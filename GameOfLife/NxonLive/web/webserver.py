# Multi Neuraxon Game of Life 5 — web server / dispatcher  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# One aiohttp app provides everything:
#   * static client + admin pages
#   * /ws         live world stream (WebSocket, broadcast ~10 Hz)
#   * /api/...    register / login / my-Nxer / export
#   * /api/admin  admin console (config, reboot, live load)
#   * /api/god    god access (inspect any NxEr)
#
# The web layer NEVER mutates engine internals — it reads thread-safe
# snapshots and calls a small, guarded GameServer API. All user strings
# are sanitised; passwords are hashed; brute-force is IP-banned.
#
# TLS: pass --cert/--key for real HTTPS. Without them it serves HTTP
# (use a reverse proxy / put certs in front for production).
# ===================================================================
import os
import ssl
import json
import time
import asyncio
import argparse

from aiohttp import web, WSMsgType

from server import np_fallback
np_fallback.install()

from server.game_server import GameServer
from web.auth import (hash_pw, sanitize,
                       coerce_overrides, IPGuard)
from web.sessions import SessionManager
from web.machine import MachineStats

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
ROOT = os.path.dirname(HERE)


class App:
    def __init__(self, gs: GameServer):
        self.gs = gs
        cfg = gs.cfg
        self.sessions = SessionManager(
            cfg.get("max_viewers", 200),
            cfg.get("max_registered_users", 100))
        self.ipguard = IPGuard(
            cfg.get("max_pw_attempts", 3),
            cfg.get("ban_hours", 24))
        self._machine = MachineStats()
        self.admin_user = cfg.get("admin_user", "admin")
        self.admin_pw_hash = hash_pw(cfg.get("admin_password", "changeme"))
        self.god_user = cfg.get("god_user", "god")
        self.god_pw_hash = hash_pw(cfg.get("god_password", "changeme"))
        self._admin_tokens = set()
        self._god_tokens = set()
        self._ws = set()

    # ---- helpers ----------------------------------------------------
    @staticmethod
    def _ip(req):
        xff = req.headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
        return req.remote or "?"

    async def _json(self, req):
        try:
            data = await req.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # ---- static -----------------------------------------------------
    @staticmethod
    def _no_cache(resp):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp

    async def index(self, req):
        return self._no_cache(web.FileResponse(
            os.path.join(STATIC, "client.html")))

    async def admin_page(self, req):
        return self._no_cache(web.FileResponse(
            os.path.join(STATIC, "admin.html")))

    async def builder_page(self, req):
        """Serve the trimmed MultiNx2Builder — Advanced Neuraxon
        model design opened in a separate window from the Create form.
        Fixed to the chc6 topology (auto-loads the prefrontal preset)
        with a 'Save for GoL Life' button that exports a JSON the
        client's Load button accepts."""
        return self._no_cache(web.FileResponse(
            os.path.join(STATIC, "builder.html")))

    async def kaleido_page(self, req):
        """v1.48 — NxonKaleido (was NxonCaliedo): the standalone
        brain-connectivity viewer, rendered entirely client-side."""
        return self._no_cache(web.FileResponse(
            os.path.join(STATIC, "kaleido.html")))

    async def _hidden_404(self, req):
        """When admin_path is customised, the default /admin URL must
        not reveal the panel — return a generic 404 page so scanners
        cannot fingerprint the admin endpoint."""
        return web.Response(status=404, text="Not Found")

    # ---- live stream ------------------------------------------------
    async def ws(self, req):
        token = self.sessions.open_viewer()
        if token is None:
            return web.json_response(
                {"error": "viewer capacity reached"}, status=503)
        ws = web.WebSocketResponse(heartbeat=30)
        try:
            await ws.prepare(req)
        except asyncio.CancelledError:
            self.sessions.close(token)
            raise
        except Exception:
            # v1.42 — a client (often a stale browser tab reconnecting the
            # instant the server reboots) frequently drops the socket
            # mid-handshake; aiohttp then raises ClientConnectionResetError
            # ("Cannot write to closing transport") from the header write.
            # That is normal connection churn, not a server fault. Release
            # the viewer slot and return quietly instead of letting it
            # bubble up as an "Error handling request" traceback.
            self.sessions.close(token)
            return ws
        self._ws.add(ws)
        try:
            await ws.send_json({"type": "hello", "token": token})
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # the only client→server msg is a lightweight ping;
                    # all auth happens over REST, not the socket.
                    pass
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._ws.discard(ws)
            self.sessions.close(token)
        return ws

    async def _broadcast_loop(self):
        hz = float(self.gs.cfg.get("broadcast_hz", 10))
        period = 1.0 / hz
        while True:
            # FAST PATH: if the snapshot worker subprocess is enabled
            # and has produced bytes, send those directly — the JSON
            # encoding ran on a different core. If the worker is
            # disabled or hasn't published yet, fall through to the
            # in-process JSON build.
            payload_bytes = self.gs.latest_bytes()
            payload_str = None
            if payload_bytes is None:
                snap = self.gs.snapshot()
                if snap and self._ws:
                    payload_str = json.dumps({"type": "world",
                                              "data": snap})
            if (payload_bytes or payload_str) and self._ws:
                # aiohttp's send_str expects a string. If we have raw
                # bytes from the worker (UTF-8 JSON), decode here. The
                # decode is cheap relative to the encode work that just
                # ran on another core, and lets us keep send_str()
                # which the clients already handle.
                if payload_str is None:
                    payload_str = payload_bytes.decode("utf-8")
                # v1.42 — fan the frame out to every viewer CONCURRENTLY,
                # with a per-send timeout. Previously this was a sequential
                # `for ws: await ws.send_str(...)`, so a single slow or
                # backpressured viewer (mobile, distant, or a tab that
                # stopped reading) stalled the whole broadcast loop until
                # its write drained. Because WS handshakes (`ws.prepare`)
                # run on this same event loop, that stall also delayed NEW
                # connections — so under load the console could fail to
                # connect even though the engine was healthy. We now send
                # concurrently and bound each send: a client that can't
                # accept a frame within the timeout is dropped (it will
                # reconnect) rather than holding up everyone else and the
                # handshakes. Worst-case loop cycle is the timeout, not
                # infinity.
                clients = list(self._ws)

                async def _send_one(_ws, _s):
                    try:
                        await asyncio.wait_for(_ws.send_str(_s),
                                               timeout=1.0)
                        return None
                    except Exception as _e:
                        return _e

                results = await asyncio.gather(
                    *(_send_one(ws, payload_str) for ws in clients))
                for ws, res in zip(clients, results):
                    if res is not None:
                        self._ws.discard(ws)
            await asyncio.sleep(period)

    async def api_world(self, req):
        return web.json_response(self.gs.world_meta())

    async def api_params(self, req):
        """Authoritative slider spec for the registration form. Brain
        topology is fixed (CHC g-capable) and intentionally absent."""
        from server.engine import USER_TUNABLE, make_params
        defp = make_params()
        spec = []
        for k, (lo, hi, typ) in USER_TUNABLE.items():
            spec.append({
                "key": k,
                "min": lo, "max": hi,
                "type": ("bool" if typ is bool
                         else "int" if typ is int else "float"),
                "default": getattr(defp, k, lo),
            })
        return web.json_response({"topology": "chc6", "params": spec})

    # ---- register / login (owners) ----------------------------------
    async def api_register(self, req):
        ip = self._ip(req)
        if self.ipguard.is_banned(ip):
            return web.json_response({"error": "ip temporarily banned"},
                                     status=429)
        d = await self._json(req)
        # Password is now auto-managed: the server picks a simple
        # 4-character alphanumeric password and returns it to the
        # client (shown in the info modal). The user doesn't have to
        # type or invent one. They still need to save it to log in
        # from another device later.
        import secrets
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I/l
        pw = "".join(secrets.choice(alphabet) for _ in range(4))
        c = self.sessions.counts()
        if c["registered"] >= c["max_registered"]:
            return web.json_response(
                {"error": "registered-user capacity reached"},
                status=503)
        overrides = coerce_overrides(d.get("params", {}))
        # v1.45 — up to max_nxers_per_user (default 3) live NxErs per
        # browser owner-token, instead of exactly one. If this token is
        # already at the cap, refuse and return the names it owns.
        otok = sanitize(d.get("owner", ""), 64)
        if not otok:
            import secrets
            otok = secrets.token_urlsafe(18)
        limit = int(self.gs.cfg.get("max_nxers_per_user", 3))
        owned = self.gs.owner_live_names(otok)
        if len(owned) >= limit:
            return web.json_response(
                {"error": "you already have %d live NxErs (max %d)"
                          % (len(owned), limit),
                 "names": owned, "owner": otok}, status=409)
        name = self.gs.register_nxer(overrides, hash_pw(pw), None)
        if not name:
            return web.json_response(
                {"error": "world is full — try again shortly"},
                status=503)
        names = self.gs.bind_owner_session(otok, name)
        return web.json_response(
            {"name": name, "owner": otok, "password": pw,
             "names": names})

    async def api_session(self, req):
        """Auto-reconnect: given the browser's owner token, return the
        NxErs it owns that are still alive (no password needed). v1.45 —
        returns the full list (`names`) plus `name` for back-compat."""
        d = await self._json(req)
        tok = sanitize(d.get("owner", ""), 64)
        names = self.gs.owner_session_names(tok)
        return web.json_response(
            {"names": names, "name": (names[0] if names else None)})

    async def api_nxbrain(self, req):
        """v1.48 — NxonKaleido: live brain topology for one NxEr. v1.51 —
        fully lock-free: marks the NxEr wanted (by name) and reads the
        cache the game loop fills. NEVER acquires the engine lock, so it
        can't stall the event loop / the world WebSocket while a viewer
        polls. The client renders everything; no server-side visual work."""
        name = sanitize(req.query.get("name", ""), 16)
        if not name:
            return web.json_response({"error": "no name"}, status=400)
        self.gs.request_topology(name)
        c = self.gs.get_topology(name)
        if c is None:
            return web.json_response({"warming": True, "name": name})
        if c.get("dead"):
            return web.json_response(
                {"error": "NxEr not alive", "name": name}, status=404)
        return web.json_response(
            {"name": c["name"], "tick": c["tick"],
             "spheres": c["topo"]["spheres"], "links": c["topo"]["links"],
             "events": c.get("events", [])})

    async def api_mynxers(self, req):
        """v1.45 — owner views for every live NxEr this browser token
        owns (up to the per-user cap), so the client can track them all
        in one panel."""
        tok = sanitize(req.query.get("owner", ""), 64)
        out = []
        for nm in self.gs.owner_session_names(tok):
            v = self.gs.get_owner_view(nm)
            if v is not None:
                out.append(v)
        return web.json_response({"nxers": out})

    async def api_login(self, req):
        ip = self._ip(req)
        if self.ipguard.is_banned(ip):
            return web.json_response({"error": "ip temporarily banned"},
                                     status=429)
        d = await self._json(req)
        name = sanitize(d.get("name", ""), 16)
        pw = d.get("password", "")
        token = sanitize(d.get("token", ""), 64)
        owner = sanitize(d.get("owner", ""), 64)
        nx = self.gs.find_nxer_by_name(name)
        if (nx is None or not nx.alive or not nx.password_hash
                or nx.password_hash != hash_pw(pw)):
            banned = self.ipguard.record_fail(ip)
            return web.json_response(
                {"error": "invalid name/password or NxEr not alive",
                 "ip_banned": banned}, status=401)
        self.ipguard.record_success(ip)
        if token and not self.sessions.promote_registered(token, name):
            return web.json_response(
                {"error": "owner-session capacity reached"}, status=503)
        # v1.45 — also bind to this browser's owner token so a NxEr you
        # log into is tracked alongside the ones you created, up to the
        # per-user cap (already-owned names re-bind freely).
        names = [name]
        if owner:
            limit = int(self.gs.cfg.get("max_nxers_per_user", 3))
            owned = self.gs.owner_live_names(owner)
            if name in owned or len(owned) < limit:
                names = self.gs.bind_owner_session(owner, name)
            else:
                names = owned + [name]
        return web.json_response({"ok": True, "name": name,
                                  "names": names})

    async def api_mynxer(self, req):
        name = sanitize(req.query.get("name", ""), 16)
        view = self.gs.get_owner_view(name)
        if view is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(view)

    async def api_mynxer_family(self, req):
        """Direct-lineage view (parents + children) for an owned
        NxEr. Names are publicly addressable already (api_mynxer);
        the family endpoint follows the same model — knowing the
        name is the gate."""
        name = sanitize(req.query.get("name", ""), 16)
        fam = self.gs.get_family(name)
        if fam is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(fam)

    async def api_mynxer_child(self, req):
        """Inspect ONE child of an owned NxEr. Server enforces the
        rule from the spec: only allowed while the parent is alive.
        That makes ancestry browsing a privilege of the surviving
        original NxEr — once it dies, the line is sealed."""
        parent = sanitize(req.query.get("name", ""), 16)
        child  = sanitize(req.query.get("child", ""), 16)
        status, view = self.gs.get_child_view(parent, child)
        if status == "ok":
            return web.json_response(view)
        codes = {"parent_dead": 410, "not_a_child": 403,
                 "not_found":   404}
        return web.json_response(
            {"error": status.replace("_", " ")},
            status=codes.get(status, 400))

    async def api_export(self, req):
        name = sanitize(req.query.get("name", ""), 16)
        model = self.gs.export_nxer(name)
        if model is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(model)

    # ---- admin ------------------------------------------------------
    async def api_admin_login(self, req):
        ip = self._ip(req)
        if self.ipguard.is_banned(ip):
            return web.json_response({"error": "ip banned"}, status=429)
        d = await self._json(req)
        if (sanitize(d.get("user", ""), 32) == self.admin_user
                and hash_pw(d.get("password", "")) == self.admin_pw_hash):
            import secrets
            tok = secrets.token_urlsafe(24)
            self._admin_tokens.add(tok)
            return web.json_response({"token": tok})
        self.ipguard.record_fail(ip)
        return web.json_response({"error": "bad credentials"},
                                 status=401)

    def _is_admin(self, d):
        return sanitize(d.get("token", ""), 64) in self._admin_tokens

    async def api_admin_load(self, req):
        d = await self._json(req)
        if not self._is_admin(d):
            return web.json_response({"error": "unauthorized"},
                                     status=403)
        m = self.gs.load_metrics()
        m["sessions"] = self.sessions.counts()
        m["banned_ips"] = self.ipguard.banned_count()
        m["machine"] = self._machine.sample()
        m["config"] = self.gs.cfg
        return web.json_response(m)

    async def api_admin_config(self, req):
        d = await self._json(req)
        if not self._is_admin(d):
            return web.json_response({"error": "unauthorized"},
                                     status=403)
        patch = d.get("config", {})
        if isinstance(patch, dict):
            cfg_path = self.gs.config_path
            with open(cfg_path, "r", encoding="utf-8") as f:
                cur = json.load(f)
            for k, v in patch.items():
                if k in cur and isinstance(
                        v, (int, float, str, bool)):
                    cur[k] = v
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cur, f, indent=2)
            self.gs.reload_config()
            self.sessions.set_caps(
                self.gs.cfg.get("max_viewers", 200),
                self.gs.cfg.get("max_registered_users", 100))
        return web.json_response({"ok": True, "config": self.gs.cfg})

    async def api_admin_reboot(self, req):
        d = await self._json(req)
        if not self._is_admin(d):
            return web.json_response({"error": "unauthorized"},
                                     status=403)
        self.gs.reboot()
        return web.json_response({"ok": True})

    async def api_admin_stats(self, req):
        """All-time / lifetime counters plus current snapshot —
        designed to be useful for days-long sessions. Updates uptime
        on every call so the admin page always shows fresh totals."""
        d = await self._json(req)
        if not self._is_admin(d):
            return web.json_response({"error": "unauthorized"},
                                     status=403)
        eng = self.gs.engine
        # refresh uptime
        import time as _t
        now = _t.time()
        eng.lifetime["uptime_seconds"] += max(0.0,
            now - eng._uptime_t0)
        eng._uptime_t0 = now
        # current snapshot of the live world
        # v1.57 — take ONE list copy first. These lines used to iterate
        # engine.nxers directly from the asyncio event loop while the game
        # loop was spawning/killing NxErs in that same dict, which raised
        # "RuntimeError: dictionary changed size during iteration" and
        # crashed the admin page at random (seen repeatedly in production
        # logs). list() over .values() is a single C-level copy, far
        # cheaper than taking the engine lock and — unlike the lock — it
        # cannot stall the world. A torn read is harmless here: these are
        # display counters, not science.
        try:
            snap = list(eng.nxers.values())
        except RuntimeError:
            snap = list(eng.nxers.values())    # one retry; mutation is brief
        alive_total = sum(1 for a in snap if a.alive)
        alive_managed = sum(1 for a in snap
                            if a.alive and a.is_managed)
        # average lifespan among the dead we still have records for
        dead_with_age = [
            (eng.tick - a.born_tick) if hasattr(a, "born_tick") else 0
            for a in snap
            if not a.alive and getattr(a, "born_tick", None) is not None
        ]
        avg_lifespan_ticks = (sum(dead_with_age) / len(dead_with_age)
                              if dead_with_age else 0.0)
        # mating efficiency (births per spawn) is a useful long-run
        # indicator of population health
        ts = eng.lifetime["total_spawns"] or 1
        mating_share = eng.lifetime["total_births_mating"] / ts
        return web.json_response({
            "lifetime": eng.lifetime,
            "current": {
                "tick": eng.tick,
                "alive_total": alive_total,
                "alive_managed": alive_managed,
                "tracked": len(eng.nxers),    # alive + dead in pool
                "foods_on_map": len(eng.foods),
            },
            "derived": {
                "avg_lifespan_ticks": round(avg_lifespan_ticks, 1),
                "share_births_via_mating": round(mating_share, 3),
                "uptime_hours": round(
                    eng.lifetime["uptime_seconds"] / 3600.0, 2),
            },
            "brain_pool": (eng.pool.mode_info()
                           if hasattr(eng.pool, "mode_info") else {}),
            "perf": (eng.get_perf()
                     if hasattr(eng, "get_perf") else {}),
            "m12": getattr(eng, "_last_m12", {}),
            "history": (eng.history.stats()
                        if getattr(eng, "history", None) is not None
                        else {"enabled": False}),
        })

    # ---- god --------------------------------------------------------
    async def api_god_login(self, req):
        ip = self._ip(req)
        if self.ipguard.is_banned(ip):
            return web.json_response({"error": "ip banned"}, status=429)
        d = await self._json(req)
        if (sanitize(d.get("user", ""), 32) == self.god_user
                and hash_pw(d.get("password", "")) == self.god_pw_hash):
            import secrets
            tok = secrets.token_urlsafe(24)
            self._god_tokens.add(tok)
            return web.json_response({"token": tok})
        self.ipguard.record_fail(ip)
        return web.json_response({"error": "bad credentials"},
                                 status=401)

    async def api_god_inspect(self, req):
        d = await self._json(req)
        if sanitize(d.get("token", ""), 64) not in self._god_tokens:
            return web.json_response({"error": "unauthorized"},
                                     status=403)
        name = sanitize(d.get("name", ""), 16)
        view = self.gs.get_owner_view(name)
        if view is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(view)

    # ---- app --------------------------------------------------------
    def build(self):
        # Domain + HTTPS enforcement. Set in world_config.json:
        #   "canonical_host": "play.example.com"   force this host
        #   "force_https":    true                 redirect HTTP -> HTTPS
        # Works behind a reverse proxy (nginx/cloudflare) — we trust
        # the X-Forwarded-Proto / Host headers the proxy sets, AND we
        # also work standalone when aiohttp is the TLS terminator.
        cfg = self.gs.cfg
        canon_host = str(cfg.get("canonical_host", "")).strip().lower()
        force_https = bool(cfg.get("force_https", False))

        @web.middleware
        async def host_https_mw(req, handler):
            host = (req.headers.get("Host") or "").lower().split(":")[0]
            xfp = (req.headers.get("X-Forwarded-Proto") or
                   req.scheme).lower()
            # 1. wrong host -> 308 to canonical
            if canon_host and host and host != canon_host:
                tgt = ("https://" if force_https or xfp == "https"
                       else "http://") + canon_host + req.rel_url.path_qs
                return web.Response(status=308,
                                    headers={"Location": tgt})
            # 2. http when https required -> 308 to https. MUST skip the
            #    WebSocket upgrade: a WS handshake is a GET, browsers do
            #    NOT follow redirects on it, and nginx already terminated
            #    TLS — so a ws:// upgrade proxied here over plain http to
            #    127.0.0.1 is expected. Redirecting it (which happens if
            #    nginx omits X-Forwarded-Proto: https) silently breaks the
            #    live stream. The previous code's comment claimed to skip
            #    WS but the condition didn't — this is the fix.
            is_ws_upgrade = (
                req.headers.get("Upgrade", "").lower() == "websocket")
            if (force_https and xfp == "http" and req.method == "GET"
                    and not is_ws_upgrade):
                tgt = ("https://" + (canon_host or host) +
                       req.rel_url.path_qs)
                return web.Response(status=308,
                                    headers={"Location": tgt})
            return await handler(req)

        app = web.Application(middlewares=[host_https_mw])
        # configurable admin URL: world_config.json -> "admin_path"
        # e.g. "ops-7a91c.html" — hides the obvious /admin route. The
        # admin API still lives under /api/admin/* and is password
        # protected as before; this is just URL obscurity for the
        # static panel.
        admin_path = str(self.gs.cfg.get("admin_path", "admin")).strip("/")
        admin_route = "/" + (admin_path or "admin")
        routes = [
            web.get("/", self.index),
            web.get("/static/builder.html", self.builder_page),
            web.get("/kaleido.html", self.kaleido_page),
            web.get("/static/kaleido.html", self.kaleido_page),
            web.get("/caleido.html", self.kaleido_page),   # back-compat alias
            web.get("/static/caleido.html", self.kaleido_page),
            web.get(admin_route, self.admin_page),
            web.get("/ws", self.ws),
            web.get("/api/world", self.api_world),
            web.get("/api/params", self.api_params),
            web.post("/api/register", self.api_register),
            web.post("/api/session", self.api_session),
            web.post("/api/login", self.api_login),
            web.get("/api/mynxer", self.api_mynxer),
            web.get("/api/mynxers", self.api_mynxers),
            web.get("/api/nxbrain", self.api_nxbrain),
            web.get("/api/mynxer/family", self.api_mynxer_family),
            web.get("/api/mynxer/child",  self.api_mynxer_child),
            web.get("/api/export", self.api_export),
            web.post("/api/admin/login", self.api_admin_login),
            web.post("/api/admin/load", self.api_admin_load),
            web.post("/api/admin/config", self.api_admin_config),
            web.post("/api/admin/reboot", self.api_admin_reboot),
            web.post("/api/admin/stats", self.api_admin_stats),
            web.post("/api/god/login", self.api_god_login),
            web.post("/api/god/inspect", self.api_god_inspect),
        ]
        # If the panel was relocated, return a generic 404 from the
        # default /admin so port-scanners don't find it by guessing.
        if admin_route != "/admin":
            routes.append(web.get("/admin", self._hidden_404))
        app.add_routes(routes)

        async def _on_start(_):
            asyncio.create_task(self._broadcast_loop())
        app.on_startup.append(_on_start)
        return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=os.path.join(ROOT, "world_config.json"))
    ap.add_argument("--state",
                    default=os.path.join(ROOT, "state"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert", default=None)
    ap.add_argument("--key", default=None)
    args = ap.parse_args()

    gs = GameServer(args.config, args.state)
    gs.start()
    app = App(gs).build()

    ssl_ctx = None
    if args.cert and args.key:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(args.cert, args.key)
        print(f"[web] HTTPS on {args.host}:{args.port}")
    else:
        print(f"[web] HTTP on {args.host}:{args.port} "
              "(no --cert/--key; put TLS in front for production)")
    web.run_app(app, host=args.host, port=args.port,
                ssl_context=ssl_ctx, print=None)


if __name__ == "__main__":
    main()
