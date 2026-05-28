# Multi Neuraxon Game of Life 5 — game server (forever loop)  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# Owns the Engine, steps it forever in a background thread, snapshots
# for crash recovery, and exposes a thread-safe public snapshot the web
# layer broadcasts. Restart vs reboot:
#   * RESTART (default on boot): if a snapshot exists, resume from it
#     (names + rankings + brains preserved).
#   * REBOOT (admin action / no snapshot): fresh world from
#     world_config.json, name space reset to A.
# ===================================================================
import os
import time
import json
import threading

from .engine import (Engine, NxEr, make_params, _params_to_dict,
                      RANK_METRICS)
from .names import NameAllocator
from .persistence import Persistence


SERVER_VERSION = "GoL Server V 1.035"   # bumped each release

class GameServer:
    def __init__(self, config_path, state_dir):
        self.config_path = config_path
        self.persist = Persistence(state_dir)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._snapshot = {}
        self._metrics = {"tick": 0, "uptime_s": 0.0, "tps": 0,
                         "alive": 0, "total_tracked": 0, "managed": 0,
                         "g": {"pc1": 0.0, "pos_manifold": 0.0,
                               "mean_r": 0.0, "lambda_ratio": 1.0,
                               "n": 0}}
        self._world_meta_cache = None
        self._owner_views = {}
        self._owner_sessions = {}    # browser token -> NxEr name
        self._reg_queue = []
        self._reg_lock = threading.Lock()
        self._last_save = 0.0
        # Hourly "best of" archive (food, explored, lived, mates,
        # fitness, g). Each entry is the per-metric highest value
        # already saved, so we never re-save the same NxEr at the
        # same score and the archive only grows when a record is
        # actually broken.
        self._best_dir = os.path.join(state_dir, "best")
        try:
            os.makedirs(self._best_dir, exist_ok=True)
        except OSError:
            pass
        self._best_saved = self._load_best_index()
        self._best_last_save = time.time()
        self._steps = 0
        self._t0 = time.time()
        self.world_epoch = 0
        self.cfg = self._load_config()
        self.engine = None
        self._boot()

    # ---- config -----------------------------------------------------
    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reload_config(self):
        with self._lock:
            self.cfg = self._load_config()
            # live-tunable knobs take effect next tick
            for k in ("max_food", "max_nxers", "min_alive",
                      "respawn_batch", "g_interval_ticks",
                      "mate_cooldown_ticks", "snapshot_secs",
                      "target_tps", "max_viewers",
                      "max_registered_users"):
                if k in self.cfg and self.engine is not None:
                    self.engine.cfg[k] = self.cfg[k]
        return True

    # ---- boot: restart (resume) or reboot (fresh) -------------------
    def _boot(self, force_reboot=False):
        # terminate the previous engine's worker processes (reboot)
        old = getattr(self, "engine", None)
        if old is not None:
            try:
                old.shutdown()
            except Exception:
                pass
        snap = None if force_reboot else self.persist.load()
        if snap:
            self._restore(snap)
            print(f"[GameServer] {SERVER_VERSION} — RESTART from "
                  f"snapshot (tick {self.engine.tick}, "
                  f"{len(self.engine.nxers)} NxErs).")
        else:
            names = NameAllocator(0)
            self.engine = Engine(self.cfg, names)
            print(f"[GameServer] {SERVER_VERSION} — REBOOT fresh "
                  f"world ({self.cfg['world_size']}², "
                  f"{self.cfg['starting_nxers']} NxErs).")
        self._refresh_snapshot()
        self._world_meta_cache = None
        self._publish_metrics()
        try:
            pool = self.engine.pool
            print(f"[GameServer] brain pool: {len(pool._procs)} step "
                  f"workers + {len(pool._builder_procs)} builders "
                  f"({os.cpu_count()} cores detected). "
                  f"Founder brains build in parallel; NxErs appear as "
                  f"translucent 'ghosts' until their brain is ready.")
        except Exception:
            pass

    def _restore(self, snap):
        names = NameAllocator.from_state(snap.get("names_state"))
        eng = Engine.__new__(Engine)
        eng.cfg = self.cfg
        eng.names = names
        from .engine import World
        eng.world = World(self.cfg["world_size"], self.cfg["sea_pct"],
                          self.cfg["rock_pct"],
                          self.cfg.get("world_seed", 12345),
                          earth_map=self.cfg.get("earth_map", False))
        eng.nxers = {}
        eng.foods = {}
        eng.tick = int(snap.get("tick", 0))
        eng.next_nxer_id = int(snap.get("next_nxer_id", 0))
        eng.next_food_id = int(snap.get("next_food_id", 0))
        eng.all_time = snap.get("all_time",
                                {m: [] for m in RANK_METRICS})
        eng._g_cache = {"pc1": 0.0, "pos_manifold": 0.0,
                        "mean_r": 0.0, "lambda_ratio": 1.0, "n": 0}
        # one source of truth for all derived runtime state
        eng._init_runtime(self.cfg)
        for fr in snap.get("foods", []):
            eng.foods[fr["id"]] = {
                "pos": fr["pos"],
                "remaining": fr.get("remaining",
                                    fr.get("amount", 25)),
            }
        for nd in snap.get("nxers", []):
            params = make_params(nd.get("params", {}))
            nx = NxEr(nd["id"], nd["name"], nd["pos"], params)
            nx.alive = nd.get("alive", True)
            nx.food = nd.get("food", 60.0)
            nx.is_managed = nd.get("is_managed", False)
            nx.is_male = nd.get("is_male", nx.is_male)
            nx.can_land = nd.get("can_land", True)
            nx.can_sea = nd.get("can_sea", False)
            nx.parents = nd.get("parents", [None, None])
            nx.offspring_ids = nd.get("offspring_ids", [])
            # rebuild the brain inside its pool worker: load the saved
            # brain if present, else build fresh from params.
            if nd.get("brain"):
                eng.pool.add(nx.id, nd.get("params", {}))
                eng.pool.load(nx.id, nd["brain"])
            else:
                eng.pool.add(nx.id, nd.get("params", {}))
            st = nd.get("stats", {})
            nx.stats.food_found = st.get("food_found", 0)
            nx.stats.food_taken = st.get("food_taken", 0)
            nx.stats.explored = st.get("explored", 0)
            nx.stats.time_lived_s = st.get("time_lived", 0.0)
            nx.stats.mates_performed = st.get("mates_performed", 0)
            nx.stats.fitness = st.get("fitness", 0.0)
            nx.stats.g_factor = st.get("g", 0.0)
            eng.nxers[nx.id] = nx
            if nx.alive:
                eng._occupied[(nx.pos[0], nx.pos[1])] = nx.id
        self.engine = eng

    def reboot(self):
        """Admin action: discard the world and start fresh."""
        with self._lock:
            self.world_epoch += 1
            self._world_meta_cache = None   # rebuilt next tick
            self.persist.clear()
            self._boot(force_reboot=True)
        return True

    # ---- main loop --------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        target_tps = float(self.cfg.get("target_tps", 20))
        period = 1.0 / target_tps
        snap_secs = float(self.cfg.get("snapshot_secs", 20))
        # rebuild the public broadcast snapshot at most at broadcast_hz
        # (not every tick) — saves O(N) serial work on the main core
        bcast_period = 1.0 / float(self.cfg.get("broadcast_hz", 10))
        last_snap = 0.0
        # optional headroom: a small per-tick sleep so the engine never
        # 100%-pegs the box, keeping the OS + web server responsive even
        # when the world is large. 0 = run as fast as possible.
        yield_ms = float(self.cfg.get("cpu_yield_ms", 0)) / 1000.0
        while not self._stop.is_set():
            t = time.time()
            try:
                with self._lock:
                    self._drain_registrations()
                    self.engine.pool.drain_loads()
                    self.engine.step()
                    self._steps += 1
                    if t - last_snap >= bcast_period:
                        self._refresh_snapshot()
                        self._publish_metrics()
                        last_snap = t
                    if t - self._last_save >= snap_secs:
                        try:
                            self.persist.save(
                                self.engine.state_dict())
                            self._last_save = t
                        except Exception as e:
                            print("[GameServer] snapshot failed:", e)
                    # Hourly best-of archive (per-metric champions).
                    if t - self._best_last_save >= 3600:
                        try:
                            self._archive_best()
                        except Exception as e:
                            print("[GameServer] best-archive error:",
                                  e)
                        self._best_last_save = t
            except Exception as e:
                # a 24/7 world must never die from one bad tick
                print("[GameServer] tick error (continuing):",
                      repr(e))
                time.sleep(0.05)
            dt = time.time() - t
            if dt < period:
                time.sleep(period - dt)
            elif yield_ms > 0:
                time.sleep(yield_ms)   # leave CPU headroom

    # ---- thread-safe accessors used by the web layer ----------------
    # The web layer must NEVER acquire the engine lock — if it did, an
    # admin poll / login-verify would block for a whole (possibly slow)
    # tick and hang the console. Instead the loop publishes immutable
    # cached dicts each tick; readers just grab the latest reference
    # (atomic in CPython), so the web server is always responsive even
    # when the simulation is fully saturated.
    def _publish_metrics(self):
        up = time.time() - self._t0
        eng = self.engine
        self._metrics = {
            "tick": eng.tick,
            "uptime_s": round(up, 1),
            "tps": round(self._steps / up, 2) if up > 0 else 0,
            "alive": sum(1 for a in eng.nxers.values() if a.alive),
            "total_tracked": len(eng.nxers),
            "managed": sum(1 for a in eng.nxers.values()
                           if a.is_managed and a.alive),
            "g": dict(eng._g_cache),
        }
        # Lock-free owner-view cache. Only OWNED NxErs can be connected
        # to, so this is bounded by the number of registered users
        # (default <=100) and owner_view() is cheap (no brain export).
        # /api/mynxer reads this WITHOUT the engine lock, so a client
        # polling it can never contend with the simulation loop — this
        # is what made connecting peg the CPU and stall the client.
        ov = {}
        # names currently in any all-time top-5 board (for highlight)
        top_names = set()
        for m, board in eng.all_time.items():
            for e in board[:5]:
                top_names.add(e["name"])
        for a in eng.nxers.values():
            if a.alive and a.is_managed:
                try:
                    v = a.owner_view()
                    v["ranks"] = eng.rank_of(a)
                    v["in_top5"] = a.name in top_names
                    v["brain_building"] = eng._brain_building(a)
                    ov[a.name] = v
                except Exception:
                    pass
        self._owner_views = ov
        if self._world_meta_cache is None:
            w = eng.world
            self._world_meta_cache = {
                "size": w.size, "earth_map": w.earth_map,
                "epoch": self.world_epoch,
                "terrain": w.terrain_rows(),
            }

    def _refresh_snapshot(self):
        snap = self.engine.world_snapshot()
        snap["world"]["epoch"] = self.world_epoch
        self._snapshot = snap

    def snapshot(self):
        return self._snapshot

    def load_metrics(self):
        return self._metrics            # lock-free cached read

    def world_meta(self):
        return self._world_meta_cache or {
            "size": 0, "earth_map": False, "epoch": self.world_epoch,
            "terrain": []}

    def _load_best_index(self):
        """Read state/best/_index.json if present so per-metric high
        water marks survive a restart (otherwise we'd re-save the same
        top NxEr the first hour after every reboot)."""
        path = os.path.join(self._best_dir, "_index.json")
        try:
            with open(path) as f:
                return {k: float(v) for k, v in json.load(f).items()}
        except (OSError, ValueError):
            return {}

    def _save_best_index(self):
        path = os.path.join(self._best_dir, "_index.json")
        try:
            with open(path, "w") as f:
                json.dump(self._best_saved, f)
        except OSError:
            pass

    def _archive_best(self):
        """Save the current top alive NxEr per metric to
        state/best/<metric>_<name>_<value>_<tick>.json — but only when
        the value beats whatever has been saved before for that
        metric. Runs every hour from the tick loop, under the engine
        lock (rare + cheap, ~6 export calls)."""
        from .engine import RANK_METRICS, _metric
        archived = []
        for m in RANK_METRICS:
            top = self.engine._rank_top.get(m, [])
            best_alive = None
            for e in top:
                a = self.engine.nxers.get(e["id"])
                if a and a.alive and a.is_managed:
                    best_alive = a
                    break
            if best_alive is None:
                continue
            value = _metric(best_alive, m)
            if value <= self._best_saved.get(m, 0.0):
                continue
            try:
                model = self.engine.export_model_for(best_alive)
            except Exception as e:
                print("[GameServer] best-archive export failed:", e)
                continue
            v_disp = f"{value:.3f}".replace(".", "_")
            fn = (f"{m}_{best_alive.name}_{v_disp}"
                  f"_t{self.engine.tick}.json")
            fpath = os.path.join(self._best_dir, fn)
            try:
                with open(fpath, "w") as f:
                    json.dump(model, f)
                self._best_saved[m] = value
                archived.append((m, best_alive.name, value))
            except OSError as e:
                print("[GameServer] best-archive write failed:", e)
        if archived:
            self._save_best_index()
            for m, nm, v in archived:
                print(f"[best] archived {m}: {nm} = {v:.3f}")

    def find_nxer_by_name(self, name):
        with self._lock:
            for a in self.engine.nxers.values():
                if a.name == name:
                    return a
            return None

    def get_owner_view(self, name):
        # lock-free fast path (refreshed every broadcast tick)
        v = self._owner_views.get(name)
        if v is not None:
            return v
        # fallback: just-registered NxEr not yet in a published frame —
        # still attach ranks/in_top5 so the client never sees them blank
        with self._lock:
            a = self.find_nxer_by_name(name)
            if not a:
                return None
            d = a.owner_view()
            try:
                d["ranks"] = self.engine.rank_of(a)
                d["brain_building"] = self.engine._brain_building(a)
                top = set()
                for board in self.engine.all_time.values():
                    for e in board[:5]:
                        top.add(e["name"])
                d["in_top5"] = a.name in top
            except Exception:
                d["ranks"] = {}
                d["in_top5"] = False
            return d

    def export_nxer(self, name):
        with self._lock:
            a = self.find_nxer_by_name(name)
            return self.engine.export_model_for(a) if a else None

    def register_nxer(self, overrides, password_hash, owner_token):
        # Decouple create-latency from the (possibly slow) step: the
        # web thread allocates the name instantly and enqueues the
        # request; the game loop builds the actual NxEr at the top of
        # its next tick. No waiting on the engine lock across a whole
        # step — this is what made NxEr creation feel slow.
        # Cap on the LIVING population. The engine keeps up to ~200
        # dead NxErs around for the all-time ranking scan, so
        # len(engine.nxers) is NOT the population — counting it made
        # registration fail with "world is full" long before the
        # world actually filled. Count alive NxErs plus the pending
        # (queued-but-not-yet-built) registrations instead.
        cap = int(self.cfg.get("max_nxers", 150))
        alive = sum(1 for a in self.engine.nxers.values() if a.alive)
        with self._reg_lock:
            pending = len(self._reg_queue)
        if alive + pending >= cap:
            return None
        name = self.engine.names.next_name()       # thread-safe
        with self._reg_lock:
            self._reg_queue.append(
                (name, overrides, password_hash, owner_token))
        return name

    def _drain_registrations(self):
        """Called by the game loop (already under the engine lock) at
        the top of each tick — fast: name is pre-allocated, pool.add is
        fire-and-forget, the CHC brain builds deferred in the worker."""
        with self._reg_lock:
            q, self._reg_queue = self._reg_queue, []
        for name, ov, pwh, otok in q:
            try:
                self.engine.register_user_nxer(
                    ov, pwh, otok, name=name)
            except Exception as e:
                print("[GameServer] register failed:", repr(e))

    def owner_session_name(self, token):
        """Return the NxEr name this browser-session token owns IFF it
        is still alive, else None (so the client can auto-reconnect
        without a password, and we can block a 2nd live NxEr)."""
        if not token:
            return None
        name = self._owner_sessions.get(token)
        if not name:
            return None
        v = self._owner_views.get(name)   # lock-free, only-if-alive
        return name if v else None

    def owner_live_name(self, token):
        """Lock-free: is this browser token's NxEr alive (or just
        queued for creation)? Used by register to block a 2nd live
        NxEr WITHOUT touching the engine lock (which the slow step
        holds), so registration stays instant."""
        if not token:
            return None
        name = self._owner_sessions.get(token)
        if not name:
            return None
        if name in self._owner_views:          # lock-free alive cache
            return name
        with self._reg_lock:                   # just-queued, not yet live
            for n, _, _, _ in self._reg_queue:
                if n == name:
                    return name
        return None

    def bind_owner_session(self, token, name):
        self._owner_sessions[token] = name
