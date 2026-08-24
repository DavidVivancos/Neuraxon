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
import glob
import threading

from .engine import (Engine, NxEr, make_params, _params_to_dict,
                      RANK_METRICS, _AGE_CKPTS)
from .names import NameAllocator
from .persistence import Persistence


SERVER_VERSION = "GoL Server V 1.078"   # bumped each release

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
        self._owner_sessions = {}    # browser token -> [NxEr names] (v1.45)
        # v1.48 — NxonKaleido brain-viewer plumbing. The web handler can't
        # touch the worker pipes (the step loop owns them), so it sets a
        # "want" and reads a cache that the GAME LOOP refreshes under the
        # lock. Only refreshed while a viewer is actively polling, so it's
        # free when nobody is watching.
        self._topo_want = None       # (nxer_id, requested_at)
        self._topo_cache = None      # {"id","name","tick","ts","topo"}
        self._topo_prev = {}         # v1.51 — per-NxEr prev state for event deltas
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
        # v1.56 — migrate/repair folders written by the v1.48-v1.55 layout
        try:
            self._repair_best_dir()
        except Exception as e:
            print("[best] repair skipped:", e)
        self._best_last_save = time.time()
        self._steps = 0
        self._t0 = time.time()
        self.world_epoch = 0
        self.cfg = self._load_config()
        self.engine = None
        # v1.34 — persistent SCIENCE history (append-only JSONL, written
        # by a background thread so it never blocks the engine). Separate
        # from the crash-recovery snapshot: the snapshot is "how to
        # resume", the history is "what happened" (trinary distribution,
        # trait evolution, g, lineage, obituaries) for offline analysis.
        from .history import HistoryLogger
        hist_dir = os.path.join(state_dir,
                                self.cfg.get("history_dir", "history"))
        self.history = HistoryLogger(
            hist_dir,
            enabled=bool(self.cfg.get("history_enabled", True)),
            max_mb_per_stream=int(self.cfg.get("history_max_mb", 200)),
            keep_rotations=int(self.cfg.get("history_keep_rotations", 4)))
        self._hist_secs = float(self.cfg.get("history_sample_secs", 60))
        self._hist_last = 0.0
        # Snapshot worker subprocess (opt-in). When enabled, the engine
        # thread pushes COMPACT raw snapshots into the worker via Pipe;
        # the worker rebuilds the broadcast dict + JSON-encodes on a
        # SEPARATE CORE (its own Python process, no GIL contention with
        # the parent). The aiohttp WS loop reads the pre-encoded bytes
        # from `self._latest_bytes` instead of building locally. If the
        # worker crashes the bridge falls back to in-process building.
        self._snap_worker = None
        self._snap_send = None        # parent -> worker (raw tuples)
        self._snap_recv = None        # worker -> parent (json bytes)
        self._snap_reader = None      # thread that drains _snap_recv
        self._latest_bytes = None     # most recent JSON broadcast frame
        # v1.41 — the parent->worker send used to run inline on the engine
        # thread. multiprocessing Pipe.send() BLOCKS when the OS pipe
        # buffer (~64 KB) fills, which happened whenever the worker fell
        # behind encoding the full food list at 10 Hz — freezing the whole
        # simulation for 1-2 s at a time. We now drop the latest raw frame
        # into a 1-slot latest-wins mailbox and let a dedicated pump thread
        # do the (still blocking) send. Only the pump blocks; the engine
        # keeps ticking and stale frames are simply overwritten.
        self._snap_pending = None     # 1-slot mailbox (latest raw frame)
        self._snap_pump = None        # thread doing the blocking send
        self._snap_pump_evt = threading.Event()
        self._saving = False          # crash-recovery save in flight (off-thread)
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
        # Attach the science logger to the engine on BOTH paths (the
        # restore path uses Engine.__new__ and bypasses __init__, so set
        # it explicitly here) and stamp provenance for this boot.
        self.engine.history = self.history
        try:
            self.engine.history_provenance(SERVER_VERSION, self.cfg)
        except Exception as e:
            print("[GameServer] history provenance failed:", repr(e))
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
                          earth_map=self.cfg.get("earth_map", False),
                          pole_frac=self.cfg.get("pole_frac", 0.01))
        eng.nxers = {}
        eng.foods = {}
        eng.tick = int(snap.get("tick", 0))
        eng.next_nxer_id = int(snap.get("next_nxer_id", 0))
        eng.next_food_id = int(snap.get("next_food_id", 0))
        # v1.53 — carry the NAS trial counter across the reboot so trial
        # ids in nas_trials.jsonl never collide (read BEFORE _init_runtime,
        # which only defaults it when absent).
        try:
            eng._nas_trial_seq = int(snap.get("nas_trial_seq", 0))
        except (TypeError, ValueError):
            eng._nas_trial_seq = 0
        eng.all_time = snap.get("all_time",
                                {m: [] for m in RANK_METRICS})
        # _restore bypasses Engine.__init__ via Engine.__new__, so we
        # must seed the lifetime dict here too. Defaults match
        # Engine.__init__'s. If the snapshot has a "lifetime" sub-dict
        # (created in v1.038+), merge its values in; older snapshots
        # silently start with fresh zeros and accumulate from this run
        # forward.
        eng.lifetime = {
            "started_at_unix": time.time(),
            "uptime_seconds":  0.0,
            "total_ticks":     0,
            "total_spawns":    0,
            "total_managed_registrations": 0,
            "total_births_mating": 0,
            "total_deaths":    0,
            "total_food_eaten":0,
            "total_food_spawned":0,
            "total_matings":   0,
            "peak_alive":      0,
            "peak_managed":    0,
        }
        eng._uptime_t0 = time.time()
        if isinstance(snap.get("lifetime"), dict):
            for k, v in snap["lifetime"].items():
                if k in eng.lifetime:
                    eng.lifetime[k] = v
        eng._g_cache = {"pc1": 0.0, "pos_manifold": 0.0,
                        "mean_r": 0.0, "lambda_ratio": 1.0, "n": 0}
        # one source of truth for all derived runtime state
        eng._init_runtime(self.cfg)
        # v1.43 — rehydrate the id->name map AND the all-time rank pool
        # from the restored board. Previously _nxer_names was only filled
        # at registration and never saved, so after a reboot the all-time
        # ranking showed "?" for every record-holder that wasn't currently
        # registered (i.e. almost all of them — the record-holders are
        # usually long dead). The board entries carry {id,name,value}, so
        # we can fully recover both the display names and the historical
        # best values that rank_of() compares against.
        for m, board in (eng.all_time or {}).items():
            pm = eng._rank_pool.setdefault(m, {})
            for e in board or []:
                try:
                    nid = e["id"]; val = e.get("value", 0.0)
                except (TypeError, KeyError):
                    continue
                nm = e.get("name")
                if nm and nm != "?":
                    eng._nxer_names.setdefault(nid, nm)
                if nid not in pm or val > pm[nid]:
                    pm[nid] = val
        # also restore the explicit id->name map saved in v1.43+
        for nid_s, nm in (snap.get("nxer_names") or {}).items():
            try:
                if nm:
                    eng._nxer_names.setdefault(int(nid_s), nm)
            except (TypeError, ValueError):
                continue
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
            # v1.53 — restore true age + within-life foraging curve. Falls
            # back to the current tick for pre-v1.53 snapshots (which never
            # stored born_tick) so a restored NxEr reads as newly born
            # rather than as old as the world.
            nx.born_tick = int(nd.get("born_tick", eng.tick) or 0)
            fba = nd.get("food_by_age") or {}
            nx.food_by_age = {int(k): v for k, v in fba.items()}
            # skip checkpoints this NxEr has already lived past, so we never
            # backfill fabricated points at the moment of reboot
            age_now = eng.tick - nx.born_tick
            nx._ck_i = sum(1 for c in _AGE_CKPTS if c <= age_now)
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
        # Opt-in snapshot worker. Spawning a separate process here is
        # the simplest architectural step toward multi-core utilisation:
        # it adds one truly-parallel Python interpreter to the picture
        # without touching the engine itself, so a bug in the worker
        # can never corrupt your world state. Enable in world_config:
        #   "snapshot_in_subprocess": true
        # v1.39: slim_broadcast does its food/colour throttling inside
        # the in-process world_snapshot(), so it requires the in-process
        # broadcast path — the subprocess worker is skipped when slim is
        # on (the two optimisations are alternatives, not stackable yet).
        if (self.cfg.get("snapshot_in_subprocess", False)
                and not self.cfg.get("slim_broadcast", False)):
            try:
                import multiprocessing as mp
                from . import snapshot_worker as _sw
                raw_recv, raw_send = mp.Pipe(duplex=False)
                bytes_recv, bytes_send = mp.Pipe(duplex=False)
                self._snap_send = raw_send
                self._snap_recv = bytes_recv
                self._snap_worker = mp.Process(
                    target=_sw.worker_main,
                    args=(raw_recv, bytes_send),
                    name="snap_worker", daemon=True)
                self._snap_worker.start()
                # Drainer thread keeps the worker→parent pipe empty so
                # the worker never blocks on a full pipe buffer. Stores
                # the latest bytes in self._latest_bytes (atomic ref).
                self._snap_reader = threading.Thread(
                    target=self._snap_reader_loop, daemon=True,
                    name="snap_reader")
                self._snap_reader.start()
                # v1.41 — pump thread owns the blocking parent->worker send
                # so the engine thread never stalls on a full pipe.
                self._snap_pump = threading.Thread(
                    target=self._snap_pump_loop, daemon=True,
                    name="snap_pump")
                self._snap_pump.start()
                print("[GameServer] snapshot worker started on PID",
                      self._snap_worker.pid)
            except Exception as e:
                print("[GameServer] snapshot worker failed to start"
                      " — falling back to in-process snapshots:", e)
                self._snap_worker = None
                self._snap_send = None
                self._snap_recv = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _snap_reader_loop(self):
        recv = self._snap_recv
        while not self._stop.is_set():
            try:
                data = recv.recv_bytes()
            except (EOFError, BrokenPipeError, OSError):
                # worker dead — the engine thread will detect the same
                # next time it tries to send and stop publishing raw
                # snapshots. We fall back to in-process JSON encoding
                # in the aiohttp broadcast loop.
                self._snap_worker = None
                return
            # Atomic reference assignment in CPython/PyPy — readers
            # never see a partial value.
            self._latest_bytes = data

    def _snap_pump_loop(self):
        """v1.41 — perform the blocking parent->worker Pipe.send() OFF the
        engine thread. The engine drops the latest raw frame into the
        1-slot mailbox (`_snap_pending`, latest-wins) and sets the event;
        this thread sends it. If the worker falls behind and the OS pipe
        buffer fills, only THIS thread blocks — the simulation keeps
        ticking and intermediate frames are dropped (the next one sent is
        always the most recent world state, which is exactly what a live
        viewer wants). This is what removes the periodic 1-2 s freezes."""
        while not self._stop.is_set():
            if not self._snap_pump_evt.wait(timeout=0.5):
                continue
            self._snap_pump_evt.clear()
            frame = self._snap_pending
            self._snap_pending = None
            if frame is None:
                continue
            send = self._snap_send
            if send is None:
                continue
            try:
                send.send(frame)
            except (BrokenPipeError, EOFError, OSError):
                self._snap_worker = None
                self._snap_send = None
                return

    def latest_bytes(self):
        """Return the most recent pre-encoded JSON broadcast frame
        from the snapshot worker, or None if the worker is disabled /
        not ready yet."""
        return self._latest_bytes

    def stop(self):
        self._stop.set()
        # flush + close the science history writer cleanly
        try:
            self.history.close()
        except Exception:
            pass
        # Shut down the snapshot worker cleanly so PyPy doesn't leak a
        # zombie process between reboots.
        if self._snap_send is not None:
            try:
                self._snap_send.send(None)
            except Exception:
                pass
        if self._snap_worker is not None:
            try:
                self._snap_worker.join(timeout=2.0)
                if self._snap_worker.is_alive():
                    self._snap_worker.terminate()
            except Exception:
                pass

    def _run(self):
        target_tps = float(self.cfg.get("target_tps", 20))
        period = 1.0 / target_tps
        snap_secs = float(self.cfg.get("snapshot_secs", 20))
        # rebuild the public broadcast snapshot at most at broadcast_hz
        # (not every tick) — saves O(N) serial work on the main core
        bcast_period = 1.0 / float(self.cfg.get("broadcast_hz", 10))
        # Auto-throttle broadcasts on big populations. world_snapshot()
        # rebuilds a public_view dict per alive NxEr each broadcast; at
        # 2500+ that becomes a real chunk of main-thread time AND a
        # large WebSocket payload. The threshold and floor are tunable.
        bcast_throttle_above = int(self.cfg.get(
            "broadcast_throttle_above_alive", 1000))
        bcast_max_period = 1.0 / float(self.cfg.get(
            "broadcast_min_hz", 3))
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
                    self._service_topo()      # v1.48 NxonKaleido refresh
                    # Effective broadcast period: linearly stretches
                    # from bcast_period to bcast_max_period as alive
                    # grows past bcast_throttle_above. Keeps the
                    # 10 Hz feel up to ~1000 alive, falls back to
                    # ~3 Hz at 4000+, with proportional interpolation
                    # in between.
                    cur_alive = sum(1 for a in self.engine.nxers.values()
                                    if a.alive)
                    if cur_alive > bcast_throttle_above:
                        frac = min(1.0,
                            (cur_alive - bcast_throttle_above) /
                            max(1.0, 3 * bcast_throttle_above))
                        eff_period = (bcast_period
                            + (bcast_max_period - bcast_period) * frac)
                    else:
                        eff_period = bcast_period
                    if t - last_snap >= eff_period:
                        # Hand off the broadcast snapshot to the worker
                        # subprocess if it's running (cheap compact
                        # tuple over a Pipe). The in-process
                        # _refresh_snapshot() still runs as a fallback
                        # so /api/snapshot and god-mode have current
                        # data even if the worker dies.
                        if self._snap_worker is not None and \
                                self._snap_send is not None:
                            # v1.41 — build the raw frame here (cheap tuple
                            # read on the engine thread) but hand the actual
                            # send to the pump thread via the latest-wins
                            # mailbox. Never block the engine on the pipe.
                            try:
                                raw = self.engine.world_snapshot_raw()
                            except Exception:
                                raw = None
                            if raw is not None:
                                self._snap_pending = raw
                                self._snap_pump_evt.set()
                        self._refresh_snapshot()
                        self._publish_metrics()
                        last_snap = t
                    if t - self._last_save >= snap_secs and not self._saving:
                        # v1.38 — the disk snapshot used to serialize the
                        # whole world (state_dict → JSON → file) inline on
                        # this thread, a 50–200 ms+ GIL hold every
                        # snapshot_secs that froze BOTH the engine loop and
                        # the aiohttp web server (the periodic "hangs on
                        # console/view"). Now we take the consistent read
                        # (state_dict) here, then JSON-encode + write on a
                        # daemon thread. That serialization runs during the
                        # engine's recv-idle windows instead of blocking it
                        # in one chunk, so the world keeps ticking and the
                        # UI stays live across a save.
                        try:
                            _sd = self.engine.state_dict()
                            self._saving = True

                            def _do_save(_d):
                                try:
                                    self.persist.save(_d)
                                except Exception as _e:
                                    print("[GameServer] snapshot failed:", _e)
                                finally:
                                    self._saving = False

                            threading.Thread(
                                target=_do_save, args=(_sd,),
                                daemon=True).start()
                            self._last_save = t
                        except Exception as e:
                            self._saving = False
                            print("[GameServer] snapshot failed:", e)
                    # Best-of archive (per-metric all-time champions).
                    # v1.43 — cadence configurable via best_archive_secs
                    # (default 300 s, was hard-coded 3600). A tighter
                    # cadence means a record-breaking NxEr is far more
                    # likely to still be alive when we sweep, so we
                    # capture its real loadable brain instead of a stub.
                    if (t - self._best_last_save >=
                            self.cfg.get("best_archive_secs", 300)):
                        try:
                            self._archive_best()
                        except Exception as e:
                            print("[GameServer] best-archive error:",
                                  e)
                        self._best_last_save = t
                    # v1.34 — periodic science sample (trinary
                    # distribution + trait/g/fitness/lifespan dists +
                    # rates). Called in the engine thread under the lock
                    # so pool.sample_firing()'s pipe round-trip is safe;
                    # it only builds + ENQUEUES one record (the file I/O
                    # is on the history background thread). Low cadence
                    # (default 60 s) so the once-a-minute worker round
                    # trip is negligible against the per-tick budget.
                    if self.history.enabled and \
                            t - self._hist_last >= self._hist_secs:
                        try:
                            self.engine.history_sample()
                        except Exception as e:
                            print("[GameServer] history sample error:",
                                  repr(e))
                        self._hist_last = t
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
        top NxEr the first hour after every reboot).

        v1.56 — a high-water mark with no corresponding record file is
        dropped. Releases v1.48-v1.55 accidentally shipped an _index.json
        from a sandbox test inside state/best/, carrying a fitness mark of
        0.9001 — effectively the saturation ceiling of the legacy metric —
        so unpacking over a live install could silently suppress archiving
        of that metric for the lifetime of the world. A mark is only
        meaningful if the record it describes is actually on disk."""
        path = os.path.join(self._best_dir, "_index.json")
        try:
            with open(path) as f:
                idx = {k: float(v) for k, v in json.load(f).items()}
        except (OSError, ValueError):
            return {}
        kept = {}
        for m, v in idx.items():
            if glob.glob(os.path.join(self._best_dir, "%s_*.json" % m)):
                kept[m] = v
        dropped = len(idx) - len(kept)
        if dropped:
            print("[best] dropped %d orphaned high-water mark(s) with no "
                  "record file; those metrics will re-archive" % dropped)
        return kept

    def _save_best_index(self):
        path = os.path.join(self._best_dir, "_index.json")
        try:
            with open(path, "w") as f:
                json.dump(self._best_saved, f)
        except OSError:
            pass

    def _brain_store_dir(self):
        d = os.path.join(self._best_dir, "brains")
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
        return d

    def _gc_brain_store(self):
        """v1.56 — delete brain blobs nothing points at any more.

        Reference counting is trivial here: a blob is live iff some
        <metric>_*.json in best/ names it in its "brain" field."""
        bdir = self._brain_store_dir()
        wanted = set()
        for f in glob.glob(os.path.join(self._best_dir, "*.json")):
            if os.path.basename(f) == "_index.json":
                continue
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            b = d.get("brain")
            if b:
                wanted.add(os.path.basename(b))
        for f in glob.glob(os.path.join(bdir, "*.json")):
            if os.path.basename(f) not in wanted:
                try:
                    os.remove(f)
                except OSError:
                    pass

    def _archive_best(self):
        """Archive the ALL-TIME record holder per metric to
        state/best/<metric>_<name>_<value>_<tick>.json, once per new
        record. Runs hourly from the tick loop under the engine lock.

        v1.43 — fixes "the best NxErs aren't saved." Uses
        `engine._record_breakers`, which captures the id+value the instant
        a live NxEr breaks a record, so genuine all-time champions (almost
        always dead by the hourly sweep) still reach the folder.

        v1.56 — fixes "best/ is full of tiny files and almost no brains."
        v1.48 introduced two optimisations that destroyed each other:
        writing metric M first deleted every `M_*.json` (to stop
        value-stamped duplicates piling up), while an NxEr holding several
        records stored its brain under ONE metric and wrote pointer stubs
        for the rest. So when a new champion took metric M, the delete also
        removed the brain that other metrics' stubs pointed at, orphaning
        them. A 10-day run ended with 1 loadable brain out of 7 records and
        3 dangling stubs — strictly worse than never de-duplicating at all
        (7 full brains is only ~4.5 MB).

        The blob is now separated from the record. Each champion brain is
        written ONCE to best/brains/<name>_t<tick>.json and every metric
        file is a small record that names it in "brain". Deleting a metric
        file can therefore never destroy a brain; blobs are removed only by
        _gc_brain_store() when no record references them. Metric files stay
        small BY DESIGN now — that is the pointer, not a failure.
        """
        from .engine import RANK_METRICS, _metric
        archived = []
        bdir = self._brain_store_dir()
        # name -> blob path, for NxErs whose brain we already stored (this
        # sweep OR any previous one — v1.48's version only looked within a
        # single sweep, so it re-exported the same brain hour after hour).
        stored = {}
        for f in glob.glob(os.path.join(bdir, "*.json")):
            base = os.path.basename(f)
            nm = base.rsplit("_t", 1)[0]
            if nm:
                stored[nm] = f
        breakers = dict(getattr(self.engine, "_record_breakers", {}) or {})
        rank_top = getattr(self.engine, "_rank_top", {}) or {}
        for m in RANK_METRICS:
            # v1.46 — `_record_breakers` is reset on reboot and only
            # repopulates when a LIVE NxEr sets a NEW record. Monotonic
            # counters (food/explored/…) repopulate immediately, but
            # plateaued metrics (mates_performed, fitness, g) whose leader
            # has stopped climbing — or has died — stayed empty, so they
            # were NEVER archived. Fall back to the current all-time #1
            # from the rank board (which survives reboots) so every metric
            # gets a champion file: a live brain if the #1 is alive, else
            # a documented stub.
            nid = value = None
            rec = breakers.get(m)
            if rec:
                nid, value = rec
            else:
                top = rank_top.get(m) or []
                if top:
                    nid, value = top[0]["id"], top[0]["value"]
            if nid is None or value is None:
                continue
            # v1.57 — require a MEANINGFUL improvement, not any improvement.
            # time_lived / explored / food_found / food_taken increase every
            # tick for any living NxEr, so a long-lived champion set a "new
            # record" on literally every sweep: the 6.87-day V1.076 run
            # archived 3,104 times (1,076 for time_lived alone), which was
            # 88% of all console output and a glob+delete+write each time.
            # A relative gain threshold collapses that to a handful of
            # writes while keeping every genuine record change.
            prev = self._best_saved.get(m)
            if prev is not None:
                need = abs(prev) * float(
                    self.cfg.get("best_min_gain", 0.02))
                if value <= prev + max(need, 1e-9):
                    continue
            elif value <= 0.0:
                continue
            a = self.engine.nxers.get(nid)
            name = (a.name if a else
                    self.engine._nxer_names.get(nid, "?"))
            # v1.56 — delete prior file(s) for THIS metric only. Safe now:
            # metric files never hold the only copy of a brain, so removing
            # one cannot orphan another metric's record.
            for old in glob.glob(os.path.join(self._best_dir,
                                              f"{m}_*.json")):
                try:
                    os.remove(old)
                except OSError:
                    pass
            # v1.56 — store the brain ONCE in best/brains/, then point at
            # it. One NxEr holding several records gets one blob and
            # several small records, none of which can dangle.
            blob = stored.get(name)
            if blob is None and a is not None and a.alive:
                try:
                    _mdl = self.engine.export_model_for(a)
                    blob = os.path.join(
                        bdir, f"{name}_t{self.engine.tick}.json")
                    with open(blob, "w") as bf:
                        json.dump(_mdl, bf)
                    stored[name] = blob
                except Exception as e:
                    print("[GameServer] best-archive export failed:", e)
                    blob = None
            model = {
                "name": name,
                "record": {"metric": m, "value": value,
                           "tick": self.engine.tick},
            }
            if blob is not None:
                model["brain"] = os.path.join(
                    "brains", os.path.basename(blob))
                model["note"] = (
                    f"full brain stored once at best/{model['brain']} "
                    f"(shared by every record this NxEr holds).")
            else:
                # champion died before this sweep — document the record so
                # it is never lost or shown as "?"
                model["note"] = ("all-time record holder; brain unavailable "
                                 "(NxEr died before the archive sweep). For "
                                 "a loadable brain, raise the archive "
                                 "cadence.")
            v_disp = f"{value:.3f}".replace(".", "_")
            fn = f"{m}_{name}_{v_disp}_t{self.engine.tick}.json"
            fpath = os.path.join(self._best_dir, fn)
            try:
                with open(fpath, "w") as f:
                    json.dump(model, f)
                self._best_saved[m] = value
                archived.append((m, name, value,
                                 bool(a and a.alive)))
            except OSError as e:
                print("[GameServer] best-archive write failed:", e)
        if archived:
            self._save_best_index()
            for m, nm, v, live in archived:
                tag = "live brain" if live else "record only (died)"
                print(f"[best] archived {m}: {nm} = {v:.3f} [{tag}]")
        # v1.56 — drop blobs no record points at any more (an NxEr that
        # lost every one of its records). Cheap: a handful of files.
        try:
            self._gc_brain_store()
        except Exception:
            pass

    def _repair_best_dir(self):
        """v1.56 — one-off migration for folders written by v1.48-v1.55.

        Those versions wrote pointer stubs ({"see": <metric>}) that were
        orphaned whenever the referenced metric changed hands, because
        taking a record deleted the previous holder's file — which was
        sometimes the only copy of a brain. Real 10-day folders ended up
        with 1 loadable brain and 3 dangling stubs.

        This scans best/, rewrites any legacy record into the new pointer
        form, and reports what is recoverable. A dangling record whose NxEr
        is still alive is simply re-archived by the next sweep (its value
        is cleared from the index so the sweep re-fires); one whose NxEr is
        long dead keeps its documented record with no brain, which is the
        best that can be done — the brain no longer exists anywhere."""
        if not os.path.isdir(self._best_dir):
            return
        fixed = dangling = ok = 0
        for f in glob.glob(os.path.join(self._best_dir, "*.json")):
            if os.path.basename(f) == "_index.json":
                continue
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            if "see" not in d:
                if d.get("brain") or "record" in d:
                    ok += 1
                continue
            # legacy stub -> did its target survive?
            nm = d.get("name")
            tgt = glob.glob(os.path.join(
                self._best_dir, "%s_%s_*.json" % (d["see"], nm)))
            live = [t for t in tgt if os.path.getsize(t) > 10000]
            d.pop("see", None)
            if live:
                # promote the legacy full file into the brain store
                blob = os.path.join(
                    self._brain_store_dir(), "%s_legacy.json" % nm)
                try:
                    if not os.path.exists(blob):
                        with open(live[0]) as sf, open(blob, "w") as df:
                            df.write(sf.read())
                    d["brain"] = os.path.join("brains",
                                              os.path.basename(blob))
                    d["note"] = ("full brain stored once at best/%s "
                                 "(migrated from v1.48 layout)."
                                 % d["brain"])
                    fixed += 1
                except OSError:
                    dangling += 1
            else:
                d["note"] = ("record preserved; brain was lost by the "
                             "pre-v1.56 archive layout (the file it "
                             "pointed to was deleted when another NxEr "
                             "took that record).")
                dangling += 1
                # let the next sweep re-archive if this NxEr still lives
                mt = (d.get("record") or {}).get("metric")
                if mt:
                    self._best_saved.pop(mt, None)
            try:
                with open(f, "w") as fh:
                    json.dump(d, fh)
            except OSError:
                pass
        if fixed or dangling:
            print("[best] v1.56 repair: %d record(s) relinked, %d had no "
                  "recoverable brain (will re-archive if the NxEr lives), "
                  "%d already fine" % (fixed, dangling, ok))
            self._save_best_index()

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

    def get_family(self, name):
        """Return the lineage view for an owned NxEr: parents (if any)
        plus the direct offspring. Used by /api/mynxer/family. Does
        NOT include grand-children — the spec is one generation +
        click-through, not a full tree dump."""
        with self._lock:
            a = self.find_nxer_by_name(name)
            if not a:
                return None
            def _mini(nid):
                o = self.engine.nxers.get(nid)
                return {
                    "id": nid,
                    "name": o.name if o else
                            self.engine._nxer_names.get(nid, "?"),
                    "alive": bool(o and o.alive),
                    "managed": bool(o and o.is_managed),
                    "born_tick": getattr(o, "born_tick", None),
                }
            parents = [_mini(pid) for pid in (a.parents or [])
                       if pid is not None]
            children = [_mini(cid) for cid in (a.offspring_ids or [])]
            return {
                "self": {"id": a.id, "name": a.name, "alive": a.alive,
                         "born_tick": getattr(a, "born_tick", None)},
                "parents": parents,
                "children": children,
                # true while the owner's NxEr is alive — the client
                # uses this to grey out child-inspection if it died.
                "can_inspect_children": bool(a.alive),
            }

    def get_child_view(self, parent_name, child_name):
        """Return the owner-style view of a CHILD of `parent_name`.
        Returns:
          ("ok", view_dict)            on success
          ("parent_dead", None)        if the original is no longer alive
          ("not_a_child", None)        if name is not in offspring_ids
          ("not_found", None)          if either NxEr is missing
        """
        with self._lock:
            parent = self.find_nxer_by_name(parent_name)
            if not parent:
                return "not_found", None
            if not parent.alive:
                return "parent_dead", None
            child = self.find_nxer_by_name(child_name)
            if not child:
                return "not_found", None
            if child.id not in (parent.offspring_ids or []):
                return "not_a_child", None
            d = child.owner_view()
            try:
                d["ranks"] = self.engine.rank_of(child)
                d["brain_building"] = self.engine._brain_building(child)
            except Exception:
                d["ranks"] = {}
            return "ok", d

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

    def request_topology(self, name):
        """v1.51 — web handler marks a NxEr (by NAME) as wanted by an open
        viewer. Pure attribute write — NEVER touches the engine lock, so it
        can't block the asyncio event loop (which also serves the world
        WebSocket). The game loop resolves the name and does the fetch."""
        self._topo_want = (name, time.time())

    def get_topology(self, name):
        """v1.51 — web handler reads the cached topology for this NAME if
        fresh; returns the cache (or a {'dead':True} marker) so the handler
        never needs the lock or a NxEr lookup."""
        c = self._topo_cache
        if c and c.get("name") == name and (time.time() - c["ts"]) < 4.0:
            return c
        return None

    def _service_topo(self):
        """Called from the game loop under the lock (so name resolution and
        the worker pipes are both safe here). Refreshes the cached topology
        ~1/s, but ONLY while a viewer polled within the last 6 s — otherwise
        it does nothing, so an idle world pays no cost."""
        want = self._topo_want
        if not want:
            return
        name, asked = want
        if time.time() - asked > 6.0:       # viewer went away
            self._topo_want = None
            return
        c = self._topo_cache
        if (c and c.get("name") == name
                and (time.time() - c["ts"]) < 0.8):
            return                          # cache still fresh enough
        # resolve name -> live NxEr (safe: we hold the lock)
        nx = None
        for a in self.engine.nxers.values():
            if a.name == name and a.alive:
                nx = a
                break
        if nx is None:                      # dead / unknown -> 404 marker
            self._topo_cache = {"name": name, "dead": True,
                                "ts": time.time()}
            return
        nid = nx.id
        try:
            topo = self.engine.pool.brain_topology(nid)
        except Exception:
            topo = None
        if topo is not None:
            try:
                events = self._topo_events(nid)
            except Exception:
                events = []
            self._topo_cache = {"name": name, "id": nid,
                                "tick": self.engine.tick,
                                "ts": time.time(), "topo": topo,
                                "events": events}

    def _topo_events(self, nid):
        """v1.51 — derive what the NxEr is experiencing right now (sees
        food, ate, hungry, mated, sang, stole) from its live game state, so
        NxonKaleido can flow the matching emoji from the right sphere to the
        cortex. Sensory states come from the last sensory vector; discrete
        actions are caught by short tick-windows or by deltas vs the last
        refresh."""
        nx = self.engine.nxers.get(nid)
        if nx is None or not nx.alive:
            self._topo_prev.pop(nid, None)
            return []
        ev = []
        tick = self.engine.tick
        first = nid not in self._topo_prev
        prev = self._topo_prev.get(nid, {})
        cap = float(self.engine.cfg.get("max_nxer_energy", 150)) or 150.0
        sens = getattr(nx, "_last_sensory", None)
        # sensory-sphere signals
        sees = False
        if sens is not None and len(sens) >= 6:
            try:
                sees = (abs(float(sens[1])) + abs(float(sens[2])) > 0.05
                        or float(sens[4]) > 0.5 or float(sens[5]) > 0.15)
            except (TypeError, ValueError):
                sees = False
        if sees:
            ev.append({"e": "food", "s": "sensory"})
        if float(getattr(nx, "food", cap)) < 0.4 * cap:
            ev.append({"e": "hungry", "s": "sensory"})
        # motor-sphere actions
        if tick - int(getattr(nx, "last_eat_tick", -10**9)) <= 3:
            ev.append({"e": "ate", "s": "motor"})
        if tick - int(getattr(nx, "last_steal_tick", -10**9)) <= 3:
            ev.append({"e": "steal", "s": "motor"})
        if float(getattr(nx, "last_sing_level", 0.0) or 0.0) > 0.1:
            ev.append({"e": "sing", "s": "motor"})
        mcu = int(getattr(nx, "mate_cooldown_until", 0))
        if not first and mcu > int(prev.get("mcu", 0)):
            ev.append({"e": "mate", "s": "motor"})
        self._topo_prev[nid] = {"mcu": mcu,
                                "noff": len(nx.offspring_ids)}
        return ev

    def owner_session_name(self, token):
        """Back-compat single-name accessor: the FIRST still-alive NxEr
        this browser token owns, else None."""
        names = self.owner_session_names(token)
        return names[0] if names else None

    def owner_session_names(self, token):
        """v1.45 — every still-alive NxEr name this browser token owns
        (so the client can auto-reconnect to all of them without a
        password). Order preserved; dead ones dropped."""
        if not token:
            return []
        names = self._owner_sessions.get(token) or []
        return [n for n in names if n in self._owner_views]

    def owner_live_name(self, token):
        """Back-compat: first NxEr (alive or just-queued) for this token."""
        names = self.owner_live_names(token)
        return names[0] if names else None

    def owner_live_names(self, token):
        """v1.45 — lock-free list of this token's NxErs that are alive OR
        queued for creation. Used by register to enforce the per-user cap
        WITHOUT touching the engine lock, so registration stays instant."""
        if not token:
            return []
        names = list(self._owner_sessions.get(token) or [])
        if not names:
            return []
        queued = set()
        with self._reg_lock:
            for n, _, _, _ in self._reg_queue:
                queued.add(n)
        live = []
        for n in names:
            if n in self._owner_views or n in queued:
                live.append(n)
        return live

    def owner_live_count(self, token):
        return len(self.owner_live_names(token))

    def bind_owner_session(self, token, name):
        """v1.45 — append a NxEr to this token's owned set (dedup). The
        stored list keeps every name (a just-registered NxEr is still
        only QUEUED, not yet in the live-view cache, so we must NOT drop
        it here or the per-user cap can never accumulate). Liveness is
        filtered at read time by owner_live_names / owner_session_names;
        the raw list is length-capped so dead names can't pile up."""
        if not token or not name:
            return self.owner_live_names(token)
        cur = list(self._owner_sessions.get(token) or [])
        if name not in cur:
            cur.append(name)
        if len(cur) > 12:
            cur = cur[-12:]
        self._owner_sessions[token] = cur
        return self.owner_live_names(token)
