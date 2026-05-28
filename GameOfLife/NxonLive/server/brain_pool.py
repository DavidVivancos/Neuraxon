# Multi Neuraxon Game of Life 5 — multi-core brain pool  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# THE performance fix. The forced CHC 6-sphere brain costs ~8 ms/step;
# stepping dozens of them sequentially in one Python thread pegs a
# single core (GIL) and starves the web server → slow TPS + a client
# that never receives data.
#
# Brain steps within a tick are independent (sensing is read-only; the
# only world mutation, _act, is cheap and stays serial). So we run the
# brain phase across N persistent worker PROCESSES — real parallelism,
# no GIL. Brains live ONLY in their worker (sharded by id % N) and are
# never pickled per tick: only small sensory/motor vectors cross the
# pipe. While the main process blocks on pipe.recv() the GIL is free,
# so aiohttp stays responsive and the client loads normally.
#
# A worker count of 0/1 falls back to a correct in-process path (works
# on a 1-core box and as a safety net).
# ===================================================================
import os
import multiprocessing as mp


# --- worker -------------------------------------------------------
def _builder_main(conn):
    """Dedicated brain-construction process. build_brain() costs
    ~250 ms (heavy CHC topology + init); doing it on a step worker
    stalls that shard's next step (single-threaded recv loop) which
    freezes the whole tick for everyone. This process does ONLY that
    build, fully in parallel, and ships the finished brain back as a
    dict — load_multisphere_from_dict is ~1 ms so the step worker
    applies it inline with no stall."""
    os.environ.setdefault("NEURAXON_HEADLESS", "1")
    import sys as _sys
    import os as _os2
    _sys.path.insert(0, _os2.path.dirname(
        _os2.path.dirname(_os2.path.abspath(__file__))))
    from server import np_fallback
    np_fallback.install()
    import architecture
    _af = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "architecture_files", "nas_best.json")
    try:
        architecture.load_architecture(_af, verbose=False)
    except Exception:
        architecture._ARCH = {}
        architecture._ARCH_PATH = None
    from neuraxon.multisphere import build_brain
    from server.engine import make_params
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg[0] == "stop":
            break
        if msg[0] == "build":
            _, i, params_dict = msg
            try:
                b = build_brain(make_params(params_dict))
            except Exception:
                b = build_brain(make_params())
            try:
                conn.send(("built", i, b.to_dict()))
            except (BrokenPipeError, OSError):
                break
    conn.close()


def _worker_main(conn):
    os.environ.setdefault("NEURAXON_HEADLESS", "1")
    # numpy shim BEFORE anything imports the substrate (PyPy fix)
    import sys as _sys
    import os as _os2
    _sys.path.insert(0, _os2.path.dirname(
        _os2.path.dirname(_os2.path.abspath(__file__))))
    from server import np_fallback
    np_fallback.install()
    import architecture
    _af = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "architecture_files", "nas_best.json")
    try:
        architecture.load_architecture(_af, verbose=False)
    except Exception:
        architecture._ARCH = {}
        architecture._ARCH_PATH = None
    from neuraxon.multisphere import load_multisphere_from_dict

    brains = {}          # id -> NeuraxonMultiSphere
    inputs = {}          # id -> [input_neuron_id, ...] (cached)

    def _ensure_inputs(i, b):
        if i not in inputs:
            inputs[i] = [n.id for n in
                         b.sensory_sphere.network.input_neurons]
        return inputs[i]

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        op = msg[0]
        if op == "stop":
            break
        elif op == "load":
            # the dedicated builder process did the heavy ~250 ms
            # build; rehydrating from the dict is ~1 ms so applying it
            # inline here never stalls this shard's step loop.
            _, i, bdict = msg
            try:
                brains[i] = load_multisphere_from_dict(bdict)
            except Exception:
                pass
            inputs.pop(i, None)
        elif op == "del":
            brains.pop(msg[1], None)
            inputs.pop(msg[1], None)
        elif op == "export":
            b = brains.get(msg[1])
            conn.send(b.to_dict() if b is not None else None)
        elif op == "step":
            out = []
            for i, sens in msg[1]:
                b = brains.get(i)
                if b is None:
                    out.append((i, [0] * 7, 1.0))   # brain not loaded yet
                    continue
                try:
                    ids = _ensure_inputs(i, b)
                    ext = {"sensory": {
                        ids[k]: (sens[k] if k < len(sens) else 0.0)
                        for k in range(len(ids))}}
                    b.simulate_step(ext)
                    outs = b.motor_sphere.network.get_output_states()
                    br = getattr(b.motor_sphere.network,
                                 "branching_ratio", 1.0)
                    out.append((i, list(outs), float(br)))
                except Exception:
                    out.append((i, [0] * 7, 1.0))
            conn.send(out)
        else:
            conn.send(("err", "unknown op"))
    conn.close()


# --- pool ---------------------------------------------------------
class BrainPool:
    """Sharded persistent worker processes. id -> shard = id % N."""

    def __init__(self, num_workers, num_builders=None, step_timeout=5.0):
        self.n = max(0, int(num_workers))
        self._parents = []
        self._procs = []
        # builder POOL — was a single process, which serialised all
        # brain construction (34 ms each) and left most cores idle on
        # a big machine. Now several builders construct brains in
        # parallel; finished brains are pumped back and loaded into
        # the owning step shard. Default scales with worker count.
        if num_builders is None:
            num_builders = max(1, min(4, self.n // 3))
        self._n_builders = int(num_builders)
        self._builder_parents = []
        self._builder_procs = []
        self._builder_rr = 0          # round-robin dispatch cursor
        self._pumps = []
        self._step_timeout = float(step_timeout)
        self._closing = False
        self._load_q = []
        import threading as _th
        self._load_lock = _th.Lock()
        if self.n >= 2:
            ctx = mp.get_context("fork")
            for _ in range(self.n):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker_main, args=(child,),
                                daemon=True)
                p.start()
                child.close()
                self._parents.append(parent)
                self._procs.append(p)
            # dedicated builder POOL: each does ONLY the heavy
            # build_brain, fully in parallel with stepping AND with
            # each other. Each ships finished brains back through its
            # own pump thread, which queues a ~1 ms "load" to the
            # owning shard. No build ever touches a step worker or the
            # game loop → creation never freezes the world.
            import threading
            for _ in range(self._n_builders):
                bp_parent, bp_child = ctx.Pipe()
                proc = ctx.Process(
                    target=_builder_main, args=(bp_child,), daemon=True)
                proc.start()
                bp_child.close()
                self._builder_parents.append(bp_parent)
                self._builder_procs.append(proc)
                pump = threading.Thread(
                    target=self._pump_built, args=(bp_parent,),
                    daemon=True)
                pump.start()
                self._pumps.append(pump)
        else:
            # in-process fallback (1-core / safety): keep brains here
            os.environ.setdefault("NEURAXON_HEADLESS", "1")
            from server import np_fallback
            np_fallback.install()
            import architecture
            _af = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))),
                "architecture_files", "nas_best.json")
            try:
                architecture.load_architecture(_af, verbose=False)
            except Exception:
                architecture._ARCH = {}
                architecture._ARCH_PATH = None
            from neuraxon.multisphere import (
                build_brain, load_multisphere_from_dict)
            self._bb = build_brain
            self._lb = load_multisphere_from_dict
            self._brains = {}
            self._inputs = {}

    def _pump_built(self, builder_parent):
        """Daemon (one per builder): drain finished brains from THIS
        builder and QUEUE them. The game-loop thread is the only
        writer of the worker pipes (it drains this queue at the top of
        each tick), so there is never a concurrent send on a shard
        pipe. A 34 ms build still never delays a tick."""
        while not self._closing:
            try:
                msg = builder_parent.recv()
            except (EOFError, OSError, BrokenPipeError, ValueError):
                break
            if not msg or msg[0] != "built":
                continue
            _, i, bdict = msg
            with self._load_lock:
                self._load_q.append((i, bdict))

    def drain_loads(self):
        """Called by the game loop (sole worker-pipe writer) each tick:
        push any builder-finished brains into their shard as a ~1 ms
        load. Bounded per tick so a burst can't stall a step."""
        if not self.parallel:
            return
        with self._load_lock:
            if not self._load_q:
                return
            cap = max(8, 4 * self._n_builders)
            batch = self._load_q[:cap]
            del self._load_q[:cap]
        for i, bdict in batch:
            k = self._shard(i)
            try:
                self._parents[k].send(("load", i, bdict))
            except (BrokenPipeError, OSError, ValueError):
                pass

    @property
    def parallel(self):
        return self.n >= 2

    def _shard(self, i):
        return i % self.n

    def _safe_call(self, i, payload, default=None):
        c = self._parents[self._shard(i)]
        try:
            c.send(payload)
            return c.recv()
        except (BrokenPipeError, OSError, EOFError, ValueError):
            return default

    # ---- lifecycle ----
    def _fire(self, i, payload):
        """Send a mutating op with NO ack (workers are silent for
        add/del/load and build brains deferred). register/loop never
        block on a CHC brain build."""
        k = self._shard(i)
        try:
            self._parents[k].send(payload)
        except (BrokenPipeError, OSError, ValueError):
            pass

    def add(self, i, params_dict):
        if self.parallel:
            # fire to the next builder (round-robin); it builds in
            # parallel and its pump thread loads the result into the
            # shard. Instant, never blocks register, the loop, or any
            # step worker. Spreading across builders means construction
            # uses many cores, not one.
            try:
                bp = self._builder_parents[
                    self._builder_rr % len(self._builder_parents)]
                self._builder_rr += 1
                bp.send(("build", i, params_dict))
            except (BrokenPipeError, OSError, ValueError):
                pass
        else:
            from server.engine import make_params
            try:
                self._brains[i] = self._bb(make_params(params_dict))
            except Exception:
                self._brains[i] = self._bb(make_params())
            self._inputs.pop(i, None)

    def load(self, i, bdict):
        if self.parallel:
            self._fire(i, ("load", i, bdict))
        else:
            try:
                self._brains[i] = self._lb(bdict)
                self._inputs.pop(i, None)
            except Exception:
                pass

    def remove(self, i):
        if self.parallel:
            self._fire(i, ("del", i))
        else:
            self._brains.pop(i, None)
            self._inputs.pop(i, None)

    def export(self, i):
        if self.parallel:
            return self._safe_call(i, ("export", i), default=None)
        b = self._brains.get(i)
        return b.to_dict() if b is not None else None

    # ---- the hot path: parallel brain step ----
    def step(self, batch):
        """batch = [(id, sens_list), ...] for all alive NxErs.
        Returns {id: (motor_list, branching)}."""
        if not batch:
            return {}
        if self.parallel:
            shards = [[] for _ in range(self.n)]
            for i, sens in batch:
                shards[i % self.n].append((i, sens))
            # fan out to all workers, THEN collect → true concurrency
            sent = [False] * self.n
            for k, sh in enumerate(shards):
                if sh:
                    try:
                        self._parents[k].send(("step", sh))
                        sent[k] = True
                    except (BrokenPipeError, OSError, EOFError):
                        sent[k] = False
            res = {}
            for k, sh in enumerate(shards):
                if not sh:
                    continue
                if sent[k]:
                    try:
                        # poll with a timeout so a single stuck worker
                        # (e.g. a pathological brain whose simulate_step
                        # spins) can't freeze the whole game loop. If it
                        # doesn't answer in time we use idle defaults for
                        # that shard this tick and carry on.
                        if self._parents[k].poll(self._step_timeout):
                            for i, outs, br in self._parents[k].recv():
                                res[i] = (outs, br)
                            continue
                    except (BrokenPipeError, OSError, EOFError,
                            ValueError):
                        pass
                # worker unreachable/slow → safe defaults (idle this
                # tick); the world keeps running, anti-extinction holds
                for i, _ in sh:
                    res[i] = ([0] * 7, 1.0)
            return res
        # in-process fallback
        res = {}
        for i, sens in batch:
            b = self._brains.get(i)
            if b is None:
                res[i] = ([0] * 7, 1.0)
                continue
            try:
                if i not in self._inputs:
                    self._inputs[i] = [
                        n.id for n in
                        b.sensory_sphere.network.input_neurons]
                ids = self._inputs[i]
                ext = {"sensory": {
                    ids[k]: (sens[k] if k < len(sens) else 0.0)
                    for k in range(len(ids))}}
                b.simulate_step(ext)
                outs = b.motor_sphere.network.get_output_states()
                br = getattr(b.motor_sphere.network,
                             "branching_ratio", 1.0)
                res[i] = (list(outs), float(br))
            except Exception:
                res[i] = ([0] * 7, 1.0)
        return res

    def close(self):
        self._closing = True
        for bp in self._builder_parents:
            try:
                bp.send(("stop",))
                bp.close()
            except Exception:
                pass
        for proc in self._builder_procs:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
        if self.parallel:
            for c in self._parents:
                try:
                    c.send(("stop",))
                    c.close()
                except Exception:
                    pass
            for p in self._procs:
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()
