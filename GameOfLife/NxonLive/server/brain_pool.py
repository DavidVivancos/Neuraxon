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
import math
import multiprocessing as mp


# --- worker -------------------------------------------------------
def _set_affinity(pid, cores):
    """Pin a process to a set of CPU cores (Linux). No-op where
    os.sched_setaffinity is unavailable (macOS/Windows)."""
    if not cores:
        return
    fn = getattr(os, "sched_setaffinity", None)
    if fn is None:
        return
    try:
        fn(pid, set(cores))
    except (OSError, ValueError):
        pass


def _builder_main(conn):
    """Dedicated brain-construction process. build_brain() costs
    ~250 ms (heavy CHC topology + init); doing it on a step worker
    stalls that shard's next step (single-threaded recv loop) which
    freezes the whole tick for everyone. This process does ONLY that
    build, fully in parallel, and ships the finished brain back as a
    dict — load_multisphere_from_dict is ~1 ms so the step worker
    applies it inline with no stall.

    v1.37: builders are pinned to a RESERVED core set and run at a
    lower OS priority (nice), so a burst of user NxEr creations builds
    on dedicated cores and can never starve the step-workers or the web
    server. This is the 'creation on separate reserved cores' fix."""
    try:
        os.nice(10)               # yield to engine/web/step-workers
    except (OSError, AttributeError):
        pass
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


def _brain_firing_counts(b):
    """Count trinary states across every neuron of a brain.
    Returns (excitatory, neutral, inhibitory, total). Defensive: works
    regardless of internal naming, returns zeros if structure differs.
    This is the live trinary firing distribution — the scientific core
    the NAS (v195) optimises toward the paper's 0.22 / 0.68 / 0.10."""
    exc = neu = inh = n = 0
    try:
        spheres = getattr(b, "spheres", {}) or {}
        for sphere in spheres.values():
            net = getattr(sphere, "network", None)
            if net is None:
                continue
            for neuron in getattr(net, "all_neurons", ()) or ():
                st = getattr(neuron, "trinary_state", None)
                if st == 1:
                    exc += 1
                elif st == -1:
                    inh += 1
                elif st == 0:
                    neu += 1
                else:
                    continue
                n += 1
    except Exception:
        pass
    return exc, neu, inh, n


def _brain_science_into(b, acc):
    """v1.44 — single defensive pass over ONE brain, accumulating its
    contributions to the offline GoL M-metrics into the shard dict `acc`.
    Runs only at history cadence (~1/min) inside the 'sci' op, folded into
    the same neuron walk `firestat` already does, so the marginal cost is
    a handful of extra adds per neuron once a minute. Every section is
    wrapped so a structural difference degrades that metric to "absent"
    rather than killing the whole sample (same philosophy as firestat).

    Metrics gathered here (the brain-internal ones):
      M1  trinary E/I/Neutral      — neuron.trinary_state
      M2  CTC inter-sphere gate    — link._communication_gate (instantaneous)
      M5  branching ratio sigma    — net.branching_ratio
      M6  ACW heterogeneity        — neuron.intrinsic_timescale spread
      M8  sphere specialisation    — active fraction per sphere role
      M10 lesion robustness        — dead-neuron fraction + motor amplitude
    (M7/M9 are behavioural and accumulated in the engine; M3/M4 are
    offline-harness only — see PERF_ISOLATION_SCOPE / changelog.)
    """
    spheres = getattr(b, "spheres", {}) or {}
    acc["n_brains"] += 1

    # ---- per-sphere neuron walk: M1, M6 ACW, M8 role activity, M10 ----
    total_n = 0
    dead_n = 0
    for sid, sphere in spheres.items():
        net = getattr(sphere, "network", None)
        if net is None:
            continue
        role = ("sensory" if "sensory" in sid else
                "motor" if "motor" in sid else "assoc")
        sph_fire = sph_tot = 0
        for nrn in getattr(net, "all_neurons", ()) or ():
            st = getattr(nrn, "trinary_state", None)
            active = getattr(nrn, "is_active", True)
            total_n += 1
            if not active:
                dead_n += 1
                continue
            # M1 trinary
            if st == 1:
                acc["e"] += 1
            elif st == -1:
                acc["h"] += 1
            elif st == 0:
                acc["z"] += 1
            if st in (1, -1, 0):
                acc["nneu"] += 1
            # M8 sphere activity (fraction firing within this sphere)
            sph_tot += 1
            if st in (1, -1):
                sph_fire += 1
            # M6 ACW — intrinsic timescale spread
            ts = getattr(nrn, "intrinsic_timescale", None)
            if ts is not None:
                acc["ts_sum"] += ts
                acc["ts_sq"] += ts * ts
                acc["ts_n"] += 1
        if sph_tot > 0:
            frac = sph_fire / sph_tot
            acc[role + "_act"] += frac
            acc[role + "_n"] += 1

    # ---- M10 lesion: dead-neuron fraction + motor amplitude bucket ----
    try:
        mnet = getattr(b, "motor_sphere", None)
        mnet = getattr(mnet, "network", None)
        outs = list(getattr(mnet, "output_neurons", ()) or ())
        amp = 0.0
        na = 0
        for nrn in outs:
            if getattr(nrn, "is_active", True):
                amp += abs(getattr(nrn, "trinary_state", 0) or 0)
                na += 1
        motor_amp = (amp / na) if na else 0.0
        dead_frac = (dead_n / total_n) if total_n else 0.0
        acc["dead_sum"] += dead_frac
        acc["dead_n"] += 1
        if dead_frac < 0.10:
            acc["mamp_ok_sum"] += motor_amp
            acc["mamp_ok_n"] += 1
        elif dead_frac >= 0.25:
            acc["mamp_les_sum"] += motor_amp
            acc["mamp_les_n"] += 1
    except Exception:
        pass

    # ---- M5 branching ratio (one scalar per brain) ----
    try:
        br = getattr(getattr(b, "motor_sphere", None), "network", None)
        br = getattr(br, "branching_ratio", None)
        if br is None:
            for sphere in spheres.values():
                net = getattr(sphere, "network", None)
                if net is not None and getattr(net, "branching_ratio", 0) > 0:
                    br = net.branching_ratio
                    break
        if br and br > 0:
            acc["br_sum"] += br
            acc["br_sq"] += br * br
            acc["br_n"] += 1
            if br < 0.92:
                acc["br_sub"] += 1
            elif br > 1.10:
                acc["br_sup"] += 1
    except Exception:
        pass

    # ---- M2 CTC inter-sphere gate (instantaneous, per link) ----
    try:
        for link in (getattr(b, "links", {}) or {}).values():
            src = spheres.get(getattr(link, "source_sphere_id", None))
            tgt = spheres.get(getattr(link, "target_sphere_id", None))
            if src is None or tgt is None:
                continue
            g = float(link._communication_gate(src.network, tgt.network))
            acc["gate_sum"] += g
            acc["gate_sq"] += g * g
            acc["gate_n"] += 1
    except Exception:
        pass

    # ---- v1.53: sampled synaptic weight scan -----------------------
    # The 44-day run showed the excitatory fraction collapsing
    # monotonically (M1_E 0.103 -> 0.033) with the inter-sphere gate
    # rock-steady, i.e. the brains went quiet without the gating
    # changing. We could not tell whether plasticity was net-depressing
    # the weights because weights were never logged. This samples up to
    # 200 synapses per sphere at the existing ~1/min science cadence,
    # so the next run can correlate weight drift against M1_E directly.
    try:
        for sph in spheres.values():
            syns = getattr(sph.network, "synapses", None)
            if not syns:
                continue
            acc["w_total"] += len(syns)      # v1.57 — true synapse count
            step = 1 if len(syns) <= 200 else len(syns) // 200
            for i in range(0, len(syns), step):
                s = syns[i]
                if getattr(s, "integrity", 1) <= 0:
                    continue
                w = float(getattr(s, "w_fast", 0.0))
                acc["w_sum"] += w
                acc["w_abs"] += w if w >= 0 else -w
                acc["w_sq"] += w * w
                acc["w_n"] += 1
                if w > 0:
                    acc["w_pos"] += 1
    except Exception:
        pass


def _brain_topology(b):
    """v1.48 — extract ONE brain's connectivity + live activity for the
    client-side NxonKaleido visualiser. Cheap: a handful of spheres, one
    pass over each sphere's neurons for the firing fractions, and the
    inter-sphere communication gates. ALL rendering/animation happens
    client-side from this compact JSON — the server only exposes the
    structure + current activity, never computes the visual."""
    spheres = getattr(b, "spheres", {}) or {}
    out_spheres = []
    idx = {}
    for sid, sphere in spheres.items():
        idx[sid] = len(out_spheres)
        net = getattr(sphere, "network", None)
        role = ("sensory" if "sensory" in sid else
                "motor" if "motor" in sid else "assoc")
        e = z = h = n = 0
        if net is not None:
            for nrn in getattr(net, "all_neurons", ()) or ():
                if not getattr(nrn, "is_active", True):
                    continue
                st = getattr(nrn, "trinary_state", 0)
                n += 1
                if st == 1:
                    e += 1
                elif st == -1:
                    h += 1
                else:
                    z += 1
        out_spheres.append({
            "id": sid, "role": role, "n": n,
            "exc": round(e / n, 3) if n else 0.0,
            "inh": round(h / n, 3) if n else 0.0,
            "act": round((e + h) / n, 3) if n else 0.0,
        })
    out_links = []
    for link in (getattr(b, "links", {}) or {}).values():
        s = getattr(link, "source_sphere_id", None)
        t = getattr(link, "target_sphere_id", None)
        if s not in idx or t not in idx:
            continue
        try:
            g = float(link._communication_gate(spheres[s].network,
                                                spheres[t].network))
        except Exception:
            g = 0.5
        out_links.append({"s": idx[s], "t": idx[t], "g": round(g, 3)})
    return {"spheres": out_spheres, "links": out_links}


def _m_from_acc(a):
    """v1.54 — turn ONE brain's science accumulator into that brain's own
    M-metric values. Same formulas compute_m12() uses on the population
    total; the difference is simply that this is not summed across brains.

    Until now every per-brain value was folded into a shard accumulator and
    discarded, so the M claims existed only as a population average and no
    individual brain could be ranked by them. That made it impossible to ask
    which architectures produce M-compliant brains, or to select for them."""
    out = {}
    nneu = a.get("nneu", 0)
    if nneu > 0:
        out["M1_E"] = a["e"] / nneu
        out["M1_I"] = a["h"] / nneu
        out["M1_N"] = a["z"] / nneu
    if a.get("br_n", 0) > 0:
        out["M5_branching"] = a["br_sum"] / a["br_n"]
    if a.get("ts_n", 0) > 1:
        tn = a["ts_n"]
        mts = a["ts_sum"] / tn
        out["M6_acw_heterogeneity"] = math.sqrt(
            max(0.0, a["ts_sq"] / tn - mts * mts))
    if a.get("gate_n", 0) > 0:
        gn = a["gate_n"]
        mg = a["gate_sum"] / gn
        out["M2_mean_gate"] = mg
        out["M2_gate_xlink_std"] = math.sqrt(
            max(0.0, a["gate_sq"] / gn - mg * mg))
    if a.get("sensory_n", 0) and a.get("assoc_n", 0):
        out["M8_sensory_vs_assoc"] = (a["sensory_act"] / a["sensory_n"]
                                      - a["assoc_act"] / a["assoc_n"])
    if a.get("dead_n", 0) > 0:
        out["M10_dead_neuron_frac"] = a["dead_sum"] / a["dead_n"]
    if a.get("mamp_ok_n", 0) and a.get("mamp_les_n", 0):
        ok = a["mamp_ok_sum"] / a["mamp_ok_n"]
        if ok > 0.05:
            out["M10_lesion_retention"] = (a["mamp_les_sum"]
                                           / a["mamp_les_n"]) / ok
    # v1.55 — per-brain synaptic weight magnitude. Population-level W_*
    # showed mean |w| tripling toward the +/-1 clip while mean w stayed
    # at 0, which tracked the excitatory collapse at r = -0.87. Exposing
    # it per brain lets the NAS correlate architecture against weight
    # runaway directly, instead of only seeing its downstream symptom.
    if a.get("w_n", 0) > 0:
        wn = a["w_n"]
        out["W_mean_abs"] = a["w_abs"] / wn
        out["W_mean"] = a["w_sum"] / wn
        out["W_pos_frac"] = a["w_pos"] / wn
        # v1.57 — the sample count, so the per-brain and population
        # computations can be reconciled. In the V1.076 run per-brain
        # W_mean_abs read 0.0107 while the population read 0.3365 (31x),
        # and the gap widened over the run, while every M-metric agreed
        # within 1.7x. Both paths use identical arithmetic on the same
        # accumulator, so the divergence has to come from WHICH synapses
        # each one walks — and without w_n that is untestable. Logging it
        # makes the discrepancy diagnosable from the data alone.
        out["W_n"] = wn
        out["W_n_syn_total"] = a.get("w_total", wn)
    return {k: round(v, 5) for k, v in out.items()}


def _brain_science(b, acc, per_brain=None, bid=None):
    """v1.54 — wrapper around the single-brain pass. Accumulates into a
    LOCAL dict first so this brain's own M values can be read off, then
    folds that local dict into the shard total exactly as before. The
    neuron walk is unchanged, so the cost is a few divisions per brain at
    the existing ~1/min science cadence."""
    loc = _new_sci_acc()
    _brain_science_into(b, loc)
    for k, v in loc.items():
        acc[k] = acc.get(k, 0) + v
    if per_brain is not None and bid is not None and loc.get("n_brains", 0):
        m = _m_from_acc(loc)
        if m:
            per_brain[bid] = m


def _new_sci_acc():
    return {k: 0 for k in (
        "n_brains", "e", "z", "h", "nneu",
        "br_sum", "br_sq", "br_n", "br_sub", "br_sup",
        "ts_sum", "ts_sq", "ts_n",
        "gate_sum", "gate_sq", "gate_n",
        "sensory_act", "sensory_n", "assoc_act", "assoc_n",
        "motor_act", "motor_n",
        "dead_sum", "dead_n",
        "mamp_ok_sum", "mamp_ok_n", "mamp_les_sum", "mamp_les_n",
        # v1.53 — sampled synaptic weights (plasticity drift diagnostic)
        "w_sum", "w_abs", "w_sq", "w_n", "w_pos", "w_total",
    )}


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
    # CRITICAL: load_multisphere_from_dict() needs the network rebuilder
    # passed explicitly. Without it, its fallback calls a non-existent
    # NeuraxonNetwork._from_dict, the AttributeError is swallowed, and
    # EVERY brain loads with ZERO spheres (inert: 0 neurons, motor
    # output all-zero, branching pinned at the 1.0 default). In parallel
    # mode every brain comes through this path, so the whole world ran
    # on the behavioural floors with dead brains until this was wired.
    try:
        from neuraxon.network import _rebuild_net_from_dict as _rebuild_net
    except Exception:
        _rebuild_net = None

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
                brains[i] = load_multisphere_from_dict(
                    bdict, rebuild_net_fn=_rebuild_net)
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
            smods = msg[2] if len(msg) > 2 else None   # v1.52 id->dopamine
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
                    if smods is not None and i in smods:
                        # v1.52 — raise phasic dopamine so the AGMP rule
                        # consolidates this NxEr's eligibility traces.
                        try:
                            b.set_global_modulator("dopamine", smods[i])
                        except Exception:
                            pass
                    b.simulate_step(ext)
                    outs = b.motor_sphere.network.get_output_states()
                    br = getattr(b.motor_sphere.network,
                                 "branching_ratio", 1.0)
                    out.append((i, list(outs), float(br)))
                except Exception:
                    out.append((i, [0] * 7, 1.0))
            conn.send(out)
        elif op == "firestat":
            # low-cadence (≈1/min) population trinary sample. Aggregates
            # this shard's brains; the engine sums across shards. Cheap
            # at history cadence — never called on the hot step path.
            e = z = h = nn = 0
            for b in brains.values():
                be, bz, bh, bn = _brain_firing_counts(b)
                e += be
                z += bz
                h += bh
                nn += bn
            conn.send(("fire", e, z, h, nn))
        elif op == "sci":
            # v1.44 — enriched science sample (M1/M2/M5/M6/M8/M10), same
            # ~1/min cadence as firestat, one pass over this shard's
            # brains. Returns summable float accumulators; the engine
            # merges across shards and turns them into the M-metrics.
            acc = _new_sci_acc()
            per_brain = {}
            for i, b in brains.items():
                try:
                    _brain_science(b, acc, per_brain, i)
                except Exception:
                    pass
            conn.send(("sci", acc, per_brain))
        elif op == "btopo":
            # v1.48 — single-brain topology for the NxonKaleido viewer.
            b = brains.get(msg[1])
            try:
                conn.send(("btopo",
                           _brain_topology(b) if b is not None else None))
            except Exception:
                conn.send(("btopo", None))
        else:
            conn.send(("err", "unknown op"))
    conn.close()


# --- pool ---------------------------------------------------------
class BrainPool:
    """Sharded persistent worker processes. id -> shard = id % N."""

    def __init__(self, num_workers, num_builders=None, step_timeout=5.0,
                 reserved_builder_cores=2, pin_main_process=False):
        self.n = max(0, int(num_workers))
        self._pin_main = bool(pin_main_process)
        self._parents = []
        self._procs = []
        # v1.37 — CPU isolation. Reserve a small core set for the
        # builders (and the main/web process) so heavy brain
        # construction from user NxEr creation runs on dedicated cores
        # and never steals a step-worker's core (which previously
        # stalled the tick past worker_step_timeout and froze the
        # server). Step-workers get the remaining cores 1:1.
        try:
            avail = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            avail = list(range(os.cpu_count() or 1))
        ncpu = len(avail)
        R = max(0, int(reserved_builder_cores))
        # never reserve so many that no worker core remains
        R = min(R, max(0, ncpu - 1))
        if R > 0 and ncpu > R:
            self._builder_cores = avail[ncpu - R:]     # last R cores
            self._worker_cores = avail[:ncpu - R]      # the rest
        else:
            self._builder_cores = []
            self._worker_cores = []
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
                _set_affinity(p.pid, self._worker_cores)   # reserved-core isolation
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
                _set_affinity(proc.pid, self._builder_cores)  # reserved cores
                self._builder_parents.append(bp_parent)
                self._builder_procs.append(proc)
                pump = threading.Thread(
                    target=self._pump_built, args=(bp_parent,),
                    daemon=True)
                pump.start()
                self._pumps.append(pump)
            # v1.38 — DO NOT pin the main process by default. v1.37
            # pinned it to the 2 reserved cores, which confined the
            # engine loop + aiohttp web server + the builder pump
            # threads all onto 2 cores. Under PyPy's GIL that means the
            # engine thread's per-tick worker `recv` wakeups had to
            # contend for those 2 cores with the web server, collapsing
            # TPS even at tiny populations. Letting the main process
            # FLOAT across all cores (workers/builders stay pinned to
            # their disjoint sets) restores throughput. Opt back in with
            # pin_main_process=true only if you have a specific reason.
            if self._pin_main and self._builder_cores:
                _set_affinity(0, self._builder_cores)
            print("[BrainPool] mode=PARALLEL  step_workers=%d  "
                  "builders=%d  cpu_count=%s" % (
                      self.n, self._n_builders, os.cpu_count()))
            print("[BrainPool]   worker cores=%s  reserved(builder+main) "
                  "cores=%s" % (self._worker_cores, self._builder_cores))
            print("[BrainPool]   step-worker pids: %s" % (
                  [p.pid for p in self._procs],))
        else:
            # in-process fallback (1-core / safety): keep brains here
            print("=" * 64)
            print("[BrainPool] *** IN-PROCESS MODE *** — brains run in the "
                  "MAIN thread on ONE core.")
            print("[BrainPool] num_workers=%d  cpu_count=%s" % (
                  self.n, os.cpu_count()))
            print("[BrainPool] For multi-core, set \"engine_workers\" in "
                  "world_config.json to (cores-1) and restart.")
            print("=" * 64)
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
            # see the worker note above — the loader needs the network
            # rebuilder or brains load empty (inert). Bind it here so the
            # in-process path (and any dict-loaded brain) is correct too.
            try:
                from neuraxon.network import _rebuild_net_from_dict as _rn
            except Exception:
                _rn = None
            self._bb = build_brain
            self._lb = (lambda bd: load_multisphere_from_dict(
                bd, rebuild_net_fn=_rn))
            self._brains = {}
            self._inputs = {}

    def mode_info(self):
        """Diagnostic snapshot for the admin console so the operator can
        SEE whether brains are parallel or stuck in-process."""
        alive_workers = 0
        pids = []
        if self.parallel:
            for p in self._procs:
                pids.append(p.pid)
                try:
                    if p.is_alive():
                        alive_workers += 1
                except Exception:
                    pass
        return {
            "mode": "parallel" if self.parallel else "in_process",
            "step_workers": self.n if self.parallel else 0,
            "step_workers_alive": alive_workers,
            "builders": self._n_builders if self.parallel else 0,
            "cpu_count": os.cpu_count(),
            "worker_cores": self._worker_cores if self.parallel else [],
            "reserved_cores": self._builder_cores if self.parallel else [],
            "worker_pids": pids,
            "brains_in_process": (0 if self.parallel
                                  else len(getattr(self, "_brains", {}))),
        }

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
    def step(self, batch, mods=None):
        """batch = [(id, sens_list), ...] for all alive NxErs.
        mods = optional {id: dopamine_level} for rewarded NxErs (v1.52).
        Returns {id: (motor_list, branching)}."""
        if not batch:
            return {}
        if self.parallel:
            shards = [[] for _ in range(self.n)]
            smods = [None] * self.n
            for i, sens in batch:
                k = i % self.n
                shards[k].append((i, sens))
                if mods is not None and i in mods:
                    if smods[k] is None:
                        smods[k] = {}
                    smods[k][i] = mods[i]
            # fan out to all workers, THEN collect → true concurrency
            sent = [False] * self.n
            for k, sh in enumerate(shards):
                if sh:
                    try:
                        self._parents[k].send(("step", sh, smods[k]))
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
                if mods is not None and i in mods:   # v1.52
                    try:
                        b.set_global_modulator("dopamine", mods[i])
                    except Exception:
                        pass
                b.simulate_step(ext)
                outs = b.motor_sphere.network.get_output_states()
                br = getattr(b.motor_sphere.network,
                             "branching_ratio", 1.0)
                res[i] = (list(outs), float(br))
            except Exception:
                res[i] = ([0] * 7, 1.0)
        return res

    def sample_firing(self):
        """Population trinary distribution: (M1_exc, M1_neutral, M1_inh)
        fractions averaged over all neurons of all living brains, or None
        if no brains. Low cadence only (history sampling) — NOT the hot
        path. Parallel: one round-trip per shard, aggregated."""
        e = z = h = nn = 0
        if self.parallel:
            sent = [False] * self.n
            for k in range(self.n):
                try:
                    self._parents[k].send(("firestat",))
                    sent[k] = True
                except (BrokenPipeError, OSError, EOFError):
                    sent[k] = False
            for k in range(self.n):
                if not sent[k]:
                    continue
                try:
                    if self._parents[k].poll(self._step_timeout):
                        tag, be, bz, bh, bn = self._parents[k].recv()
                        if tag == "fire":
                            e += be
                            z += bz
                            h += bh
                            nn += bn
                except (BrokenPipeError, OSError, EOFError, ValueError):
                    pass
        else:
            for b in self._brains.values():
                be, bz, bh, bn = _brain_firing_counts(b)
                e += be
                z += bz
                h += bh
                nn += bn
        if nn <= 0:
            return None
        return (e / nn, z / nn, h / nn, nn)

    def sample_science(self):
        """v1.44 — enriched M-metric sample. Fans the 'sci' op to every
        shard (or runs in-process), merges the summable accumulators, and
        returns the merged dict (or None if no brains). Low cadence only
        (history sampling), one round-trip per shard — never the hot
        path.

        v1.54 — also returns each brain's OWN M values, so individual NxErs
        can be scored against the M bands instead of only the population
        average. Returns (acc, per_brain) where per_brain is {nxer_id: {...}}.
        """
        acc = _new_sci_acc()
        per_brain = {}

        def _merge(part, pb=None):
            if part:
                for k, v in part.items():
                    acc[k] = acc.get(k, 0) + v
            if pb:
                per_brain.update(pb)

        if self.parallel:
            sent = [False] * self.n
            for k in range(self.n):
                try:
                    self._parents[k].send(("sci",))
                    sent[k] = True
                except (BrokenPipeError, OSError, EOFError):
                    sent[k] = False
            for k in range(self.n):
                if not sent[k]:
                    continue
                try:
                    if self._parents[k].poll(self._step_timeout):
                        msg = self._parents[k].recv()
                        if msg and msg[0] == "sci":
                            _merge(msg[1],
                                   msg[2] if len(msg) > 2 else None)
                except (BrokenPipeError, OSError, EOFError, ValueError):
                    pass
        else:
            for i, b in self._brains.items():
                try:
                    _brain_science(b, acc, per_brain, i)
                except Exception:
                    pass
        if acc.get("n_brains", 0) > 0:
            return acc, per_brain
        return None, {}

    def brain_topology(self, nxer_id):
        """v1.48 — fetch ONE NxEr's brain topology for the NxonKaleido
        viewer. Only the shard holding that brain returns data. MUST be
        called from the engine/game-loop thread (it shares the worker
        pipes with the step loop), exactly like sample_science."""
        if self.parallel:
            for k in range(self.n):
                try:
                    self._parents[k].send(("btopo", nxer_id))
                except (BrokenPipeError, OSError, EOFError):
                    pass
            found = None
            for k in range(self.n):
                try:
                    if self._parents[k].poll(self._step_timeout):
                        tag, topo = self._parents[k].recv()
                        if tag == "btopo" and topo is not None:
                            found = topo
                except (BrokenPipeError, OSError, EOFError, ValueError):
                    pass
            return found
        b = self._brains.get(nxer_id)
        return _brain_topology(b) if b is not None else None

    def close(self):
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
