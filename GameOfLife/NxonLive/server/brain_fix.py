# Multi Neuraxon Game of Life 5 — v1.67 "living, connected brain"
# ===================================================================
# Everything here is applied FROM OUR CODE to brains built by the vendored
# substrate (neuraxon/*.py is never edited). It answers the six questions
# the V 1.086 run left open:
#
#   Q3 neurons freeze      -> rescue_energy() each brain step (P1) and a
#                             firing energy cost that matches the M1 target
#                             activity (P2, apply_living)
#   Q4 synapses lost early -> consolidating slow weights: tau_slow raised for
#                             every synapse and for the network's params, so
#                             synapses formed later inherit it (P3, apply_living)
#   Q5 dopamine -> AGMP    -> reward_phasic(): the eating reward is written
#                             into the PHASIC dopamine pool, the one AGMP reads
#   Q1/Q2/Q6 senses -> movement:
#                           -> relay input layer: the 10 world-facing input
#                             neurons follow their input (P8, apply_living)
#                           -> add_sensorimotor(): an innate projection from the
#                             fdx/fdy INPUT neurons of the sensory sphere to the
#                             mvx/mvy OUTPUT neurons of the motor sphere, built
#                             with the substrate's own connect_spheres() and
#                             weighted by the NxEr's inherited reflex genes.
#                             Food direction then reaches the motor output
#                             through the brain's own neurons.
#
# Why each piece (V 1.086 final report, §7):
#   * a neuron active on more than ~9-17% of steps ran out of energy, and at
#     energy 0 Neuraxon.update returns before its own recovery code, so it
#     never updated again (0 of 368 frozen neurons thawed in 480k ticks);
#   * w_fast/w_slow are leaky traces (tau_slow ~50 steps): ~97% of synapses
#     were gone by ~3k ticks in every plastic arm;
#   * set_global_modulator('dopamine', x) lands in TONIC dopamine, AGMP reads
#     levels['DA']['phasic'];
#   * the input neurons fired on 15-25% of steps (refractory period +
#     after-hyperpolarisation), and even with a relay input layer the
#     direction died inside the sensory sphere: its output neurons decoded at
#     chance in every living brain tested. No plasticity rule tested built the
#     map within a life. A direct, genetically specified pathway does.
# ===================================================================
import math
import threading

DEFAULTS = {
    "exact_reload": True,             # P0: brains keep their shape through to_dict -> load
    "energy_rescue": True,            # P1
    "firing_energy_cost": 1.5,        # P2 (substrate default ~5-6)
    "tau_slow": 1.0e5,                # P3 (substrate default ~50)
    "relay_inputs": True,             # P8: world-facing input neurons follow their input
    "tonic_motor": True,              # mvx/mvy output neurons fire while driven (no refractory/AHP)
    "clean_topdown": True,            # assoc_fluid -> sensory lands on channels 6-9 only
    "sm_link_gain_scale": 1.0,        # innate sensorimotor projection gain x rfx_gain
    "sm_link_plasticity": 0.0,        # 0 = innate (weights are genes)
    "sm_same_step": True,             # the projection is delivered in the same step (step_brain)
    "reward_phasic": True,            # Q5
    "reward_phasic_cap": 1.0,
}

SM_LINK_ID = "sensorimotor__v167"
# sensory input channels (engine._sense order) and motor output channels
CH_FDX, CH_FDY = 1, 2
MO_X, MO_Y = 0, 1


def cfg_get(cfg, k):
    if cfg and k in cfg:
        return cfg[k]
    return DEFAULTS[k]


# ---------------------------------------------------------------------------
# P0 — exact reload.
# The substrate's network loader (_rebuild_net_from_dict) rebuilds every
# sphere with at least 10 input and 7 output neurons ("always upgrade to
# v4.5 shape"). The chc6 builder makes smaller spheres (visual/auditory/
# intero 6/12/5, assoc 8/24/6 and 8/38/6, motor 6/12/7, sensory 10/12/6).
# Neuron states and synapses are then restored BY ID into the inflated
# network, so ids change role: in the motor sphere hidden ids 6-9 become
# input neurons, the mvx/mvy/social/mate outputs (ids 18-21) become hidden
# neurons, and the engine's mvx/mvy/social read ids 22-24, which were built
# as give/rest/sing. Link sources built on output ids now read hidden
# neurons. Every brain passes through this loader once (builder -> worker),
# and again at every snapshot restore. The fix: construct each network with
# the counts it was saved with; everything else is the substrate's own loader.
# ---------------------------------------------------------------------------
_REBUILD_LOCK = threading.Lock()
# The lock serialises rebuilds only. While a rebuild runs, the module-level
# NeuraxonNetwork name is the shape-fixing subclass, so a brain BUILT on
# another thread of the same process at that instant would get this
# network's counts. Nothing does that today (the engine thread does every
# in-process build and load; workers are single-threaded; builders never
# rebuild) — keep it that way.


def rebuild_net_exact(d):
    """Drop-in rebuild_net_fn for load_multisphere_from_dict that keeps the
    saved input/hidden/output counts (and therefore every neuron id's role)."""
    from neuraxon import network as _nw
    ni = len(d["neurons"]["input"])
    nh = len(d["neurons"]["hidden"])
    no = len(d["neurons"]["output"])
    orig = _nw.NeuraxonNetwork

    class _ExactShape(orig):
        def __init__(self, params, *a, **k):
            params.num_input_neurons = ni
            params.num_hidden_neurons = nh
            params.num_output_neurons = no
            super().__init__(params, *a, **k)

    with _REBUILD_LOCK:
        _nw.NeuraxonNetwork = _ExactShape
        try:
            net = _nw._rebuild_net_from_dict(d)
        finally:
            _nw.NeuraxonNetwork = orig
    net.__class__ = orig      # no state was added; keep the substrate's class
    return net


def rebuild_fn(cfg=None):
    """The network loader to pass as rebuild_net_fn (exact unless disabled)."""
    if cfg_get(cfg, "exact_reload"):
        return rebuild_net_exact
    from neuraxon.network import _rebuild_net_from_dict
    return _rebuild_net_from_dict


def load_brain(bdict, cfg=None):
    """load_multisphere_from_dict with the exact loader, then apply_living."""
    from neuraxon.multisphere import load_multisphere_from_dict
    b = load_multisphere_from_dict(bdict, rebuild_net_fn=rebuild_fn(cfg))
    apply_living(b, cfg)
    return b


def wiring_check(b):
    """How many link endpoints sit on the neuron role they were built for:
    link sources should be output neurons of the source sphere (except the
    innate sensorimotor projection, which reads input neurons) and targets
    input neurons (except the sensorimotor projection, which drives motor
    outputs). Returns {'src_ok','src_n','tgt_ok','tgt_n', 'shapes'}."""
    r = {"src_ok": 0, "src_n": 0, "tgt_ok": 0, "tgt_n": 0, "shapes": {}}
    for sid, sp in b.spheres.items():
        net = sp.network
        r["shapes"][sid] = (len(net.input_neurons), len(net.hidden_neurons), len(net.output_neurons))
    for lid, L in b.links.items():
        s = b.spheres[L.source_sphere_id].network
        t = b.spheres[L.target_sphere_id].network
        sm = lid == SM_LINK_ID
        s_ok = {n.id for n in (s.input_neurons if sm else s.output_neurons)}
        t_ok = {n.id for n in (t.output_neurons if sm else t.input_neurons)}
        r["src_n"] += len(L.source_output_ids)
        r["src_ok"] += sum(1 for i in L.source_output_ids if i in s_ok)
        r["tgt_n"] += len(L.target_input_ids)
        r["tgt_ok"] += sum(1 for i in L.target_input_ids if i in t_ok)
    return r


def _networks(b):
    for sp in getattr(b, "spheres", {}).values():
        net = getattr(sp, "network", None)
        if net is not None:
            yield net


def apply_living(b, cfg=None):
    """Idempotent. Call after every build AND every load (the per-neuron
    refractory attributes are not serialised by the substrate, the rest is).
    Returns a small dict describing what was applied."""
    out = {"cost": 0, "tau": 0, "relay": 0}
    target_cost = float(cfg_get(cfg, "firing_energy_cost") or 0)
    tau = float(cfg_get(cfg, "tau_slow") or 0)
    for net in _networks(b):
        p = net.params
        # P2 — scale each neuron's individual cost so its variation is kept
        if target_cost > 0:
            cur = float(getattr(p, "firing_energy_cost", target_cost) or target_cost)
            if cur > 0 and abs(cur - target_cost) > 1e-9:
                f = target_cost / cur
                for n in net.all_neurons:
                    try:
                        n.firing_energy_cost = n.firing_energy_cost * f
                    except Exception:
                        pass
                p.firing_energy_cost = target_cost
                out["cost"] += 1
        # P3 — slow weights consolidate; params so later synapses inherit it
        if tau > 0:
            try:
                p.tau_slow = tau
            except Exception:
                pass
            for s in net.synapses:
                if getattr(s, "tau_slow", tau) != tau:
                    s.tau_slow = tau
                    out["tau"] += 1
    # tonic movement effectors: mvx/mvy fire for as long as they are driven.
    # With the substrate's refractory period (1-5 steps) + after-
    # hyperpolarisation a driven output neuron fires once and then sits at 0
    # for several decisions (measured: mvx fired on 4 of 23 steps under a
    # constant +1.2 drive), so the movement command is rhythmic noise, not a
    # readout of the drive. Same treatment as the input relays, for every
    # brain arm alike (plain and wired share one output layer).
    if cfg_get(cfg, "tonic_motor"):
        try:
            for n in b.motor_sphere.network.output_neurons[:2]:
                _relay(n)
                out["tonic"] = out.get("tonic", 0) + 1
        except Exception:
            pass
    # the top-down assoc_fluid -> sensory link adds the brain's own state to
    # the world's value on the same 10 input neurons before rounding; on fdx
    # and fdy that creates phantom food directions (20-25% of zero inputs in
    # living brains, 74-75% in frozen ones). Keep its context on channels 6-9.
    if cfg_get(cfg, "clean_topdown"):
        try:
            out["topdown"] = clean_topdown(b)
        except Exception:
            out["topdown"] = -1
    # P8 — relay input layer (world-facing inputs of the sensory sphere only)
    if cfg_get(cfg, "relay_inputs"):
        try:
            for n in b.sensory_sphere.network.input_neurons:
                _relay(n)
                # the stimulus dominates the receptor: the external drive
                # enters as s*tanh(s*x); s = 2 gives 1.93 at x = 1 instead of
                # 0.76, above what the recurrent synapses onto the input
                # layer deliver (47 synapses, |w| ~0.39 at birth; a frozen
                # brain keeps them, and its fdx/fdy inputs followed the world
                # on 41% of steps instead of 93-100%)
                if getattr(n, "_sensory_boost_scale", 1.0) < 2.0:
                    n._sensory_boost_scale = 2.0
                # transducers: a relay input fires on almost every step while
                # its input lasts; at any firing cost it drains to the energy
                # floor, where the substrate's energy-threshold coupling raises
                # its threshold and it stops following the world (measured:
                # follow rate 0.61 -> 0.00 within 40 steps). Receptor cells are
                # not limited by their own firing budget, so they pay none.
                out["relay"] += 1
        except Exception:
            pass
    return out


def _relay(n):
    """A neuron that follows its drive: no refractory period, no after-
    hyperpolarisation, no adaptation, symmetric thresholds, and no firing energy cost (it
    would otherwise sit at the energy floor, where the substrate's
    energy-threshold coupling silences it)."""
    n._refractory_period = 0
    n.refractory_counter = 0
    n._post_spike_mp_reset = 0.0
    # spike-frequency adaptation (target 0.83 under sustained firing, tau 20
    # steps) is subtracted from the drive: a relay held at +1 fired on only
    # ~half the steps and the food direction flickered. Tonic receptors and
    # effectors do not adapt here.
    n._adaptation_tau_ticks = 1.0e12
    n.adaptation = 0.0
    # no memory of past input: the readout is the membrane potential (no CTSN
    # complement, which carried the previous direction for tens of steps and
    # flipped fdx against the world after a turn), integrated with tau 1 and
    # without the DSN input buffer
    n._ctsn_enabled = False
    n.complement_h = 0.0
    # the substrate decides the state from params.ctsn_enabled's readout,
    # state_tilde, which is written at the END of the previous update: every
    # neuron's state lags its own input by one step (a relay held the old
    # direction on 63% of steps when the direction changed every step). A
    # relay reads its current potential instead: its own copy of the
    # network parameters with the CTSN readout off (not serialised; the
    # network's shared parameters are untouched)
    if getattr(n.params, "ctsn_enabled", False):
        import copy
        p = copy.copy(n.params)
        p.ctsn_enabled = False
        n.params = p
    n._dsn_enabled = False
    n._max_intrinsic_timescale = 1.0
    n.intrinsic_timescale = 1.0
    # fixed thresholds: the network's criticality homeostasis moves every
    # active neuron's thresholds (a relay's went 0.31 -> 0.46 in 200 steps);
    # hold_relays() puts these back each step
    th = abs(n.firing_threshold_excitatory)
    if not hasattr(n, "_relay_th"):
        n._relay_th = th
    n.firing_threshold_excitatory = n._relay_th
    n.firing_threshold_inhibitory = -n._relay_th
    n.firing_energy_cost = 0.0
    if n.energy_level < n.energy_baseline:
        n.energy_level = n.energy_baseline


def relay_list(b, cfg=None):
    """The relay neurons of a brain (world-facing sensory inputs, mvx/mvy)."""
    out = []
    if cfg_get(cfg, "relay_inputs"):
        try:
            out.extend(b.sensory_sphere.network.input_neurons)
        except Exception:
            pass
    if cfg_get(cfg, "tonic_motor"):
        try:
            out.extend(b.motor_sphere.network.output_neurons[:2])
        except Exception:
            pass
    return out


def hold_relays(relays):
    """Each step: undo the network's threshold homeostasis on the relays
    (both thresholds; the inhibitory-deficit rule moves only the negative
    one) and keep them supplied (metabolism still drains a relay: 107 -> 71
    in 400 steps, and below the energy floor the substrate raises its
    threshold)."""
    for n in relays:
        th = getattr(n, "_relay_th", None)
        if th is not None:
            if n.firing_threshold_excitatory != th:
                n.firing_threshold_excitatory = th
            if n.firing_threshold_inhibitory != -th:
                n.firing_threshold_inhibitory = -th
        if n.energy_level < n.energy_baseline:
            n.energy_level = n.energy_baseline


def clean_topdown(b):
    """Re-target the assoc_fluid -> sensory link onto input channels 6-9
    (day/night, heading, neighbour, song). Idempotent."""
    from collections import deque
    k = 0
    for lid, L in list(b.links.items()):
        if not lid.startswith("assoc_fluid__to__sensory"):
            continue
        if len(L.target_input_ids) <= 4:
            continue
        L.target_input_ids = L.target_input_ids[6:]
        L.weight_matrix = L._normalise_rows([r[:] for r in L.weight_matrix[6:]])
        L.delay_buffer = deque([[0.0] * len(L.target_input_ids)
                                for _ in range(max(1, L.params.delay_steps))],
                               maxlen=max(1, L.params.delay_steps + 1))
        k += 1
    return k


def neuron_list(b):
    """Flat list of every neuron (cache it per brain; the lists never change
    size in this substrate — neurons are deactivated, not removed)."""
    out = []
    for net in _networks(b):
        out.extend(net.all_neurons)
    return out


def rescue_energy(neurons):
    """P1 — the recovery the substrate skips: Neuraxon.update returns at its
    first line when energy_level <= 0, before _update_energy runs, so an
    active neuron at 0 energy is frozen for good. Give it one step's boosted
    recovery so it can update again. Returns the number rescued."""
    k = 0
    for n in neurons:
        if n.energy_level <= 0.0 and n.is_active:
            try:
                n.energy_level = n.recovery_rate * n.params.energy_recovery_boost
            except Exception:
                n.energy_level = 1.0
            k += 1
    return k


def sm_weights(genes):
    """Rows = targets (mvx, mvy), columns = sources (fdx, fdy)."""
    g = genes or {}
    return [[float(g.get("rfx_xx", 0.0)), float(g.get("rfx_xy", 0.0))],
            [float(g.get("rfx_yx", 0.0)), float(g.get("rfx_yy", 0.0))]]


def add_sensorimotor(b, genes, cfg=None):
    """Innate sensorimotor projection: sensory-sphere fdx/fdy INPUT neurons ->
    motor-sphere mvx/mvy OUTPUT neurons, weights = reflex genes, gain =
    rfx_gain x sm_link_gain_scale. The payload (gain x sum w*state) is added
    to the motor neuron's input and rounded to a trinary value at +-0.25 by
    the substrate, exactly like every other link. Idempotent (re-applies the
    genes if the link already exists)."""
    from neuraxon.multisphere import SphereLinkParameters
    g = genes or {}
    gain = float(g.get("rfx_gain", 0.0)) * float(cfg_get(cfg, "sm_link_gain_scale"))
    W = sm_weights(g)
    try:
        src = [n.id for n in b.sensory_sphere.network.input_neurons]
        tgt = [n.id for n in b.motor_sphere.network.output_neurons]
        src_ids = [src[CH_FDX], src[CH_FDY]]
        tgt_ids = [tgt[MO_X], tgt[MO_Y]]
    except Exception:
        return False
    L = b.links.get(SM_LINK_ID)
    if L is None:
        prm = SphereLinkParameters(
            kind="sensorimotor", topology="dense", delay_steps=0, gain=gain,
            coherence_strength=0.0, plasticity_rate=float(cfg_get(cfg, "sm_link_plasticity")),
            allow_negative_weights=True, normalize_rows=False, weight_clip=1.5,
            integrity_decay_rate=0.0)
        b.connect_spheres("sensory", "motor", source_output_ids=src_ids,
                          target_input_ids=tgt_ids, params=prm,
                          weight_matrix=W, link_id=SM_LINK_ID)
    else:
        L.params.gain = gain
    return True


def remove_sensorimotor(b):
    return b.links.pop(SM_LINK_ID, None) is not None


def step_brain(b, ext=None, same_step=True):
    """One brain step. Brains without the sensorimotor projection: exactly
    b.simulate_step(ext). Brains with it: the same pipeline (it mirrors
    NeuraxonMultiSphere.simulate_step of the vendored substrate), except that
    the projection is delivered in the SAME step — computed after the sensory
    sphere has updated and added to the motor sphere's input — instead of
    from the previous step's states like every other link.
    Why: the substrate aggregates all link inputs before any sphere steps,
    so a link carries the previous step's states. A receptor -> effector arc
    then answers the food direction of one brain decision earlier (8 ticks,
    up to 2 cells of movement): offline the command matched the previous
    direction (+0.90) and the current one at ~0 near food. The innate arc is
    the short-latency pathway (a spinal reflex is monosynaptic); every other
    link keeps the substrate's one-step conduction delay."""
    L = b.links.get(SM_LINK_ID) if same_step else None
    if L is None:
        return b.simulate_step(ext)
    sp_ids = list(b.spheres.keys())
    src_id, tgt_id = L.source_sphere_id, L.target_sphere_id
    if (src_id not in b.spheres or tgt_id not in b.spheres
            or sp_ids.index(src_id) >= sp_ids.index(tgt_id)):
        return b.simulate_step(ext)
    ext = ext or {}
    # 1. inter-sphere propagation, without the projection
    del b.links[SM_LINK_ID]
    try:
        routed = b._aggregate_link_inputs()
    finally:
        b.links[SM_LINK_ID] = L
    # 2. global neuromodulation
    b._diffuse_global_neuromodulators()
    # 3. intra-sphere simulation; the projection joins the target's input
    for sid, sphere in b.spheres.items():
        li = routed.get(sid, {})
        if sid == tgt_id and L.integrity > 0.0:
            pl = L.project(b.spheres[src_id], sphere)
            li = dict(li)
            for nid, v in pl.items():
                li[nid] = li.get(nid, 0.0) + float(v)
        prepared = b._prepare_external_inputs(sphere, ext.get(sid, {}), li)
        sphere.network.simulate_step(prepared)
    # 4. projection plasticity, 5. structural plasticity — as the substrate
    global_da = b.global_neuromodulators.get("dopamine", 0.15)
    for link in b.links.values():
        if link.integrity <= 0.0:
            continue
        s_ = b.spheres.get(link.source_sphere_id)
        t_ = b.spheres.get(link.target_sphere_id)
        if s_ and t_:
            link.update_plasticity(s_, t_, global_da)
    prune = []
    for lid, link in b.links.items():
        link.update_structural_plasticity()
        if link.integrity <= 0.0:
            prune.append(lid)
    for lid in prune:
        del b.links[lid]
    b.step_count += 1
    b.time = max((s.network.time for s in b.spheres.values()), default=0.0)
    if b.spheres:
        n_ = len(b.spheres)
        for mod in b.global_neuromodulators:
            b.global_neuromodulators[mod] = sum(
                s.network.neuromodulators.get(mod, 0.12)
                for s in b.spheres.values()) / n_


def reward_phasic(b, amount, cap=None):
    """Q5 — write an eating/mating reward into the PHASIC dopamine pool of
    every sphere, keeping tonic where it is. The substrate's
    _sync_neuromod_to_system() sets tonic = level - phasic before each step,
    so the level is raised by the same amount; AGMP then reads the raised
    phasic (delta_w = eta * phasic * astrocyte * eligibility). The global
    pool (link plasticity gate, volume transmission) is raised to the new
    sphere mean so diffusion does not pull the level back."""
    amt = float(amount)
    if amt <= 0.0:
        return 0.0
    cap = float(cap if cap is not None else DEFAULTS["reward_phasic_cap"])
    tot = 0.0
    n = 0
    for net in _networks(b):
        try:
            lv = net.neuromod_system.levels["DA"]
            ph = min(cap, lv["phasic"] + amt)
            lv["phasic"] = ph
            level = min(2.0, lv["tonic"] + ph)
            net.neuromodulators["dopamine"] = level
            tot += level
            n += 1
        except Exception:
            pass
    if n:
        try:
            b.global_neuromodulators["dopamine"] = tot / n
        except Exception:
            pass
    return amt


def agmp_rate(b):
    """Q5 readout: the mean size of the AGMP weight update the next step would
    make, |eta * DA_phasic * astrocyte(post) * eligibility|, averaged over the
    live synapses of every sphere (the substrate's own terms). 0 when phasic
    dopamine is 0, however much tonic dopamine there is."""
    tot = 0.0
    k = 0
    for net in _networks(b):
        try:
            ph = float(net.neuromod_system.levels["DA"]["phasic"])
            eta = float(getattr(net.params, "agmp_eta", 0.005))
            nm = net._neuron_map
        except Exception:
            continue
        for s in net.synapses:
            if s.integrity <= 0:
                continue
            post = nm.get(s.post_id)
            if post is None:
                continue
            tot += abs(eta * ph * getattr(post, "astrocyte_state", 0.0)
                       * getattr(s, "eligibility", 0.0))
            k += 1
    return (tot / k) if k else 0.0


def measure(b):
    """P10 — per-brain health and wiring readout for brain_samples.
    Cheap: one pass over neurons, no synapse scan."""
    m = {}
    tot = fz = 0
    for net in _networks(b):
        for n in net.all_neurons:
            tot += 1
            if n.is_active and n.energy_level <= 0.0:
                fz += 1
    m["frozen_frac"] = round(fz / tot, 4) if tot else None
    try:
        ins = b.sensory_sphere.network.input_neurons
        m["frozen_in"] = sum(1 for n in ins if n.is_active and n.energy_level <= 0.0)
        outs = b.motor_sphere.network.output_neurons
        m["frozen_mo"] = sum(1 for n in outs if n.is_active and n.energy_level <= 0.0)
    except Exception:
        pass
    t = p = 0.0
    k = 0
    for net in _networks(b):
        try:
            lv = net.neuromod_system.levels["DA"]
            t += lv["tonic"]
            p += lv["phasic"]
            k += 1
        except Exception:
            pass
    if k:
        m["DA_tonic"] = round(t / k, 4)
        m["DA_phasic"] = round(p / k, 4)
    try:
        m["kappa"] = round(float(b.sensory_sphere.network.params.cross_sphere_coupling), 4)
    except Exception:
        pass
    m["sm_link"] = 1 if SM_LINK_ID in getattr(b, "links", {}) else 0
    try:
        m["agmp"] = float("%.3e" % agmp_rate(b))
        w = wiring_check(b)
        m["link_src_ok"] = round(w["src_ok"] / max(1, w["src_n"]), 3)
    except Exception:
        pass
    # what the fdx/fdy input neurons received vs what the world wrote (one
    # step's snapshot; the population mean is the faithfulness rate)
    try:
        S = b.sensory_sphere
        ids = [n.id for n in S.network.input_neurons]
        ok = nz = 0
        for ch in (CH_FDX, CH_FDY):
            w = S.last_user_inputs.get(ids[ch], 0.0)
            if w:
                nz += 1
                c = S.last_combined_inputs.get(ids[ch], 0)
                ok += 1 if (c > 0) == (w > 0) and c != 0 else 0
        if nz:
            m["in_faith"] = round(ok / nz, 3)
        st = [S.network.input_neurons[ch].trinary_state for ch in (CH_FDX, CH_FDY)]
        ws = [S.last_user_inputs.get(ids[ch], 0.0) for ch in (CH_FDX, CH_FDY)]
        f = [1 if (wv and s == (1 if wv > 0 else -1)) else 0 for s, wv in zip(st, ws) if wv]
        if f:
            m["in_follow"] = round(sum(f) / len(f), 3)
    except Exception:
        pass
    return m
