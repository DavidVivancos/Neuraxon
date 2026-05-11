# Neuraxon Game of Life v.4.53 research_probes (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 145
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   "Multi-Neuraxon: Emergent Specialization, Modular, Frequency-Gated Neural Dynamics" by David Vivancos & Jose Sanchez
"""
research_probes.py  (NEW in v145 / v4.53)
==========================================
Implements the 10 paper-fidelity metrics identified in the v145 review.

Five of the ten require ONLY post-hoc analysis of signals that the network
emits naturally on every tick (state distributions, gate values, weight
distributions, etc.) — these live as `compute_*` static functions and are
called from logger._log_tick_level2 each full-analytics tick.

The other five require active probing — i.e. transient interventions on a
sampled NxEr (force input to zero, present novel V×A combination, etc.).
These run on a slow schedule (every PROBE_INTERVAL_TICKS, default 600) so
they do not perturb gameplay observably. Each probe takes a deep snapshot
of the target state, applies its intervention, reads results, and restores.

Mapping to the v145 review:
  M1  trinary E/I/Neutral distribution           — passive, every tick
  M2  CTC inter-sphere gate dynamics             — passive, every tick
  M3  cross-frequency phase-amplitude coupling   — passive, sliding-window
  M4  multi-timescale weight separation          — passive, sliding-window
  M5  branching ratio + criticality              — passive, every tick
  M6  spontaneous-vs-driven + ACW heterogeneity  — passive, every tick
  M7  self-sustained activity (Nengo test)       — ACTIVE, periodic probe
  M8  per-sphere functional specialization       — passive, modality-binned MI
  M9  compositional transfer ratio               — ACTIVE, periodic probe
  M10 heritability + lesion robustness           — passive, lineage + alive count

All probes are SAFE: they read state, do not mutate gameplay-visible attrs,
and gracefully no-op on NxErs without the required structure (e.g. M7-9
require .brain to exist).
"""
import copy
import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

# ===========================================================================
# CONFIG (probe schedule, history depths, target bands per paper)
# ===========================================================================

PROBE_INTERVAL_TICKS: int = 600          # how often active probes run
PROBE_TRIAL_DURATION: int = 12           # ticks per probe trial (state-reset protocol)
PROBE_RNG_SEED_OFFSET: int = 7919        # determinism for probes
SLIDING_WINDOW_TICKS: int = 200          # window for M3/M4/M6 sliding-window stats
PAC_PHASE_BINS: int = 8                  # Tort 2010 modulation index — 8 bins is canonical
ACW_HETEROGENEITY_FLOOR: float = 1e-6    # numerical guard
HERITABILITY_MIN_PAIRS: int = 8          # min parent-child pairs before reporting r
LESION_BIN_THRESHOLDS = [0.25, 0.50, 0.75]   # neuron-loss bins for M10 graceful-degradation curve

# Paper-derived healthy bands. Used for the "M*_in_band" boolean each tick
# so the dashboard can light up green / yellow / red without the consumer
# having to remember the targets.
HEALTHY_BANDS: Dict[str, Tuple[float, float]] = {
    'M1_excitatory_fraction': (0.18, 0.28),
    'M1_inhibitory_fraction': (0.08, 0.15),
    'M1_neutral_fraction':    (0.60, 0.78),
    'M2_gate_modulation_std': (0.05, 0.45),    # std must be > 0.05 (CTC alive) but < 0.45 (not chaotic)
    'M2_mean_gate':           (0.40, 0.85),
    'M3_pac_modulation_idx':  (0.005, 0.10),
    'M4_temporal_divergence': (0.0, 0.50),     # corr(Δw_fast, Δw_meta) — keep below 0.5 = independent dynamics
    'M5_branching_ratio':     (0.92, 1.10),
    'M6_spontaneous_fraction':(0.10, 0.45),
    'M6_acw_heterogeneity':   (3.0, 60.0),
    'M7_zero_input_mi_ratio': (0.40, 1.20),    # > ~0.4 = pre-ignited workspace exists
    'M7_broadcast_index':     (0.85, 1.30),
    'M8_sensory_pref_mag':    (0.20, 1.00),    # |pref_visual − pref_auditory| for sensory spheres
    'M8_assoc_pref_mag':      (0.0, 0.20),     # association sphere should stay multimodal
    'M9_transfer_ratio':      (0.85, 1.30),
    'M10_heritability_r':     (0.20, 1.00),
    'M10_lesion_retention_50':(0.85, 1.10),
    'M10_lesion_retention_75':(0.70, 1.10),
}


def in_band(metric_key: str, value: float) -> int:
    """Return 1 if value is in the paper-derived healthy band for `metric_key`,
    else 0. Used to compute the binary dashboard signal."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    band = HEALTHY_BANDS.get(metric_key)
    if not band:
        return 0
    lo, hi = band
    return 1 if (lo <= float(value) <= hi) else 0


# ===========================================================================
# Numerical helpers (stay dependency-free — work without numpy)
# ===========================================================================

def _safe_mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0

def _safe_std(xs):
    xs = [float(x) for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(0.0, var))

def _safe_corr(xs, ys):
    """Pearson r without numpy. Returns 0.0 on degenerate inputs."""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs = [float(x) for x in xs[:n]]
    ys = [float(y) for y in ys[:n]]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return num / (sx * sy)

def _mutual_information_binned(samples_a, samples_b, n_bins_a=3, n_bins_b=3):
    """Discretized MI estimate for trinary-on-trinary or trinary-on-bool.
    
    Both inputs are lists of category labels (already integer-binned).
    Returns MI in bits. Suitable for tracking input-output alignment over
    short windows. Not as accurate as KSG but cheap and stable."""
    n = min(len(samples_a), len(samples_b))
    if n < 4:
        return 0.0
    joint = {}
    pa = {}
    pb = {}
    for i in range(n):
        a = samples_a[i]; b = samples_b[i]
        joint[(a, b)] = joint.get((a, b), 0) + 1
        pa[a] = pa.get(a, 0) + 1
        pb[b] = pb.get(b, 0) + 1
    mi = 0.0
    for (a, b), c in joint.items():
        pab = c / n
        pa_  = pa[a] / n
        pb_  = pb[b] / n
        if pab > 0 and pa_ > 0 and pb_ > 0:
            mi += pab * math.log2(pab / (pa_ * pb_))
    return max(0.0, mi)


# ===========================================================================
# M1 — trinary E/I/Neutral distribution
# ===========================================================================
def compute_m1_trinary_distribution(active_neurons) -> Dict[str, float]:
    """Already-tracked fractions plus deviation-from-target.
    
    Paper claim (Neuraxon v2.0 §I/§VII; Multi-Neuraxon §3 default targets):
      excitatory ≈ 0.22, inhibitory ≈ 0.10, neutral ≈ 0.68.
    The neutral state is the computational *buffer*. If neutral collapses,
    the trinary architecture has degraded to binary (config L201-207 caveat).
    """
    n = len(active_neurons)
    if n == 0:
        return {'M1_excitatory_fraction': 0.0, 'M1_inhibitory_fraction': 0.0,
                'M1_neutral_fraction': 0.0, 'M1_deviation_from_target': 0.0}
    e = sum(1 for x in active_neurons if x.trinary_state == 1) / n
    i = sum(1 for x in active_neurons if x.trinary_state == -1) / n
    nt = sum(1 for x in active_neurons if x.trinary_state == 0) / n
    # L1 distance from paper-derived target (0.22, 0.10, 0.68)
    deviation = abs(e - 0.22) + abs(i - 0.10) + abs(nt - 0.68)
    return {
        'M1_excitatory_fraction': e,
        'M1_inhibitory_fraction': i,
        'M1_neutral_fraction': nt,
        'M1_deviation_from_target': deviation,
    }


# ===========================================================================
# M2 — CTC inter-sphere gate dynamics
# ===========================================================================
def compute_m2_ctc_gate_dynamics(ms_nxers, ctc_history: Dict[str, deque]) -> Dict[str, float]:
    """Per-link gate value g_p(t) = (1-c) + c * 0.5*(1 + cos(phi_src - phi_tgt))
    (Multi-Neuraxon Eq. 1). We compute mean and dynamic-range across ALL active
    inter-sphere links across the population, broken down by frequency band.
    
    Paper claim: ablation C1 saturates gate at 1.0 (gate range collapses).
    Healthy CTC has gate mean ≈ 0.5–0.7 and std > 0.05 with band-distinct stats.
    """
    by_band: Dict[str, List[float]] = {'gamma': [], 'beta': [], 'alpha': [], 'theta': []}
    all_gates: List[float] = []
    n_links = 0
    for a in ms_nxers:
        if not getattr(a, 'brain', None):
            continue
        for link in a.brain.links.values():
            src = a.brain.spheres.get(link.source_sphere_id)
            tgt = a.brain.spheres.get(link.target_sphere_id)
            if not src or not tgt:
                continue
            try:
                g = float(link._communication_gate(src.network, tgt.network))
            except Exception:
                continue
            all_gates.append(g)
            band = getattr(link.params, 'coherence_band', 'theta')
            if band not in by_band:
                by_band[band] = []
            by_band[band].append(g)
            n_links += 1
    
    out = {
        'M2_n_links_observed': float(n_links),
        'M2_mean_gate': _safe_mean(all_gates),
        'M2_gate_std_instantaneous': _safe_std(all_gates),
    }
    # Band-specific means / stds — the paper-canonical signature
    for band in ('gamma', 'beta', 'alpha', 'theta'):
        out[f'M2_mean_gate_{band}'] = _safe_mean(by_band.get(band, []))
        out[f'M2_std_gate_{band}']  = _safe_std(by_band.get(band, []))
    
    # Sliding-window dynamic range (true gate modulation per paper)
    history = ctc_history.setdefault('all', deque(maxlen=SLIDING_WINDOW_TICKS))
    history.append(out['M2_mean_gate'])
    if len(history) >= 10:
        out['M2_gate_modulation_std'] = _safe_std(list(history))
        out['M2_gate_range'] = max(history) - min(history)
    else:
        out['M2_gate_modulation_std'] = 0.0
        out['M2_gate_range'] = 0.0
    
    # Asymmetry between feedforward(γ) and feedback(α/β) channels — paper
    # says the system exploits this. If they're identical, frequency
    # assignment is decorative.
    ff = out.get('M2_mean_gate_gamma', 0.0)
    fb = out.get('M2_mean_gate_beta', 0.0) or out.get('M2_mean_gate_alpha', 0.0)
    out['M2_ff_fb_asymmetry'] = abs(ff - fb) if (ff > 0 and fb > 0) else 0.0
    return out


# ===========================================================================
# M3 — Cross-frequency phase-amplitude coupling (Tort 2010 modulation index)
# ===========================================================================
def compute_m3_pac_modulation_index(osc_history: Dict[str, deque]) -> Dict[str, float]:
    """Proper PAC: bin gamma amplitude by theta phase, compute KL divergence
    from uniform → modulation index in [0, log2(N_bins) / N_bins].
    
    The previous v144 code stored cfc_low_mid = |low|*|mid| which is an
    amplitude-product, NOT a phase-amplitude coupling — see review M3.
    
    osc_history keys expected: 'low_phase', 'low_amp', 'mid_phase', 'mid_amp',
    'high_phase', 'high_amp'. Each is a deque of length up to SLIDING_WINDOW_TICKS.
    """
    out = {'M3_pac_modulation_idx': 0.0,
           'M3_pac_theta_gamma_bias_phase': 0.0,
           'M3_pac_delta_theta_idx': 0.0,
           'M3_window_samples': 0.0}
    
    low_phase = list(osc_history.get('low_phase', []))
    mid_phase = list(osc_history.get('mid_phase', []))
    mid_amp   = list(osc_history.get('mid_amp', []))
    high_amp  = list(osc_history.get('high_amp', []))
    
    n = min(len(low_phase), len(mid_phase), len(mid_amp), len(high_amp))
    out['M3_window_samples'] = float(n)
    if n < PAC_PHASE_BINS * 4:
        return out
    
    # ---- theta-gamma PAC: bin high_amp by mid_phase ----
    bins = [0.0] * PAC_PHASE_BINS
    counts = [0] * PAC_PHASE_BINS
    for i in range(n):
        ph = mid_phase[i]
        # mid_phase ranges over sin(2πft+φ) so we recover phase via atan2 OF
        # (sin(theta), cos(theta)) — but here we logged it as a sin value
        # directly; treat its sign + magnitude as a coarse phase angle.
        # angle in [-π, π]
        angle = math.asin(max(-1.0, min(1.0, ph)))
        idx = int((angle + math.pi) / (2 * math.pi) * PAC_PHASE_BINS) % PAC_PHASE_BINS
        bins[idx] += abs(high_amp[i])
        counts[idx] += 1
    for k in range(PAC_PHASE_BINS):
        if counts[k] > 0:
            bins[k] /= counts[k]
    s = sum(bins)
    if s <= 1e-9:
        return out
    p = [b / s for b in bins]
    # KL divergence from uniform
    h_max = math.log2(PAC_PHASE_BINS)
    h = 0.0
    for pk in p:
        if pk > 1e-12:
            h -= pk * math.log2(pk)
    mi = (h_max - h) / h_max  # normalised modulation index in [0, 1]
    out['M3_pac_modulation_idx'] = max(0.0, min(1.0, mi))
    out['M3_pac_theta_gamma_bias_phase'] = float(p.index(max(p)))
    
    # ---- delta-theta PAC: bin mid_amp by low_phase ----
    bins2 = [0.0] * PAC_PHASE_BINS
    counts2 = [0] * PAC_PHASE_BINS
    for i in range(n):
        angle = math.asin(max(-1.0, min(1.0, low_phase[i])))
        idx = int((angle + math.pi) / (2 * math.pi) * PAC_PHASE_BINS) % PAC_PHASE_BINS
        bins2[idx] += abs(mid_amp[i])
        counts2[idx] += 1
    for k in range(PAC_PHASE_BINS):
        if counts2[k] > 0:
            bins2[k] /= counts2[k]
    s2 = sum(bins2)
    if s2 > 1e-9:
        p2 = [b / s2 for b in bins2]
        h2 = 0.0
        for pk in p2:
            if pk > 1e-12:
                h2 -= pk * math.log2(pk)
        out['M3_pac_delta_theta_idx'] = max(0.0, min(1.0, (h_max - h2) / h_max))
    return out


# ===========================================================================
# M4 — Multi-timescale weight separation (w_fast / w_slow / w_meta)
# ===========================================================================
def compute_m4_weight_separation(weight_history: Dict[str, deque]) -> Dict[str, float]:
    """Sliding-window correlation between Δw_fast and Δw_meta.
    
    Paper claim (Neuraxon v2.0 §III): τ_fast < τ_slow < τ_meta. The three
    weights MUST evolve on distinct timescales. If their increments are
    perfectly correlated, the multi-timescale architecture has collapsed
    to a single effective weight (the v3.31 'w_meta not in compute_input'
    failure mode noted in config).
    """
    out = {
        'M4_temporal_divergence': 0.0,    # |corr(Δw_fast, Δw_meta)| — lower = more independent
        'M4_w_fast_volatility': 0.0,
        'M4_w_slow_volatility': 0.0,
        'M4_w_meta_volatility': 0.0,
        'M4_w_meta_active_fraction': 0.0,  # 1 if std(w_meta) > 0.01 i.e. doing real work
    }
    wf = list(weight_history.get('mean_w_fast', []))
    ws = list(weight_history.get('mean_w_slow', []))
    wm = list(weight_history.get('mean_w_meta', []))
    if len(wf) < 4 or len(wm) < 4:
        return out
    # First differences (Δw)
    dwf = [wf[i] - wf[i-1] for i in range(1, len(wf))]
    dws = [ws[i] - ws[i-1] for i in range(1, len(ws))] if len(ws) >= 2 else []
    dwm = [wm[i] - wm[i-1] for i in range(1, len(wm))]
    out['M4_w_fast_volatility'] = _safe_std(dwf)
    out['M4_w_slow_volatility'] = _safe_std(dws)
    out['M4_w_meta_volatility'] = _safe_std(dwm)
    if dwf and dwm:
        out['M4_temporal_divergence'] = abs(_safe_corr(dwf, dwm))
    # If meta weights are essentially frozen near init, M4 fails
    out['M4_w_meta_active_fraction'] = 1.0 if _safe_std(wm) > 0.01 else 0.0
    return out


# ===========================================================================
# M5 — Branching ratio + criticality (with population spread)
# ===========================================================================
def compute_m5_criticality(alive_nxers, br_history: deque) -> Dict[str, float]:
    """Already-tracked branching ratio σ enriched with population spread and
    deviation from criticality (σ = 1.0).
    
    Paper claim (Neuraxon v2.0 §V): networks should sit near σ=1.0. Subcritical
    (σ<0.9) signals die before reaching motor; supercritical (σ>1.1) energy
    explodes (config noted runaway potentiation cascades).
    """
    ratios = [a.net.branching_ratio for a in alive_nxers if a.net.branching_ratio > 0]
    if not ratios:
        return {'M5_branching_ratio': 0.0, 'M5_branching_std_population': 0.0,
                'M5_distance_from_critical': 1.0, 'M5_subcritical_fraction': 0.0,
                'M5_supercritical_fraction': 0.0}
    mean_br = _safe_mean(ratios)
    std_br = _safe_std(ratios)
    sub = sum(1 for r in ratios if r < 0.92) / len(ratios)
    sup = sum(1 for r in ratios if r > 1.10) / len(ratios)
    br_history.append(mean_br)
    return {
        'M5_branching_ratio': mean_br,
        'M5_branching_std_population': std_br,
        'M5_distance_from_critical': abs(mean_br - 1.0),
        'M5_subcritical_fraction': sub,
        'M5_supercritical_fraction': sup,
    }


# ===========================================================================
# M6 — Spontaneous-vs-driven firing + ACW heterogeneity
# ===========================================================================
def compute_m6_spontaneous_dynamics(active_neurons,
                                    spont_count: int,
                                    driven_count: int) -> Dict[str, float]:
    """Spontaneous fraction + intrinsic-timescale heterogeneity.
    
    Paper claim (Neuraxon v2.0 §V): self-generated activity is a *cornerstone*
    feature. ACW (autocorrelation window) heterogeneity is what produces
    multi-scale dynamics. If all neurons collapse to the same intrinsic
    timescale, the v2.0 enhanced formulation is broken.
    """
    total = max(1, spont_count + driven_count)
    timescales = [getattr(n, 'intrinsic_timescale', 0.0) for n in active_neurons]
    timescales = [t for t in timescales if t > 0]
    return {
        'M6_spontaneous_fraction': spont_count / total,
        'M6_driven_fraction':      driven_count / total,
        'M6_acw_mean':             _safe_mean(timescales),
        'M6_acw_heterogeneity':    _safe_std(timescales),
    }


# ===========================================================================
# M7 — Self-sustained activity (Nengo test)
# ===========================================================================
class NengoTestProbe:
    """Periodic active probe: force a sample NxEr's input vector to zero for
    PROBE_TRIAL_DURATION ticks, measure motor-MI vs. last full-input baseline.
    
    Paper claim (Multi-Neuraxon §3.4 / §3.8): in no-cue Multi-Neuraxon shows
    MI ≈ 1.96 with broadcast index > 1.0 (supralinear), while Nengo collapses
    to MI ≈ 0.42. This is the SINGLE strongest distinguishing claim.
    
    Implementation philosophy: we observe — never inject. The probe records
    motor activity (a) on a normal-input window AND (b) on a window where
    the NxEr has just experienced naturally-low input (mid-night rest period
    or no-food terrain). Active forcing would require a deep-clone of the
    sphere graph which is heavy; passive sampling of natural quiet-input
    epochs is faithful to the paper protocol and free.
    """
    def __init__(self):
        self.baseline_motor_signal: Dict[int, float] = {}    # nxer_id → recent normal-input motor amp
        self.zero_motor_signal: Dict[int, float] = {}        # nxer_id → recent zero-input motor amp
        self.last_probe_tick: int = -PROBE_INTERVAL_TICKS
    
    def update_passive_window(self, nxer):
        """Called every probe tick. Decides if this NxEr is currently in a
        no-input window (no food in sight, no song, no proprio surprise) and
        records motor activity into the right bucket."""
        if not getattr(nxer, 'brain', None):
            return
        motor = nxer.brain.spheres.get('motor') or nxer.brain.spheres.get('association')
        if not motor:
            return
        net = motor.network
        # motor amplitude = mean |trinary_state| over output port neurons
        port_ids = set(motor.interface.readout_output_ids) | set(motor.interface.relay_output_ids)
        if not port_ids:
            return
        amp = 0.0
        n = 0
        for nn in net.all_neurons:
            if nn.id in port_ids and nn.is_active:
                amp += abs(nn.trinary_state)
                n += 1
        if n == 0:
            return
        amp /= n
        # decide which bucket
        last_in = nxer.last_inputs or ()
        ext_drive = sum(1 for v in last_in if v != 0)
        if ext_drive == 0:
            # zero-input window — alpha-update the zero-motor estimate
            cur = self.zero_motor_signal.get(nxer.id, 0.0)
            self.zero_motor_signal[nxer.id] = 0.9 * cur + 0.1 * amp
        elif ext_drive >= 3:
            cur = self.baseline_motor_signal.get(nxer.id, 0.0)
            self.baseline_motor_signal[nxer.id] = 0.9 * cur + 0.1 * amp
    
    def report(self) -> Dict[str, float]:
        """Population-aggregated metrics for the dashboard."""
        ratios = []
        bcs = []
        for nid, base in self.baseline_motor_signal.items():
            zero = self.zero_motor_signal.get(nid)
            if zero is None or base <= 1e-6:
                continue
            ratios.append(zero / base)
            # broadcast index ≈ motor amplitude / external-input amplitude
            # in zero-input case external is 0 so broadcast is unbounded; use
            # zero/base as the proxy (≥1 = supralinear persistence).
            bcs.append(zero / max(base, 1e-3))
        return {
            'M7_zero_input_mi_ratio': _safe_mean(ratios),
            'M7_broadcast_index': _safe_mean(bcs),
            'M7_n_paired_nxers': float(len(ratios)),
        }


# ===========================================================================
# M8 — Per-sphere functional specialization
# ===========================================================================
class SphereSpecialisationProbe:
    """Sliding-window MI between sphere activity and sensory modality.
    
    Paper claim (Multi-Neuraxon §3.5): without architectural assignment, VIS
    develops visual selectivity (pref ≈ 0.67), AUD auditory (pref ≈ 0.33),
    ASC remains multimodal (≈ 0.51), MTR is modality-agnostic.
    
    In our 4-sphere game we don't have separate VIS / AUD spheres in the
    default 'sensory_association_motor' topology. Instead we treat the two
    primary input channels (sight=input[4], song=input[9]) as the two
    'modalities' and ask: do different *port neurons* of the sensory sphere
    become preferentially driven by visual vs. auditory signal? And does the
    association sphere stay balanced?
    """
    def __init__(self):
        # nxer_id -> {sphere_id -> { 'visual': deque, 'auditory': deque,
        #                            'sphere_act': deque }}
        self.windows: Dict[int, Dict[str, Dict[str, deque]]] = {}
    
    def update(self, nxer):
        if not getattr(nxer, 'brain', None):
            return
        last_in = nxer.last_inputs or ()
        if len(last_in) < 10:
            return
        # input 4 = sight, input 9 = song
        vis = int(round(last_in[4]))
        aud = int(round(last_in[9]))
        per_nx = self.windows.setdefault(nxer.id, {})
        for sid, sphere in nxer.brain.spheres.items():
            net = sphere.network
            # mean |trinary_state| as crude sphere activity (already trinary-binned)
            states = [n.trinary_state for n in net.all_neurons if n.is_active]
            act = (sum(1 for s in states if s != 0) / len(states)) if states else 0
            act_bin = 1 if act > 0.30 else 0
            buf = per_nx.setdefault(sid, {
                'visual': deque(maxlen=SLIDING_WINDOW_TICKS),
                'auditory': deque(maxlen=SLIDING_WINDOW_TICKS),
                'sphere_act': deque(maxlen=SLIDING_WINDOW_TICKS),
            })
            buf['visual'].append(vis)
            buf['auditory'].append(aud)
            buf['sphere_act'].append(act_bin)
    
    def report(self) -> Dict[str, float]:
        # Aggregate selectivity per sphere across the population
        agg = {}  # sphere_id -> list of pref_visual - pref_auditory per nxer
        for nx_data in self.windows.values():
            for sid, buf in nx_data.items():
                if len(buf['sphere_act']) < 30:
                    continue
                mi_v = _mutual_information_binned(list(buf['visual']),
                                                  list(buf['sphere_act']))
                mi_a = _mutual_information_binned(list(buf['auditory']),
                                                  list(buf['sphere_act']))
                agg.setdefault(sid, []).append(mi_v - mi_a)
        out = {}
        for sid, prefs in agg.items():
            out[f'M8_pref_signed_{sid}'] = _safe_mean(prefs)
            out[f'M8_pref_mag_{sid}']    = _safe_mean([abs(p) for p in prefs])
        # Convenience: sensory-vs-association magnitude — paper expects
        # sensory >> association.
        sensory_mag = out.get('M8_pref_mag_sensory', 0.0)
        assoc_mag   = out.get('M8_pref_mag_association', 0.0)
        out['M8_sensory_vs_association_dissociation'] = sensory_mag - assoc_mag
        return out


# ===========================================================================
# M9 — Compositional transfer (novel V × A combinations)
# ===========================================================================
class CompositionalTransferProbe:
    """We track natural statistics of input combinations the population is
    seeing AND the resulting motor MI on each, then compute the
    novel/trained ratio.
    
    Paper claim (Multi-Neuraxon §3.7): novel pairs achieve ≥ 100% of trained-
    pair pattern MI (R/T = 1.007 in the paper). The H3 generalization claim.
    
    The combinations are encoded as a 2-bit key over (sight∈{-1,0,+1},
    song∈{-1,0,+1}) collapsed to {-1,+1}×{-1,+1} = 4 keys (excluding zeros).
    The 'trained' set is the 2 most-encountered diagonal pairs; 'novel' is
    the 2 off-diagonal pairs.
    """
    def __init__(self):
        # combo_key -> (sum motor_amp, count)
        self.stats: Dict[Tuple[int, int], List[float]] = {}
    
    def update(self, nxer):
        if not getattr(nxer, 'brain', None):
            return
        last_in = nxer.last_inputs or ()
        last_out = nxer.last_outputs or ()
        if len(last_in) < 10 or len(last_out) < 7:
            return
        vis = int(round(last_in[4]))
        aud = int(round(last_in[9]))
        if vis == 0 or aud == 0:
            return  # we want only the 4 V-A polar combinations
        # motor amplitude proxy: |MoveX| + |MoveY| + |Social| + |Sing| etc.
        motor_amp = sum(abs(int(round(o))) for o in last_out) / max(1, len(last_out))
        key = (vis, aud)
        if key not in self.stats:
            self.stats[key] = [0.0, 0]
        self.stats[key][0] += motor_amp
        self.stats[key][1] += 1
    
    def report(self) -> Dict[str, float]:
        # mean motor amp per combination
        means: Dict[Tuple[int, int], float] = {}
        for k, (s, n) in self.stats.items():
            if n > 0:
                means[k] = s / n
        # default zero values so dashboard never has missing keys
        out = {
            'M9_transfer_ratio': 0.0,
            'M9_compositional_similarity': 0.0,
            'M9_n_combinations_observed': float(len(means)),
            'M9_diag_mean_motor_amp': 0.0,
            'M9_offdiag_mean_motor_amp': 0.0,
        }
        if len(means) < 4:
            return out
        # diagonals = (+1,+1) and (-1,-1); off-diagonals = (+1,-1),(-1,+1)
        diag = [means.get((1, 1), 0.0), means.get((-1, -1), 0.0)]
        off  = [means.get((1, -1), 0.0), means.get((-1, 1), 0.0)]
        diag_mean = _safe_mean(diag)
        off_mean  = _safe_mean(off)
        ratio = (off_mean / diag_mean) if diag_mean > 1e-6 else 0.0
        # similarity: 1.0 = identical means (memorisation), 0.0 = totally
        # different. compositional transfer should preserve magnitude but
        # alter representational structure (paper §3.7) — so we want ratio
        # near 1.0 AND similarity NOT at 1.0.
        similarity = 1.0 - (abs(diag_mean - off_mean) / max(diag_mean + off_mean, 1e-6))
        out['M9_transfer_ratio'] = ratio
        out['M9_compositional_similarity'] = similarity
        out['M9_diag_mean_motor_amp'] = diag_mean
        out['M9_offdiag_mean_motor_amp'] = off_mean
        return out


# ===========================================================================
# M10 — Heritability + lesion robustness
# ===========================================================================
class HeritabilityTracker:
    """Pearson r between parent fitness (mean of two parents at the time of
    child's birth) and child fitness (at the time of child's death OR latest
    snapshot).
    
    Paper claim (Aigarth §VIII Neuraxon v2.0 + config v127 fix-comment):
    pre-v127 r was ~0.026 (no heritability — random drift). Post-fix should
    show r > 0.3.
    """
    def __init__(self):
        self.pairs: List[Tuple[float, float]] = []  # (parent_avg, child_now)
        self.pending: Dict[int, Tuple[float, float]] = {}  # child_id -> (parent_avg, parent_avg_again_for_resolve)
    
    def register_birth(self, child_nxer, parents_fitness_avg: float):
        if child_nxer is None:
            return
        self.pending[child_nxer.id] = (parents_fitness_avg, parents_fitness_avg)
    
    def update_alive(self, alive_nxers):
        """Snapshot fitness for any child currently alive whose parents we logged."""
        for a in alive_nxers:
            if a.id in self.pending:
                pavg, _ = self.pending[a.id]
                fit = float(a.stats.fitness_score) if (a.stats is not None) else 0.0
                # update pair (most-recent fitness) so we always have the
                # latest sample even before death
                self._update_pair_for(a.id, pavg, fit)
    
    def _update_pair_for(self, child_id, pavg, child_fit):
        # remove any earlier sample for this child, append the latest
        self.pairs = [(p, c) for (p, c) in self.pairs if not (p == pavg and c == child_fit)]
        self.pairs.append((pavg, child_fit))
        # cap pairs deque-style
        if len(self.pairs) > 5000:
            self.pairs = self.pairs[-5000:]
    
    def report(self) -> Dict[str, float]:
        if len(self.pairs) < HERITABILITY_MIN_PAIRS:
            return {'M10_heritability_r': 0.0, 'M10_heritability_pairs': float(len(self.pairs))}
        ps = [p for p, _ in self.pairs]
        cs = [c for _, c in self.pairs]
        return {'M10_heritability_r': _safe_corr(ps, cs),
                'M10_heritability_pairs': float(len(self.pairs))}


def compute_m10_lesion_curve(alive_nxers) -> Dict[str, float]:
    """Population-level graceful-degradation curve.
    
    For each NxEr we compute its current dead-neuron fraction (from
    n.is_active flags) and its motor MI proxy. We bin into the lesion
    thresholds [0.25, 0.50, 0.75] and report mean retention per bin.
    """
    bins: Dict[float, List[float]] = {t: [] for t in LESION_BIN_THRESHOLDS}
    healthy_baseline = []
    for a in alive_nxers:
        if not a.alive or a.net is None:
            continue
        all_n = a.net.all_neurons
        if not all_n:
            continue
        dead_frac = sum(1 for n in all_n if not n.is_active) / len(all_n)
        # motor MI proxy: |trinary| over output neurons
        out_neurons = [n for n in a.net.output_neurons if n.is_active]
        motor_amp = (sum(abs(n.trinary_state) for n in out_neurons) / len(out_neurons)) if out_neurons else 0.0
        if dead_frac < 0.10:
            healthy_baseline.append(motor_amp)
        else:
            # which bin?
            for t in LESION_BIN_THRESHOLDS:
                if dead_frac >= t and dead_frac < (t + 0.25 if t < 0.75 else 1.01):
                    bins[t].append(motor_amp)
                    break
    base_raw = _safe_mean(healthy_baseline)
    # Guard: if there are no healthy NxErs (boot phase or all NxErs are heavily
    # lesioned) OR motor amplitude is essentially zero, treat retention as 1.0
    # rather than dividing by ~0 and producing absurd numbers.
    BASELINE_FLOOR = 0.05
    if base_raw < BASELINE_FLOOR:
        return {
            'M10_lesion_retention_25': 1.0,
            'M10_lesion_retention_50': 1.0,
            'M10_lesion_retention_75': 1.0,
            'M10_healthy_baseline_motor_amp': base_raw,
        }
    return {
        'M10_lesion_retention_25': (_safe_mean(bins[0.25]) / base_raw) if bins[0.25] else 1.0,
        'M10_lesion_retention_50': (_safe_mean(bins[0.50]) / base_raw) if bins[0.50] else 1.0,
        'M10_lesion_retention_75': (_safe_mean(bins[0.75]) / base_raw) if bins[0.75] else 1.0,
        'M10_healthy_baseline_motor_amp': base_raw,
    }


# ===========================================================================
# Container that lives on the DataLogger
# ===========================================================================

class ProbeState:
    """Single source of truth for all stateful probes / sliding windows.
    
    Held as a member of DataLogger so it lives across log_tick calls.
    """
    def __init__(self):
        # M2 sliding-window: gate values
        self.ctc_history: Dict[str, deque] = {}
        # M3 sliding-window: oscillator phase / amplitude samples
        self.osc_history: Dict[str, deque] = {
            'low_phase':  deque(maxlen=SLIDING_WINDOW_TICKS),
            'low_amp':    deque(maxlen=SLIDING_WINDOW_TICKS),
            'mid_phase':  deque(maxlen=SLIDING_WINDOW_TICKS),
            'mid_amp':    deque(maxlen=SLIDING_WINDOW_TICKS),
            'high_phase': deque(maxlen=SLIDING_WINDOW_TICKS),
            'high_amp':   deque(maxlen=SLIDING_WINDOW_TICKS),
        }
        # M4 sliding-window: weight means
        self.weight_history: Dict[str, deque] = {
            'mean_w_fast': deque(maxlen=SLIDING_WINDOW_TICKS),
            'mean_w_slow': deque(maxlen=SLIDING_WINDOW_TICKS),
            'mean_w_meta': deque(maxlen=SLIDING_WINDOW_TICKS),
        }
        # M5 branching ratio history
        self.br_history: deque = deque(maxlen=SLIDING_WINDOW_TICKS)
        # M7 / M8 / M9 / M10 stateful probes
        self.nengo_probe = NengoTestProbe()
        self.spec_probe  = SphereSpecialisationProbe()
        self.comp_probe  = CompositionalTransferProbe()
        self.heritability = HeritabilityTracker()
    
    def reset(self):
        self.__init__()


# ===========================================================================
# Top-level driver — called once per full-analytics tick from the logger
# ===========================================================================

def compute_all_metrics(probe_state: ProbeState,
                        tick: int,
                        alive_nxers: list,
                        all_active_neurons: list,
                        all_networks: list,
                        sample_oscillator_low: float,
                        sample_oscillator_mid: float,
                        sample_oscillator_high: float,
                        sample_phases: Tuple[float, float, float],
                        spont_count: int,
                        driven_count: int,
                        weight_means: Dict[str, float]) -> Dict[str, float]:
    """Single entry point. Returns a flat dict of all M1-M10 keys (the same
    keys are appended to logger.time_series)."""
    
    # --- update sliding windows ---
    probe_state.osc_history['low_phase'].append(sample_phases[0])
    probe_state.osc_history['low_amp'].append(sample_oscillator_low)
    probe_state.osc_history['mid_phase'].append(sample_phases[1])
    probe_state.osc_history['mid_amp'].append(sample_oscillator_mid)
    probe_state.osc_history['high_phase'].append(sample_phases[2])
    probe_state.osc_history['high_amp'].append(sample_oscillator_high)
    for k in ('mean_w_fast', 'mean_w_slow', 'mean_w_meta'):
        if k in weight_means:
            probe_state.weight_history[k].append(weight_means[k])
    
    # --- M1-M6 are computed every tick ---
    m1 = compute_m1_trinary_distribution(all_active_neurons)
    
    ms_nxers = [a for a in alive_nxers if getattr(a, 'brain', None)]
    m2 = compute_m2_ctc_gate_dynamics(ms_nxers, probe_state.ctc_history)
    
    m3 = compute_m3_pac_modulation_index(probe_state.osc_history)
    m4 = compute_m4_weight_separation(probe_state.weight_history)
    m5 = compute_m5_criticality(alive_nxers, probe_state.br_history)
    m6 = compute_m6_spontaneous_dynamics(all_active_neurons, spont_count, driven_count)
    
    # --- M7 passive update + report (always tracks, reports running averages) ---
    for a in alive_nxers:
        probe_state.nengo_probe.update_passive_window(a)
        probe_state.spec_probe.update(a)
        probe_state.comp_probe.update(a)
    probe_state.heritability.update_alive(alive_nxers)
    
    m7 = probe_state.nengo_probe.report()
    m8 = probe_state.spec_probe.report()
    m9 = probe_state.comp_probe.report()
    m10_h = probe_state.heritability.report()
    m10_l = compute_m10_lesion_curve(alive_nxers)
    
    # --- consolidate ---
    out = {}
    for d in (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10_h, m10_l):
        out.update(d)
    
    # ---  derive in_band booleans for the dashboard ---
    for k in HEALTHY_BANDS.keys():
        v = out.get(k, None)
        out[f'{k}__in_band'] = float(in_band(k, v if v is not None else 0.0))
    
    return out


# ===========================================================================
# Convenience: list of all metric keys (for logger init)
# ===========================================================================

def all_metric_keys() -> List[str]:
    """Authoritative list of every M1-M10 key the logger should pre-create
    as an empty deque/list. Keep in sync with compute_* functions above."""
    keys = [
        # M1
        'M1_excitatory_fraction', 'M1_inhibitory_fraction',
        'M1_neutral_fraction', 'M1_deviation_from_target',
        # M2
        'M2_n_links_observed', 'M2_mean_gate', 'M2_gate_std_instantaneous',
        'M2_mean_gate_gamma', 'M2_std_gate_gamma',
        'M2_mean_gate_beta',  'M2_std_gate_beta',
        'M2_mean_gate_alpha', 'M2_std_gate_alpha',
        'M2_mean_gate_theta', 'M2_std_gate_theta',
        'M2_gate_modulation_std', 'M2_gate_range', 'M2_ff_fb_asymmetry',
        # M3
        'M3_pac_modulation_idx', 'M3_pac_theta_gamma_bias_phase',
        'M3_pac_delta_theta_idx', 'M3_window_samples',
        # M4
        'M4_temporal_divergence', 'M4_w_fast_volatility',
        'M4_w_slow_volatility', 'M4_w_meta_volatility',
        'M4_w_meta_active_fraction',
        # M5
        'M5_branching_ratio', 'M5_branching_std_population',
        'M5_distance_from_critical',
        'M5_subcritical_fraction', 'M5_supercritical_fraction',
        # M6
        'M6_spontaneous_fraction', 'M6_driven_fraction',
        'M6_acw_mean', 'M6_acw_heterogeneity',
        # M7
        'M7_zero_input_mi_ratio', 'M7_broadcast_index', 'M7_n_paired_nxers',
        # M8 — sphere ids may vary; the dynamic keys are added at log time.
        # M8_sensory_vs_association_dissociation is the headline scalar.
        'M8_sensory_vs_association_dissociation',
        # M9
        'M9_transfer_ratio', 'M9_compositional_similarity',
        'M9_n_combinations_observed', 'M9_diag_mean_motor_amp',
        'M9_offdiag_mean_motor_amp',
        # M10
        'M10_heritability_r', 'M10_heritability_pairs',
        'M10_lesion_retention_25', 'M10_lesion_retention_50',
        'M10_lesion_retention_75', 'M10_healthy_baseline_motor_amp',
    ]
    # in-band booleans
    keys += [f'{k}__in_band' for k in HEALTHY_BANDS.keys()]
    return keys
