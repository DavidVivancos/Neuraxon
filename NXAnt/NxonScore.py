# Neuraxon Ant Colony 1.04 internal version 11
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonScore.py — PUBLIC integer band-score over the trit metrics (v1.03).

Same public-law role as before: every node imports this and recomputes the same
integer score, so there is no secret. What changed in v1.03: the metrics arrive
as FIXED-POINT INTEGERS (value x METRIC_SCALE) straight from the integer trit
sim, so scoring is integer end-to-end — no float appears in the state, the
update, the metrics, or the score. This is what makes the consensus scalar
bit-identical across all hardware without any FP-pinning caveat.

RESOLUTION: the trit state is only 3-valued, but each metric is a ratio of
integer counts accumulated over every non-input neuron across a whole phase
(dozens of neurons x dozens of ticks), so the numerators/denominators are large
and the fixed-point ratio keeps ~6 digits of resolution. Coarse state, fine
metrics.
"""

METRIC_SCALE = 1_000_000          # fixed-point grain for metrics AND bands
PENALTY_RESOLUTION = 1000         # per-metric reward resolution


# Public healthy-bands (human-readable floats; converted to fixed-point once).
# Tuned so the ROOT is beatable and the search has gradient on the trit CA.
TARGET_BANDS_F = {
    "excitatory_fraction":  [0.35, 0.65],   # balanced +/- among firing neurons
    "active_fraction":      [0.30, 0.72],   # driven: alive but not fully saturated
    "branching":            [0.80, 1.20],   # criticality (~1) across driven ticks
    "spontaneous_fraction": [0.20, 0.75],   # silence: self-sustained but decaying
    "transfer_ratio":       [0.40, 1.30],   # responds to novel transfer input
    "saturation":           [0.00, 0.55],   # not frozen in one state
}

METRIC_WEIGHTS_INT = {
    "excitatory_fraction":  10,
    "active_fraction":      10,
    "branching":            15,
    "spontaneous_fraction": 10,
    "transfer_ratio":       10,
    "saturation":           10,
}


def _to_fp_bands():
    return {k: [int(round(lo * METRIC_SCALE)), int(round(hi * METRIC_SCALE))]
            for k, (lo, hi) in TARGET_BANDS_F.items()}


# Fixed-point bands (integers), what the consensus score actually compares.
TARGET_BANDS = _to_fp_bands()


# ============================================================================
# OBJECTIVE MODE  (v1.04, Q1 + Q5)
# ============================================================================
# "banded"    — bounded band-score. SATURATES at a ceiling once every metric is
#               centred; fine for a self-contained trit demo, but the colony
#               STALLS at the ceiling because no child can then strictly beat its
#               parent (exactly the Q5 problem).
# "unbounded" — keeps climbing past the band ceiling. On top of the in-band
#               reward it adds a small MARGIN term (how deep inside each band the
#               metric sits) so there is always a strictly-better move. Mirrors
#               the real ARC NAS fitness, which is unbounded ("best score wins").
# "external"  — the score is supplied by an external evaluator (the real
#               ARC-AGI3 NAS fitness). See register_external_scorer().
OBJECTIVE_MODE = "unbounded"

# Weight of the margin term in "unbounded" mode. Small so band membership still
# dominates ranking, but nonzero so gradient never disappears past the ceiling.
MARGIN_WEIGHT = 1

_EXTERNAL_SCORER = None


def register_external_scorer(fn):
    """Plug in the real ARC-AGI3 NAS fitness as the consensus scalar. `fn` takes
    the fixed-point metrics dict and returns an INTEGER (higher = better). Use
    with OBJECTIVE_MODE='external'."""
    global _EXTERNAL_SCORER
    _EXTERNAL_SCORER = fn


def _band_reward(metrics):
    """In-band reward (bounded part). 0 = floor; ceiling when all centred."""
    total = 0
    for key, (lo, hi) in TARGET_BANDS.items():
        if key not in metrics:
            continue
        w = METRIC_WEIGHTS_INT.get(key, 10)
        v = metrics[key]
        width = hi - lo
        if width < 1:
            width = 1
        if v < lo:
            dist = lo - v
        elif v > hi:
            dist = v - hi
        else:
            dist = 0
        reward = 0 if dist >= width else (PENALTY_RESOLUTION * (width - dist)) // width
        total += w * reward
    return total


def _margin_reward(metrics):
    """Centre-seeking term: how deep INSIDE each band a metric sits (0 at edge,
    max at exact centre). A walk can always nudge toward centres, so this term
    keeps a strictly-better move available past the band ceiling. Pure integer."""
    total = 0
    for key, (lo, hi) in TARGET_BANDS.items():
        if key not in metrics:
            continue
        w = METRIC_WEIGHTS_INT.get(key, 10)
        v = metrics[key]
        if v < lo or v > hi:
            continue
        centre = (lo + hi) // 2
        half = max((hi - lo) // 2, 1)
        depth = half - abs(v - centre)
        total += w * ((PENALTY_RESOLUTION * depth) // half)
    return total


def consensus_score_int(metrics):
    """THE consensus scalar (integer, higher = healthier, 0 = floor).

    `metrics` values are FIXED-POINT integers (value x METRIC_SCALE) from the
    trit sim. Mode-dependent (OBJECTIVE_MODE). In 'unbounded'/'external' there is
    no saturating ceiling, which is what lets the ant colony keep growing (Q5).
    Every node computes this identically from the same metrics.
    """
    if OBJECTIVE_MODE == "external" and _EXTERNAL_SCORER is not None:
        return int(_EXTERNAL_SCORER(metrics))
    band = _band_reward(metrics)
    if OBJECTIVE_MODE == "banded":
        return band
    return band + MARGIN_WEIGHT * _margin_reward(metrics)


def band_satisfaction(metrics):
    """Fraction of bands satisfied (float, human display only). Expects
    fixed-point metrics."""
    if not TARGET_BANDS:
        return 0.0
    inside = counted = 0
    for key, (lo, hi) in TARGET_BANDS.items():
        if key not in metrics:
            continue
        counted += 1
        if lo <= metrics[key] <= hi:
            inside += 1
    return inside / max(counted, 1)


def fp_to_float(v):
    """Convert a fixed-point metric back to float for display only."""
    return v / METRIC_SCALE
