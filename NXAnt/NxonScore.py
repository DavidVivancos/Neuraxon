# Neuraxon Ant Colony 1.03 internal version 10
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


def consensus_score_int(metrics):
    """THE consensus scalar. Pure-integer, NON-NEGATIVE, higher = healthier,
    0 = floor (so ROOT can start at 0 and children climb above it).

    `metrics` values are FIXED-POINT integers (value x METRIC_SCALE) produced by
    the trit sim. Each metric rewards up to PENALTY_RESOLUTION x weight when
    centred in its band, decaying to 0 one band-width outside. Every node
    computes this identically from the same metrics.
    """
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
        if dist >= width:
            reward = 0
        else:
            reward = (PENALTY_RESOLUTION * (width - dist)) // width
        total += w * reward
    return total


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
