# Neuraxon Ant Colony 1.02 internal version 09
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonScore.py — PUBLIC, DETERMINISTIC scoring law (the consensus scalar).

In the Qubic node-consensus model EVERY node recomputes the score, so the score
function, the healthy-band targets, and the weights are PUBLIC and identical on
every node. THERE IS NO SECRET. This module is that shared public law. It is
imported unchanged by:
  - NxonNode.py            (the on-node consensus verifier)
  - NxonOverseerOffline.py (your offline curation tool)
  - NxonReview1.py         (independent re-verification)
so the on-node scalar and the offline re-check are computed by the SAME code —
a built-in consistency guarantee.

WHY THE SCALAR IS AN INTEGER
----------------------------
Floating-point results can differ across heterogeneous hardware (compiler,
libm, FMA, rounding). If nodes compared raw floats they could disagree and break
consensus. `consensus_score_int` quantizes the metrics to a fixed grain and does
the band arithmetic in pure integers, so the scalar every node compares is
bit-identical given the same metrics.

WHAT INTEGER SCORING DOES *NOT* FIX
-----------------------------------
Integer scoring removes nondeterminism in the SCORING step only. Full cross-node
agreement ALSO requires the SIMULATION to be deterministic across hardware
(so every node produces the same metrics to begin with). This reference impl is
deterministic within one Python build; for real deployment the chc6 sim must run
in fixed-point or a pinned FP mode, and the metric-quantization grain
(METRIC_SCALE / QUANT) must be coarser than any residual cross-node drift so a
1-ULP difference can never flip the integer score. This is the one hard
engineering requirement to carry into production.
"""

# Public healthy-band targets (the "5 weighted fitness components" analogue:
# the bands the ant's raw metrics are judged against). PUBLIC, on every node.
TARGET_BANDS = {
    "M1_excitatory_fraction": [0.18, 0.28],
    "M2_mean_gate": [0.40, 0.85],
    "M5_branching_ratio": [0.92, 1.10],
    "M6_spontaneous_fraction": [0.10, 0.45],
    "M7_zero_input_mi_ratio": [0.40, 1.20],
    "M9_transfer_ratio": [0.85, 1.30],
    "sensory_motor_corr": [0.20, 1.00],
    "input_saturation_fraction": [0.00, 0.30],
}

# Integer weights (the float weights x10, so 1.0 -> 10 and 1.5 -> 15) so all
# scoring arithmetic stays in integers. PUBLIC, on every node.
METRIC_WEIGHTS_INT = {
    "M1_excitatory_fraction": 10,
    "M2_mean_gate": 10,
    "M5_branching_ratio": 15,
    "M6_spontaneous_fraction": 10,
    "M7_zero_input_mi_ratio": 10,
    "M9_transfer_ratio": 10,
    "sensory_motor_corr": 15,
    "input_saturation_fraction": 10,
}

# Fixed-point grain. Metrics are quantized to 1/METRIC_SCALE before scoring.
METRIC_SCALE = 1_000_000
# Per-metric penalty resolution (band-widths normalized to this many steps).
PENALTY_RESOLUTION = 1000


def _q(v):
    """Quantize a metric to the fixed integer grain (deterministic)."""
    return int(round(float(v) * METRIC_SCALE))


def consensus_score_int(metrics):
    """THE consensus scalar. Pure-integer function of the (quantized) metrics.

    Returns a NON-NEGATIVE int FITNESS where HIGHER IS BETTER and 0 is the floor.
    This convention lets the ROOT start at score 0 (the worst possible) and lets
    every accepted child climb strictly above its parent, exactly as the recorded
    protocol specifies ("ROOT score = 0", "score > parent score (STRICT)").

    Construction: each metric contributes 0 when out-of-band and up to a fixed
    per-metric maximum when perfectly centred in its band; the score is the
    weighted sum. Every node computes it identically from the same metrics.
    """
    total = 0
    for key, (lo, hi) in TARGET_BANDS.items():
        if key not in metrics:
            continue
        w = METRIC_WEIGHTS_INT.get(key, 10)
        v = _q(metrics[key])
        loq, hiq = _q(lo), _q(hi)
        width = hiq - loq
        if width < 1:
            width = 1
        if v < loq:
            dist = loq - v
        elif v > hiq:
            dist = v - hiq
        else:
            dist = 0
        # In-band reward: PENALTY_RESOLUTION at the edge of the band, decaying to
        # 0 once the metric is one full band-width outside. Pure integer.
        if dist >= width:
            reward = 0
        else:
            reward = (PENALTY_RESOLUTION * (width - dist)) // width
        total += w * reward
    return total


def band_satisfaction(metrics):
    """Fraction of bands satisfied (FLOAT — for human display only, never used
    in the consensus decision)."""
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


def band_score_float(metrics):
    """Continuous fitness (FLOAT — for human display / offline analysis only,
    NOT the consensus scalar). Higher is better, matching consensus_score_int."""
    total = 0.0
    for key, (lo, hi) in TARGET_BANDS.items():
        if key not in metrics:
            continue
        w = METRIC_WEIGHTS_INT.get(key, 10) / 10.0
        v = metrics[key]
        width = max(hi - lo, 1e-6)
        if v < lo:
            dist = lo - v
        elif v > hi:
            dist = v - hi
        else:
            dist = 0.0
        reward = max(0.0, (width - dist) / width)
        total += w * reward
    return total
