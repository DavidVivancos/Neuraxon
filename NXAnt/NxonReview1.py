# Neuraxon Ant Colony 1.02 internal version 09
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonReview1.py — independent verification for NxonAnt 1.02 (chc6, consensus edition).

Loads a best_architecture.json produced by the colony and INDEPENDENTLY
re-evaluates it on the compressed Neuraxon proxy, using seeds that are
completely disjoint from the colony's. Reports:

  - the metric vector across N independent trials (mean +/- std)
  - which target bands the architecture satisfies on average
  - a band-satisfaction rate and an overall PASS / WEAK / FAIL verdict

This is the external check that the architecture the colony found is genuinely
good — not an artifact of a lucky seed. Because the proxy is deterministic
given (genome, seed) but stochastic across seeds (random connectivity, jitter,
spontaneous firing), averaging over many independent seeds reveals whether the
architecture RELIABLY lands in the healthy bands.
"""

import sys
import json
import math
import statistics

import NxonAnt
import NxonOverseer


def review(path, n_trials=30, review_seed=99999):
    with open(path) as f:
        record = json.load(f)

    genome = record["genome"]
    md = record.get("metadata", {})
    bands = md.get("target_bands", NxonOverseer.DEFAULT_TARGET_BANDS)
    colony_score = record.get("score", "?")

    print("=" * 72)
    print("NXON REVIEW 1.02 (chc6, consensus) — Independent Architecture Verification")
    print("=" * 72)
    print("Architecture:    {}".format(path))
    print("Found in epoch:  {}".format(md.get("found_in_epoch", "?")))
    print("Registered by:   {}".format(md.get("registered_by", "?")))
    print("Hash:            {}".format(md.get("hash", "?")))
    print("Colony score:    {}".format(colony_score))
    print("Review seed:     {}  (independent of colony)".format(review_seed))
    print("N trials:        {}".format(n_trials))

    # Show the architecture's key parameters.
    neural = genome.get("neural", {})
    print("\nArchitecture parameters (searchable):")
    print("  hidden neurons:        {}".format(neural.get("num_hidden_neurons_default")))
    print("  connection prob:       {:.3f}".format(neural.get("connection_probability", 0)))
    print("  firing threshold (E):  {:.3f}".format(neural.get("firing_threshold_excitatory", 0)))
    print("  sensorimotor coupling: {:.3f}".format(neural.get("sensorimotor_coupling", 0)))
    print("  refractory ticks:      {}".format(neural.get("refractory_period_ticks")))
    print("  symmetric STDP:        {}".format(neural.get("symmetric_stdp")))

    # Run N independent trials.
    print("\nRunning {} independent evaluations...".format(n_trials))
    import time
    metric_samples = {}
    for trial in range(n_trials):
        vseed = review_seed + trial * 101 + 7
        metrics, _, timed_out = NxonAnt._evaluate_architecture(
            genome, vseed, time.time() + 5.0)
        if timed_out or metrics is None:
            continue
        for k, v in metrics.items():
            metric_samples.setdefault(k, []).append(v)

    # Aggregate.
    print("\n" + "=" * 72)
    print("REVIEW RESULTS — metric distributions across independent seeds")
    print("=" * 72)
    print("  {:<30} {:>9} {:>9} {:>17} {:>6}".format(
        "metric", "mean", "std", "target band", "in?"))
    bands_satisfied = 0
    bands_counted = 0
    for k in sorted(metric_samples.keys()):
        samples = metric_samples[k]
        mean = statistics.mean(samples)
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        band = bands.get(k)
        if band:
            bands_counted += 1
            inside = band[0] <= mean <= band[1]
            if inside:
                bands_satisfied += 1
            band_str = "[{:.3f},{:.3f}]".format(band[0], band[1])
            mark = "IN " if inside else "OUT"
        else:
            band_str = "--"
            mark = ""
        print("  {:<30} {:>9.4f} {:>9.4f} {:>17} {:>6}".format(
            k, mean, std, band_str, mark))

    sat_rate = bands_satisfied / max(bands_counted, 1)
    # Overall band-distance score on the mean metrics.
    mean_metrics = {k: statistics.mean(v) for k, v in metric_samples.items()}
    overall_score = NxonOverseer.band_score(
        mean_metrics, bands, NxonOverseer.METRIC_WEIGHTS)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("Bands satisfied:      {}/{}  ({:.0%})".format(
        bands_satisfied, bands_counted, sat_rate))
    print("Overall band score:   {:+.4f}  (0 = all bands satisfied)".format(overall_score))

    if sat_rate >= 0.75:
        print("VERDICT: PASS — architecture reliably lands in the healthy bands.")
        print("         The colony found a genuinely good Neuraxon architecture.")
    elif sat_rate >= 0.5:
        print("VERDICT: WEAK — architecture satisfies about half the bands.")
        print("         Promising but needs more search (more epochs / compute).")
    else:
        print("VERDICT: FAIL — architecture satisfies few bands on independent seeds.")
        print("         The colony needs more search or a different starting point.")

    return sat_rate, overall_score


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "nxon_outputs/best_architecture.json"
    n = 30
    seed = 99999
    for i, a in enumerate(sys.argv):
        if a == "--n-trials" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        if a == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
    review(path, n_trials=n, review_seed=seed)
