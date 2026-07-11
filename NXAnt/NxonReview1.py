# Neuraxon Ant Colony 1.03 internal version 10
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonReview1.py — independent verification (v1.03, integer trit-LUT).

Rebuilds a solution's LUT from its derivation (root + pubkey + nonce by replaying
the walk) and re-runs the integer trit sim across several digests, reporting the
recomputed integer score and band satisfaction. This is the SAME recompute the
consensus nodes do, run independently — a cross-check that the recorded score is
reproducible from public data.

Usage:
  python3 NxonReview1.py <system_file.json>            # verify all solutions
  python3 NxonReview1.py <system_file.json> <hash>     # verify one solution
"""

import sys
import json

import NxonGenome as G
import NxonScore
import NxonTrit as TR


def main():
    if len(sys.argv) < 2:
        print("usage: python3 NxonReview1.py <system_file.json> [solution_hash]")
        sys.exit(1)
    path = sys.argv[1]
    only = sys.argv[2] if len(sys.argv) > 2 else None
    with open(path) as f:
        data = json.load(f)

    epoch = G.build_epoch(data["salted_digest"], N=data["N"], ticks=data["ticks"])
    walk_steps = data["walk_steps"]

    print("=" * 74)
    print("NXON REVIEW 1.03 (trit) — Independent Reproducibility Check")
    print("=" * 74)
    print("System file:  {}".format(path))
    print("Salted digest:{}".format(data["salted_digest"]))
    print("Sim: N={} ticks={} walk={}".format(data["N"], data["ticks"], walk_steps))

    luts = {data["root"]: G.root_lut(epoch)}
    recs = sorted(data["solutions"],
                  key=lambda r: (r["accept_tick"], r["accept_age"], r["hash"]))
    mism = 0
    checked = 0
    shown = 0
    for r in recs:
        if r["submitter"] == "ROOT":
            continue
        parent = luts.get(r["parent"])
        if parent is None:
            continue
        best_lut, best_score, best_m, _ = TR.mining_walk(
            parent, r["pubkey"], r["nonce"], epoch, walk_steps)
        luts[r["hash"]] = best_lut
        checked += 1
        ok = (best_score == r["score"]) and (G.hash_lut(best_lut, epoch) == r["hash"])
        if not ok:
            mism += 1
        if only is not None and r["hash"].startswith(only):
            print("\nSolution {}  by {}".format(r["hash"][:12], r["submitter"]))
            print("  recorded score: {}".format(r["score"]))
            print("  recomputed:     {}  (match: {})".format(best_score, best_score == r["score"]))
            print("  bands satisfied:{:.0%}".format(NxonScore.band_satisfaction(best_m)))
            for k, v in best_m.items():
                print("    {:<22} {:.3f}".format(k, NxonScore.fp_to_float(v)))
            shown += 1

    print("\nChecked {} solutions | score+hash mismatches: {}".format(checked, mism))
    print("VERDICT: {}".format("PASS — fully reproducible from public data"
                               if mism == 0 else "FAIL — {} mismatch(es)".format(mism)))


if __name__ == "__main__":
    main()
