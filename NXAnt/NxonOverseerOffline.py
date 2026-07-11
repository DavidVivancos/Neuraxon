# Neuraxon Ant Colony 1.03 internal version 10
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonOverseerOffline.py — the OFFLINE half of the "Overseer" role (v1.03).

The nodes run consensus (verify + score + validity + record). This tool is what
YOU run offline against the recorded system file. Nothing here runs on-node.

Because the system file stores only the DERIVATION of each solution
(pubkey, nonce, parent) plus its hash and score, this tool first REBUILDS every
LUT by replaying the anti-attractor walk from the ROOT — reproducing exactly what
the nodes computed (and confirming its own scores match the recorded scores,
0 mismatches). Then it does the genuinely-offline work:

  - multi-objective / Pareto re-ranking among accepted solutions (the consensus
    tree ratchets on ONE integer scalar; near-equal winners are separated here);
  - MultiNeuraxon2 brain ASSEMBLY: compose the recorded winner LUTs into a larger
    multi-sphere brain spec (there is no separate MultiNeuraxon2.py — assembly
    stitches recorded modules into a bigger composition);
  - curriculum tightening PROPOSAL (offline decision; its OUTPUT must then be
    published deterministically to nodes so the next epoch stays reproducible).

Usage:  python3 NxonOverseerOffline.py system_file_node_00.json
"""

import sys
import json

import NxonGenome as G
import NxonScore
import NxonTrit as TR


def load(path):
    with open(path) as f:
        return json.load(f)


def rebuild_luts(data):
    """Replay the walk from ROOT to reconstruct every solution's LUT. Returns
    {hash: lut}, the epoch, and a mismatch count (recorded vs recomputed score)."""
    epoch = G.build_epoch(data["salted_digest"], N=data["N"], ticks=data["ticks"])
    walk_steps = data["walk_steps"]
    luts = {data["root"]: G.root_lut(epoch)}
    # Solutions must be rebuilt parent-before-child; sort by accept order.
    recs = sorted(data["solutions"],
                  key=lambda r: (r["accept_tick"], r["accept_age"], r["hash"]))
    mismatches = 0
    checked = 0
    for r in recs:
        if r["submitter"] == "ROOT":
            continue
        parent = luts.get(r["parent"])
        if parent is None:
            continue
        best_lut, best_score, _, _ = TR.mining_walk(
            parent, r["pubkey"], r["nonce"], epoch, walk_steps)
        luts[r["hash"]] = best_lut
        checked += 1
        if best_score != r["score"]:
            mismatches += 1
    return luts, epoch, mismatches, checked


def pareto_front(data, luts, epoch):
    enriched = []
    for r in data["solutions"]:
        if r["submitter"] == "ROOT" or r["hash"] not in luts:
            continue
        m = TR.run_sim(luts[r["hash"]], epoch)
        obj = (
            r["score"],
            NxonScore.band_satisfaction(m),
            # parsimony: fewer non-NEUTRAL LUT lines (a simpler brain)
            -sum(1 for n in epoch["noninput_ids"] for t in luts[r["hash"]][n]
                 if t != G.NEUTRAL),
        )
        enriched.append((r, obj, m))
    front = []
    for i, (ri, oi, mi) in enumerate(enriched):
        dominated = False
        for j, (rj, oj, mj) in enumerate(enriched):
            if i == j:
                continue
            if all(oj[k] >= oi[k] for k in range(len(oi))) and any(
                    oj[k] > oi[k] for k in range(len(oi))):
                dominated = True
                break
        if not dominated:
            front.append((ri, oi, mi))
    front.sort(key=lambda t: (-t[1][0], -t[1][1]))
    return front


def assemble_multineuraxon2(front, luts, epoch, max_modules=4):
    roles = ["assoc_fluid", "assoc_cryst", "sensory", "motor"]
    modules = []
    for i, (r, obj, m) in enumerate(front[:max_modules]):
        modules.append({
            "module_index": i, "role": roles[i % len(roles)],
            "source_hash": r["hash"], "consensus_score": r["score"],
            "lut": G.lut_to_str(luts[r["hash"]], epoch),
        })
    return {
        "assembly": "MultiNeuraxon2 (chc6-style composition of recorded trit modules)",
        "note": "Realize in the real codebase via NEURAXON_ARCH=<spec> for a "
                "full-fidelity confirmation. Not run on-node.",
        "N_per_module": epoch["N"], "module_count": len(modules), "modules": modules,
    }


def propose_curriculum_tightening(front):
    if not front:
        return None
    keys = list(NxonScore.TARGET_BANDS_F.keys())
    acc = {k: [] for k in keys}
    for (r, obj, m) in front:
        for k in keys:
            if k in m:
                acc[k].append(NxonScore.fp_to_float(m[k]))
    proposal = {}
    for k in keys:
        lo, hi = NxonScore.TARGET_BANDS_F[k]
        if acc[k]:
            vals = sorted(acc[k])
            med = vals[len(vals) // 2]
            nlo = lo + 0.2 * (min(med, (lo + hi) / 2) - lo)
            nhi = hi - 0.2 * (hi - max(med, (lo + hi) / 2))
            proposal[k] = [round(nlo, 4), round(nhi, 4)] if nlo < nhi else [lo, hi]
        else:
            proposal[k] = [lo, hi]
    return proposal


def main():
    if len(sys.argv) < 2:
        print("usage: python3 NxonOverseerOffline.py <system_file.json>")
        sys.exit(1)
    path = sys.argv[1]
    data = load(path)

    print("=" * 76)
    print("NxonAnt 1.03 — OFFLINE OVERSEER (curation of the agreed population)")
    print("=" * 76)
    print("System file:        {}".format(path))
    print("Recorded solutions: {}".format(len(data["solutions"])))
    print("Salted digest:      {}".format(data["salted_digest"]))
    print("Sim: N={} ticks={} walk={}".format(data["N"], data["ticks"], data["walk_steps"]))
    print("(Nothing here runs on the node — this is downstream of consensus.)")

    print("\n[1] Rebuild LUTs by replaying the walk from ROOT + re-verify scores:")
    luts, epoch, mismatches, checked = rebuild_luts(data)
    print("    rebuilt {} LUTs; recorded-vs-recomputed score mismatches: {}".format(
        checked, mismatches))
    print("    (0 mismatches == the system file is reproducible from public data")
    print("     and our offline scoring == the on-node scoring)")

    print("\n[2] Pareto front (consensus score / bands / parsimony):")
    front = pareto_front(data, luts, epoch)
    print("    {:<14} {:>10} {:>8} {:>12}".format("hash", "score", "bands", "lit_lines"))
    for (r, obj, m) in front[:8]:
        print("    {:<14} {:>10} {:>7.0%} {:>12}".format(
            r["hash"][:12], r["score"], obj[1], -obj[2]))

    print("\n[3] MultiNeuraxon2 assembly (compose winner modules):")
    spec = assemble_multineuraxon2(front, luts, epoch)
    out_spec = path.replace(".json", "") + "_assembly.json"
    with open(out_spec, "w") as f:
        json.dump(spec, f, indent=2)
    print("    composed {} modules -> {}".format(spec["module_count"], out_spec))

    print("\n[4] Curriculum tightening proposal (publish deterministically to nodes):")
    prop = propose_curriculum_tightening(front)
    if prop:
        for k, band in list(prop.items())[:4]:
            print("    {:<24} -> {}".format(k, band))
        out_prop = path.replace(".json", "") + "_next_bands.json"
        with open(out_prop, "w") as f:
            json.dump(prop, f, indent=2)
        print("    full proposal -> {}".format(out_prop))
    print("\nDone. Node side stays minimal (verify + score + rules + record);")
    print("all of the above is offline curation.")


if __name__ == "__main__":
    main()
