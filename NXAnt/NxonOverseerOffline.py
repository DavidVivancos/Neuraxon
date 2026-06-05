# Neuraxon Ant Colony 1.02 internal version 09
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonOverseerOffline.py — the OFFLINE half of the "Overseer" role.

The nodes run consensus (verify + score + validity + record). This tool is what
YOU run offline, reading the recorded system file. None of this runs on the node
or in consensus — it is downstream consumption of the agreed population:

  - Multi-objective / Pareto re-ranking among accepted solutions. The CONSENSUS
    tree ratchets on the single integer scalar (strict-greater); near-equal
    winners are separated here by additional objectives the node never computes.
  - MultiNeuraxon2 brain ASSEMBLY: compose the recorded winners into a larger
    multi-sphere brain (there is no separate MultiNeuraxon2.py — "Multi-Neuraxon
    2.0" is the chc6 multi-sphere architecture; assembly stitches recorded
    sphere-genomes into a bigger composition).
  - Curriculum tightening PROPOSAL for the next epoch (the design decision is
    offline; its OUTPUT must then be published deterministically to the nodes,
    e.g. derived from the consensus digest, so the scored sim stays reproducible).
  - Analytics / reputation.

Usage:  python3 NxonOverseerOffline.py system_file_node_00.json
"""

import sys
import json
import time

import NxonAnt
import NxonScore


def load_system_file(path):
    with open(path) as f:
        return json.load(f)


def rescore_metrics(genome, seed=99):
    """Recompute raw metrics offline (same sim the nodes ran). Lets us verify
    our offline scoring agrees with the on-node integer score, and gives us the
    raw metric vector for multi-objective analysis."""
    m, _, timed_out = NxonAnt._evaluate_architecture(
        genome, seed, time.time() + 5.0)
    return None if timed_out else m


def pareto_front(records):
    """Multi-objective Pareto selection among accepted solutions.

    Objectives (all 'more is better' after transforms):
      - consensus score (already integer; higher better)
      - bands satisfied (higher better)
      - parsimony: fewer hidden neurons (we negate)
    A record is on the front if no other record dominates it on all objectives.
    """
    enriched = []
    for r in records:
        if r["submitter"] == "ROOT":
            continue
        m = rescore_metrics(r["genome"])
        if m is None:
            continue
        obj = (
            r["score"],
            NxonScore.band_satisfaction(m),
            -int(r["genome"]["neural"]["num_hidden_neurons_default"]),
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


def assemble_multineuraxon2(front, max_modules=4):
    """Compose the top recorded winners into a larger Multi-Neuraxon 2.0 brain.

    A million-neuron brain is many ant-sized modules; here we take the Pareto
    winners as candidate MODULES and describe a composition (sphere roles +
    inter-sphere wiring). This produces an assembly SPEC (JSON) the real codebase
    can realize via NEURAXON_ARCH=; the heavy full-fidelity run happens there,
    never in the 1-second ant or on the consensus node.
    """
    roles = ["assoc_fluid_donor", "assoc_cryst_donor", "sensory_donor", "motor_donor"]
    modules = []
    for i, (r, obj, m) in enumerate(front[:max_modules]):
        modules.append({
            "module_index": i,
            "role": roles[i % len(roles)],
            "source_hash": r["hash"],
            "consensus_score": r["score"],
            "genome": r["genome"],
        })
    spec = {
        "assembly": "MultiNeuraxon2 (chc6 composition of recorded winners)",
        "note": "Realize in the real codebase via NEURAXON_ARCH=<spec> for a "
                "full-fidelity 600-tick confirmation. Not run on-node.",
        "module_count": len(modules),
        "modules": modules,
    }
    return spec


def propose_curriculum_tightening(front):
    """Propose tighter healthy-bands for the next epoch from where the current
    winners landed. OFFLINE proposal only — to take effect, the OUTPUT must be
    published deterministically to the nodes (e.g. folded into the consensus
    digest) so the next epoch's scored sim stays reproducible on every node."""
    if not front:
        return None
    keys = list(NxonScore.TARGET_BANDS.keys())
    acc = {k: [] for k in keys}
    for (r, obj, m) in front:
        for k in keys:
            if k in m:
                acc[k].append(m[k])
    proposal = {}
    for k in keys:
        lo, hi = NxonScore.TARGET_BANDS[k]
        if acc[k]:
            vals = sorted(acc[k])
            med = vals[len(vals) // 2]
            # Tighten 20% toward the median of where winners actually landed.
            nlo = lo + 0.2 * (min(med, (lo + hi) / 2) - lo)
            nhi = hi - 0.2 * (hi - max(med, (lo + hi) / 2))
            if nlo < nhi:
                proposal[k] = [round(nlo, 4), round(nhi, 4)]
            else:
                proposal[k] = [lo, hi]
        else:
            proposal[k] = [lo, hi]
    return proposal


def main():
    if len(sys.argv) < 2:
        print("usage: python3 NxonOverseerOffline.py <system_file.json>")
        sys.exit(1)
    path = sys.argv[1]
    data = load_system_file(path)
    records = data["solutions"]

    print("=" * 74)
    print("NxonAnt 1.02 (consensus edition) — OFFLINE OVERSEER (curation)")
    print("=" * 74)
    print("System file:        {}".format(path))
    print("Recorded solutions: {}".format(len(records)))
    print("Age clock:          {} productive ticks".format(data.get("age_clock")))
    print("Root:               {}".format((data.get("root") or "")[:12]))
    print("(Nothing here runs on the node — this is downstream of consensus.)")

    # 1) Verify our offline scoring agrees with the on-node integer score.
    print("\n[1] Offline re-verification (our score function == the node's):")
    mism = 0
    checked = 0
    for r in records:
        if r["submitter"] == "ROOT":
            continue
        m = rescore_metrics(r["genome"], seed=_seed_for(r))
        if m is None:
            continue
        checked += 1
        s = NxonScore.consensus_score_int(m)
        if s != r["score"]:
            mism += 1
    print("    checked {} solutions; integer-score mismatches: {}".format(checked, mism))
    print("    (0 mismatches == offline tool and nodes share the same public law)")

    # 2) Multi-objective Pareto re-ranking (offline-only selection).
    print("\n[2] Pareto front (consensus score / bands satisfied / parsimony):")
    front = pareto_front(records)
    print("    {:<14} {:>12} {:>10} {:>8}".format("hash", "score", "bands", "hidden"))
    for (r, obj, m) in front[:8]:
        print("    {:<14} {:>12} {:>9.0%} {:>8}".format(
            r["hash"][:12], r["score"], obj[1], -obj[2]))

    # 3) MultiNeuraxon2 assembly spec from the winners.
    print("\n[3] MultiNeuraxon2 assembly (compose winners into a larger brain):")
    spec = assemble_multineuraxon2(front)
    out_spec = path.replace(".json", "") + "_assembly.json"
    with open(out_spec, "w") as f:
        json.dump(spec, f, indent=2)
    print("    composed {} modules -> {}".format(spec["module_count"], out_spec))
    print("    realize full-fidelity via NEURAXON_ARCH=<spec> in the real codebase")

    # 4) Curriculum tightening proposal for next epoch.
    print("\n[4] Curriculum tightening proposal (publish deterministically to nodes):")
    prop = propose_curriculum_tightening(front)
    if prop:
        shown = list(prop.items())[:4]
        for k, band in shown:
            print("    {:<28} -> {}".format(k, band))
        out_prop = path.replace(".json", "") + "_next_bands.json"
        with open(out_prop, "w") as f:
            json.dump(prop, f, indent=2)
        print("    full proposal -> {}".format(out_prop))
    print()
    print("Done. The node side stays minimal (verify + score + rules + record);")
    print("all of the above is offline curation of the agreed population.")


def _seed_for(record):
    """Reproduce the eval seed a node used for this record, from public fields,
    so offline re-scoring matches the on-node integer score exactly."""
    import NxonGenome as G
    meta = record["genome"].get("_meta", {})
    pubkey = meta.get("pubkey")
    nonce = meta.get("nonce")
    parent = record["parent"]
    if pubkey is not None and nonce is not None and parent is not None:
        return G.derive_eval_seed(pubkey, nonce, parent)
    return 99


if __name__ == "__main__":
    main()
