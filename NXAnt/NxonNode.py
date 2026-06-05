# Neuraxon Ant Colony 1.02 internal version 09
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonNode.py — the ON-NODE consensus verifier (Qubic node-consensus model).

This is the consensus half of what older versions called the "Overseer". There
is NO single Overseer service and NO secret here. Every node runs THIS code,
independently re-verifies every submission, and writes an IDENTICAL system file.

What a node does, per submission (input-type-12 analogue):
  1. Look up the parent solution (must exist and be live).
  2. Re-apply the K12(pubkey||nonce) mutation to the parent  -> child genome.
  3. Re-run the ~1-second chc6 simulation on the public derived seed -> metrics.
  4. Recompute the INTEGER consensus score from the metrics (NxonScore).
  5. Apply the validity rules on that integer score.
The score is NOT in the submission — the node derives it from public material
(pubkey, nonce, parentRef), so there is nothing for a miner to fabricate.

Validity rules (judged on the integer scalar, against the PRE-TICK snapshot):
  R1 strict improvement     child score > parent score
  R2 earlier-tick sibling   child score >= max score of siblings (same parent)
     floor                   accepted at a STRICTLY EARLIER tick
  R3 same-tick siblings      children at the SAME tick are judged vs the same
     do not compete          pre-tick snapshot, so order of arrival is irrelevant
                             -> every node agrees regardless of arrival order
  R4 independent re-verify   the score is recomputed by every node (this file)
  R5 parent age              parent younger than PARENT_AGE_LIMIT productive
                             ticks (keeps exploration near the frontier)

The productive-tick age clock advances once per tick that accepts >= 1 solution,
so quiet periods do not age the tree.

Accepted solutions are appended to the registry and written to the system file;
because every node judges identically, every node's system file is byte-identical.
"""

import os
import sys
import json
import time
import copy
import random
import hashlib
import argparse
from collections import defaultdict

import NxonAnt
import NxonScore
import NxonGenome as G


PARENT_AGE_LIMIT = 676      # parent must be younger than this many productive ticks
SAME_TICK_WINDOW = 676      # same-tick counter window (documentation constant)
MUT_STRENGTH = 0.16         # public mutation strength (on every node)
ANT_BUDGET_S = 1.0          # non-negotiable 1-second ant budget


# =============================================================================
# REGISTRY — the recorded population (identical on every node)
# =============================================================================

class Registry:
    def __init__(self):
        self.solutions = {}                  # hash -> record
        self.children = defaultdict(list)    # parent_hash -> [child_hash]
        self.age_clock = 0                   # productive-tick clock (for R5)
        self.root_hash = None

    def seed_root(self, genome):
        h = NxonAnt.hash_genome(genome)
        self.solutions[h] = {
            "hash": h, "parent": None, "score": 0,
            "accept_tick": 0, "accept_age": 0,
            "submitter": "ROOT", "genome": genome,
        }
        self.root_hash = h
        return h

    def is_live(self, parent_hash):
        rec = self.solutions.get(parent_hash)
        if rec is None:
            return False
        return (self.age_clock - rec["accept_age"]) < PARENT_AGE_LIMIT

    def earlier_tick_sibling_floor(self, parent_hash, tick):
        """Max score among siblings under this parent accepted at a STRICTLY
        earlier tick (R2). Same-tick siblings are excluded (R3)."""
        floor = None
        for ch in self.children.get(parent_hash, ()):
            rec = self.solutions[ch]
            if rec["accept_tick"] < tick:
                floor = rec["score"] if floor is None else max(floor, rec["score"])
        return floor

    def digest(self):
        """Deterministic fingerprint of the whole registry — two nodes in
        consensus must produce the identical digest."""
        items = sorted(
            (h, r["parent"] or "", r["score"], r["accept_tick"], r["accept_age"])
            for h, r in self.solutions.items())
        return hashlib.sha256(repr(items).encode()).hexdigest()[:16]

    def best(self):
        if not self.solutions:
            return None
        return max(self.solutions.values(), key=lambda r: r["score"])

    def live_solutions(self):
        return [r for h, r in self.solutions.items() if self.is_live(h)]

    def write_system_file(self, path):
        """Write the recorded population to a system file. Deterministic + sorted
        so every node's file is byte-identical."""
        rows = []
        for h in sorted(self.solutions):
            r = self.solutions[h]
            rows.append({"hash": h, "parent": r["parent"], "score": r["score"],
                         "accept_tick": r["accept_tick"], "accept_age": r["accept_age"],
                         "submitter": r["submitter"], "genome": r["genome"]})
        payload = {"version": "1.02-consensus", "age_clock": self.age_clock,
                   "root": self.root_hash, "solutions": rows}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path) if os.path.exists(path) else os.rename(tmp, path)


# =============================================================================
# NODE — independent deterministic verifier
# =============================================================================

class Node:
    def __init__(self, node_id, mut_strength=MUT_STRENGTH, ant_budget=ANT_BUDGET_S):
        self.id = node_id
        self.registry = Registry()
        self.mut_strength = mut_strength
        self.ant_budget = ant_budget

    def seed_root(self, genome):
        return self.registry.seed_root(genome)

    def verify(self, submission, tick):
        """Re-verify ONE submission against the current pre-tick snapshot.

        submission = {"pubkey", "nonce", "parentRef"}  (no score — node derives it)
        Returns (accepted: bool, reason: str, score: int|None, child_genome|None).
        """
        reg = self.registry
        parent_ref = submission["parentRef"]
        parent = reg.solutions.get(parent_ref)
        if parent is None:
            return (False, "unknown_parent", None, None)
        if not reg.is_live(parent_ref):                       # R5
            return (False, "parent_too_old", None, None)

        # 2) re-apply the K12(pubkey||nonce) mutation (deterministic, public).
        child = G.mutate_genome_k12(parent["genome"], submission["pubkey"],
                                    submission["nonce"], self.mut_strength)
        # 3) re-run the 1s chc6 sim on the public derived seed.
        eval_seed = G.derive_eval_seed(submission["pubkey"], submission["nonce"], parent_ref)
        metrics, _, timed_out = NxonAnt._evaluate_architecture(
            child, eval_seed, time.time() + self.ant_budget * 3)
        if timed_out or metrics is None:
            return (False, "timeout", None, None)
        # 4) recompute the INTEGER consensus score.
        score = NxonScore.consensus_score_int(metrics)
        # 5) validity rules on the scalar (vs pre-tick snapshot).
        if not (score > parent["score"]):                     # R1
            return (False, "not_improvement", score, child)
        floor = reg.earlier_tick_sibling_floor(parent_ref, tick)   # R2 / R3
        if floor is not None and score < floor:
            return (False, "below_sibling_floor", score, child)
        child_hash = NxonAnt.hash_genome(child)
        if child_hash in reg.solutions:
            return (False, "duplicate", score, child)
        return (True, "ok", score, child)

    def process_tick(self, submissions, tick, order=None):
        """Judge all tick submissions against the SAME pre-tick snapshot, then
        commit the accepted set. Because judging never reads mid-tick mutations,
        the accept set is independent of arrival order (R3) — the property that
        makes every node agree.

        `order` lets the caller shuffle arrival order to demonstrate that the
        outcome is identical regardless.
        """
        reg = self.registry
        idx = list(range(len(submissions)))
        if order is not None:
            idx = order
        accepted = []
        results = []
        for i in idx:
            sub = submissions[i]
            ok, reason, score, child = self.verify(sub, tick)
            results.append((sub, ok, reason, score))
            if ok:
                accepted.append((sub, score, child))
        # Commit (append) the accepted set.
        committed = 0
        for sub, score, child in accepted:
            ch = NxonAnt.hash_genome(child)
            if ch in reg.solutions:           # guard identical same-tick genomes
                continue
            reg.solutions[ch] = {
                "hash": ch, "parent": sub["parentRef"], "score": score,
                "accept_tick": tick, "accept_age": reg.age_clock,
                "submitter": sub["pubkey"], "genome": child,
            }
            reg.children[sub["parentRef"]].append(ch)
            committed += 1
        if committed > 0:                     # productive tick -> advance age clock
            reg.age_clock += 1
        return results, committed


# =============================================================================
# DEPOSIT LEDGER — anti-spam (the submission deposit)
# =============================================================================

class DepositLedger:
    """Each submission attaches a deposit. Accepted -> refunded. Rejected ->
    forfeited (anti-spam). No slashing-of-secrets machinery is needed: in full
    re-verification there is nothing to fake, so the deposit only has to make
    junk submissions costly."""

    def __init__(self, deposit=1.0, start_balance=1000.0):
        self.deposit = deposit
        self.balance = defaultdict(lambda: start_balance)
        self.forfeited = defaultdict(float)
        self.submitted = defaultdict(int)
        self.accepted = defaultdict(int)

    def on_submit(self, pubkey):
        self.balance[pubkey] -= self.deposit
        self.submitted[pubkey] += 1

    def on_result(self, pubkey, accepted):
        if accepted:
            self.balance[pubkey] += self.deposit     # refunded
            self.accepted[pubkey] += 1
        else:
            self.forfeited[pubkey] += self.deposit    # forfeited


# =============================================================================
# MULTI-NODE NETWORK SIMULATION  (proves consensus without a secret)
# =============================================================================

def run_network(num_nodes=3, num_miners=6, ticks=15, ant_budget=1.0,
                mut_strength=MUT_STRENGTH, seed=42, output_dir="nxon_consensus_out",
                salted_digest="salted_spectrum_digest_PUBLIC_v1"):
    import Miner_nxon as MN
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 74)
    print("NxonAnt 1.02 (consensus edition) — NODE-CONSENSUS NETWORK")
    print("=" * 74)
    print("No secret. No single Overseer. Every node re-runs the sim, recomputes")
    print("the INTEGER score, applies the validity rules, and must agree.")
    print("Nodes: {}   Miners: {}   Ticks: {}   Ant budget: {:.1f}s".format(
        num_nodes, num_miners, ticks, ant_budget))
    print("Parent-age limit: {} productive ticks".format(PARENT_AGE_LIMIT))
    print()

    # Build nodes; seed the identical ROOT from the public salted digest.
    root = G.root_genome(salted_digest)
    nodes = [Node("node_{:02d}".format(i), mut_strength, ant_budget)
             for i in range(num_nodes)]
    for nd in nodes:
        nd.seed_root(root)
    print("ROOT seeded from salted spectrum digest -> hash {} (score 0)".format(
        nodes[0].registry.root_hash[:12]))

    # Miners. miner 0 is a SPAMMER (submits without self-checking); the rest are
    # honest (run the ant locally and only submit genuine improvements).
    miners = []
    for i in range(num_miners):
        mode = "spam" if i == 0 else "honest"
        miners.append(MN.Miner("miner_{:02d}".format(i), mode=mode,
                               mut_strength=mut_strength, rng_seed=seed + i,
                               ant_budget=ant_budget))
    deposits = DepositLedger()

    rng = random.Random(seed)
    agree_fail = 0

    for tick in range(1, ticks + 1):
        # Shared agreed view (all registries identical): use node 0's.
        shared = nodes[0].registry
        submissions = []
        for m in miners:
            sub = m.make_submission(shared, tick)
            if sub is not None:
                submissions.append(sub)
                deposits.on_submit(sub["pubkey"])

        if not submissions:
            continue

        # Each node verifies the SAME submissions in a DIFFERENT (shuffled)
        # order, to prove order-independence -> identical outcome.
        per_node_accepts = []
        for nd in nodes:
            order = list(range(len(submissions)))
            rng.shuffle(order)
            results, committed = nd.process_tick(submissions, tick, order=order)
            per_node_accepts.append({s["pubkey"] + ":" + str(s["nonce"])
                                     for s, ok, r, sc in results if ok})

        # Deposit accounting from node 0's view (all nodes agree on accept set).
        node0_results, _ = (lambda r: r)((None, None)) if False else (None, None)
        # Recompute node 0 accept reasons for deposit + logging (re-judge cheaply
        # is avoided; reuse the agreed set).
        agreed = per_node_accepts[0]
        for sub in submissions:
            key = sub["pubkey"] + ":" + str(sub["nonce"])
            deposits.on_result(sub["pubkey"], key in agreed)

        # CONSENSUS CHECK: every node's registry digest must match.
        digests = {nd.registry.digest() for nd in nodes}
        accept_sets_match = all(s == per_node_accepts[0] for s in per_node_accepts)
        ok = (len(digests) == 1) and accept_sets_match
        if not ok:
            agree_fail += 1
        best = nodes[0].registry.best()
        print("tick {:3d}: subs={:2d} accepted={:2d} | all {} nodes agree: {} "
              "| best_score={} (depth-ish n={})".format(
                  tick, len(submissions), len(agreed), num_nodes,
                  "YES" if ok else "NO  <-- DIVERGENCE",
                  best["score"], len(nodes[0].registry.solutions)))

    # Write each node's system file; confirm byte-identical.
    paths = []
    for nd in nodes:
        p = os.path.join(output_dir, "system_file_{}.json".format(nd.id))
        nd.registry.write_system_file(p)
        paths.append(p)
    hashes = []
    for p in paths:
        with open(p, "rb") as f:
            hashes.append(hashlib.sha256(f.read()).hexdigest()[:16])
    identical = len(set(hashes)) == 1

    print()
    print("-" * 74)
    print("CONSENSUS RESULT")
    print("-" * 74)
    print("Ticks with full agreement: {}/{}".format(ticks - agree_fail, ticks))
    print("System files byte-identical across nodes: {}  ({})".format(
        identical, hashes[0] if identical else hashes))
    print("Recorded solutions: {}".format(len(nodes[0].registry.solutions)))
    best = nodes[0].registry.best()
    if best and best["submitter"] != "ROOT":
        m = NxonAnt._evaluate_architecture(
            best["genome"], 99, time.time() + ant_budget * 3)[0]
        print("Best recorded: score={} by {} | bands satisfied {:.0%}".format(
            best["score"], best["submitter"], NxonScore.band_satisfaction(m)))

    print()
    print("DEPOSITS (anti-spam): spammer forfeits, honest miners are refunded")
    print("  {:<12} {:>9} {:>9} {:>10} {:>11}".format(
        "miner", "submit", "accepted", "balance", "forfeited"))
    for m in miners:
        pk = m.pubkey
        print("  {:<12} {:>9} {:>9} {:>10.1f} {:>11.1f}".format(
            pk, deposits.submitted[pk], deposits.accepted[pk],
            deposits.balance[pk], deposits.forfeited[pk]))

    print()
    print("System files: {}".format(", ".join(os.path.basename(p) for p in paths)))
    print("Hand these to the OFFLINE tool:  python3 NxonOverseerOffline.py {}".format(
        paths[0]))
    return nodes[0].registry


def main():
    p = argparse.ArgumentParser(
        description="NxonAnt 1.02 consensus edition — node-consensus network")
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--miners", type=int, default=6)
    p.add_argument("--ticks", type=int, default=15)
    p.add_argument("--ant-budget", type=float, default=1.0)
    p.add_argument("--mutation", type=float, default=MUT_STRENGTH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="nxon_consensus_out")
    args = p.parse_args()
    run_network(num_nodes=args.nodes, num_miners=args.miners, ticks=args.ticks,
                ant_budget=args.ant_budget, mut_strength=args.mutation,
                seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
