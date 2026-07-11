# Neuraxon Ant Colony 1.03 internal version 10
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonNode.py — the ON-NODE consensus verifier (v1.03, integer trit-LUT model).

Same node-consensus role as 1.02-consensus, now over the integer trit CA:
every node independently re-runs the miner's anti-attractor WALK, recomputes the
integer score, applies the validity rules, and writes an identical system file.
No secret. No single Overseer. Because the sim is pure integer, cross-node
agreement is exact with no FP caveat.

Per submission {pubkey, nonce, parentRef} (no score in the message):
  1. look up the parent solution (must exist and be live);
  2. re-run mining_walk(parent_lut, pubkey, nonce, epoch)  -> best child LUT;
  3. the score is the walk's best integer score (recomputed here, not trusted);
  4. apply the validity rules on that integer scalar vs the pre-tick snapshot.

Validity rules (identical to the recorded spec, on the integer scalar):
  R1 strict improvement      child score > parent score
  R2 earlier-tick sibling     child score >= max score of siblings (same parent)
     floor                     accepted at a strictly earlier tick
  R3 same-tick non-competition children at the same tick judged vs the same
                              pre-tick snapshot -> order-independent -> agreement
  R4 independent re-verify     the score is recomputed by every node (this file)
  R5 parent age                parent younger than PARENT_AGE_LIMIT productive ticks
The productive-tick age clock advances once per tick that accepts >= 1 solution.
"""

import os
import json
import time
import random
import hashlib
import argparse
from collections import defaultdict

import NxonGenome as G
import NxonScore
import NxonTrit as TR


PARENT_AGE_LIMIT = 676
WALK_STEPS = 120                 # network-fixed anti-attractor walk length
ANT_BUDGET_S = 1.0


# =============================================================================
# REGISTRY — recorded population (identical on every node)
# =============================================================================

class Registry:
    def __init__(self, epoch):
        self.epoch = epoch
        self.solutions = {}                  # hash -> record (incl. in-memory LUT)
        self.children = defaultdict(list)
        self.age_clock = 0
        self.root_hash = None

    def seed_root(self, lut):
        h = G.hash_lut(lut, self.epoch)
        self.solutions[h] = {
            "hash": h, "parent": None, "score": 0,
            "accept_tick": 0, "accept_age": 0, "submitter": "ROOT",
            "pubkey": "ROOT", "nonce": "ROOT", "lut": lut,
        }
        self.root_hash = h
        return h

    def is_live(self, parent_hash):
        rec = self.solutions.get(parent_hash)
        if rec is None:
            return False
        return (self.age_clock - rec["accept_age"]) < PARENT_AGE_LIMIT

    def earlier_tick_sibling_floor(self, parent_hash, tick):
        floor = None
        for ch in self.children.get(parent_hash, ()):
            rec = self.solutions[ch]
            if rec["accept_tick"] < tick:
                floor = rec["score"] if floor is None else max(floor, rec["score"])
        return floor

    def best(self):
        if not self.solutions:
            return None
        return max(self.solutions.values(), key=lambda r: r["score"])

    def live_solutions(self):
        return [r for h, r in self.solutions.items() if self.is_live(h)]

    def digest(self):
        items = sorted(
            (h, r["parent"] or "", r["score"], r["accept_tick"], r["accept_age"])
            for h, r in self.solutions.items())
        return hashlib.sha256(repr(items).encode()).hexdigest()[:16]

    def write_system_file(self, path):
        """Deterministic, compact system file. Stores the DERIVATION
        (pubkey, nonce, parent) + hash + score, NOT the full LUT — any node
        rebuilds every LUT by replaying the walk from ROOT. Byte-identical on
        every node."""
        rows = []
        for h in sorted(self.solutions):
            r = self.solutions[h]
            rows.append({"hash": h, "parent": r["parent"], "score": r["score"],
                         "accept_tick": r["accept_tick"], "accept_age": r["accept_age"],
                         "submitter": r["submitter"],
                         "pubkey": r["pubkey"], "nonce": r["nonce"]})
        payload = {
            "version": "1.03-trit-consensus",
            "salted_digest": self.epoch["salted_digest"],
            "N": self.epoch["N"], "ticks": self.epoch["ticks"],
            "walk_steps": WALK_STEPS,
            "age_clock": self.age_clock, "root": self.root_hash,
            "solutions": rows,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path) if os.path.exists(path) else os.rename(tmp, path)


# =============================================================================
# NODE — independent deterministic verifier
# =============================================================================

class Node:
    def __init__(self, node_id, epoch, walk_steps=WALK_STEPS, ant_budget=ANT_BUDGET_S):
        self.id = node_id
        self.epoch = epoch
        self.registry = Registry(epoch)
        self.walk_steps = walk_steps
        self.ant_budget = ant_budget

    def seed_root(self, lut):
        return self.registry.seed_root(lut)

    def verify(self, submission, tick):
        """Re-run the miner's walk and judge vs the pre-tick snapshot.

        Returns (accepted, reason, score, child_lut, child_hash)."""
        reg = self.registry
        parent_ref = submission["parentRef"]
        parent = reg.solutions.get(parent_ref)
        if parent is None:
            return (False, "unknown_parent", None, None, None)
        if not reg.is_live(parent_ref):                          # R5
            return (False, "parent_too_old", None, None, None)

        # 2) re-run the identical anti-attractor walk from the parent LUT.
        best_lut, best_score, _, _ = TR.mining_walk(
            parent["lut"], submission["pubkey"], submission["nonce"],
            self.epoch, self.walk_steps,
            deadline=time.time() + self.ant_budget * 3)
        child_hash = G.hash_lut(best_lut, self.epoch)

        # 4) validity rules on the integer scalar.
        if not (best_score > parent["score"]):                  # R1
            return (False, "not_improvement", best_score, best_lut, child_hash)
        floor = reg.earlier_tick_sibling_floor(parent_ref, tick)  # R2 / R3
        if floor is not None and best_score < floor:
            return (False, "below_sibling_floor", best_score, best_lut, child_hash)
        if child_hash in reg.solutions:
            return (False, "duplicate", best_score, best_lut, child_hash)
        return (True, "ok", best_score, best_lut, child_hash)

    def process_tick(self, submissions, tick, order=None):
        """Judge all submissions vs the SAME pre-tick snapshot, then commit the
        accepted set (order-independent -> every node agrees)."""
        reg = self.registry
        idx = list(range(len(submissions))) if order is None else order
        accepted = []
        results = []
        for i in idx:
            sub = submissions[i]
            ok, reason, score, child_lut, child_hash = self.verify(sub, tick)
            results.append((sub, ok, reason, score))
            if ok:
                accepted.append((sub, score, child_lut, child_hash))
        committed = 0
        for sub, score, child_lut, child_hash in accepted:
            if child_hash in reg.solutions:
                continue
            reg.solutions[child_hash] = {
                "hash": child_hash, "parent": sub["parentRef"], "score": score,
                "accept_tick": tick, "accept_age": reg.age_clock,
                "submitter": sub["pubkey"], "pubkey": sub["pubkey"],
                "nonce": sub["nonce"], "lut": child_lut,
            }
            reg.children[sub["parentRef"]].append(child_hash)
            committed += 1
        if committed > 0:
            reg.age_clock += 1
        return results, committed


# =============================================================================
# DEPOSIT LEDGER — anti-spam
# =============================================================================

class DepositLedger:
    def __init__(self, deposit=1.0, start_balance=1000.0):
        self.deposit = deposit
        self.balance = defaultdict(lambda: start_balance)
        self.forfeited = defaultdict(float)
        self.submitted = defaultdict(int)
        self.accepted = defaultdict(int)

    def on_submit(self, pk):
        self.balance[pk] -= self.deposit
        self.submitted[pk] += 1

    def on_result(self, pk, accepted):
        if accepted:
            self.balance[pk] += self.deposit
            self.accepted[pk] += 1
        else:
            self.forfeited[pk] += self.deposit


# =============================================================================
# MULTI-NODE NETWORK SIMULATION
# =============================================================================

def run_network(num_nodes=3, num_miners=6, ticks=15, ant_budget=1.0,
                walk_steps=WALK_STEPS, seed=42, N=48, sim_ticks=128,
                output_dir="nxon103_out",
                salted_digest="salted_spectrum_digest_PUBLIC_v1"):
    import Miner_nxon as MN
    os.makedirs(output_dir, exist_ok=True)

    epoch = G.build_epoch(salted_digest, N=N, ticks=sim_ticks)

    print("=" * 76)
    print("NxonAnt 1.03 — INTEGER trit-LUT Neuraxon, NODE-CONSENSUS NETWORK")
    print("=" * 76)
    print("Pure-integer trinary CA (no floats) -> exact cross-node determinism.")
    print("Every node re-runs the anti-attractor WALK, recomputes the integer")
    print("score, applies the validity rules, and must agree. No secret.")
    print("Nodes: {}  Miners: {}  Ticks: {}  Walk: {} steps  Sim: N={} ticks={}".format(
        num_nodes, num_miners, ticks, walk_steps, N, sim_ticks))
    print()

    root = G.root_lut(epoch)
    nodes = [Node("node_{:02d}".format(i), epoch, walk_steps, ant_budget)
             for i in range(num_nodes)]
    for nd in nodes:
        nd.seed_root(root)
    print("ROOT LUT from salted digest -> hash {} (score 0)".format(
        nodes[0].registry.root_hash[:12]))

    miners = []
    for i in range(num_miners):
        mode = "spam" if i == 0 else "honest"
        miners.append(MN.Miner("miner_{:02d}".format(i), epoch, mode=mode,
                               walk_steps=walk_steps, rng_seed=seed + i,
                               ant_budget=ant_budget))
    deposits = DepositLedger()
    rng = random.Random(seed)
    agree_fail = 0

    for tick in range(1, ticks + 1):
        shared = nodes[0].registry
        submissions = []
        for m in miners:
            sub = m.make_submission(shared, tick)
            if sub is not None:
                submissions.append(sub)
                deposits.on_submit(sub["pubkey"])
        if not submissions:
            continue

        per_node_accept = []
        for nd in nodes:
            order = list(range(len(submissions)))
            rng.shuffle(order)
            results, _ = nd.process_tick(submissions, tick, order=order)
            per_node_accept.append({s["pubkey"] + ":" + str(s["nonce"])
                                    for s, ok, r, sc in results if ok})

        agreed = per_node_accept[0]
        for sub in submissions:
            key = sub["pubkey"] + ":" + str(sub["nonce"])
            deposits.on_result(sub["pubkey"], key in agreed)

        digests = {nd.registry.digest() for nd in nodes}
        accept_match = all(s == per_node_accept[0] for s in per_node_accept)
        ok = (len(digests) == 1) and accept_match
        if not ok:
            agree_fail += 1
        best = nodes[0].registry.best()
        print("tick {:3d}: subs={:2d} accepted={:2d} | all {} nodes agree: {} "
              "| best_score={} n={}".format(
                  tick, len(submissions), len(agreed), num_nodes,
                  "YES" if ok else "NO <-- DIVERGENCE",
                  best["score"], len(nodes[0].registry.solutions)))

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
    print("-" * 76)
    print("CONSENSUS RESULT")
    print("-" * 76)
    print("Ticks with full agreement: {}/{}".format(ticks - agree_fail, ticks))
    print("System files byte-identical across nodes: {}  ({})".format(
        identical, hashes[0] if identical else hashes))
    print("Recorded solutions: {}".format(len(nodes[0].registry.solutions)))
    best = nodes[0].registry.best()
    if best and best["submitter"] != "ROOT":
        bm = TR.run_sim(best["lut"], epoch)
        print("Best recorded: score={} by {} | bands satisfied {:.0%}".format(
            best["score"], best["submitter"], NxonScore.band_satisfaction(bm)))

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
    print("System files -> offline tool:  python3 NxonOverseerOffline.py {}".format(
        paths[0]))
    return nodes[0].registry


def main():
    p = argparse.ArgumentParser(
        description="NxonAnt 1.03 — integer trit-LUT node-consensus network")
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--miners", type=int, default=6)
    p.add_argument("--ticks", type=int, default=15)
    p.add_argument("--walk-steps", type=int, default=WALK_STEPS)
    p.add_argument("--ant-budget", type=float, default=1.0)
    p.add_argument("--neurons", type=int, default=48)
    p.add_argument("--sim-ticks", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="nxon103_out")
    args = p.parse_args()
    run_network(num_nodes=args.nodes, num_miners=args.miners, ticks=args.ticks,
                walk_steps=args.walk_steps, ant_budget=args.ant_budget,
                seed=args.seed, N=args.neurons, sim_ticks=args.sim_ticks,
                output_dir=args.output_dir)


if __name__ == "__main__":
    main()
