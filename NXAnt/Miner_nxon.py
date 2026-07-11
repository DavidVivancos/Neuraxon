# Neuraxon Ant Colony 1.03 internal version 10
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
Miner_nxon.py — the MINER client (v1.03, integer trit-LUT model).

Per the recorded protocol the miner:
  (1) picks any LIVE parent solution from the agreed registry and fetches its LUT;
  (2) chooses a nonce = base_L_K (L lut lines mutated per walk step, K
      anti-attractor escape steps) and runs the mining WALK locally (the "ant");
  (5) broadcasts {pubkey, nonce, parentRef} with a deposit if the walk's best
      beats the parent.

The submission carries NO score — every node re-runs the identical walk (it is
deterministic from K12(pubkey||nonce)) and recomputes the integer score, so
there is nothing to fabricate. The only adversarial behaviour is spam, which the
deposit makes costly and the nodes reject.

Modes:
  - "honest": runs the walk locally, submits only genuine improvements.
  - "spam":   fires nonces without checking; mostly rejected, deposit drains.
"""

import time
import random

import NxonGenome as G
import NxonScore
import NxonTrit as TR


class Miner:
    def __init__(self, pubkey, epoch, mode="honest", walk_steps=120,
                 rng_seed=0, ant_budget=1.0):
        self.pubkey = pubkey
        self.epoch = epoch
        self.mode = mode
        self.walk_steps = walk_steps
        self.rng = random.Random(rng_seed)
        self.ant_budget = ant_budget
        self._ctr = 0

    def _new_nonce(self):
        self._ctr += 1
        base = self.rng.randrange(1 << 30)
        L = 1 + self.rng.randrange(4)          # 1..4 lut lines per step
        K = self.rng.randrange(self.walk_steps // 2 + 1)   # escape steps
        return G.make_nonce("{}b{}".format(base, self._ctr), L, K)

    def _pick_parent(self, registry):
        live = registry.live_solutions()
        if not live:
            return None
        live.sort(key=lambda r: -r["score"])
        k = max(1, len(live) // 2)
        top = live[:k]
        weights = [pow(0.7, i) for i in range(len(top))]
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        chosen = top[0]
        for rec, w in zip(top, weights):
            acc += w
            if r <= acc:
                chosen = rec
                break
        return chosen

    def make_submission(self, registry, tick):
        parent = self._pick_parent(registry)
        if parent is None:
            return None
        parent_ref = parent["hash"]

        if self.mode == "spam":
            nonce = self._new_nonce()
            return {"pubkey": self.pubkey, "nonce": nonce, "parentRef": parent_ref}

        # Honest: run the walk locally, submit only if it beats the parent.
        nonce = self._new_nonce()
        _, best_score, _, _ = TR.mining_walk(
            parent["lut"], self.pubkey, nonce, self.epoch, self.walk_steps,
            deadline=time.time() + self.ant_budget * 3)
        if best_score > parent["score"]:
            return {"pubkey": self.pubkey, "nonce": nonce, "parentRef": parent_ref}
        return None
