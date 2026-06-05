# Neuraxon Ant Colony 1.02 internal version 09
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
Miner_nxon.py — the MINER client (consensus edition).

A miner is the Qubic node's mining role. Per the recorded protocol it:
  (1) picks any LIVE parent solution from the agreed registry,
  (2) chooses a nonce and applies K12(pubkey||nonce) mutations to the parent ANN,
  (3) runs the ~1-second chc6 simulation locally (the "ant"),
  (5) broadcasts a submission {pubkey, nonce, parentRef} with a deposit attached.

Crucially the submission carries NO score. The score is recomputed by every node
from the public (pubkey, nonce, parentRef) material, so there is nothing for a
miner to fabricate — the secret-model "fabricator" attack is structurally
impossible here. The only adversarial behaviour left is spam (submitting junk
nonces), which the deposit makes costly and which the nodes simply reject.

Modes:
  - "honest": runs the ant locally and only submits when it finds a genuine
    improvement over the chosen parent (self-filtering — efficient).
  - "spam":   submits a fresh nonce every tick without checking, hoping to get
    lucky. Almost everything is rejected; its deposit drains.
"""

import time
import random

import NxonAnt
import NxonScore
import NxonGenome as G


class Miner:
    def __init__(self, pubkey, mode="honest", mut_strength=0.16,
                 rng_seed=0, ant_budget=1.0, local_tries=10):
        self.pubkey = pubkey
        self.mode = mode
        self.mut_strength = mut_strength
        self.rng = random.Random(rng_seed)
        self.ant_budget = ant_budget
        self.local_tries = local_tries
        self._nonce_counter = 0

    def _next_nonce(self):
        self._nonce_counter += 1
        # Nonce is arbitrary; mix in identity + counter + a random draw.
        return "{}-{}-{}".format(self._nonce_counter,
                                 self.rng.randrange(1 << 30),
                                 self.pubkey[-2:])

    def _pick_parent(self, registry):
        """Pick a live parent, biased toward higher-scoring solutions (exploit)
        with exploration over the top half of the live set."""
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
        """Produce a submission for this tick, or None if the miner declines."""
        parent = self._pick_parent(registry)
        if parent is None:
            return None
        parent_ref = parent["hash"]

        if self.mode == "spam":
            # No local check — just fire a nonce and hope. Mostly rejected.
            nonce = self._next_nonce()
            return {"pubkey": self.pubkey, "nonce": nonce, "parentRef": parent_ref}

        # Honest: try a few nonces locally, submit the first genuine improvement.
        best = None
        for _ in range(self.local_tries):
            nonce = self._next_nonce()
            child = G.mutate_genome_k12(parent["genome"], self.pubkey, nonce,
                                        self.mut_strength)
            eval_seed = G.derive_eval_seed(self.pubkey, nonce, parent_ref)
            metrics, _, timed_out = NxonAnt._evaluate_architecture(
                child, eval_seed, time.time() + self.ant_budget * 3)
            if timed_out or metrics is None:
                continue
            score = NxonScore.consensus_score_int(metrics)
            if score > parent["score"] and (best is None or score > best[1]):
                best = (nonce, score)
        if best is None:
            return None      # found no improvement this tick; stay quiet
        return {"pubkey": self.pubkey, "nonce": best[0], "parentRef": parent_ref}
