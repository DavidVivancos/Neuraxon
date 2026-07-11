# NxonAnt 1.03 — Integer trit-LUT Neuraxon (node-consensus)

v1.03 replaces the floating-point chc6 brain with a **pure-integer trinary
cellular automaton with per-neuron lookup tables**

State is a **trit** per neuron: `NEUTRAL=0, POS=1, NEG=2` (Neruaxon -1/0/+1).
Each non-input neuron has 3 source neighbours and a 27-entry lookup table (one
entry per combination of its 3 neighbour trits). The update is:

```
idx = state[src[n][0]] + 3*state[src[n][1]] + 9*state[src[n][2]]   # base-3, 0..26
next[n] = lut[n][idx]
```

## Files

| File | Side | Role |
|---|---|---|
| `NxonGenome.py` | on-node | trit-LUT genome, epoch structure from digest, `K12`, deterministic mutation-with-undo, ROOT (blank brain). |
| `NxonTrit.py` | on-node | the integer trit-CA `run_sim` + the `mining_walk`. The "ant". |
| `NxonScore.py` | on-node | public integer band-score over fixed-point metrics. |
| `NxonNode.py` | on-node | consensus verifier: re-run the walk, re-score, validity rules, registry, system file, + multi-node network demo. |
| `Miner_nxon.py` | client | miner: pick parent, `base_L_K` nonce, run the walk, broadcast `{pubkey,nonce,parentRef}` + deposit. honest / spam. |
| `NxonOverseerOffline.py` | offline | rebuild LUTs from derivations, Pareto re-rank, MultiNeuraxon2 assembly, curriculum tightening. |
| `NxonReview1.py` | offline | independent reproducibility check. |

## Run it

```bash
# 3-node network, 6 miners. Each node re-runs every miner's walk, recomputes the
# integer score, and must agree; order is shuffled per node to prove
# order-independence; system files are checked byte-identical.
python3 NxonNode.py --nodes 3 --miners 6 --ticks 16 --walk-steps 120 --neurons 48 --sim-ticks 128

# Offline curation on the agreed system file (rebuilds LUTs, 0 mismatches):
python3 NxonOverseerOffline.py nxon103_out/system_file_node_00.json

# Independent reproducibility check:
python3 NxonReview1.py nxon103_out/system_file_node_00.json

# Bench / self-test the integer sim + walk:
python3 NxonTrit.py 48 128 120
```

ROOT is a **blank all-NEUTRAL brain** (score 0, does nothing) — a true floor with
maximum headroom, so the colony visibly evolves it toward healthy dynamics
(0 → all-bands in a few ticks). The single integer trit sim runs in ~1 ms; a
120-step walk in ~0.15 s, well inside the 1-second budget.

## The validity rules (unchanged, on the integer scalar)

Judged against the pre-tick registry snapshot, so arrival order can't change the
outcome (proven by shuffling order per node → identical registry digest):

- **R1 strict improvement** — child score > parent score.
- **R2 earlier-tick sibling floor** — child ≥ max score of strictly-earlier-tick
  siblings of the same parent.
- **R3 same-tick non-competition** — same-tick children judged vs the same
  snapshot; order-independent → every node agrees.
- **R4 independent re-verify** — every node recomputes the score.
- **R5 parent age** — parent younger than 676 productive ticks.

The productive-tick age clock advances once per tick that accepts ≥ 1 solution.

## Reconciliation with the recorded spec (point-by-point)

- `sourceNeuron[N][3]`, `T[ticks][nInput]`, placement, phase map → `build_epoch`
  (from digest, on every node).
- `lut[N][27]` mined from `K12(pubkey‖nonce)` → `seed_lut`; child LUT = walk
  result stored per solution.
- RUN SIMULATION (inject held drive, base-3 LUT update, commit non-input,
  accumulate by phase) → `NxonTrit.run_sim`, exactly the loop in the spec.
- MINING (anti-attractor walk, `nonce[1]=L`, `nonce[2]=K`, allow-worse then
  better-only) → `NxonTrit.mining_walk`.
- `score = bandScore(metrics)` higher=healthier → `NxonScore.consensus_score_int`
  (integer).
- No training data / no output comparison → we measure neuron-state dynamics per
  phase, never compare to an expected output.


