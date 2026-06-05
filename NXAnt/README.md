# NxonAnt 1.02 — Consensus Edition (Qubic node-consensus model)

Match the **node-consensus** trust model in AntColony spec: every node independently re-runs the ~1-second simulation,
recomputes the score, applies the validity rules, and they all must agree.
**There is no secret and no single Overseer service.** Security comes from
redundant re-execution plus the submission deposit using the Qubic model.

## The on-node / offline split

**ON-NODE (consensus, runs on every machine, all public):**
- the ~1-second chc6 simulation (`NxonAnt.py`) — re-run to re-verify;
- the deterministic mutation `mutate_genome_k12` and eval-seed derivation, both
  from public `K12(pubkey‖nonce‖parentRef)` material (`NxonGenome.py`);
- the **public integer score** `consensus_score_int` and the healthy-bands +
  weights it uses (`NxonScore.py`);
- the validity rules and the registry / system file (`NxonNode.py`).

**OFFLINE (tool, downstream of consensus — `NxonOverseerOffline.py`):**
- multi-objective / Pareto re-ranking among accepted solutions;
- MultiNeuraxon2 brain assembly (compose recorded winners into a larger brain);
- curriculum tightening proposal for the next epoch;
- analytics / reputation.

The same `NxonScore` module is imported by the nodes and by the offline tool, so
the offline re-check reproduces the on-node integer score exactly (the demo
reports 0 mismatches).

## Files

| File | Side | Role |
|---|---|---|
| `NxonAnt.py` | on-node | chc6 ant: 1-second sim, returns RAW metrics. Unchanged runtime. |
| `NxonGenome.py` | on-node | public genome template, search space, `K12` derivations, deterministic mutation, ROOT seeding. |
| `NxonScore.py` | on-node | **public** healthy-bands + weights + **integer** consensus score. The shared scoring law. |
| `NxonNode.py` | on-node | the consensus verifier: re-run, re-score, validity rules, registry, system file, + a multi-node network demo. |
| `Miner_nxon.py` | client | miner: pick parent, `K12(pubkey‖nonce)` mutation, run ant, broadcast `{pubkey, nonce, parentRef}` + deposit. honest / spam modes. |
| `NxonOverseerOffline.py` | offline | curation: Pareto, assembly, curriculum tightening, analytics. |
| `NxonReview1.py` | offline | independent re-verification (same recompute the nodes do). |

## Run it

```bash
# A 3-node network with 6 miners. Every node re-verifies every submission and
# they must agree; the demo shuffles arrival order PER NODE to prove the
# outcome is order-independent, and checks the system files are byte-identical.
python3 NxonNode.py --nodes 3 --miners 6 --ticks 14 --ant-budget 1.0 --mutation 0.18

# Then run the offline curation on the agreed system file:
python3 NxonOverseerOffline.py nxon_consensus_out/system_file_node_00.json

# Independent re-verification of a recorded solution:
python3 NxonReview1.py <a genome json> --n-trials 20
```

The network run prints, per tick, `all N nodes agree: YES` and ends with
`System files byte-identical across nodes: True`. miner_00 is a spammer (fires
blind nonces, mostly rejected, forfeits the most deposit); the rest are honest
(run the ant locally and only submit genuine improvements).

## The validity rules (exactly your spec, on the integer scalar)

Judged against the **pre-tick registry snapshot**, so arrival order can't change
the outcome:

- **R1 strict improvement** — child score > parent score.
- **R2 earlier-tick sibling floor** — child score ≥ max score of siblings (same
  parent) accepted at a strictly earlier tick.
- **R3 same-tick non-competition** — same-tick children are judged against the
  same snapshot, so they don't compete and every node agrees regardless of order
  (proven in the demo by shuffling order per node → identical registry digest).
- **R4 independent re-verify** — the score is recomputed by every node.
- **R5 parent age** — parent younger than 676 productive ticks (frontier-keeping).

The productive-tick age clock advances once per tick that accepts ≥ 1 solution,
so quiet periods don't age the tree. ROOT score = 0 (the floor); the score is a
non-negative integer fitness where higher is better, so children climb above 0.

## Determinism — the one production requirement

The consensus scalar is an **integer** computed on quantized metrics, so the
scoring step is bit-identical across nodes. Full cross-node agreement **also**
requires the SIMULATION to be deterministic across hardware (compiler/libm/FMA
differences). This reference is deterministic within one Python build; for real
deployment the chc6 sim must run in fixed-point or a pinned FP mode, and the
metric-quantization grain (`METRIC_SCALE`) must be coarser than any residual
cross-node drift so a 1-ULP difference can never flip the integer score. K12 is
stubbed with SHA3-256 here; swap in real KangarooTwelve in deployment (call
sites don't change).

