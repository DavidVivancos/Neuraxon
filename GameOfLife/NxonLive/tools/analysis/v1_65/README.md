# Analysis scripts behind report_v1.65.md

Standard-library Python unless noted. Run them from the `mngol5` folder, and pass the run's
`state` folder (or a file inside it) as the argument.

| script | what it computes | report section |
|---|---|---|
| a1_ledger.py `state/history/obituaries.jsonl` | energy ledger by arm and lifespan class; collisions by kind | §3 |
| a2_nas.py `state/history/nas_trials.jsonl` | NAS knob ↔ outcome rank correlations by arm | §13 |
| a3_chemo.py `…/obituaries.jsonl` | chemotaxis by collision band; bites by m_fit tercile | §5 |
| a4_cohort.py `…/obituaries.jsonl` | birth-cohort trends; explorers vs mated | §9 |
| a5_bs.py `state/history/brain_samples.jsonl` | brains by arm × age (synapses, lock, M1, M5) | §10 |
| a6_ts.py `state/history` | 12-window timeline | §11 |
| a7_elders.py `snapshot.prev.json obituaries.jsonl`; a8_founders.py `snapshot.prev.json brain_samples.jsonl` | the longest lives, surviving founders | §4 |
| a9_identity.py `state` | run identity and totals | §1 |
| reflex_test.py, rfx_lock.py | the in-brain reflex current, open loop (substrate; run from mngol5) | §14.3 |
| sweep_gain.py | sensory gain × coupling probe (substrate) | §14.2 |
| trace_signal.py | where a sensory difference propagates sphere by sphere (substrate) | §14.2 |
| geom.py | stale-target bias of the chemotaxis metric for a blind walker | §5 |
| forage_sim.py / forage_sim2.py / forage_sim3.py `15` | ideal vs random forager under 1–8 cells per decision | §14.5 |
| pilot66.py `<mngol5> <seed> <ticks> <out.json>`, pilot_sum.py, pilot_genes.py | small-world v1.66 pilot | §16.1 |
| a10_ramp.py `…/obituaries.jsonl 0.002` | metabolism paid by resting vs non-resting NxErs (idle ramp) | §3 |
| drain_diag.py `<mngol5> <ticks>` | per-arm metabolism and idle time in a small world | §18 |
| pilot_report.py `p11.json p23.json` | the §16.1 tables | §16.1 |
| mk_rfx.py | writes synthetic reflex brains for testing probe_brain.py | — |

The closed-loop learning and reflex tests are `tools/learn_test.py` (§14.1, §14.4).

These are the exact scripts used for the report, kept as they ran (some carry run-specific
constants, e.g. the end tick in a4_cohort.py). For new runs use `tools/arm_report.py`.
