# state/history — science logs

Empty in a fresh distribution. The server appends:
  timeseries.jsonl     ~1/min metric samples (M-metrics, g-structure, W_*, Mc_*,
                       and from v1.65 a per-arm `arms` block)
  obituaries.jsonl     one record per NxEr death (v1.65: exactly one; v1.64 runs
                       logged every collision death twice — deduplicate by id)
  lineage.jsonl        parent -> child links (v1.65: arm, parent_arms,
                       genetic_parent = the parent whose genome was cloned)
  nas_trials.jsonl     one record per NAS explorer, architecture + outcomes
                       (v1.65: id and name, so it joins the obituaries)
  brain_samples.jsonl  ~12 living brains per minute: M-metrics + brain markers
  provenance.jsonl     one record per boot (full config + science stamp)

v1.65 brain markers count FRESH DECISIONS (the brain decides every
brain_step_every ticks; replayed ticks are not counted). Behaviour fields in
brain_samples are lifetime running values: difference consecutive readings
before any repeatability analysis (tools/arm_report.py does this).

Never overwrite this folder when upgrading — it is the run's entire dataset.
