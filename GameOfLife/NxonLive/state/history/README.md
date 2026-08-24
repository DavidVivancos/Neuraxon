# state/history — science logs

Empty in a fresh distribution. The server appends:
  timeseries.jsonl  ~1/min metric samples (M-metrics, g-structure, W_*, Mc_*)
  obituaries.jsonl  one record per NxEr death
  lineage.jsonl     parent -> child links
  nas_trials.jsonl  one record per NAS explorer, architecture + outcomes
  provenance.jsonl  one record per boot (full config + science stamp)

Never overwrite this folder when upgrading — it is the run's entire dataset.
