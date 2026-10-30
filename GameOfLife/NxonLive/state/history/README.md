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

v1.66 adds:
  - chemo_exec / chemo_exec_n / approach_frac: executed-step chemotaxis.
    It is the cosine between the step the body really took and the TRUE food
    direction, scored on the first step of each brain decision only. The
    intended-step `chemotaxis` is still logged; it is biased by blocked
    steps.
  - `reflex`: the innate reflex genes (rfx_xx, rfx_xy, rfx_yx, rfx_yy,
    rfx_rest, rfx_gain), in lineage and obituaries. Every NxEr carries them;
    only the reflex and reflex_sham arms express them.
  - the timeseries `arms` block: chemo_exec, out_rest, bites_per_1k and the
    reflex gene means.
  - provenance: nas_objective and the reflex settings.

Never overwrite this folder when upgrading — it is the run's entire dataset.

v1.67 adds:
  - brain_samples `fx_*` (brain_fix.measure, per living brain): fx_frozen_frac
    (active neurons at energy 0), fx_frozen_in / fx_frozen_mo (frozen sensory
    inputs / motor outputs), fx_DA_tonic / fx_DA_phasic (dopamine pools, mean of
    the spheres), fx_agmp (mean size of the AGMP update the next step would
    make), fx_kappa, fx_sm_link (1 = carries the innate sensorimotor
    projection), fx_link_src_ok (share of link sources on the neuron role they
    were built for; 1.0 with the exact loader), fx_in_faith / fx_in_follow (do
    fdx/fdy reach the input neurons, and do those neurons follow them).
  - chemo_lag / chemo_lag_n (per NxEr, obituaries and the `arms` block): the
    first executed step of a decision against the food direction sensed at
    the PREVIOUS decision (pathways through the association spheres answer an
    earlier direction; the wired arms' projection is same-step).
  - arms wired / wired_sham / wired_frozen (see world_config.json _v167_notes).
