# state/best — all-time champion archive

Empty in a fresh distribution: a new world has no records yet.

The server fills this automatically. For each all-time record it writes a small
record file `<metric>_<name>_<value>_t<tick>.json`, and stores that NxEr's full
brain ONCE under `brains/<name>_t<tick>.json`, which the record points to via
its `"brain"` field. One NxEr holding several records therefore produces several
small record files and a single brain blob — the small files are pointers by
design, not truncated brains.

`_index.json` holds the per-metric high-water marks. Do NOT copy one in from
another world: a stale index makes the server skip archiving until the new world
beats the old world's values.

`elite/` (v1.59+, rebuilt in v1.65, v1.66) holds three pools of LIVING brains, indexed in
`elite/INDEX.csv`:
- score slots (elite_brains - elite_brains/4): the most M-compliant brains, each
  re-read once after elite_reverify_ticks (`verify_*` columns);
- age slots (elite_brains/4): a hall of fame of the longest-lived brains. Older
  brains displace younger ones, and a living elder's blob is re-exported every
  time its age doubles, so the file holds the OLD brain.
- forage slots (v1.66, elite_brains/5): the best foragers driven by their OWN senses (brain,
  frozen and reflex arms), by food found per 1000 ticks of age, at age >= 5k
  ticks. Theft and gifts do not count. This is the pool to probe for function.
Each row records the NxEr's experiment `arm`, `motor_frac`, `lock_frac`,
`forage_rate` and, from v1.66, its reflex genes. The table reloads from
INDEX.csv after a restart.

Never overwrite this folder when upgrading — it holds your champions.
