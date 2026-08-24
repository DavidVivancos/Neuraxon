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

Never overwrite this folder when upgrading — it holds your champions.
