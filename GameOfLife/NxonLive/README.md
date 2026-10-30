# Multi Neuraxon Game of Life 5 — client / server

**Version mngol5-v1.67 / GoL Server V 1.087** 

A 24/7 server that runs one Neuraxon Game-of-Life world forever and
streams it to many web clients. Built on the proven **v184** neural
substrate (`neuraxon/` — neuron, network, components, multi-sphere,
g-factor), with **all visuals and diagnostic logging stripped** — just
the fundamentals: world dynamics, the neural substrate, the general
factor *g*, and an **all-time** ranking over live *and* dead NxErs.

```
mngol5/
├── world_config.json     # world + server settings (admin-editable)
├── run_server.sh         # launcher (HTTP dev / HTTPS prod)
├── requirements.txt      # aiohttp
├── neuraxon/             # reused v184 neural substrate (pygame-free)
├── logger.py             # tiny no-op logger shim (replaces the 3.3k-line
│                          #  research logger — "no unnecessary logging")
├── server/
│   ├── engine.py         # headless world + agents (+ g + ranking)
│   ├── game_server.py    # forever loop, snapshots, restart vs reboot
│   ├── persistence.py    # atomic crash-safe JSON snapshots (2 slots)
│   └── names.py          # unique A,B,…Z,AA,… allocator (persisted)
└── web/
    ├── webserver.py      # aiohttp: WS stream + REST + static
    ├── auth.py           # pw hashing, 3-fail/24h IP ban, sanitisation
    ├── sessions.py       # concurrent viewer + registered-user caps
    └── static/
        ├── client.html   # trimmed client (view / register / my-Nxer)
        └── admin.html    # admin console
```

## Run

```bash
cd mngol5
./run_server.sh                                   # HTTP  (dev)  :8443
./run_server.sh --cert fullchain.pem --key privkey.pem   # HTTPS (prod)
```

Open `https://host:8443/` (client) and `https://host:8443/admin`
(admin). On a fresh Ubuntu only one dependency is needed —
`pip install aiohttp` (the launcher does this automatically).

> Built-in TLS works with a cert/key pair. For production, terminating
> TLS at nginx/Caddy in front and running the app on plain HTTP behind
> it is also fine — the app honours `X-Forwarded-For` for IP bans.

## Science tools (v1.65 / v1.66 / v1.67)

```bash
python3 tools/selftest_arms.py            # 26 headless checks, in-process (~5-10 min)
python3 tools/selftest_arms.py --parallel # the same checks through worker processes
python3 tools/arm_report.py state         # per-arm answer from a run's history (stdlib only)
python3 tools/probe_brain.py --newborn state/best/brains/*.json state/best/elite/*.json
python3 tools/probe_brain.py --reflex both state/best/elite/*.json   # reflex arc vs bare brain
python3 tools/learn_test.py --conditions none reflex body             # closed-loop foraging test
python3 tools/test_gene_roundtrip.py      # every searched gene survives save -> load
# v1.67 — one offline closed-loop life per arm, server-faithful, with the six
# readouts (frozen share, synapses, tonic/phasic DA and the AGMP update size,
# input faithfulness, executed and one-step-lag chemotaxis, bites) and an
# open-loop probe (MI food direction -> motor step) with/without the projection
python3 tools/v167_life.py . - out.json --arm wired --genes chemo --steps 300
python3 tools/v167_life.py . - out.json --arm legacy --steps 300   # the V 1.086 brain
# v1.67 — headless pilot of the real engine (builder + workers): per-arm
# foraging, split by reflex genome, and the live fx_* brain-health readouts
python3 tools/pilot_v167.py --ticks 2500 --n 80 --parallel
```

**v1.67 brain fixes** (`brain_fix` block of `world_config.json`, all on by
default; `server/brain_fix.py` explains each): `exact_reload`,
`energy_rescue`, `firing_energy_cost`, `tau_slow`, `relay_inputs`,
`tonic_motor`, `clean_topdown`, `reward_phasic` (+ `reward_phasic_cap`),
`sm_link_gain_scale`, `sm_link_plasticity`, `sm_same_step`. Brains exported by a V 1.087
worker carry `mngol5_fix` (the settings they ran with); `probe_brain.py`
re-applies them. Arms added: `wired`, `wired_sham`, `wired_frozen`.
`script_max_share` and `mate_reserve` keep NAS and the brain-driven arms
alive at the population cap; `brain_detour` / `brain_food_stop` are the
body rules of the arms whose body executes the brain alone.

**Speed (v1.67).** Brains now keep their ~1,300 synapses (V 1.086 brains
lost ~97% within a few thousand ticks), so an aged brain's step costs ~2.2x
what it did. At 1,000 NxErs expect about half the V 1.086 TPS (~0.5-0.6
instead of 1.1 on the same machine); `max_nxers` ~600 keeps ~1 TPS.

The experiment arms (script / mixed / brain / frozen / sham / shaped /
reflex / reflex_sham / wired / wired_sham / wired_frozen) are set by `arms_enabled` and `arm_shares` in
`world_config.json`. Set `"arms_enabled": false` to run every NxEr on the
mixed controller (autopilot = `autopilot`). `reflex_mode` ("body" or
"brain") sets where the innate reflex acts, and `brain_move_steps` how many
cells one brain decision may move the body (0 = the whole interval).

## Performance & multi-core (important)

Every NxEr's brain is the CHC g-capable 6-sphere network (required for
g). It is ~2.4x heavier than a plain 3-sphere brain, so a single
Python thread cannot step many of them at the target TPS, and (because
of the GIL) a single-threaded engine would also starve the web server,
freezing clients.

The engine therefore steps all brains in **parallel across worker
processes** (`server/brain_pool.py`). Brains live only inside their
worker (sharded by id), are never serialised per tick, and the engine
blocks on a pipe while they run — which releases the GIL so the web
server stays responsive and clients keep streaming.

`engine_workers` in `world_config.json`:
- `0` (default) = auto = **CPU cores - 1** (recommended).
- `N` = exactly N worker processes.
- `1` = in-process fallback (no extra processes; for a 1-core box).

On a 4-core server, leave it `0` (→ 3 workers): the heavy brain phase
runs ~3x faster and the world view loads smoothly even at 100% sim CPU.
One bad tick or a dead worker can never kill the 24/7 loop.
