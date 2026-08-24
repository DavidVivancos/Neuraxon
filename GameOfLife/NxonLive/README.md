# Multi Neuraxon Game of Life 5 — client / server

**Version mngol5-v1.58 / GoL Server V 1.078** 

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

