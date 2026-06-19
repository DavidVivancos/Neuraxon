#!/usr/bin/env bash
# Multi Neuraxon Game of Life 5 — launcher
# Server release: GoL Server V 1.053  (v189-compat substrate, NAS arch nas_best_t005)
# ===================================================================
# SPEED: the brain compute is pure Python and is the bottleneck. The
# single biggest speedup on Ubuntu — with ZERO code changes — is to
# run under PyPy (a JIT for Python). For this workload PyPy is
# typically 5-10x faster than CPython. This launcher uses pypy3
# automatically if it is installed, otherwise falls back to python3.
#
# numpy is NOT required: the substrate uses a tiny numpy surface and
# ships a pure-Python fallback (server/np_fallback.py) that activates
# automatically when a usable numpy is absent — so PyPy needs only
# aiohttp. (On CPython a real numpy, if present, is used unchanged.)
#
# Install PyPy on Ubuntu (recommended):
#     sudo apt update && sudo apt install -y pypy3 pypy3-dev
#     pypy3 -m ensurepip
#     pypy3 -m pip install --break-system-packages aiohttp
#
# Then just run ./run_server.sh as usual — it will pick pypy3 up.
# (To start a fresh world, delete the ./state directory first.)
# ===================================================================
set -e
cd "$(dirname "$0")"
export NEURAXON_HEADLESS=1

if command -v pypy3 >/dev/null 2>&1; then
    PY="pypy3"
    echo "[run_server] using PyPy ($(pypy3 --version 2>&1 | head -1)) — JIT enabled"
else
    PY="python3"
    echo "[run_server] using CPython. For ~5-10x more speed install PyPy:"
    echo "             sudo apt install -y pypy3 pypy3-dev && pypy3 -m ensurepip"
    echo "             pypy3 -m pip install --break-system-packages aiohttp"
fi

$PY -c "import aiohttp" 2>/dev/null || \
    $PY -m pip install --break-system-packages -q aiohttp || true

# HTTP (dev):     ./run_server.sh --port 8080
# HTTPS (prod):   ./run_server.sh --cert fullchain.pem --key privkey.pem

# Auto-bind to 127.0.0.1 when a domain is configured: that means a
# reverse proxy (nginx) is in front and the backend MUST NOT be open
# on the public interface. If canonical_host is empty (single-machine
# dev) we keep 0.0.0.0 so localhost access still works.
HOST_ARG=""
if [ -f world_config.json ]; then
    HAS_DOMAIN=$($PY -c "import json; c=json.load(open('world_config.json')); print('1' if c.get('canonical_host','').strip() else '0')" 2>/dev/null || echo 0)
    if [ "$HAS_DOMAIN" = "1" ]; then
        HOST_ARG="--host 127.0.0.1"
        echo "[run_server] canonical_host set → binding to 127.0.0.1 (put nginx in front; see install_nginx.sh)"
    fi
fi

exec $PY -m web.webserver --config world_config.json --state state $HOST_ARG "$@"
