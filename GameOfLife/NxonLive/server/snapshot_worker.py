# --------------------------------------------------------------------
# Snapshot / broadcast worker subprocess.
#
# Lives on its own core. Receives a COMPACT raw snapshot tuple via Pipe
# from the engine thread (positions + colors + flags, no dict-building),
# builds the full WS broadcast payload (public_view dicts + foods +
# events + ranking), JSON-encodes it, and pushes the bytes back through
# a second Pipe.
#
# This frees the engine thread of the per-broadcast dict-build (~50 ms
# at 5000 alive) AND moves the JSON encoding off the aiohttp loop.
# Both pieces of work now run concurrently with engine.step() on a
# different core, since they live in a separate Python interpreter
# (no shared GIL with the parent).
#
# Failure mode: if the subprocess crashes, the GameServer's bridge
# detects the broken pipe and gracefully falls back to in-process
# snapshot building on the next tick. Worst case: a few seconds of
# stale broadcasts.
# --------------------------------------------------------------------

import json
import sys


def worker_main(raw_recv, bytes_send):
    """Entry point for the snapshot worker subprocess.

    Args:
        raw_recv:   multiprocessing.Connection — receives compact tuples
                    from the engine thread.
        bytes_send: multiprocessing.Connection — sends JSON-encoded
                    broadcast bytes back to the main process.
    """
    while True:
        try:
            msg = raw_recv.recv()
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            return
        if msg is None:                     # explicit shutdown signal
            return
        try:
            (tick, world_size, raw_nxers, raw_foods,
             alive_n, g_cache, events, ranking) = msg
        except (ValueError, TypeError) as e:
            print("[snapshot_worker] malformed input:", e, file=sys.stderr)
            continue
        try:
            nxers = []
            ap = nxers.append
            for r in raw_nxers:
                # r layout matches Engine.world_snapshot_raw() — keep in
                # sync: (id, name, x, y, alive, managed, color,
                #        singing, brain_building)
                d = {"id": r[0], "name": r[1],
                     "x": r[2], "y": r[3],
                     "alive": r[4],
                     "managed": r[5],
                     "c": r[6],
                     "s": r[7]}
                if r[8]:
                    d["b"] = 1
                ap(d)
            foods = [{"x": fx, "y": fy} for fx, fy in raw_foods]
            payload = {"type": "world", "data": {
                "tick": tick,
                "world": {"size": world_size},
                "nxers": nxers,
                "foods": foods,
                "alive": alive_n,
                "g": g_cache,
                "events": events,
                "ranking": ranking,
            }}
            data = json.dumps(payload).encode()
        except Exception as e:
            print("[snapshot_worker] build error:", e, file=sys.stderr)
            continue
        try:
            bytes_send.send_bytes(data)
        except (BrokenPipeError, EOFError):
            return
