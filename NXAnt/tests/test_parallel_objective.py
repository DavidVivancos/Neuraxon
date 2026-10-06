"""The scoring objective must survive the trip into a worker process.

OBJECTIVE_MODE, MARGIN_WEIGHT and the external scorer are module globals in
NxonScore, set at runtime by --objective in main(). A worker started with
"fork" inherits them. A worker started with "spawn" (default on Windows and
macOS) or "forkserver" (default on Linux from Python 3.14) re-imports the
module and gets the DEFAULT instead.

That makes the score depend on the OS and the Python version, which is exactly
the class of divergence #3 removed from the clock. Caught in review of #4:
with --objective banded, one submission scored 41,600 serially and 60,950 in
parallel.

test_parallel_verify.py did not catch it because every test there runs with the
default objective, where parent and worker happen to agree.
"""

import multiprocessing as mp
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import NxonGenome as G        # noqa: E402
import NxonNode as ND         # noqa: E402
import NxonScore              # noqa: E402

METHODS = [m for m in ("fork", "forkserver", "spawn")
           if m in mp.get_all_start_methods()]

WALK_STEPS = 40
SALT = "salted_spectrum_digest_PUBLIC_v1"


def _run(workers):
    epoch = G.build_epoch(SALT, N=24, ticks=32,
                          warmup=8, driven=12, silence=6, transfer=6)
    root = G.root_lut(epoch)
    root_hash = G.hash_lut(root, epoch)
    subs = [{"pubkey": "pk%d" % i,
             "nonce": G.make_nonce(i, 1, 0),
             "parentRef": root_hash}
            for i in range(6)]

    node = ND.Node("n", epoch, WALK_STEPS, 1.0, verify_workers=workers)
    node.seed_root(root)
    try:
        results, _ = node.process_tick(subs, tick=1)
    finally:
        node.close()
    trimmed = [(ok, reason, score) for _, ok, reason, score in results]
    return trimmed, node.registry.digest()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("objective", ["banded", "unbounded"])
def test_parallel_honours_objective(method, objective, monkeypatch):
    """Parallel must match serial under every start method and objective."""
    monkeypatch.setattr(NxonScore, "OBJECTIVE_MODE", objective)
    old = mp.get_start_method(allow_none=True)
    mp.set_start_method(method, force=True)
    try:
        parallel = _run(4)
        serial = _run(1)
        assert parallel == serial, (
            "objective %r did not reach the workers under %r start method:\n"
            "  serial   %s\n  parallel %s" % (objective, method, serial, parallel))
    finally:
        mp.set_start_method(old, force=True)


def test_pool_is_sized_from_config_not_first_batch():
    """A small first batch must not permanently shrink the pool.

    _walk_pool creates the pool once and reuses it, so passing
    min(workers, len(jobs)) would freeze a 4-worker node at 2 workers forever
    if its first parallel tick happened to carry 2 submissions.
    """
    epoch = G.build_epoch(SALT, N=24, ticks=32,
                          warmup=8, driven=12, silence=6, transfer=6)
    root = G.root_lut(epoch)
    root_hash = G.hash_lut(root, epoch)

    node = ND.Node("n", epoch, WALK_STEPS, 1.0, verify_workers=4)
    node.seed_root(root)
    try:
        small = [{"pubkey": "pk%d" % i,
                  "nonce": G.make_nonce(i, 1, 0),
                  "parentRef": root_hash}
                 for i in range(2)]
        node.process_tick(small, tick=1)
        assert node._pool is not None, "expected a pool after a 2-job batch"
        assert node._pool._max_workers == 4, (
            "pool froze at %d workers after a 2-job first batch; it should be "
            "sized from the configured count" % node._pool._max_workers)
    finally:
        node.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
