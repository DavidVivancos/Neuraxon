"""Regression tests: running the verify walks concurrently must not change
anything a node commits.

process_tick judges every submission against the same pre-tick snapshot and
defers commits to a second pass, so the per-submission walks are mutually
independent and mining_walk is pure. That is what makes them safe to run in
worker processes. These tests pin that claim: a node with workers must produce
the identical results tuple and the identical registry digest as a serial one,
because parallelism here is a speed decision and nothing else.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import NxonGenome as G        # noqa: E402
import NxonNode as ND         # noqa: E402
import NxonTrit as TR         # noqa: E402

WALK_STEPS = 40
SALT = "salted_spectrum_digest_PUBLIC_v1"


def _epoch():
    # phase lengths must sum to ticks: 8 + 12 + 6 + 6 == 32
    return G.build_epoch(SALT, N=24, ticks=32,
                         warmup=8, driven=12, silence=6, transfer=6)


def _node(epoch, verify_workers=1):
    node = ND.Node("node_test", epoch, WALK_STEPS, 1.0,
                   verify_workers=verify_workers)
    node.seed_root(G.root_lut(epoch))
    return node


def _submissions(node, count=6):
    """A mixed batch: some that improve on the root, plus a bad parent ref.

    The unknown parent is deliberate -- it exercises the pre-check pass, which
    short-circuits before any walk is queued.
    """
    parent_ref = node.registry.root_hash
    parent = node.registry.solutions[parent_ref]
    subs = []
    for i in range(64):
        if len(subs) >= count:
            break
        nonce = G.make_nonce("par{}".format(i), 2, WALK_STEPS // 4)
        _, score, _, _ = TR.mining_walk(
            parent["lut"], "pk_test", nonce, node.epoch, WALK_STEPS)
        if score > parent["score"]:
            subs.append({"pubkey": "pk_test", "nonce": nonce,
                         "parentRef": parent_ref})
    if not subs:
        pytest.skip("no improving nonce found")
    subs.append({"pubkey": "pk_bad", "nonce": G.make_nonce("bad", 1, 1),
                 "parentRef": "0" * 16})
    return subs


def _run(epoch, subs, workers, order=None):
    node = _node(epoch, verify_workers=workers)
    try:
        results, committed = node.process_tick(subs, tick=1, order=order)
    finally:
        node.close()
    # Drop the LUTs: compare the parts that consensus actually turns on.
    trimmed = [(s["pubkey"], s["nonce"], ok, reason, score)
               for s, ok, reason, score in results]
    return trimmed, committed, node.registry.digest()


def test_parallel_matches_serial():
    epoch = _epoch()
    node = _node(epoch)
    subs = _submissions(node)
    node.close()

    serial = _run(epoch, subs, workers=1)
    parallel = _run(epoch, subs, workers=4)

    assert parallel == serial, (
        "workers changed the outcome:\n  serial   {}\n  parallel {}".format(
            serial, parallel))


def test_parallel_matches_serial_under_shuffled_order():
    """Order-independence has to survive parallelism, not just precede it."""
    epoch = _epoch()
    node = _node(epoch)
    subs = _submissions(node)
    node.close()

    order = list(reversed(range(len(subs))))
    serial = _run(epoch, subs, workers=1, order=order)
    parallel = _run(epoch, subs, workers=4, order=order)

    assert parallel == serial

    # And the committed state must match the unshuffled run's digest too.
    assert parallel[2] == _run(epoch, subs, workers=4)[2]


def test_auto_workers_matches_serial():
    """verify_workers=0 means one per core; still the same answer."""
    epoch = _epoch()
    node = _node(epoch)
    subs = _submissions(node)
    node.close()

    assert _run(epoch, subs, workers=0) == _run(epoch, subs, workers=1)


def test_walk_and_hash_is_pure():
    """The unit the pool distributes has to be a pure function."""
    epoch = _epoch()
    root = G.root_lut(epoch)
    job = (root, "pk_test", G.make_nonce("pure", 2, 10))

    a = TR.walk_and_hash(job, epoch, WALK_STEPS)
    b = TR.walk_and_hash(job, epoch, WALK_STEPS)

    assert a[1] == b[1] and a[2] == b[2]


def test_falls_back_to_serial_when_no_pool(monkeypatch):
    """A pool that cannot start must not change the answer, only the speed."""
    epoch = _epoch()
    node = _node(epoch)
    subs = _submissions(node)
    node.close()

    expected = _run(epoch, subs, workers=1)

    broken = _node(epoch, verify_workers=4)
    monkeypatch.setattr(broken, "_walk_pool", lambda workers: None)
    try:
        results, committed = broken.process_tick(subs, tick=1)
    finally:
        broken.close()
    trimmed = [(s["pubkey"], s["nonce"], ok, reason, score)
               for s, ok, reason, score in results]

    assert (trimmed, committed, broken.registry.digest()) == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
