"""Regression tests: the consensus verifier must never read the wall clock.

Node.verify() re-runs the miner's walk, and the score it returns *is* the value
every node has to agree on. If that walk can be truncated by a wall-clock
deadline, then two nodes running at different speeds -- heterogeneous hardware,
a GC pause, a noisy neighbour -- stop at different steps, return different
scores, and commit different system files.

These tests pin the invariant by simulating a node that is arbitrarily slow.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import NxonGenome as G        # noqa: E402
import NxonNode as ND         # noqa: E402
import NxonTrit as TR         # noqa: E402

WALK_STEPS = 60
SALT = "salted_spectrum_digest_PUBLIC_v1"


def _epoch():
    # phase lengths must sum to ticks: 8 + 12 + 6 + 6 == 32
    return G.build_epoch(SALT, N=24, ticks=32,
                         warmup=8, driven=12, silence=6, transfer=6)


def _node(epoch, ant_budget=1.0):
    node = ND.Node("node_test", epoch, WALK_STEPS, ant_budget)
    node.seed_root(G.root_lut(epoch))
    return node


def _submission(node):
    """A submission whose walk genuinely improves on the root."""
    parent_ref = node.registry.root_hash
    parent = node.registry.solutions[parent_ref]
    for i in range(64):
        nonce = G.make_nonce("regress{}".format(i), 2, WALK_STEPS // 4)
        _, score, _, _ = TR.mining_walk(
            parent["lut"], "pk_test", nonce, node.epoch, WALK_STEPS)
        if score > parent["score"]:
            return {"pubkey": "pk_test", "nonce": nonce, "parentRef": parent_ref}
    pytest.skip("no improving nonce found in 64 tries")


def _expired_clock():
    """A clock far enough ahead that any deadline built from it has passed.

    Stands in for a node so slow that a wall-clock budget runs out mid-walk.
    """
    return float("inf")


def test_verify_ignores_a_slow_clock(monkeypatch):
    epoch = _epoch()
    node = _node(epoch)
    sub = _submission(node)

    baseline = node.verify(sub, tick=1)
    assert baseline[0] is True, "fixture should produce an accepted submission"

    # Same node, same submission -- but every clock read now reports that the
    # budget is already blown. A correct verifier does not care.
    monkeypatch.setattr(time, "time", _expired_clock)
    stalled = node.verify(sub, tick=1)

    assert stalled == baseline, (
        "verify() changed its answer when the clock moved: "
        "score {} -> {}, hash {} -> {}".format(
            baseline[2], stalled[2], baseline[4], stalled[4]))


def test_verify_agrees_across_budgets():
    """ant_budget is a local knob; it must not reach the consensus value."""
    epoch = _epoch()
    reference = _node(epoch)
    sub = _submission(reference)

    results = []
    for budget in (1e-9, 0.001, 1.0, 60.0):
        node = _node(epoch, ant_budget=budget)
        ok, _, score, _, child_hash = node.verify(sub, tick=1)
        results.append((budget, ok, score, child_hash))

    scores = set(r[2] for r in results)
    hashes = set(r[3] for r in results)
    assert len(scores) == 1 and len(hashes) == 1, (
        "ant_budget leaked into consensus: {}".format(results))


def test_mining_walk_runs_every_step_without_a_deadline():
    """walk_steps, not the clock, is what bounds the verifier's work."""
    epoch = _epoch()
    root = G.root_lut(epoch)
    nonce = G.make_nonce("budget", 2, 10)

    _, _, _, sims = TR.mining_walk(root, "pk_test", nonce, epoch, WALK_STEPS)

    # one initial sim, plus one per walk step
    assert sims == WALK_STEPS + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
