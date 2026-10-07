"""Invariants for Neuraxon v2.0.

The repository publishes code alongside a paper, so the properties worth
protecting are the ones a reader depends on when reproducing a figure: that a
seeded run is repeatable, and that a network written to disk is the network
that comes back. Both hold today; these tests are here so that stays true.

neuraxon2.py had no test coverage before this file.
"""

import json
import random

import pytest

from neuraxon2 import (
    NetworkParameters,
    NeuraxonNetwork,
    load_network,
    save_network,
)

SEED = 4242


def _small_params() -> NetworkParameters:
    return NetworkParameters(
        num_input_neurons=3,
        num_hidden_neurons=6,
        num_output_neurons=2,
    )


def _build_and_run(seed: int, steps: int = 20) -> NeuraxonNetwork:
    """A fully determined experiment: seed, construct, drive, return."""
    random.seed(seed)
    net = NeuraxonNetwork(_small_params())
    for i in range(steps):
        net.set_input_states([1, -1, 1] if i % 2 == 0 else [-1, 1, 0])
        net.simulate_step()
    return net


def _fingerprint(net: NeuraxonNetwork) -> dict:
    """Everything that should survive a save/load or repeat a seeded run."""
    return {
        "synapse_count": len(net.synapses),
        "step_count": net.step_count,
        "energy": round(net.energy_usage, 10),
        "outputs": list(net.get_output_states()),
        "w_fast": [round(s.w_fast, 10) for s in net.synapses],
        "w_slow": [round(s.w_slow, 10) for s in net.synapses],
        "integrity": [round(s.integrity, 10) for s in net.synapses],
        "trinary": [n.trinary_state for n in net.all_neurons],
        "health": [round(n.health, 10) for n in net.all_neurons],
    }


def test_seeded_run_is_reproducible():
    """Same seed, same trajectory.

    Weight initialisation, topology and the spontaneous-activity draws all come
    from the `random` module, so seeding it has to be sufficient to pin the
    whole run. If this fails, a published figure cannot be regenerated.
    """
    assert _fingerprint(_build_and_run(SEED)) == _fingerprint(_build_and_run(SEED))


def test_different_seeds_diverge():
    """Guards the test above from passing vacuously.

    If the network ignored the RNG entirely, reproducibility would hold for a
    trivial reason. Two different seeds must give different networks.
    """
    assert _fingerprint(_build_and_run(SEED)) != _fingerprint(_build_and_run(SEED + 1))


def test_save_load_round_trip(tmp_path):
    """A loaded network equals the one that was saved.

    load_network() rebuilds a fresh randomly-initialised network and then
    overwrites it from the file, so anything the restore path forgets is
    silently replaced by random values rather than raising.
    """
    net = _build_and_run(SEED)
    path = tmp_path / "net.json"
    save_network(net, str(path))

    before = _fingerprint(net)
    after = _fingerprint(load_network(str(path)))

    for key in before:
        assert after[key] == before[key], f"{key} did not survive the round trip"


def test_round_trip_is_idempotent(tmp_path):
    """Saving a loaded network and loading it again changes nothing.

    The restore path renormalises dsn_kernel_weights, so a round trip that is
    not idempotent would let state drift a little further on every save/load
    cycle.
    """
    net = _build_and_run(SEED)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    save_network(net, str(first))
    once = load_network(str(first))
    save_network(once, str(second))
    twice = load_network(str(second))

    assert _fingerprint(twice) == _fingerprint(once)


def test_saved_file_is_json_with_expected_sections(tmp_path):
    """The on-disk format is a contract for anything reading these files."""
    net = _build_and_run(SEED)
    path = tmp_path / "net.json"
    save_network(net, str(path))

    data = json.loads(path.read_text())
    for section in ("parameters", "neurons", "synapses"):
        assert section in data, f"saved network is missing '{section}'"
    for group in ("input", "hidden", "output"):
        assert group in data["neurons"]


def test_network_honours_requested_sizes():
    random.seed(SEED)
    net = NeuraxonNetwork(_small_params())
    assert len(net.input_neurons) == 3
    assert len(net.hidden_neurons) == 6
    assert len(net.output_neurons) == 2
    assert len(net.all_neurons) == 11


def test_states_stay_trinary():
    """Trinary states are the central claim of the model: -1, 0 or 1 only."""
    net = _build_and_run(SEED, steps=30)
    assert {n.trinary_state for n in net.all_neurons} <= {-1, 0, 1}
    assert set(net.get_output_states()) <= {-1, 0, 1}


def test_energy_is_monotonic_and_finite():
    """Energy accounting only accumulates; it should never go backwards."""
    random.seed(SEED)
    net = NeuraxonNetwork(_small_params())
    previous = net.get_energy()
    for i in range(15):
        net.set_input_states([1, 0, -1])
        net.simulate_step()
        current = net.get_energy()
        assert current == pytest.approx(current)  # not NaN
        assert current >= previous
        previous = current


# ---------------------------------------------------------------------------
# Golden trajectory
# ---------------------------------------------------------------------------
#
# test_seeded_run_is_reproducible above proves a run repeats itself, but it
# cannot notice if the result *changes*: both runs would simply agree on a new
# value. For a repository published alongside a paper that is the dangerous
# case, because figures stop matching the code without anything failing.
#
# This pins the actual trajectory. If it fails and the change was deliberate,
# regenerate the digest with:
#
#     python -c "import tests.test_neuraxon2 as t; print(t._trajectory_digest())"
#
# and update GOLDEN_TRAJECTORY in the same commit as the behaviour change, so
# the diff records that results moved.

GOLDEN_SEED = 4242
GOLDEN_STEPS = 20
GOLDEN_TRAJECTORY = "e3f09d116702fa75dec50a0a207efc24f33eacbc08a9c35f99256a2f1a23ef73"


def _trajectory_digest(seed: int = GOLDEN_SEED, steps: int = GOLDEN_STEPS) -> str:
    import hashlib

    random.seed(seed)
    net = NeuraxonNetwork(_small_params())
    trace = []
    for i in range(steps):
        net.set_input_states([1, -1, 1] if i % 2 == 0 else [-1, 1, 0])
        net.simulate_step()
        trace.append(tuple(net.get_output_states()))
    payload = {
        "trace": trace,
        "w_fast": [round(s.w_fast, 9) for s in net.synapses],
        "trinary": [n.trinary_state for n in net.all_neurons],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def test_golden_trajectory_is_unchanged():
    assert _trajectory_digest() == GOLDEN_TRAJECTORY, (
        "the seeded trajectory moved; if that was intended, regenerate "
        "GOLDEN_TRAJECTORY (see the note above this test) in the same commit"
    )
