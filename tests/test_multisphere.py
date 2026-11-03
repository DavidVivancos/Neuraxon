"""Invariants for MultiNeuraxon 2.0.

Multi-Sphere composes unchanged Neuraxon networks into a graph, so the things
worth pinning are the ones the graph layer owns and the single-sphere tests
cannot reach: that the whole graph survives a save/load, that an inter-sphere
link delays by the number of steps it was asked for, and that the trinary
conversion treats its two thresholds symmetrically.

MultiNeuraxon2.py had no test coverage before this file.
"""

import json
import random

import pytest

from MultiNeuraxon2 import (
    NetworkParameters,
    NeuraxonMultiSphere,
    SphereLinkParameters,
    _continuous_to_trinary,
    load_multisphere,
    save_multisphere,
)

SEED = 909


def _sphere_params() -> NetworkParameters:
    return NetworkParameters(
        num_input_neurons=3,
        num_hidden_neurons=5,
        num_output_neurons=2,
    )


def _build(seed: int = SEED, steps: int = 12) -> NeuraxonMultiSphere:
    random.seed(seed)
    ms = NeuraxonMultiSphere("fixture")
    for sphere_id in ("A", "B", "C"):
        ms.add_sphere(sphere_id, params=_sphere_params())
    ms.connect_spheres("A", "B")
    ms.connect_spheres("B", "C")
    for _ in range(steps):
        ms.simulate_step()
    return ms


def _fingerprint(ms: NeuraxonMultiSphere) -> dict:
    return {
        "spheres": sorted(ms.spheres),
        "links": sorted(ms.links),
        "layers": sorted(ms.layers),
        "step_count": ms.step_count,
        "time": round(ms.time, 10),
        "energy": round(ms.get_energy(), 10),
        "synapse_counts": {k: len(v.network.synapses) for k, v in sorted(ms.spheres.items())},
        "w_fast": {
            k: [round(s.w_fast, 10) for s in v.network.synapses]
            for k, v in sorted(ms.spheres.items())
        },
        "outputs": {
            k: list(v.network.get_output_states()) for k, v in sorted(ms.spheres.items())
        },
    }


def test_graph_is_built_as_requested():
    ms = _build(steps=0)
    assert sorted(ms.spheres) == ["A", "B", "C"]
    assert len(ms.links) == 2
    assert len(ms.layers) == 1, "spheres default into a single layer L0"


def test_duplicate_sphere_id_is_rejected():
    """Silently replacing a sphere would discard a trained network."""
    random.seed(SEED)
    ms = NeuraxonMultiSphere("dup")
    ms.add_sphere("A", params=_sphere_params())
    with pytest.raises(ValueError):
        ms.add_sphere("A", params=_sphere_params())


def test_multisphere_round_trip(tmp_path):
    """The whole graph — spheres, links, layers and clock — survives a save."""
    ms = _build()
    path = tmp_path / "ms.json"
    save_multisphere(ms, str(path))

    before = _fingerprint(ms)
    after = _fingerprint(load_multisphere(str(path)))

    for key in before:
        assert after[key] == before[key], f"{key} did not survive the round trip"


def test_saved_multisphere_has_expected_sections(tmp_path):
    ms = _build(steps=3)
    path = tmp_path / "ms.json"
    save_multisphere(ms, str(path))

    data = json.loads(path.read_text())
    for section in ("spheres", "links"):
        assert section in data, f"saved multisphere is missing '{section}'"


def test_seeded_multisphere_is_reproducible():
    assert _fingerprint(_build(SEED)) == _fingerprint(_build(SEED))


def test_link_delay_holds_signal_for_requested_steps():
    """A link with delay_steps=N must emit N steps of silence first.

    project() keeps a pre-filled deque and pops one payload per call. If the
    buffer were ever built empty, append-then-popleft would return the payload
    just pushed and the delay would silently vanish, which is invisible from
    the outside because the values stay plausible.
    """
    random.seed(SEED)
    ms = NeuraxonMultiSphere("delay")
    ms.add_sphere("src", params=_sphere_params())
    ms.add_sphere("dst", params=_sphere_params())
    delay = 3
    link_id = ms.connect_spheres(
        "src", "dst", params=SphereLinkParameters(delay_steps=delay)
    )
    link = ms.links[link_id]

    # Drive the source so its payload is not all zeros by coincidence.
    for neuron, state in zip(ms.spheres["src"].network.output_neurons, (1, -1)):
        neuron.trinary_state = state

    emitted = [
        link.project(ms.spheres["src"], ms.spheres["dst"]) for _ in range(delay + 2)
    ]

    for step in range(delay):
        assert all(v == 0.0 for v in emitted[step].values()), (
            f"link emitted a non-zero payload at step {step}, "
            f"before its delay of {delay} elapsed"
        )
    assert any(v != 0.0 for v in emitted[delay].values()), (
        "link never emitted the payload it buffered"
    )


def test_zero_delay_link_passes_straight_through():
    random.seed(SEED)
    ms = NeuraxonMultiSphere("nodelay")
    ms.add_sphere("src", params=_sphere_params())
    ms.add_sphere("dst", params=_sphere_params())
    link_id = ms.connect_spheres(
        "src", "dst", params=SphereLinkParameters(delay_steps=0)
    )
    link = ms.links[link_id]

    for neuron, state in zip(ms.spheres["src"].network.output_neurons, (1, -1)):
        neuron.trinary_state = state

    first = link.project(ms.spheres["src"], ms.spheres["dst"])
    assert any(v != 0.0 for v in first.values()), (
        "a zero-delay link should deliver on the first step"
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.0, 1),
        (0.26, 1),
        (0.25, 0),      # exactly at the threshold is not above it
        (0.0, 0),
        (-0.25, 0),     # and the negative side must match
        (-0.26, -1),
        (-1.0, -1),
    ],
)
def test_continuous_to_trinary_thresholds(value, expected):
    """The two thresholds must be symmetric, including on the boundary."""
    assert _continuous_to_trinary(value) == expected


def test_multisphere_states_stay_trinary():
    ms = _build(steps=20)
    for sphere in ms.spheres.values():
        assert {n.trinary_state for n in sphere.network.all_neurons} <= {-1, 0, 1}
