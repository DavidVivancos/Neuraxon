"""
Tests for the Image Evolution Engine.
"""

import json
import os
import random
import tempfile

import numpy as np
import pytest

from image_evolution import (
    FILTERS,
    FilterNeuron,
    FilterOp,
    FilterPopulation,
    ImageEvolutionEngine,
    MAX_CHAIN,
    filter_names,
)
from image_evolution.filter_neuron import MIN_CHAIN
from image_evolution.filters import INT_PARAMS, random_params, mutate_one_param


@pytest.fixture
def rgb_img():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", list(FILTERS.keys()))
def test_every_filter_preserves_shape_and_dtype(rgb_img, name):
    params = random_params(name)
    out = FILTERS[name]["apply"](rgb_img, params)
    assert out.shape == rgb_img.shape, f"{name} changed shape"
    assert out.dtype == np.uint8, f"{name} changed dtype"


@pytest.mark.parametrize("angle", [0.0, 90.0, 33.0])
def test_pixel_sort_runs_at_multiple_angles(rgb_img, angle):
    params = {"threshold": 0.2, "angle": angle, "key": 0, "reverse": 0}
    out = FILTERS["pixel_sort"]["apply"](rgb_img, params)
    assert out.shape == rgb_img.shape
    assert not np.array_equal(out, rgb_img), f"pixel_sort at {angle}° had zero effect"


def test_pixel_sort_reverse_changes_output(rgb_img):
    p_asc = {"threshold": 0.2, "angle": 0.0, "key": 0, "reverse": 0}
    p_desc = {"threshold": 0.2, "angle": 0.0, "key": 0, "reverse": 1}
    asc = FILTERS["pixel_sort"]["apply"](rgb_img, p_asc)
    desc = FILTERS["pixel_sort"]["apply"](rgb_img, p_desc)
    assert not np.array_equal(asc, desc)


def test_neon_pixels_zero_density_is_identity(rgb_img):
    p = {"density": 0.0, "brightness": 1.0, "saturation": 1.0, "glow": 0.0}
    out = FILTERS["neon_pixels"]["apply"](rgb_img, p)
    assert np.array_equal(out, rgb_img)


def test_neon_pixels_high_density_changes_many_pixels(rgb_img):
    np.random.seed(0)
    p = {"density": 0.1, "brightness": 1.0, "saturation": 1.0, "glow": 0.0}
    out = FILTERS["neon_pixels"]["apply"](rgb_img, p)
    changed = (out != rgb_img).any(axis=-1).sum()
    total = rgb_img.shape[0] * rgb_img.shape[1]
    assert changed / total > 0.02


def test_mutate_one_param_stays_in_range():
    for name in filter_names():
        spec = FILTERS[name]["params"]
        if not spec:
            continue
        params = random_params(name)
        for _ in range(200):
            params = mutate_one_param(name, params)
        for k, v in params.items():
            lo, hi, _ = spec[k]
            assert lo <= v <= hi, f"{name}.{k}={v} out of [{lo},{hi}]"
            if k in INT_PARAMS:
                assert isinstance(v, int), f"{name}.{k} should be int"


def test_filter_neuron_random_has_valid_chain():
    random.seed(1)
    n = FilterNeuron.random(nid=7)
    assert n.id == 7
    assert MIN_CHAIN <= len(n.chain) <= MAX_CHAIN
    for op in n.chain:
        assert op.filter_type in FILTERS


def test_filter_neuron_apply_matches_manual_chain(rgb_img):
    random.seed(2)
    n = FilterNeuron.random(nid=0)
    n.strength = 1.0
    manual = rgb_img
    for op in n.chain:
        manual = op.apply(manual)
    auto = n.apply(rgb_img)
    assert np.array_equal(manual, auto)


def test_mutate_chain_insert_and_remove_bounds():
    random.seed(3)
    n = FilterNeuron.random(nid=0, chain_len=3)
    for _ in range(500):
        n.mutate()
        assert MIN_CHAIN <= len(n.chain) <= MAX_CHAIN


def test_clone_preserves_parent_id():
    random.seed(4)
    parent = FilterNeuron.random(nid=10)
    parent.age = 5
    child = parent.clone(new_id=11, new_parent=parent.id)
    assert child.id == 11
    assert child.parent_id == 10
    assert child.age == 0
    assert child.health == 1.0
    assert len(child.chain) == len(parent.chain)


def test_population_seeding_creates_distinct_neurons():
    random.seed(5)
    pop = FilterPopulation(size=10)
    pop.seed()
    ids = [n.id for n in pop.neurons]
    assert len(ids) == 10
    assert len(set(ids)) == 10
    signatures = {(op.filter_type for op in n.chain) for n in pop.neurons}
    assert len(signatures) >= 1


def test_unselected_neurons_die_within_five_generations():
    random.seed(6)
    pop = FilterPopulation(size=6)
    pop.seed()
    tracked_id = pop.neurons[0].id
    for _ in range(5):
        pop.evolve(selected_ids=[])
    assert pop.by_id(tracked_id) is None, "neuron never selected should have died"


def test_evolve_children_reference_parent():
    random.seed(7)
    pop = FilterPopulation(size=6)
    pop.seed()
    parent_id = pop.neurons[0].id
    pop.evolve(selected_ids=[parent_id])
    parent_ids = {n.parent_id for n in pop.neurons if n.parent_id is not None}
    assert parent_id in parent_ids


def test_next_id_is_monotonic():
    random.seed(8)
    pop = FilterPopulation(size=5)
    pop.seed()
    first_max = max(n.id for n in pop.neurons)
    pop.evolve(selected_ids=[pop.neurons[0].id])
    new_ids = [n.id for n in pop.neurons if n.parent_id is not None]
    if new_ids:
        assert max(new_ids) > first_max


def test_population_json_roundtrip(tmp_path):
    random.seed(9)
    pop = FilterPopulation(size=4)
    pop.seed()
    pop.evolve(selected_ids=[pop.neurons[0].id])
    path = tmp_path / "pop.json"
    pop.save(str(path))
    loaded = FilterPopulation.load(str(path))
    assert loaded.size == pop.size
    assert loaded.generation == pop.generation
    assert loaded.next_id == pop.next_id
    assert [n.id for n in loaded.neurons] == [n.id for n in pop.neurons]
    assert [len(n.chain) for n in loaded.neurons] == [len(n.chain) for n in pop.neurons]


def test_engine_end_to_end_flow(rgb_img):
    random.seed(10)
    from PIL import Image
    engine = ImageEvolutionEngine(population_size=4, max_side=64)
    engine.load_image(Image.fromarray(rgb_img))
    engine.seed()
    previews = engine.previews()
    assert len(previews) == 4
    for p in previews:
        assert p.shape == rgb_img.shape
    first_ids = set(engine.neuron_ids())
    engine.step(selected_ids=[engine.neuron_ids()[0]])
    second_ids = set(engine.neuron_ids())
    assert first_ids != second_ids
    stats = engine.stats()
    assert stats["generation"] == 1
    assert stats["population"] == 4

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        saved = engine.save_best(path)
        assert saved == path
        assert os.path.exists(path) and os.path.getsize(path) > 0
    finally:
        os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
