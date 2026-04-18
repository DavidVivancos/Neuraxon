"""
FilterPopulation — clone-and-mutate evolution of FilterNeurons.

Selection loop mirrors NeuraxonAigarthHybrid.evolve (neuraxon2.py:1273-1285)
and the clone-and-mutate genetics from GameOfLife/4/neuraxon/genetics.py.
Selection is interactive: the user passes the IDs they like; everyone else
loses health and dies.
"""

from __future__ import annotations

import json
import random
from typing import Iterable, Optional

import numpy as np

from .filter_neuron import FilterNeuron


SPONTANEOUS_BIRTH_PROB = 0.05


class FilterPopulation:
    def __init__(self, size: int = 12):
        self.size = size
        self.neurons: list[FilterNeuron] = []
        self.generation: int = 0
        self.next_id: int = 0
        self.history: list[dict] = []

    def _new_id(self) -> int:
        nid = self.next_id
        self.next_id += 1
        return nid

    def seed(self, n: Optional[int] = None) -> None:
        if n is not None:
            self.size = n
        self.neurons = [FilterNeuron.random(self._new_id()) for _ in range(self.size)]
        self.generation = 0
        self.history = [{"gen": 0, "births": self.size, "deaths": 0, "survivors": self.size}]

    def by_id(self, nid: int) -> Optional[FilterNeuron]:
        for n in self.neurons:
            if n.id == nid:
                return n
        return None

    def render_all(self, img: np.ndarray) -> list[np.ndarray]:
        return [n.apply(img) for n in self.neurons]

    def evolve(self, selected_ids: Iterable[int]) -> dict:
        """One generation.

        If the user picked ≥1 neuron: the unpicked are rejected and die this
        round; the picks survive (age += 1) and the rest of the population is
        filled with clone-and-mutate children of the picks.

        If no neurons were picked: everyone decays, dead ones are removed,
        and the remainder is filled from the healthiest survivors — the
        population drifts rather than being steered.
        """
        selected = set(int(s) for s in selected_ids)

        survivors: list[FilterNeuron] = []
        deaths = 0

        if selected:
            for n in self.neurons:
                if n.id in selected:
                    n.age += 1
                    survivors.append(n)
                else:
                    deaths += 1
            parents = [n for n in survivors]
        else:
            for n in self.neurons:
                n.decay()
                if n.is_dead():
                    deaths += 1
                else:
                    survivors.append(n)
            parents = sorted(survivors, key=lambda x: -x.health)[: max(1, len(survivors) // 2)]

        if not parents:
            self.seed(self.size)
            self.generation += 1  # seed() reset it, so we bump past gen 0
            stats = {
                "gen": self.generation,
                "births": self.size,
                "deaths": deaths,
                "survivors": 0,
                "picked": len(selected),
                "extinct": True,
            }
            self.history[-1] = {**self.history[-1], **stats}
            return stats

        next_gen = list(survivors)
        births = 0
        while len(next_gen) < self.size:
            parent = random.choice(parents)
            child = parent.clone(self._new_id(), parent.id)
            child.mutate()
            next_gen.append(child)
            births += 1

        if random.random() < SPONTANEOUS_BIRTH_PROB and len(next_gen) > 0:
            victim_idx = random.randrange(len(next_gen))
            if next_gen[victim_idx].id not in selected:
                next_gen[victim_idx] = FilterNeuron.random(self._new_id())
                births += 1

        self.neurons = next_gen[: self.size]
        self.generation += 1
        stats = {
            "gen": self.generation,
            "births": births,
            "deaths": deaths,
            "survivors": len(survivors),
            "picked": len(selected),
        }
        self.history.append(stats)
        return stats

    def filter_histogram(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self.neurons:
            for op in n.chain:
                out[op.filter_type] = out.get(op.filter_type, 0) + 1
        return out

    def best_by_age(self) -> Optional[FilterNeuron]:
        if not self.neurons:
            return None
        return max(self.neurons, key=lambda n: (n.age, n.health))

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "generation": self.generation,
            "next_id": self.next_id,
            "history": self.history,
            "neurons": [n.to_dict() for n in self.neurons],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FilterPopulation":
        pop = cls(size=d["size"])
        pop.generation = d.get("generation", 0)
        pop.next_id = d.get("next_id", 0)
        pop.history = d.get("history", [])
        pop.neurons = [FilterNeuron.from_dict(nd) for nd in d.get("neurons", [])]
        return pop

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FilterPopulation":
        with open(path) as f:
            return cls.from_dict(json.load(f))
