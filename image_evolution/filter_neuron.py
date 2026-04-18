"""
FilterNeuron — one evolvable image effect.

Structurally modeled after the mutate/birth/death pattern in
`AigarthITU.mutate` (neuraxon2.py:1237-1256) and the neuron health-decay
in `Neuraxon.simulate` (neuraxon2.py:906-912), scaled down from
"hidden neurons inside a network" to "filter ops inside a chain."
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .filters import FILTERS, filter_names, random_params, mutate_one_param, apply_op


MAX_CHAIN = 6
MIN_CHAIN = 1
DEFAULT_CHAIN_LEN_RANGE = (1, 3)
HEALTH_DECAY = 0.25
DEATH_THRESHOLD = 0.2


@dataclass
class FilterOp:
    filter_type: str
    params: dict

    def apply(self, img: np.ndarray) -> np.ndarray:
        return apply_op(self.filter_type, self.params, img)

    def mutate_params(self) -> None:
        self.params = mutate_one_param(self.filter_type, self.params)

    @classmethod
    def random(cls) -> "FilterOp":
        ft = random.choice(filter_names())
        return cls(filter_type=ft, params=random_params(ft))

    def describe(self) -> str:
        if not self.params:
            return self.filter_type
        parts = []
        for k, v in self.params.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v}")
        return f"{self.filter_type}({', '.join(parts)})"


@dataclass
class FilterNeuron:
    id: int
    chain: list[FilterOp]
    strength: float = 1.0
    health: float = 1.0
    age: int = 0
    parent_id: Optional[int] = None

    @classmethod
    def random(cls, nid: int, chain_len: Optional[int] = None) -> "FilterNeuron":
        if chain_len is None:
            chain_len = random.randint(*DEFAULT_CHAIN_LEN_RANGE)
        chain_len = max(MIN_CHAIN, min(MAX_CHAIN, chain_len))
        chain = [FilterOp.random() for _ in range(chain_len)]
        return cls(id=nid, chain=chain, strength=random.uniform(0.6, 1.0))

    def apply(self, source: np.ndarray) -> np.ndarray:
        img = source
        for op in self.chain:
            img = op.apply(img)
        if self.strength >= 0.999:
            return img
        a = float(self.strength)
        blended = source.astype(np.float32) * (1 - a) + img.astype(np.float32) * a
        return blended.clip(0, 255).astype(np.uint8)

    def clone(self, new_id: int, new_parent: Optional[int]) -> "FilterNeuron":
        return FilterNeuron(
            id=new_id,
            chain=[FilterOp(op.filter_type, dict(op.params)) for op in self.chain],
            strength=self.strength,
            health=1.0,
            age=0,
            parent_id=new_parent,
        )

    def mutate(self) -> None:
        r = random.random()
        if r < 0.50:
            op = random.choice(self.chain)
            op.mutate_params()
        elif r < 0.60:
            self.strength = max(0.0, min(1.0, self.strength + random.gauss(0.0, 0.15)))
        elif r < 0.70:
            op = random.choice(self.chain)
            op.filter_type = random.choice(filter_names())
            op.params = random_params(op.filter_type)
        elif r < 0.80 and len(self.chain) < MAX_CHAIN:
            idx = random.randint(0, len(self.chain))
            self.chain.insert(idx, FilterOp.random())
        elif r < 0.90 and len(self.chain) > MIN_CHAIN:
            idx = random.randrange(len(self.chain))
            self.chain.pop(idx)
        elif r < 0.95 and len(self.chain) >= 2:
            i = random.randrange(len(self.chain) - 1)
            self.chain[i], self.chain[i + 1] = self.chain[i + 1], self.chain[i]
        else:
            if len(self.chain) < MAX_CHAIN:
                src = random.choice(self.chain)
                dup = FilterOp(src.filter_type, dict(src.params))
                dup.mutate_params()
                self.chain.insert(random.randint(0, len(self.chain)), dup)
            else:
                random.choice(self.chain).mutate_params()

    def decay(self) -> None:
        self.health -= HEALTH_DECAY

    def is_dead(self) -> bool:
        return self.health < DEATH_THRESHOLD

    def describe(self) -> str:
        body = " > ".join(op.describe() for op in self.chain)
        return f"#{self.id} [s={self.strength:.2f}, h={self.health:.2f}] {body}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "chain": [asdict(op) for op in self.chain],
            "strength": self.strength,
            "health": self.health,
            "age": self.age,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FilterNeuron":
        chain = [FilterOp(filter_type=o["filter_type"], params=dict(o["params"])) for o in d["chain"]]
        return cls(
            id=d["id"],
            chain=chain,
            strength=d.get("strength", 1.0),
            health=d.get("health", 1.0),
            age=d.get("age", 0),
            parent_id=d.get("parent_id"),
        )
