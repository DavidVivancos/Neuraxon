"""
ImageEvolutionEngine — thin façade used by the Gradio UI.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from PIL import Image

from .population import FilterPopulation


DEFAULT_MAX_SIDE = 512


class ImageEvolutionEngine:
    def __init__(self, population_size: int = 12, max_side: int = DEFAULT_MAX_SIDE):
        self.max_side = max_side
        self.population = FilterPopulation(size=population_size)
        self.source: Optional[np.ndarray] = None

    def load_image(self, pil_img: Image.Image) -> np.ndarray:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        scale = min(1.0, self.max_side / max(w, h))
        if scale < 1.0:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        self.source = np.asarray(pil_img, dtype=np.uint8)
        return self.source

    def seed(self, size: Optional[int] = None) -> None:
        self.population.seed(size)

    def previews(self) -> list[np.ndarray]:
        if self.source is None:
            raise RuntimeError("Load an image first with load_image().")
        return self.population.render_all(self.source)

    def labels(self) -> list[str]:
        return [n.describe() for n in self.population.neurons]

    def neuron_ids(self) -> list[int]:
        return [n.id for n in self.population.neurons]

    def step(self, selected_ids: Iterable[int]) -> dict:
        return self.population.evolve(selected_ids)

    def stats(self) -> dict:
        hist = self.population.filter_histogram()
        last = self.population.history[-1] if self.population.history else {}
        return {
            "generation": self.population.generation,
            "population": len(self.population.neurons),
            "filter_histogram": hist,
            "last_gen_stats": last,
        }

    def save_best(self, path: str) -> Optional[str]:
        if self.source is None:
            return None
        best = self.population.best_by_age()
        if best is None:
            return None
        img = best.apply(self.source)
        Image.fromarray(img).save(path)
        return path
