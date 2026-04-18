"""
Image Evolution Engine for Neuraxon.

Population of filter-neurons (chains of image ops) that mutate, reproduce,
and die based on interactive user selection.

Public API:
    ImageEvolutionEngine  — top-level façade used by app.py
    FilterPopulation      — population with evolve() step
    FilterNeuron, FilterOp — single evolvable neuron + atomic op
    FILTERS               — catalogue of available filter ops
"""

from .engine import ImageEvolutionEngine
from .population import FilterPopulation
from .filter_neuron import FilterNeuron, FilterOp, MAX_CHAIN
from .filters import FILTERS, filter_names

__all__ = [
    "ImageEvolutionEngine",
    "FilterPopulation",
    "FilterNeuron",
    "FilterOp",
    "FILTERS",
    "filter_names",
    "MAX_CHAIN",
]
