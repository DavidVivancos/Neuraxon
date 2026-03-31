# Neuraxon Game of Life 3.5 Simulation Package (Neuraxon 2.0 Compliant) Internal version 125
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife

"""
Simulation Logic Package
------------------------
This package contains the world generation, physics, and entities (agents, food)
that make up the environment for the Neuraxon networks.
"""

from .entities import NxEr, NxErStats, Food, Proprioceptron
from .world import World

__all__ = ['NxEr', 'NxErStats', 'Food', 'World', 'Proprioceptron']
