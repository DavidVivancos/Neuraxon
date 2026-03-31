# Neuraxon Game of Life 3.5 Neuron Package (Neuraxon 2.0 Compliant) Internal version 125
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife

"""
Neuraxon Neural Network Library
-------------------------------
Implements the core neural computation model based on the Neuraxon research paper.
Includes neurons, synapses, dendritic branches, and network-level plasticity rules.
"""

# Expose key classes for easier importing
from .neuron import Neuraxon
from .network import NeuraxonNetwork
from .components import Synapse, DendriticBranch, ITUCircle, MSTHState, ReceptorSubtype, OscillatorBank, NeuromodulatorSystem
from .enums import NeuronType, SynapseType, TrinaryState
from .genetics import Inheritance

__all__ = [
    'Neuraxon',
    'NeuraxonNetwork',
    'Synapse',
    'DendriticBranch',
    'ITUCircle',
    'NeuronType',
    'SynapseType',
    'TrinaryState',
    'Inheritance'
]
