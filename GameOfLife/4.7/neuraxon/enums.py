# Neuraxon Game of Life v.4.79 enums (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 171
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
from enum import Enum

class NeuronType(Enum):
    """Defines the structural role of a neuron within the network."""
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"

class SynapseType(Enum):
    """
    Defines the functional type of a synapse.
    
    - IONOTROPIC_FAST: Rapid signal transmission (AMPA-like).
    - IONOTROPIC_SLOW: Slower, sustained transmission (NMDA-like).
    - METABOTROPIC: Modulates learning rates and excitability rather than direct voltage.
    - SILENT: Structurally present but functionally inactive until potentiated.
    """
    IONOTROPIC_FAST = "ionotropic_fast"
    IONOTROPIC_SLOW = "ionotropic_slow"
    METABOTROPIC = "metabotropic"
    SILENT = "silent"

class TrinaryState(Enum):
    """
    Represents the discrete output state of a Neuraxon.
    Unlike binary neurons (0/1), Neuraxons can be Inhibitory (-1), Neutral (0), or Excitatory (1).
    """
    INHIBITORY = -1
    NEUTRAL = 0
    EXCITATORY = 1
