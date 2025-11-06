"""
Neuraxon: Bio-inspired Neural Network with Trinary States 
Based on the paper "Neuraxon"
Hibridized with Aigarth Intelligent Tissue https://github.com/Aigarth/aigarth-it

This implementation includes:
- Trinary neuron states (-1, 0, 1)
- Ring architecture (input, hidden, output neurons)
- Multiple synapse types (ionotropic fast/slow, metabotropic)
- Neuromodulators (dopamine, serotonin, acetylcholine, norepinephrine)
- Synaptic plasticity (growth, death, silent synapses)
- Spontaneous neural activity
- Continuous processing model
"""

import json
import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


# =============================================================================
# NETWORK PARAMETERS - Biologically Plausible Ranges   
# =============================================================================

@dataclass
class NetworkParameters:
    """Default network parameters with biologically plausible ranges"""
    
    # Network Architecture
    num_input_neurons: int = 5         # Range: [1, 100]
    num_hidden_neurons: int = 20      # Range: [1, 1000]
    num_output_neurons: int = 5      # Range: [1, 100]
    connection_probability: float = 0.05  # Range: [0.0, 1.0]
    
    # Neuron Parameters
    membrane_time_constant: float = 20.0      # ms, Range: [5.0, 50.0]
    firing_threshold_excitatory: float = 1.0  # Range: [0.5, 2.0]
    firing_threshold_inhibitory: float = -1.0 # Range: [-2.0, -0.5]
    adaptation_rate: float = 0.05             # Range: [0.0, 0.2]
    spontaneous_firing_rate: float = 0.01     # Range: [0.0, 0.1]
    neuron_health_decay: float = 0.001        # Range: [0.0, 0.01]
    
    # Synapse Parameters - Fast (Ionotropic)
    tau_fast: float = 5.0              # ms, Range: [1.0, 10.0]
    w_fast_init_min: float = -1.0     # Range: [-1.0, 0.0]
    w_fast_init_max: float = 1.0      # Range: [0.0, 1.0]
    
    # Synapse Parameters - Slow (NMDA-like)
    tau_slow: float = 50.0             # ms, Range: [20.0, 100.0]
    w_slow_init_min: float = -0.5     # Range: [-1.0, 0.0]
    w_slow_init_max: float = 0.5      # Range: [0.0, 1.0]
    
    # Synapse Parameters - Metabotropic
    tau_meta: float = 1000.0           # ms, Range: [500.0, 5000.0]
    w_meta_init_min: float = -0.3     # Range: [-0.5, 0.0]
    w_meta_init_max: float = 0.3      # Range: [0.0, 0.5]
    
    # Plasticity Parameters
    learning_rate: float = 0.01        # Range: [0.0, 0.1]
    stdp_window: float = 20.0         # ms, Range: [10.0, 50.0]
    synapse_integrity_threshold: float = 0.1  # Range: [0.0, 0.5]
    synapse_formation_prob: float = 0.05      # Range: [0.0, 0.2]
    synapse_death_prob: float = 0.01          # Range: [0.0, 0.1]
    neuron_death_threshold: float = 0.1       # Range: [0.0, 0.3]
    
    # Neuromodulator Parameters
    dopamine_baseline: float = 0.1     # Range: [0.0, 1.0]
    serotonin_baseline: float = 0.1    # Range: [0.0, 1.0]
    acetylcholine_baseline: float = 0.1  # Range: [0.0, 1.0]
    norepinephrine_baseline: float = 0.1  # Range: [0.0, 1.0]
    neuromod_decay_rate: float = 0.1   # Range: [0.0, 0.5]
    
    # Simulation Parameters
    dt: float = 1.0                    # ms, Range: [0.1, 10.0]
    simulation_steps: int = 100        # Range: [1, 10000]


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class NeuronType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class SynapseType(Enum):
    IONOTROPIC_FAST = "ionotropic_fast"
    IONOTROPIC_SLOW = "ionotropic_slow"
    METABOTROPIC = "metabotropic"
    SILENT = "silent"


class TrinaryState(Enum):
    INHIBITORY = -1
    NEUTRAL = 0
    EXCITATORY = 1


# =============================================================================
# SYNAPSE CLASS
# =============================================================================

class Synapse:
    """
    Bio-inspired synapse with multiple weight types and dynamics
    """
    
    def __init__(self, pre_id: int, post_id: int, params: NetworkParameters):
        self.pre_id = pre_id
        self.post_id = post_id
        self.params = params
        
        # Triple weight system
        self.w_fast = random.uniform(params.w_fast_init_min, params.w_fast_init_max)
        self.w_slow = random.uniform(params.w_slow_init_min, params.w_slow_init_max)
        self.w_meta = random.uniform(params.w_meta_init_min, params.w_meta_init_max)
        
        # Synapse properties
        self.is_silent = random.random() < 0.1  # 10% silent synapses
        self.is_modulatory = random.random() < 0.2  # 20% modulatory
        self.integrity = 1.0  # Health of the synapse
        
        # Traces for STDP
        self.pre_trace = 0.0
        self.post_trace = 0.0
        
        # Determine primary type
        self.synapse_type = self._determine_type()
    
    def _determine_type(self) -> SynapseType:
        """Determine the primary synapse type"""
        if self.is_silent:
            return SynapseType.SILENT
        elif self.is_modulatory:
            return SynapseType.METABOTROPIC
        elif abs(self.w_fast) > abs(self.w_slow):
            return SynapseType.IONOTROPIC_FAST
        else:
            return SynapseType.IONOTROPIC_SLOW
    
    def compute_input(self, pre_state: int) -> float:
        """
        Compute synaptic input based on presynaptic state
        
        Args:
            pre_state: Trinary state of presynaptic neuron (-1, 0, 1)
        
        Returns:
            Synaptic current contribution
        """
        if self.is_silent:
            return 0.0
        
        # Combined ionotropic contribution
        return (self.w_fast + self.w_slow) * pre_state
    
    def update(self, pre_state: int, post_state: int, 
               neuromodulators: Dict[str, float], dt: float):
        """
        Update synaptic weights based on activity and neuromodulation
        
        Args:
            pre_state: Presynaptic neuron state
            post_state: Postsynaptic neuron state
            neuromodulators: Dict of neuromodulator concentrations
            dt: Time step
        """
        # Update traces
        tau_trace = self.params.stdp_window
        self.pre_trace += (-self.pre_trace / tau_trace + (1 if pre_state == 1 else 0)) * dt
        self.post_trace += (-self.post_trace / tau_trace + (1 if post_state == 1 else 0)) * dt
        
        # STDP-like plasticity
        if pre_state == 1 and post_state == 1:
            # LTP: strengthen synapse
            delta_w = self.params.learning_rate * neuromodulators.get('dopamine', 0.5)
        elif pre_state == 1 and post_state == -1:
            # LTD: weaken synapse
            delta_w = -self.params.learning_rate * neuromodulators.get('dopamine', 0.5)
        else:
            delta_w = 0.0
        
        # Update fast weight (ionotropic)
        self.w_fast += dt / self.params.tau_fast * (-self.w_fast + delta_w * 0.3)
        self.w_fast = max(-1.0, min(1.0, self.w_fast))
        
        # Update slow weight (NMDA-like)
        self.w_slow += dt / self.params.tau_slow * (-self.w_slow + delta_w * 0.1)
        self.w_slow = max(-1.0, min(1.0, self.w_slow))
        
        # Update metabotropic weight (very slow, modulated by serotonin)
        serotonin_effect = neuromodulators.get('serotonin', 0.5)
        self.w_meta += dt / self.params.tau_meta * (
            -self.w_meta + delta_w * 0.05 * serotonin_effect
        )
        self.w_meta = max(-0.5, min(0.5, self.w_meta))
        
        # Update integrity
        if abs(self.w_fast) < 0.01 and abs(self.w_slow) < 0.01:
            self.integrity -= self.params.synapse_death_prob * dt
        else:
            self.integrity = min(1.0, self.integrity + 0.001 * dt)
        
        # Unsilence silent synapses with LTP
        if self.is_silent and pre_state == 1 and post_state == 1:
            if random.random() < 0.01:  # Small probability
                self.is_silent = False
                self.synapse_type = self._determine_type()
    
    def get_modulatory_effect(self) -> float:
        """Get the modulatory effect on postsynaptic neuron"""
        if self.is_modulatory:
            return self.w_meta
        return 0.0
    
    def to_dict(self) -> dict:
        """Convert synapse to dictionary for JSON serialization"""
        return {
            'pre_id': self.pre_id,
            'post_id': self.post_id,
            'w_fast': self.w_fast,
            'w_slow': self.w_slow,
            'w_meta': self.w_meta,
            'is_silent': self.is_silent,
            'is_modulatory': self.is_modulatory,
            'integrity': self.integrity,
            'synapse_type': self.synapse_type.value
        }


# =============================================================================
# NEURON (NEURAXON) CLASS
# =============================================================================

class Neuraxon:
    """
    Bio-inspired neuron with trinary states and complex dynamics
    """
    
    def __init__(self, neuron_id: int, neuron_type: NeuronType, 
                 params: NetworkParameters):
        self.id = neuron_id
        self.type = neuron_type
        self.params = params
        
        # State variables
        self.membrane_potential = 0.0
        self.trinary_state = TrinaryState.NEUTRAL.value
        self.adaptation = 0.0
        self.autoreceptor = 0.0
        
        # Health and activity
        self.health = 1.0
        self.is_active = True
        
        # History for visualization
        self.state_history = []
        self.potential_history = []
    
    def update(self, synaptic_inputs: List[float], 
               modulatory_inputs: List[float],
               external_input: float,
               neuromodulators: Dict[str, float],
               dt: float):
        """
        Update neuron state based on inputs and internal dynamics
        
        Args:
            synaptic_inputs: List of synaptic current contributions
            modulatory_inputs: List of modulatory effects
            external_input: External drive (for input neurons)
            neuromodulators: Global neuromodulator levels
            dt: Time step
        """
        if not self.is_active:
            return
        
        # Sum all inputs
        total_synaptic = sum(synaptic_inputs)
        total_modulatory = sum(modulatory_inputs)
        
        # Spontaneous activity
        spontaneous = 0.0
        if random.random() < self.params.spontaneous_firing_rate * dt:
            spontaneous = random.uniform(-0.5, 0.5)
        
        # Membrane potential dynamics
        tau = self.params.membrane_time_constant
        self.membrane_potential += dt / tau * (
            -self.membrane_potential 
            + total_synaptic 
            + external_input 
            - self.adaptation 
            + spontaneous
        )
        
        # Adaptation (frequency adaptation)
        tau_adapt = 100.0  # ms
        self.adaptation += dt / tau_adapt * (
            -self.adaptation + 0.1 * abs(self.trinary_state)
        )
        
        # Autoreceptor (self-inhibition/modulation)
        tau_auto = 200.0  # ms
        self.autoreceptor += dt / tau_auto * (
            -self.autoreceptor + 0.2 * self.trinary_state
        )
        
        # Effective thresholds (modulated by neuromodulators and autoreceptor)
        acetylcholine = neuromodulators.get('acetylcholine', 0.5)
        norepinephrine = neuromodulators.get('norepinephrine', 0.5)
        
        threshold_mod = (acetylcholine - 0.5) * 0.5 + total_modulatory * 0.3
        
        theta_exc = self.params.firing_threshold_excitatory - threshold_mod - 0.1 * self.autoreceptor
        theta_inh = self.params.firing_threshold_inhibitory - threshold_mod + 0.1 * self.autoreceptor
        
        # Determine trinary state
        if self.membrane_potential > theta_exc:
            self.trinary_state = TrinaryState.EXCITATORY.value
        elif self.membrane_potential < theta_inh:
            self.trinary_state = TrinaryState.INHIBITORY.value
        else:
            self.trinary_state = TrinaryState.NEUTRAL.value
        
        # Update health
        activity_level = abs(self.membrane_potential) / 2.0
        if activity_level < 0.01:
            self.health -= self.params.neuron_health_decay * dt
        else:
            self.health = min(1.0, self.health + 0.0005 * dt)
        
        # Check if neuron should die (only for hidden neurons)
        if self.type == NeuronType.HIDDEN and self.health < self.params.neuron_death_threshold:
            if random.random() < 0.001:  # Very rare
                self.is_active = False
        
        # Store history
        self.state_history.append(self.trinary_state)
        self.potential_history.append(self.membrane_potential)
        if len(self.state_history) > 1000:
            self.state_history.pop(0)
            self.potential_history.pop(0)
    
    def set_state(self, state: int):
        """Manually set neuron state (for input neurons)"""
        if state in [-1, 0, 1]:
            self.trinary_state = state
            self.membrane_potential = state * self.params.firing_threshold_excitatory
    
    def to_dict(self) -> dict:
        """Convert neuron to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'type': self.type.value,
            'membrane_potential': self.membrane_potential,
            'trinary_state': self.trinary_state,
            'adaptation': self.adaptation,
            'health': self.health,
            'is_active': self.is_active
        }


# =============================================================================
# NEURAXON NETWORK CLASS
# =============================================================================

class NeuraxonNetwork:
    """
    Complete Neuraxon network with ring architecture
    """
    
    def __init__(self, params: Optional[NetworkParameters] = None):
        self.params = params or NetworkParameters()
        
        # Neurons
        self.input_neurons: List[Neuraxon] = []
        self.hidden_neurons: List[Neuraxon] = []
        self.output_neurons: List[Neuraxon] = []
        self.all_neurons: List[Neuraxon] = []
        
        # Synapses
        self.synapses: List[Synapse] = []
        
        # Neuromodulators (global state)
        self.neuromodulators = {
            'dopamine': self.params.dopamine_baseline,
            'serotonin': self.params.serotonin_baseline,
            'acetylcholine': self.params.acetylcholine_baseline,
            'norepinephrine': self.params.norepinephrine_baseline
        }
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        
        # Initialize network
        self._initialize_neurons()
        self._initialize_synapses()
    
    def _initialize_neurons(self):
        """Create neurons with ring architecture"""
        neuron_id = 0
        
        # Input neurons
        for _ in range(self.params.num_input_neurons):
            neuron = Neuraxon(neuron_id, NeuronType.INPUT, self.params)
            self.input_neurons.append(neuron)
            self.all_neurons.append(neuron)
            neuron_id += 1
        
        # Hidden neurons
        for _ in range(self.params.num_hidden_neurons):
            neuron = Neuraxon(neuron_id, NeuronType.HIDDEN, self.params)
            self.hidden_neurons.append(neuron)
            self.all_neurons.append(neuron)
            neuron_id += 1
        
        # Output neurons
        for _ in range(self.params.num_output_neurons):
            neuron = Neuraxon(neuron_id, NeuronType.OUTPUT, self.params)
            self.output_neurons.append(neuron)
            self.all_neurons.append(neuron)
            neuron_id += 1
    
    def _initialize_synapses(self):
        """
        Create synapses following constraints:
        - Output neurons cannot connect to input neurons
        - Ring architecture with forward/backward connections
        """
        for pre_neuron in self.all_neurons:
            for post_neuron in self.all_neurons:
                if pre_neuron.id == post_neuron.id:
                    continue
                
                # Constraint: Output cannot connect to Input
                if (pre_neuron.type == NeuronType.OUTPUT and 
                    post_neuron.type == NeuronType.INPUT):
                    continue
                
                # Probabilistic connection
                if random.random() < self.params.connection_probability:
                    synapse = Synapse(pre_neuron.id, post_neuron.id, self.params)
                    self.synapses.append(synapse)
    
    def simulate_step(self, external_inputs: Optional[Dict[int, float]] = None):
        """
        Simulate one time step
        
        Args:
            external_inputs: Dict mapping neuron IDs to external input values
        """
        external_inputs = external_inputs or {}
        
        # Collect inputs for each neuron
        neuron_synaptic_inputs = {n.id: [] for n in self.all_neurons}
        neuron_modulatory_inputs = {n.id: [] for n in self.all_neurons}
        
        # Process synapses
        for synapse in self.synapses:
            if synapse.integrity <= 0:
                continue
            
            pre_neuron = self.all_neurons[synapse.pre_id]
            if not pre_neuron.is_active:
                continue
            
            # Compute synaptic input
            syn_input = synapse.compute_input(pre_neuron.trinary_state)
            neuron_synaptic_inputs[synapse.post_id].append(syn_input)
            
            # Add modulatory effect
            mod_effect = synapse.get_modulatory_effect()
            if mod_effect != 0:
                neuron_modulatory_inputs[synapse.post_id].append(mod_effect)
        
        # Update neurons
        for neuron in self.all_neurons:
            if not neuron.is_active:
                continue
            
            external_input = external_inputs.get(neuron.id, 0.0)
            
            neuron.update(
                synaptic_inputs=neuron_synaptic_inputs[neuron.id],
                modulatory_inputs=neuron_modulatory_inputs[neuron.id],
                external_input=external_input,
                neuromodulators=self.neuromodulators,
                dt=self.params.dt
            )
        
        # Update synapses
        for synapse in self.synapses:
            if synapse.integrity <= 0:
                continue
            
            pre_neuron = self.all_neurons[synapse.pre_id]
            post_neuron = self.all_neurons[synapse.post_id]
            
            if pre_neuron.is_active and post_neuron.is_active:
                synapse.update(
                    pre_state=pre_neuron.trinary_state,
                    post_state=post_neuron.trinary_state,
                    neuromodulators=self.neuromodulators,
                    dt=self.params.dt
                )
        
        # Update neuromodulators (decay)
        for key in self.neuromodulators:
            baseline = getattr(self.params, f'{key}_baseline')
            self.neuromodulators[key] += (
                (baseline - self.neuromodulators[key]) * 
                self.params.neuromod_decay_rate * self.params.dt / 100.0
            )
        
        # Structural plasticity
        self._apply_structural_plasticity()
        
        # Update time
        self.time += self.params.dt
        self.step_count += 1
    
    def _apply_structural_plasticity(self):
        """Apply synapse formation and death"""
        # Remove dead synapses
        self.synapses = [s for s in self.synapses 
                        if s.integrity > self.params.synapse_integrity_threshold]
        
        # Synapse formation (rare)
        if random.random() < self.params.synapse_formation_prob:
            active_neurons = [n for n in self.all_neurons if n.is_active]
            if len(active_neurons) >= 2:
                pre = random.choice(active_neurons)
                post = random.choice(active_neurons)
                
                # Check constraints
                if (pre.id != post.id and 
                    not (pre.type == NeuronType.OUTPUT and post.type == NeuronType.INPUT)):
                    # Check if synapse doesn't already exist
                    exists = any(s.pre_id == pre.id and s.post_id == post.id 
                               for s in self.synapses)
                    if not exists:
                        new_synapse = Synapse(pre.id, post.id, self.params)
                        self.synapses.append(new_synapse)
    
    def set_input_states(self, states: List[int]):
        """
        Set states of input neurons
        
        Args:
            states: List of trinary states (-1, 0, 1)
        """
        for i, state in enumerate(states[:len(self.input_neurons)]):
            self.input_neurons[i].set_state(state)
    
    def get_output_states(self) -> List[int]:
        """Get current states of output neurons"""
        return [n.trinary_state for n in self.output_neurons if n.is_active]
    
    def modulate(self, neuromodulator: str, level: float):
        """
        Adjust neuromodulator level
        
        Args:
            neuromodulator: Name of neuromodulator
            level: New level (0.0 to 1.0)
        """
        if neuromodulator in self.neuromodulators:
            self.neuromodulators[neuromodulator] = max(0.0, min(1.0, level))
    
    def to_dict(self) -> dict:
        """Convert network to dictionary for JSON serialization"""
        return {
            'parameters': asdict(self.params),
            'neurons': {
                'input': [n.to_dict() for n in self.input_neurons],
                'hidden': [n.to_dict() for n in self.hidden_neurons],
                'output': [n.to_dict() for n in self.output_neurons]
            },
            'synapses': [s.to_dict() for s in self.synapses],
            'neuromodulators': self.neuromodulators,
            'time': self.time,
            'step_count': self.step_count
        }


# =============================================================================
# JSON SAVE/LOAD FUNCTIONS
# =============================================================================

def save_network(network: NeuraxonNetwork, filename: str):
    """
    Save network to JSON file
    
    Args:
        network: NeuraxonNetwork instance
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(network.to_dict(), f, indent=2)
    print(f"Network saved to {filename}")


def load_network(filename: str) -> NeuraxonNetwork:
    """
    Load network from JSON file
    
    Args:
        filename: Input filename
    
    Returns:
        Reconstructed NeuraxonNetwork
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Reconstruct parameters
    params = NetworkParameters(**data['parameters'])
    
    # Create network
    network = NeuraxonNetwork(params)
    
    # Restore neuron states
    for neuron_data in data['neurons']['input']:
        neuron = network.input_neurons[neuron_data['id'] % len(network.input_neurons)]
        neuron.membrane_potential = neuron_data['membrane_potential']
        neuron.trinary_state = neuron_data['trinary_state']
        neuron.health = neuron_data['health']
        neuron.is_active = neuron_data['is_active']
    
    # Similar for hidden and output neurons
    for neuron_data in data['neurons']['hidden']:
        idx = neuron_data['id'] - len(network.input_neurons)
        if 0 <= idx < len(network.hidden_neurons):
            neuron = network.hidden_neurons[idx]
            neuron.membrane_potential = neuron_data['membrane_potential']
            neuron.trinary_state = neuron_data['trinary_state']
            neuron.health = neuron_data['health']
            neuron.is_active = neuron_data['is_active']
    
    for neuron_data in data['neurons']['output']:
        idx = neuron_data['id'] - len(network.input_neurons) - len(network.hidden_neurons)
        if 0 <= idx < len(network.output_neurons):
            neuron = network.output_neurons[idx]
            neuron.membrane_potential = neuron_data['membrane_potential']
            neuron.trinary_state = neuron_data['trinary_state']
            neuron.health = neuron_data['health']
            neuron.is_active = neuron_data['is_active']
    
    # Restore synapse states
    network.synapses = []
    for syn_data in data['synapses']:
        synapse = Synapse(syn_data['pre_id'], syn_data['post_id'], params)
        synapse.w_fast = syn_data['w_fast']
        synapse.w_slow = syn_data['w_slow']
        synapse.w_meta = syn_data['w_meta']
        synapse.is_silent = syn_data['is_silent']
        synapse.is_modulatory = syn_data['is_modulatory']
        synapse.integrity = syn_data['integrity']
        network.synapses.append(synapse)
    
    # Restore neuromodulators
    network.neuromodulators = data['neuromodulators']
    network.time = data['time']
    network.step_count = data['step_count']
    
    print(f"Network loaded from {filename}")
    return network



if __name__ == "__main__":    
    print("="*70)
    print("NEURAXON - Bio-Inspired Neural Network")
    print("="*70)
    
    # Create network with default parameters
    print("\n1. Creating network...")
    params = NetworkParameters()
    network = NeuraxonNetwork(params)
    
    print(f"   - Input neurons: {len(network.input_neurons)}")
    print(f"   - Hidden neurons: {len(network.hidden_neurons)}")
    print(f"   - Output neurons: {len(network.output_neurons)}")
    print(f"   - Total synapses: {len(network.synapses)}")
    
    # Set input states
    print("\n2. Setting input states...")
    input_pattern = [1, -1, 0, 1, -1]
    network.set_input_states(input_pattern)
    print(f"   Input pattern: {input_pattern}")
    
    # Run simulation
    print("\n3. Running simulation...")
    for step in range(10):
        network.simulate_step()
        
        if step % 20 == 0:
            outputs = network.get_output_states()
            print(f"   Step {step}: Outputs = {outputs}")
    
    # Modulate network
    print("\n4. Testing neuromodulation...")
    network.modulate('dopamine', 0.8)
    print(f"   Dopamine level set to 0.8")
    
    # Save network
    print("\n5. Saving network...")
    save_network(network, "neuraxon_network.json")    