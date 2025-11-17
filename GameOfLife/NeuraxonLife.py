# Neuraxon Game of Life v.1.0 (Research Version): A Trinary Bioinspired Neural Unit Implementation of initial life dynamics
# Based on the Paper "Neuraxon: A New Neural Growth & Computation Blueprint" by David Vivancos https://vivancos.com/  & Jose Sanchez  https://josesanchezgarcia.com/
# https://www.researchgate.net/publication/397331336_Neuraxon
# Play the Lite Version of the Game of Life at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife

import os, sys, time, json, math, random, pathlib
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp
import numpy as np
from collections import deque, defaultdict
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
try:
    import pygame
except Exception as e:
    raise RuntimeError("This program requires pygame. Install with: pip install pygame") from e

@dataclass
class NetworkParameters:
    """
    A dataclass holding all configurable parameters for the Neuraxon network and the simulation environment.
    This centralized approach allows for easy serialization, loading, and modification of network behavior,
    reflecting the biologically-plausible parameters discussed throughout the Neuraxon paper.
    """
    # --- General Network Architecture ---
    network_name: str = "Neuraxon NxEr"
    num_input_neurons: int = 3  # Number of neurons dedicated to receiving external stimuli.
    num_hidden_neurons: int = 10 # Number of neurons in the processing layer.
    num_output_neurons: int = 4 # Number of neurons that produce the final output/action.
    
    # --- Temporal Dynamics & Simulation ---
    dt: float = 1.0  # Base time step for the simulation, can be dynamically adjusted.
    min_dt: float = 0.1  # The minimum time step for adaptive time-stepping.
    max_dt: float = 2.0  # The maximum time step for adaptive time-stepping.
    activity_threshold: float = 0.5 # A general threshold used for various activity checks.
    simulation_steps: int = 30 # Number of internal network updates per single game world tick.

    # --- Network Topology ---
    connection_probability: float = 0.15 # Probability of forming random connections beyond the initial topology.
    small_world_k: int = 6  # The 'k' parameter for the Watts-Strogatz small-world network generator, defining initial local connectivity.
    small_world_rewire_prob: float = 0.15 # The probability of rewiring an edge in the small-world model, creating shortcuts.
    preferential_attachment: bool = True # If true, rewiring prefers to connect to high-degree nodes, creating a scale-free-like topology.
    
    # --- Core Neuron Properties (Neuraxon) ---
    membrane_time_constant: float = 20.0 # Tau (τ) in the neuron's differential equation, governing the decay rate of the membrane potential (Section 2).
    firing_threshold_excitatory: float = 0.9 # The threshold (θ₁) for the membrane potential to enter the excitatory (+1) state (Section 2).
    firing_threshold_inhibitory: float = -0.9 # The threshold (θ₂) for the membrane potential to enter the inhibitory (-1) state (Section 2).
    adaptation_rate: float = 0.05 # Governs the strength of spike-frequency adaptation, a negative feedback mechanism.
    spontaneous_firing_rate: float = 0.02 # The base rate of self-generated, non-stimulus-driven activity (Section 6).
    neuron_health_decay: float = 0.001 # The rate at which a neuron's health degrades due to inactivity or energy depletion.
    
    # --- Dendritic Branch Properties ---
    num_dendritic_branches: int = 3 # Each neuron has multiple dendritic compartments for non-linear input integration.
    branch_threshold: float = 0.6 # The threshold for a dendritic branch to generate a local dendritic spike (Section 1).
    plateau_decay: float = 500.0 # The decay rate of the sustained plateau potential following a dendritic spike.

    # --- Synaptic Properties & Plasticity (Section 3) ---
    tau_fast: float = 5.0  # Time constant for the fast synaptic weight component (w_fast), modeling fast ionotropic receptors (e.g., AMPA).
    tau_slow: float = 50.0  # Time constant for the slow synaptic weight component (w_slow), modeling slow ionotropic receptors (e.g., NMDA).
    tau_meta: float = 1000.0 # Time constant for the metabotropic weight component (w_meta), modeling slow neuromodulatory effects.
    tau_ltp: float = 15.0 # Time constant for the potentiation (LTP) trace in the STDP rule.
    tau_ltd: float = 35.0 # Time constant for the depression (LTD) trace in the STDP rule.
    
    # --- Synaptic Weight Initialization Ranges ---
    w_fast_init_min: float = -1.0
    w_fast_init_max: float = 1.0
    w_slow_init_min: float = -0.5
    w_slow_init_max: float = 0.5
    w_meta_init_min: float = -0.3
    w_meta_init_max: float = 0.3
    
    # --- Learning and Plasticity Rules (Section 4) ---
    learning_rate: float = 0.01 # The learning rate (η) governing the magnitude of STDP weight updates.
    stdp_window: float = 20.0 # The effective time window for spike-timing-dependent plasticity.
    learning_rate_mod: float = 1.0 # A factor that can modulate the learning rate, often influenced by neuromodulators.
    plasticity_threshold: float = 0.5 # An activity threshold that must be met for plasticity to occur.
    associativity_strength: float = 0.1 # The strength of plasticity influence from neighboring synapses sharing the same presynaptic neuron.
    
    # --- Structural Plasticity ---
    synapse_integrity_threshold: float = 0.1 # If a synapse's integrity falls below this, it is a candidate for pruning.
    synapse_formation_prob: float = 0.02 # Probability per step of forming a new synapse in the network.
    synapse_death_prob: float = 0.01 # Probability per step of a synapse losing integrity due to inactivity.
    neuron_death_threshold: float = 0.1 # Health threshold below which a neuron is a candidate for removal.
    
    # --- Neuromodulation (Section 1 & 8) ---
    # Baseline levels for global neuromodulators.
    dopamine_baseline: float = 0.12
    # Dual-receptor affinity model: high affinity for tonic (sustained) signals, low affinity for phasic (burst) signals.
    dopamine_high_affinity_threshold: float = 0.01 # Models nM-range receptors for tonic background levels.
    dopamine_low_affinity_threshold: float = 1.0  # Models µM-range receptors for high-concentration phasic events.
    serotonin_baseline: float = 0.12
    serotonin_high_affinity_threshold: float = 0.01
    serotonin_low_affinity_threshold: float = 1.0
    acetylcholine_baseline: float = 0.12
    acetylcholine_high_affinity_threshold: float = 0.01
    acetylcholine_low_affinity_threshold: float = 1.0
    norepinephrine_baseline: float = 0.12
    norepinephrine_high_affinity_threshold: float = 0.01
    norepinephrine_low_affinity_threshold: float = 1.0
    neuromod_decay_rate: float = 0.1 # The rate at which neuromodulator concentrations decay back to their baseline.
    diffusion_rate: float = 0.05 # The rate at which neuromodulators diffuse spatially in the environment grid.
    
    # --- Oscillators & Synchronization (Section 7) ---
    # Frequencies for the three nested global oscillators, modeling cross-frequency coupling.
    oscillator_low_freq: float = 0.05  # e.g., Delta/Theta band for large temporal windows.
    oscillator_mid_freq: float = 0.5   # e.g., Alpha/Beta band.
    oscillator_high_freq: float = 4.0  # e.g., Gamma band for fast, local processing.
    oscillator_strength: float = 0.15 # The amplitude of the global oscillatory drive signal.
    
    # --- Energy Metabolism ---
    energy_baseline: float = 100.0 # The starting energy level of a neuron.
    firing_energy_cost: float = 5.0  # The energy consumed when a neuron is in an active state (+1 or -1).
    plasticity_energy_cost: float = 10.0 # The energy consumed by synaptic weight changes.
    metabolic_rate: float = 1.0 # The base rate of energy consumption for maintaining neuron function.
    recovery_rate: float = 0.5 # The rate at which energy is replenished during inactivity.
    
    # --- Homeostasis ---
    target_firing_rate: float = 0.1 # The desired average firing rate for neurons, maintained by homeostatic plasticity.
    homeostatic_plasticity_rate: float = 0.001 # The rate at which firing thresholds are adjusted to meet the target rate.
    
    # --- Aigarth Hybridization (Section 8) ---
    itu_circle_radius: int = 8 # The number of neurons that form a single Intelligent Tissue Unit (ITU).
    evolution_interval: int = 1000 # The number of simulation steps between evolutionary cycles (mutation and selection).
    # Weights for the components of the evolutionary fitness function.
    fitness_temporal_weight: float = 0.4 
    fitness_energy_weight: float = 0.3
    fitness_pattern_weight: float = 0.3
    
    # --- Miscellaneous ---
    phase_coupling_strength: float = 0.1 # The strength of the coupling between an individual neuron's oscillator and the global oscillator.
    max_axonal_delay: float = 10.0 # The maximum signal propagation delay for a synapse.

# Defines the core types within the model, aligning with the paper's terminology.
class NeuronType(Enum): INPUT = "input"; HIDDEN = "hidden"; OUTPUT = "output"
class SynapseType(Enum): IONOTROPIC_FAST = "ionotropic_fast"; IONOTROPIC_SLOW = "ionotropic_slow"; METABOTROPIC = "metabotropic"; SILENT = "silent"
class TrinaryState(Enum): INHIBITORY = -1; NEUTRAL = 0; EXCITATORY = 1

class DendriticBranch:
    """
    Models a single dendritic compartment of a Neuraxon.
    This is crucial for implementing non-linear input integration and local plasticity,
    as described in Section 1 of the Neuraxon paper. Dendritic spikes provide a "third, intermediate
    signaling state" that can trigger plasticity without causing a full somatic spike.
    """
    def __init__(self, branch_id: int, parent_neuron_id: int, params: NetworkParameters):
        self.branch_id = branch_id
        self.parent_neuron_id = parent_neuron_id
        self.params = params
        self.branch_potential = 0.0  # The local membrane potential of this specific branch.
        self.branch_threshold = params.branch_threshold # Threshold to trigger a local dendritic spike.
        self.plateau_potential = 0.0 # A sustained depolarization following a local spike, important for plasticity.
        self.local_spike_history = deque(maxlen=10) # A short-term memory of local spiking activity.
    
    def integrate_inputs(self, synaptic_inputs: List[float], dt: float) -> float:
        """
        Integrates incoming synaptic signals for this branch. If the integrated potential
        crosses a threshold, it generates a local dendritic spike and a plateau potential.
        """
        # If there are no inputs, potentials decay towards their resting state.
        if not synaptic_inputs:
            self.plateau_potential += dt / self.params.plateau_decay * (-self.plateau_potential)
            self.branch_potential += dt / (self.params.membrane_time_constant * 0.5) * (-self.branch_potential)
            return self.branch_potential + self.plateau_potential
        
        # Sum inputs and apply a non-linear saturation function (tanh).
        branch_signal = math.tanh(sum(synaptic_inputs))
        
        # If the local signal is strong enough, trigger a plateau potential (a form of local spike).
        if abs(branch_signal) > self.branch_threshold:
            self.plateau_potential = branch_signal * 0.8
            
        # Update the transient branch potential.
        tau_branch = max(1.0, self.params.membrane_time_constant * 0.3)
        self.branch_potential += dt / tau_branch * (branch_signal - self.branch_potential)
        
        # Record the occurrence of a local spike for plasticity calculations.
        self.local_spike_history.append(1.0 if abs(self.branch_potential) > self.branch_threshold else 0.0)
        
        # The total output to the soma is the sum of the transient potential and the sustained plateau.
        return self.branch_potential + self.plateau_potential
    
    def get_local_ca_influx(self) -> float:
        """
        Estimates the local calcium influx based on recent dendritic spike activity.
        This serves as a biological proxy for signals that trigger local synaptic plasticity.
        """
        return sum(self.local_spike_history) / len(self.local_spike_history) if len(self.local_spike_history) >= 3 else 0.0
    
    def to_dict(self) -> dict:
        """Serializes the branch's current state into a dictionary for saving."""
        return {'branch_id': self.branch_id, 'branch_potential': self.branch_potential, 'plateau_potential': self.plateau_potential, 'threshold': self.branch_threshold}

class Synapse:
    """
    Implements the Neuraxon synapse, which is an active computational unit, not a passive weight.
    As per Section 3 ("Integrated Neuron-Synapse Computation"), it features three distinct weights
    (w_fast, w_slow, w_meta) to model different biological receptor dynamics and timescales.
    It manages its own plasticity based on local activity and global neuromodulatory signals.
    """
    def __init__(self, pre_id: int, post_id: int, params: NetworkParameters):
        self.pre_id = pre_id
        self.post_id = post_id
        self.params = params
        
        # The three-component weight system, central to the Neuraxon model.
        self.w_fast = random.uniform(params.w_fast_init_min, params.w_fast_init_max) # Models fast ionotropic channels (e.g., AMPA).
        self.w_slow = random.uniform(params.w_slow_init_min, params.w_slow_init_max) # Models slow ionotropic channels (e.g., NMDA).
        self.w_meta = random.uniform(params.w_meta_init_min, params.w_meta_init_max) # Models slow, modulatory metabotropic effects.
        
        self.is_silent = random.random() < 0.1 # Models "silent synapses" which exist structurally but are not functional until unsilenced by plasticity (Section 1).
        self.is_modulatory = random.random() < 0.2 # If true, this synapse's primary effect is through w_meta, modulating the postsynaptic neuron's properties.
        
        self.integrity = 1.0 # A health metric for the synapse, used in structural plasticity.
        self.axonal_delay = random.uniform(0, params.max_axonal_delay) # Signal propagation delay.
        
        # Traces used for calculating STDP.
        self.pre_trace = 0.0  # Trace of presynaptic excitatory activity for LTP.
        self.post_trace = 0.0 # Trace of postsynaptic excitatory activity for LTP.
        self.pre_trace_ltd = 0.0 # A separate presynaptic trace with a longer time constant for LTD.
        
        self.learning_rate_mod = 1.0 # The learning rate, dynamically modulated by neuromodulators like dopamine.
        self.plasticity_threshold = params.plasticity_threshold
        self.associative_strength = 0.0 # A temporary store for associative plasticity effects.
        self.neighbor_synapses = [] # A list of other synapses originating from the same presynaptic neuron.
        self.potential_delta_w = 0.0 # The calculated weight change, which is applied in a separate step.
        self.synapse_type = self._determine_type()
    
    def _determine_type(self) -> SynapseType:
        """Dynamically determines the functional type of the synapse based on its current state."""
        if self.is_silent: return SynapseType.SILENT
        if self.is_modulatory: return SynapseType.METABOTROPIC
        # If not silent or modulatory, its type is determined by the dominant ionotropic weight component.
        return SynapseType.IONOTROPIC_FAST if abs(self.w_fast) >= abs(self.w_slow) else SynapseType.IONOTROPIC_SLOW
    
    def compute_input(self, pre_state: int, current_time: float) -> Tuple[float, float]:
        """Calculates the postsynaptic potential contribution of this synapse."""
        if self.is_silent: return 0.0, 0.0
        # A simple model where signal strength is attenuated by delay.
        delay_factor = max(0.0, 1.0 - self.axonal_delay / 10.0)
        # The total ionotropic effect is the sum of the fast and slow components.
        w = self.w_fast + self.w_slow
        # Returns the immediate and delayed portions of the signal.
        return w * pre_state * delay_factor, w * pre_state * (1.0 - delay_factor)
    
    def calculate_delta_w(self, pre_state: int, post_state: int, neuromodulators: Dict[str, float], dt: float) -> float:
        """
        Calculates the potential change in synaptic weight (Δw) using a trinary STDP rule.
        This implements the core of the plasticity mechanism described in Section 4. The change
        depends on the coincidence of presynaptic and postsynaptic trinary states and is
        gated by neuromodulators (specifically dopamine here).
        """
        # Update activity traces based on current presynaptic state.
        self.pre_trace += (-self.pre_trace / self.params.tau_ltp + (1 if pre_state == 1 else 0)) * dt
        self.pre_trace_ltd += (-self.pre_trace_ltd / self.params.tau_ltd + (1 if pre_state == 1 else 0)) * dt
        self.post_trace += (-self.post_trace / self.params.tau_ltp + (1 if post_state == 1 else 0)) * dt
        
        # Dopamine level modulates the learning rate based on the dual-affinity model.
        da = neuromodulators.get('dopamine', 0.5)
        da_high = 1.0 if da > self.params.dopamine_high_affinity_threshold else 0.0 # Tonic response
        da_low = 1.0 if da > self.params.dopamine_low_affinity_threshold else 1.0 # Phasic response (in this simplified model, it's always on)
        self.learning_rate_mod = 1.0 + (da_high * 0.5) + (da_low * 0.2)
        
        # Hebbian rule: pre-excitatory (+1) followed by post-excitatory (+1) causes potentiation (LTP).
        if pre_state == 1 and post_state == 1:
            return self.params.learning_rate * self.learning_rate_mod * da_high * self.pre_trace
        # Anti-Hebbian rule: pre-excitatory (+1) followed by post-inhibitory (-1) causes depression (LTD).
        elif pre_state == 1 and post_state == -1:
            return -self.params.learning_rate * self.learning_rate_mod * da_low * self.pre_trace_ltd
        return 0.0
    
    def apply_update(self, dt: float, neuromodulators: Dict[str, float], neighbor_deltas: List[float] = None):
        """
        Applies the calculated weight change (Δw) to the three weight components.
        This function solves the differential equations for the weights from Section 3,
        and includes the associativity term from Section 4.
        """
        delta_w = self.potential_delta_w
        
        # Associativity: plasticity at this synapse is influenced by plasticity at neighboring synapses.
        if neighbor_deltas:
            delta_w += self.params.associativity_strength * sum(dw / (i + 1) for i, dw in enumerate(neighbor_deltas[:3]))
        
        # Define the targets for the fast and slow weights based on the activity traces.
        h_fast = self.pre_trace
        h_slow = 0.5 * self.pre_trace + 0.5 * self.post_trace
        
        # Update each weight component according to its differential equation and timescale.
        self.w_fast = max(-1.0, min(1.0, self.w_fast + dt / self.params.tau_fast * (-self.w_fast + h_fast + delta_w * 0.3)))
        self.w_slow = max(-1.0, min(1.0, self.w_slow + dt / self.params.tau_slow * (-self.w_slow + h_slow + delta_w * 0.1)))
        
        # The metabotropic weight's update is modulated by serotonin.
        serotonin = neuromodulators.get('serotonin', 0.5)
        meta_target = delta_w * 0.05 * (1.0 if serotonin > 0.5 else 0.5)
        self.w_meta = max(-0.5, min(0.5, self.w_meta + dt / self.params.tau_meta * (-self.w_meta + meta_target)))
        
        # Update synaptic integrity for structural plasticity. Active synapses become stronger, inactive ones decay.
        activity = abs(self.w_fast) + abs(self.w_slow)
        self.integrity = min(1.0, self.integrity + 0.001 * dt * activity) if activity >= 0.01 else self.integrity - self.params.synapse_death_prob * dt

        # A mechanism for "unsilencing" a silent synapse based on correlated activity.
        if self.is_silent and self.pre_trace > 0.5 and random.random() < 0.01:
            self.is_silent = False
            
        # Re-determine the functional type of the synapse after the updates.
        self.synapse_type = self._determine_type()
    
    def get_modulatory_effect(self) -> float:
        """Returns the modulatory signal from this synapse, driven by its metabotropic weight."""
        return self.w_meta * 0.5 if self.is_modulatory else 0.0
    
    def to_dict(self) -> dict:
        """Serializes the synapse's state into a dictionary."""
        return {'pre_id': self.pre_id, 'post_id': self.post_id, 'w_fast': self.w_fast, 'w_slow': self.w_slow, 'w_meta': self.w_meta, 'is_silent': self.is_silent, 'is_modulatory': self.is_modulatory, 'integrity': self.integrity, 'axonal_delay': self.axonal_delay, 'learning_rate_mod': self.learning_rate_mod, 'synapse_type': self.synapse_type.value, 'potential_delta_w': self.potential_delta_w}

class Neuraxon:
    """
    The core computational unit of the model, as detailed in the paper.
    It's a trinary-state (-1, 0, 1), continuous-time neuron that integrates inputs
    non-linearly through dendritic branches. It features intrinsic dynamics like adaptation,
    spontaneous activity, and metabolic energy management.
    """
    def __init__(self, neuron_id: int, neuron_type: NeuronType, params: NetworkParameters):
        self.id = neuron_id
        self.type = neuron_type
        self.params = params
        
        # Core state variables of the neuron.
        self.membrane_potential = 0.0 # The internal state s(t) from the differential equation in Section 2.
        self.trinary_state = TrinaryState.NEUTRAL.value # The neuron's output state: -1, 0, or 1.
        self.adaptation = 0.0 # A slow variable that provides negative feedback to firing, preventing hyperactivity.
        self.autoreceptor = 0.0 # A variable modeling self-regulation via autoreceptors.
        self.health = 1.0 # A health metric used for structural plasticity (neuron death).
        self.is_active = True # A flag to enable or disable the neuron.
        
        # Structural and metabolic components.
        self.dendritic_branches = [DendriticBranch(i, neuron_id, params) for i in range(params.num_dendritic_branches)]
        self.energy_level = params.energy_baseline # Models the neuron's metabolic energy.
        
        # Temporal and oscillatory properties.
        self.last_firing_time = -1000.0
        self.phase = random.random() * 2 * math.pi # The current phase of the neuron's intrinsic oscillator.
        self.natural_frequency = random.uniform(0.5, 2.0) # The neuron's preferred oscillation frequency.
        self.state_history = deque(maxlen=50) # A history of its recent output states.
        self.intrinsic_timescale = params.membrane_time_constant # The effective membrane time constant, which can be dynamically adjusted.
        
        # Properties for Aigarth Hybridization (Section 8).
        self.circle_id = None # The ID of the ITU this neuron belongs to.
        self.fitness_score = 0.0 # Its fitness score for evolutionary selection.
    
    def _nonlinear_dendritic_integration(self, synaptic_inputs: List[float], modulatory_inputs: List[float], dt: float) -> Tuple[float, List[float]]:
        """
        Sums the outputs from all dendritic branches to get the total synaptic drive.
        This two-stage, non-linear integration is a key feature distinguishing Neuraxons from simpler models.
        """
        branch_outputs = []
        total_synaptic = 0.0
        for i, branch in enumerate(self.dendritic_branches):
            # Distribute the total list of synaptic inputs among the branches.
            branch_syn_inputs = synaptic_inputs[i::len(self.dendritic_branches)]
            branch_out = branch.integrate_inputs(branch_syn_inputs, dt)
            branch_outputs.append(branch_out)
            total_synaptic += branch_out
        # Modulatory inputs can multiplicatively scale the total synaptic drive.
        return total_synaptic * (1.0 + sum(modulatory_inputs) * 0.2), branch_outputs
    
    def _update_phase_oscillator(self, dt: float, global_osc: float):
        """
        Updates the neuron's internal phase, coupling it to the global network oscillation.
        This mechanism is essential for achieving the time-centric synchronization described in Section 7 of the Neuraxon paper.
        """
        self.phase += 2 * math.pi * self.natural_frequency * dt + self.params.phase_coupling_strength * math.sin(global_osc - self.phase) * dt
        self.phase %= 2 * math.pi
    
    def _update_energy(self, activity: float, plasticity_cost: float, dt: float):
        """Models the consumption and replenishment of the neuron's metabolic energy."""
        if not self.is_active: return
        # Energy consumption is proportional to firing activity and plasticity.
        consumption = self.params.metabolic_rate * (self.params.firing_energy_cost * activity + self.params.plasticity_energy_cost * plasticity_cost) * dt
        # Energy recovers during periods of inactivity.
        recovery = self.params.recovery_rate * (1.0 - activity) * dt
        self.energy_level = max(0.0, min(self.params.energy_baseline * 1.5, self.energy_level + recovery - consumption))
        # Critically low energy degrades health and impairs function.
        if self.energy_level < 10.0:
            self.health -= self.params.neuron_health_decay * dt * 2.0
            self.membrane_potential *= 0.9
    
    def _update_autocorrelation(self):
        """
        Dynamically adjusts the neuron's intrinsic timescale based on the autocorrelation of its recent activity.
        This implements the "autocorrelation window" concept from Section 6, allowing a neuron to adapt its
        own temporal integration properties. A more rhythmic neuron will have a longer timescale.
        """
        if len(self.state_history) >= 10:
            states = list(self.state_history)
            autocorr = np.corrcoef(states[:-1], states[1:])[0, 1]
            if not np.isnan(autocorr):
                self.intrinsic_timescale = self.params.membrane_time_constant * (1.0 + abs(autocorr) * 2.0)
    
    def update(self, synaptic_inputs: List[float], modulatory_inputs: List[float], external_input: float, neuromodulators: Dict[str, float], dt: float, global_osc: float):
        """
        Executes one update step for the neuron. This function is an implementation of the
        continuous-time dynamics described by the differential equation in Section 2 of the paper.
        """
        if not self.is_active or self.energy_level <= 0: return

        self._update_phase_oscillator(dt, global_osc)
        
        # 1. Integrate all inputs non-linearly.
        total_synaptic, branch_outputs = self._nonlinear_dendritic_integration(synaptic_inputs, modulatory_inputs, dt)
        
        # 2. Add spontaneous, self-generated activity (Section 6).
        spont_prob = self.params.spontaneous_firing_rate * dt * (1.0 + math.cos(self.phase) * 0.3)
        spontaneous = random.uniform(-0.5, 0.5) if random.random() < spont_prob else 0.0
        
        # 3. Apply the effects of global neuromodulators.
        acetylcholine = neuromodulators.get('acetylcholine', 0.5)
        norepi = neuromodulators.get('norepinephrine', 0.5)
        threshold_mod = (acetylcholine - 0.5) * 0.5 + sum(modulatory_inputs) * 0.3 # Modulates firing threshold.
        gain = 1.0 + (norepi - 0.5) * 0.4 # Modulates the gain (responsiveness) of the neuron.
        
        # 4. Calculate the total drive affecting the membrane potential.
        drive = (total_synaptic + external_input + spontaneous) * gain
        tau_eff = max(1.0, self.intrinsic_timescale)
        prev_state = self.trinary_state
        
        # 5. Update the membrane potential using the forward Euler method on the core differential equation:
        # τ * ds/dt = -s + drive - adaptation
        self.membrane_potential += dt / tau_eff * (-self.membrane_potential + drive - self.adaptation)
        
        # 6. Update the slow adaptation and autoreceptor variables.
        self.adaptation += dt / 100.0 * (-self.adaptation + 0.1 * abs(self.trinary_state))
        self.autoreceptor += dt / 200.0 * (-self.autoreceptor + 0.2 * self.trinary_state)
        
        # 7. Apply dynamic thresholds to determine the new trinary output state.
        theta_exc = self.params.firing_threshold_excitatory - threshold_mod - 0.1 * self.autoreceptor
        theta_inh = self.params.firing_threshold_inhibitory - threshold_mod + 0.1 * self.autoreceptor
        
        if self.membrane_potential > theta_exc: self.trinary_state = TrinaryState.EXCITATORY.value
        elif self.membrane_potential < theta_inh: self.trinary_state = TrinaryState.INHIBITORY.value
        else: self.trinary_state = TrinaryState.NEUTRAL.value
        
        # 8. Update internal history, health, and energy based on the new state.
        self.state_history.append(self.trinary_state)
        self._update_autocorrelation()
        activity_level = abs(self.trinary_state)
        self.health = min(1.0, self.health + 0.0005 * dt) if activity_level >= 0.01 else self.health - self.params.neuron_health_decay * dt
        self._update_energy(activity_level, abs(self.trinary_state - prev_state) * 0.1, dt)
        
        # Hidden neurons can "die" (become inactive) if their health or energy is critically low.
        if self.type == NeuronType.HIDDEN and (self.health < self.params.neuron_death_threshold or self.energy_level < 1.0):
            if random.random() < 0.001: self.is_active = False
    
    def set_state(self, state: int):
        """
        Directly sets the neuron's trinary state and membrane potential.
        Used primarily for clamping input neurons to external values.
        """
        if state in [-1, 0, 1]:
            self.trinary_state = state
            self.membrane_potential = state * self.params.firing_threshold_excitatory
    
    def to_dict(self) -> dict:
        """Serializes the neuron's state into a dictionary."""
        return {'id': self.id, 'type': self.type.value, 'membrane_potential': self.membrane_potential, 'trinary_state': self.trinary_state, 'adaptation': self.adaptation, 'health': self.health, 'is_active': self.is_active, 'energy_level': self.energy_level, 'phase': self.phase, 'natural_frequency': self.natural_frequency, 'intrinsic_timescale': self.intrinsic_timescale, 'circle_id': self.circle_id, 'fitness_score': self.fitness_score, 'dendritic_branches': [b.to_dict() for b in self.dendritic_branches]}

class ITUCircle:
    """
    Implements an Intelligent Tissue Unit (ITU) circle, a concept from the Aigarth hybridization (Section 8 of the Neuraxon paper).
    This class manages a group of neurons that are subject to evolutionary pressures. It calculates
    a collective fitness score and applies mutation and selection (pruning) to its member neurons.
    """
    def __init__(self, circle_id: int, neurons: List[Neuraxon], params: NetworkParameters):
        self.circle_id = circle_id
        self.neurons = neurons
        self.params = params
        self.fitness_history = []
        self.mutation_rate = 0.01
        for n in neurons: n.circle_id = circle_id
    
    def compute_fitness(self, network_activity: List[float], temporal_sync: float, energy_efficiency: float) -> float:
        """Calculates the fitness of this ITU based on its performance on multiple objectives."""
        fitness = self.params.fitness_pattern_weight * (np.mean(network_activity) if network_activity else 0.0) + self.params.fitness_temporal_weight * temporal_sync + self.params.fitness_energy_weight * energy_efficiency
        self.fitness_history.append(fitness)
        return fitness
    
    def mutate(self):
        """Applies small, random changes (mutations) to the parameters of the neurons in the circle."""
        for neuron in self.neurons:
            if random.random() < self.mutation_rate:
                # Mutating intrinsic properties allows the evolution of different dynamic behaviors.
                neuron.natural_frequency = max(0.1, min(5.0, neuron.natural_frequency + random.uniform(-0.2, 0.2)))
                neuron.intrinsic_timescale *= random.uniform(0.9, 1.1)
                if hasattr(neuron.params, 'firing_threshold_excitatory'):
                    neuron.params.firing_threshold_excitatory *= random.uniform(0.95, 1.05)
    
    def prune_unfit_neurons(self) -> List[int]:
        """
        Implements selection by "pruning" (deactivating) neurons whose fitness is consistently low
        compared to the circle's average. This is a form of structural optimization.
        """
        pruned_ids = []
        avg_fitness = np.mean(self.fitness_history[-10:]) if len(self.fitness_history) > 10 else 0.5
        for neuron in self.neurons:
            if neuron.fitness_score < avg_fitness * 0.3 and random.random() < 0.001:
                neuron.is_active = False
                pruned_ids.append(neuron.id)
        return pruned_ids

class NeuraxonNetwork:
    """
    The main class that orchestrates the entire network simulation. It manages the collections
    of neurons and synapses, global signals like neuromodulators and oscillators, and executes
    the simulation steps, including all forms of plasticity and evolution.
    """
    def __init__(self, params: Optional[NetworkParameters] = None):
        self.params = params or NetworkParameters()
        self.input_neurons: List[Neuraxon] = []
        self.hidden_neurons: List[Neuraxon] = []
        self.output_neurons: List[Neuraxon] = []
        self.all_neurons: List[Neuraxon] = []
        self.synapses: List[Synapse] = []
        self.itu_circles: List[ITUCircle] = [] # A list of evolutionary units.
        
        # Initialize global neuromodulator levels.
        self.neuromodulators = {mod: getattr(self.params, f'{mod}_baseline') for mod in ['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']}
        self.modulator_grid = np.full((10, 10, 4), 0.5) # A spatial grid for modeling modulator diffusion.
        
        self.time = 0.0
        self.step_count = 0
        self.total_energy_consumed = 0.0
        
        # Initialize random phase offsets for the three global oscillators.
        self.oscillator_phase_offsets = (random.random() * 2 * math.pi, random.random() * 2 * math.pi, random.random() * 2 * math.pi)
        self.activation_history = deque(maxlen=1000)
        self.branching_ratio = 1.0 # A measure of network criticality. A value near 1.0 is often optimal.
        
        # Construct the network.
        self._initialize_neurons()
        self._initialize_synapses()
        self._initialize_itu_circles()
    
    def _initialize_neurons(self):
        """Creates the populations of input, hidden, and output neurons."""
        nid = 0
        for _ in range(self.params.num_input_neurons):
            n = Neuraxon(nid, NeuronType.INPUT, self.params)
            self.input_neurons.append(n)
            self.all_neurons.append(n)
            nid += 1
        for _ in range(self.params.num_hidden_neurons):
            n = Neuraxon(nid, NeuronType.HIDDEN, self.params)
            self.hidden_neurons.append(n)
            self.all_neurons.append(n)
            nid += 1
        for _ in range(self.params.num_output_neurons):
            n = Neuraxon(nid, NeuronType.OUTPUT, self.params)
            self.output_neurons.append(n)
            self.all_neurons.append(n)
            nid += 1
    
    def _initialize_synapses(self):
        """
        Generates the initial synaptic connectivity using a Watts-Strogatz small-world model.
        This creates a network with high local clustering and short average path lengths, a
        topology common in biological brains and mentioned in the paper's abstract and Section 7 of the Neuraxon paper.
        """
        num_neurons = len(self.all_neurons)
        if num_neurons <= 1: return
        
        k = max(2, min(num_neurons - 1, int(self.params.small_world_k)))
        rewire_p = max(0.0, min(1.0, float(self.params.small_world_rewire_prob)))
        existing = set()
        neuron_degrees = defaultdict(int)
        
        # Step 1: Create a regular ring lattice where each neuron connects to its 'k' nearest neighbors.
        for idx, pre in enumerate(self.all_neurons):
            for offset in range(1, k // 2 + 1):
                j = (idx + offset) % num_neurons
                post = self.all_neurons[j]
                
                # Step 2: With probability 'rewire_p', rewire the connection to a new random target neuron.
                if random.random() < rewire_p:
                    # Optional: Use preferential attachment to create hubs ("rich get richer").
                    if self.params.preferential_attachment:
                        candidates = []; weights = []
                        for _ in range(10): # Sample a few potential new targets.
                            j_new = random.randrange(num_neurons)
                            cand = self.all_neurons[j_new]
                            if cand.id != pre.id:
                                candidates.append(cand)
                                # Weight candidates by their current degree, making hubs more attractive targets.
                                weights.append(neuron_degrees[cand.id] + 1)
                        if candidates: post = random.choices(candidates, weights=weights, k=1)[0]
                        else: # Fallback if no valid candidates are found.
                            for _ in range(6):
                                j_new = random.randrange(num_neurons)
                                cand = self.all_neurons[j_new]
                                if cand.id != pre.id:
                                    post = cand
                                    break
                    else: # Simple random rewiring.
                        for _ in range(6):
                            j_new = random.randrange(num_neurons)
                            cand = self.all_neurons[j_new]
                            if cand.id != pre.id:
                                post = cand
                                break
                
                # Create the synapse, avoiding self-connections and other invalid pairings.
                if pre.id == post.id or (pre.type == NeuronType.OUTPUT and post.type == NeuronType.INPUT) or (pre.id, post.id) in existing: continue
                syn = Synapse(pre.id, post.id, self.params)
                self.synapses.append(syn)
                existing.add((pre.id, post.id))
                neuron_degrees[pre.id] += 1
                neuron_degrees[post.id] += 1
                
        # Step 3: Add some purely random connections to ensure the graph is not fragmented.
        for pre in self.all_neurons:
            for post in self.all_neurons:
                if pre.id == post.id or (pre.type == NeuronType.OUTPUT and post.type == NeuronType.INPUT) or (pre.id, post.id) in existing: continue
                if random.random() < self.params.connection_probability * 0.25:
                    syn = Synapse(pre.id, post.id, self.params)
                    self.synapses.append(syn)
                    existing.add((pre.id, post.id))
                    neuron_degrees[pre.id] += 1
                    neuron_degrees[post.id] += 1
                    
        # After all synapses are created, find neighbors for the associativity rule.
        for s in self.synapses:
            s.neighbor_synapses = [ns for ns in self.synapses if ns.pre_id == s.pre_id and ns.post_id != s.post_id]
    
    def _initialize_itu_circles(self):
        """Groups hidden neurons into ITUs for the Aigarth evolutionary process."""
        if len(self.hidden_neurons) < self.params.itu_circle_radius * 2: return
        num_circles = max(1, len(self.hidden_neurons) // self.params.itu_circle_radius)
        neurons_per_circle = len(self.hidden_neurons) // num_circles
        for circle_idx in range(num_circles):
            start = circle_idx * neurons_per_circle
            end = start + neurons_per_circle
            self.itu_circles.append(ITUCircle(circle_idx, self.hidden_neurons[start:end], self.params))
    
    def _global_oscillatory_drive(self) -> float:
        """
        Generates a complex, global oscillatory signal by combining three sine waves of different frequencies.
        This models the concept of nested oscillations (Cross-Frequency Coupling) from Section 7 of the Neuraxon paper, providing a
        shared temporal context for the entire network.
        """
        t = self.time
        low = math.sin(2.0 * math.pi * self.params.oscillator_low_freq * t + self.oscillator_phase_offsets[0])
        mid = math.sin(2.0 * math.pi * self.params.oscillator_mid_freq * t + self.oscillator_phase_offsets[1])
        high = math.sin(2.0 * math.pi * self.params.oscillator_high_freq * t + self.oscillator_phase_offsets[2])
        # The combination creates a complex waveform that is more biologically realistic than a simple sine wave.
        return self.params.oscillator_strength * (low * 0.5 + low * mid * 0.3 + mid * high * 0.2)
    
    def _update_neuromodulator_diffusion(self, dt: float):
        """Simulates the spatial diffusion of neuromodulators across the environment grid."""
        for i, mod in enumerate(['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']):
            grid = self.modulator_grid[:, :, i]
            laplacian = np.zeros_like(grid)
            # A simple finite difference approximation of the 2D Laplacian operator to model diffusion.
            laplacian[1:-1, 1:-1] = grid[2:, 1:-1] + grid[:-2, 1:-1] + grid[1:-1, 2:] + grid[1:-1, :-2] - 4 * grid[1:-1, 1:-1]
            self.modulator_grid[:, :, i] += self.params.diffusion_rate * laplacian * dt
            # The global average level also decays back towards its baseline.
            base = getattr(self.params, f'{mod}_baseline')
            self.neuromodulators[mod] += (base - self.neuromodulators[mod]) * self.params.neuromod_decay_rate * dt / 100.0
    
    def _apply_homeostatic_plasticity(self):
        """
        A slow-acting plasticity rule that adjusts neuron excitability to maintain a target
        average firing rate, preventing states of pathological hyperactivity or silence.
        """
        if self.step_count % 100 != 0: return # This is a slow process, so it's only applied periodically.
        for neuron in self.all_neurons:
            if neuron.type == NeuronType.HIDDEN and len(neuron.state_history) > 0:
                recent_activity = sum(abs(s) for s in neuron.state_history) / len(neuron.state_history)
                # If firing too much, increase the firing threshold to make it harder to fire.
                if recent_activity > self.params.target_firing_rate * 1.2:
                    neuron.params.firing_threshold_excitatory += self.params.homeostatic_plasticity_rate
                # If firing too little, decrease the threshold to make it easier to fire.
                elif recent_activity < self.params.target_firing_rate * 0.8:
                    neuron.params.firing_threshold_excitatory -= self.params.homeostatic_plasticity_rate
    
    def _compute_branching_ratio(self):
        """
        Calculates the branching ratio (sigma), a measure of how activity propagates through the network.
        A value near 1.0 suggests the network is in a "critical" state, balanced between dying out and
        exploding, which is believed to be optimal for information processing.
        """
        if len(self.activation_history) < 2: return 1.0
        activations = list(self.activation_history)
        denominator = sum(activations[:-1])
        self.branching_ratio = max(0.1, min(10.0, sum(activations[1:]) / denominator if denominator != 0 else 1.0))
        return self.branching_ratio
    
    def simulate_step(self, external_inputs: Optional[Dict[int, float]] = None):
        """
        Executes one full, orchestrated simulation step for the entire network.
        The order of operations is critical: signal propagation -> neuron update -> plasticity calculation -> plasticity application.
        """
        external_inputs = external_inputs or {}
        
        # Use an adaptive time step that shrinks during periods of high activity variance for stability.
        dt = max(self.params.min_dt, min(self.params.max_dt, self.params.dt / (1.0 + np.var(list(self.activation_history)[-10:]) if len(self.activation_history) > 10 else self.params.dt)))
        
        # 1. Compute global signals for this time step.
        osc_drive = self._global_oscillatory_drive()
        
        # 2. Propagate signals through all synapses.
        syn_inputs = {n.id: [] for n in self.all_neurons}
        mod_inputs = {n.id: [] for n in self.all_neurons}
        delayed_inputs = defaultdict(list)
        for s in self.synapses:
            if s.integrity <= 0: continue
            pre = self.all_neurons[s.pre_id]
            if not pre.is_active: continue
            immediate, delayed = s.compute_input(pre.trinary_state, self.time)
            syn_inputs[s.post_id].append(immediate)
            if delayed > 0: delayed_inputs[s.post_id].append((delayed, self.time + s.axonal_delay))
            me = s.get_modulatory_effect()
            if me != 0: mod_inputs[s.post_id].append(me)
        
        # Add any delayed signals that are due to arrive in this time step.
        for post_id, delays in list(delayed_inputs.items()):
            for value, delivery_time in delays:
                if abs(delivery_time - self.time) < dt:
                    syn_inputs[post_id].append(value)
                    delayed_inputs[post_id].remove((value, delivery_time))

        # 3. Update the state of all neurons based on their integrated inputs.
        for n in self.all_neurons:
            if not n.is_active: continue
            ext = external_inputs.get(n.id, 0.0) + osc_drive
            n.update(syn_inputs[n.id], mod_inputs[n.id], ext, self.neuromodulators, dt, osc_drive)
            self.activation_history.append(abs(n.trinary_state))

        # 4. Calculate the potential weight changes for all synapses based on the new neuron states.
        # This is done in a separate loop to ensure all neurons are updated before plasticity is calculated.
        for s in self.synapses:
            if s.integrity <= 0: continue
            pre = self.all_neurons[s.pre_id]
            post = self.all_neurons[s.post_id]
            if pre.is_active and post.is_active:
                s.potential_delta_w = s.calculate_delta_w(pre.trinary_state, post.trinary_state, self.neuromodulators, dt)
                
        # 5. Apply the calculated weight changes to all synapses.
        for s in self.synapses:
            if s.integrity <= 0: continue
            pre = self.all_neurons[s.pre_id]
            post = self.all_neurons[s.post_id]
            if pre.is_active and post.is_active:
                neighbor_deltas = [ns.potential_delta_w for ns in s.neighbor_synapses]
                s.apply_update(dt, self.neuromodulators, neighbor_deltas)
                
        # 6. Update global state and apply slower forms of plasticity.
        self._update_neuromodulator_diffusion(dt)
        for k in ('dopamine', 'serotonin', 'acetylcholine', 'norepinephrine'):
            self.neuromodulators[k] = max(0.0, min(2.0, self.neuromodulators[k] + osc_drive * (0.02 if k == 'dopamine' else 0.01 if k in ('serotonin', 'acetylcholine') else 0.015)))
        self._apply_homeostatic_plasticity()
        self._apply_structural_plasticity()
        
        # 7. Periodically run the evolutionary algorithm for the Aigarth hybridization.
        if self.step_count % self.params.evolution_interval == 0: self._evolve_itu_circles()
        
        self._compute_branching_ratio()
        self.time += dt
        self.step_count += 1
    
    def _apply_structural_plasticity(self):
        """
        Modifies the network's topology over time by removing weak synapses (pruning)
        and adding new ones (formation). This allows the network to self-organize and
        optimize its structure.
        """
        # Prune synapses whose integrity has fallen below the threshold.
        self.synapses = [s for s in self.synapses if s.integrity > self.params.synapse_integrity_threshold]
        
        # With a small probability, form a new synapse between two active, high-energy neurons.
        if random.random() < self.params.synapse_formation_prob:
            act = [n for n in self.all_neurons if n.is_active and n.energy_level > 20.0]
            if len(act) >= 2:
                if self.params.preferential_attachment:
                    degrees = [sum(1 for s in self.synapses if s.pre_id == n.id or s.post_id == n.id) for n in act]
                    pre = random.choices(act, weights=[d+1 for d in degrees], k=1)[0]
                    post = random.choices(act, weights=[d+1 for d in degrees], k=1)[0]
                else:
                    pre = random.choice(act)
                    post = random.choice(act)
                # Ensure the new connection is valid before creating it.
                if pre.id != post.id and not (pre.type == NeuronType.OUTPUT and post.type == NeuronType.INPUT) and not any(ss.pre_id == pre.id and ss.post_id == post.id for ss in self.synapses):
                    self.synapses.append(Synapse(pre.id, post.id, self.params))
                    
        # With a very small probability, "kill" neurons that are unhealthy or out of energy.
        for n in self.hidden_neurons:
            if (n.health < self.params.neuron_death_threshold or n.energy_level < 5.0) and random.random() < 0.001:
                n.is_active = False
    
    def _evolve_itu_circles(self):
        """
        Executes one evolutionary cycle for all ITUs in the network, involving
        fitness calculation, mutation, and selection (pruning).
        """
        for circle in self.itu_circles:
            # Calculate the fitness of the circle based on its recent performance.
            network_activity = [abs(n.trinary_state) for n in circle.neurons if n.is_active]
            temporal_sync = abs(math.cos(self.time * 0.1)) # A simple proxy for synchronization.
            avg_energy = np.mean([n.energy_level for n in circle.neurons if n.is_active]) if circle.neurons else 0.0
            energy_efficiency = avg_energy / self.params.energy_baseline
            fitness = circle.compute_fitness(network_activity, temporal_sync, energy_efficiency)
            
            for n in circle.neurons: n.fitness_score = fitness
            
            # Apply mutation and selection.
            circle.mutate()
            pruned_ids = circle.prune_unfit_neurons()
            if pruned_ids: print(f"[EVOLUTION] Circle {circle.circle_id} pruned {len(pruned_ids)} neurons")
    
    def set_input_states(self, states: List[int]):
        """Clamps the input neurons to the given trinary states."""
        for i, s in enumerate(states[:len(self.input_neurons)]):
            self.input_neurons[i].set_state(s)
    
    def get_output_states(self) -> List[int]:
        """Returns the current trinary states of all active output neurons."""
        return [n.trinary_state for n in self.output_neurons if n.is_active]
    
    def get_energy_status(self) -> Dict[str, float]:
        """Returns a summary of the network's current energy state."""
        active_neurons = [n for n in self.all_neurons if n.is_active]
        total_energy = sum(n.energy_level for n in active_neurons)
        avg_energy = total_energy / max(1, len(active_neurons))
        return {'total_energy': total_energy, 'average_energy': avg_energy, 'efficiency': self.total_energy_consumed / max(1, self.step_count), 'branching_ratio': self.branching_ratio}
    
    def to_dict(self) -> dict:
        """Serializes the entire network state into a single dictionary."""
        return {'parameters': asdict(self.params), 'neurons': {'input': [n.to_dict() for n in self.input_neurons], 'hidden': [n.to_dict() for n in self.hidden_neurons], 'output': [n.to_dict() for n in self.output_neurons]}, 'synapses': [s.to_dict() for s in self.synapses], 'neuromodulators': self.neuromodulators, 'time': self.time, 'step_count': self.step_count, 'energy_consumed': self.total_energy_consumed, 'branching_ratio': self.branching_ratio, 'itu_circles': [c.circle_id for c in self.itu_circles]}

def _rebuild_net_from_dict(d: dict) -> NeuraxonNetwork:
    """A utility function to reconstruct a complete NeuraxonNetwork object from a dictionary."""
    params = NetworkParameters(**d['parameters'])
    net = NeuraxonNetwork(params)
    
    # Helper function to apply saved state to a list of neurons.
    def _apply(neu_list, src_list, off=0):
        for nd in src_list:
            idx = nd['id'] - off
            if 0 <= idx < len(neu_list):
                n = neu_list[idx]
                n.membrane_potential = nd['membrane_potential']
                n.trinary_state = nd['trinary_state']
                n.health = nd['health']
                n.is_active = nd['is_active']
                n.energy_level = nd.get('energy_level', params.energy_baseline)
                n.phase = nd.get('phase', random.random() * 2 * math.pi)
                n.fitness_score = nd.get('fitness_score', 0.0)
                if 'dendritic_branches' in nd:
                    for i, bd in enumerate(nd['dendritic_branches']):
                        if i < len(n.dendritic_branches):
                            n.dendritic_branches[i].branch_potential = bd['branch_potential']
                            n.dendritic_branches[i].plateau_potential = bd['plateau_potential']
                            
    # Apply the saved state to each neuron population.
    _apply(net.input_neurons, d['neurons']['input'], 0)
    _apply(net.hidden_neurons, d['neurons']['hidden'], len(net.input_neurons))
    _apply(net.output_neurons, d['neurons']['output'], len(net.input_neurons) + len(net.hidden_neurons))
    
    # Reconstruct all synapses from the saved data.
    net.synapses = []
    for sd in d['synapses']:
        s = Synapse(sd['pre_id'], sd['post_id'], params)
        s.w_fast = sd['w_fast']
        s.w_slow = sd['w_slow']
        s.w_meta = sd['w_meta']
        s.is_silent = sd['is_silent']
        s.is_modulatory = sd['is_modulatory']
        s.integrity = sd['integrity']
        s.axonal_delay = sd.get('axonal_delay', random.uniform(0, params.max_axonal_delay))
        s.learning_rate_mod = sd.get('learning_rate_mod', 1.0)
        s.potential_delta_w = sd.get('potential_delta_w', 0.0)
        net.synapses.append(s)
        
    # Re-link neighbor synapses, which is necessary for the associativity rule.
    for s in net.synapses:
        s.neighbor_synapses = [ns for ns in net.synapses if ns.pre_id == s.pre_id and ns.post_id != s.post_id]
        
    # Restore the rest of the global network state.
    net.neuromodulators = d['neuromodulators']
    net.time = d['time']
    net.step_count = d['step_count']
    net.total_energy_consumed = d.get('energy_consumed', 0.0)
    net.branching_ratio = d.get('branching_ratio', 1.0)
    return net

# --- General Utility and Helper Functions ---
def _clamp(v, a, b): return max(a, min(b, v))
def _now_str(): return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
def _safe_path(name: str) -> str: return str((pathlib.Path(os.getcwd()) / name).resolve())
def _base26_name(n: int) -> str:
    """Generates a unique, human-readable name from an integer (e.g., 0->'A', 26->'AA')."""
    letters = []
    n0 = n
    while True:
        n, r = divmod(n0, 26)
        letters.append(chr(ord('A') + r))
        if n == 0: break
        n0 = n - 1
    return ''.join(reversed(letters))
def _strip_leading_digits(name: str) -> str:
    """Removes leading digits from a string, used for generating names for offspring."""
    i = 0
    while i < len(name) and name[i].isdigit(): i += 1
    return name[i:] if i < len(name) else ""
def _rand_color(exclude: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Generates a random color that is visually distinct from a list of excluded colors."""
    while True:
        c = (random.randint(30, 235), random.randint(30, 235), random.randint(30, 235))
        if any(sum((c[i] - e[i]) ** 2 for i in range(3)) < 1200 for e in exclude): continue
        if c in [(20, 120, 255), (40, 180, 60), (130, 130, 130), (220, 40, 40)]: continue
        return c
def _rot(x, y, a):
    """Rotates a 2D point (x, y) around the origin by angle 'a'."""
    ca, sa = math.cos(a), math.sin(a)
    return (x * ca - y * sa, x * sa + y * ca)
def _chunked(seq, n):
    """Yields successive n-sized chunks from a sequence."""
    n = max(1, int(n))
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
def _pick_save_file(default_name: str) -> Optional[str]:
    """Opens a native OS file dialog to choose a save location."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=default_name, filetypes=[("JSON files", "*.json")])
        root.destroy()
        return path if path else None
    except Exception:
        return _safe_path(default_name)
def _pick_open_file() -> Optional[str]:
    """Opens a native OS file dialog to choose a file to open."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        root.destroy()
        return path if path else None
    except Exception:
        # A simple fallback for environments without a GUI.
        cand = sorted([p for p in os.listdir(os.getcwd()) if p.endswith(".json") and p.startswith("nxer_")])
        return cand[-1] if cand else None

class Slider:
    """A simple UI slider widget implemented with Pygame for the configuration screen."""
    def __init__(self, rect: pygame.Rect, min_val: float, max_val: float, default_val: float, label: str, is_int: bool = True):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.is_int = is_int
        self.label = label
        self.handle_radius = 10
        self.dragging = False
        range_size = max_val - min_val
        self.normalized_pos = (default_val - min_val) / range_size if range_size != 0 else 0.5
        self.normalized_pos = _clamp(self.normalized_pos, 0.0, 1.0)
        track_y = rect.centery
        self.track_left = rect.x + self.handle_radius
        self.track_right = rect.x + rect.width - self.handle_radius
        self.track_top = track_y
        self.track_bottom = track_y
        self.handle_x = self.track_left + self.normalized_pos * (self.track_right - self.track_left)
        self.handle_y = track_y
    
    def handle_event(self, event: pygame.event.Event):
        """Handles mouse input for dragging the slider."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos
            if math.hypot(mouse_x - self.handle_x, mouse_y - self.handle_y) <= self.handle_radius:
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x, _ = event.pos
            self.handle_x = _clamp(mouse_x, self.track_left, self.track_right)
            self.normalized_pos = (self.handle_x - self.track_left) / (self.track_right - self.track_left)
            return True
        return False
    
    def get_value(self) -> float:
        """Returns the current numerical value of the slider."""
        value = self.min_val + self.normalized_pos * (self.max_val - self.min_val)
        return int(round(value)) if self.is_int else float(value)
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        """Renders the slider onto a Pygame surface."""
        track_y = self.rect.centery
        pygame.draw.line(surface, (100, 100, 100), (self.track_left, track_y), (self.track_right, track_y), 3)
        pygame.draw.circle(surface, (200, 200, 200), (int(self.handle_x), int(self.handle_y)), self.handle_radius)
        pygame.draw.circle(surface, (150, 150, 150), (int(self.handle_x), int(self.handle_y)), self.handle_radius, 2)
        label_surf = font.render(self.label, True, (220, 220, 220))
        label_x = self.rect.x + self.rect.width // 2 - label_surf.get_width() // 2
        surface.blit(label_surf, (label_x, self.rect.y - 20))
        value = self.get_value()
        value_str = str(int(value)) if self.is_int else f"{value:.2f}"
        value_surf = font.render(value_str, True, (255, 255, 0))
        surface.blit(value_surf, (self.rect.x + self.rect.width + 10, self.rect.y))

# --- Game of Life Simulation: World, Agent, and Object Definitions ---

# Constants for terrain types.
T_SEA, T_LAND, T_ROCK = 0, 1, 2
# A list of colors reserved for UI or special objects to ensure agent colors are distinct.
RESERVED_COLORS = [(20, 120, 255), (40, 180, 60), (130, 130, 130), (220, 40, 40)]

@dataclass
class NxErStats:
    """A dataclass to store performance statistics for a single NxEr agent."""
    food_found: float = 0.0
    food_taken: float = 0.0
    explored: int = 0
    time_lived_s: float = 0.0
    mates_performed: int = 0
    energy_efficiency: float = 0.0
    temporal_sync_score: float = 0.0
    fitness_score: float = 0.0

@dataclass
class NxEr:
    """
    Represents a single agent (a "Neuraxon-enabled life form" or NxEr) in the simulation.
    Each NxEr is an individual entity whose behavior is governed by its own internal NeuraxonNetwork.
    """
    id: int
    name: str
    color: Tuple[int, int, int]
    pos: Tuple[int, int]  # Current (x, y) position in the world.
    can_land: bool  # Can this agent move on land tiles?
    can_sea: bool   # Can this agent move on sea tiles?
    net: NeuraxonNetwork  # The "brain" of the agent.
    food: float  # The agent's energy level.
    is_male: bool
    alive: bool = True
    born_ts: float = time.time()
    died_ts: Optional[float] = None
    last_inputs: Tuple[int, int, int] = (0, 0, 0) # The last trinary inputs provided to the network.
    ticks_per_action: int = 1 # How many simulation steps pass between each of this agent's actions.
    tick_accum: int = 0
    harvesting: Optional[int] = None
    mating_with: Optional[int] = None # The ID of the agent it is currently mating with.
    mating_end_tick: Optional[int] = None
    stats: NxErStats = None
    visited: set = None # A set of all coordinates this agent has visited.
    dopamine_boost_ticks: int = 0 # A counter for a temporary dopamine boost (e.g., after finding food).
    _last_O4: int = 0 # Stores the last state of the 4th output neuron (mate/attack).
    mating_intent_until_tick: int = 0 # A timer for showing mating intent.
    parents: Tuple[Optional[int], Optional[int]] = (None, None)
    mate_cooldown_until_tick: int = 0 # A timer preventing an agent from mating too frequently.
    last_move_tick: int = 0
    last_pos: Tuple[int, int] = (0, 0)

@dataclass
class Food:
    """Represents a source of food in the world."""
    id: int
    anchor: Tuple[int, int] # The original spawn location, used for respawning nearby.
    pos: Tuple[int, int] # The current location.
    alive: bool = True
    respawn_at_tick: Optional[int] = None # The simulation tick at which this food will respawn.
    remaining: int = 25 # How much food is left at this source.
    progress: Dict[int, int] = field(default_factory=dict) # Tracks harvesting progress by different NxErs.

class World:
    """Generates and holds the state of the 2D grid-based environment."""
    def __init__(self, N: int, sea_pct: float, rock_pct: float, rnd_seed=None):
        self.N = N
        if rnd_seed is not None: random.seed(rnd_seed)
        # Use multiple layers of a simple noise function to generate natural-looking terrain.
        self.noise_offsets = [(random.random() * 100, random.random() * 100) for _ in range(3)]
        self.grid = [[T_LAND for _ in range(N)] for _ in range(N)]
        self._gen(sea_pct, rock_pct)
    
    def _noise(self, x, y, s, offset_idx):
        """A simple noise function based on sine and cosine waves."""
        ox, oy = self.noise_offsets[offset_idx]
        x_off, y_off = x + ox, y + oy
        return (math.sin(x_off * 0.15 * s) + math.cos(y_off * 0.13 * s) + math.sin((x_off + y_off) * 0.07 * s)) * 0.333
    
    def _gen(self, sea_pct, rock_pct):
        """Procedurally generates the world map with land, sea, and rocks."""
        N = self.N
        values = [[0.0] * N for _ in range(N)]
        # Generate a heightmap using layered noise, with a radial gradient to form an island.
        for y in range(N):
            for x in range(N):
                r = math.hypot(x - N / 2, y - N / 2) / (N * 0.7)
                v = (self._noise(x, y, 0.5, 0) + self._noise(x, y, 1.0, 1) + self._noise(x, y, 1.8, 2)) / 3.0 - r * 0.9
                values[y][x] = v
        
        # Determine the sea level threshold based on the desired percentage of sea.
        flat = sorted(v for row in values for v in row)
        k = int(len(flat) * sea_pct)
        sea_thresh = flat[k] if 0 <= k < len(flat) else min(flat)
        
        # Assign terrain types based on height relative to the sea level.
        for y in range(N):
            for x in range(N):
                self.grid[y][x] = T_SEA if values[y][x] <= sea_thresh else T_LAND
                
        # Randomly place rocks on a percentage of the land tiles.
        land = [(x, y) for y in range(N) for x in range(N) if self.grid[y][x] == T_LAND]
        num_rocks = int(len(land) * rock_pct)
        if num_rocks > 0:
            for (x, y) in random.sample(land, k=min(num_rocks, len(land))):
                self.grid[y][x] = T_ROCK
    
    def in_bounds(self, p):
        """Checks if a coordinate is within the world boundaries."""
        x, y = p
        return 0 <= x < self.N and 0 <= y < self.N
    
    def terrain(self, p):
        """Returns the terrain type at a given coordinate, with toroidal (wrapping) boundaries."""
        x, y = p
        return self.grid[y % self.N][x % self.N]

def _net_batch_step_and_outputs(batch_items: List[Tuple[int, dict, Tuple[int, int, int], int]]):
    """
    This is the target function for the worker processes in the multiprocessing pool.
    It receives a batch of serialized networks, deserializes them, runs the simulation step,
    and returns the serialized results. This is the core of the parallel computation.
    """
    out = []
    for aid, serialized_net, input_states, steps in batch_items:
        try:
            # Reconstruct the network object from its dictionary representation.
            net = _rebuild_net_from_dict(serialized_net)
            # Run the network simulation for the specified number of steps.
            for _ in range(max(1, steps)):
                net.set_input_states(list(input_states))
                net.simulate_step()
            # Serialize the results and append to the output list.
            out.append((aid, net.to_dict(), net.get_output_states(), net.get_energy_status()))
        except Exception as e:
            # Gracefully handle any errors that occur within a worker process.
            print(f"Worker error: {e}")
            out.append((aid, serialized_net, [0, 0, 0, 0], {}))
    return out

class Renderer:
    """Handles all Pygame-based rendering and user input for the main simulation window."""
    def __init__(self, world: World, textures: Dict[str, Optional[str]], textures_alpha: float):
        pygame.init()
        pygame.display.set_caption("Neuraxon Game of Life v 1.0 By David Vivancos & Dr Jose Sanchez for Qubic Science")
        self.screen = pygame.display.set_mode((1920, 1080))
        self.clock = pygame.time.Clock()
        self.world = world
        # Camera state variables.
        self.zoom = max(2.0, 800.0 / world.N)
        self.pan = [world.N * 0.5, world.N * 0.5]
        self.rot = 0.0
        self.dt = 1 / 60.0
        self.dragging = False
        self.drag_start = (0, 0)
        self.textures_alpha = _clamp(int(textures_alpha * 255) if textures_alpha <= 1 else int(textures_alpha), 0, 255)
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("consolas", 20, bold=True)
        self._load_textures(textures)
        # Rectangles for clickable UI elements.
        self.button_rects = {}
        self.overlay_buttons = {}
        self.selected_nxer_id: Optional[int] = None # The ID of the currently selected NxEr for the detail view.
        self.detail_buttons: Dict[str, pygame.Rect] = {}
        self.ranking_click_areas: List[Tuple[pygame.Rect, int]] = []
    
    def _load_textures(self, tex):
        """Loads optional image files to be used as textures for world elements."""
        def load_one(path):
            if not path or str(path).lower() == "none": return None
            try:
                s = pygame.image.load(path).convert_alpha()
                s.set_alpha(self.textures_alpha)
                return s
            except: return None
        self.tex_land = load_one(tex.get("TextureLand"))
        self.tex_sea = load_one(tex.get("TextureSea"))
        self.tex_rock = load_one(tex.get("TextureRock"))
        self.tex_food = load_one(tex.get("TextureFood"))
        self.tex_nxer = load_one(tex.get("TextureNxEr"))
    
    def world_to_screen(self, x, y):
        """Converts world coordinates to screen coordinates, applying camera pan, zoom, and rotation."""
        cx, cy = self.pan
        dx, dy = (x - cx), (y - cy)
        rx, ry = _rot(dx, dy, self.rot)
        return (int(self.screen.get_width() / 2 + rx * self.zoom), int(self.screen.get_height() / 2 + ry * self.zoom))
    
    def screen_to_world(self, sx, sy) -> Tuple[float, float]:
        """Converts screen coordinates back to world coordinates, reversing the camera transform."""
        cx, cy = self.pan
        rx = (sx - self.screen.get_width() / 2) / self.zoom
        ry = (sy - self.screen.get_height() / 2) / self.zoom
        wx, wy = _rot(rx, ry, -self.rot)
        return (cx + wx, cy + wy)
    
    def _draw_effects(self, effects: List[dict], step_tick: int, GlobalTimeSteps: int):
        """Renders temporary visual effects like hearts for mating or skulls for death."""
        for ef in effects:
            age = step_tick - ef['start']
            if age < 0 or age >= GlobalTimeSteps: continue
            rise_px = int(-40 * (age / max(1, GlobalTimeSteps)))
            sx, sy = self.world_to_screen(ef['pos'][0], ef['pos'][1])
            sy += rise_px
            if ef['kind'] == 'heart':
                r = max(6, int(self.zoom * 0.5))
                pygame.draw.circle(self.screen, (220, 40, 60), (sx - r // 2, sy - r // 4), r // 2)
                pygame.draw.circle(self.screen, (220, 40, 60), (sx + r // 2, sy - r // 4), r // 2)
                pygame.draw.polygon(self.screen, (220, 40, 60), [(sx - r, sy), (sx + r, sy), (sx, sy + r)])
            elif ef['kind'] == 'skull':
                r = max(6, int(self.zoom * 0.45))
                pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), r)
                pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(sx - r // 2, sy, r, r // 2), border_radius=3)
                eye_r = max(2, r // 5)
                pygame.draw.circle(self.screen, (200, 200, 200), (sx - r // 3, sy - r // 4), eye_r)
                pygame.draw.circle(self.screen, (200, 200, 200), (sx + r // 3, sy - r // 4), eye_r)
    
    def _draw_restart_modal(self):
        """Renders the "Game Over" modal dialog with options to restart or quit."""
        self.overlay_buttons = {}
        W, H = self.screen.get_size()
        bw, bh = 200, 48
        rect = pygame.Rect(W // 2 - 260, H // 2 - 140, 520, 280)
        srf = pygame.Surface((W, H), pygame.SRCALPHA)
        srf.fill((0, 0, 0, 160))
        self.screen.blit(srf, (0, 0))
        pygame.draw.rect(self.screen, (15, 15, 18), rect, border_radius=12)
        pygame.draw.rect(self.screen, (90, 90, 100), rect, 2, border_radius=12)
        title = self.big.render("All NxErs have perished.", True, (235, 235, 240))
        subtitle = self.small.render("Restart? (Will do in 2 minutes if no response)", True, (220, 220, 220))
        self.screen.blit(title, (rect.x + (rect.w - title.get_width()) // 2, rect.y + 28))
        self.screen.blit(subtitle, (rect.x + (rect.w - subtitle.get_width()) // 2, rect.y + 60))
        y = rect.y + 130
        yes_rect = pygame.Rect(rect.x + 40, y, bw - 30, bh)
        no_rect = pygame.Rect(rect.x + rect.w - 60 - bw, y, bw - 30, bh)
        for r, lab, key in [(yes_rect, "Yes", "restart_yes"), (no_rect, "No", "restart_no")]:
            pygame.draw.rect(self.screen, (35, 35, 45), r, border_radius=10)
            pygame.draw.rect(self.screen, (110, 110, 130), r, 2, border_radius=10)
            tx = self.big.render(lab, True, (235, 235, 240))
            self.screen.blit(tx, (r.x + (bw - tx.get_width()) // 2, r.y + (bh - tx.get_height()) // 2))
            self.overlay_buttons[key] = r
    
    def draw_world(self, foods: Dict[int, Food], nxers: Dict[int, NxEr], hud: Dict[str, List[Tuple[str, str]]], alive_count: int, dead_count: int, born_count: int, paused: bool, effects: List[dict], step_tick: int, GlobalTimeSteps: int, game_over: bool, game_index: int, best_scores: Optional[Dict[str, float]] = None):
        """The main rendering function, called once per frame to draw the entire scene."""
        self.screen.fill((0, 0, 0))
        w, h = self.screen.get_size()
        cx, cy = self.pan
        
        # --- Draw World Terrain ---
        # Calculate the visible portion of the world to avoid drawing off-screen tiles.
        radius = max(w, h) / self.zoom * 1.5
        x0 = int(max(0, cx - radius)); x1 = int(min(self.world.N, cx + radius))
        y0 = int(max(0, cy - radius)); y1 = int(min(self.world.N, cy + radius))
        
        # Use Level of Detail (LOD) to speed up rendering when zoomed out.
        lod = 1
        tile = max(2, int(self.zoom))
        if tile < 4: lod = 3
        elif tile < 2: lod = 6
        
        for y in range(y0, y1, lod):
            for x in range(x0, x1, lod):
                t = self.world.grid[y][x]
                base = (40, 180, 60) if t == T_LAND else ((25, 100, 200) if t == T_SEA else (110, 110, 110))
                height = 0 if t == T_SEA else (2 if t == T_ROCK else 1)
                c = tuple(_clamp(int(b * (0.85 + 0.08 * height)), 0, 255) for b in base)
                sx, sy = self.world_to_screen(x, y)
                if sx < -tile or sx > w + tile or sy < -tile or sy > h + tile: continue
                pygame.draw.rect(self.screen, c, pygame.Rect(sx, sy, int(self.zoom * lod) + 1, int(self.zoom * lod) + 1))
        
        # --- Draw Objects (Food and NxErs) ---
        for f in foods.values():
            if not f.alive: continue
            sx, sy = self.world_to_screen(f.pos[0], f.pos[1])
            if sx < -50 or sx > w + 50 or sy < -50 or sy > h + 50: continue
            s = max(6, int(self.zoom * 0.8))
            pygame.draw.polygon(self.screen, (220, 40, 40), [(sx, sy - s), (sx - s // 2, sy), (sx + s // 2, sy)])
        for a in nxers.values():
            if not a.alive: continue
            sx, sy = self.world_to_screen(a.pos[0], a.pos[1])
            if sx < -50 or sx > w + 50 or sy < -50 or sy > h + 50: continue
            rad = max(4, int(self.zoom * 0.45))
            pygame.draw.circle(self.screen, a.color, (sx, sy), rad)
            pygame.draw.circle(self.screen, (20, 20, 20), (sx, sy), rad, 1)
            # Draw an inner yellow circle representing the agent's energy level.
            if hasattr(a.net, 'get_energy_status'):
                energy = a.net.get_energy_status().get('average_energy', 0.0)
                energy_rad = max(2, int(rad * energy / 100.0))
                pygame.draw.circle(self.screen, (255, 255, 0), (sx, sy), energy_rad, 1)
                
        self._draw_effects(effects, step_tick, GlobalTimeSteps)
        
        # --- Draw Heads-Up Display (HUD) Side Panel ---
        panel_w = 300
        x = self.screen.get_width() - panel_w - 16; y = 12
        rows = 1
        for _, lst in hud.items(): rows += 1 + min(3, len(lst))
        rows += 11
        panel_h = 26 + rows * 18 + 24
        base_rect = pygame.Rect(x - 10, y - 8, panel_w + 20, panel_h-50)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), base_rect, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 60), base_rect, 1, border_radius=8)
        round_text = self.big.render(f"Game Metrics: Round #{game_index}", True, (230, 230, 230))
        self.screen.blit(round_text, (x, y)); y += 28
        
        # Draw Rankings.
        name2color = {a.name: a.color for a in nxers.values()}
        self.ranking_click_areas = []
        for title, rows in hud.items():
            display_title = title
            score = best_scores.get(title) if best_scores else None
            if score is not None: display_title = f"{title} ({score:.2f})" if isinstance(score, float) else f"{title} ({int(score)})"
            self.screen.blit(self.small.render(display_title, True, (180, 180, 180)), (x, y)); y += 18
            for name, val in rows[:3]:
                base_name = name.replace(" [Die]", "")
                dot_c = name2color.get(base_name, (200, 200, 200))
                pygame.draw.circle(self.screen, dot_c, (x + 8, y + 8), 6)
                name_text = self.small.render(f"{name}", True, (230, 230, 230))
                val_text = self.small.render(f"{val}", True, (220, 220, 220))
                name_rect = name_text.get_rect(topleft=(x + 20, y))
                val_rect = val_text.get_rect(topleft=(x + 180, y))
                clicked_nxer_name = None
                for nxer_obj in nxers.values():
                    if nxer_obj.name == base_name: clicked_nxer_name = nxer_obj.name; break
                if clicked_nxer_name: # Store the clickable area for this ranking entry.
                    combined_rect = name_rect.union(val_rect)
                    self.ranking_click_areas.append((combined_rect, clicked_nxer_name))
                self.screen.blit(name_text, (x + 20, y))
                self.screen.blit(val_text, (x + 180, y))
                y += 16
        y += 6
        
        # Draw general statistics.
        self.screen.blit(self.small.render(f"Alive: {alive_count}", True, (220, 220, 220)), (x, y)); y += 18
        self.screen.blit(self.small.render(f"Dead : {dead_count}", True, (220, 220, 220)), (x, y)); y += 18
        self.screen.blit(self.small.render(f"Born : {born_count}", True, (220, 220, 220)), (x, y)); y += 24
        
        # Draw aggregate network statistics.
        if nxers:
            alive_nxers = [a for a in nxers.values() if a.alive]
            if alive_nxers:
                avg_energy = np.mean([a.net.get_energy_status().get('average_energy', 0.0) for a in alive_nxers])
                avg_branching = np.mean([a.net.branching_ratio for a in alive_nxers])
                self.screen.blit(self.small.render(f"Avg Energy: {avg_energy:.1f}", True, (200, 200, 0)), (x, y)); y += 18
                self.screen.blit(self.small.render(f"Branching: {avg_branching:.2f}", True, (180, 180, 180)), (x, y)); y += 24
        
        # Draw control buttons.
        self.button_rects = {}
        button_rows = [[("playpause", "Pause" if not paused else "Play"), ("exit", "Exit")], [("save", "Save Game"), ("load", "Load Game")], [("save_best", "Save Bests")]]
        bx, by, bw, bh, pad = x, y, 120, 28, 8
        for row in button_rows:
            row_x = bx
            for key, lab in row:
                r = pygame.Rect(row_x, by, bw, bh)
                pygame.draw.rect(self.screen, (35, 35, 40), r, border_radius=6)
                pygame.draw.rect(self.screen, (90, 90, 100), r, 1, border_radius=6)
                tx = self.small.render(lab, True, (230, 230, 230))
                self.screen.blit(tx, (r.x + (bw - tx.get_width()) // 2, r.y + (bh - tx.get_height()) // 2))
                self.button_rects[key] = r
                row_x += bw + pad
            by += bh + pad
        
        # --- Draw Detail Panel for Selected NxEr ---
        self.detail_buttons = {}
        if paused and self.selected_nxer_id is not None and self.selected_nxer_id in nxers:
            a = nxers[self.selected_nxer_id]
            px, py, pw, ph = x, by + 12, panel_w, 340
            rect = pygame.Rect(px - 10, py - 8, pw + 20, ph+50)
            pygame.draw.rect(self.screen, (0, 0, 0, 200), rect, border_radius=8)
            pygame.draw.rect(self.screen, (80, 80, 80), rect, 1, border_radius=8)
            
            gender_str = "Male" if a.is_male else "Female"
            title = f"{a.name} (id {a.id}) - {gender_str}"
            self.screen.blit(self.big.render(title, True, (230, 230, 230)), (px, py)); py += 28
            
            # Display detailed stats for the selected agent.
            terrain_type = "Land" if a.can_land and not a.can_sea else ("Sea" if a.can_sea and not a.can_land else "Both")
            energy_status = a.net.get_energy_status() if hasattr(a.net, 'get_energy_status') else {}
            facts = [f"Color : {a.color}", f"Pos : {a.pos} Food : {a.food:.1f}", f"Alive : {a.alive} Terr: {terrain_type} Lived : {a.stats.time_lived_s:.1f}s", f"Found : {a.stats.food_found:.1f} Taken: {a.stats.food_taken:.1f}", f"Mates : {a.stats.mates_performed} Explr : {a.stats.explored}", f"Energy: {energy_status.get('average_energy', 0):.1f} Fitness: {a.stats.fitness_score:.3f}", f"Branching: {energy_status.get('branching_ratio', 0):.2f}"]
            for line in facts:
                self.screen.blit(self.small.render(line, True, (220, 220, 220)), (px, py)); py += 18
            py += 6
            
            # Display key parameters of the agent's neural network.
            self.screen.blit(self.small.render("Network params:", True, (200, 200, 200)), (px, py)); py += 18
            P = a.net.params
            main_params = [f"inputs={P.num_input_neurons} hidden={P.num_hidden_neurons} outputs={P.num_output_neurons}", f"conn_prob={P.connection_probability:.2f} steps={P.simulation_steps}", f"tau_fast={P.tau_fast:.2f} slow={P.tau_slow:.2f} meta={P.tau_meta:.2f}", f"thr_exc={P.firing_threshold_excitatory:.3f} thr_inh={P.firing_threshold_inhibitory:.3f}", f"learn={P.learning_rate:.3f} stdp_win={P.stdp_window:.3f}", f"dopamine={P.dopamine_baseline:.3f} serotonin={P.serotonin_baseline:.3f}", f"energy_cost={P.firing_energy_cost:.1f} meta_rate={P.metabolic_rate:.2f}", f"circles={len(a.net.itu_circles)} evolve_int={P.evolution_interval}"]
            for line in main_params:
                self.screen.blit(self.small.render(line, True, (210, 210, 210)), (px, py)); py += 16
            py += 10
            
            # Draw buttons specific to the detail panel (e.g., save this specific NxEr).
            bw2, bh2, pad2 = 120, 26, 8
            detail_button_rows = [[("save_nxer", "Save NxEr"), ("load_nxer", "Load NxEr")], [("save_nxvizer", "Save NxVizer"), ("load_nxvizer", "Load NxVizer")]]
            for row in detail_button_rows:
                row_x = px
                for key, lab in row:
                    r = pygame.Rect(row_x, py, bw2, bh2)
                    pygame.draw.rect(self.screen, (35, 35, 40), r, border_radius=6)
                    pygame.draw.rect(self.screen, (90, 90, 100), r, 1, border_radius=6)
                    tx = self.small.render(lab, True, (230, 230, 230))
                    self.screen.blit(tx, (r.x + (bw2 - tx.get_width()) // 2, r.y + (bh2 - tx.get_height()) // 2))
                    self.detail_buttons[key] = r
                    row_x += bw2 + pad2
                py += bh2 + pad2
                
        if game_over: self._draw_restart_modal()
        pygame.display.flip()
    
    def handle_input(self, dt):
        """Handles continuous keyboard input for camera panning."""
        keys = pygame.key.get_pressed()
        pstep = (50.0 / self.zoom) * dt
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: self.pan[0] -= pstep
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.pan[0] += pstep
        if keys[pygame.K_w] or keys[pygame.K_UP]: self.pan[1] -= pstep
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: self.pan[1] += pstep
    
    def event_zoom_rotate_pan(self, ev):
        """Handles discrete user input events for camera control (zoom, rotation, drag-pan)."""
        if ev.type == pygame.MOUSEWHEEL:
            self.zoom *= 1.1 if ev.y > 0 else 0.9
            self.zoom = _clamp(self.zoom, 0.5, 64.0)
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_q: self.rot -= 0.04
            elif ev.key == pygame.K_e: self.rot += 0.04
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3: # Right mouse button for drag-pan.
            self.dragging = True
            self.drag_start = ev.pos
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            dx = ev.pos[0] - self.drag_start[0]
            dy = ev.pos[1] - self.drag_start[1]
            self.drag_start = ev.pos
            wx, wy = _rot(dx / self.zoom, dy / self.zoom, -self.rot)
            self.pan[0] -= wx
            self.pan[1] -= wy
    
    def button_clicked(self, pos) -> Optional[str]:
        """Checks if a click position collides with any of the main UI buttons."""
        for k, r in self.button_rects.items():
            if r.collidepoint(pos): return k
        for k, r in self.overlay_buttons.items():
            if r.collidepoint(pos): return k
        return None
    
    def detail_button_clicked(self, pos) -> Optional[str]:
        """Checks if a click position collides with any buttons in the detail panel."""
        for k, r in self.detail_buttons.items():
            if r.collidepoint(pos): return k
        return None
    
    def ranking_clicked(self, pos) -> Optional[int]:
        """Checks if a click position collides with any of the names in the ranking list."""
        for rect, name in self.ranking_click_areas:
            if rect.collidepoint(pos): return name
        return None
    
    def clear_detail(self):
        """Deselects the current NxEr and clears the detail panel."""
        self.selected_nxer_id = None
        self.detail_buttons = {}
    
    def tick(self, fps_cap=60):
        """Advances the Pygame clock, enforces an FPS cap, and returns the frame's delta time."""
        self.dt = self.clock.tick(fps_cap) / 1000.0
        return self.dt

def GameOfLife(NxWorldSize: int = 100, NxWorldSea: float = 0.60, NxWorldRocks: float = 0.05, StartingNxErs: int = 30, MaxNxErs: int = 400, MaxFood: int = 300, FoodRespan: int = 600, StartFood: float = 40.0, MaxNeurons: int = 12, GlobalTimeSteps: int = 60, TextureLand: Optional[str] = None, TextureSea: Optional[str] = None, TextureRock: Optional[str] = None, TextureFood: Optional[str] = None, TextureNxEr: Optional[str] = None, TexturesAlpha: float = 0.7, MateCooldownSeconds: int = 10, random_seed: Optional[int] = None):
    """
    The main function that initializes and runs the entire Game of Life simulation.
    It orchestrates the world generation, agent creation, the main simulation loop,
    and the interaction between the simulation logic and the renderer.
    """
    # Clamp user-provided parameters to ensure they are within a sensible range.
    NxWorldSize = _clamp(int(NxWorldSize), 30, 1000)
    NxWorldSea = _clamp(float(NxWorldSea), 0.0, 0.95)
    NxWorldRocks = _clamp(float(NxWorldRocks), 0.0, 0.9)
    StartingNxErs = _clamp(int(StartingNxErs), 10, 100)
    MaxNxErs = _clamp(int(MaxNxErs), 200, 1000)
    MaxFood = _clamp(int(MaxFood), 10, 1000)
    FoodRespan = _clamp(int(FoodRespan), 10, 3000)
    StartFood = _clamp(float(StartFood), 10.0, 100.0)
    MaxNeurons = _clamp(int(MaxNeurons), 1, 25)
    GlobalTimeSteps = _clamp(int(GlobalTimeSteps), 30, 300)
    MateCooldownSeconds = _clamp(int(MateCooldownSeconds), 0, 300)
    if random_seed is not None: random.seed(int(random_seed))
    
    mate_cooldown_ticks = MateCooldownSeconds * GlobalTimeSteps
    textures = {"TextureLand": TextureLand, "TextureSea": TextureSea, "TextureRock": TextureRock, "TextureFood": TextureFood, "TextureNxEr": TextureNxEr}
    
    # Initialize the core components of the simulation.
    world = World(NxWorldSize, NxWorldSea, NxWorldRocks, rnd_seed=random_seed)
    renderer = Renderer(world, textures, TexturesAlpha)
    nxers: Dict[int, NxEr] = {}
    foods: Dict[int, Food] = {}
    occupied = set()
    births_count = 0
    deaths_count = 0
    effects: List[dict] = []
    champion_counts: Dict[str, int] = {}
    game_index = 1
    all_time_best: Dict[str, List[NxEr]] = {'food_found': [], 'food_taken': [], 'explored': [], 'time_lived_s': [], 'mates_performed': [], 'fitness_score': []}
    
    def wrap_pos(p: Tuple[int, int]) -> Tuple[int, int]: return (p[0] % world.N, p[1] % world.N)
    
    def find_free(allow_sea=True, allow_land=True, forbid=None, near=None, search_radius=5):
        """Finds a valid, unoccupied coordinate in the world, optionally near a specific point."""
        forbid = forbid or set()
        if near is not None:
            nx, ny = near
            for r in range(0, max(2, search_radius)):
                cand = []
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        x, y = wrap_pos((nx + dx, ny + dy))
                        if (x, y) in forbid or (x, y) in occupied: continue
                        t = world.terrain((x, y))
                        if t == T_ROCK or (t == T_LAND and not allow_land) or (t == T_SEA and not allow_sea): continue
                        cand.append((x, y))
                random.shuffle(cand)
                for c in cand: return c
            return None
        else:
            for _ in range(4000): # Try a number of random positions.
                x = random.randrange(world.N)
                y = random.randrange(world.N)
                if (x, y) in forbid or (x, y) in occupied: continue
                t = world.terrain((x, y))
                if t == T_ROCK or (t == T_LAND and not allow_land) or (t == T_SEA and not allow_sea): continue
                return (x, y)
            return None
            
    def place_initial_food(count):
        """Populates the world with the initial set of food sources."""
        foods.clear()
        for i in range(count):
            p = find_free(allow_sea=True, allow_land=True)
            if not p: break
            foods[i] = Food(id=i, anchor=p, pos=p, alive=True, respawn_at_tick=None, remaining=25, progress={})
    place_initial_food(min(MaxFood, max(30, MaxFood // 2)))
    used_colors = set(RESERVED_COLORS)
    
    def _random_params(hidden_max: int) -> NetworkParameters:
        """Creates a randomized set of network parameters to generate diversity in the initial population."""
        p = NetworkParameters(network_name="Neuraxon NxEr", num_input_neurons=3, num_hidden_neurons=random.randint(1, hidden_max), num_output_neurons=4)
        p.connection_probability = random.uniform(0.05, 0.20)
        p.membrane_time_constant = random.uniform(5.0, 50.0)
        p.firing_threshold_excitatory = random.uniform(0.5, 2.0)
        p.firing_threshold_inhibitory = random.uniform(-2.0, -0.5)
        p.adaptation_rate = random.uniform(0.0, 0.2)
        p.spontaneous_firing_rate = random.uniform(0.0, 0.1)
        p.neuron_health_decay = random.uniform(0.0002, 0.01)
        p.tau_fast = random.uniform(1.0, 10.0)
        p.tau_slow = random.uniform(20.0, 100.0)
        p.tau_meta = random.uniform(500.0, 5000.0)
        p.learning_rate = random.uniform(0.002, 0.05)
        p.stdp_window = random.uniform(10.0, 50.0)
        p.synapse_integrity_threshold = random.uniform(0.05, 0.2)
        p.synapse_formation_prob = random.uniform(0.01, 0.08)
        p.synapse_death_prob = random.uniform(0.005, 0.05)
        p.neuron_death_threshold = random.uniform(0.05, 0.2)
        p.dopamine_baseline = random.uniform(0.05, 0.3)
        p.serotonin_baseline = random.uniform(0.05, 0.3)
        p.acetylcholine_baseline = random.uniform(0.05, 0.3)
        p.norepinephrine_baseline = random.uniform(0.05, 0.3)
        p.neuromod_decay_rate = random.uniform(0.05, 0.25)
        p.dt = 1.0
        p.simulation_steps = random.randint(10, max(20, 2 * GlobalTimeSteps))
        p.firing_energy_cost = random.uniform(3.0, 8.0)
        p.metabolic_rate = random.uniform(0.8, 1.5)
        p.associativity_strength = random.uniform(0.05, 0.2)
        p.evolution_interval = random.randint(500, 2000)
        return p
    
    def make_nxer(idx: int) -> NxEr:
        """Factory function to create a single new NxEr with a randomized brain."""
        p = _random_params(MaxNeurons)
        net = NeuraxonNetwork(p)
        pos = find_free(allow_sea=True, allow_land=True) or (random.randrange(world.N), random.randrange(world.N))
        terrain = world.terrain(pos)
        if terrain == T_LAND: can_land, can_sea = True, False
        elif terrain == T_SEA: can_land, can_sea = False, True
        else: can_land, can_sea = True, False
        is_male = random.random() < 0.5
        nx = NxEr(id=idx, name=_base26_name(idx), color=_rand_color(list(used_colors)), pos=pos, can_land=can_land, can_sea=can_sea, net=net, food=float(StartFood), is_male=is_male, ticks_per_action=max(1, int(GlobalTimeSteps / max(1, p.simulation_steps))), stats=NxErStats(), visited=set([pos]), parents=(None, None), mate_cooldown_until_tick=0, last_move_tick=0, last_pos=pos)
        used_colors.add(nx.color)
        nx.last_inputs = (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
        return nx
    
    def spawn_child(A: NxEr, B: NxEr, near_pos: Tuple[int, int]) -> Optional[NxEr]:
        """Creates a new NxEr from two parents, with a simple model of parameter inheritance."""
        nonlocal births_count
        if len(nxers) >= MaxNxErs: return None
        child_id = (max(nxers.keys()) + 1) if nxers else 0
        p = _random_params(MaxNeurons)
        # A simple form of "genetic" inheritance: the child's parameters are randomly chosen from its parents or a new random value.
        p.connection_probability = random.choice([A.net.params.connection_probability, B.net.params.connection_probability, p.connection_probability])
        p.simulation_steps = random.choice([A.net.params.simulation_steps, B.net.params.simulation_steps, p.simulation_steps])
        A_land_specialist = A.can_land and not A.can_sea
        A_sea_specialist = A.can_sea and not A.can_land
        B_land_specialist = B.can_land and not B.can_sea
        B_sea_specialist = B.can_sea and not B.can_land
        terrain_A = world.terrain(A.pos)
        terrain_B = world.terrain(B.pos)
        is_shoreline_mating = (A_land_specialist and B_sea_specialist and terrain_A == T_LAND and terrain_B == T_SEA) or (A_sea_specialist and B_land_specialist and terrain_A == T_SEA and terrain_B == T_LAND)
        # Inherit terrain traversal abilities. Mating across a shoreline can produce an amphibious child.
        if is_shoreline_mating: can_land, can_sea = True, True
        else:
            if A_land_specialist and B_land_specialist: can_land, can_sea = True, False
            elif A_sea_specialist and B_sea_specialist: can_land, can_sea = False, True
            else:
                can_land = A.can_land or B.can_land
                can_sea = A.can_sea or B.can_sea
        if not (can_land or can_sea): can_land = True
        is_male_child = random.random() < 0.5
        child = NxEr(id=child_id, name=f"{_strip_leading_digits(A.name)}-{_strip_leading_digits(B.name)}", color=_rand_color(list(RESERVED_COLORS) + [a.color for a in nxers.values()]), pos=near_pos, can_land=can_land, can_sea=can_sea, net=NeuraxonNetwork(p), food=0.0, is_male=is_male_child, ticks_per_action=max(1, int(GlobalTimeSteps / max(1, p.simulation_steps))), stats=NxErStats(), visited=set(), parents=(A.id, B.id), mate_cooldown_until_tick=0, last_move_tick=step_tick, last_pos=near_pos)
        # Parents provide the child with some starting food.
        transfer = min(5.0, min(A.food / 2, B.food / 2))
        A.food -= transfer
        B.food -= transfer
        child.food += transfer * 2
        child.pos = find_free(allow_sea=can_sea, allow_land=can_land, near=near_pos, search_radius=3) or near_pos
        nxers[child.id] = child
        occupied.add(child.pos)
        A.stats.mates_performed += 1
        B.stats.mates_performed += 1
        births_count += 1
        child.stats.fitness_score = (A.stats.fitness_score + B.stats.fitness_score) / 2
        return child
        
    # Create the initial population.
    for i in range(StartingNxErs):
        a = make_nxer(i)
        nxers[a.id] = a
        occupied.add(a.pos)
        
    # Set up the multiprocessing pool for parallel network updates.
    cpu = os.cpu_count() or 4
    nx_workers = max(1, min(15, cpu - 2))
    ctx = mp.get_context("spawn")
    nx_pool = ProcessPoolExecutor(max_workers=nx_workers, mp_context=ctx)
    
    # Main simulation loop variables.
    running, paused = True, True
    step_tick = 0
    boot_random_until = 5 * GlobalTimeSteps # Period at the start where agents take random actions to bootstrap activity.
    game_over = False
    game_over_start_time = None
    user_declined_restart = False
    
    def push_effect(kind: str, pos: Tuple[int, int]): effects.append({'kind': kind, 'pos': pos, 'start': step_tick})
    
    # --- Save/Load and Game Management Functions ---
    def save_nxer_to_file(a: NxEr, save_name: str = None):
        """Saves a single NxEr's complete state to a JSON file."""
        default = save_name or f"nxer_{a.name}_{_now_str()}.json"
        path = _pick_save_file(default)
        if not path: return
        data = {"meta": {"created": _now_str(), "type": "NxEr"}, "nxer": {"id": a.id, "name": a.name, "color": a.color, "pos": a.pos, "can_land": a.can_land, "can_sea": a.can_sea, "food": a.food, "is_male": a.is_male, "alive": a.alive, "born_ts": a.born_ts, "died_ts": a.died_ts, "last_inputs": a.last_inputs, "ticks_per_action": a.ticks_per_action, "tick_accum": a.tick_accum, "harvesting": a.harvesting, "mating_with": a.mating_with, "mating_end_tick": a.mating_end_tick, "visited": list(a.visited), "dopamine_boost_ticks": a.dopamine_boost_ticks, "_last_O4": a._last_O4, "mating_intent_until_tick": a.mating_intent_until_tick, "parents": list(a.parents) if a.parents else [None, None], "mate_cooldown_until_tick": a.mate_cooldown_until_tick, "last_move_tick": a.last_move_tick, "last_pos": a.last_pos, "stats": asdict(a.stats), "net": a.net.to_dict()}}
        with open(path, "w") as f: json.dump(data, f)
        print(f"[SAVE NxEr] {path}")
    
    def load_nxer_from_file(spawn_near: Tuple[int, int] = None):
        """Loads an NxEr from a file and adds it to the current simulation."""
        path = _pick_open_file()
        if not path: return
        with open(path, "r") as f: data = json.load(f)
        nd = data.get("nxer", data)
        net = _rebuild_net_from_dict(nd["net"])
        pos0 = tuple(nd.get("pos", (world.N // 2, world.N // 2)))
        if spawn_near is not None: pos0 = spawn_near
        pos = find_free(allow_sea=True, allow_land=True, near=pos0, search_radius=6) or wrap_pos(pos0)
        base_name = str(nd.get("name", f"N{len(nxers)}"))
        name = base_name; counter = 1
        alive_names = {a.name for a in nxers.values() if a.alive}
        while name in alive_names: name = f"{base_name}{counter}"; counter += 1
        a = NxEr(id=(max(nxers.keys()) + 1) if nxers else 0, name=name, color=tuple(nd.get("color", (200, 200, 200))), pos=pos, can_land=nd["can_land"], can_sea=nd["can_sea"], net=net, food=float(StartFood), is_male=bool(nd.get("is_male", random.random() < 0.5)), alive=True, born_ts=time.time(), died_ts=None, last_inputs=tuple(nd.get("last_inputs", (0, 0, 0))), ticks_per_action=int(nd.get("ticks_per_action", 1)), tick_accum=int(nd.get("tick_accum", 0)), harvesting=nd.get("harvesting"), mating_with=nd.get("mating_with"), mating_end_tick=nd.get("mating_end_tick"), stats=NxErStats(**nd.get("stats", {})), visited=set(map(tuple, nd.get("visited", []))), dopamine_boost_ticks=int(nd.get("dopamine_boost_ticks", 0)), _last_O4=int(nd.get("_last_O4", 0)), mating_intent_until_tick=int(nd.get("mating_intent_until_tick", 0)), parents=tuple(nd.get("parents", [None, None])), mate_cooldown_until_tick=int(nd.get("mate_cooldown_until_tick", 0)), last_move_tick=int(nd.get("last_move_tick", step_tick)), last_pos=tuple(nd.get("last_pos", pos)))
        a.tick_accum = 0; a.harvesting = None; a.mating_with = None; a.mating_end_tick = None; a.mating_intent_until_tick = 0; a.mate_cooldown_until_tick = 0
        nxers[a.id] = a
        occupied.add(a.pos)
        print(f"[LOAD NxEr] spawned {a.name} at {a.pos}")
    
    def save_nxvizer_to_file(a: NxEr, save_name: str = None):
        """Saves only the network parameters of an NxEr, for use with the visualizer tool."""
        default = save_name or f"nxvizer_{a.name}_{_now_str()}.json"
        path = _pick_save_file(default)
        if not path: return
        params = a.net.params
        data = {"network_name": params.network_name, "num_input_neurons": params.num_input_neurons, "num_hidden_neurons": params.num_hidden_neurons, "num_output_neurons": params.num_output_neurons, "connection_probability": params.connection_probability, "membrane_time_constant": params.membrane_time_constant, "firing_threshold_excitatory": params.firing_threshold_excitatory, "firing_threshold_inhibitory": params.firing_threshold_inhibitory, "adaptation_rate": params.adaptation_rate, "spontaneous_firing_rate": params.spontaneous_firing_rate, "neuron_health_decay": params.neuron_health_decay, "tau_fast": params.tau_fast, "w_fast_init_min": params.w_fast_init_min, "w_fast_init_max": params.w_fast_init_max, "tau_slow": params.tau_slow, "w_slow_init_min": params.w_slow_init_min, "w_slow_init_max": params.w_slow_init_max, "tau_meta": params.tau_meta, "w_meta_init_min": params.w_meta_init_min, "w_meta_init_max": params.w_meta_init_max, "learning_rate": params.learning_rate, "stdp_window": params.stdp_window, "synapse_integrity_threshold": params.synapse_integrity_threshold, "synapse_formation_prob": params.synapse_formation_prob, "synapse_death_prob": params.synapse_death_prob, "neuron_death_threshold": params.neuron_death_threshold, "dopamine_baseline": params.dopamine_baseline, "serotonin_baseline": params.serotonin_baseline, "acetylcholine_baseline": params.acetylcholine_baseline, "norepinephrine_baseline": params.norepinephrine_baseline, "neuromod_decay_rate": params.neuromod_decay_rate, "dt": params.dt, "simulation_steps": params.simulation_steps}
        with open(path, "w") as f: json.dump(data, f, indent=2)
        print(f"[SAVE NxVizer] {path}")
        
    def load_nxvizer_from_file(spawn_near: Tuple[int, int] = None):
        """Loads a set of network parameters and creates a new, random NxEr with that "brain" configuration."""
        path = _pick_open_file()
        if not path: return
        with open(path, "r") as f: data = json.load(f)
        params = NetworkParameters(network_name=data.get("network_name", "Neuraxon NxEr"), num_input_neurons=data.get("num_input_neurons", 3), num_hidden_neurons=data.get("num_hidden_neurons", 10), num_output_neurons=data.get("num_output_neurons", 4), connection_probability=data.get("connection_probability", 0.15), membrane_time_constant=data.get("membrane_time_constant", 20.0), firing_threshold_excitatory=data.get("firing_threshold_excitatory", 0.9), firing_threshold_inhibitory=data.get("firing_threshold_inhibitory", -0.9), adaptation_rate=data.get("adaptation_rate", 0.05), spontaneous_firing_rate=data.get("spontaneous_firing_rate", 0.02), neuron_health_decay=data.get("neuron_health_decay", 0.001), tau_fast=data.get("tau_fast", 5.0), w_fast_init_min=data.get("w_fast_init_min", -1.0), w_fast_init_max=data.get("w_fast_init_max", 1.0), tau_slow=data.get("tau_slow", 50.0), w_slow_init_min=data.get("w_slow_init_min", -0.5), w_slow_init_max=data.get("w_slow_init_max", 0.5), tau_meta=data.get("tau_meta", 1000.0), w_meta_init_min=data.get("w_meta_init_min", -0.3), w_meta_init_max=data.get("w_meta_init_max", 0.3), learning_rate=data.get("learning_rate", 0.01), stdp_window=data.get("stdp_window", 20.0), synapse_integrity_threshold=data.get("synapse_integrity_threshold", 0.1), synapse_formation_prob=data.get("synapse_formation_prob", 0.02), synapse_death_prob=data.get("synapse_death_prob", 0.01), neuron_death_threshold=data.get("neuron_death_threshold", 0.1), dopamine_baseline=data.get("dopamine_baseline", 0.12), serotonin_baseline=data.get("serotonin_baseline", 0.12), acetylcholine_baseline=data.get("acetylcholine_baseline", 0.12), norepinephrine_baseline=data.get("norepinephrine_baseline", 0.12), neuromod_decay_rate=data.get("neuromod_decay_rate", 0.1), dt=data.get("dt", 1.0), simulation_steps=data.get("simulation_steps", 30))
        net = NeuraxonNetwork(params)
        pos0 = spawn_near or (world.N // 2, world.N // 2)
        pos = find_free(allow_sea=True, allow_land=True, near=pos0, search_radius=6) or wrap_pos(pos0)
        terrain_choice = random.random()
        if terrain_choice < 0.33: can_land, can_sea = True, False
        elif terrain_choice < 0.66: can_land, can_sea = False, True
        else: can_land, can_sea = True, True
        base_name = "NxVizer"; name = base_name; counter = 1
        alive_names = {a.name for a in nxers.values() if a.alive}
        while name in alive_names: name = f"{base_name}{counter}"; counter += 1
        a = NxEr(id=(max(nxers.keys()) + 1) if nxers else 0, name=name, color=_rand_color(list(used_colors)), pos=pos, can_land=can_land, can_sea=can_sea, net=net, food=float(StartFood), is_male=random.random() < 0.5, alive=True, born_ts=time.time(), died_ts=None, last_inputs=(0, 0, 0), ticks_per_action=max(1, int(GlobalTimeSteps / max(1, params.simulation_steps))), tick_accum=0, harvesting=None, mating_with=None, mating_end_tick=None, stats=NxErStats(), visited=set([pos]), dopamine_boost_ticks=0, _last_O4=0, mating_intent_until_tick=0, parents=(None, None), mate_cooldown_until_tick=0, last_move_tick=step_tick, last_pos=pos)
        used_colors.add(a.color); nxers[a.id] = a; occupied.add(a.pos)
        print(f"[LOAD NxVizer] spawned {a.name} at {a.pos}")
        
    def save_state(name=None):
        """Saves the entire simulation state to a single JSON file."""
        name = name or f"nx_world_save_{_now_str()}.json"
        data = {"meta": {"created": _now_str(), "step_tick": step_tick, "GlobalTimeSteps": GlobalTimeSteps, "births_count": births_count, "deaths_count": deaths_count, "game_index": game_index}, "world": {"N": world.N, "grid": world.grid}, "nxers": [{"id": a.id, "name": a.name, "color": a.color, "pos": a.pos, "can_land": a.can_land, "can_sea": a.can_sea, "food": a.food, "is_male": a.is_male, "alive": a.alive, "born_ts": a.born_ts, "died_ts": a.died_ts, "last_inputs": a.last_inputs, "ticks_per_action": a.ticks_per_action, "tick_accum": a.tick_accum, "harvesting": a.harvesting, "mating_with": a.mating_with, "mating_end_tick": a.mating_end_tick, "visited": list(a.visited), "dopamine_boost_ticks": a.dopamine_boost_ticks, "_last_O4": a._last_O4, "mating_intent_until_tick": a.mating_intent_until_tick, "parents": list(a.parents) if a.parents else [None, None], "mate_cooldown_until_tick": a.mate_cooldown_until_tick, "last_move_tick": a.last_move_tick, "last_pos": a.last_pos, "stats": asdict(a.stats), "net": a.net.to_dict()} for a in nxers.values()], "foods": [{"id": f.id, "anchor": f.anchor, "pos": f.pos, "alive": f.alive, "respawn_at_tick": f.respawn_at_tick, "remaining": f.remaining, "progress": f.progress} for f in foods.values()], "all_time_best": {k: [{"name": a.name, "is_male": a.is_male, "stats": asdict(a.stats), "net": a.net.to_dict(), "can_land": a.can_land, "can_sea": a.can_sea} for a in v] for k, v in all_time_best.items()}}
        path = _safe_path(name)
        with open(path, "w") as f: json.dump(data, f)
        print(f"[SAVE] {path}")
    
    def load_state(path):
        """Loads and restores the entire simulation state from a JSON file."""
        nonlocal step_tick, nxers, foods, occupied, births_count, deaths_count, world, game_index, all_time_best
        with open(path, "r") as f: data = json.load(f)
        step_tick = data["meta"]["step_tick"]
        births_count = int(data["meta"].get("births_count", 0)); deaths_count = int(data["meta"].get("deaths_count", 0)); game_index = int(data["meta"].get("game_index", 1))
        world.grid = data["world"]["grid"]; world.N = len(world.grid)
        renderer.pan = [world.N * 0.5, world.N * 0.5]; renderer.zoom = max(2.0, 800.0 / world.N)
        if "all_time_best" in data:
            all_time_best = {k: [] for k in data["all_time_best"]}
            for category, champs in data["all_time_best"].items():
                for champ_data in champs:
                    net = _rebuild_net_from_dict(champ_data["net"])
                    champ_nxer = NxEr(id=-1, name=champ_data["name"], color=(200, 200, 200), pos=(0, 0), can_land=champ_data.get("can_land", True), can_sea=champ_data.get("can_sea", False), net=net, food=0, is_male=champ_data.get("is_male", random.random() < 0.5), stats=NxErStats(**champ_data["stats"]))
                    all_time_best[category].append(champ_nxer)
        nxers = {}; occupied = set()
        for nd in data["nxers"]:
            net = _rebuild_net_from_dict(nd["net"])
            pos_wrapped = wrap_pos(tuple(nd["pos"]))
            a = NxEr(id=nd["id"], name=nd["name"], color=tuple(nd["color"]), pos=pos_wrapped, can_land=nd["can_land"], can_sea=nd["can_sea"], net=net, food=float(nd["food"]), is_male=nd.get("is_male", random.random() < 0.5), alive=nd["alive"], born_ts=nd["born_ts"], died_ts=nd["died_ts"], last_inputs=tuple(nd["last_inputs"]), ticks_per_action=int(nd["ticks_per_action"]), tick_accum=int(nd["tick_accum"]), harvesting=nd["harvesting"], mating_with=nd["mating_with"], mating_end_tick=nd["mating_end_tick"], stats=NxErStats(**nd["stats"]), visited=set(map(tuple, nd.get("visited", []))), dopamine_boost_ticks=int(nd.get("dopamine_boost_ticks", 0)), _last_O4=int(nd.get("_last_O4", 0)), mating_intent_until_tick=int(nd.get("mating_intent_until_tick", 0)), parents=tuple(nd.get("parents", [None, None])), mate_cooldown_until_tick=int(nd.get("mate_cooldown_until_tick", 0)), last_move_tick=int(nd.get("last_move_tick", step_tick)), last_pos=tuple(nd.get("last_pos", pos_wrapped)))
            nxers[a.id] = a
            if a.alive: occupied.add(a.pos)
        foods = {}
        for fd in data["foods"]:
            f = Food(id=fd["id"], anchor=wrap_pos(tuple(fd["anchor"])), pos=wrap_pos(tuple(fd["pos"])), alive=fd["alive"], respawn_at_tick=fd["respawn_at_tick"], remaining=int(fd.get("remaining", 25)), progress={int(k): int(v) for k, v in fd.get("progress", {}).items()})
            foods[f.id] = f
            
    def schedule_respawn(food: Food, cur_tick: int):
        """Schedules a depleted food source to respawn after a delay."""
        food.alive = False
        food.respawn_at_tick = cur_tick + FoodRespan * GlobalTimeSteps
        food.progress.clear()
        
    def try_respawns(cur_tick: int):
        """Checks for and respawns food sources whose respawn timer has elapsed."""
        if sum(1 for f in foods.values() if f.alive) >= MaxFood: return
        for f in foods.values():
            if not f.alive and f.respawn_at_tick and cur_tick >= f.respawn_at_tick:
                p = find_free(allow_sea=True, allow_land=True, near=f.anchor, search_radius=6)
                if p:
                    f.pos = p; f.alive = True; f.respawn_at_tick = None; f.remaining = 25; f.progress.clear()
                if sum(1 for ff in foods.values() if ff.alive) >= MaxFood: break
                
    def update_all_time_best():
        """Updates the hall-of-fame list of the best agents ever seen across all games."""
        nonlocal all_time_best
        current_agents = list(nxers.values())
        if not current_agents: return
        for a in current_agents: # Recalculate fitness scores.
            energy_status = a.net.get_energy_status()
            energy_efficiency = energy_status.get('efficiency', 0.0); branching_ratio = energy_status.get('branching_ratio', 1.0)
            normalized_food = min(a.stats.food_found / 100.0, 1.0); normalized_explored = min(a.stats.explored / 1000.0, 1.0)
            normalized_time = min(a.stats.time_lived_s / 1000.0, 1.0); normalized_energy = min(energy_efficiency / 10.0, 1.0) if energy_efficiency else 0.0
            normalized_sync = min(branching_ratio / 2.0, 1.0)
            a.stats.fitness_score = normalized_food * 0.3 + normalized_explored * 0.2 + normalized_time * 0.2 + normalized_energy * 0.15 + normalized_sync * 0.15
        categories = {'food_found': sorted(current_agents, key=lambda a: a.stats.food_found, reverse=True)[:3], 'food_taken': sorted(current_agents, key=lambda a: a.stats.food_taken, reverse=True)[:3], 'explored': sorted(current_agents, key=lambda a: a.stats.explored, reverse=True)[:3], 'time_lived_s': sorted(current_agents, key=lambda a: a.stats.time_lived_s, reverse=True)[:3], 'mates_performed': sorted(current_agents, key=lambda a: a.stats.mates_performed, reverse=True)[:3], 'fitness_score': sorted(current_agents, key=lambda a: a.stats.fitness_score, reverse=True)[:3]}
        for stat_name, top_champs in categories.items():
            combined_champs = all_time_best.get(stat_name, []) + top_champs
            seen_names = set()
            unique_champs = []
            for champ in combined_champs: # Ensure only one entry per unique agent lineage.
                base_name = _strip_leading_digits(champ.name).lstrip('-') or champ.name
                if base_name not in seen_names:
                    seen_names.add(base_name)
                    unique_champs.append(champ)
            all_time_best[stat_name] = sorted(unique_champs, key=lambda a: getattr(a.stats, stat_name), reverse=True)[:5]
            
    def rankings():
        """Generates the formatted data for the HUD ranking panel."""
        all_nxers = list(nxers.values())
        if not all_nxers: return {}
        food_found = sorted(all_nxers, key=lambda a: a.stats.food_found, reverse=True)
        explored = sorted(all_nxers, key=lambda a: a.stats.explored, reverse=True)
        lived = sorted(all_nxers, key=lambda a: a.stats.time_lived_s, reverse=True)
        mated = sorted(all_nxers, key=lambda a: a.stats.mates_performed, reverse=True)
        food_taken = sorted(all_nxers, key=lambda a: a.stats.food_taken, reverse=True)
        fitness = sorted(all_nxers, key=lambda a: a.stats.fitness_score, reverse=True)
        fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)
        def format_entry(agent, value): return (f"{agent.name} [Die]" if not agent.alive else agent.name, fmt(value))
        return {"Food found": [format_entry(a, a.stats.food_found) for a in food_found], "Food taken": [format_entry(a, a.stats.food_taken) for a in food_taken], "World explored": [format_entry(a, a.stats.explored) for a in explored], "Time lived (s)": [format_entry(a, a.stats.time_lived_s) for a in lived], "Mates": [format_entry(a, a.stats.mates_performed) for a in mated], "Fitness": [format_entry(a, a.stats.fitness_score) for a in fitness]}
    
    def _is_parent_child(A: NxEr, B: NxEr) -> bool: return (A.id in (B.parents or (None, None))) or (B.id in (A.parents or (None, None)))
    
    def can_mate(A: NxEr, B: NxEr, now_tick: int) -> bool:
        """Checks if two agents meet all the conditions to be able to mate."""
        if A.id == B.id or not A.alive or not B.alive or A.is_male == B.is_male or A.mating_with is not None or B.mating_with is not None or A.food < 5 or B.food < 5 or _is_parent_child(A, B) or now_tick < A.mate_cooldown_until_tick or now_tick < B.mate_cooldown_until_tick: return False
        return True
        
    def champions_from_last_game() -> List[NxEr]:
        """Selects the best-performing agents from the completed game to carry over to the next one."""
        all_agents = list(nxers.values())
        if not all_agents: return []
        by_food = sorted(all_agents, key=lambda a: a.stats.food_found, reverse=True)[:3]
        by_expl = sorted(all_agents, key=lambda a: a.stats.explored, reverse=True)[:3]
        by_lived = sorted(all_agents, key=lambda a: a.stats.time_lived_s, reverse=True)[:3]
        by_fitness = sorted(all_agents, key=lambda a: a.stats.fitness_score, reverse=True)[:3]
        potential_champs = []
        for category_champs in all_time_best.values(): potential_champs.extend(category_champs)
        potential_champs.extend(by_food); potential_champs.extend(by_expl); potential_champs.extend(by_lived); potential_champs.extend(by_fitness)
        seen_names = set()
        unique_champs = []
        for a in potential_champs:
            base_name_with_hyphen = _strip_leading_digits(a.name)
            base_name = base_name_with_hyphen.lstrip('-') or a.name
            if base_name not in seen_names:
                seen_names.add(base_name)
                unique_champs.append(a)
        return unique_champs[:9]
    
    def restart_game_with_champions():
        """Resets the simulation world and populates it with champions from the previous round and new random agents."""
        nonlocal world, nxers, foods, occupied, births_count, deaths_count, effects, game_index, used_colors, champion_counts
        update_all_time_best()
        champs = champions_from_last_game()
        game_index += 1
        effects.clear(); births_count = 0; deaths_count = 0; used_colors = set(RESERVED_COLORS)
        world = World(NxWorldSize, NxWorldSea, NxWorldRocks, rnd_seed=None)
        renderer.world = world; renderer.pan = [world.N * 0.5, world.N * 0.5]; renderer.zoom = max(2.0, 800.0 / world.N)
        nxers = {}; occupied = set()
        place_initial_food(min(MaxFood, max(30, MaxFood // 2)))
        next_id = 0
        for a in champs:
            base_name_with_hyphen = _strip_leading_digits(a.name)
            base_name = base_name_with_hyphen.lstrip('-') or a.name
            champion_counts[base_name] = champion_counts.get(base_name, 0) + 1
            new_name = f"{champion_counts[base_name]}{base_name}"
            net_copy = _rebuild_net_from_dict(a.net.to_dict())
            allow_land, allow_sea = a.can_land, a.can_sea
            pos = find_free(allow_sea=allow_sea, allow_land=allow_land) or (random.randrange(world.N), random.randrange(world.N))
            nx = NxEr(id=next_id, name=new_name, color=_rand_color(list(used_colors)), pos=pos, can_land=allow_land, can_sea=allow_sea, net=net_copy, food=float(StartFood * 1.1), is_male=a.is_male, ticks_per_action=max(1, int(GlobalTimeSteps / max(1, net_copy.params.simulation_steps))), stats=NxErStats(), visited=set([pos]), parents=(None, None), mate_cooldown_until_tick=0, last_move_tick=0, last_pos=pos)
            used_colors.add(nx.color); nxers[nx.id] = nx; occupied.add(nx.pos); next_id += 1
        while len(nxers) < StartingNxErs:
            a = make_nxer(next_id)
            nxers[a.id] = a; occupied.add(a.pos); next_id += 1
        print(f"[RESTART] Game #{game_index} started with {len(champs)} champions and {len(nxers) - len(champs)} new NxErs.")
        
    FIXED_DT = 1.0 / GlobalTimeSteps
    accumulator = 0.0
    
    # --- The Main Simulation Loop ---
    try:
        while running:
            # --- 1. Time Management and Input Handling ---
            frame_dt = renderer.tick(60) # Get real time elapsed and cap FPS.
            accumulator += frame_dt # Add real time to the accumulator.
            
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE: running = False
                    elif ev.key == pygame.K_SPACE:
                        if not game_over: paused = not paused; renderer.clear_detail() if not paused else None
                    elif ev.key == pygame.K_s:
                        was_paused = paused; paused = True; save_state(); paused = was_paused
                    elif ev.key == pygame.K_l:
                        candidates = sorted([p for p in os.listdir(os.getcwd()) if p.startswith("nx_world_save_") and p.endswith(".json")])
                        if candidates: paused = True; load_state(candidates[-1]); paused = False; renderer.clear_detail()
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    btn = renderer.button_clicked(ev.pos)
                    if btn == "playpause":
                        if not game_over: paused = not paused; renderer.clear_detail() if not paused else None
                    elif btn == "save":
                        was_paused = paused; paused = True; save_state(); paused = was_paused
                    elif btn == "load":
                        candidates = sorted([p for p in os.listdir(os.getcwd()) if p.startswith("nx_world_save_") and p.endswith(".json")])
                        if candidates: paused = True; load_state(candidates[-1]); paused = False; renderer.clear_detail()
                    elif btn == "save_best":
                        file_mapping = {'food_found': 'BestFoodFound.json', 'food_taken': 'BestFoodTaken.json', 'explored': 'BestWorldExplorer.json', 'time_lived_s': 'BestTimeLived.json', 'mates_performed': 'BestMates.json', 'fitness_score': 'BestFitness.json'}
                        update_all_time_best()
                        for category, champs in all_time_best.items():
                            if category not in file_mapping: continue
                            filename = file_mapping[category]
                            for champ in champs[:1]: save_nxer_to_file(champ, save_name=filename)
                        print(f"[SAVE BEST] Saved champions to JSON files")
                    elif btn == "exit": running = False
                    elif btn == "restart_yes" and game_over:
                        game_over = False; paused = False; restart_game_with_champions(); step_tick = 0; accumulator = 0.0
                        renderer.clear_detail(); game_over_start_time = None; user_declined_restart = False
                    elif btn == "restart_no" and game_over:
                        user_declined_restart = True; running = False
                    else:
                        dbtn = renderer.detail_button_clicked(ev.pos)
                        if dbtn and renderer.selected_nxer_id is not None and renderer.selected_nxer_id in nxers:
                            a = nxers[renderer.selected_nxer_id]
                            if dbtn == "save_nxer": save_nxer_to_file(a, save_name=a.name)
                            elif dbtn == "load_nxer": load_nxer_from_file(spawn_near=a.pos)
                            elif dbtn == "save_nxvizer": save_nxvizer_to_file(a)
                            elif dbtn == "load_nxvizer": load_nxvizer_from_file(spawn_near=a.pos)
                        elif paused and not game_over:
                            clicked_name = renderer.ranking_clicked(ev.pos)
                            if clicked_name:
                                for a in nxers.values():
                                    if a.name == clicked_name: renderer.selected_nxer_id = a.id; break
                            else: # Handle clicking on an agent in the world view.
                                mx, my = ev.pos
                                best_id, best_d = None, 1e9
                                for a in nxers.values():
                                    if not a.alive: continue
                                    sx, sy = renderer.world_to_screen(a.pos[0], a.pos[1])
                                    d2 = (sx - mx) ** 2 + (sy - my) ** 2
                                    if d2 < best_d: best_d = d2; best_id = a.id
                                if best_id is not None and best_d <= (14 * 14): renderer.selected_nxer_id = best_id
                                else: renderer.clear_detail()
                renderer.event_zoom_rotate_pan(ev)
            
            if game_over and game_over_start_time is not None and not user_declined_restart:
                if time.time() - game_over_start_time > 120: # Auto-restart after 2 minutes.
                    game_over = False; paused = False; restart_game_with_champions(); step_tick = 0; accumulator = 0.0
                    renderer.clear_detail(); game_over_start_time = None; user_declined_restart = False
                    
            renderer.handle_input(frame_dt)
            
            # --- 2. Simulation Step Execution ---
            # This loop ensures the simulation runs at a constant rate (FIXED_DT) regardless of frame rate.
            steps_to_process = 0
            if not paused and not game_over:
                while accumulator >= FIXED_DT:
                    steps_to_process += 1
                    accumulator -= FIXED_DT
                    if steps_to_process >= 10: # Cap steps per frame to prevent a "spiral of death" if rendering is slow.
                        accumulator = 0.0
                        break
            
            for _ in range(steps_to_process):
                step_tick += 1
                
                # --- A. Update Game World State & Agent Vitals ---
                effects[:] = [ef for ef in effects if (step_tick - ef['start']) < GlobalTimeSteps]
                for a in nxers.values(): # Apply food decay and check for starvation.
                    if not a.alive: continue
                    a.stats.time_lived_s += FIXED_DT; a.food -= 0.01 * FIXED_DT
                    if a.food <= 0 and a.alive:
                        a.alive = False; a.died_ts = time.time(); deaths_count += 1
                        if a.pos in occupied: occupied.discard(a.pos)
                        push_effect('skull', a.pos)
                for a in nxers.values(): # Agents that haven't moved in a long time die.
                    if not a.alive: continue
                    if step_tick - a.last_move_tick > 10 * GlobalTimeSteps:
                        a.alive = False; a.died_ts = time.time(); deaths_count += 1
                        if a.pos in occupied: occupied.discard(a.pos)
                        push_effect('skull', a.pos)
                for a in nxers.values(): # End mating sessions.
                    if not a.alive: continue
                    if a.mating_with is not None and step_tick >= (a.mating_end_tick or 0):
                        a.mating_with = None; a.mating_end_tick = None
                        a.mate_cooldown_until_tick = max(a.mate_cooldown_until_tick, step_tick + mate_cooldown_ticks)
                        
                # --- B. Gather and Execute Network Updates in Parallel ---
                jobs = {}
                for a in nxers.values():
                    if not a.alive or a.mating_with is not None: continue
                    a.tick_accum += 1
                    if a.tick_accum >= a.ticks_per_action:
                        a.tick_accum = 0
                        if step_tick < boot_random_until: a.last_inputs = (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
                        netdict = a.net.to_dict()
                        if a.dopamine_boost_ticks > 0: # Apply temporary dopamine boost.
                            nd = netdict['neuromodulators']
                            nd['dopamine'] = max(nd.get('dopamine', 0.12), 0.9); nd['serotonin'] = max(nd.get('serotonin', 0.12), 0.6)
                            a.dopamine_boost_ticks -= 1
                        jobs[a.id] = (netdict, a.last_inputs, max(1, a.net.params.simulation_steps // GlobalTimeSteps))
                
                results: Dict[int, Tuple[dict, List[int], dict]] = {}
                if jobs:
                    items = [(aid, nd, ins, st) for aid, (nd, ins, st) in jobs.items()]
                    num_jobs = len(items); target_batches = max(1, nx_workers * 2); chunk = max(1, (num_jobs + target_batches - 1) // target_batches)
                    futures = []
                    for batch in _chunked(items, chunk): futures.append(nx_pool.submit(_net_batch_step_and_outputs, batch))
                    for fut in as_completed(futures):
                        try:
                            for aid, net_dict, outs, energy_status in fut.result(): results[aid] = (net_dict, outs, energy_status)
                        except Exception as e: print(f"Error in worker future: {e}")
                
                # --- C. Apply Network Outputs to Agent Actions ---
                for aid, (net_dict, outs, energy_status) in results.items():
                    a = nxers.get(aid)
                    if not a or not a.alive: continue
                    a.net = _rebuild_net_from_dict(net_dict) # Update the agent's brain with the new state from the worker.
                    if energy_status:
                        a.stats.energy_efficiency = energy_status.get('efficiency', a.stats.energy_efficiency)
                        a.stats.temporal_sync_score = energy_status.get('branching_ratio', a.stats.temporal_sync_score)
                        normalized_food = min(a.stats.food_found / 100.0, 1.0); normalized_explored = min(a.stats.explored / 1000.0, 1.0)
                        normalized_time = min(a.stats.time_lived_s / 1000.0, 1.0); normalized_energy = min(a.stats.energy_efficiency / 10.0, 1.0) if a.stats.energy_efficiency else 0.0
                        normalized_sync = min(a.stats.temporal_sync_score / 2.0, 1.0)
                        a.stats.fitness_score = normalized_food * 0.3 + normalized_explored * 0.2 + normalized_time * 0.2 + normalized_energy * 0.15 + normalized_sync * 0.15
                    o = (outs + [0, 0, 0, 0])[:4]
                    O1, O2, O3, O4 = o # Decode the 4 output neurons into actions.
                    # Add some randomness to prevent agents from getting stuck.
                    if O1 == 0 and O2 == 0 and random.random() < 0.4:
                        if random.random() < 0.5: O1 = random.choice([-1, 1])
                        else: O2 = random.choice([-1, 1])
                    if O4 == 0 and random.random() < 0.08: O4 = random.choice([-1, 1])
                    dx = -O1; dy = -O2 # O1/O2 control movement direction.
                    a._pending_move = (dx, dy, O3, O4) # O3: Share/Cooperate, O4: Mate/Attack
                
                # --- D. Resolve Agent Interactions and Movements ---
                intents = []; move_target = {}
                for a in nxers.values(): # Gather all intended moves.
                    if not a.alive: continue
                    pm = getattr(a, "_pending_move", None)
                    if pm is None: a.last_inputs = (-1, 0, (1 if world.terrain(a.pos) == T_LAND else (0 if world.terrain(a.pos) == T_SEA else -1))); continue
                    dx, dy, O3, O4 = pm; delattr(a, "_pending_move")
                    if dx == 0 and dy == 0: a.last_inputs = (-1, 0, (1 if world.terrain(a.pos) == T_LAND else (0 if world.terrain(a.pos) == T_SEA else -1))); continue
                    tx, ty = wrap_pos((a.pos[0] + dx, a.pos[1] + dy))
                    intents.append((a.id, (tx, ty), O3, O4)); move_target[a.id] = (tx, ty)
                    
                by_pos: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
                for aid, tgt, O3, O4 in intents: by_pos.setdefault(tgt, []).append((aid, O3, O4))
                
                occupant_at = {a.pos: a.id for a in nxers.values() if a.alive}; food_pos_to_id = {f.pos: f.id for f in foods.values() if f.alive}
                valid_intents = []
                for tgt, lst in by_pos.items(): # Filter out invalid moves (e.g., onto rocks).
                    tt = world.terrain(tgt)
                    if tt == T_ROCK:
                        for (aid, _, _) in lst: nxers[aid].last_inputs = (-1, 0, -1)
                        continue
                    for (aid, O3, O4) in lst:
                        a = nxers[aid]
                        origin_terrain = world.terrain(a.pos); target_terrain = tt
                        is_seashore_crossing = (origin_terrain == T_LAND and target_terrain == T_SEA) or (origin_terrain == T_SEA and target_terrain == T_LAND)
                        if not is_seashore_crossing:
                            if target_terrain == T_LAND and not a.can_land: a.last_inputs = (-1, 0, 1); continue
                            if target_terrain == T_SEA and not a.can_sea: a.last_inputs = (-1, 0, 0); continue
                        valid_intents.append((tgt, aid, O3, O4, tt))
                
                handled_swap = set()
                for aid, tgt, O3, O4 in intents: # Handle head-on collisions (swaps).
                    if aid in handled_swap: continue
                    occ = occupant_at.get(tgt)
                    if occ is None: continue
                    A = nxers[aid]; B = nxers[occ]
                    if A.id == B.id: continue
                    # Interaction gives a dopamine boost.
                    A.dopamine_boost_ticks = max(A.dopamine_boost_ticks, int(3 * GlobalTimeSteps)); B.dopamine_boost_ticks = max(B.dopamine_boost_ticks, int(3 * GlobalTimeSteps))
                    A.last_inputs = (-1, 1, (1 if world.terrain(tgt) == T_LAND else 0)); B.last_inputs = (-1, 1, (1 if world.terrain(A.pos) == T_LAND else 0))
                    if O4 == 1 or random.random() < 0.03: A.mating_intent_until_tick = step_tick + 3 * GlobalTimeSteps
                    if getattr(B, "_last_O4", 0) == 1 or random.random() < 0.03: B.mating_intent_until_tick = step_tick + 3 * GlobalTimeSteps
                    # If both have mating intent and can mate, they mate.
                    if (A.mating_intent_until_tick > step_tick and B.mating_intent_until_tick > step_tick and can_mate(A, B, step_tick)):
                        A.mating_with = B.id; B.mating_with = A.id; dur = max(A.net.params.simulation_steps, B.net.params.simulation_steps)
                        A.mating_end_tick = step_tick + dur; B.mating_end_tick = step_tick + dur; A.food -= 1; B.food -= 1
                        push_effect('heart', A.pos); spawn_child(A, B, A.pos)
                    # If O4 is -1 (attack) or one agent is starving, food can be taken.
                    if (O4 == -1 or O3 == -1 or (A.food < 2 and B.food > 3 and random.random() < 0.3)) and B.food > 0:
                        B.food -= 1; A.food += 1; A.stats.food_taken += 1
                    A._last_O4 = O4; handled_swap.add(aid); handled_swap.add(occ)
                    
                winners = []; tgt_map: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
                for tgt, aid, O3, O4, tt in valid_intents: tgt_map.setdefault(tgt, []).append((aid, O3, O4, tt))
                for tgt, lst in tgt_map.items():
                    fid = food_pos_to_id.get(tgt)
                    if fid is not None:
                        f = foods[fid]
                        lst_ordered = sorted(lst, key=lambda it: nxers[it[0]].ticks_per_action)
                        for (aid, _, _, _) in lst_ordered: nxers[aid].dopamine_boost_ticks = max(nxers[aid].dopamine_boost_ticks, int(1.5 * GlobalTimeSteps))
                        for (aid, O3, O4, tt) in lst_ordered:
                            a = nxers[aid]
                            a.last_inputs = (1, 0, (1 if tt == T_LAND else 0))
                            if f.remaining > 0:
                                f.progress[aid] = f.progress.get(aid, 0) + 1
                                f.remaining -= 1
                                a.food += 1.0
                                a.stats.food_found += 1.0
                                a.food -= 0.1
                                if a.food <= 0 and a.alive:
                                    a.alive = False
                                    a.died_ts = time.time()
                                    deaths_count += 1
                                    if a.pos in occupied: occupied.discard(a.pos)
                                    push_effect('skull', a.pos)
                        if f.remaining <= 0:
                            winner_id = None
                            if f.progress:
                                max_prog = max(f.progress.values())
                                candidates = [k for k, v in f.progress.items() if v == max_prog]
                                winner_id = random.choice(candidates)
                            wnx = nxers.get(winner_id) if winner_id is not None else None
                            if wnx and wnx.alive:
                                if wnx.pos in occupied: occupied.discard(wnx.pos)
                                wnx.pos = f.pos
                                occupied.add(wnx.pos)
                                wnx.visited.add(wnx.pos)
                                wnx.stats.explored = len(wnx.visited)
                            schedule_respawn(f, step_tick)
                        continue
                    occ = occupant_at.get(tgt)
                    if occ is not None:
                        for (aid, O3, O4, tt) in lst:
                            a = nxers[aid]
                            b = nxers.get(occ)
                            if not b: continue
                            a.last_inputs = (-1, 1, (1 if tt == T_LAND else 0))
                            if not b.alive or a.id == b.id: continue
                            a.dopamine_boost_ticks = max(a.dopamine_boost_ticks, int(3 * GlobalTimeSteps))
                            b.dopamine_boost_ticks = max(b.dopamine_boost_ticks, int(3 * GlobalTimeSteps))
                            if O4 == 1 or random.random() < 0.03: a.mating_intent_until_tick = step_tick + 3 * GlobalTimeSteps
                            if b._last_O4 == 1 or random.random() < 0.03: b.mating_intent_until_tick = step_tick + 3 * GlobalTimeSteps
                            if (a.mating_intent_until_tick > step_tick and b.mating_intent_until_tick > step_tick and can_mate(a, b, step_tick)):
                                a.mating_with = b.id
                                b.mating_with = a.id
                                dur = max(a.net.params.simulation_steps, b.net.params.simulation_steps)
                                a.mating_end_tick = step_tick + dur
                                b.mating_end_tick = step_tick + dur
                                a.food -= 1
                                b.food -= 1
                                push_effect('heart', tgt)
                                spawn_child(a, b, a.pos)
                            elif (O4 == -1 or O3 == -1 or (a.food < 2 and b.food > 3 and random.random() < 0.3)) and b.food > 0:
                                b.food -= 1
                                a.food += 1
                                a.stats.food_taken += 1
                            a._last_O4 = O4
                        continue
                    contenders = lst[:]
                    for (aid, _, _, _) in contenders: nxers[aid].dopamine_boost_ticks = max(nxers[aid].dopamine_boost_ticks, int(2.5 * GlobalTimeSteps))
                    want = []
                    for (aid, O3, O4, tt) in contenders:
                        a = nxers[aid]
                        if ((O4 == 1 or a.mating_intent_until_tick > step_tick or random.random() < 0.02) and a.food >= 5 and a.mating_with is None):
                            want.append(aid)
                    if len(want) >= 2:
                        want = list(dict.fromkeys(want))
                        A = nxers[want[0]]
                        B = nxers[want[1]]
                        if A.id != B.id and can_mate(A, B, step_tick):
                            dur = max(A.net.params.simulation_steps, B.net.params.simulation_steps)
                            A.mating_with = B.id
                            B.mating_with = A.id
                            A.mating_end_tick = step_tick + dur
                            B.mating_end_tick = step_tick + dur
                            A.food -= 1
                            B.food -= 1
                            push_effect('heart', tgt)
                            spawn_child(A, B, tgt)
                        for (aid, O3, O4, tt) in contenders:
                            nxers[aid]._last_O4 = O4
                            nxers[aid].last_inputs = (-1, 1, (1 if tt == T_LAND else 0))
                        continue
                    want = contenders
                    if len(want) > 1:
                        for (aid, O3, O4, tt) in want:
                            giver = nxers[aid]
                            if O3 == 1 and giver.food > 1.0:
                                slowest = max(want, key=lambda it: nxers[it[0]].ticks_per_action)[0]
                                if slowest != aid:
                                    giver.food -= 1.0
                                    nxers[slowest].food += 1.0
                        want.sort(key=lambda it: nxers[it[0]].ticks_per_action)
                        top = [w for w in want if nxers[w[0]].ticks_per_action == nxers[want[0][0]].ticks_per_action]
                        winner = random.choice(top)
                        winners.append((winner[0], tgt, winner[3]))
                        for (aid, O3, O4, tt) in want:
                            if aid == winner[0]: continue
                            a = nxers[aid]
                            a.last_inputs = (-1, 1, (1 if tt == T_LAND else 0))
                            a._last_O4 = O4
                    elif want:
                        (aid, O3, O4, tt) = want[0]
                        winners.append((aid, tgt, tt))
                        nxers[aid]._last_O4 = O4
                for (aid, tgt, tt) in winners:
                    a = nxers.get(aid)
                    if not a or not a.alive: continue
                    if tgt in occupied:
                        a.last_inputs = (-1, 1, (1 if tt == T_LAND else 0))
                    else:
                        if a.pos in occupied: occupied.discard(a.pos)
                        if a.pos != tgt:
                            a.last_move_tick = step_tick
                            a.last_pos = a.pos
                        a.pos = tgt
                        occupied.add(a.pos)
                        a.visited.add(a.pos)
                        a.stats.explored = len(a.visited)
                        a.last_inputs = (-1, 0, (1 if tt == T_LAND else 0))
                        a.food -= 0.1
                        if a.food <= 0 and a.alive:
                            a.alive = False
                            a.died_ts = time.time()
                            deaths_count += 1
                            if a.pos in occupied: occupied.discard(a.pos)
                            push_effect('skull', a.pos)
                try_respawns(step_tick)
            # --- 3. Rendering ---
            alive_count = sum(1 for a in nxers.values() if a.alive)
            if alive_count == 0 and not game_over:
                paused = True; game_over = True; game_over_start_time = time.time(); user_declined_restart = False
            best_scores = {}
            title_to_stat = {"Food found": "food_found", "Food taken": "food_taken", "World explored": "explored", "Time lived (s)": "time_lived_s", "Mates": "mates_performed", "Fitness": "fitness_score"}
            for title, stat_name in title_to_stat.items():
                all_candidates = list(nxers.values())
                if stat_name in all_time_best: all_candidates.extend(all_time_best[stat_name])
                if not all_candidates: best_scores[title] = 0
                else: best_scores[title] = max(getattr(a.stats, stat_name) for a in all_candidates)
            renderer.draw_world(foods, nxers, rankings(), alive_count, deaths_count, births_count, paused, effects, step_tick, GlobalTimeSteps, game_over, game_index, best_scores)
    except Exception as ex:
        print("Fatal error:", ex)
        import traceback; traceback.print_exc()
    finally:
        # Ensure the worker pool is shut down gracefully on exit.
        try: nx_pool.shutdown(wait=False, cancel_futures=True)
        except: pass
        pygame.quit()

def run_config_screen() -> Optional[Dict[str, any]]:
    """
    Displays a pre-simulation configuration screen using Pygame, allowing the user
    to adjust key parameters with sliders before starting the game.
    """
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    pygame.display.set_caption("Neuraxon Game Of Life v 1.0 By David Vivancos & Dr Jose Sanchez for Qubic Science")
    # Display a splash screen for a few seconds.
    try:
        splash_image_path = "GameOfLifeByDavidVivancosAndJoseSanchez.png"
        if os.path.exists(splash_image_path):
            splash_image = pygame.image.load(splash_image_path).convert_alpha()
            img_width, img_height = splash_image.get_size()
            screen_width, screen_height = screen.get_size()
            scale = min(screen_width / img_width, screen_height / img_height)
            new_width = int(img_width * scale); new_height = int(img_height * scale)
            splash_image = pygame.transform.scale(splash_image, (new_width, new_height))
            x_pos = (screen_width - new_width) // 2; y_pos = (screen_height - new_height) // 2
            start_time = time.time(); clock = pygame.time.Clock()
            while time.time() - start_time < 10.0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); return None
                    if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN): start_time = 0; break
                screen.fill((0, 0, 0)); screen.blit(splash_image, (x_pos, y_pos)); pygame.display.flip(); clock.tick(60)
    except Exception as e: print(f"Could not load splash screen: {e}")
    
    pygame.display.set_caption("Neuraxon Game Of Life v 1.0 By David Vivancos & Dr Jose Sanchez for Qubic Science - Configuration")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16); title_font = pygame.font.SysFont("consolas", 32, bold=True)
    # Define the parameters that will be configurable via sliders.
    param_specs = [("World Size", 30, 100, 50, True, lambda x: x), ("Sea Percentage", 20, 80, 55, True, lambda x: x / 100.0), ("Rock Percentage", 1, 10, 2, True, lambda x: x / 100.0), ("Starting NxErs", 10, 100, 70, True, lambda x: x), ("Food Sources", 50, 300, 250, True, lambda x: x), ("Food Respawn", 200, 600, 400, True, lambda x: x), ("Start Food", 25, 100, 50, True, lambda x: float(x)), ("Max Neurons", 5, 25, 12, True, lambda x: x), ("Global Time Steps", 30, 90, 60, True, lambda x: x), ("Mate Cooldown (sec)", 6, 20, 12, True, lambda x: x)]
    screen_width, screen_height = screen.get_size()
    slider_container_width = 700; slider_width = 600
    slider_start_x = (screen_width - slider_container_width) // 2 + (slider_container_width - slider_width) // 2
    start_y = 200; slider_height = 50
    sliders = []
    for i, (label, min_val, max_val, default_val, is_int, _) in enumerate(param_specs):
        rect = pygame.Rect(slider_start_x, start_y + i * slider_height, slider_width, 20)
        sliders.append(Slider(rect, min_val, max_val, default_val, label, is_int))
    play_button_width = 250; play_button_height = 50
    play_button_x = (screen_width - play_button_width) // 2; play_button_y = 950
    play_button_rect = pygame.Rect(play_button_x, play_button_y, play_button_width, play_button_height)
    running = True
    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return None
            for slider in sliders: slider.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if play_button_rect.collidepoint(event.pos):
                    params = {}
                    for i, slider in enumerate(sliders): # Collect values from all sliders.
                        raw_value = slider.get_value()
                        conversion_func = param_specs[i][5]
                        param_name = ["NxWorldSize", "NxWorldSea", "NxWorldRocks", "StartingNxErs", "MaxFood", "FoodRespan", "StartFood", "MaxNeurons", "GlobalTimeSteps", "MateCooldownSeconds"][i]
                        params[param_name] = conversion_func(raw_value)
                    return params # Return the dictionary of parameters to the main function.
        screen.fill((15, 15, 18))
        title_surf = title_font.render("Neuraxon Game Of Life - World Configuration", True, (235, 235, 240))
        screen.blit(title_surf, (screen.get_width() // 2 - title_surf.get_width() // 2, 50))
        for slider in sliders: slider.draw(screen, font)
        instr_text = font.render("Adjust parameters with sliders, then click 'Start Game'", True, (180, 180, 180))
        screen.blit(instr_text, (screen.get_width() // 2 - instr_text.get_width() // 2, 900))
        pygame.draw.rect(screen, (35, 180, 60), play_button_rect, border_radius=8)
        pygame.draw.rect(screen, (60, 220, 90), play_button_rect, 2, border_radius=8)
        play_text = font.render("Start Game", True, (255, 255, 255))
        screen.blit(play_text, (play_button_rect.x + (play_button_rect.width - play_text.get_width()) // 2, play_button_rect.y + (play_button_rect.height - play_text.get_height()) // 2))
        pygame.display.flip()

if __name__ == "__main__":
    # This is required for multiprocessing to work correctly when the script is packaged into an executable.
    mp.freeze_support()
    # First, show the configuration screen to the user.
    config_params = run_config_screen()
    # If the user confirmed the settings start the main simulation.
    if config_params is not None:
        GameOfLife(**config_params)
    else:
        # If the user closed the configuration window, exit gracefully.
        print("Game cancelled by user.")
        pygame.quit()
        sys.exit(0)

