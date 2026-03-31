# Neuraxon Game of Life 3.5 Neuron Genetics (Neuraxon 2.0 Compliant) Internal version 125
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife

import random
import math
from typing import Any, Tuple, Optional
import numpy as np

# Import local modules
from config import NetworkParameters
from .network import NeuraxonNetwork

# New Version 2.2: Enhanced Full Feldged Inheritance
def Inheritance(father: 'NxEr', mother: 'NxEr') -> 'NeuraxonNetwork':
    """
    Creates a child NeuraxonNetwork by inheriting 50% of ALL parameters from each parent.
    
    HANDLES DIFFERENT-SIZED NETWORKS:
    - Uses proportional mapping for neurons (not modulo)
    - Intelligently matches synapses by relative position
    - Weighted sampling when structures don't align
    
    Args:
        father: The male NxEr parent
        mother: The female NxEr parent
    
    Returns:
        A new NeuraxonNetwork with 50% inheritance from each parent
    """
    
    father_net = father.net
    mother_net = mother.net
    father_params = father_net.params
    mother_params = mother_net.params
    
    # ==========================================================================
    # DETERMINE FITTER PARENT (NEW in v2.34)
    # ==========================================================================
    father_fitness = father.stats.food_found + father.stats.time_lived_s * 0.1 + father.stats.explored * 0.05
    mother_fitness = mother.stats.food_found + mother.stats.time_lived_s * 0.1 + mother.stats.explored * 0.05
    father_is_fitter = father_fitness >= mother_fitness
    
    # ==========================================================================
    # HELPER FUNCTIONS
    # ==========================================================================
    
    def pick(father_val, mother_val):
        """50% chance to pick from father or mother."""
        return father_val if random.random() < 0.5 else mother_val
    
    def pick_biased(father_val, mother_val, bias: float = 0.75):
        """Pick from fitter parent with bias probability."""
        if father_is_fitter:
            return father_val if random.random() < bias else mother_val
        else:
            return mother_val if random.random() < bias else father_val
    
    def blend(father_val, mother_val, variation: float = 0.03):
        """v111 FIX (M21/M23): Blend biased toward fitter parent.
        
        OLD: 50/50 average + 5% noise → regression to mean, zero heritability.
        NEW: 70/30 weighted toward fitter parent + 3% noise.
        BIOINSPIRED: Biological inheritance preserves more of the fitter
        genotype via dominance effects. Paper §7 Algorithm 7 prescribes
        'mutated deep-copies of top half' — children should resemble
        successful parents, not average the population.
        """
        if father_is_fitter:
            avg = father_val * 0.7 + mother_val * 0.3
        else:
            avg = mother_val * 0.7 + father_val * 0.3
        return avg * random.uniform(1.0 - variation, 1.0 + variation)
    
    def blend_int(father_val, mother_val, variation: float = 0.1):
        """Blend two integer values, returning an integer result."""
        result = blend(father_val, mother_val, variation)
        return max(1, int(round(result)))
    
    def blend_bounded(father_val, mother_val, low: float, high: float, variation: float = 0.05):
        """v111 FIX (M21/M23): Reduced default variation 0.05→0.03."""
        result = blend(father_val, mother_val, min(variation, 0.03))
        return max(low, min(high, result))
    
    def pick_or_blend(father_val, mother_val, blend_prob: float = 0.7):
        """v111 FIX (M21/M23): Replaced pick() with pick_biased() for non-blend path."""
        if random.random() < blend_prob:
            return blend(father_val, mother_val)
        return pick_biased(father_val, mother_val)
    
    def proportional_index(child_idx: int, child_count: int, parent_count: int) -> int:
        """
        Map a child index to a parent index using proportional mapping.
        
        Example: If child has 15 neurons and parent has 10:
        - child[0] -> parent[0]
        - child[7] -> parent[4 or 5]
        - child[14] -> parent[9]
        
        This preserves the relative position in the network.
        """
        if parent_count == 0:
            return 0
        if child_count <= 1:
            return 0
        # Map child position [0, child_count-1] to parent position [0, parent_count-1]
        ratio = child_idx / (child_count - 1) if child_count > 1 else 0
        parent_idx = int(round(ratio * (parent_count - 1)))
        return min(parent_idx, parent_count - 1)
    
    def get_parent_neuron_pair(child_idx: int, child_count: int, 
                                father_neurons: list, mother_neurons: list) -> Tuple[Any, Any]:
        """
        Get the corresponding parent neurons for a child neuron index.
        Uses proportional mapping to handle different sizes.
        
        Returns (father_neuron, mother_neuron) - either can be None if parent list is empty.
        """
        f_neuron = None
        m_neuron = None
        
        if father_neurons:
            f_idx = proportional_index(child_idx, child_count, len(father_neurons))
            f_neuron = father_neurons[f_idx]
        
        if mother_neurons:
            m_idx = proportional_index(child_idx, child_count, len(mother_neurons))
            m_neuron = mother_neurons[m_idx]
        
        return f_neuron, m_neuron
    
    def weighted_random_choice(father_list: list, mother_list: list) -> Any:
        """
        Pick a random element from combined parent lists (50/50 parent selection).
        """
        # First decide which parent to sample from
        use_father = random.random() < 0.5
        
        if use_father and father_list:
            return random.choice(father_list)
        elif mother_list:
            return random.choice(mother_list)
        elif father_list:
            return random.choice(father_list)
        return None
    
    def find_similar_synapse(child_pre_id: int, child_post_id: int,
                             child_neuron_count: int,
                             parent_synapses: list,
                             parent_neuron_count: int) -> Optional[Any]:
        """
        Find a synapse in parent that has similar relative pre/post positions.
        
        This handles the case where child and parent have different neuron counts
        by matching based on relative position rather than absolute IDs.
        """
        if not parent_synapses or parent_neuron_count == 0:
            return None
        
        # Calculate relative positions for child synapse
        child_pre_rel = child_pre_id / max(1, child_neuron_count - 1)
        child_post_rel = child_post_id / max(1, child_neuron_count - 1)
        
        # Find the parent synapse with closest relative positions
        best_synapse = None
        best_distance = float('inf')
        
        for syn in parent_synapses:
            parent_pre_rel = syn.pre_id / max(1, parent_neuron_count - 1)
            parent_post_rel = syn.post_id / max(1, parent_neuron_count - 1)
            
            # Euclidean distance in relative position space
            distance = math.sqrt(
                (child_pre_rel - parent_pre_rel) ** 2 + 
                (child_post_rel - parent_post_rel) ** 2
            )
            
            if distance < best_distance:
                best_distance = distance
                best_synapse = syn
        
        # Only use if reasonably close (threshold: 0.3 in relative space)
        if best_distance < 0.3:
            return best_synapse
        
        # Fallback to random selection
        return random.choice(parent_synapses)
    
    # ==========================================================================
    # LEVEL 1: INHERIT NETWORK PARAMETERS
    # ==========================================================================
    
    child_params = NetworkParameters()
    
    # --- General Network Architecture ---
    child_params.network_name = pick(father_params.network_name, mother_params.network_name)
    child_params.num_input_neurons = 9  # FIXED v3.1: Always 9 inputs
    child_params.num_hidden_neurons = blend_int(father_params.num_hidden_neurons, mother_params.num_hidden_neurons)
    child_params.num_output_neurons = 6  # FIXED v3.1: Always 6 outputs
    
    # --- Temporal Dynamics & Simulation ---
    child_params.dt = pick_or_blend(father_params.dt, mother_params.dt)
    child_params.min_dt = pick_or_blend(father_params.min_dt, mother_params.min_dt)
    child_params.max_dt = pick_or_blend(father_params.max_dt, mother_params.max_dt)
    child_params.activity_threshold = blend_bounded(father_params.activity_threshold, mother_params.activity_threshold, 0.1, 0.9)
    child_params.simulation_steps = blend_int(father_params.simulation_steps, mother_params.simulation_steps)
    
    # --- Network Topology ---
    child_params.connection_probability = blend_bounded(father_params.connection_probability, mother_params.connection_probability, 0.05, 0.5)
    child_params.small_world_k = blend_int(father_params.small_world_k, mother_params.small_world_k)
    child_params.small_world_rewire_prob = blend_bounded(father_params.small_world_rewire_prob, mother_params.small_world_rewire_prob, 0.0, 0.5)
    child_params.preferential_attachment = pick(father_params.preferential_attachment, mother_params.preferential_attachment)
    
    # --- Core Neuron Properties ---
    child_params.membrane_time_constant = pick_or_blend(father_params.membrane_time_constant, mother_params.membrane_time_constant)
    # v111 FIX (M23): Firing thresholds use pick_biased, not blend.
    # MUTATIONAL MELTDOWN ROOT CAUSE: Blending thresholds averaged them
    # toward population mean, destroying beneficial threshold configurations
    # that underlie exploration and avoidance behaviors.
    child_params.firing_threshold_excitatory = pick_biased(father_params.firing_threshold_excitatory, mother_params.firing_threshold_excitatory)
    child_params.firing_threshold_inhibitory = pick_biased(father_params.firing_threshold_inhibitory, mother_params.firing_threshold_inhibitory)
    child_params.adaptation_rate = blend_bounded(father_params.adaptation_rate, mother_params.adaptation_rate, 0.0, 0.3)
    child_params.spontaneous_firing_rate = blend_bounded(father_params.spontaneous_firing_rate, mother_params.spontaneous_firing_rate, 0.0, 0.15)
    child_params.neuron_health_decay = blend_bounded(father_params.neuron_health_decay, mother_params.neuron_health_decay, 0.0001, 0.01)
    
    # --- Dendritic Branch Properties ---
    child_params.num_dendritic_branches = blend_int(father_params.num_dendritic_branches, mother_params.num_dendritic_branches)
    child_params.branch_threshold = blend_bounded(father_params.branch_threshold, mother_params.branch_threshold, 0.3, 1.0)
    child_params.plateau_decay = pick_or_blend(father_params.plateau_decay, mother_params.plateau_decay)
    
    # --- Synaptic Properties & Plasticity ---
    # v111 FIX (M23): Time constants use pick_biased to preserve
    # the fitter parent's temporal dynamics. Blending tau values
    # averages intrinsic timescales, collapsing the multi-timescale
    # separation (§3) that evolution may have optimized.
    child_params.tau_fast = pick_biased(father_params.tau_fast, mother_params.tau_fast)
    child_params.tau_slow = pick_biased(father_params.tau_slow, mother_params.tau_slow)
    child_params.tau_meta = pick_biased(father_params.tau_meta, mother_params.tau_meta)
    child_params.tau_ltp = pick_biased(father_params.tau_ltp, mother_params.tau_ltp)
    child_params.tau_ltd = pick_biased(father_params.tau_ltd, mother_params.tau_ltd)
    
    # v3.31: Inherit meta-plasticity, LTP/LTD, and meta-behavior coupling params
    child_params.meta_target_gain = blend_bounded(
        getattr(father_params, 'meta_target_gain', 0.30),
        getattr(mother_params, 'meta_target_gain', 0.30), 0.1, 0.6)
    child_params.meta_accumulation_rate = blend_bounded(
        getattr(father_params, 'meta_accumulation_rate', 0.35),
        getattr(mother_params, 'meta_accumulation_rate', 0.35), 0.1, 0.7)
    child_params.meta_clamp_max = blend_bounded(
        getattr(father_params, 'meta_clamp_max', 1.0),
        getattr(mother_params, 'meta_clamp_max', 1.0), 0.5, 1.0)
    child_params.hebbian_ltp_rate = blend_bounded(
        getattr(father_params, 'hebbian_ltp_rate', 0.12),
        getattr(mother_params, 'hebbian_ltp_rate', 0.12), 0.05, 0.20)
    child_params.ltd_neutral_scale = blend_bounded(
        getattr(father_params, 'ltd_neutral_scale', 0.12),
        getattr(mother_params, 'ltd_neutral_scale', 0.12), 0.03, 0.25)
    child_params.ltd_inhibitory_scale = blend_bounded(
        getattr(father_params, 'ltd_inhibitory_scale', 0.6),
        getattr(mother_params, 'ltd_inhibitory_scale', 0.6), 0.3, 1.0)
    child_params.w_slow_post_trace_fraction = blend_bounded(
        getattr(father_params, 'w_slow_post_trace_fraction', 0.85),
        getattr(mother_params, 'w_slow_post_trace_fraction', 0.85), 0.5, 0.95)
    
    # v3.31: NEW — Inherit meta-behavior coupling params
    child_params.meta_influence_gain = blend_bounded(
        getattr(father_params, 'meta_influence_gain', 0.25),
        getattr(mother_params, 'meta_influence_gain', 0.25), 0.1, 0.5)
    child_params.meta_da_boost = blend_bounded(
        getattr(father_params, 'meta_da_boost', 2.0),
        getattr(mother_params, 'meta_da_boost', 2.0), 1.0, 3.0)
    child_params.w_fast_delta_share = blend_bounded(
        getattr(father_params, 'w_fast_delta_share', 0.5),
        getattr(mother_params, 'w_fast_delta_share', 0.5), 0.3, 0.7)
    child_params.w_slow_delta_share = blend_bounded(
        getattr(father_params, 'w_slow_delta_share', 0.02),
        getattr(mother_params, 'w_slow_delta_share', 0.02), 0.005, 0.05)
    
    # --- Synaptic Weight Initialization Ranges ---
    child_params.w_fast_init_min = pick_or_blend(father_params.w_fast_init_min, mother_params.w_fast_init_min)
    child_params.w_fast_init_max = pick_or_blend(father_params.w_fast_init_max, mother_params.w_fast_init_max)
    child_params.w_slow_init_min = pick_or_blend(father_params.w_slow_init_min, mother_params.w_slow_init_min)
    child_params.w_slow_init_max = pick_or_blend(father_params.w_slow_init_max, mother_params.w_slow_init_max)
    child_params.w_meta_init_min = pick_or_blend(father_params.w_meta_init_min, mother_params.w_meta_init_min)
    child_params.w_meta_init_max = pick_or_blend(father_params.w_meta_init_max, mother_params.w_meta_init_max)
    
    # --- Learning and Plasticity Rules ---
    # v111 FIX (M21): Learning rate inherits from fitter parent.
    child_params.learning_rate = pick_biased(father_params.learning_rate, mother_params.learning_rate)
    child_params.stdp_window = pick_or_blend(father_params.stdp_window, mother_params.stdp_window)
    child_params.learning_rate_mod = pick_biased(father_params.learning_rate_mod, mother_params.learning_rate_mod)
    child_params.plasticity_threshold = blend_bounded(father_params.plasticity_threshold, mother_params.plasticity_threshold, 0.2, 0.9)
    child_params.associativity_strength = blend_bounded(father_params.associativity_strength, mother_params.associativity_strength, 0.01, 0.3)
    
    # --- Structural Plasticity ---
    child_params.synapse_integrity_threshold = blend_bounded(father_params.synapse_integrity_threshold, mother_params.synapse_integrity_threshold, 0.02, 0.3)
    child_params.synapse_formation_prob = blend_bounded(father_params.synapse_formation_prob, mother_params.synapse_formation_prob, 0.005, 0.1)
    child_params.synapse_death_prob = blend_bounded(father_params.synapse_death_prob, mother_params.synapse_death_prob, 0.005, 0.05)
    child_params.neuron_death_threshold = blend_bounded(father_params.neuron_death_threshold, mother_params.neuron_death_threshold, 0.02, 0.3)
    
    # --- Neuromodulation ---
    child_params.dopamine_baseline = blend_bounded(father_params.dopamine_baseline, mother_params.dopamine_baseline, 0.05, 0.4)
    child_params.dopamine_high_affinity_threshold = pick_or_blend(father_params.dopamine_high_affinity_threshold, mother_params.dopamine_high_affinity_threshold)
    child_params.dopamine_low_affinity_threshold = pick_or_blend(father_params.dopamine_low_affinity_threshold, mother_params.dopamine_low_affinity_threshold)
    child_params.serotonin_baseline = blend_bounded(father_params.serotonin_baseline, mother_params.serotonin_baseline, 0.05, 0.4)
    child_params.serotonin_high_affinity_threshold = pick_or_blend(father_params.serotonin_high_affinity_threshold, mother_params.serotonin_high_affinity_threshold)
    child_params.serotonin_low_affinity_threshold = pick_or_blend(father_params.serotonin_low_affinity_threshold, mother_params.serotonin_low_affinity_threshold)
    child_params.acetylcholine_baseline = blend_bounded(father_params.acetylcholine_baseline, mother_params.acetylcholine_baseline, 0.05, 0.4)
    child_params.acetylcholine_high_affinity_threshold = pick_or_blend(father_params.acetylcholine_high_affinity_threshold, mother_params.acetylcholine_high_affinity_threshold)
    child_params.acetylcholine_low_affinity_threshold = pick_or_blend(father_params.acetylcholine_low_affinity_threshold, mother_params.acetylcholine_low_affinity_threshold)
    child_params.norepinephrine_baseline = blend_bounded(father_params.norepinephrine_baseline, mother_params.norepinephrine_baseline, 0.05, 0.4)
    child_params.norepinephrine_high_affinity_threshold = pick_or_blend(father_params.norepinephrine_high_affinity_threshold, mother_params.norepinephrine_high_affinity_threshold)
    child_params.norepinephrine_low_affinity_threshold = pick_or_blend(father_params.norepinephrine_low_affinity_threshold, mother_params.norepinephrine_low_affinity_threshold)
    # v110 FIX (M18): Tighter blending bounds [0.08, 0.40] prevent offspring
    # from inheriting fatally low decay rates that cause chronic saturation.
    child_params.neuromod_decay_rate = blend_bounded(father_params.neuromod_decay_rate, mother_params.neuromod_decay_rate, 0.08, 0.40)
    # v3.33: Inherit reuptake transporter kinetics
    child_params.reuptake_vmax_ne = blend_bounded(father_params.reuptake_vmax_ne, mother_params.reuptake_vmax_ne, 0.03, 0.15)
    child_params.reuptake_vmax_da = blend_bounded(father_params.reuptake_vmax_da, mother_params.reuptake_vmax_da, 0.10, 0.30)
    child_params.reuptake_vmax_5ht = blend_bounded(father_params.reuptake_vmax_5ht, mother_params.reuptake_vmax_5ht, 0.01, 0.07)
    child_params.reuptake_vmax_ach = blend_bounded(father_params.reuptake_vmax_ach, mother_params.reuptake_vmax_ach, 0.04, 0.20)
    child_params.reuptake_km = blend_bounded(father_params.reuptake_km, mother_params.reuptake_km, 0.2, 1.0)
    child_params.autoreceptor_strength = blend_bounded(father_params.autoreceptor_strength, mother_params.autoreceptor_strength, 0.5, 1.5)
    child_params.diffusion_rate = blend_bounded(father_params.diffusion_rate, mother_params.diffusion_rate, 0.005, 0.1)
    
    # --- Oscillators & Synchronization ---
    child_params.oscillator_low_freq = pick_or_blend(father_params.oscillator_low_freq, mother_params.oscillator_low_freq)
    child_params.oscillator_mid_freq = pick_or_blend(father_params.oscillator_mid_freq, mother_params.oscillator_mid_freq)
    child_params.oscillator_high_freq = pick_or_blend(father_params.oscillator_high_freq, mother_params.oscillator_high_freq)
    child_params.oscillator_strength = blend_bounded(father_params.oscillator_strength, mother_params.oscillator_strength, 0.05, 0.4)
    
    # --- Energy Metabolism ---
    child_params.energy_baseline = pick_or_blend(father_params.energy_baseline, mother_params.energy_baseline)
    child_params.firing_energy_cost = pick_or_blend(father_params.firing_energy_cost, mother_params.firing_energy_cost)
    child_params.plasticity_energy_cost = pick_or_blend(father_params.plasticity_energy_cost, mother_params.plasticity_energy_cost)
    child_params.metabolic_rate = blend_bounded(father_params.metabolic_rate, mother_params.metabolic_rate, 0.4, 1.8)
    child_params.recovery_rate = pick_or_blend(father_params.recovery_rate, mother_params.recovery_rate)
    
    # --- Proprioceptron Parameters (UPDATED v3.1) ---
    child_params.proprioceptron_rock_memory = blend_int(father_params.proprioceptron_rock_memory, mother_params.proprioceptron_rock_memory)
    child_params.proprioceptron_force_turn_threshold = blend_int(father_params.proprioceptron_force_turn_threshold, mother_params.proprioceptron_force_turn_threshold)
    child_params.proprioceptron_clear_path_threshold = blend_int(
        getattr(father_params, 'proprioceptron_clear_path_threshold', 5),
        getattr(mother_params, 'proprioceptron_clear_path_threshold', 5)
    )
    
    # --- NEW v3.1: Brain-Instinct Balance ---
    child_params.brain_movement_base_weight = blend_bounded(
        getattr(father_params, 'brain_movement_base_weight', 0.7),
        getattr(mother_params, 'brain_movement_base_weight', 0.7), 0.3, 0.95)
    child_params.brain_rest_override_threshold = blend_bounded(
        getattr(father_params, 'brain_rest_override_threshold', 0.3),
        getattr(mother_params, 'brain_rest_override_threshold', 0.3), 0.15, 0.5)
    child_params.circadian_rest_tendency = blend_bounded(
        getattr(father_params, 'circadian_rest_tendency', 0.7),
        getattr(mother_params, 'circadian_rest_tendency', 0.7), 0.4, 0.9)
    child_params.temp_cold_threshold = blend_bounded(
        getattr(father_params, 'temp_cold_threshold', 35.5),
        getattr(mother_params, 'temp_cold_threshold', 35.5), 34.0, 36.5)
    child_params.temp_hot_threshold = blend_bounded(
        getattr(father_params, 'temp_hot_threshold', 38.5),
        getattr(mother_params, 'temp_hot_threshold', 38.5), 37.5, 40.0)
    child_params.temp_movement_bonus = blend_bounded(
        getattr(father_params, 'temp_movement_bonus', 0.15),
        getattr(mother_params, 'temp_movement_bonus', 0.15), 0.05, 0.3)
    
    # --- Homeostasis ---
    child_params.target_firing_rate = blend_bounded(father_params.target_firing_rate, mother_params.target_firing_rate, 0.02, 0.3)
    child_params.homeostatic_plasticity_rate = blend_bounded(father_params.homeostatic_plasticity_rate, mother_params.homeostatic_plasticity_rate, 0.0001, 0.005)
    child_params.min_excitatory_fraction = blend_bounded(getattr(father_params, 'min_excitatory_fraction', 0.17), getattr(mother_params, 'min_excitatory_fraction', 0.17), 0.10, 0.25)
    child_params.max_excitatory_fraction = blend_bounded(getattr(father_params, 'max_excitatory_fraction', 0.27), getattr(mother_params, 'max_excitatory_fraction', 0.27), 0.22, 0.35)
    child_params.target_excitatory_fraction = blend_bounded(getattr(father_params, 'target_excitatory_fraction', 0.22), getattr(mother_params, 'target_excitatory_fraction', 0.22), 0.15, 0.28)
    child_params.min_inhibitory_fraction = blend_bounded(getattr(father_params, 'min_inhibitory_fraction', 0.16), getattr(mother_params, 'min_inhibitory_fraction', 0.16), 0.10, 0.28)
    child_params.target_inhibitory_fraction = blend_bounded(getattr(father_params, 'target_inhibitory_fraction', 0.24), getattr(mother_params, 'target_inhibitory_fraction', 0.24), 0.16, 0.40)
    child_params.adaptive_threshold_adjustment = blend_bounded(getattr(father_params, 'adaptive_threshold_adjustment', 0.04), getattr(mother_params, 'adaptive_threshold_adjustment', 0.04), 0.005, 0.08)
    
    # --- ChronoPlasticity ---
    child_params.chrono_alpha_f = pick_or_blend(father_params.chrono_alpha_f, mother_params.chrono_alpha_f)
    child_params.chrono_alpha_s = pick_or_blend(father_params.chrono_alpha_s, mother_params.chrono_alpha_s)
    child_params.chrono_lambda_f = pick_or_blend(father_params.chrono_lambda_f, mother_params.chrono_lambda_f)
    child_params.chrono_lambda_s = pick_or_blend(father_params.chrono_lambda_s, mother_params.chrono_lambda_s)
    child_params.chrono_omega_min = blend_bounded(father_params.chrono_omega_min, mother_params.chrono_omega_min, 0.01, 0.25)
    child_params.chrono_omega_max = blend_bounded(father_params.chrono_omega_max, mother_params.chrono_omega_max, 0.50, 0.99)
    child_params.chrono_omega_smoothing = blend_bounded(father_params.chrono_omega_smoothing, mother_params.chrono_omega_smoothing, 0.05, 0.5)
    child_params.chrono_carry_gain = blend_bounded(getattr(father_params, 'chrono_carry_gain', 0.24), getattr(mother_params, 'chrono_carry_gain', 0.24), 0.08, 0.45)
    child_params.chrono_plasticity_gain = blend_bounded(getattr(father_params, 'chrono_plasticity_gain', 0.28), getattr(mother_params, 'chrono_plasticity_gain', 0.28), 0.08, 0.50)
    
    # --- Aigarth Hybridization ---
    child_params.itu_circle_radius = blend_int(father_params.itu_circle_radius, mother_params.itu_circle_radius)
    child_params.evolution_interval = blend_int(father_params.evolution_interval, mother_params.evolution_interval)
    child_params.fitness_temporal_weight = blend_bounded(father_params.fitness_temporal_weight, mother_params.fitness_temporal_weight, 0.1, 0.6)
    child_params.fitness_energy_weight = blend_bounded(father_params.fitness_energy_weight, mother_params.fitness_energy_weight, 0.1, 0.5)
    child_params.fitness_pattern_weight = blend_bounded(father_params.fitness_pattern_weight, mother_params.fitness_pattern_weight, 0.1, 0.5)
    
    # --- Miscellaneous ---
    child_params.phase_coupling_strength = blend_bounded(father_params.phase_coupling_strength, mother_params.phase_coupling_strength, 0.05, 0.3)
    child_params.max_axonal_delay = pick_or_blend(father_params.max_axonal_delay, mother_params.max_axonal_delay)
    
    # ==========================================================================
    # LEVEL 2: CREATE BASE CHILD NETWORK
    # ==========================================================================
    
    child_net = NeuraxonNetwork(child_params)
    
    # Get counts for proportional mapping
    child_total_neurons = len(child_net.all_neurons)
    father_total_neurons = len(father_net.all_neurons)
    mother_total_neurons = len(mother_net.all_neurons)
    
    # ==========================================================================
    # LEVEL 3: INHERIT NEURON-LEVEL PARAMETERS (with proportional mapping)
    # ==========================================================================
    
    def inherit_neuron_properties(child_neuron, f_neuron, m_neuron):
        """
        Inherit individual neuron properties from parent neurons.
        Handles None parents gracefully.
        """
        # If both parents are None, keep defaults
        if f_neuron is None and m_neuron is None:
            return
        
        # If only one parent available, use that one with variation
        if f_neuron is None:
            f_neuron = m_neuron
        if m_neuron is None:
            m_neuron = f_neuron
        
        # --- Individualized Membrane Properties ---
        child_neuron.membrane_time_constant = pick_or_blend(f_neuron.membrane_time_constant, m_neuron.membrane_time_constant)
        child_neuron.firing_threshold_excitatory = pick_biased(f_neuron.firing_threshold_excitatory, m_neuron.firing_threshold_excitatory)
        child_neuron.firing_threshold_inhibitory = pick_biased(f_neuron.firing_threshold_inhibitory, m_neuron.firing_threshold_inhibitory)
        child_neuron.adaptation_rate = pick_or_blend(f_neuron.adaptation_rate, m_neuron.adaptation_rate)
        child_neuron.spontaneous_firing_rate = pick_or_blend(f_neuron.spontaneous_firing_rate, m_neuron.spontaneous_firing_rate)
        child_neuron.neuron_health_decay = pick_or_blend(f_neuron.neuron_health_decay, m_neuron.neuron_health_decay)
        
        # --- Individualized Energy Metabolism ---
        child_neuron.energy_baseline = pick_or_blend(f_neuron.energy_baseline, m_neuron.energy_baseline)
        child_neuron.firing_energy_cost = pick_or_blend(f_neuron.firing_energy_cost, m_neuron.firing_energy_cost)
        child_neuron.plasticity_energy_cost = pick_or_blend(f_neuron.plasticity_energy_cost, m_neuron.plasticity_energy_cost)
        child_neuron.metabolic_rate = pick_or_blend(f_neuron.metabolic_rate, m_neuron.metabolic_rate)
        child_neuron.recovery_rate = pick_or_blend(f_neuron.recovery_rate, m_neuron.recovery_rate)
        
        # --- Phase and Frequency Properties ---
        child_neuron.natural_frequency = pick_or_blend(f_neuron.natural_frequency, m_neuron.natural_frequency)
        child_neuron.intrinsic_timescale = pick_or_blend(f_neuron.intrinsic_timescale, m_neuron.intrinsic_timescale)
        child_neuron.fitness_score = blend(f_neuron.fitness_score, m_neuron.fitness_score, variation=0.2)
        
        # --- Inherit Dendritic Branch Properties ---
        inherit_dendritic_branches(child_neuron, f_neuron, m_neuron)
    
    def inherit_dendritic_branches(child_neuron, f_neuron, m_neuron):
        """
        Inherit dendritic branch properties using proportional mapping.
        Handles different numbers of branches between parents.
        """
        f_branches = f_neuron.dendritic_branches if f_neuron else []
        m_branches = m_neuron.dendritic_branches if m_neuron else []
        
        if not f_branches and not m_branches:
            return
        
        child_branch_count = len(child_neuron.dendritic_branches)
        f_branch_count = len(f_branches)
        m_branch_count = len(m_branches)
        
        for i, branch in enumerate(child_neuron.dendritic_branches):
            # Use proportional mapping for branches too
            f_b = None
            m_b = None
            
            if f_branches:
                f_idx = proportional_index(i, child_branch_count, f_branch_count)
                f_b = f_branches[f_idx]
            
            if m_branches:
                m_idx = proportional_index(i, child_branch_count, m_branch_count)
                m_b = m_branches[m_idx]
            
            # Handle None cases
            if f_b is None and m_b is None:
                continue
            if f_b is None:
                f_b = m_b
            if m_b is None:
                m_b = f_b
            
            branch.branch_threshold = pick_or_blend(f_b.branch_threshold, m_b.branch_threshold)
            branch.plateau_decay = pick_or_blend(f_b.plateau_decay, m_b.plateau_decay)
    
    # Apply neuron inheritance using proportional mapping
    # Input neurons
    for i, neuron in enumerate(child_net.input_neurons):
        f_n, m_n = get_parent_neuron_pair(
            i, len(child_net.input_neurons),
            father_net.input_neurons, mother_net.input_neurons
        )
        inherit_neuron_properties(neuron, f_n, m_n)
    
    # Hidden neurons (most likely to have different counts)
    for i, neuron in enumerate(child_net.hidden_neurons):
        f_n, m_n = get_parent_neuron_pair(
            i, len(child_net.hidden_neurons),
            father_net.hidden_neurons, mother_net.hidden_neurons
        )
        inherit_neuron_properties(neuron, f_n, m_n)
    
    # Output neurons
    for i, neuron in enumerate(child_net.output_neurons):
        f_n, m_n = get_parent_neuron_pair(
            i, len(child_net.output_neurons),
            father_net.output_neurons, mother_net.output_neurons
        )
        inherit_neuron_properties(neuron, f_n, m_n)
    
    # ==========================================================================
    # LEVEL 4: INHERIT SYNAPSE-LEVEL PARAMETERS (with relative position matching)
    # ==========================================================================
    
    f_synapses = list(father_net.synapses) if father_net.synapses else []
    m_synapses = list(mother_net.synapses) if mother_net.synapses else []
    
    def inherit_synapse_properties(child_synapse):
        """
        Inherit synapse properties using relative position matching.
        This handles different-sized networks by finding synapses with
        similar relative pre/post positions.
        """
        # Try to find similar synapses in each parent
        f_s = find_similar_synapse(
            child_synapse.pre_id, child_synapse.post_id,
            child_total_neurons, f_synapses, father_total_neurons
        )
        m_s = find_similar_synapse(
            child_synapse.pre_id, child_synapse.post_id,
            child_total_neurons, m_synapses, mother_total_neurons
        )
        
        # Handle None cases
        if f_s is None and m_s is None:
            return  # Keep default initialization
        if f_s is None:
            f_s = m_s
        if m_s is None:
            m_s = f_s
        
        # --- Inherit Synaptic Weights (BIASED toward fitter parent v2.34) ---
        child_synapse.w_fast = pick_biased(f_s.w_fast, m_s.w_fast, 0.7)
        child_synapse.w_slow = pick_biased(f_s.w_slow, m_s.w_slow, 0.7)
        child_synapse.w_meta = pick_biased(f_s.w_meta, m_s.w_meta, 0.7)
        
        # --- Inherit Individualized Time Constants ---
        child_synapse.tau_fast = pick_or_blend(f_s.tau_fast, m_s.tau_fast)
        child_synapse.tau_slow = pick_or_blend(f_s.tau_slow, m_s.tau_slow)
        child_synapse.tau_meta = pick_or_blend(f_s.tau_meta, m_s.tau_meta)
        child_synapse.tau_ltp = pick_or_blend(f_s.tau_ltp, m_s.tau_ltp)
        child_synapse.tau_ltd = pick_or_blend(f_s.tau_ltd, m_s.tau_ltd)
        
        # --- Inherit Learning Parameters ---
        child_synapse.learning_rate = pick_or_blend(f_s.learning_rate, m_s.learning_rate)
        child_synapse.plasticity_threshold = pick_or_blend(f_s.plasticity_threshold, m_s.plasticity_threshold)
        
        # --- Inherit Synapse Type Properties ---
        child_synapse.is_silent = pick(f_s.is_silent, m_s.is_silent)
        child_synapse.is_modulatory = pick(f_s.is_modulatory, m_s.is_modulatory)
        child_synapse.axonal_delay = pick_or_blend(f_s.axonal_delay, m_s.axonal_delay)
        child_synapse.integrity = blend_bounded(f_s.integrity, m_s.integrity, 0.0, 1.0, variation=0.02)
        
        # Update synapse type based on inherited properties
        child_synapse.synapse_type = child_synapse._determine_type()
    
    # Apply synapse inheritance
    for synapse in child_net.synapses:
        inherit_synapse_properties(synapse)
    
    # ==========================================================================
    # LEVEL 5: INHERIT NEUROMODULATOR STATES
    # ==========================================================================
    
    for mod in ['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']:
        f_level = father_net.neuromodulators.get(mod, 0.12)
        m_level = mother_net.neuromodulators.get(mod, 0.12)
        baseline = getattr(child_params, f'{mod}_baseline')
        inherited = blend(f_level, m_level, variation=0.1)
        child_net.neuromodulators[mod] = inherited * 0.7 + baseline * 0.3
    
    # Inherit modulator grid (with handling for different sizes)
    try:
        f_grid = np.array(father_net.modulator_grid)
        m_grid = np.array(mother_net.modulator_grid)
        child_shape = child_net.modulator_grid.shape
        
        if f_grid.shape == m_grid.shape == child_shape:
            # Same size - direct blend
            avg = (f_grid + m_grid) / 2.0
            noise = np.random.uniform(0.9, 1.1, avg.shape)
            child_net.modulator_grid = avg * noise
        else:
            # Different sizes - interpolate to child size
            from scipy.ndimage import zoom as scipy_zoom
            
            def resize_grid(grid, target_shape):
                if grid.shape == target_shape:
                    return grid
                zoom_factors = [t / s for t, s in zip(target_shape, grid.shape)]
                return scipy_zoom(grid, zoom_factors, order=1)
            
            try:
                f_resized = resize_grid(f_grid, child_shape)
                m_resized = resize_grid(m_grid, child_shape)
                avg = (f_resized + m_resized) / 2.0
                noise = np.random.uniform(0.9, 1.1, avg.shape)
                child_net.modulator_grid = avg * noise
            except ImportError:
                # scipy not available - use simple average and add noise
                child_net.modulator_grid = child_net.modulator_grid * random.uniform(0.9, 1.1)
    except Exception:
        pass  # Keep default grid on any error
    
    # ==========================================================================
    # LEVEL 6: INHERIT OSCILLATOR CONFIGURATIONS
    # ==========================================================================
    
    f_phases = father_net.oscillator_phase_offsets
    m_phases = mother_net.oscillator_phase_offsets
    
    child_phases = []
    for i in range(3):
        f_p = f_phases[i] if i < len(f_phases) else random.random() * 2 * math.pi
        m_p = m_phases[i] if i < len(m_phases) else random.random() * 2 * math.pi
        
        if random.random() < 0.5:
            child_phases.append(pick(f_p, m_p))
        else:
            # Circular average for phases
            avg_phase = math.atan2(
                (math.sin(f_p) + math.sin(m_p)) / 2,
                (math.cos(f_p) + math.cos(m_p)) / 2
            )
            child_phases.append((avg_phase + random.uniform(-0.2, 0.2)) % (2 * math.pi))
    
    child_net.oscillator_phase_offsets = tuple(child_phases)
    
    # ==========================================================================
    # LEVEL 7: INHERIT ITU CIRCLE PARAMETERS (with different counts)
    # ==========================================================================
    
    f_circles = father_net.itu_circles if father_net.itu_circles else []
    m_circles = mother_net.itu_circles if mother_net.itu_circles else []
    
    if child_net.itu_circles and (f_circles or m_circles):
        child_circle_count = len(child_net.itu_circles)
        f_circle_count = len(f_circles)
        m_circle_count = len(m_circles)
        
        for i, circle in enumerate(child_net.itu_circles):
            # Use proportional mapping for ITU circles
            f_c = None
            m_c = None
            
            if f_circles:
                f_idx = proportional_index(i, child_circle_count, f_circle_count)
                f_c = f_circles[f_idx]
            
            if m_circles:
                m_idx = proportional_index(i, child_circle_count, m_circle_count)
                m_c = m_circles[m_idx]
            
            # Handle None cases
            if f_c is None and m_c is None:
                continue
            if f_c is None:
                f_c = m_c
            if m_c is None:
                m_c = f_c
            
            # Inherit mutation rate
            circle.mutation_rate = blend_bounded(f_c.mutation_rate, m_c.mutation_rate, 0.001, 0.1)
            
            # Initialize fitness history with inherited baseline
            f_fitness = f_c.fitness_history[-1] if f_c.fitness_history else 0.5
            m_fitness = m_c.fitness_history[-1] if m_c.fitness_history else 0.5
            avg_fitness = (f_fitness + m_fitness) / 2
            circle.fitness_history = [avg_fitness * random.uniform(0.9, 1.1)]
    
    # ==========================================================================
    # LEVEL 8: INHERIT ADDITIONAL NETWORK STATE
    # ==========================================================================
    
    # Inherit branching ratio tendency
    child_net.branching_ratio = blend(
        father_net.branching_ratio,
        mother_net.branching_ratio,
        variation=0.15
    )
    
    # Reset time-based counters for new child
    child_net.time = 0.0
    child_net.step_count = 0
    child_net.total_energy_consumed = 0.0
    
    # ==========================================================================
    # v3.2: INHERIT METABOLIC PREFERENCES (for NxEr to copy after birth)
    # ==========================================================================
    child_net._inherited_metadata = {
        'temperature_tolerance_cold': blend_bounded(
            getattr(father, 'temperature_tolerance_cold', 35.5),
            getattr(mother, 'temperature_tolerance_cold', 35.5),
            34.0, 36.5, 0.05),
        'temperature_tolerance_hot': blend_bounded(
            getattr(father, 'temperature_tolerance_hot', 38.5),
            getattr(mother, 'temperature_tolerance_hot', 38.5),
            37.5, 40.0, 0.05),
        'resting_metabolism_multiplier': blend_bounded(
            getattr(father, 'resting_metabolism_multiplier', 0.3),
            getattr(mother, 'resting_metabolism_multiplier', 0.3),
            0.15, 0.5, 0.08),
    }
    
    return child_net


def inherit_v2_neuron_params(child_neuron, parent1_neuron, parent2_neuron, mutation_rate=0.1):
    """Neuraxon v2.0: Inherit DSN/CTSN learnable parameters."""
    for attr in ['ctsn_phi_gain', 'ctsn_phi_bias']:
        p1_val = getattr(parent1_neuron, attr, 0.0)
        p2_val = getattr(parent2_neuron, attr, 0.0)
        # Crossover: weighted average with slight random bias
        child_val = p1_val * 0.5 + p2_val * 0.5
        # Mutation
        if random.random() < mutation_rate:
            child_val += random.gauss(0, 0.1)
        setattr(child_neuron, attr, max(-5.0, min(5.0, child_val)))
