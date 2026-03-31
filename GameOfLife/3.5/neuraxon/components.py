# Neuraxon Game of Life 3.5 Neuron Components (Neuraxon 2.0 Compliant) Internal version 125
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife

import random
import math
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

# Import local modules
from .enums import SynapseType
from config import NetworkParameters, SYNAPSE_SILENCING_ACTIVITY_THRESHOLD
from utils import _variate
from logger import get_data_logger

class DendriticBranch:
    """
    Models a single dendritic compartment of a Neuraxon.
    Now includes individualized threshold and decay properties.
    """
    def __init__(self, branch_id: int, parent_neuron_id: int, params: NetworkParameters):
        self.branch_id = branch_id
        self.parent_neuron_id = parent_neuron_id
        self.params = params
        self.branch_potential = 0.0
        
        # INDIVIDUALIZED: Each branch has slight variations in geometry/physics
        self.branch_threshold = _variate(params.branch_threshold)
        self.plateau_decay = _variate(params.plateau_decay)
        
        self.plateau_potential = 0.0
        self.local_spike_history = deque(maxlen=10)
    
    def integrate_inputs(self, synaptic_inputs: List[float], dt: float) -> float:
        """
        Integrates incoming synaptic signals for this branch.
        """
        if not synaptic_inputs:
            # Use localized plateau_decay
            self.plateau_potential += dt / self.plateau_decay * (-self.plateau_potential)
            self.branch_potential += dt / (self.params.membrane_time_constant * 0.5) * (-self.branch_potential)
            return self.branch_potential + self.plateau_potential
        
        branch_signal = math.tanh(sum(synaptic_inputs))
        
        # Use localized branch_threshold
        if abs(branch_signal) > self.branch_threshold:
            self.plateau_potential = branch_signal * 0.8
            
        tau_branch = max(1.0, self.params.membrane_time_constant * 0.3)
        self.branch_potential += dt / tau_branch * (branch_signal - self.branch_potential)
        
        self.local_spike_history.append(1.0 if abs(self.branch_potential) > self.branch_threshold else 0.0)
        
        return self.branch_potential + self.plateau_potential
    
    def get_local_ca_influx(self) -> float:
        return sum(self.local_spike_history) / len(self.local_spike_history) if len(self.local_spike_history) >= 3 else 0.0
    
    def to_dict(self) -> dict:
        # Save individualized parameters
        return {
            'branch_id': self.branch_id, 
            'branch_potential': self.branch_potential, 
            'plateau_potential': self.plateau_potential, 
            'branch_threshold': self.branch_threshold,
            'plateau_decay': self.plateau_decay,
            'local_spike_history': list(self.local_spike_history) # Save spike history v 2.03
        }

class Synapse:
    """
    Implements the Neuraxon synapse with individualized temporal dynamics and weights.
    Time constants and learning rates are now specific to this synapse instance.
    """
    def __init__(self, pre_id: int, post_id: int, params: NetworkParameters):
        self.pre_id = pre_id
        self.post_id = post_id
        self.is_afferent = False
        self.params = params
        
        self.w_fast = random.uniform(params.w_fast_init_min, params.w_fast_init_max)
        self.w_slow = random.uniform(params.w_slow_init_min, params.w_slow_init_max)
        self.w_meta = random.uniform(params.w_meta_init_min, params.w_meta_init_max)
        
        # INDIVIDUALIZED: Synaptic Time Constants & Plasticity Properties
        self.tau_fast = _variate(params.tau_fast)
        self.tau_slow = _variate(params.tau_slow)
        self.tau_meta = _variate(params.tau_meta)
        self.tau_ltp = _variate(params.tau_ltp)
        self.tau_ltd = _variate(params.tau_ltd)
        self.learning_rate = _variate(params.learning_rate)
        self.plasticity_threshold = _variate(params.plasticity_threshold)
        
        self.is_silent = random.random() < 0.1
        self.is_modulatory = random.random() < 0.2
        self.integrity = 1.0
        self.axonal_delay = random.uniform(0, params.max_axonal_delay)
        
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.pre_trace_ltd = 0.0
        
        self.learning_rate_mod = 1.0
        self.associative_strength = 0.0
        self.neighbor_synapses = []

        # --- Neuraxon v2.0: ChronoPlasticity (Eqs 5-7) ---
        self.chrono_fast_trace = 0.0
        self.chrono_slow_trace = 0.0
        self.chrono_omega = 0.5
        # --- Neuraxon v2.0: AGMP eligibility (Eq 8) ---
        self.eligibility = 0.0
        # --- Neuraxon v2.0: Dendritic branch assignment ---
        self.branch_id = random.randint(0, max(0, getattr(params, 'num_dendritic_branches', 3) - 1))
        self.branch_index = 0
        self.potential_delta_w = 0.0
        self.synapse_type = self._determine_type()
        
        # NEW: Track previous weights for evolution logging
        self._prev_w_fast = self.w_fast
        self._prev_w_slow = self.w_slow
        self._prev_w_meta = self.w_meta
    
    def mark_as_afferent(self):
        """Mark as afferent synapse and strengthen."""
        self.is_afferent = True
        self.w_fast *= self.params.afferent_synapse_strength
        self.w_slow *= self.params.afferent_synapse_strength
        self.is_silent = random.random() > self.params.afferent_synapse_reliability
    
    def _determine_type(self) -> SynapseType:
        if self.is_silent: return SynapseType.SILENT
        if self.is_modulatory: return SynapseType.METABOTROPIC
        return SynapseType.IONOTROPIC_FAST if abs(self.w_fast) >= abs(self.w_slow) else SynapseType.IONOTROPIC_SLOW

    def update_chrono_traces(self, pre_state: int):
        """Neuraxon v2.0: update adaptive fast/slow traces and a smoothed warp gate."""
        if not getattr(self.params, 'chrono_enabled', False):
            return
        a_f = getattr(self.params, 'chrono_alpha_f', 0.95)
        a_s = getattr(self.params, 'chrono_alpha_s', 0.99)
        gate_norm = max(1e-6, float(getattr(self.params, 'chrono_gate_norm', 10.0)))
        raw_clip = float(getattr(self.params, 'chrono_raw_clip', 8.0))
        omega_min = float(getattr(self.params, 'chrono_omega_min', 0.05))
        omega_max = float(getattr(self.params, 'chrono_omega_max', 0.95))
        omega_smoothing = float(getattr(self.params, 'chrono_omega_smoothing', 0.2))

        omega_raw = (
            float(pre_state)
            + getattr(self.params, 'chrono_lambda_f', 0.15) * self.chrono_fast_trace
            + getattr(self.params, 'chrono_lambda_s', 0.08) * self.chrono_slow_trace
        ) / gate_norm
        omega_raw = max(-raw_clip, min(raw_clip, omega_raw))
        omega_target = 1.0 / (1.0 + math.exp(-omega_raw))
        omega_target = max(omega_min, min(omega_max, omega_target))
        self.chrono_omega = (1.0 - omega_smoothing) * self.chrono_omega + omega_smoothing * omega_target
        self.chrono_fast_trace = a_f * self.chrono_fast_trace + float(pre_state)
        warped_alpha = a_s ** self.chrono_omega
        self.chrono_slow_trace = warped_alpha * self.chrono_slow_trace + float(pre_state)
        clip = getattr(self.params, 'chrono_trace_clip', 6.0)
        self.chrono_fast_trace = max(-clip, min(clip, self.chrono_fast_trace))
        self.chrono_slow_trace = max(-clip, min(clip, self.chrono_slow_trace))

    def update_eligibility(self, pre_state: int, post_state: int, params):
        """Neuraxon v2.0: AGMP eligibility trace (Algorithm 1, Eq 8)."""
        if not getattr(params, 'agmp_enabled', False):
            return
        lam_e = getattr(params, 'agmp_lambda_e', 0.95)
        self.eligibility = lam_e * self.eligibility + (1.0 - lam_e) * (1.0 if (pre_state == 1 and post_state == 1) else 0.0)
        self.eligibility = max(-1.0, min(1.0, self.eligibility))
    
    def compute_input(self, pre_state: int, current_time: float) -> Tuple[float, float]:
        if self.is_silent: return 0.0, 0.0
        chrono_carry = 0.0
        if getattr(self.params, 'chrono_plastic_enabled', False):
            carry_gain = float(getattr(self.params, 'chrono_carry_gain', 0.24))
            gate_norm = max(1e-6, float(getattr(self.params, 'chrono_gate_norm', 10.0)))
            carry_raw = (self.chrono_slow_trace - 0.5 * self.chrono_fast_trace) / gate_norm
            chrono_carry = max(-carry_gain, min(carry_gain, carry_raw))
        if pre_state == 0 and abs(chrono_carry) < 1e-6:
            return 0.0, 0.0
        delay_factor = max(0.0, 1.0 - self.axonal_delay / 10.0)
        if self.is_afferent:
            delay_factor = max(0.5, delay_factor)
        try:
            meta_gain = self._cached_meta_gain
        except AttributeError:
            meta_gain = getattr(self.params, 'meta_influence_gain', 0.25)
            self._cached_meta_gain = meta_gain
        w = self.w_fast + self.w_slow + self.w_meta * meta_gain
        effective_state = float(pre_state) + chrono_carry
        signal = w * effective_state
        memory_bias = max(0.0, chrono_carry) * float(getattr(self.params, 'chrono_plasticity_gain', 0.28))
        immediate_factor = max(0.0, min(1.0, delay_factor - 0.5 * memory_bias))
        delayed_factor = max(0.0, min(1.0, (1.0 - delay_factor) + memory_bias))
        return signal * immediate_factor, signal * delayed_factor
    
    def calculate_delta_w(self, pre_state: int, post_state: int, neuromodulators: Dict[str, float], dt: float, receptor_activations: Dict = None, tick: int = 0) -> float:
        receptor_activations = receptor_activations or {}

        # v4.1: Fast path — when both pre and post are neutral, only decay traces
        if pre_state == 0 and post_state == 0:
            self.pre_trace *= (1.0 - dt / self.tau_ltp)
            self.pre_trace_ltd *= (1.0 - dt / self.tau_ltd)
            self.post_trace *= (1.0 - dt / self.tau_ltp)
            self._pending_delta_w = 0.0
            return 0.0
        # Use localized tau_ltp / tau_ltd
        self.pre_trace += (-self.pre_trace / self.tau_ltp + (1 if pre_state == 1 else 0)) * dt
        self.pre_trace_ltd += (-self.pre_trace_ltd / self.tau_ltd + (1 if pre_state == 1 else 0)) * dt
        self.post_trace += (-self.post_trace / self.tau_ltp + (1 if post_state == 1 else 0)) * dt

        d1_act = receptor_activations.get('D1', 0.5)
        d2_act = receptor_activations.get('D2', 0.5)
        ach = neuromodulators.get('acetylcholine', 0.5)
        
        da = neuromodulators.get('dopamine', 0.5)
        da_threshold = self.params.dopamine_low_affinity_threshold
        if da > da_threshold:
            da_high = min(1.0, (da - da_threshold) / da_threshold)
        else:
            da_high = 0.0
        da_low = 1.0 if da > self.params.dopamine_high_affinity_threshold else 0.0
        self._last_da_high = da_high  # v3.31: Store for meta DA-boost in apply_update
        chrono_gain = 1.0
        if getattr(self.params, 'chrono_plastic_enabled', False):
            memory_signal = abs(self.chrono_slow_trace) / (1.0 + abs(self.chrono_fast_trace))
            chrono_gain = 1.0 + float(getattr(self.params, 'chrono_plasticity_gain', 0.28)) * min(1.0, memory_signal + self.chrono_omega * 0.5)

        # v111 FIX (M25): True DA gating — suppress LTP when D1 is weak.
        # Paper Algorithm 3: Δw = η · A⁺ · receptor_act[D1] − η · A⁻ · receptor_act[D2]
        # D1_act already scales the DA component, but with gentle slopes (old v110)
        # and a Hebbian floor, LTP was effectively DA-independent.
        # Now with steep slopes (v111 config) + reduced Hebbian (0.02), D1_act
        # ranges from ~0.05 (DA low) to ~0.95 (DA high), creating true gating.

        # ACh gain: consolidates learning (Paper: ACh modulates plasticity)
        ach_gain = 1.0 + (ach if ach > 0.5 else 0.0)
        self.learning_rate_mod = (1.0 + (d1_act * 0.5) + (d2_act * 0.2)) * ach_gain
        
        # NEW: Log threshold crossings
        logger = get_data_logger()
        if logger.log_level >= 2:
            if da_high > 0 and pre_state == 1 and post_state == 1:
                logger.log_neuromodulator_event(
                    tick=tick,
                    modulator='dopamine',
                    level=da,
                    crossed_threshold='low_affinity',
                    effect='ltp_enabled'
                )
            if da_low > 0 and pre_state == 1 and post_state == -1:
                logger.log_neuromodulator_event(
                    tick=tick,
                    modulator='dopamine', 
                    level=da,
                    crossed_threshold='high_affinity',
                    effect='ltd_enabled'
                )
        
        # Track for weight evolution logging
        self._pending_delta_w = 0.0
        
        if pre_state == 1 and post_state == 1:
            # v112 FIX (M25+M7+M1): LTP = DA-gated + moderate Hebbian floor.
            # Algorithm 3: Δw = η · A⁺ · receptor_act[D1]
            # v111 AUTOPSY: D1²+hebbian=0.02 killed ALL learning (D1²=0.03).
            # v112: Linear D1 (not squared) + moderate Hebbian floor (0.08).
            # With D1 EC50=0.18 in the actual DA range:
            #   DA=0.13 (quiet):  D1=0.38, LTP ∝ 0.38+0.08 = 0.46
            #   DA=0.30 (burst):  D1=0.77, LTP ∝ 0.77+0.08 = 0.85 (×1.8)
            #   DA=0.45 (peak):   D1=0.92, LTP ∝ 0.92+0.08 = 1.00 (×2.2)
            # DA-gated component dominates (5-10:1 ratio) → M25 sees differential
            # Hebbian floor ensures baseline learning → M1, M14 preserved
            hebbian_frac = getattr(self.params, 'hebbian_ltp_rate', 0.18)
            da_component = self.learning_rate * self.learning_rate_mod * d1_act * self.pre_trace
            # v112 FIX: Hebbian NOT gated by D1 — this is the DA-independent
            # residual that keeps the network plastic. Paper §4 distinguishes
            # basic Hebbian LTP (coincident firing) from DA-modulated LTP.
            hebbian_component = self.learning_rate * hebbian_frac * self.pre_trace * self.post_trace
            delta = (da_component + hebbian_component) * chrono_gain
            self._pending_delta_w = delta
            # FIX v2.2505: Log LTP event when delta is significant
            if delta > 0.0001 and logger.log_level >= 2:
                logger.log_plasticity_event(
                    tick=tick, event_type='LTP',
                    pre_id=self.pre_id, post_id=self.post_id,
                    delta_w=delta,
                    details={
                        'dopamine': da,
                        'acetylcholine': ach,
                        'receptor_D1': d1_act,
                        'receptor_D2': d2_act,
                        'pre_trace': self.pre_trace,
                        'post_trace': self.post_trace,
                        'pre_state': pre_state,
                        'post_state': post_state,
                    }
                )
            return delta
        elif pre_state == 1 and post_state <= 0:
            # v112 FIX (M25+M7): LTD linear D2-gated per Algorithm 3.
            # v111 AUTOPSY: D2² at tonic=0.12 gave D2²=0.004 → LTD suppressed
            # 8× more than LTP, breaking M7 ratio (2.17→3.58).
            # v112: Linear D2 (not squared). With D2 EC50=0.12:
            #   D2=0.50 at baseline → moderate LTD always present
            #   D2 unchanged during phasic bursts (tonic-only receptor)
            #   → LTD events spread evenly across DA levels
            #   → Mean DA during LTD ≈ population mean
            #   → While LTP concentrates at burst DA → M25 differential
            ach_forgetting_mult = 1.5 if ach > 0.6 else 1.0
            ltd_neutral = getattr(self.params, 'ltd_neutral_scale', 0.12)
            ltd_inhib = getattr(self.params, 'ltd_inhibitory_scale', 0.6)
            ltd_scale = (ltd_inhib if post_state == -1 else ltd_neutral) * ach_forgetting_mult
            # v112 FIX: Linear D2 gating. D2≈0.50 at baseline → healthy LTD.
            chrono_ltd_gain = max(0.85, min(1.05, chrono_gain))
            delta = -self.learning_rate * self.learning_rate_mod * d2_act * self.pre_trace_ltd * ltd_scale * chrono_ltd_gain
            self._pending_delta_w = delta
            # FIX v2.2505: Log LTD event when delta is significant
            if delta < -0.0001 and logger.log_level >= 2:
                logger.log_plasticity_event(
                    tick=tick, event_type='LTD',
                    pre_id=self.pre_id, post_id=self.post_id,
                    delta_w=delta,
                    details={
                        'dopamine': da,
                        'acetylcholine': ach,
                        'receptor_D1': d1_act,
                        'receptor_D2': d2_act,
                        'pre_trace_ltd': self.pre_trace_ltd,
                        'pre_state': pre_state,
                        'post_state': post_state,
                    }
                )
            return delta
        self._pending_delta_w = 0.0
        return 0.0
    
    def apply_update(self, dt: float, neuromodulators: Dict[str, float], neighbor_deltas: List[float] = None, receptor_activations: Dict = None, tick: int = 0):
        receptor_activations = receptor_activations or {}
        # Store old weights for evolution logging
        old_w_fast = self.w_fast
        old_w_slow = self.w_slow
        old_w_meta = self.w_meta
        
        delta_w = self.potential_delta_w
        own_delta_w = delta_w  # Store for associativity logging
        neighbor_contribution = 0.0
        ach = neuromodulators.get('acetylcholine', 0.5)
        
        if neighbor_deltas:
            # ACh Modulation: Prioritize neighbors/directions (Paper Claim)
            associativity_gain = 1.0 + (ach * 0.5)
            neighbor_contribution = self.params.associativity_strength * associativity_gain * sum(
                dw / (i + 1) for i, dw in enumerate(neighbor_deltas[:3]))
            delta_w += neighbor_contribution
        
        # Use localized tau_fast, tau_slow, tau_meta with saturation prevention
        max_w = self.params.max_weight_magnitude
        min_w = self.params.min_weight_magnitude
        
        # 106: ODE weight form — fast/slow track delta_w with time constants
        self.w_fast += (dt / self.tau_fast) * (-self.w_fast + 0.3 * delta_w)
        self.w_fast = max(min_w, min(max_w, self.w_fast))
        self.w_slow += (dt / self.tau_slow) * (-self.w_slow + 0.1 * delta_w)
        self.w_slow = max(min_w, min(max_w, self.w_slow))
        
        # 106: w_meta — 5-HT receptor gating
        ht2a = receptor_activations.get('5HT2A', 0.5)
        ht1a = receptor_activations.get('5HT1A', 0.5)
        ht_factor = 0.5 * ht2a + 0.1 * (1.0 - ht1a)
        meta_clamp = getattr(self.params, 'meta_clamp_max', 1.0)
        self.w_meta += (dt / self.tau_meta) * (-self.w_meta + 0.05 * delta_w * ht_factor)
        self.w_meta = max(-meta_clamp, min(meta_clamp, self.w_meta))
        
        activity = abs(self.w_fast) + abs(self.w_slow)
        self.integrity = min(1.0, self.integrity + 0.001 * dt * activity) if activity >= 0.01 else self.integrity - self.params.synapse_death_prob * dt

        # NEW: Log weight evolution if significant change
        logger = get_data_logger()
        if logger.log_level >= 2:
            total_change = (abs(self.w_fast - old_w_fast) + 
                           abs(self.w_slow - old_w_slow) + 
                           abs(self.w_meta - old_w_meta))
            if total_change > 0.01:  # Threshold for logging
                logger.log_weight_evolution_event(
                    tick=tick,
                    synapse_pre_id=self.pre_id,
                    synapse_post_id=self.post_id,
                    w_fast_old=old_w_fast, w_fast_new=self.w_fast,
                    w_slow_old=old_w_slow, w_slow_new=self.w_slow,
                    w_meta_old=old_w_meta, w_meta_new=self.w_meta,
                    details={
                        'dopamine': neuromodulators.get('dopamine', 0.0),
                        'serotonin': neuromodulators.get('serotonin', 0.0),
                        'acetylcholine': neuromodulators.get('acetylcholine', 0.0),
                        'norepinephrine': neuromodulators.get('norepinephrine', 0.0),
                        'receptor_D1': receptor_activations.get('D1', 0.0),
                        'receptor_D2': receptor_activations.get('D2', 0.0),
                        'receptor_5HT2A': receptor_activations.get('5HT2A', 0.0),
                        'neighbor_contribution': neighbor_contribution,
                        'potential_delta_w': delta_w,
                    }
                )
            
            # Log associativity if neighbor contribution was significant
            if abs(neighbor_contribution) > 0.001:
                logger.log_associativity_event(
                    tick=tick,
                    synapse_pre_id=self.pre_id,
                    synapse_post_id=self.post_id,
                    own_delta_w=own_delta_w,
                    neighbor_contribution=neighbor_contribution,
                    final_delta_w=delta_w
                )

        # Track previous silent state
        was_silent = self.is_silent

        if self.is_silent and self.pre_trace > 0.5 and random.random() < 0.01:  # silent → active
            self.is_silent = False
            # NEW: Log the event
            logger = get_data_logger()
            if logger.log_level >= 2:
                logger.log_silent_synapse_event(
                    tick=tick,
                    pre_id=self.pre_id,
                    post_id=self.post_id,
                    became_active=True,
                    trigger="pre_trace_threshold"
                )

        # v3.32 FIX: Active → silent transition.
        # BIOINSPIRED: Synapses with prolonged inactivity undergo functional silencing
        # (AMPA receptor internalization). Mirrors LTD-driven silent synapse formation
        # observed in hippocampal circuits (Paper Section 5 — complex signaling).
        if not self.is_silent and not was_silent:
            activity = abs(self.w_fast) + abs(self.w_slow)
            if activity < SYNAPSE_SILENCING_ACTIVITY_THRESHOLD and self.integrity < 0.3 and random.random() < 0.008:
                self.is_silent = True
                logger = get_data_logger()
                if logger.log_level >= 2:
                    logger.log_silent_synapse_event(
                        tick=tick,
                        pre_id=self.pre_id,
                        post_id=self.post_id,
                        became_active=False,
                        trigger="inactivity_silencing"
                    )

        self.synapse_type = self._determine_type()
    
    def get_modulatory_effect(self) -> float:
        return self.w_meta * 0.5 if self.is_modulatory else 0.0
    
    def to_dict(self) -> dict:        
        return {
            'pre_id': self.pre_id, 'post_id': self.post_id, 
            'w_fast': self.w_fast, 'w_slow': self.w_slow, 'w_meta': self.w_meta, 
            'is_silent': self.is_silent, 'is_modulatory': self.is_modulatory, 
            'integrity': self.integrity, 'axonal_delay': self.axonal_delay, 
            'learning_rate_mod': self.learning_rate_mod, 'synapse_type': self.synapse_type.value, 
            'potential_delta_w': self.potential_delta_w,            
            'tau_fast': self.tau_fast, 'tau_slow': self.tau_slow, 'tau_meta': self.tau_meta,
            'tau_ltp': self.tau_ltp, 'tau_ltd': self.tau_ltd,
            'learning_rate': self.learning_rate, 'plasticity_threshold': self.plasticity_threshold,
            'pre_trace': self.pre_trace,# Updated Save states in v 2.03
        'post_trace': self.post_trace,# Updated Save states in v 2.03
        'pre_trace_ltd': self.pre_trace_ltd,# Updated Save # Updated Save states in v 2.1
        'associative_strength': self.associative_strength, # Updated Save states in v 2.1
         # NEW: Save neighbor synapse references as (pre_id, post_id) tuples
        'neighbor_synapse_ids': [(ns.pre_id, ns.post_id) for ns in self.neighbor_synapses], # Updated Save states in v 2.1
        }

class ITUCircle:
    """
    Implements an Intelligent Tissue Unit (ITU) circle, a concept from the Aigarth hybridization (Section 8 of the Neuraxon paper).
    This class manages a group of neurons that are subject to evolutionary pressures. It calculates
    a collective fitness score and applies mutation and selection (pruning) to its member neurons.
    """
    def __init__(self, circle_id: int, neurons: list, params: NetworkParameters, total_circles: int = 1):
        self.circle_id = circle_id
        self.neurons = neurons
        self.params = params
        self.total_circles = max(1, int(total_circles))
        self.fitness_history = []
        self.mutation_rate = 0.01

        freq_min = float(getattr(params, 'natural_freq_range_min', 0.85))
        freq_max = float(getattr(params, 'natural_freq_range_max', 1.15))
        circle_pos = (circle_id + 0.5) / self.total_circles
        self.target_frequency = freq_min + circle_pos * max(1e-6, (freq_max - freq_min))
        self.target_timescale = max(2.0, float(params.membrane_time_constant) * (0.75 + 0.5 * circle_pos))

        # Seed each circle around a different dynamical niche so specialization can emerge
        # through selection instead of every circle performing a symmetric random walk.
        init_spread = max(0.01, (freq_max - freq_min) / (self.total_circles * 2.5))
        for n in neurons:
            n.circle_id = circle_id
            n.natural_frequency = max(freq_min, min(freq_max, random.gauss(self.target_frequency, init_spread)))
            n.intrinsic_timescale = max(1.0, min(getattr(params, 'max_intrinsic_timescale', 80.0),
                                                   random.gauss(self.target_timescale, self.target_timescale * 0.08)))
    
    def compute_fitness(self, network_activity: List[float], temporal_sync: float, energy_efficiency: float) -> float:
        """Calculates the fitness of this ITU based on performance plus circle specialization."""
        base_fitness = (
            self.params.fitness_pattern_weight * (np.mean(network_activity) if network_activity else 0.0)
            + self.params.fitness_temporal_weight * temporal_sync
            + self.params.fitness_energy_weight * energy_efficiency
        )
        if self.neurons:
            mean_freq = float(np.mean([n.natural_frequency for n in self.neurons]))
            mean_tau = float(np.mean([n.intrinsic_timescale for n in self.neurons]))
            freq_span = max(1e-6, float(getattr(self.params, 'natural_freq_range_max', 1.15)) - float(getattr(self.params, 'natural_freq_range_min', 0.85)))
            tau_cap = max(1.0, float(getattr(self.params, 'max_intrinsic_timescale', 80.0)))
            freq_match = max(0.0, 1.0 - abs(mean_freq - self.target_frequency) / freq_span)
            tau_match = max(0.0, 1.0 - abs(mean_tau - self.target_timescale) / tau_cap)
            within_var = float(np.var([n.natural_frequency for n in self.neurons]))
            cohesion = max(0.0, 1.0 - within_var / max(1e-6, (freq_span / 3.0) ** 2))
        else:
            freq_match = tau_match = cohesion = 0.0

        niche_strength = float(getattr(self.params, 'itu_niche_strength', 0.30))
        cohesion_strength = float(getattr(self.params, 'itu_target_cohesion', 0.20))
        fitness = base_fitness + niche_strength * (0.65 * freq_match + 0.35 * tau_match) + cohesion_strength * cohesion
        self.fitness_history.append(fitness)
        return fitness
    
    def mutate(self):
        """Applies small, niche-preserving mutations so circles diverge instead of homogenising."""
        freq_min = float(getattr(self.params, 'natural_freq_range_min', 0.85))
        freq_max = float(getattr(self.params, 'natural_freq_range_max', 1.15))
        tau_cap = max(1.0, float(getattr(self.params, 'max_intrinsic_timescale', 80.0)))
        freq_scale = float(getattr(self.params, 'itu_freq_mutation_scale', 0.06))
        tau_scale = float(getattr(self.params, 'itu_timescale_mutation_scale', 0.08))
        niche_strength = float(getattr(self.params, 'itu_niche_strength', 0.30))
        for neuron in self.neurons:
            if random.random() < self.mutation_rate:
                freq_pull = (self.target_frequency - neuron.natural_frequency) * niche_strength
                tau_pull = (self.target_timescale - neuron.intrinsic_timescale) * niche_strength
                neuron.natural_frequency = max(
                    freq_min,
                    min(freq_max, neuron.natural_frequency + freq_pull + random.uniform(-freq_scale, freq_scale))
                )
                neuron.intrinsic_timescale = max(
                    1.0,
                    min(tau_cap, neuron.intrinsic_timescale + tau_pull + neuron.intrinsic_timescale * random.uniform(-tau_scale, tau_scale))
                )
                if hasattr(neuron.params, 'firing_threshold_excitatory'):
                    neuron.params.firing_threshold_excitatory *= random.uniform(0.97, 1.03)
                
                logger = get_data_logger()
                if logger.log_level >= 2:
                    logger.log_itu_evolution_event(
                        0, self.circle_id, 'mutation',
                        self.fitness_history[-1] if self.fitness_history else 0.0,
                        self.fitness_history[-1] if self.fitness_history else 0.0,
                        1
                    )
    
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
        
        # NEW: Log pruning events
        if pruned_ids:
            logger = get_data_logger()
            if logger.log_level >= 2:
                logger.log_itu_evolution_event(
                    0, self.circle_id, 'pruning',
                    avg_fitness, avg_fitness,
                    len(pruned_ids)
                )
        
        return pruned_ids


# =============================================================================
# MSTH: Multi-Scale Temporal Homeostasis (Neuraxon v2.0 Section 5)
# =============================================================================

class MSTHState:
    """
    Four coordinated regulatory loops (Neuraxon v2.0):
      Ultra-fast (~5ms): emergency suppression / runaway prevention
      Fast (~2s): rapid Ca2+/homeostatic control of excitability
      Medium (~5min): synaptic scaling / gain normalisation
      Slow (~1-24h): structural adjustments and long-horizon stability
    """
    def __init__(self, params):
        self.params = params
        self.ultrafast_activity = 0.0
        self.fast_excitability = 0.0
        self.medium_gain = 1.0
        self.slow_structural = 0.0

    def update(self, current_state_abs: float, dt: float) -> dict:
        p = self.params
        uf_tau = getattr(p, 'msth_ultrafast_tau', 5.0)
        f_tau = getattr(p, 'msth_fast_tau', 2000.0)
        m_tau = getattr(p, 'msth_medium_tau', 300000.0)
        s_tau = getattr(p, 'msth_slow_tau', 3600000.0)
        target = getattr(p, 'target_firing_rate', 0.2)

        alpha_uf = dt / uf_tau
        self.ultrafast_activity = (1.0 - alpha_uf) * self.ultrafast_activity + alpha_uf * current_state_abs
        ultrafast_suppress = self.ultrafast_activity > getattr(p, 'msth_ultrafast_ceiling', 2.0)

        alpha_f = dt / f_tau
        self.fast_excitability = (1.0 - alpha_f) * self.fast_excitability + alpha_f * current_state_abs
        fast_threshold_shift = getattr(p, 'msth_fast_gain', 0.1) * (self.fast_excitability - target)

        alpha_m = dt / m_tau
        target_dev = current_state_abs - target
        self.medium_gain += alpha_m * (-getattr(p, 'msth_medium_gain', 0.001) * target_dev * self.medium_gain)
        self.medium_gain = max(0.5, min(2.0, self.medium_gain))

        alpha_s = dt / s_tau
        self.slow_structural = (1.0 - alpha_s) * self.slow_structural + alpha_s * abs(target_dev)

        return {
            'ultrafast_suppress': ultrafast_suppress,
            'fast_threshold_shift': fast_threshold_shift,
            'medium_gain': self.medium_gain,
            'slow_structural_pressure': self.slow_structural,
        }

    def to_dict(self) -> dict:
        return {
            'ultrafast_activity': self.ultrafast_activity,
            'fast_excitability': self.fast_excitability,
            'medium_gain': self.medium_gain,
            'slow_structural': self.slow_structural,
        }


# =============================================================================
# RECEPTOR SUBTYPE (Neuraxon v2.0 Algorithm 2)
# =============================================================================

class ReceptorSubtype:
    """Nonlinear receptor activation with Hill-like sigmoid."""
    def __init__(self, name: str, parent_modulator: str,
                 threshold: float, gain: float, is_tonic: bool, slope: float = 0.0):
        self.name = name
        self.parent_modulator = parent_modulator
        self.threshold = threshold
        self.gain = gain
        self.is_tonic = is_tonic
        self.slope = float(slope)
        self.activation = 0.0

    def compute_activation(self, concentration: float) -> float:
        k = self.slope if self.slope > 0.0 else (20.0 if self.is_tonic else 10.0)
        exponent = -k * (concentration - self.threshold)
        exponent = max(-50.0, min(50.0, exponent))
        self.activation = self.gain / (1.0 + math.exp(exponent))
        return self.activation

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'parent_modulator': self.parent_modulator,
            'threshold': self.threshold, 'gain': self.gain, 'slope': self.slope,
            'is_tonic': self.is_tonic, 'activation': self.activation,
        }


# =============================================================================
# OSCILLATOR BANK (Neuraxon v2.0 Algorithm 2 - Multi-band PAC)
# =============================================================================

class OscillatorBank:
    """Multi-band oscillator (infraslow-gamma) with Phase-Amplitude Coupling."""
    DEFAULT_BANDS = {
        'infraslow': 0.05, 'slow': 0.5, 'theta': 6.0,
        'alpha': 10.0, 'gamma': 40.0,
    }

    def __init__(self, coupling: float = 0.15, bands=None):
        self.coupling = coupling
        self.bands = {}
        for name, freq in (bands or self.DEFAULT_BANDS).items():
            self.bands[name] = {
                'freq': freq,
                'phase': random.uniform(0.0, 2.0 * math.pi),
                'amplitude': 1.0,
            }

    def update(self, dt: float):
        for b in self.bands.values():
            b['phase'] = (b['phase'] + 2.0 * math.pi * b['freq'] * dt / 1000.0) % (2.0 * math.pi)

    def get_drive(self, neuron_id: int, total_neurons: int) -> float:
        phi = 2.0 * math.pi * neuron_id / max(total_neurons, 1)
        theta_phase = self.bands['theta']['phase']
        gamma_phase = self.bands['gamma']['phase']
        slow_phase = self.bands['slow']['phase']
        infra_phase = self.bands['infraslow']['phase']
        gate_theta = max(0.0, math.cos(theta_phase + phi))
        gamma_sig = self.bands['gamma']['amplitude'] * gate_theta * math.sin(gamma_phase + 2.0 * phi)
        slow_sig = self.bands['slow']['amplitude'] * math.sin(slow_phase + 0.3 * phi)
        infra_sig = self.bands['infraslow']['amplitude'] * math.sin(infra_phase)
        return self.coupling * (gamma_sig + 0.5 * slow_sig + 0.3 * infra_sig)

    def to_dict(self) -> dict:
        return {'coupling': self.coupling, 'bands': self.bands}


# =============================================================================
# NEUROMODULATOR SYSTEM (Neuraxon v2.0 Algorithm 5)
# =============================================================================

class NeuromodulatorSystem:
    """4 neuromodulators with tonic/phasic, 9 receptor subtypes, crosstalk."""
    def __init__(self, params):
        self.params = params
        self.levels = {
            'DA':  {'tonic': getattr(params, 'dopamine_baseline', 0.15),       'phasic': 0.0},
            '5HT': {'tonic': getattr(params, 'serotonin_baseline', 0.12),      'phasic': 0.0},
            'ACh': {'tonic': getattr(params, 'acetylcholine_baseline', 0.12),   'phasic': 0.0},
            'NA':  {'tonic': getattr(params, 'norepinephrine_baseline', 0.12),  'phasic': 0.0},
        }
        st = float(getattr(params, 'receptor_slope_tonic', 4.0))
        sp = float(getattr(params, 'receptor_slope_phasic', 3.0))
        self.receptors = {
            # v113 FIX (M25): D1 EC50 0.18→0.45, D2 EC50 0.12→0.15.
            # v112 AUTOPSY: D1 at EC50=0.18 with DA mean=0.53 gave D1=0.97
            # at ALL times — no differential gating possible.
            # v113: EC50=0.45 sits in the middle of the actual DA operating
            # range (~0.20-0.55 after injection reduction). This creates:
            #   DA=0.20 (quiet):  D1≈0.18 → LTP suppressed (×0.18)
            #   DA=0.35 (active): D1≈0.35 → moderate LTP (×0.35)
            #   DA=0.50 (burst):  D1≈0.57 → strong LTP (×0.57)
            #   DA=0.65 (peak):   D1≈0.77 → peak LTP (×0.77)
            # This 4× D1 swing between quiet and peak is the differential
            # gating the paper requires (Algorithm 3: Δw ∝ D1_act).
            # D2 EC50=0.15 keeps it stably active at tonic DA baseline.
            'D1':     ReceptorSubtype('D1',     'DA',  0.45, 1.0, False, sp),
            'D2':     ReceptorSubtype('D2',     'DA',  0.15, 1.0, True,  st),
            '5HT1A':  ReceptorSubtype('5HT1A',  '5HT', 0.05, 1.0, True,  st),
            '5HT2A':  ReceptorSubtype('5HT2A',  '5HT', 0.30, 1.0, False, sp),
            '5HT4':   ReceptorSubtype('5HT4',   '5HT', 0.20, 1.0, False, sp),
            'M1':     ReceptorSubtype('M1',     'ACh', 0.30, 1.0, False, sp),
            'M2':     ReceptorSubtype('M2',     'ACh', 0.10, 1.0, True,  st),
            'beta1':  ReceptorSubtype('beta1',  'NA',  0.20, 1.0, False, sp),
            'alpha2': ReceptorSubtype('alpha2', 'NA',  0.08, 1.0, True,  st),
        }

    def get_flat_levels(self) -> dict:
        return {
            'dopamine':       self.levels['DA']['tonic']  + self.levels['DA']['phasic'],
            'serotonin':      self.levels['5HT']['tonic'] + self.levels['5HT']['phasic'],
            'acetylcholine':  self.levels['ACh']['tonic'] + self.levels['ACh']['phasic'],
            'norepinephrine': self.levels['NA']['tonic']  + self.levels['NA']['phasic'],
        }

    def set_level(self, name: str, value: float):
        mapping = {'dopamine': 'DA', 'serotonin': '5HT',
                   'acetylcholine': 'ACh', 'norepinephrine': 'NA'}
        key = mapping.get(name, name)
        if key in self.levels:
            # v110 FIX: Clamp to [0, 2.0] matching the system ceiling.
            # Was [0, 1.0] which truncated synced-back values (DA mean ~1.6)
            # making the sync ineffective — tonic was always capped at 1.0
            # while get_flat_levels() returned tonic+phasic up to 2.0.
            self.levels[key]['tonic'] = max(0.0, min(2.0, value))

    def update(self, network_activity: dict, dt: float):
        p = self.params
        mean_act = network_activity.get('mean_activity', 0.0)
        exc_frac = network_activity.get('excitatory_fraction', 0.0)
        change_rate = network_activity.get('state_change_rate', 0.0)
        # v110 FIX: Fallback default matches new param value (200.0, was 5000.0)
        tau_ton = getattr(p, 'tau_tonic', 200.0)
        tau_pha = getattr(p, 'tau_phasic', 200.0)
        rr = getattr(p, 'neuromod_release_rate', 0.005)

        for key, bl_attr in [('DA', 'dopamine_baseline'), ('5HT', 'serotonin_baseline'),
                             ('ACh', 'acetylcholine_baseline'), ('NA', 'norepinephrine_baseline')]:
            baseline = getattr(p, bl_attr, 0.12)
            # v113 FIX (M25): DA-specific faster tonic decay.
            # BIOINSPIRED: DA tonic clearance is faster than other monoamines
            # because DAT has highest affinity and density in striatum/NAcc.
            # 5-HT, ACh, NE use standard tau_ton; DA uses tau_ton/2.
            effective_tau_ton = tau_ton * 0.5 if key == 'DA' else tau_ton
            self.levels[key]['tonic'] += dt / effective_tau_ton * (baseline - self.levels[key]['tonic'])
            self.levels[key]['phasic'] += dt / tau_pha * (0.0 - self.levels[key]['phasic'])

        # v112 FIX (M25): DA phasic release gets dedicated multiplier.
        # DA encodes salience/reward with burst-firing 3-5× baseline (§1).
        # Without this, DA barely varies (v111: 90%-range=0.085) and D1
        # never crosses its EC50 threshold. With 4× multiplier, phasic DA
        # peaks at ~0.30-0.45 during active periods, creating the D1 swing
        # (0.38→0.92) needed for differential LTP gating.
        da_phasic_mult = getattr(self.params, 'da_phasic_release_multiplier', 4.0)
        self.levels['DA']['phasic']  += rr * da_phasic_mult * change_rate * dt
        self.levels['5HT']['tonic'] += rr * mean_act * dt
        self.levels['ACh']['phasic'] += rr * exc_frac * dt
        self.levels['NA']['phasic']  += rr * change_rate * dt

        # Crosstalk (Algorithm 5)
        da_ach_strength = float(getattr(self.params, 'da_ach_crosstalk_strength', 0.35))
        da_ach_tonic = float(getattr(self.params, 'da_ach_tonic_suppression', 0.08))
        da_drive = max(0.0, self.levels['DA']['phasic'])
        self.levels['ACh']['phasic'] *= max(0.0, 1.0 - da_ach_strength * da_drive)
        self.levels['ACh']['tonic'] = max(0.0, self.levels['ACh']['tonic'] - da_ach_tonic * da_drive * dt)
        self.levels['5HT']['tonic'] += 0.02 * (self.levels['NA']['tonic'] + self.levels['NA']['phasic']) * dt

        for key in self.levels:
            for comp in ('tonic', 'phasic'):
                self.levels[key][comp] = max(0.0, min(2.0, self.levels[key][comp]))

    def compute_receptor_activations(self) -> dict:
        activations = {}
        cap = float(getattr(self.params, 'receptor_concentration_cap', 1.0))
        for rname, receptor in self.receptors.items():
            parent = receptor.parent_modulator
            conc = self.levels[parent]['tonic'] if receptor.is_tonic else (
                self.levels[parent]['tonic'] + self.levels[parent]['phasic'])
            if cap > 0.0:
                conc = max(0.0, min(cap, float(conc)))
            else:
                conc = max(0.0, float(conc))
            activations[rname] = receptor.compute_activation(conc)
        return activations

    def to_dict(self) -> dict:
        return {'levels': self.levels, 'receptors': {k: v.to_dict() for k, v in self.receptors.items()}}
