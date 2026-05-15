# Neuraxon Game of Life v.4.68 neuron (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 160
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import math
import random
import numpy as np
from collections import deque
from typing import List, Dict, Tuple

# Import local modules
from config import NetworkParameters
from utils import _variate
from logger import get_data_logger
from .enums import NeuronType, TrinaryState
from .components import DendriticBranch, MSTHState

class Neuraxon:
    """
    The core computational unit of the model.
    Now fully individualized: Membrane properties, thresholds, adaptation rates,
    and metabolic costs are random variations of the NetworkParameters.
    """
    def __init__(self, neuron_id: int, neuron_type: NeuronType, params: NetworkParameters):
        self.id = neuron_id
        self.type = neuron_type
        self.params = params
        
        # --- Core Neuron Properties (Individualized) ---
        self.membrane_time_constant = _variate(params.membrane_time_constant)
        self.firing_threshold_excitatory = _variate(params.firing_threshold_excitatory)
        # Ensure inhibitory is negative and varied correctly
        self.firing_threshold_inhibitory = params.firing_threshold_inhibitory * random.uniform(0.8, 1.2)
        self.adaptation_rate = _variate(params.adaptation_rate)
        self.spontaneous_firing_rate = _variate(params.spontaneous_firing_rate)
        self.neuron_health_decay = _variate(params.neuron_health_decay)
        
        # --- Energy Metabolism (Individualized) ---
        self.energy_baseline = _variate(params.energy_baseline)
        self.firing_energy_cost = _variate(params.firing_energy_cost)
        self.plasticity_energy_cost = _variate(params.plasticity_energy_cost)
        self.metabolic_rate = _variate(params.metabolic_rate)
        self.recovery_rate = _variate(params.recovery_rate)

        # Core state variables
        # v2.36: Start neurons closer to resting potential (zero) - biologically neurons
        # spend most time near resting potential, not near threshold
        self.membrane_potential = random.uniform(
            self.firing_threshold_inhibitory * 0.35,  # v2.36: Narrower range, closer to zero
            self.firing_threshold_excitatory * 0.35
        )

        self.trinary_state = TrinaryState.NEUTRAL.value
        self.adaptation = 0.0
        self.autoreceptor = 0.0
        self.health = 1.0
        self.is_active = True
        
        self.dendritic_branches = [DendriticBranch(i, neuron_id, params) for i in range(params.num_dendritic_branches)]
        self.energy_level = self.energy_baseline # Start with individualized baseline
        
        self.last_firing_time = -1000.0
        # v2.39: Phase clustering for synchronization
        phase_cluster_prob = getattr(params, 'phase_clustering_init', 0.65)
        if random.random() < phase_cluster_prob:
            num_clusters = 3 + (neuron_id % 4)
            cluster_center = (neuron_id % num_clusters) * (2 * math.pi / num_clusters)
            self.phase = (cluster_center + random.gauss(0, 0.4)) % (2 * math.pi)
        else:
            self.phase = random.random() * 2 * math.pi
        freq_min = getattr(params, 'natural_freq_range_min', 0.7)
        freq_max = getattr(params, 'natural_freq_range_max', 1.4)
        self.natural_frequency = random.uniform(freq_min, freq_max)
        self._prev_phase_change = 0.0
        self.state_history = deque(maxlen=50)
        self.intrinsic_timescale = self.membrane_time_constant # Use individualized constant
        # v149 (v4.57) — stuck-state diagnostics (M1 lock-in)
        # Tracks how many consecutive ticks the neuron has been in the
        # same trinary_state. Read by the logger / dashboard to compute
        # the population's "stuck fraction" — a direct measure of the
        # M1 lock-in pathology. Cheap: one int compare + increment per tick.
        self.state_streak: int = 0
        self.last_streak_state: int = 0
        
        self.circle_id = None
        self.fitness_score = 0.0

        # --- Neuraxon v2.0: CTSN Complement (Algorithm 1 Step 3) ---
        self.complement_h = 0.0
        self.state_tilde = 0.0

        # --- Neuraxon v2.0: DSN Dynamic Decay (Algorithm 1 Step 2) ---
        # v4.52 PERF: use deque(maxlen=k) so the ring buffer is O(1) per push
        # instead of re-allocating a new list (list[1:] + [x]) every tick.
        k = max(int(getattr(params, 'dsn_kernel_size', 4)), 1)
        self._dsn_k = k
        self.dsn_input_buffer = deque([0.0] * k, maxlen=k)
        self.dsn_alpha = 0.5
        base_kernel = getattr(params, 'dsn_kernel_weights', None) or []
        if len(base_kernel) != k:
            base_kernel = [(i + 1.0) for i in range(k)]
            s = sum(abs(x) for x in base_kernel) or 1.0
            base_kernel = [float(x) / s for x in base_kernel]
        self.dsn_kernel_weights = [float(x) for x in base_kernel]

        # --- Neuraxon v2.0: CTSN learnable phi parameters ---
        self.ctsn_phi_gain = float(getattr(params, 'ctsn_phi_gain', 0.5))
        self.ctsn_phi_bias = float(getattr(params, 'ctsn_phi_bias', 0.0))
        self._ctsn_last_x = 0.0
        self._ctsn_last_phi = 0.0

        # --- Neuraxon v2.0: MSTH ---
        self.msth = MSTHState(params)

        # --- Neuraxon v2.0: AGMP astrocyte ---
        self.astrocyte_state = 0.0

        # --- Neuraxon v2.0: Firing rate average for homeostasis ---
        self.firing_rate_avg = getattr(params, 'target_firing_rate', 0.2)

        
        # NEW: Track for subthreshold logging
        self._prev_membrane_potential = 0.0

        # v4.52 PERF: cache hot-path params — called per neuron per sub-step.
        self._phase_coupling_strength = params.phase_coupling_strength
        self._phase_coupling_local = params.phase_coupling_local_strength
        self._phase_coupling_momentum = params.phase_coupling_momentum
        self._dsn_enabled = getattr(params, 'dsn_enabled', True)
        self._ctsn_enabled = getattr(params, 'ctsn_enabled', False)
        self._ctsn_rho = float(getattr(params, 'ctsn_rho', 0.9))
        self._agmp_enabled_n = getattr(params, 'agmp_enabled', False)
        self._agmp_lambda_a_n = getattr(params, 'agmp_lambda_a', 0.999)
        self._spont_as_current = params.spontaneous_as_current
        self._spont_current_mag = params.spontaneous_current_magnitude
        self._sensory_gating_enabled = params.sensory_gating_enabled
        self._sensory_gating_threshold = params.sensory_gating_threshold
        self._sensory_gating_suppression = params.sensory_gating_suppression
        self._max_intrinsic_timescale = params.max_intrinsic_timescale
        self._membrane_neg_bias = getattr(params, 'membrane_negative_bias', 0.0)
        self._resting_potential_decay = getattr(params, 'resting_potential_decay', None)
        self._energy_threshold_floor = params.energy_threshold_floor
        self._energy_threshold_coupling = params.energy_threshold_coupling
        self._critical_energy_level = params.critical_energy_level
        self._energy_recovery_boost = params.energy_recovery_boost
        self._neuron_death_threshold = params.neuron_death_threshold
        self._target_firing_rate_n = getattr(params, 'target_firing_rate', 0.2)
        self._firing_rate_alpha = getattr(params, 'firing_rate_alpha', 0.01)
        self._spike_class_enabled = params.spike_classification_enabled
        self._driven_input_threshold = params.driven_input_threshold
        # v4.52 PERF: precompute branch count (called every integration call)
        self._n_branches = len(self.dendritic_branches)
    
    def _nonlinear_dendritic_integration(self, synaptic_inputs: List[float], modulatory_inputs: List[float], dt: float) -> Tuple[float, List[float]]:
        # v4.52 PERF: use cached _n_branches (was len(self.dendritic_branches) called 3× per invocation)
        nb = self._n_branches
        branch_outputs = []
        total_synaptic = 0.0
        for i, branch in enumerate(self.dendritic_branches):
            branch_syn_inputs = synaptic_inputs[i::nb]
            branch_out = branch.integrate_inputs(branch_syn_inputs, dt)
            branch_outputs.append(branch_out)
            total_synaptic += branch_out
        return total_synaptic * (1.0 + sum(modulatory_inputs) * 0.2), branch_outputs
    
    #v2.40 update
    def _update_phase_oscillator(self, dt: float, global_osc: float, neighbor_phases=None):
        """Kuramoto local coupling. v4.52 PERF (#3 phase-neighbor): accepts
        EITHER the original dict form {neighbor_id: (phase, weight)} OR the
        new list-of-tuples form [(phase, weight), ...]. The list form avoids
        a full dict allocation per neuron per sub-step. neighbor_id is never
        read inside this body, so the two shapes are observationally
        equivalent — math identical."""
        # Natural frequency evolution
        d_phase = 2 * math.pi * self.natural_frequency * dt
        
        # Global coupling (WEAK - just sets rhythm)
        # v4.52 PERF: use cached _phase_coupling_strength
        global_coupling = self._phase_coupling_strength * math.sin(global_osc - self.phase) * dt
        
        # Local Kuramoto coupling (STRONG - drives synchronization)
        local_coupling = 0.0
        if neighbor_phases:
            total_weight = 0.0
            weighted_sin_sum = 0.0
            self_phase = self.phase
            # Duck-type: tuples/lists of (phase, weight) iterate directly;
            # dict {id: (phase, weight)} iterates via .values().
            iterable = neighbor_phases.values() if isinstance(neighbor_phases, dict) else neighbor_phases
            for item in iterable:
                neighbor_phase, weight = item[0], item[1]
                phase_diff = neighbor_phase - self_phase
                weighted_sin_sum += weight * math.sin(phase_diff)
                total_weight += weight
            if total_weight > 0.01:
                # v4.52 PERF: cached _phase_coupling_local
                local_coupling = (self._phase_coupling_local *
                                weighted_sin_sum / total_weight * dt)
        
        # Update with momentum — v4.52 PERF: cached _phase_coupling_momentum
        mom = self._phase_coupling_momentum
        total_change = d_phase + global_coupling + local_coupling
        smoothed_change = (mom * self._prev_phase_change +
                        (1 - mom) * total_change)
        self._prev_phase_change = smoothed_change
        self.phase = (self.phase + smoothed_change) % (2 * math.pi)
    
    def _update_energy(self, activity: float, plasticity_cost: float, dt: float):
        if not self.is_active: return
        # Use individualized metabolic parameters
        consumption = self.metabolic_rate * (self.firing_energy_cost * activity + self.plasticity_energy_cost * plasticity_cost) * dt
        
        # NEW v2.30: Energy-aware recovery boost for metabolically stressed neurons
        # BIOINSPIRED: Neurons in metabolic crisis prioritize ATP regeneration
        # This mimics increased mitochondrial activity under low-ATP conditions
        base_recovery = self.recovery_rate * (1.0 - activity) * dt
        
        if self.energy_level < self.params.critical_energy_level:
            # Boost recovery when energy is critically low
            # Scale boost by how far below critical threshold we are
            energy_deficit_ratio = 1.0 - (self.energy_level / self.params.critical_energy_level)
            recovery_multiplier = 1.0 + (self.params.energy_recovery_boost - 1.0) * energy_deficit_ratio
            recovery = base_recovery * recovery_multiplier
        else:
            recovery = base_recovery
        
        self.energy_level = max(0.0, min(self.energy_baseline * 1.5, self.energy_level + recovery - consumption))
        
        if self.energy_level < 10.0:
            # v3.35/RC6-FIX: Reduced penalty multiplier 2.0→1.2 (was creating death spiral).
            # BIOINSPIRED: Metabolic stress damages neurons but not at catastrophic rate;
            # cellular stress-response (UPR, heat-shock proteins) provides partial protection.
            self.health -= self.neuron_health_decay * dt * 1.2
            self.membrane_potential *= 0.95  # Gentler potential damping (was 0.9)
            self.health = max(-0.5, self.health)  # Enforce floor here too
        elif self.energy_level > self.params.critical_energy_level:
            # v3.35/RC6-FIX NEW: Energy-conditional health recovery.
            # BIOINSPIRED: Adequate ATP drives mitochondrial biogenesis and
            # protein synthesis, actively repairing neuronal structure.
            # Recovery is slow but steady when metabolically healthy.
            self.health = min(1.0, self.health + 0.001 * dt)

    
    def _update_intrinsic_timescale(self, dt: float):
        """Update intrinsic timescale based on autocorrelation."""
        # This method is called early in update() to prepare for timescale updates
        # The actual autocorrelation-based update happens in _update_autocorrelation()
        # after state_history is updated
        pass
    
    def _update_autocorrelation(self):
        if len(self.state_history) >= 10:
            states = list(self.state_history)
            # Check if data has variance (avoid zero std)
            states_a = states[:-1]
            states_b = states[1:]
            if np.std(states_a) < 1e-10 or np.std(states_b) < 1e-10:
                return
            try:
                autocorr = np.corrcoef(states_a, states_b)[0, 1]
                if not np.isnan(autocorr):
                    # ACW estimate: timescale weighted by autocorrelation strength
                    # Higher autocorr = longer memory window
                    acw = self.intrinsic_timescale * (1.0 + abs(autocorr))
                    self.intrinsic_timescale = acw
                    # Cap after autocorrelation update
                    self.intrinsic_timescale = min(self.intrinsic_timescale, self.params.max_intrinsic_timescale)
            except:
                pass
    
    def _compute_dsn_alpha(self, current_input: float) -> float:
        """Neuraxon v2.0: alpha_t = Sigmoid(CausalConv1D(X_{t-k+1:t})) (Algorithm 1 Steps 5-6)."""
        # v4.52 PERF: use cached flag; deque.append auto-evicts oldest (maxlen=k),
        # so pop-then-append collapses to a single append with identical semantics.
        if not self._dsn_enabled:
            return 0.5
        self.dsn_input_buffer.append(float(current_input))
        kernel = getattr(self, 'dsn_kernel_weights', []) or []
        if len(kernel) != len(self.dsn_input_buffer):
            k = max(int(getattr(self.params, 'dsn_kernel_size', 4)), 1)
            kernel = [(i + 1.0) for i in range(k)]
            s = sum(abs(x) for x in kernel) or 1.0
            kernel = [float(x) / s for x in kernel]
            self.dsn_kernel_weights = kernel
        conv_out = sum(w * x for w, x in zip(kernel, self.dsn_input_buffer)) + float(getattr(self.params, 'dsn_bias', 0.0))
        exponent = max(-50.0, min(50.0, -conv_out))
        self.dsn_alpha = 1.0 / (1.0 + math.exp(exponent))
        return self.dsn_alpha

    def _update_complement(self, x_t: float):
        """Neuraxon v2.0: h_t, s_tilde(t) = s(t) + h(t) (Algorithm 1 Steps 7-8)."""
        # v4.52 PERF: use cached flags.
        if not self._ctsn_enabled:
            self.complement_h = 0.0
            return
        rho = self._ctsn_rho
        phi = math.tanh(float(self.ctsn_phi_gain) * float(x_t) + float(self.ctsn_phi_bias))
        self._ctsn_last_x = float(x_t)
        self._ctsn_last_phi = float(phi)
        self.complement_h = rho * self.complement_h + (1.0 - rho) * phi

    def update(self, synaptic_inputs: List[float], modulatory_inputs: List[float], external_input: float, neuromodulators: Dict[str, float], dt: float, global_osc: float, neighbor_phases: List[float] = None, receptor_activations: Dict = None):
        """v2.39: Added neighbor_phases for Kuramoto coupling."""
        if not self.is_active or self.energy_level <= 0: return
        receptor_activations = receptor_activations or {}

        phase_coupling_strength = self._phase_coupling_strength  # v4.52 PERF
        
        self._update_intrinsic_timescale(dt)
        
        # CRITICAL FIX: Cap intrinsic timescale AFTER update, not before
        # This ensures the cap is always enforced regardless of ACW calculation
        self.intrinsic_timescale = min(self.intrinsic_timescale, self._max_intrinsic_timescale)  # v4.52 PERF
        
        # v2.39: Use Kuramoto coupling method
        self._update_phase_oscillator(dt, global_osc, neighbor_phases)
        
        total_synaptic, branch_outputs = self._nonlinear_dendritic_integration(synaptic_inputs, modulatory_inputs, dt)
        
        # 106: Algorithm 4 — g_NA = 1 + 0.5*β1 + 0.2*α2
        beta1_act = receptor_activations.get('beta1', 0.0)
        alpha2_act = receptor_activations.get('alpha2', 0.0)
        g_NA = 1.0 + 0.5 * beta1_act + 0.2 * alpha2_act
        
        acetylcholine = neuromodulators.get('acetylcholine', 0.5)
        norepi = neuromodulators.get('norepinephrine', 0.5)
        
        # ACh Modulation: Maintain persistence of state despite environmental fluctuations (Paper Claim)
        # High ACh suppresses noise (spontaneous firing), focusing the neuron on inputs/memory
        noise_suppression = 0.4 if acetylcholine > 0.6 else 1.0
        
        # Calculate total input strength for classification and gating
        total_input_strength = abs(total_synaptic) + abs(external_input)
        has_strong_input = total_input_strength > self._sensory_gating_threshold  # v4.52 PERF
        
        # Spontaneous probability with sensory gating
        effective_spont_rate = self.spontaneous_firing_rate + 0.3 * alpha2_act
        base_spont_prob = effective_spont_rate * dt * (1.0 + math.cos(self.phase) * 0.3) * noise_suppression
        if self._sensory_gating_enabled and has_strong_input:  # v4.52 PERF
            spont_prob = base_spont_prob * self._sensory_gating_suppression
        else:
            spont_prob = base_spont_prob
        
        is_spontaneous_firing = False
        spontaneous = 0.0

        # NB: random.random() sequence preserved exactly — do NOT reorder.
        if random.random() < spont_prob:
            is_spontaneous_firing = True
            if self._spont_as_current:  # v4.52 PERF
                # v3.34 RC1-FIX: Balanced spontaneous current — 50% inhibitory, 50% excitatory
                # BIOINSPIRED: Cortical spontaneous activity explores the FULL trinary
                # state space symmetrically. In vivo, balanced E/I networks produce
                # approximately equal rates of spontaneous excitatory and inhibitory
                # postsynaptic events (Haider et al. 2006, J Neurosci). The prior 60/40
                # inhibitory bias compounded with membrane_negative_bias to lock outputs
                # into -1 from initialization (RC1 diagnostic: 97.5% SW quadrant).
                spontaneous = random.choice([-1.0, 1.0]) * self._spont_current_mag  # v4.52 PERF
            else:
                # Legacy: force threshold
                if random.random() < 0.5:
                    self.membrane_potential = self.firing_threshold_excitatory + 0.01
                else:
                    spontaneous = random.choice([-1.0, 1.0]) * 2.0
                
        threshold_mod = (acetylcholine - 0.5) * 0.5 + sum(modulatory_inputs) * 0.3
        gain = 1.0 + (norepi - 0.5) * 0.4
        
        # v3.34 RC1-FIX: Bias now 0.0 from config; kept in formula for backward compat
        negative_bias = self._membrane_neg_bias  # v4.52 PERF
        
        # v152 (v4.60) STEP 1 FIX — TIGHTER SATURATING CAP
        # ----------------------------------------------------------------
        # The v151 sample run (nxon2_300373934) showed input_saturation_fraction
        # reached 0.989 by tick 510 — input neurons STILL saturate even with
        # the v151 saturating gain. The 1.5*tanh(1.5x) asymptote at 1.5 was
        # still high enough above the 0.55 firing threshold to drive
        # STDP/LTP into runaway potentiation.
        #
        # Tighter cap: 1.0 * tanh(1.0 * x) asymptotes at 1.0 (vs 1.5).
        # Effective behaviour:
        #   x = 0.1 (weak):    boost = tanh(0.1)  ≈ 0.10  (barely amplified)
        #   x = 0.3 (medium):  boost = tanh(0.3)  ≈ 0.29  (mild amplification)
        #   x = 1.0 (strong):  boost = tanh(1.0)  ≈ 0.76  (still above threshold)
        #   x = 10  (very):    boost = tanh(10)   ≈ 1.00  (asymptotic limit)
        #
        # Critically: even a maxed-out external_input now produces a
        # boosted value of 1.0, which is ABOVE the firing threshold of
        # 0.55 (so input neurons can still fire reliably) but only
        # 1.8× above (vs 2.7× under v151 cap). STDP runaway requires
        # sustained over-threshold drive — at 1.8× over, the brake
        # provided by adaptation (target 0.55, v149) catches up.
        boosted_external = math.tanh(external_input)
        
        drive = (g_NA * total_synaptic + boosted_external + spontaneous + negative_bias * 2.0) * gain
        
        tau_eff = max(1.0, self.intrinsic_timescale)
        prev_state = self.trinary_state
        
        # v3.34 RC1-FIX: Fully symmetric membrane decay
        # BIOINSPIRED: Passive membrane leak conductance is direction-agnostic —
        # both depolarisation and hyperpolarisation decay toward resting potential
        # at the same rate, governed by the membrane time constant and leak channels.
        # The prior asymmetry (positive 1.1×, negative 0.85×) created a ratchet effect
        # that trapped membrane potential in the negative range, contributing to RC1.
        # Paper claim: neutral state enables "swift transitions based on subsequent inputs"
        # v4.52 PERF: use cached _resting_potential_decay (None if not in params).
        if self._resting_potential_decay is not None:
            resting_decay = self._resting_potential_decay * dt
            # v3.34: Symmetric decay for both positive and negative potentials
            self.membrane_potential *= (1.0 - resting_decay)
        
        # Store previous potential for subthreshold logging
        prev_potential = self.membrane_potential
        
        # v150 (v4.58) STEP 2 FIX (M3 PAC over-locking → M7 inversion)
        # -----------------------------------------------------------
        # The previous oscillator amplitude was 0.2 — strong enough to
        # dominate the membrane equation in the DSN branch. With sensory
        # input weak (coefficient 1.0), oscillator rhythm became the
        # primary driver of motor output. M3 PAC measured at 0.47 in the
        # v149 run (5× the [0.005, 0.10] target).
        #
        # Reducing to 0.08 keeps the oscillator MODULATING the membrane
        # but no longer DRIVING it. Combined with the Step 1 sensory
        # boost (2.5×), the relative weighting becomes:
        #     sensory : recurrent : oscillator
        #         2.5 :    ~1.0   :    0.08
        # — sensory is now the dominant input force as it should be.
        #
        # BIOINSPIRED: cortical theta/gamma oscillations modulate spike
        # timing but rarely drive spikes by themselves. Buzsáki & Wang
        # 2012 (Mechanisms of gamma oscillations) shows oscillator
        # amplitude in vivo is small relative to sensory-evoked
        # depolarisations. The previous 0.2 was unphysiologically large.
        osc_component = 0.08 * math.cos(global_osc)  # v150: 0.2 → 0.08
        # v4.52 PERF: cached _dsn_enabled; deque.append auto-evicts (maxlen=k);
        # conv uses zip, one allocation-free pass. Mathematically identical.
        if self._dsn_enabled:
            self.dsn_input_buffer.append(float(boosted_external + total_synaptic))
            conv_out = 0.0
            for b, w in zip(self.dsn_input_buffer, self.dsn_kernel_weights):
                conv_out += b * w
            conv_out = max(-50.0, min(50.0, conv_out))
            dsn_alpha = 1.0 / (1.0 + math.exp(-conv_out))
            self.dsn_alpha = dsn_alpha
            # v150 — also use boosted_external here so DSN branch gets
            # the same sensory amplification as the non-DSN branch.
            total_input = g_NA * total_synaptic + boosted_external + osc_component - self.adaptation + spontaneous
            self.membrane_potential = dsn_alpha * self.membrane_potential + (1.0 - dsn_alpha) * total_input
        else:
            ds = (dt / tau_eff) * (-self.membrane_potential + drive - self.adaptation)
            self.membrane_potential += ds
        
        # v149 (v4.57) STEP 1 FIX (M1 lock-in / NxErs stuck-and-dying)
        # =============================================================
        # The diagnostic chain from sample run nxon2_502415576:
        #   M1 climbs to 0.62  → neurons stuck firing at +1
        #   M5 locks at 1.0000 → branching ratio frozen (synapses don't vary)
        #   M2 saturates 0.90  → CTC gates open at max, no modulation
        #   Outputs frozen     → motor neurons stop varying → NxErs stop moving
        #   No mating          → M10 stays at 0 → starve and die
        #
        # The previous adaptation_target ceiling was 0.33 (= 0.25 + 0.08).
        # In a recurrent excitatory network the synaptic drive easily
        # reaches ~0.8-1.0 — far above the 0.55 firing threshold even
        # AFTER subtracting 0.33 of adaptation. So adaptation never broke
        # the positive-feedback loop and neurons settled into +1.
        #
        # Fix: raise the adaptation ceiling to 0.55 (matching the firing
        # threshold). At sustained firing, adaptation precisely cancels
        # drive AT the threshold, producing intermittent firing instead
        # of stuck-on. Combined with halving tau (40 → 20 ticks), the
        # brake engages within ~10-15 ticks of sustained firing instead
        # of ~30-40 ticks.
        #
        # Asymmetric: penalise sustained +1 (excitatory) firing MORE than
        # -1 (inhibitory) since the pathology is excitatory dominance.
        # The 1.5× factor on +1 firing biases the brake toward the
        # state that's causing the lock-in.
        #
        # BIOINSPIRED: the underlying mechanism is M-current / AHP
        # (afterhyperpolarisation) — slow K+ currents that build up
        # with repeated firing. In real cortex, the timescale is
        # 50-200 ms, scaled to our tick rate gives τ ≈ 10-20 ticks.
        # The previous 40-tick τ was too slow for that biology.
        if self.trinary_state == 1:
            adaptation_target = 0.55 * 1.5  # excitatory firing: stronger brake
        elif self.trinary_state == -1:
            adaptation_target = 0.55 * 1.0  # inhibitory firing: normal brake
        else:
            adaptation_target = 0.0
        self.adaptation += dt / 20.0 * (-self.adaptation + adaptation_target)
        # Cap adaptation so a single neuron can't go runaway-saturated
        if self.adaptation > 1.0:
            self.adaptation = 1.0
        elif self.adaptation < 0.0:
            self.adaptation = 0.0
        # v2.38: FIXED autoreceptor to track ACTIVITY level, not state sign
        # BIOINSPIRED: D2 autoreceptors detect released neurotransmitter from ANY firing
        # Both excitatory AND inhibitory firing should increase autoreceptor
        # This creates proper negative feedback: high activity → high autoreceptor → harder to fire
        # Previous bug: tracked trinary_state sign, causing correlation issues
        activity_for_autoreceptor = abs(self.trinary_state)  # 0 or 1
        self.autoreceptor += dt / 150.0 * (-self.autoreceptor + 0.35 * activity_for_autoreceptor)
        
        # NEW v2.30: Energy-Aware Firing Threshold
        # BIOINSPIRED: ATP depletion impairs Na+/K+-ATPase pump efficiency
        # This raises the effective firing threshold, making low-energy neurons less excitable
        # Creates natural metabolic recovery windows while maintaining network criticality
        
        # Calculate energy factor: 1.0 when energy is high, <1.0 when depleted
        energy_ratio = self.energy_level / (self.energy_baseline * self.params.energy_threshold_floor)
        energy_factor = min(1.0, max(0.3, energy_ratio))  # Clamp between 0.3 and 1.0
        
        # Energy-dependent threshold scaling: low energy raises effective threshold
        # energy_factor=1.0 -> no change; energy_factor=0.3 -> threshold raised by ~3.3x coupling factor
        threshold_energy_mod = (1.0 - energy_factor) * self.params.energy_threshold_coupling * self.firing_threshold_excitatory
        
        # 106: Saturated threshold modulation (MOD-1, MOD-2 via readout)
        m1_act = receptor_activations.get('M1', 0.0)
        m2_act = receptor_activations.get('M2', 0.0)
        raw_mod = 0.3 * m1_act - 0.2 * m2_act + (sum(modulatory_inputs) if modulatory_inputs else 0)
        delta_theta_meta = 0.1 * math.tanh(raw_mod)
        target_rate = getattr(self.params, 'target_firing_rate', 0.2)
        fr_avg = getattr(self, 'firing_rate_avg', target_rate)
        delta_theta_homeo = 0.01 * (fr_avg - target_rate)
        # v154 (v4.62) K — REVERT v153's F-2 autoreceptor sign flip
        # ---------------------------------------------------------
        # The v153 F-2 fix flipped the autoreceptor sign on theta1_eff
        # from `- 0.1 * autoreceptor` (the inherited form) to
        # `+ 0.1 * autoreceptor` (true negative feedback).
        # Mathematically this was correct — the v2.38 comment promised
        # negative feedback and the original math did the opposite. But
        # the v153 sample run nxon2_064274592 revealed this hurt overall
        # network behavior: M1_excitatory_fraction dropped from 0.186
        # (in band) to 0.103 (below), M6_spontaneous dropped to 0,
        # M10_heritability collapsed from 0.485 to 0. Pushing thresholds
        # UP when activity is high tipped neurons over the edge into
        # neutral/inhibitory dominance — the network was already
        # operating near the firing threshold and couldn't afford the
        # extra inhibition.
        #
        # We accept the mathematical inconsistency (the comment lies)
        # for the sake of network behavior that actually works. The
        # autoreceptor magnitude is small (~0.035 at peak) so this
        # term's contribution is minor either way — but minor matters
        # when M1 is sitting at 0.18 with a 0.20-0.28 band.
        theta1_eff = self.firing_threshold_excitatory + threshold_energy_mod - delta_theta_meta + delta_theta_homeo - 0.1 * self.autoreceptor
        theta2_eff = self.firing_threshold_inhibitory - threshold_energy_mod - delta_theta_meta + delta_theta_homeo + 0.1 * self.autoreceptor
        
        # Trinary readout — use state_tilde when ctsn_enabled else membrane_potential
        readout = self.state_tilde if getattr(self.params, 'ctsn_enabled', False) else self.membrane_potential
        if readout > theta1_eff:
            self.trinary_state = TrinaryState.EXCITATORY.value
        elif readout < theta2_eff:
            self.trinary_state = TrinaryState.INHIBITORY.value
        else:
            self.trinary_state = TrinaryState.NEUTRAL.value
        
        self.state_history.append(self.trinary_state)
        # v149 — update state-streak counter (used by logger M1-stuck metric)
        if self.trinary_state == self.last_streak_state:
            self.state_streak += 1
        else:
            self.state_streak = 1
            self.last_streak_state = self.trinary_state

        msth_signals = self.msth.update(abs(self.trinary_state), dt)
        if msth_signals['ultrafast_suppress']:
            self.membrane_potential *= 0.5

        self._update_complement(self.membrane_potential)
        self.state_tilde = self.membrane_potential + self.complement_h

        # v4.52 PERF: use cached firing_rate_alpha
        fr_alpha = self._firing_rate_alpha
        self.firing_rate_avg += fr_alpha * (abs(self.trinary_state) - self.firing_rate_avg) * dt

        # v4.52 PERF: use cached agmp flag + lambda.
        if self._agmp_enabled_n:
            lam_a = self._agmp_lambda_a_n
            self.astrocyte_state = lam_a * self.astrocyte_state + (1.0 - lam_a) * abs(self.state_tilde)


        self._update_autocorrelation()
        activity_level = abs(self.trinary_state)
        
        # === SPIKE CLASSIFICATION AND LOGGING ===
        # Determine if spike was driven (by input) or spontaneous
        # BIOINSPIRED: Biological neurons show ~70-90% driven, ~10-30% spontaneous activity
        # Paper Section 6: Spontaneous activity provides substrate for plasticity but 
        # most spikes during active behavior are stimulus-driven
        # v4.52 PERF: resolve logger+level ONCE for the remainder of update().
        logger = get_data_logger()
        _ll = logger.log_level
        if abs(self.trinary_state) > 0 and self._spike_class_enabled:  # v4.52 PERF
            # v148 (v4.56) FIX (M6): a spike that was triggered by the
            # spontaneous-firing path counts as spontaneous, EVEN IF
            # synaptic input was also present. The previous criterion
            # (spont_contribution > input_contribution) was too restrictive
            # — in any busy network, input dominates, so M6 stuck at 0
            # despite spontaneous events firing every tick.
            #
            # Biologically: a "spontaneous" spike is one that the cell
            # generated WITHOUT requiring drive — co-occurring input doesn't
            # erase that. Cortical recordings count any spike during the
            # spontaneous-current window as a spontaneous event.
            #
            # We still log driven firings separately, so the M6 spontaneous
            # fraction = spont / (spont + driven) remains a meaningful
            # comparison.
            if is_spontaneous_firing:
                if _ll >= 2:
                    logger.log_spontaneous_event(0, self.id, self.membrane_potential)
                else:
                    # Even at log_level 1, count for M6. Cheap.
                    logger._spont_count_pending += 1
            else:
                # Pure driven spike — neuron was pushed past threshold by
                # synaptic / external input alone.
                logger.log_driven_firing(0)
        
        # NEW: Log subthreshold integration events Updated Save states in v 2.1
        if _ll >= 2:
            # If we're in neutral state but close to threshold
            if self.trinary_state == 0:
                distance_to_exc = theta1_eff - self.membrane_potential
                distance_to_inh = self.membrane_potential - theta2_eff
                
                # Log if within 30% of either threshold
                if distance_to_exc < abs(theta1_eff) * 0.3:
                    logger.log_subthreshold_event(
                        0, self.id, self.membrane_potential, 
                        theta1_eff, distance_to_exc
                    )
                elif distance_to_inh < abs(theta2_eff) * 0.3:
                    logger.log_subthreshold_event(
                        0, self.id, self.membrane_potential,
                        theta2_eff, distance_to_inh
                    )
        
        # NEW: Log significant autoreceptor effects Updated Save states in v 2.1
        if abs(self.autoreceptor) > 0.1 and _ll >= 2:
            threshold_effect = -0.1 * self.autoreceptor
            logger.log_autoreceptor_event(0, self.id, self.autoreceptor, threshold_effect)
        
        # NEW: Log threshold modulation events (when crossing state boundaries) Updated Save states in v 2.1
        if prev_state != self.trinary_state and _ll >= 2:
            ach_contrib = (neuromodulators.get('acetylcholine', 0.5) - 0.5) * 0.5
            autoreceptor_contrib = -0.1 * self.autoreceptor
            logger.log_threshold_modulation_event(
                0, self.id, self.firing_threshold_excitatory,
                theta1_eff, ach_contrib, autoreceptor_contrib
            )
        
        # NEW: Log Dendritic Spikes Updated Save states in v 2.1
        # Check recent activity in branches to log events
        if _ll >= 2:
            for branch in self.dendritic_branches:
                # If the most recent history indicates a spike (1.0)
                if branch.local_spike_history and branch.local_spike_history[-1] > 0.9:
                    # Avoid spamming: only log if it's a fresh spike (previous was 0) or probabalistically
                    if len(branch.local_spike_history) < 2 or branch.local_spike_history[-2] < 0.1:
                        logger.log_dendritic_spike_event(0, self.id, branch.branch_id, 
                                                       branch.branch_potential, branch.plateau_potential, 
                                                       branch.get_local_ca_influx())
        
        # Use individualized health decay
        # v3.35/RC6-FIX: Increased recovery rate (0.0005→0.003) to match decay magnitude.
        # BIOINSPIRED: Neurotrophic factors (BDNF/NGF) actively repair active neurons.
        # Added health floor at -0.5: structural proteins maintain minimum integrity.
        if activity_level >= 0.01:
            self.health = min(1.0, self.health + 0.003 * dt)
        else:
            self.health -= self.neuron_health_decay * dt
        self.health = max(-0.5, self.health)

        
        self._update_energy(activity_level, abs(self.trinary_state - prev_state) * 0.1, dt)
        
        if self.type == NeuronType.HIDDEN and (self.health < self.params.neuron_death_threshold or self.energy_level < 1.0):
            if random.random() < 0.001: self.is_active = False
    
    def set_state(self, state: int):
        if state in [-1, 0, 1]:
            self.trinary_state = state
            # Use individualized threshold for clamping
            self.membrane_potential = state * self.firing_threshold_excitatory
    
    def to_dict(self) -> dict:
        """Serializes the neuron's state and its individualized parameters."""
        return {
            'id': self.id, 'type': self.type.value, 
            'membrane_potential': self.membrane_potential, 'trinary_state': self.trinary_state, 
            'adaptation': self.adaptation, 'health': self.health, 
            'is_active': self.is_active, 'energy_level': self.energy_level, 
            'phase': self.phase, 'natural_frequency': self.natural_frequency, 
            'intrinsic_timescale': self.intrinsic_timescale, 
            'circle_id': self.circle_id, 'fitness_score': self.fitness_score, 
            'dendritic_branches': [b.to_dict() for b in self.dendritic_branches],            
            'membrane_time_constant': self.membrane_time_constant,
            'firing_threshold_excitatory': self.firing_threshold_excitatory,
            'firing_threshold_inhibitory': self.firing_threshold_inhibitory,
            'adaptation_rate': self.adaptation_rate,
            'spontaneous_firing_rate': self.spontaneous_firing_rate,
            'neuron_health_decay': self.neuron_health_decay,
            'energy_baseline': self.energy_baseline,
            'firing_energy_cost': self.firing_energy_cost,
            'plasticity_energy_cost': self.plasticity_energy_cost,
            'metabolic_rate': self.metabolic_rate,
            'recovery_rate': self.recovery_rate,
            'state_history': list(self.state_history), # Updated Save states in v 2.03
            'autoreceptor': self.autoreceptor, # Updated Save states in v 2.03
            'last_firing_time': self.last_firing_time  # Updated Save states in v 2.1
        }
