# Neuraxon Game of Life v.4.79 network (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 171
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import math
import random
import cmath
import numpy as np
from collections import defaultdict, deque
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

# Import local modules
from config import NetworkParameters
from logger import get_data_logger
from .enums import NeuronType, TrinaryState
from .neuron import Neuraxon
from .components import Synapse, ITUCircle, NeuromodulatorSystem, OscillatorBank, MSTHState

class NeuraxonNetwork:
    """
    The main class that orchestrates the entire network simulation. It manages the collections
    of neurons and synapses, global signals like neuromodulators and oscillators, and executes
    the simulation steps, including all forms of plasticity and evolution.
    """
    def __init__(self, params: Optional[NetworkParameters] = None):
        self.params = params or NetworkParameters()
        # UPDATED v4.5: Input neurons now include Song/Hearing
        # Indices: 0-2=physical, 3=hunger, 4=sight, 5=smell, 6=daynight, 7=temp, 8=proprio, 9=song
        self.input_neurons: List[Neuraxon] = []
        self.hidden_neurons: List[Neuraxon] = []
        # UPDATED v4.5: Output neurons now include Sing
        # Indices: 0=MoveX, 1=MoveY, 2=Social, 3=Mate, 4=GiveFood, 5=Resting, 6=Sing
        self.output_neurons: List[Neuraxon] = []
        self.all_neurons: List[Neuraxon] = []
        self.synapses: List[Synapse] = []
        self.itu_circles: List[ITUCircle] = [] # A list of evolutionary units.
        
        # Initialize global neuromodulator levels.
        self.neuromodulators = {mod: getattr(self.params, f'{mod}_baseline') for mod in ['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']}
        self.modulator_grid = np.full((10, 10, 4), 0.5)

        # --- Neuraxon v2.0: Full Neuromodulator System with receptor subtypes ---
        self.neuromod_system = NeuromodulatorSystem(self.params)
        self.receptor_activations = {}
        self.current_game_tick = 0

        # --- Neuraxon v2.0: Multi-band Oscillator Bank with PAC ---
        osc_coupling = getattr(self.params, 'oscillator_coupling', 0.1)
        self.oscillator_bank = OscillatorBank(coupling=osc_coupling)

        self.time = 0.0
        self.step_count = 0

        # === NEURAXON v2.0: AGMP Astrocyte-Gated Plasticity (Algorithm 1 Step 5) ===
        if getattr(self.params, 'agmp_enabled', False):
            neuron_map = {n.id: n for n in self.all_neurons}
            for syn in self.synapses:
                if syn.integrity <= 0: continue
                pre = neuron_map.get(syn.pre_id)
                post = neuron_map.get(syn.post_id)
                if pre is None or post is None or not pre.is_active or not post.is_active: continue
                syn.update_chrono_traces(pre.trinary_state)
                syn.update_eligibility(pre.trinary_state, post.trinary_state, self.params)
                m_t = self.neuromod_system.levels['DA']['phasic']
                lambda_a = getattr(self.params, 'agmp_lambda_a', 0.95)
                s_tilde = getattr(post, 'state_tilde', float(abs(post.trinary_state)))
                prev_astro = getattr(post, 'astrocyte_state', 0.0)
                a_t = lambda_a * prev_astro + (1.0 - lambda_a) * abs(s_tilde)
                post.astrocyte_state = a_t
                eta = getattr(self.params, 'agmp_eta', 0.005)
                delta_agmp = eta * m_t * a_t * syn.eligibility
                syn.w_fast = max(-1.0, min(1.0, syn.w_fast + delta_agmp * 0.3))
                syn.w_slow = max(-1.0, min(1.0, syn.w_slow + delta_agmp * 0.1))

        # === NEURAXON v2.0: Homeostatic Synaptic Scaling ===
        h_rate = getattr(self.params, 'homeostatic_rate', 0.0005)
        target_fr = getattr(self.params, 'target_firing_rate', 0.2)
        for neuron in self.all_neurons:
            if not neuron.is_active or neuron.type == NeuronType.INPUT: continue
            fr_avg = getattr(neuron, 'firing_rate_avg', target_fr)
            med_gain = getattr(neuron, 'msth', None)
            mg = med_gain.medium_gain if med_gain else 1.0
            scale = 1.0 + h_rate * (target_fr - fr_avg) * mg
            for syn in self.synapses:
                if syn.post_id == neuron.id and syn.integrity > 0:
                    syn.w_fast = max(-1.0, min(1.0, syn.w_fast * scale))
                    syn.w_slow = max(-1.0, min(1.0, syn.w_slow * scale))

        self.total_energy_consumed = 0.0
        
        # v2.39: Phase coherence tracking
        self.phase_coherence_history = deque(maxlen=100)
        self.last_phase_coherence = 0.0
        
        # Initialize random phase offsets for the three global oscillators.
        self.oscillator_phase_offsets = (random.random() * 2 * math.pi, random.random() * 2 * math.pi, random.random() * 2 * math.pi)
        self.activation_history = deque(maxlen=1000)
        self.branching_ratio = 1.0 # A measure of network criticality. A value near 1.0 is often optimal.
        
        # NEW v2.2503: Track excitatory fraction history for adaptive threshold homeostasis
        self.excitatory_fraction_history = deque(maxlen=200)
        self.last_adaptive_threshold_tick = 0
        
        # Construct the network.
        self._initialize_neurons()
        self._initialize_synapses()
        self._initialize_itu_circles()
        
        # v4.1: Pre-indexed synapse lookup maps — eliminates O(N*S) scans
        self._rebuild_synapse_indexes()
    
    def _rebuild_synapse_indexes(self):
        """v4.1: Build O(1) lookup maps for synapses. Called after init and structural changes."""
        from collections import defaultdict
        self._syn_by_pre = defaultdict(list)
        self._syn_by_post = defaultdict(list)
        self._neuron_map = {n.id: n for n in self.all_neurons}
        self._afferent_by_pre = defaultdict(list)
        for s in self.synapses:
            self._syn_by_pre[s.pre_id].append(s)
            self._syn_by_post[s.post_id].append(s)
            if s.is_afferent:
                self._afferent_by_pre[s.pre_id].append(s)
    
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
        """
        num_neurons = len(self.all_neurons)
        if num_neurons <= 1: return
        
        k = max(2, min(num_neurons - 1, int(self.params.small_world_k)))
        rewire_p = max(0.0, min(1.0, float(self.params.small_world_rewire_prob)))
        existing = set()
        neuron_degrees = defaultdict(int)
        
        # Step 1: Regular ring lattice
        for idx, pre in enumerate(self.all_neurons):
            for offset in range(1, k // 2 + 1):
                j = (idx + offset) % num_neurons
                post = self.all_neurons[j]
                
                # Step 2: Rewiring
                if random.random() < rewire_p:
                    if self.params.preferential_attachment:
                        candidates = []; weights = []
                        for _ in range(10): 
                            j_new = random.randrange(num_neurons)
                            cand = self.all_neurons[j_new]
                            if cand.id != pre.id:
                                candidates.append(cand)
                                weights.append(neuron_degrees[cand.id] + 1)
                        if candidates: post = random.choices(candidates, weights=weights, k=1)[0]
                        else:
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
                
                # Create synapse
                if pre.id == post.id or (pre.type == NeuronType.OUTPUT and post.type == NeuronType.INPUT) or (pre.id, post.id) in existing: continue
                syn = Synapse(pre.id, post.id, self.params)
                self.synapses.append(syn)
                if pre.type == NeuronType.INPUT:
                    syn.mark_as_afferent()
                existing.add((pre.id, post.id))
                neuron_degrees[pre.id] += 1
                neuron_degrees[post.id] += 1
                
        # v164 (v4.72) — sensorimotor_coupling architectural bias.
        # The v161-v163 fitness component sensory_motor_corr stayed near
        # zero (max 0.125 in 1425 trials at 900s) because the network had
        # NO architectural preference for input→output pathways: direct
        # sensory→motor synapses formed at the same uniform rate as any
        # other random link. v164 reads `neural.sensorimotor_coupling`
        # (default 0.0 = pre-v164 behaviour) and multiplies the
        # input→output edge probability by (1 + sensorimotor_coupling).
        # This gives sensory pathways an architectural head-start; the
        # plasticity rule then amplifies whichever correlations develop.
        try:
            from architecture import get_param as _arch_get
            _sm_coupling = float(_arch_get('neural', 'sensorimotor_coupling', 0.0))
            _sm_coupling = max(0.0, min(4.0, _sm_coupling))  # clamp to sane bounds
        except Exception:
            _sm_coupling = 0.0
        
        # Step 3: Random connections
        for pre in self.all_neurons:
            for post in self.all_neurons:
                if pre.id == post.id or (pre.type == NeuronType.OUTPUT and post.type == NeuronType.INPUT) or (pre.id, post.id) in existing: continue
                # v164 — boost input→output direct connections by the
                # architecture's sensorimotor_coupling factor. Other
                # pairs use the original probability.
                if pre.type == NeuronType.INPUT and post.type == NeuronType.OUTPUT:
                    p_edge = self.params.connection_probability * 0.25 * (1.0 + _sm_coupling)
                    p_edge = min(0.95, p_edge)  # never saturate
                else:
                    p_edge = self.params.connection_probability * 0.25
                if random.random() < p_edge:
                    syn = Synapse(pre.id, post.id, self.params)
                    self.synapses.append(syn)
                    if pre.type == NeuronType.INPUT:
                        syn.mark_as_afferent()
                    existing.add((pre.id, post.id))
                    neuron_degrees[pre.id] += 1
                    neuron_degrees[post.id] += 1
                    
        # v4.1: Build neighbor_synapses using pre-indexed groups — O(S*degree) not O(S²)
        _pre_groups = defaultdict(list)
        for s in self.synapses:
            _pre_groups[s.pre_id].append(s)
        for s in self.synapses:
            s.neighbor_synapses = [ns for ns in _pre_groups[s.pre_id] if ns.post_id != s.post_id]
    
    def _initialize_itu_circles(self):
        """Groups hidden neurons into ITUs for the Aigarth evolutionary process."""
        if len(self.hidden_neurons) < self.params.itu_circle_radius * 2: return
        num_circles = max(1, len(self.hidden_neurons) // self.params.itu_circle_radius)
        neurons_per_circle = max(1, len(self.hidden_neurons) // num_circles)
        for circle_idx in range(num_circles):
            start = circle_idx * neurons_per_circle
            end = len(self.hidden_neurons) if circle_idx == num_circles - 1 else start + neurons_per_circle
            self.itu_circles.append(ITUCircle(circle_idx, self.hidden_neurons[start:end], self.params, total_circles=num_circles))
    
    def _global_oscillatory_drive(self) -> float:
        """Generates a complex, global oscillatory signal."""
        t = self.time
        low = math.sin(2.0 * math.pi * self.params.oscillator_low_freq * t + self.oscillator_phase_offsets[0])
        mid = math.sin(2.0 * math.pi * self.params.oscillator_mid_freq * t + self.oscillator_phase_offsets[1])
        high = math.sin(2.0 * math.pi * self.params.oscillator_high_freq * t + self.oscillator_phase_offsets[2])
        return self.params.oscillator_strength * (low * 0.5 + low * mid * 0.3 + mid * high * 0.2)
    
    def _sync_neuromod_to_system(self):
        """v110 FIX (M18): Bidirectional sync — dict → NeuromodulatorSystem.
        
        ROOT CAUSE: simulate_step() line 7167 does:
            self.neuromodulators = self.neuromod_system.get_flat_levels()
        This overwrites the dict (where all decay and behavioral boosts live)
        with the v2.0 system's internal state (which was never decayed).
        
        This method syncs the authoritative dict values BACK into the v2.0
        system so that:
          1. Decay from _update_neuromodulator_diffusion persists across ticks
          2. Game-loop behavioral boosts survive through simulate_step
          3. Receptor activations are computed from actual (decayed) levels
        
        BIOINSPIRED: In vivo, there is ONE extracellular concentration per
        monoamine. Reuptake, enzymatic degradation, diffusion, and receptor
        binding all operate on the same pool. The tonic/phasic decomposition
        is an analytical convenience, not a separate physical reservoir.
        """
        _mapping = {'dopamine': 'DA', 'serotonin': '5HT',
                    'acetylcholine': 'ACh', 'norepinephrine': 'NA'}
        for name, key in _mapping.items():
            total = self.neuromodulators.get(name, 0.12)
            phasic = self.neuromod_system.levels[key]['phasic']
            # Attribute the total to tonic, preserving the current phasic component.
            # Tonic = total - phasic (clamped to [0, 2.0]).
            # This way phasic transients from the v2.0 update() are preserved,
            # while the tonic baseline tracks the actual (decayed) concentration.
            new_tonic = max(0.0, min(2.0, total - phasic))
            self.neuromod_system.levels[key]['tonic'] = new_tonic

    def _update_neuromodulator_diffusion(self, dt: float):
        """Simulates the spatial diffusion of neuromodulators across the environment grid."""
        for i, mod in enumerate(['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']):
            grid = self.modulator_grid[:, :, i]
            laplacian = np.zeros_like(grid)
            laplacian[1:-1, 1:-1] = grid[2:, 1:-1] + grid[:-2, 1:-1] + grid[1:-1, 2:] + grid[1:-1, :-2] - 4 * grid[1:-1, 1:-1]
            self.modulator_grid[:, :, i] += self.params.diffusion_rate * laplacian * dt
            base = getattr(self.params, f'{mod}_baseline')
            
            # --- Antagonistic Decay Dynamics (Paper Claim) ---
            # DA and 5-HT are antagonists. To each natural decay, the opposite one can be added.
            decay_factor = 1.0
            if mod == 'dopamine':
                antagonist = self.neuromodulators.get('serotonin', 0.12)
                if antagonist > 0.5: decay_factor += (antagonist - 0.5)
            elif mod == 'serotonin':
                antagonist = self.neuromodulators.get('dopamine', 0.12)
                if antagonist > 0.5: decay_factor += (antagonist - 0.5)
            
            local_decay = self.params.neuromod_decay_rate  # new update v2.47 3.0
            if mod == 'norepinephrine':
                local_decay *= 3.0   
            # v109 FIX (M18): REMOVED the residual /10.0 divisor entirely.
            # v108 changed /100.0 to /10.0 but that still left decay 10x too weak.
            # With neuromod_decay_rate=0.6, effective per-tick pull is now:
            #   (0.15 - 1.87) * 0.6 * 1.0 * 1.0 = -1.032 per tick for DA at saturation
            # This robustly counterbalances circadian injection (~0.08/tick) and
            # behavioral boosts, allowing neuromodulators to return to baseline.
            # BIOINSPIRED: Tonic neuromodulator clearance in vivo is fast —
            # monoamine half-lives range from seconds (DA in striatum) to minutes.
            self.neuromodulators[mod] += (base - self.neuromodulators[mod]) * local_decay * decay_factor * dt
            
            # v108 NEW: Concentration-dependent excess decay (KEPT — serves as safety net)
            # BIOINSPIRED: When monoamine concentration exceeds ~3× baseline,
            # enzymatic degradation (MAO/COMT) and overflow metabolism kick in.
            # This prevents ceiling-pinning while preserving dynamics near baseline.
            excess_threshold = base * getattr(self.params, 'neuromod_excess_threshold_mult', 3.0)
            if self.neuromodulators[mod] > excess_threshold:
                excess = self.neuromodulators[mod] - excess_threshold
                excess_rate = getattr(self.params, 'neuromod_excess_decay_rate', 0.15)
                self.neuromodulators[mod] -= excess * excess_rate * dt
            
            # v3.33: Enzymatic degradation + reuptake transporter (MAO/COMT/NET/SERT/DAT/AChE)
            # BIOINSPIRED: Monoamine clearance follows Michaelis-Menten kinetics —
            # degradation rate is concentration-dependent and saturates at Vmax.
            # Each transporter has distinct kinetics: AChE > NET > DAT > SERT
            # At low concentrations: clearance ≈ (Vmax/Km) * [NT] (first-order)
            # At high concentrations: clearance → Vmax (zero-order, saturated enzyme)
            excess = max(0.0, self.neuromodulators[mod] - base)
            if excess > 0:
                vmax_key = {'norepinephrine': 'reuptake_vmax_ne', 'dopamine': 'reuptake_vmax_da',
                            'serotonin': 'reuptake_vmax_5ht', 'acetylcholine': 'reuptake_vmax_ach'}.get(mod)
                vmax = getattr(self.params, vmax_key, 0.05) if vmax_key else 0.05
                km = getattr(self.params, 'reuptake_km', 0.5)
                clearance = vmax * excess / (km + excess)
                self.neuromodulators[mod] = max(base, self.neuromodulators[mod] - clearance * dt)

    
    def _apply_homeostatic_plasticity(self):
        """A slow-acting plasticity rule that adjusts neuron excitability."""
        if self.step_count % 100 != 0: return 
        
        # NEW v2.2503: Apply network-wide adaptive threshold homeostasis
        self._apply_adaptive_threshold_homeostasis()
        
        
        logger = get_data_logger()
        
        for neuron in self.all_neurons:
            if neuron.type == NeuronType.HIDDEN and len(neuron.state_history) > 0:
                recent_activity = sum(abs(s) for s in neuron.state_history) / len(neuron.state_history)
                old_threshold = neuron.params.firing_threshold_excitatory
                
                if recent_activity > self.params.target_firing_rate * 1.2:
                    neuron.params.firing_threshold_excitatory += self.params.homeostatic_plasticity_rate
                elif recent_activity < self.params.target_firing_rate * 0.8:
                    neuron.params.firing_threshold_excitatory -= self.params.homeostatic_plasticity_rate
                
                # NEW: Log if threshold changed
                new_threshold = neuron.params.firing_threshold_excitatory
                if old_threshold != new_threshold and logger.log_level >= 2:
                    logger.log_homeostatic_event(
                        self.step_count, 
                        neuron.id, 
                        old_threshold, 
                        new_threshold, 
                        recent_activity
                    )
    
    def _apply_adaptive_threshold_homeostasis(self):
        """
        IMPROVED v2.2507: Criticality-seeking threshold homeostasis.
        
        Bioinspired Rationale (Paper Section 7):
        Biological neural networks maintain criticality through homeostatic scaling
        of intrinsic excitability. When activity is too low (subcritical), neurons
        become more excitable; when too high (supercritical), they become less excitable.
        
        This implementation directly targets branching ratio σ→1.0 using proportional
        control, mimicking the continuous adjustment seen in biological homeostasis.
        
        Changes from v2.2503:
        - Targets branching ratio directly instead of just excitatory fraction
        - Uses proportional control (stronger correction when further from target)
        - More frequent checks (every 20 ticks vs 50)
        - Larger adjustment magnitude with bounds
        - Also modulates oscillator strength as secondary drive
        """
        if not self.params.adaptive_threshold_enabled:
            return
        
        # More frequent checking for responsive homeostasis
        check_interval = max(20, self.params.adaptive_threshold_check_interval // 2)
        if self.step_count - self.last_adaptive_threshold_tick < check_interval:
            return
        
        self.last_adaptive_threshold_tick = self.step_count
        
        # Need sufficient history for reliable branching ratio
        if len(self.activation_history) < 15:
            return
        
        active_neurons = [n for n in self.all_neurons if n.is_active]
        if not active_neurons:
            return
        
        # === PRIMARY TARGET: Branching Ratio (Criticality) ===
        sigma = self.branching_ratio
        
        # v2.36: Also check excitatory fraction to ensure neutral dominance
        excitatory_count = sum(1 for n in active_neurons if n.trinary_state == 1)
        excitatory_fraction = excitatory_count / len(active_neurons)
        
        # v2.37b: Track inhibitory for E/I balance homeostasis
        inhibitory_count = sum(1 for n in active_neurons if n.trinary_state == -1)
        inhibitory_fraction = inhibitory_count / len(active_neurons)
        
        # Critical regime for branching ratio: σ ∈ [0.85, 1.15]
        sigma_ok = 0.85 <= sigma <= 1.15
        
        # v2.36: NEW - Also target appropriate excitatory fraction for neutral dominance
        target_exc = getattr(self.params, 'target_excitatory_fraction', 0.22)
        exc_ok = abs(excitatory_fraction - target_exc) < 0.08
        
        # Only skip adjustment if BOTH criticality AND excitatory fraction are good
        # v2.37b: Also check inhibitory fraction
        min_inh = getattr(self.params, 'min_inhibitory_fraction', 0.10)
        inh_ok = inhibitory_fraction >= min_inh
        
        if sigma_ok and exc_ok and inh_ok:
            return
        
        # v2.37b: PRIORITY 1 - Fix inhibitory deficit (most critical for E/I balance)
        if not inh_ok and inhibitory_fraction < min_inh:
            # Need more inhibitory - lower inhibitory threshold for ALL neurons
            deficit = min_inh - inhibitory_fraction
            adjustment_strength = min(0.05, deficit * 0.5)  # Proportional to deficit
            
            for neuron in active_neurons:
                # Move inhibitory threshold toward zero (make it easier to reach)
                new_inh = neuron.firing_threshold_inhibitory + adjustment_strength
                neuron.firing_threshold_inhibitory = max(-0.50, min(-0.10, new_inh))
            
            # Log the adjustment
            logger = get_data_logger()
            if logger.log_level >= 2:
                logger.log_homeostatic_event(
                    tick=self.step_count,
                    neuron_id=-2,  # -2 = inhibitory homeostasis
                    old_value=inhibitory_fraction,
                    new_value=min_inh,
                    activity=excitatory_fraction
                )
            return  # Don't compound adjustments
        
        # v2.37b: PRIORITY 2 - Excitatory fraction correction
        # This ensures neutral state dominance even at criticality
        adjustment = 0.0
        reason = None
        
        if sigma_ok and not exc_ok:
            # Sigma is good but excitatory fraction needs correction
            if excitatory_fraction > self.params.max_excitatory_fraction:
                # Too much excitatory - raise thresholds even though sigma is OK
                adjustment = self.params.adaptive_threshold_adjustment * 1.5
                reason = f"excess_excitatory_{excitatory_fraction:.3f}_target_{target_exc:.3f}"
            elif excitatory_fraction < self.params.min_excitatory_fraction:
                # Too little excitatory - lower thresholds
                adjustment = -self.params.adaptive_threshold_adjustment
                reason = f"deficit_excitatory_{excitatory_fraction:.3f}_target_{target_exc:.3f}"
            else:
                return
        else:
            # Original criticality-based adjustment continues below...
            # Proportional control: adjustment scales with distance from criticality
            # This mimics biological homeostasis where larger deviations cause stronger responses
            sigma_error = 1.0 - sigma
            
            # Base adjustment magnitude, scaled by error (max 2x base)
            base_adjustment = self.params.adaptive_threshold_adjustment
            proportional_gain = min(2.0, 1.0 + abs(sigma_error))
            adjustment_magnitude = base_adjustment * proportional_gain
            
            # === ALSO CHECK: Excitatory Fraction as Secondary Signal ===
            self.excitatory_fraction_history.append(excitatory_fraction)
            
            # Determine adjustment direction based on sigma
            if sigma < 0.85:  # Subcritical - network too quiet
                # Lower thresholds to increase excitability (biological: upregulate Na+ channels)
                adjustment = -adjustment_magnitude
                reason = f"subcritical_sigma_{sigma:.3f}_exc_{excitatory_fraction:.3f}"
                
                # Secondary boost: increase oscillator drive if very subcritical
                if sigma < 0.7:
                    self.params.oscillator_strength = min(0.40, 
                        self.params.oscillator_strength + 0.005)
            
            elif sigma > 1.15:  # Supercritical - network too active
                # Raise thresholds to decrease excitability (biological: upregulate K+ channels)
                adjustment = adjustment_magnitude
                reason = f"supercritical_sigma_{sigma:.3f}_exc_{excitatory_fraction:.3f}"
                
                # Secondary damping: reduce oscillator drive if very supercritical
                if sigma > 1.3:
                    self.params.oscillator_strength = max(0.10,
                        self.params.oscillator_strength - 0.005)
        
        # === APPLY GLOBAL THRESHOLD ADJUSTMENT ===
        if adjustment != 0.0:
            neurons_adjusted = 0
            for neuron in active_neurons:
                # Excitatory threshold: biological range [0.25, 1.2]
                new_exc_threshold = neuron.firing_threshold_excitatory + adjustment
                neuron.firing_threshold_excitatory = max(0.25, min(1.2, new_exc_threshold))
                
                # Inhibitory threshold: adjust symmetrically, range [-1.2, -0.25]
                new_inh_threshold = neuron.firing_threshold_inhibitory - adjustment
                neuron.firing_threshold_inhibitory = max(-1.2, min(-0.25, new_inh_threshold))
                
                neurons_adjusted += 1
            
            # === LOG THE HOMEOSTATIC EVENT ===
            logger = get_data_logger()
            if logger.log_level >= 2:
                logger.log_homeostatic_event(
                    tick=self.step_count,
                    neuron_id=-1,  # -1 indicates network-wide adjustment
                    old_value=sigma,  # Using sigma as reference
                    new_value=sigma + adjustment,  # Direction indicator
                    activity=excitatory_fraction
                )
                
                # Additional detail logging for analysis
                if hasattr(logger, 'time_series') and 'adaptive_threshold_adjustments' not in logger.time_series:
                    logger.time_series['adaptive_threshold_adjustments'] = []
                if hasattr(logger, 'time_series'):
                    logger.time_series.setdefault('adaptive_threshold_adjustments', []).append({
                        'tick': self.step_count,
                        'sigma': sigma,
                        'excitatory_fraction': excitatory_fraction,
                        'adjustment': adjustment,
                        'reason': reason,
                        'neurons_affected': neurons_adjusted,
                        'new_oscillator_strength': self.params.oscillator_strength
                    })
    
    def _apply_synaptic_weight_homeostasis(self):
        """
        NEW v2.31: Synaptic scaling for weight homeostasis.
        
        BIOINSPIRED RATIONALE (Turrigiano et al., 1998; Turrigiano & Nelson, 2004):
        Biological synapses undergo "synaptic scaling" - a homeostatic mechanism where
        total synaptic strength is multiplicatively normalized to prevent runaway
        excitation while preserving the relative differences between weights
        (which encode learned information).
        
        This addresses the weight drift problem identified in HF52/HF53 analysis:
        - HF52: Δw = 0.294, weights approaching saturation (0.82-0.90)
        - HF53: Δw = 0.441 (50% worse after energy patch)
        
        The mechanism works in three stages:
        1. DETECTION: Monitor per-neuron outgoing weight distribution
        2. SOFT SCALING: When mean |weight| exceeds threshold, apply multiplicative scaling
        3. DRIFT CORRECTION: Apply small bias toward zero to counteract systematic drift
        
        Key property: Multiplicative scaling preserves relative weight ratios,
        so learned patterns (which depend on relative strengths) are maintained.
        """
        if not self.params.weight_homeostasis_enabled:
            return
        
        # Apply at regular intervals (not every step for efficiency)
        if self.step_count % self.params.weight_homeostasis_interval != 0:
            return
        
        if not self.synapses:
            return
        
        # === STAGE 1: COMPUTE GLOBAL WEIGHT STATISTICS ===
        all_w_fast = [s.w_fast for s in self.synapses if s.integrity > 0]
        all_w_slow = [s.w_slow for s in self.synapses if s.integrity > 0]
        
        if not all_w_fast:
            return
        
        mean_abs_w_fast = sum(abs(w) for w in all_w_fast) / len(all_w_fast)
        mean_abs_w_slow = sum(abs(w) for w in all_w_slow) / len(all_w_slow)
        mean_w_fast = sum(all_w_fast) / len(all_w_fast)  # Signed mean (for drift detection)
        mean_w_slow = sum(all_w_slow) / len(all_w_slow)
        
        # === STAGE 2: PER-NEURON SYNAPTIC SCALING ===
        # v4.52 PERF: use pre-indexed _syn_by_pre instead of rebuilding from scratch;
        # filter integrity>0 on iteration (same output as the old defaultdict build).
        scaling_events = 0

        for pre_id, all_syns in self._syn_by_pre.items():
            synapses = [s for s in all_syns if s.integrity > 0]
            if not synapses:
                continue
            
            # Calculate mean absolute weight for this neuron's outgoing synapses
            local_w_fast = [s.w_fast for s in synapses]
            local_w_slow = [s.w_slow for s in synapses]
            
            local_mean_fast = sum(abs(w) for w in local_w_fast) / len(local_w_fast)
            local_mean_slow = sum(abs(w) for w in local_w_slow) / len(local_w_slow)
            
            # Check if scaling needed (weights approaching saturation)
            needs_scaling_fast = local_mean_fast > self.params.weight_saturation_threshold
            needs_scaling_slow = local_mean_slow > self.params.weight_saturation_threshold
            
            if needs_scaling_fast or needs_scaling_slow:
                scaling_events += 1
                
                for s in synapses:
                    # SOFT MULTIPLICATIVE SCALING
                    # Scale factor approaches target/current ratio gradually
                    # This preserves relative weight differences (learned patterns)
                    
                    if needs_scaling_fast and abs(s.w_fast) > 0.01:
                        # Compute scaling factor to move toward target
                        current_scale = abs(s.w_fast) / local_mean_fast
                        target_scale = self.params.weight_homeostasis_target / local_mean_fast
                        
                        # Soft scaling: gradually approach target
                        scale_factor = 1.0 - self.params.weight_homeostasis_rate * (1.0 - target_scale)
                        scale_factor = max(0.8, min(1.0, scale_factor))  # Limit scaling per step
                        
                        s.w_fast *= scale_factor
                    
                    if needs_scaling_slow and abs(s.w_slow) > 0.01:
                        current_scale = abs(s.w_slow) / local_mean_slow
                        target_scale = self.params.weight_homeostasis_target / local_mean_slow
                        
                        scale_factor = 1.0 - self.params.weight_homeostasis_rate * (1.0 - target_scale)
                        scale_factor = max(0.8, min(1.0, scale_factor))
                        
                        s.w_slow *= scale_factor
        
        # === STAGE 3: GLOBAL DRIFT CORRECTION ===
        # If mean weight is systematically positive (drift), apply small correction
        # This counteracts the LTP > LTD imbalance seen in HF53 (ratio 1.29)
        
        drift_correction = self.params.weight_drift_correction
        
        # Detect systematic drift direction
        if abs(mean_w_fast) > 0.1:  # Significant drift detected
            drift_sign = 1.0 if mean_w_fast > 0 else -1.0
            for s in self.synapses:
                if s.integrity > 0:
                    # Apply small bias opposite to drift direction
                    s.w_fast -= drift_sign * drift_correction * abs(s.w_fast)
        
        if abs(mean_w_slow) > 0.1:
            drift_sign = 1.0 if mean_w_slow > 0 else -1.0
            for s in self.synapses:
                if s.integrity > 0:
                    s.w_slow -= drift_sign * drift_correction * abs(s.w_slow)
        
        # === LOGGING ===
        logger = get_data_logger()
        if logger.log_level >= 2 and scaling_events > 0:
            # Log as homeostatic event (repurposing existing infrastructure)
            logger.log_homeostatic_event(
                tick=self.step_count,
                neuron_id=-1,  # -1 indicates network-wide event
                old_value=mean_abs_w_fast,
                new_value=self.params.weight_homeostasis_target,
                activity=scaling_events / max(1, len(self._syn_by_pre))
            )

    def _get_neighbor_phases(self, neuron_id: int):
        """v2.40: Phases of synaptically connected neighbors for Kuramoto.
        v4.1: Uses pre-indexed synapse maps — O(degree) instead of O(S).

        v4.52 PERF (#3 phase-neighbor): returns a list of (phase, weight)
        tuples instead of a dict keyed by neighbor_id. The downstream consumer
        (_update_phase_oscillator) does not read the id, so the list form is
        equivalent — and avoids per-call dict allocation + hash work.
        Per-neighbor weights are still accumulated when multiple synapses
        connect the same pair (one list entry per pair, weights summed via
        an intermediate dict only if duplicates are detected)."""
        nm_get = self._neuron_map.get
        out = []
        # Small optimisation: dedup+accumulate only when the same neighbor-id
        # appears twice (rare in this topology). We start list-only and
        # escalate to dict-backed merge only on conflict.
        seen_ids = None
        # Incoming synapses — syn.post_id == neuron_id by construction
        for syn in self._syn_by_post.get(neuron_id, ()):
            if syn.is_silent or syn.integrity < 0.1:
                continue
            weight = abs(syn.w_fast) + abs(syn.w_slow) * 0.5
            if weight < 0.05:
                continue
            pre_n = nm_get(syn.pre_id)
            if pre_n is not None and pre_n.is_active:
                if seen_ids is None:
                    seen_ids = {syn.pre_id: len(out)}
                    out.append((pre_n.phase, weight))
                else:
                    idx = seen_ids.get(syn.pre_id)
                    if idx is None:
                        seen_ids[syn.pre_id] = len(out)
                        out.append((pre_n.phase, weight))
                    else:
                        ph, w = out[idx]
                        out[idx] = (ph, w + weight)
        # Outgoing synapses — syn.pre_id == neuron_id by construction
        for syn in self._syn_by_pre.get(neuron_id, ()):
            if syn.is_silent or syn.integrity < 0.1:
                continue
            weight = abs(syn.w_fast) + abs(syn.w_slow) * 0.5
            if weight < 0.05:
                continue
            post_n = nm_get(syn.post_id)
            if post_n is not None and post_n.is_active:
                if seen_ids is None:
                    seen_ids = {syn.post_id: len(out)}
                    out.append((post_n.phase, weight))
                else:
                    idx = seen_ids.get(syn.post_id)
                    if idx is None:
                        seen_ids[syn.post_id] = len(out)
                        out.append((post_n.phase, weight))
                    else:
                        ph, w = out[idx]
                        out[idx] = (ph, w + weight)
        return out
    
    def _compute_phase_coherence(self) -> float:
        """v2.39: Compute Kuramoto order parameter R for phase coherence."""
        active = [n for n in self.all_neurons if n.is_active]
        if len(active) < 2:
            return 0.0
        complex_sum = sum(cmath.exp(1j * n.phase) for n in active)
        return abs(complex_sum) / len(active)

    def _compute_branching_ratio(self):
        """Calculates the branching ratio (sigma): A(t) / A(t-1)."""
        if len(self.activation_history) < 2:
            self.branching_ratio = 1.0
            return 1.0
        
        # Get the two most recent time steps
        a_prev = self.activation_history[-2]
        a_now  = self.activation_history[-1]
        
        # Calculate instantaneous ratio (add small epsilon to avoid div/0)
        sigma_inst = a_now / (a_prev + 1e-9)
        
        # Apply EWMA smoothing (0.95 keep, 0.05 update) for stability
        # Clamping between 0.1 and 10.0 prevents massive spikes during quiet periods
        self.branching_ratio = 0.95 * self.branching_ratio + 0.05 * max(0.1, min(10.0, sigma_inst))
        
        return self.branching_ratio
    
    def _get_sensory_input_dict(self) -> Dict[int, float]:
        """Generate external inputs from clamped input neuron states.
        v4.1: Uses pre-indexed afferent map — O(afferents) instead of O(inputs * S).

        v117 FIX: Afferents must respect their actual synaptic weights instead of adding
        the same scalar to every target. Otherwise sensorimotor couplings cannot emerge —
        all sensory channels look equivalent at the neuron interface."""
        external_inputs = {}
        meta_gain = getattr(self.params, 'meta_influence_gain', 0.25)
        reliability = float(getattr(self.params, 'afferent_synapse_reliability', 1.0))
        for inp_neuron in self.input_neurons:
            if not inp_neuron.is_active or inp_neuron.trinary_state == 0:
                continue
            input_signal = float(inp_neuron.trinary_state) * float(self.params.sensory_input_gain)
            for syn in self._afferent_by_pre.get(inp_neuron.id, ()):
                if syn.integrity <= 0.0 or syn.is_silent:
                    continue
                if reliability < 1.0 and random.random() > reliability:
                    continue
                aff_gain = getattr(self.params, 'proprioceptive_afferent_gain', 1.8) if inp_neuron.id == 8 else getattr(self.params, 'afferent_synapse_strength', 1.1)
                effective_weight = syn.w_fast + syn.w_slow + syn.w_meta * meta_gain
                external_inputs[syn.post_id] = external_inputs.get(syn.post_id, 0.0) + input_signal * effective_weight * aff_gain
        return external_inputs

    def _assign_branch_positions(self):
        """Neuraxon v2.0: Assign stable position index for neighbour plasticity d_ij."""
        groups = {}
        for syn in self.synapses:
            key = (syn.post_id, getattr(syn, 'branch_id', 0))
            groups.setdefault(key, []).append(syn)
        for syns in groups.values():
            syns_sorted = sorted(syns, key=lambda s: (s.pre_id, s.post_id))
            for pos, syn in enumerate(syns_sorted):
                syn.branch_index = pos
    
    def simulate_step(self, external_inputs: Optional[Dict[int, float]] = None):
        """Executes one full, orchestrated simulation step for the entire network."""
        sensory_inputs = self._get_sensory_input_dict()
        external_inputs = external_inputs or {}
        for nid, val in sensory_inputs.items():
            external_inputs[nid] = external_inputs.get(nid, 0.0) + val
        
        # v4.1: Avoid list() + np.var overhead — compute variance inline
        _ah = self.activation_history
        if len(_ah) > 10:
            _slice = list(_ah)[-10:]
            _mean = sum(_slice) / 10.0
            _var = sum((x - _mean) ** 2 for x in _slice) / 10.0
            dt = max(self.params.min_dt, min(self.params.max_dt, self.params.dt / (1.0 + _var)))
        else:
            dt = max(self.params.min_dt, min(self.params.max_dt, self.params.dt))
        
        osc_drive = self._global_oscillatory_drive()
        event_tick = int(getattr(self, 'current_game_tick', self.step_count))

        # v110 FIX (M18): Sync dict → v2.0 system BEFORE update.
        # Game-loop code (circadian boosts, behavioral scenarios, reward boosts)
        # modifies self.neuromodulators between ticks. Without this sync, all
        # those changes are overwritten by get_flat_levels() below.
        self._sync_neuromod_to_system()

        # v4.52 PERF: hoist active_neurons ONCE per simulate_step — was recomputed
        # several times (v2_activity loop, main update loop, energy accounting).
        active_neurons = [n for n in self.all_neurons if n.is_active]

        # === NEURAXON v2.0: Neuromodulator System Update ===
        # v4.52 PERF: single-pass aggregation — was 3 separate passes over `states`.
        v2_activity = {
            'mean_activity': 0.0, 'excitatory_fraction': 0.0, 'state_change_rate': 0.0
        }
        if active_neurons:
            _sum_abs = 0
            _count_exc = 0
            _count_changed = 0
            _n_active = len(active_neurons)
            for _n in active_neurons:
                _s = _n.trinary_state
                if _s > 0:
                    _sum_abs += _s
                    _count_exc += 1
                elif _s < 0:
                    _sum_abs += -_s
                if _s != getattr(_n, 'prev_state', 0):
                    _count_changed += 1
            v2_activity['mean_activity'] = _sum_abs / _n_active
            v2_activity['excitatory_fraction'] = _count_exc / _n_active
            v2_activity['state_change_rate'] = _count_changed / _n_active
        self.neuromod_system.update(v2_activity, dt)
        self.receptor_activations = self.neuromod_system.compute_receptor_activations()
        # v109 FIX (M16): Actually log receptor activations to time-series output.
        # The NeuromodulatorSystem computes all 9 receptor activations correctly,
        # but they were never persisted to the per_nxer_time_series parquet.
        # This caused M16 to report all NaN values.
        # v4.52 PERF: resolve logger + level once.
        logger = get_data_logger()
        _ll = logger.log_level
        if _ll >= 2:
            for r_name, r_val in self.receptor_activations.items():
                logger.current_step_data[f'receptor_{r_name}'] = r_val
        self.neuromodulators = self.neuromod_system.get_flat_levels()

        # === NEURAXON v2.0: Oscillator Bank Update ===
        self.oscillator_bank.update(dt)

        # ------------------------------------------------------------------
        # v4.52 PERF (#1 fuse-passes + skip-neutral, #2 pre-unpack receptors)
        # ------------------------------------------------------------------
        # Before: self.synapses was scanned up to FOUR times per simulate_step:
        #   (1) compute_input → syn_inputs / mod_inputs
        #   (2) calculate_delta_w
        #   (3) apply_update
        #   (4) homeostatic scaling
        # Each pass re-resolved pre/post via self.all_neurons[id] and re-ran
        # the `if s.integrity <= 0 or not pre.is_active or not post.is_active`
        # filter. We now build the filtered tuple list ONCE per step, and
        # reuse it across passes. `active_syns` entries are (synapse, pre, post).
        #
        # Skip-neutral fast path (#1c): when pre.trinary_state == 0, compute_input
        # already returns (0, 0) (components.py). We now short-circuit at the
        # LOOP level too so we skip the function call entirely — this is where
        # the cost is in CPython (method dispatch + arg packing).
        #
        # Receptor pre-unpack (#2): D1/D2/5HT2A/5HT1A values are the SAME for
        # every synapse in a single pass. We precompute them once here and pass
        # them as a tuple `_ra_cache`; Synapse.calculate_delta_w / apply_update
        # use the tuple when supplied instead of re-doing 4 dict lookups per
        # synapse per tick. Mathematically identical.
        # ------------------------------------------------------------------
        _nm = self._neuron_map
        _syns = self.synapses
        _all_n = self.all_neurons
        _ra = self.receptor_activations
        # Pre-unpack receptor values used by Synapse hot path.
        _ra_d1   = _ra.get('D1',    0.5)
        _ra_d2   = _ra.get('D2',    0.5)
        _ra_5ht2a = _ra.get('5HT2A', 0.5)
        _ra_5ht1a = _ra.get('5HT1A', 0.5)
        _ra_cache = (_ra_d1, _ra_d2, _ra_5ht2a, _ra_5ht1a)
        _nm_cache = (
            self.neuromodulators.get('dopamine',      0.5),
            self.neuromodulators.get('serotonin',     0.5),
            self.neuromodulators.get('acetylcholine', 0.5),
            self.neuromodulators.get('norepinephrine',0.5),
        )

        # PASS 0: build active-synapse list once, also accumulate synaptic
        # inputs per neuron inline (Pass-1-fused). Skip-neutral is applied
        # here so quiescent presynaptic neurons incur zero cost.
        syn_inputs = defaultdict(list)
        mod_inputs = defaultdict(list)
        _cur_time = self.time
        active_syns: List[Tuple] = []
        for s in _syns:
            if s.integrity <= 0: continue
            pre = _nm.get(s.pre_id)
            if pre is None or not pre.is_active: continue
            post = _nm.get(s.post_id)
            if post is None or not post.is_active:
                # Post-inactive: still include in active_syns? Historically
                # calculate_delta_w / apply_update skip when post inactive, so
                # it's safe to drop the synapse for later passes too. compute_input
                # would normally still run but its output would go to an inactive
                # post-neuron which does nothing with it. Drop.
                continue
            active_syns.append((s, pre, post))
            pre_state = pre.trinary_state
            if pre_state == 0:
                # Skip-neutral fast path — compute_input returns (0,0) here;
                # avoid the call & the axonal_delay branch entirely.
                # get_modulatory_effect only depends on w_meta * is_modulatory,
                # which is NON-ZERO even at neutral pre. Preserve that branch.
                me = s.get_modulatory_effect()
                if me != 0: mod_inputs[s.post_id].append(me)
                continue
            immediate, delayed = s.compute_input(pre_state, _cur_time)
            syn_inputs[s.post_id].append(immediate)
            if delayed > 0 and s.axonal_delay < dt:
                syn_inputs[s.post_id].append(delayed)
            me = s.get_modulatory_effect()
            if me != 0: mod_inputs[s.post_id].append(me)

        step_activity_sum = 0.0
        active_neuron_count = 0

        # PASS: per-neuron update (unchanged math; receptor_activations still
        # passed as dict because Neuraxon.update consumes several keys and
        # the per-call cost is already small relative to the neuron update).
        for n in active_neurons:
            ext = external_inputs.get(n.id, 0.0) + osc_drive
            neighbor_phases = self._get_neighbor_phases(n.id)
            n.update(syn_inputs.get(n.id, ()), mod_inputs.get(n.id, ()), ext, self.neuromodulators, dt, osc_drive, neighbor_phases,
                     receptor_activations=self.receptor_activations)
            step_activity_sum += abs(n.trinary_state)
            active_neuron_count += 1

        current_step_activity = step_activity_sum / max(1, active_neuron_count)
        self.activation_history.append(current_step_activity)

        # PASS 2: calculate_delta_w over the prebuilt active_syns tuple list.
        # Pre-compute the proprio-motor output-id set once (previously done
        # inside the loop per synapse).
        _proprio_motor_ids = (
            self.params.num_input_neurons + self.params.num_hidden_neurons,
            self.params.num_input_neurons + self.params.num_hidden_neurons + 1,
        )
        _proprio_boost = getattr(self.params, 'proprioceptive_stdp_boost', 2.5)
        for s, pre, post in active_syns:
            delta_w = s.calculate_delta_w(
                pre.trinary_state, post.trinary_state, self.neuromodulators, dt,
                receptor_activations=self.receptor_activations, tick=event_tick,
                _ra_cache=_ra_cache)
            # v107 FIX (M02): Boost STDP for proprioceptive→motor pathways.
            if (pre.type == NeuronType.INPUT
                and pre.id == 8
                and post.type == NeuronType.OUTPUT
                and post.id in _proprio_motor_ids):
                delta_w *= _proprio_boost
            s.potential_delta_w = delta_w

        # PASS 3: apply_update — reuses active_syns list.
        for s, pre, post in active_syns:
            neighbor_deltas = [ns.potential_delta_w for ns in s.neighbor_synapses]
            s.apply_update(dt, self.neuromodulators, neighbor_deltas,
                           receptor_activations=self.receptor_activations, tick=event_tick,
                           _ra_cache=_ra_cache)

        # === NEURAXON v2.0: AGMP Astrocyte-Gated Plasticity (Algorithm 1 Step 5) ===
        if getattr(self.params, 'agmp_enabled', False):
            neuron_map = self._neuron_map  # v4.1: Use pre-built map
            for syn in self.synapses:
                if syn.integrity <= 0: continue
                pre = neuron_map.get(syn.pre_id)
                post = neuron_map.get(syn.post_id)
                if pre is None or post is None or not pre.is_active or not post.is_active: continue
                syn.update_chrono_traces(pre.trinary_state)
                syn.update_eligibility(pre.trinary_state, post.trinary_state, self.params)
                m_t = self.neuromod_system.levels['DA']['phasic']
                lambda_a = getattr(self.params, 'agmp_lambda_a', 0.95)
                s_tilde = getattr(post, 'state_tilde', float(abs(post.trinary_state)))
                prev_astro = getattr(post, 'astrocyte_state', 0.0)
                a_t = lambda_a * prev_astro + (1.0 - lambda_a) * abs(s_tilde)
                post.astrocyte_state = a_t
                eta = getattr(self.params, 'agmp_eta', 0.005)
                delta_agmp = eta * m_t * a_t * syn.eligibility
                syn.w_fast = max(-1.0, min(1.0, syn.w_fast + delta_agmp * 0.3))
                syn.w_slow = max(-1.0, min(1.0, syn.w_slow + delta_agmp * 0.1))

        # === NEURAXON v2.0: Homeostatic Synaptic Scaling ===
        # v4.1: O(S) via pre-indexed _syn_by_post instead of O(N*S) nested scan
        # v4.52 PERF: iterate active_neurons (already filtered) instead of re-scanning all.
        h_rate = getattr(self.params, 'homeostatic_rate', 0.0005)
        target_fr = getattr(self.params, 'target_firing_rate', 0.2)
        for neuron in active_neurons:
            if neuron.type == NeuronType.INPUT: continue
            fr_avg = getattr(neuron, 'firing_rate_avg', target_fr)
            med_gain = getattr(neuron, 'msth', None)
            mg = med_gain.medium_gain if med_gain else 1.0
            scale = 1.0 + h_rate * (target_fr - fr_avg) * mg
            for syn in self._syn_by_post.get(neuron.id, ()):
                if syn.integrity > 0:
                    syn.w_fast = max(-1.0, min(1.0, syn.w_fast * scale))
                    syn.w_slow = max(-1.0, min(1.0, syn.w_slow * scale))

        # v4.52 PERF: reuse active_neurons list for energy accumulation.
        for n in active_neurons:
            activity = abs(n.trinary_state)
            self.total_energy_consumed += n.firing_energy_cost * activity * dt
                        
        self._update_neuromodulator_diffusion(dt)
        # v110 FIX (M18): Sync dict → v2.0 system AFTER diffusion decay.
        # _update_neuromodulator_diffusion applies baseline decay, excess decay,
        # and Michaelis-Menten reuptake to self.neuromodulators. Without syncing
        # back, all this decay is overwritten next tick by get_flat_levels().
        # This is the second half of the bidirectional sync — it closes the loop.
        self._sync_neuromod_to_system()

        # v113 FIX (M25): Reduce DA oscillatory coupling from 0.02→0.005.
        # The old 0.02 added ~0.01 DA per tick via oscillator drive,
        # contributing ~30% of the chronic DA elevation.
        for k in ('dopamine', 'serotonin', 'acetylcholine', 'norepinephrine'):
            osc_mod = 0.005 if k == 'dopamine' else 0.01 if k in ('serotonin', 'acetylcholine') else 0.0
            self.neuromodulators[k] = max(0.0, min(2.0, self.neuromodulators[k] + osc_drive * osc_mod))
        self._apply_homeostatic_plasticity()
        
        # NEW v2.31: Apply synaptic weight homeostasis after plasticity updates
        self._apply_synaptic_weight_homeostasis()
        
        self._apply_structural_plasticity()
        
        if self.step_count % self.params.evolution_interval == 0: self._evolve_itu_circles()
        
        self._compute_branching_ratio()
        self.time += dt
        self.step_count += 1
    
    def _apply_structural_plasticity(self):
        """Modifies the network's topology over time."""
        _old_count = len(self.synapses)
        self.synapses = [s for s in self.synapses if s.integrity > self.params.synapse_integrity_threshold]
        _pruned = _old_count != len(self.synapses)
        
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
                if pre.id != post.id and not (pre.type == NeuronType.OUTPUT and post.type == NeuronType.INPUT) and not any(ss.pre_id == pre.id and ss.post_id == post.id for ss in self.synapses):
                    self.synapses.append(Synapse(pre.id, post.id, self.params))
                    
        _structure_changed = False
        for n in self.hidden_neurons:
            if (n.health < self.params.neuron_death_threshold or n.energy_level < 5.0) and random.random() < 0.001:
                n.is_active = False
                _structure_changed = True
        
        # v4.1: Rebuild indexes if structure changed
        if _structure_changed or _pruned:
            self._rebuild_synapse_indexes()
    

    def _circle_metrics(self, circle):
        active = [n for n in circle.neurons if n.is_active]
        if not active:
            return 0.0, 0.0, 0.0

        # pattern/activity term (same spirit as logger)
        activity = sum(abs(n.trinary_state) for n in active) / len(active)

        # phase order parameter (this matches what you log in get_energy_status)
        phases = [n.phase for n in active]
        if len(phases) >= 2:
            sync = abs(sum(cmath.exp(1j * p) for p in phases) / len(phases))
        else:
            sync = 0.0

        # efficiency should NOT be "avg energy level" (leaks energy into fitness)
        # approximate per-tick energy spent by this circle
        spent = sum(n.firing_energy_cost * abs(n.trinary_state) for n in active) + 1e-9
        efficiency = activity / spent

        return activity, sync, efficiency
    
    def _evolve_itu_circles(self):
        """
        Executes one evolutionary cycle for all ITUs in the network.
        UPDATED: Uses real physics metrics instead of proxies.
        """
        for circle in self.itu_circles:
            # Calculate REAL metrics using the new helper
            activity_val, temporal_sync, energy_efficiency = self._circle_metrics(circle)
            
            # Note: compute_fitness expects a list for network_activity for the pattern weight,
            # but usually just takes the mean. We pass the calculated scalar relative to the circle.
            # To keep signature valid with existing circle.compute_fitness logic:
            network_activity_proxy = [activity_val] 
            
            fitness = circle.compute_fitness(network_activity_proxy, temporal_sync, energy_efficiency)
            
            for n in circle.neurons: 
                n.fitness_score = fitness
            
            # Apply mutation and selection.
            circle.mutate()
            pruned_ids = circle.prune_unfit_neurons()            
    
    def set_input_states(self, input_vector: List[int]):
        """
        Clamps the input neuron states to the provided trinary values.
        UPDATED v4.5: input_vector should have 10 values in {-1, 0, 1}:
        [Movement, Encounter, Terrain, Hunger, Sight, Smell, DayNight, Temperature, Proprioception, Song]
        """
        # Ensure we have the right number of inputs
        # v4.5: Validate input vector length matches expected (10 inputs)
        assert len(input_vector) == self.params.num_input_neurons, f"Expected {self.params.num_input_neurons} inputs, got {len(input_vector)}"
        while len(input_vector) < len(self.input_neurons):
            input_vector = list(input_vector) + [0]
        for i, neuron in enumerate(self.input_neurons):
            if i < len(input_vector):
                neuron.set_state(input_vector[i])
            else:
                neuron.set_state(0)
    
    def get_output_states(self) -> List[int]:
        """
        Returns the current trinary states of all output neurons.
        UPDATED v4.5: Returns 7 values:
        [MoveX, MoveY, Social, MateIntent, GiveFood, Resting, Sing]
        """
        return [n.trinary_state for n in self.output_neurons]
    
    def get_energy_status(self) -> Dict[str, float]:
        """Returns a summary of the network's current energy state."""
        # v4.52 PERF (#26): cmath already imported at module scope.

        active_neurons = [n for n in self.all_neurons if n.is_active]
        total_energy = sum(n.energy_level for n in active_neurons)
        avg_energy = total_energy / max(1, len(active_neurons))
                
        energy_spent = sum(max(0, n.energy_baseline - n.energy_level) for n in active_neurons)
        energy_spent += self.step_count * 0.01 * len(active_neurons)
        recent_activity = sum(self.activation_history) if self.activation_history else 0
        efficiency = recent_activity / max(1, energy_spent) if energy_spent > 0 else 0.0
        
        phases = [n.phase for n in active_neurons]
        if len(phases) >= 2:
            complex_phases = [cmath.exp(1j * p) for p in phases]
            order_param = abs(sum(complex_phases) / len(complex_phases))
            temporal_sync = order_param
        else:
            temporal_sync = 0.0
        
        return {
            'total_energy': total_energy, 
            'average_energy': avg_energy, 
            'efficiency': efficiency,
            'branching_ratio': self.branching_ratio,
            'temporal_sync': temporal_sync
        }
            
   
    def to_dict(self) -> dict:
        """Serializes the entire network state into a single dictionary."""
        return {
            'version': '4.5',  # UPDATED v4.5 (Voice/Song channels added)
            'num_inputs': len(self.input_neurons),
            'num_outputs': len(self.output_neurons),
            'parameters': asdict(self.params), 'neurons': {'input': [n.to_dict() for n in self.input_neurons], 'hidden': [n.to_dict() for n in self.hidden_neurons], 'output': [n.to_dict() for n in self.output_neurons]}, 'synapses': [s.to_dict() for s in self.synapses], 'neuromodulators': self.neuromodulators, 'time': self.time, 'step_count': self.step_count, 'energy_consumed': self.total_energy_consumed, 
        'branching_ratio': self.branching_ratio, 
        'modulator_grid': self.modulator_grid.tolist(),  # Updated Save states in v 2.1
        'oscillator_phase_offsets': self.oscillator_phase_offsets,  # Updated Save states in v 2.1
          'itu_circles': [{
            'circle_id': c.circle_id,
            'fitness_history': c.fitness_history,  # Updated Save states in v 2.1
            'mutation_rate': c.mutation_rate  # Updated Save states in v 2.1
        } for c in self.itu_circles],
        'activation_history': list(self.activation_history)    # Updated Save states in v 2.03
         }


def _rebuild_net_from_dict(d: dict) -> NeuraxonNetwork:
    """A utility function to reconstruct a complete NeuraxonNetwork object from a dictionary."""
    params = NetworkParameters(**d['parameters'])
    
    # UPDATED v4.5: Ensure correct input/output counts (10 / 7)
    params.num_input_neurons = d.get('num_inputs', 10)  # Default to v4.5 count
    params.num_output_neurons = d.get('num_outputs', 7)  # Default to v4.5 count
    
    # Handle loading older saves with fewer inputs/outputs — always upgrade to v4.5 shape
    if params.num_input_neurons < 10:
        params.num_input_neurons = 10
    if params.num_output_neurons < 7:
        params.num_output_neurons = 7
    
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
                
                # Individualized neuron parameters
                n.membrane_time_constant = nd.get('membrane_time_constant', params.membrane_time_constant)
                n.firing_threshold_excitatory = nd.get('firing_threshold_excitatory', params.firing_threshold_excitatory)
                n.firing_threshold_inhibitory = nd.get('firing_threshold_inhibitory', params.firing_threshold_inhibitory)
                n.adaptation_rate = nd.get('adaptation_rate', params.adaptation_rate)
                n.spontaneous_firing_rate = nd.get('spontaneous_firing_rate', params.spontaneous_firing_rate)
                n.neuron_health_decay = nd.get('neuron_health_decay', params.neuron_health_decay)
                n.energy_baseline = nd.get('energy_baseline', params.energy_baseline)
                n.firing_energy_cost = nd.get('firing_energy_cost', params.firing_energy_cost)
                n.plasticity_energy_cost = nd.get('plasticity_energy_cost', params.plasticity_energy_cost)
                n.metabolic_rate = nd.get('metabolic_rate', params.metabolic_rate)
                n.recovery_rate = nd.get('recovery_rate', params.recovery_rate)
                n.intrinsic_timescale = nd.get('intrinsic_timescale', n.membrane_time_constant)
                n.adaptation = nd.get('adaptation', 0.0)
                n.autoreceptor = nd.get('autoreceptor', 0.0)
                # ADD: Restore last firing time
                n.last_firing_time = nd.get('last_firing_time', -1000.0)
                
                # ADD: Restore state_history
                if 'state_history' in nd:
                    n.state_history = deque(nd['state_history'], maxlen=50)
                
                # Dendritic branches
                if 'dendritic_branches' in nd:
                    for i, bd in enumerate(nd['dendritic_branches']):
                        if i < len(n.dendritic_branches):
                            b = n.dendritic_branches[i]
                            b.branch_potential = bd['branch_potential']
                            b.plateau_potential = bd['plateau_potential']
                            b.branch_threshold = bd.get('branch_threshold', params.branch_threshold)
                            b.plateau_decay = bd.get('plateau_decay', params.plateau_decay)
                            # ADD: Restore local_spike_history
                            if 'local_spike_history' in bd:
                                b.local_spike_history = deque(bd['local_spike_history'], maxlen=10)
                            
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
        
        # Individualized synapse parameters
        s.tau_fast = sd.get('tau_fast', params.tau_fast)
        s.tau_slow = sd.get('tau_slow', params.tau_slow)
        s.tau_meta = sd.get('tau_meta', params.tau_meta)
        s.tau_ltp = sd.get('tau_ltp', params.tau_ltp)
        s.tau_ltd = sd.get('tau_ltd', params.tau_ltd)
        s.learning_rate = sd.get('learning_rate', params.learning_rate)
        s.plasticity_threshold = sd.get('plasticity_threshold', params.plasticity_threshold)
        
        # ADD: Restore synaptic traces
        s.pre_trace = sd.get('pre_trace', 0.0)
        s.post_trace = sd.get('post_trace', 0.0)
        s.pre_trace_ltd = sd.get('pre_trace_ltd', 0.0)
         # ADD: Restore associative strength
        s.associative_strength = sd.get('associative_strength', 0.0)
        # Temporarily store neighbor IDs for second pass
        s._saved_neighbor_ids = sd.get('neighbor_synapse_ids', None)
        net.synapses.append(s)
    # Build a lookup dictionary for fast synapse retrieval (Updated Save states in v 2.1)
    synapse_lookup = {(s.pre_id, s.post_id): s for s in net.synapses}
    
    # Second pass: Restore neighbor_synapses relationships (Updated Save states in v 2.1)
    for s in net.synapses:
        if hasattr(s, '_saved_neighbor_ids') and s._saved_neighbor_ids is not None:
            # Restore from saved IDs
            s.neighbor_synapses = []
            for (pre_id, post_id) in s._saved_neighbor_ids:
                neighbor = synapse_lookup.get((pre_id, post_id))
                if neighbor is not None:
                    s.neighbor_synapses.append(neighbor)
            delattr(s, '_saved_neighbor_ids')  # Clean up temporary attribute
        else:
            # Fallback: Rebuild from scratch (backward compatibility)
            s.neighbor_synapses = [ns for ns in net.synapses 
                                   if ns.pre_id == s.pre_id and ns.post_id != s.post_id]
        
    # Restore the rest of the global network state.
    net.neuromodulators = d['neuromodulators']
    net.time = d['time']
    net.step_count = d['step_count']
    net.total_energy_consumed = d.get('energy_consumed', 0.0)
    net.branching_ratio = d.get('branching_ratio', 1.0)
    
    # ADD: Restore activation_history v2.1
    if 'activation_history' in d:
        net.activation_history = deque(d['activation_history'], maxlen=1000)
     # ADD: Restore modulator diffusion grid v2.1
    if 'modulator_grid' in d:
        net.modulator_grid = np.array(d['modulator_grid'])
    
    # ADD: Restore oscillator phase offsets v2.1
    if 'oscillator_phase_offsets' in d:
        net.oscillator_phase_offsets = tuple(d['oscillator_phase_offsets'])
    
    # ADD: Restore ITU circles v2.1
    if 'itu_circles' in d and isinstance(d['itu_circles'], list):
        for i, circle_data in enumerate(d['itu_circles']):
            if i < len(net.itu_circles):
                if isinstance(circle_data, dict):
                    net.itu_circles[i].fitness_history = circle_data.get('fitness_history', [])
                    net.itu_circles[i].mutation_rate = circle_data.get('mutation_rate', 0.01)

    # === NEURAXON v2.0: Restore subsystem state ===
    data = d
    if 'neuromodulator_system' in data:
        nm = data['neuromodulator_system']
        if 'levels' in nm:
            for k in net.neuromod_system.levels:
                if k in nm['levels']:
                    net.neuromod_system.levels[k] = nm['levels'][k]
        if 'receptors' in nm:
            for rname, rd in nm['receptors'].items():
                if rname in net.neuromod_system.receptors and isinstance(rd, dict):
                    try:
                        net.neuromod_system.receptors[rname].activation = float(rd.get('activation', 0.0))
                    except Exception:
                        pass
    if 'oscillator_bank' in data:
        osc = data['oscillator_bank']
        if isinstance(osc, dict):
            if 'coupling' in osc:
                net.oscillator_bank.coupling = float(osc['coupling'])
            if 'bands' in osc:
                net.oscillator_bank.bands = osc['bands']

    for syn in net.synapses:
        for sd in data.get('synapses', []):
            if sd.get('pre_id') == syn.pre_id and sd.get('post_id') == syn.post_id:
                for attr in ['chrono_fast_trace', 'chrono_slow_trace', 'chrono_omega',
                             'eligibility', 'branch_id', 'branch_index']:
                    if attr in sd:
                        setattr(syn, attr, sd[attr])
                break

    for n in net.all_neurons:
        for group_name in ['input', 'hidden', 'output']:
            for nd in data.get('neurons', {}).get(group_name, []):
                if nd.get('id') == n.id:
                    for attr in ['complement_h', 'state_tilde', 'dsn_alpha',
                                 'ctsn_phi_gain', 'ctsn_phi_bias',
                                 'firing_rate_avg', 'astrocyte_state']:
                        if attr in nd:
                            setattr(n, attr, nd[attr])
                    if 'dsn_input_buffer' in nd and isinstance(nd['dsn_input_buffer'], list):
                        # v4.52: DSN buffer is now a deque(maxlen=k); preserve on load.
                        k = max(1, len(nd['dsn_input_buffer']))
                        n.dsn_input_buffer = deque((float(x) for x in nd['dsn_input_buffer']), maxlen=k)
                        n._dsn_k = k
                    if 'dsn_kernel_weights' in nd and isinstance(nd['dsn_kernel_weights'], list):
                        n.dsn_kernel_weights = [float(x) for x in nd['dsn_kernel_weights']]
                    if 'msth' in nd and isinstance(nd['msth'], dict):
                        md = nd['msth']
                        n.msth.ultrafast_activity = md.get('ultrafast_activity', 0.0)
                        n.msth.fast_excitability = md.get('fast_excitability', 0.0)
                        n.msth.medium_gain = md.get('medium_gain', 1.0)
                        n.msth.slow_structural = md.get('slow_structural', 0.0)
                    break

    if hasattr(net, '_assign_branch_positions'):
        net._assign_branch_positions()

    return net
