# Neuraxon Game of Life v.4.53 config (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 145
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import random
from dataclasses import dataclass, field
from typing import Set, Dict, Optional, List, Set, Tuple

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

# Constants for terrain types.
T_SEA = 0
T_LAND = 1
T_ROCK = 2

# A list of colors reserved for UI or special objects to ensure agent colors are distinct.
RESERVED_COLORS = [(20, 120, 255), (40, 180, 60), (130, 130, 130), (220, 40, 40)]

# ============================================================================
# SESSION AND NAMING HELPERS
# ============================================================================

def _generate_session_id() -> str:
    """Generate a 10-digit random session ID."""
    return "".join([str(random.randint(0, 9)) for _ in range(10)])

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

# ============================================================================
# GLOBAL SESSION VARIABLES
# ============================================================================

_session_id: Optional[str] = None
_global_name_counter: int = 0  # Tracks the next available name index (A=0, B=1, ..., Z=25, AA=26, etc.)
_used_names: Set[str] = set()  # All names ever used in this session
_clan_history: Dict[int, Dict] = {}  # clan_id -> {members: set, merged_from: list, created_at_round: int}
_next_clan_id: int = 1
_current_round: int = 1
_game_id: str = ""
_next_nxer_id: int = 0  # Globally unique NxEr ID counter

def _get_next_global_name() -> str:
    """Get the next unique uppercase name (A, B, ..., Z, AA, AB, ...)."""
    global _global_name_counter, _used_names
    name = _base26_name(_global_name_counter)
    _global_name_counter += 1
    _used_names.add(name)
    return name

def _reset_session_globals():
    """Reset all session globals for a fresh start."""
    global _session_id, _global_name_counter, _used_names, _clan_history, _next_clan_id, _current_round, _game_id, _next_nxer_id
    _session_id = _generate_session_id()
    _global_name_counter = 0
    _used_names = set()
    _clan_history = {}
    _next_clan_id = 1
    _current_round = 1
    _game_id = "".join([str(random.randint(0, 9)) for _ in range(9)])
    _next_nxer_id = 0

# ============================================================================
# NETWORK PARAMETERS
# ============================================================================

# ============================================================================
# CIRCADIAN / DAY-NIGHT CYCLE CONSTANTS (NEW v3.0)
# ============================================================================
# BIOINSPIRED: Circadian rhythms are fundamental to all biological life.
# The suprachiasmatic nucleus (SCN) orchestrates 24-hour cycles affecting:
# - Melatonin/serotonin balance (sleep-wake regulation)
# - Dopamine release patterns (reward sensitivity varies by time of day)
# - Metabolic rate (lower at night, higher during day)
# - Core body temperature (drops at night, rises during day)

CIRCADIAN_CYCLE_TICKS = 600  # Default: 10 minutes real-time = 1 day cycle at 60 tps
DAWN_PHASE = 0.0              # 0.0 = start of day
NOON_PHASE = 0.25             # Peak activity
DUSK_PHASE = 0.5              # Transition to night
MIDNIGHT_PHASE = 0.75         # Deepest night

# Temperature constants (in abstract units, ~35-40 maps to biological 35-40°C)
TEMP_BASELINE = 37.0          # Normal body temperature
TEMP_MIN = 34.0               # Hypothermia threshold
TEMP_MAX = 41.0               # Hyperthermia threshold
TEMP_NIGHT_DROP = 1.5         # Temperature drops at night (circadian)
TEMP_ACTIVITY_GAIN = 0.3      # Heat from movement/activity
TEMP_SOCIAL_GAIN = 0.2        # Heat from proximity to others
TEMP_FOOD_GAIN = 0.5          # Thermogenic effect of eating
TEMP_DECAY_RATE = 0.02        # Return to baseline rate

# Proprioceptron constants
ROCK_HIT_MEMORY = 5           # How many recent rock hits to track
ROCK_HIT_THRESHOLD = 3        # Hits before forced direction change

# v3.2: Resting metabolism constants
# v3.3: Kept but referenced from NxEr attribute for inheritance
RESTING_METABOLISM_MULTIPLIER_DEFAULT = 0.3
RESTING_TEMP_DROP_RATE = 0.08  # v3.3: Slightly faster drop when resting

# v3.3: Temperature dynamics rebalance — data showed temp stuck at ~37.74
# with near-zero circadian correlation. Gains must exceed decay for variance.
TEMP_ACTIVITY_GAIN_V32 = 0.8   # v3.31: Kept from v3.3
TEMP_FOOD_GAIN_V32 = 1.0       # v3.31: Kept from v3.3
TEMP_DECAY_RATE_V32 = 0.008    # v3.31: Kept from v3.3

# v3.31: RESTING_METABOLISM_MULT alias for backward compat (game_loop uses this)
RESTING_METABOLISM_MULT = 0.3

# v3.31: Resting metabolism constants (BIOINSPIRED: torpor/sleep energy conservation)
RESTING_METABOLISM_MULTIPLIER = 0.3
RESTING_TEMP_DROP_RATE = 0.1  # Body temp drops slowly when resting
RESTING_FOOD_THRESHOLD = 0.2  # Min food fraction to enter voluntary rest

# v3.31: FIX - Temperature constants rebalanced further from results-97 analysis
# Results-97: temp_circadian_corr STILL ≈0, circadian_phase vs body_temp r=-0.106
# Root cause: night drop and day warming still too weak relative to decay
TEMP_ACTIVITY_GAIN = 0.8       # v3.3: Up from 0.5 — must outpace decay
TEMP_SOCIAL_GAIN = 0.4         # v3.3: Up from 0.3
TEMP_FOOD_GAIN = 1.0           # v3.3: Up from 0.7
TEMP_DECAY_RATE = 0.008        # v3.3: Down from 0.015 — slower homeostasis
TEMP_BASELINE_VARIANCE = 1.0   # v3.3: Up from 0.5 — more individual variation
TEMP_NIGHT_DROP = 4.0          # v3.31: Up from 2.5 — much stronger circadian drop
TEMP_DAY_WARMING = 1.5         # v3.31: NEW — explicit day warming constant

# v3.32: Temporal correlation rolling window (ticks) for temperature–circadian correlation
TEMP_CIRCADIAN_CORR_WINDOW = 50  # ~8% of a 600-tick circadian cycle — enough phase variation
# v3.32: Silencing threshold — active synapses with activity below this AND low integrity can become silent
SYNAPSE_SILENCING_ACTIVITY_THRESHOLD = 0.005  # Below this total |w_fast|+|w_slow|, eligible for silencing

@dataclass
class NetworkParameters:
    """
    A dataclass holding all configurable parameters for the Neuraxon network and the simulation environment.
    """
    # --- General Network Architecture ---
    network_name: str = "Neuraxon NxEr"
    # UPDATED v4.5: 6 original + DayNight, Temperature, Proprioception, Song = 10 inputs
    # Input neurons (trinary -1/0/1):
    #   0: Movement result (-1=blocked, 0=moved/idle, 1=food found)
    #   1: Terrain encounter (-1=rock/threat, 0=none, 1=salient encounter)
    #   2: Terrain constraint (-1=blocked/unsafe substrate, 0=traversable context, 1=unused spare channel)
    #   3: Hunger (-1=starving, 0=normal, 1=satiated) 
    #   4: Sight (-1=aversive/threatening cue, 0=no salient cue, 1=food/reward cue)
    #   5: Smell (-1=aversive cue, 0=no salient cue, 1=food/reward cue)
    #   6: DayNight (-1=night drive, 0=background/transition, 1=day drive)
    #   7: Temperature (-1=cold/hypothermic, 0=normal, 1=hot/hyperthermic)
    #   8: Proprioception (-1=repeatedly blocked, 0=normal, 1=clear path history)
    #   9: Song/Hearing (-1=dissonant song nearby, 0=no song, 1=harmonic song nearby)  [NEW v4.5]
    num_input_neurons: int = 10
    num_hidden_neurons: int = 10 
    # UPDATED v4.5: 5 original + Resting + Sing = 7 outputs
    # Output neurons (trinary -1/0/1):
    #   0: Move X (-1=west, 0=stay, 1=east)
    #   1: Move Y (-1=north, 0=stay, 1=south)
    #   2: Social (-1=aggressive/steal, 0=neutral, 1=friendly)
    #   3: Mate intent (-1=reject, 0=neutral, 1=seek mate)
    #   4: Give food (-1=take, 0=neutral, 1=give to clan)
    #   5: Resting (-1=force active/wake, 0=normal, 1=rest/sleep)
    #   6: Sing (-1=silent, 0=hum softly, 1=sing full voice)  [NEW v4.5]
    num_output_neurons: int = 7
    
    # v3.2: Resting/Metabolism parameters
    resting_metabolism_multiplier: float = 0.3
    resting_temp_drop_rate: float = 0.1
    temp_activity_gain_v32: float = 0.5
    temp_food_gain_v32: float = 0.7
    temp_decay_rate_v32: float = 0.015
    
    # --- Temporal Dynamics & Simulation ---
    dt: float = 1.0 
    min_dt: float = 0.1 
    max_dt: float = 2.0  
    activity_threshold: float = 0.5 
    simulation_steps: int = 30 

    # --- Network Topology ---
    connection_probability: float = 0.15 
    small_world_k: int = 6  
    small_world_rewire_prob: float = 0.15 
    preferential_attachment: bool = True 
    
    # --- Core Neuron Properties (Neuraxon) ---
    membrane_time_constant: float = 20.0 
    # v2.92: REBALANCED thresholds for bio-inspired trinary distribution
    # BIOINSPIRED: Target ~20-25% excitatory, 30-40% neutral, 35-45% inhibitory
    # Paper claim: neutral state acts as "buffer enabling rapid transitions"
    # v129 FIX (M6): Reverted toward v2.91 value. v2.92 lowered to 0.45 for
    # "easier excitation" but this produced 75% excitatory (target: 20-25%).
    # The neutral zone [-0.40, 0.45] = 0.85 was too narrow — most membrane
    # potentials exceeded 0.45 due to oscillator drive + spontaneous activity.
    # Widened to [-0.50, 0.60] = 1.10 (+29%) to restore neutral dominance.
    # v132 FIX (M6): Aligned with v130's _random_params range [0.45, 0.65].
    # v131's [0.55, 0.80] push regressed M12. Reverting to v130 threshold range.
    firing_threshold_excitatory: float = 0.55  # v132: midpoint of [0.45, 0.65]
    firing_threshold_inhibitory: float = -0.50  # v132: midpoint of [-0.60, -0.40]
    adaptation_rate: float = 0.08  # v2.36: Raised from 0.02 - stronger spike-frequency adaptation
    # v2.92: Moderate spontaneous rate for balanced trinary exploration
    # BIOINSPIRED: Cortical spontaneous activity explores state space
    # Paper claim: "spontaneous firing sustains intrinsic activity"
    # v129 FIX (M6): Reduced from 0.006 to 0.003 — less tonic excitatory push.
    # With 0.006, spontaneous spikes contributed ~0.6% of state transitions per tick,
    # but cumulatively biased the membrane positive (excitatory). Halving this
    # reduces the tonic excitatory floor without eliminating state exploration.
    spontaneous_firing_rate: float = 0.003  # v129: Down from 0.006
    neuron_health_decay: float = 0.001 
    
    # --- Membrane Potential Dynamics (NEW v2.36) ---
    # v132 FIX (M6): KEY NEW LEVER — faster membrane return to neutral.
    # v130 got E from 75%→54% via threshold. v131 tried higher threshold but
    # regressed M12. Different approach: reduce TIME each neuron spends above
    # threshold by making the membrane decay to resting potential FASTER.
    # At 0.10 (v129), membrane retains 90% of its value per tick.
    # At 0.20, membrane retains only 80% — halving effective excursion duration.
    # This reduces E without changing the threshold or input sensitivity.
    # BIOINSPIRED: Na+/K+-ATPase pump activity; faster restoration = shorter spikes.
    resting_potential_decay: float = 0.20  # v132: Up from 0.10 — faster neutral return
    
    # v3.34 RC1-FIX: Set to 0.0 — the constant negative bias was the primary driver of
    # output locking into inhibitory state (97.5% SW quadrant, ~40% outputs at -1).
    # BIOINSPIRED: Biological resting potential is maintained by Na+/K+-ATPase pump
    # equilibrium, not by a tonic inhibitory current. The resting_potential_decay
    # parameter already models passive return to resting potential via leak channels.
    membrane_negative_bias: float = 0.0  # v3.34: Was -0.06; removed — see RC1 diagnostics
    
    # --- Dendritic Branch Properties ---
    num_dendritic_branches: int = 3 
    branch_threshold: float = 0.72  # v2.36: Raised - dendritic spikes require more accumulated input 
    plateau_decay: float = 500.0 

    # --- Synaptic Properties & Plasticity (Section 3) ---
    tau_fast: float = 5.0  
    tau_slow: float = 50.0  
    tau_meta: float = 150.0   # v3.31: Down from 200 — faster meta tracking (was 1000 in v3.2)
    tau_ltp: float = 15.0 
    tau_ltd: float = 35.0 
    
    # v3.31: Meta-plasticity dynamics (Paper Section 3 multi-timescale)
    # BIOINSPIRED: Metabotropic receptors integrate over seconds-minutes, not hours
    meta_target_gain: float = 0.30        # v3.31: Up from 0.25 — stronger meta signal
    meta_accumulation_rate: float = 0.35  # v3.31: Up from 0.3 — more w_slow→meta flow
    meta_clamp_max: float = 1.0           # v3.31: Up from 0.8 — full [-1,1] range
    
    # v3.31: NEW — Meta-behavior coupling (CRITICAL FIX)
    # Results-97: w_meta not in compute_input → zero behavioral effect for 80% of synapses
    meta_influence_gain: float = 0.25     # v3.31: How much w_meta contributes to effective weight
    meta_da_boost: float = 2.0            # v3.31: DA multiplier for meta accumulation during reward
    
    # --- Synaptic Weight Initialization Ranges ---
    w_fast_init_min: float = -1.0
    w_fast_init_max: float = 1.0
    w_slow_init_min: float = -0.5
    w_slow_init_max: float = 0.5
    w_meta_init_min: float = -0.3
    w_meta_init_max: float = 0.3
    
    # v3.31: LTP/LTD rebalance — Results-97 overcorrected to 70.6% LTP, target ~55-60%
    # v112 FIX (M25+M7+M1): hebbian_ltp_rate 0.02→0.08.
    # v111 AUTOPSY: 0.02 was too aggressive — with D1²≈0.03 at operating DA,
    # ALL learning (LTP+LTD) was suppressed to ~3%, killing M1 (C→F),
    # M7 (A→D), M14 (A→B). The DA-gated component must DOMINATE but
    # a moderate Hebbian floor (0.08) ensures baseline connectivity forms.
    # At D1≈0.49 (baseline DA), ratio is DA:Hebbian = 6:1 → DA dominates.
    # At D1≈0.77 (burst DA), ratio = 10:1 → strong differential.
    # Paper §4: "coincident firing" drives basic Hebbian; DA modulates magnitude.
    hebbian_ltp_rate: float = 0.12        # v113: Raise to compensate lower average D1 activation.
                                           # With D1 EC50=0.45, quiet D1≈0.23 vs old D1≈0.97.
                                           # Hebbian floor keeps baseline learning alive.
    ltd_neutral_scale: float = 0.12       # v3.31: Kept
    ltd_inhibitory_scale: float = 0.6     # v3.31: Kept from v3.3
    # v3.31: Slow weight differentiation
    w_slow_post_trace_fraction: float = 0.85 # v3.31: Up from 0.8 — more post-trace in w_slow
    w_fast_delta_share: float = 0.5       # v3.31: NEW — fraction of delta_w going to w_fast
    w_slow_delta_share: float = 0.02      # v3.31: NEW — fraction of delta_w going to w_slow (was 0.03)
    
    # --- Learning and Plasticity Rules (Section 4) ---
    learning_rate: float = 0.01 
    stdp_window: float = 20.0 
    learning_rate_mod: float = 1.0 
    plasticity_threshold: float = 0.5 
    associativity_strength: float = 0.1 
    
    # --- Structural Plasticity ---
    synapse_integrity_threshold: float = 0.1 
    synapse_formation_prob: float = 0.02 
    synapse_death_prob: float = 0.01 
    neuron_death_threshold: float = 0.1 
    
    # --- Neuromodulation (Section 1 & 8) ---
    dopamine_baseline: float = 0.15
    dopamine_high_affinity_threshold: float = 0.01
    dopamine_low_affinity_threshold: float = 0.25   
    serotonin_baseline: float = 0.12
    serotonin_high_affinity_threshold: float = 0.01
    serotonin_low_affinity_threshold: float = 1.0
    acetylcholine_baseline: float = 0.12
    acetylcholine_high_affinity_threshold: float = 0.01
    acetylcholine_low_affinity_threshold: float = 1.0
    norepinephrine_baseline: float = 0.12
    norepinephrine_high_affinity_threshold: float = 0.01
    norepinephrine_low_affinity_threshold: float = 1.0
    # v109: Reduced from 0.6 to 0.15. With the /10.0 divisor removed,
    # 0.6 would cause instantaneous snap-to-baseline (too aggressive).
    # v110 FIX (M18): Increased from 0.15 to 0.25 to match the stronger
    # sync-backed decay. With bidirectional dict↔v2.0 sync, decay now
    # persists across ticks and 0.25 gives ~4 tick time constant.
    neuromod_decay_rate: float = 0.25
    
    # v108 NEW: Concentration-dependent excess decay rate
    # BIOINSPIRED: At high extracellular concentrations, enzymatic degradation
    # and reuptake both increase. Models the non-linear clearance seen in vivo
    # when monoamine overflow exceeds transporter capacity.
    neuromod_excess_decay_rate: float = 0.15  # Additional decay per unit above 3× baseline
    neuromod_excess_threshold_mult: float = 3.0  # Trigger excess decay at 3× baseline
    
    # v107 FIX (M02): Dopamine negative prediction error on obstacle collision
    # BIOINSPIRED: DA encodes reward prediction errors (Schultz 1997).
    # Unexpected obstacle = negative prediction error = DA dip.
    # Paper: "Dopamine – triggered by... prediction errors"
    collision_da_punishment: float = 0.15       # DA reduction on rock collision
    collision_ne_boost: float = 0.08            # NE surprise/startle on collision
    proprioception_lead_ticks: int = 1          # Ticks of warning before forced turn
    # v3.33: Enzymatic clearance (MAO/COMT analog) — Michaelis-Menten per-transporter
    # BIOINSPIRED: Each monoamine has a distinct reuptake transporter with different kinetics:
    #   NET (norepinephrine transporter): fast clearance
    #   DAT (dopamine transporter): moderate
    #   SERT (serotonin transporter): slow (SSRIs block this)
    #   AChE (acetylcholinesterase): fastest enzymatic degradation
    # v108 FIX: Increased Vmax across the board to counterbalance circadian injection
    reuptake_vmax_ne: float = 0.15
    reuptake_vmax_da: float = 0.20        # v113 FIX (M25): 0.10→0.20. Dopamine transporter
                                           # (DAT) has the highest Vmax of all monoamine
                                           # transporters. Doubling clearance rate pulls
                                           # DA mean down faster after behavioral boosts.
                                           # BIOINSPIRED: Striatal DA half-life ~200ms
                                           # vs 5-HT ~1-2s (Cragg & Rice 2004).
    # v107 FIX (M04): Increased from 0.03 to match NET kinetics.
    # Old value caused chronic 5-HT saturation (~1.97) destroying all modulatory range.
    # BIOINSPIRED: SERT reuptake is slower than NET but not 3x slower.
    # Paper says 5-HT modulates risk/social behavior — needs dynamic range.
    reuptake_vmax_5ht: float = 0.12
    reuptake_vmax_ach: float = 0.15
    reuptake_km: float = 0.5  # Half-saturation constant (shared, Michaelis-Menten)
    # v3.33: Autoreceptor negative feedback strength
    # BIOINSPIRED: Presynaptic autoreceptors (5-HT1A, α2-adrenergic, D2-short)
    # detect high extracellular concentration and suppress further vesicle release
    autoreceptor_strength: float = 1.0  # Scaling exponent for feedback (1.0 = quadratic)
    diffusion_rate: float = 0.05
    dopamine_reward_magnitude: float = 0.10   # v113 FIX (M25): 0.25→0.10. With surprise×2.0,
                                               # old peak was +0.50 — catapulting DA to ceiling.
                                               # 0.10 × 2.0 = 0.20 peak is a genuine phasic burst
                                               # without pushing DA permanently above D1 EC50.
                                               # BIOINSPIRED: VTA phasic bursts are transient
                                               # (Schultz 1997), not sustained step functions. 
    
    # --- Oscillators & Synchronization (Section 7) ---
    oscillator_low_freq: float = 0.05  
    oscillator_mid_freq: float = 0.5   
    oscillator_high_freq: float = 4.0  
    oscillator_strength: float = 0.10  # v2.36: Reduced from 0.25 - less tonic drive, more stimulus-dependent 
    
    # --- Energy Metabolism ---
    energy_baseline: float = 100.0 
    firing_energy_cost: float = 5.0  
    
    # --- Synaptic Weight Homeostasis (NEW in v2.31) ---
    # BIOINSPIRED: Models synaptic scaling (Turrigiano & Nelson, 2004)
    # Biological neurons maintain stable total synaptic drive through
    # multiplicative scaling that preserves relative weight differences
    weight_homeostasis_enabled: bool = True  # Enable synaptic scaling
    weight_homeostasis_interval: int = 50  # Apply scaling every N steps
    weight_saturation_threshold: float = 0.75  # Mean |weight| triggering scaling
    weight_homeostasis_target: float = 0.5  # Target mean |weight| after scaling
    weight_homeostasis_rate: float = 0.02  # How fast to approach target (soft scaling)
    weight_drift_correction: float = 0.005  # Per-step drift toward zero for Δw balance
    max_weight_magnitude: float = 0.95  # Hard ceiling to prevent saturation
    min_weight_magnitude: float = -0.95  # Hard floor to prevent saturation
    ltp_ltd_balance_target: float = 1.0  # Target LTP/LTD ratio
    ltp_ltd_correction_rate: float = 0.01  # Rate of correcting LTP/LTD imbalance
    plasticity_energy_cost: float = 10.0 
    metabolic_rate: float = 1.0 
    
    # --- Circadian Rhythm Parameters (NEW v3.0) ---
    # BIOINSPIRED: Configurable day/night cycle
    circadian_cycle_ticks: int = 2400  # Ticks per full day/night cycle
    circadian_enabled: bool = True     # Toggle circadian effects
    
    # Neuromodulator circadian modulation strengths
    circadian_serotonin_amplitude: float = 0.3   # Night: +serotonin (sleep)
    circadian_dopamine_amplitude: float = 0.25  # Day: +dopamine (activity)
    circadian_norepinephrine_amplitude: float = 0.2  # Day: +NE (alertness)
    circadian_acetylcholine_amplitude: float = 0.15  # Day: +ACh (attention)
    
    # Metabolic circadian effects
    circadian_metabolic_day_multiplier: float = 1.2   # Higher metabolism during day
    circadian_metabolic_night_multiplier: float = 0.7 # Lower metabolism at night
    
    # --- Temperature Parameters (NEW v3.0) ---
    # BIOINSPIRED: Body temperature affects enzyme kinetics, neural firing rates
    temperature_enabled: bool = True
    temperature_baseline: float = 37.0
    temperature_min: float = 34.0
    temperature_max: float = 41.0
    temperature_neural_sensitivity: float = 0.1  # How much temp affects firing
    temperature_metabolic_q10: float = 2.0  # Q10 coefficient for metabolism
    
    # --- Proprioceptron Parameters (UPDATED v3.1) ---
    proprioceptron_rock_memory: int = 5
    proprioceptron_force_turn_threshold: int = 3
    proprioceptron_clear_path_threshold: int = 5  # Consecutive successful moves for "clear path" signal
    
    # --- Brain-Instinct Balance (NEW v3.1) ---
    # BIOINSPIRED: Balance between learned brain outputs and instinctive survival behaviors
    brain_movement_base_weight: float = 0.7  # How much brain controls movement vs instinct
    brain_rest_override_threshold: float = 0.3  # Food level below which rest is overridden
    circadian_rest_tendency: float = 0.7  # Base probability of resting during night phase
    
    # --- Temperature Behavioral Thresholds (NEW v3.1) ---
    temp_cold_threshold: float = 35.5  # Below this = hypothermic signal
    temp_hot_threshold: float = 38.5   # Above this = hyperthermic signal
    temp_movement_bonus: float = 0.15  # Extra movement when cold (to generate heat)
    

    # =================================================================
    # NEURAXON V2.0 PARAMETERS (Vivancos & Sanchez, 2026)
    # =================================================================

    # --- DSN Dynamic Decay (Algorithm 1, Eq 4-5) ---
    dsn_kernel_size: int = 4
    dsn_enabled: bool = True
    dsn_bias: float = 0.0
    dsn_kernel_weights: Optional[List[float]] = None
    dsn_learn_enabled: bool = False
    dsn_learn_lr: float = 0.01
    dsn_target_sensitivity: float = 4.0
    dsn_target_bias: float = 2.0
    dsn_kernel_clip: float = 5.0

    # --- CTSN Complement (Algorithm 1, Eq 2) ---
    ctsn_rho: float = 0.9
    ctsn_enabled: bool = True
    ctsn_phi_gain: float = 0.5
    ctsn_phi_bias: float = 0.0
    ctsn_learn_enabled: bool = False
    ctsn_learn_lr: float = 0.005
    ctsn_phi_gain_clip: float = 5.0
    ctsn_phi_bias_clip: float = 5.0

    # --- ChronoPlasticity / Synaptic Time Warping (Algorithm 1, Eqs 5-7) ---
    chrono_alpha_f: float = 0.95
    chrono_alpha_s: float = 0.99
    chrono_lambda_f: float = 0.15
    chrono_lambda_s: float = 0.08
    chrono_enabled: bool = True
    chrono_trace_clip: float = 6.0
    chrono_gate_norm: float = 10.0
    chrono_raw_clip: float = 8.0
    chrono_omega_min: float = 0.05
    chrono_omega_max: float = 0.95
    chrono_omega_smoothing: float = 0.2
    chrono_plastic_enabled: bool = True

    # --- AGMP Astrocyte-Gated Plasticity (Eqs 8-10) ---
    agmp_lambda_e: float = 0.9
    agmp_lambda_a: float = 0.95
    agmp_eta: float = 0.005
    agmp_enabled: bool = True

    # --- MSTH: Multi-Scale Temporal Homeostasis (Section 5) ---
    msth_ultrafast_tau: float = 2.5
    msth_ultrafast_ceiling: float = 2.0
    msth_fast_tau: float = 2000.0
    msth_fast_gain: float = 0.1
    msth_medium_tau: float = 300000.0
    msth_medium_gain: float = 0.001
    msth_slow_tau: float = 3600000.0
    msth_slow_gain: float = 0.0001

    # v112 FIX (M25): DA-specific phasic release multiplier.
    # v111 AUTOPSY: tau_phasic=50 correctly made phasic transients short,
    # but without increasing injection magnitude, DA barely varied
    # (mean=0.13, 90%-range=0.085). D1 sat at 0.15 constantly → no gating.
    # With 4× DA phasic release: active periods push DA to ~0.30-0.45,
    # creating genuine D1 swing (0.38→0.92). Other modulators unchanged.
    # BIOINSPIRED: DA release in VTA is 3-5× higher per spike than
    # cortical glutamate release (Garris et al. 1994), reflecting
    # DA's role as a salience/reward signal, not a basal transmitter.
    da_phasic_release_multiplier: float = 2.0  # v113 FIX (M25): 4.0→2.0. The 4× was compensating
                                                # for D1 EC50=0.18 being below the DA range.
                                                # With EC50=0.45 properly centered, 2× provides
                                                # enough phasic swing (DA 0.15→0.35 during bursts)
                                                # to cross D1's switching zone.
    # v117: Stronger DA→ACh antagonism to match Algorithm 5 and prevent reward periods
    # from drifting into an exploratory/high-ACh state. This is architecture-level
    # crosstalk, not a game-scripted behavior rule.
    da_ach_crosstalk_strength: float = 0.35
    da_ach_tonic_suppression: float = 0.08

    # --- Receptor Subtypes (Algorithm 2/5) ---
    # Slopes control Hill-sigmoid steepness (tonic=high-affinity, phasic=low-affinity)
    receptor_concentration_cap: float = 1.0
    # v111 FIX (M25): receptor_slope_tonic 4→12, phasic 3→10.
    # Paper §1 Neuromodulation describes high-affinity (nM) tonic receptors
    # versus low-affinity (μM) phasic receptors. With slopes 3-4, the Hill
    # sigmoid was too shallow: D1 varied only 0.35→0.60 across the full DA
    # range, making D1-gated LTP almost DA-independent.
    # At slope 12/10, D1 switches sharply around its EC50 (0.35):
    #   DA=0.15 → D1≈0.05 (LTP suppressed)
    #   DA=0.50 → D1≈0.82 (LTP enabled)
    # D2 (tonic, EC50=0.25) also sharpens:
    #   DA_tonic=0.10 → D2≈0.10 (LTD suppressed)
    #   DA_tonic=0.30 → D2≈0.85 (LTD enabled)
    # This creates the phasic/tonic separation the paper requires.
    receptor_slope_tonic: float = 12.0
    receptor_slope_phasic: float = 6.0    # v113 FIX (M25): 10.0→6.0. Steep slope compressed
                                           # D1's useful range to a narrow DA window. With slope=6:
                                           #   DA=0.20: D1=0.18 (LTP suppressed)
                                           #   DA=0.35: D1=0.35 (moderate LTP)
                                           #   DA=0.55: D1=0.65 (strong LTP)
                                           #   DA=0.70: D1=0.82 (peak LTP)
                                           # This spreads D1 across the full DA operating range.
    # v110 FIX (M18): tau_tonic 5000→200. With tau=5000, internal tonic decay
    # was dt/5000 = 0.0002/tick — 60× weaker than activity injection.
    # Tonic levels rose monotonically to ceiling and never returned.
    # tau=200 gives dt/200 = 0.005/tick, matching biological seconds-scale
    # clearance for tonic (volume transmission) monoamine components.
    tau_tonic: float = 200.0
    # v111 FIX (M25): tau_phasic 200→50. Phasic transients must decay
    # faster than tonic baseline to create distinct DA episodes.
    # BIOINSPIRED: Synaptic DA clearance is ~100-400ms (Garris 1994).
    # At dt=1, tau=50 gives ~50-tick phasic transient half-life,
    # creating sharp phasic bursts that differentially activate D1.
    # This directly enables the tonic/phasic separation in §1.
    tau_phasic: float = 50.0
    # v110 FIX (M18): release_rate 0.02→0.005. With bidirectional sync,
    # game-loop behavioral boosts now persist into the v2.0 system (they
    # were previously overwritten). Internal activity-driven release can
    # be reduced to avoid double-counting. Net release rate stays similar
    # because external boosts (circadian, reward, satiety) now stick.
    neuromod_release_rate: float = 0.005

    # --- Oscillator Bank (Multi-band PAC, Section 6) ---
    oscillator_coupling: float = 0.1

    # --- Watts-Strogatz Topology (Algorithm 6) ---
    ws_k: int = 6
    ws_beta: float = 0.3

    # --- Homeostatic Plasticity (Section 4) ---
    homeostatic_rate: float = 0.0005
    firing_rate_alpha: float = 0.01
    threshold_mod_k: float = 0.3

    # --- Aigarth v2.0 Hybrid (Algorithm 7) ---
    aigarth_pop_size: int = 10
    aigarth_itu_size: int = 12
    aigarth_tick_cap: int = 20
    aigarth_mutation_wf_prob: float = 0.3
    aigarth_mutation_ws_prob: float = 0.1
    aigarth_mutation_wm_prob: float = 0.05
    # v117: Encourage circle-level functional niches so ITUs diversify instead of all
    # collapsing onto the same oscillator regime.
    itu_niche_strength: float = 0.30
    itu_target_cohesion: float = 0.20
    itu_freq_mutation_scale: float = 0.06
    itu_timescale_mutation_scale: float = 0.08

    recovery_rate: float = 0.5 
    
    # --- Energy-Aware Threshold Homeostasis (NEW in v2.30) ---
    # BIOINSPIRED: Models ATP depletion effects on ion channel dynamics
    # In biology, low ATP impairs Na+/K+-ATPase, raising effective firing threshold
    energy_threshold_coupling: float = 0.5  # How strongly energy affects threshold (0=none, 1=strong)
    energy_threshold_floor: float = 0.3  # Minimum energy fraction before threshold scaling activates
    energy_recovery_boost: float = 2.0  # Enhanced recovery rate multiplier for low-energy neurons
    critical_energy_level: float = 30.0  # Energy level below which recovery boost activates
    
    # --- Homeostasis ---
    target_firing_rate: float = 0.1 
    homeostatic_plasticity_rate: float = 0.001 
    
    # --- Adaptive Network-Wide Threshold Homeostasis (REBALANCED v2.36) ---
    # BIOINSPIRED: Cortical neurons maintain ~20-25% average firing rate for optimal information
    # processing. The neutral state is the computational "buffer" where integration occurs.
    adaptive_threshold_enabled: bool = True
    # v129 FIX (M6/M27): More frequent checks and stronger correction.
    # M27 showed 88% of homeostatic events are suppressive (up-adjustments)
    # but the 6% correction every 12 ticks couldn't keep up with the excitatory drive.
    # Doubling check frequency and increasing correction strength gives homeostasis
    # a fighting chance to actually enforce the target excitatory fraction.
    adaptive_threshold_check_interval: int = 8  # v129: Down from 12 — check more often
    adaptive_threshold_adjustment: float = 0.10  # v129: Up from 0.06 — stronger correction
    min_excitatory_fraction: float = 0.15  # v2.36: Floor for maintaining some activity
    max_excitatory_fraction: float = 0.28  # v2.37b: Slightly lower
    target_excitatory_fraction: float = 0.22  # v2.36: NEW - optimal target for criticality
    min_inhibitory_fraction: float = 0.10  # v2.37b: NEW - minimum inhibitory for E/I balance
    target_inhibitory_fraction: float = 0.10  # v2.36: NEW - ensures inhibitory presence
    
    # --- Aigarth Hybridization (Section 8) ---
    itu_circle_radius: int = 8 
    evolution_interval: int = 1000 
    fitness_temporal_weight: float = 0.4 
    fitness_energy_weight: float = 0.3
    fitness_pattern_weight: float = 0.3
    
    # --- Phase Synchronization Parameters (v2.40 Kuramoto Model) ---
    phase_coupling_strength: float = 0.08  # REDUCED global coupling
    phase_coupling_local_strength: float = 1.0  # STRONG local Kuramoto coupling  reduced fro 1.5 
    max_axonal_delay: float = 10.0
    phase_clustering_init: float = 0.85  # 85% neurons start clustered
    natural_freq_range_min: float = 0.85  # Narrow frequency range
    natural_freq_range_max: float = 1.15  # Spread of 0.3 (was 1.5)
    phase_coupling_momentum: float = 0.6  # Smooth phase transitions
    
    # --- Sensory-Motor Coupling (NEW in v2.35) ---
    sensory_input_gain: float = 0.9  # v2.36: Reduced - sensory shouldn't overwhelm network
    afferent_synapse_strength: float = 1.1  # v2.36: Reduced from 1.5
    afferent_synapse_reliability: float = 0.95
    sensory_gating_enabled: bool = True
    sensory_gating_threshold: float = 0.45  # v2.36: Raised - stronger gating
    sensory_gating_suppression: float = 0.25  # v2.36: Stronger suppression during driven input
    
    # v107 FIX (M02): Stronger proprioceptive-to-motor pathway 
    # BIOINSPIRED: Proprioceptive afferents have strong, reliable connections
    # to motor neurons (stretch reflex, withdrawal reflex).
    proprioceptive_afferent_gain: float = 1.8   # Boosted vs normal afferent_synapse_strength=1.1
    proprioceptive_stdp_boost: float = 2.5      # STDP multiplier for proprio→motor synapses
    max_intrinsic_timescale: float = 80.0  # Reduced from 100 for stricter bound
    spontaneous_as_current: bool = True
    spontaneous_current_magnitude: float = 1.2  # Reduced from 1.5
    
    # --- Spike Classification Thresholds (UPDATED v2.38) ---
    # Used to determine if a spike was driven vs spontaneous
    # BIOINSPIRED: Even small synaptic inputs should count as "driven"
    # v2.38: Lowered threshold to capture weak but real synaptic drive
    driven_input_threshold: float = 0.05  # v2.38: Reduced from 0.2 for better classification
    spike_classification_enabled: bool = True

    # =================================================================
    # MULTI-SPHERE ARCHITECTURE PARAMETERS (Paper Sections 7-8)
    # =================================================================
    # BIOINSPIRED: Real brains comprise tens to hundreds of modules
    # interconnected by long-range projections. The Multi-Sphere
    # architecture groups multiple NeuraxonNetwork instances into a
    # directed graph with biologically-grounded inter-sphere links.
    
    multisphere_enabled: bool = True       # Enable multi-sphere brain architecture
    multisphere_topology: str = "sensory_association_motor"  # Default 3-sphere hierarchy
    
    # Per-sphere hidden neuron counts (override num_hidden_neurons per sphere)
    ms_sensory_hidden: int = 6             # Sensory sphere hidden neurons
    ms_association_hidden: int = 10        # Association sphere hidden neurons (usually largest)
    ms_motor_hidden: int = 6              # Motor sphere hidden neurons
    
    # Inter-sphere link parameters
    ms_ff_gain: float = 1.0               # Feedforward projection gain
    ms_fb_gain: float = 0.8               # Feedback projection gain
    ms_ff_delay: int = 1                  # Feedforward conduction delay (steps)
    ms_fb_delay: int = 2                  # Feedback conduction delay (steps)
    ms_coherence_strength: float = 0.25   # Phase-coherence gating strength
    ms_projection_plasticity: float = 0.001  # Inter-sphere projection learning rate
    ms_structural_plasticity: bool = True  # Enable link pruning/sprouting
    ms_volume_diffusion_rate: float = 0.1  # Neuromodulator volume transmission rate
    
    # Sphere modularity (for brain surgery / transplant)
    ms_sphere_save_independent: bool = True  # Allow individual sphere save/load
    ms_sphere_swap_enabled: bool = True      # Allow sphere hot-swapping between NxErs

    # =================================================================
    # VOICE / SONG / AUDIO PARAMETERS (NEW v4.5)
    # =================================================================
    # BIOINSPIRED: Tonal vocalisations couple social behaviour to an
    # acoustic dimension. See simulation/voice.py for the consonance
    # model grounded in Plomp–Levelt / Kameoka–Kuriyagawa psychoacoustics.

    voice_enabled: bool = True                 # Global on/off for the voice subsystem
    voice_hearing_radius: int = 4              # Square radius of the "song near me" scan
    voice_max_listeners: int = 6               # Cap nearby-voices list cost per tick
    voice_clan_affinity_weight: float = 0.35   # Tone-similarity bias added to movement
    voice_choir_nearby_threshold: int = 3      # NxErs needed to earn choir serotonin boost
    voice_choir_serotonin_boost: float = 0.02  # Per-tick boost when choir conditions met
    voice_sing_food_threshold: float = 0.20    # Fraction of StartFood below which singing suppresses (sick/starving don't sing)
    voice_food_pitch_semitones: float = 3.0    # Max pitch elevation when fully fed
    voice_discovery_pitch_semitones: float = 2.0  # Transient spike on new food
    voice_discovery_duration_ticks: int = 30   # Length of that transient
    voice_inherit_novel_tones: int = 2         # # of fresh tones per child (rest = 4 inherited)
    voice_mate_tone_exchange: bool = True      # Partners copy 1 tone from each other at mating

    # Audio (rendering only — OFF by default, toggled with M key in-game)
    audio_enabled_default: bool = False
    audio_min_zoom: float = 8.0                # Below this, audio is silent
    audio_max_screen_dist: float = 450.0       # Pixels; fade to zero here
    audio_max_voices: int = 8                  # Hard ceiling on channels
    audio_max_new_per_frame: int = 4           # Throttle for cold-cache inits

    def __post_init__(self):
        """Initialize derived parameters (DSN kernel weights)."""
        k = max(int(self.dsn_kernel_size), 1)
        if not self.dsn_kernel_weights:
            w = [(i + 1.0) for i in range(k)]
        else:
            w = list(self.dsn_kernel_weights)[:k]
            if len(w) < k:
                w = w + [w[-1]] * (k - len(w))
        s = float(sum(abs(x) for x in w)) if w else 1.0
        if s <= 0:
            s = 1.0
        self.dsn_kernel_weights = [float(x) / s for x in w]
