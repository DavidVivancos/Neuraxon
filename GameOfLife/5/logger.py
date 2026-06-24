# Neuraxon Game of Life v.5.06 logger (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 191
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import time
import os
import json
import math
import cmath
import threading
import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Type Checking imports (avoid circular dependency at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation.entities import NxEr
    from neuraxon.network import NeuraxonNetwork

from config import TEMP_CIRCADIAN_CORR_WINDOW

# v145: Research probes — implements the 10-metric paper-fidelity dashboard.
# See neuraxon/research_probes.py for full documentation.
#
# IMPLEMENTATION NOTE:
#   logger.py is imported by neuraxon/neuron.py (for get_data_logger), and
#   neuraxon/__init__.py imports .neuron. Therefore importing
#   neuraxon.research_probes at logger module-load time triggers
#   neuraxon/__init__.py, which triggers neuron.py, which tries to import
#   logger.get_data_logger before logger has finished its top-level
#   execution → circular ImportError.
#
#   Resolution: defer the import to first use via _ensure_research_probes_loaded().
#   This costs one extra dict lookup the first time DataLogger.reset() runs and
#   nothing thereafter.
_RESEARCH_PROBES_AVAILABLE = False
_RESEARCH_PROBES_IMPORT_ERROR: Optional[str] = None
_ResearchProbeState = None
_research_compute_all_metrics = None
_research_all_metric_keys = None
_RESEARCH_HEALTHY_BANDS = {}

# v147 (v4.55) — threaded metrics worker (also lazy-loaded)
_METRICS_WORKER_AVAILABLE = False
_METRICS_WORKER_IMPORT_ERROR: Optional[str] = None
_get_metrics_worker = None
_shutdown_metrics_worker = None
_THREADED_METRICS_DEFAULT = True       # set to False here to disable globally

def _ensure_research_probes_loaded() -> bool:
    """Import the research-probes module and the metrics-worker module on
    first call. Idempotent. Returns True if research probes are available."""
    global _RESEARCH_PROBES_AVAILABLE, _RESEARCH_PROBES_IMPORT_ERROR
    global _ResearchProbeState, _research_compute_all_metrics
    global _research_all_metric_keys, _RESEARCH_HEALTHY_BANDS
    global _METRICS_WORKER_AVAILABLE, _METRICS_WORKER_IMPORT_ERROR
    global _get_metrics_worker, _shutdown_metrics_worker
    if _RESEARCH_PROBES_AVAILABLE:
        return True
    if _RESEARCH_PROBES_IMPORT_ERROR is not None:
        return False
    try:
        import importlib
        mod = importlib.import_module('neuraxon.research_probes')
        _ResearchProbeState = mod.ProbeState
        _research_compute_all_metrics = mod.compute_all_metrics
        _research_all_metric_keys = mod.all_metric_keys
        _RESEARCH_HEALTHY_BANDS = mod.HEALTHY_BANDS
        _RESEARCH_PROBES_AVAILABLE = True
    except BaseException as exc:
        import traceback as _tb
        _RESEARCH_PROBES_IMPORT_ERROR = f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"
        return False
    # v147 — also try to load the threaded worker. Failure is non-fatal:
    # the logger falls back to inline computation.
    try:
        import importlib
        wmod = importlib.import_module('neuraxon.metrics_worker')
        _get_metrics_worker = wmod.get_worker
        _shutdown_metrics_worker = wmod.shutdown_worker
        _METRICS_WORKER_AVAILABLE = True
    except BaseException as wexc:
        import traceback as _tb
        _METRICS_WORKER_IMPORT_ERROR = f"{type(wexc).__name__}: {wexc}\n{_tb.format_exc()}"
        _METRICS_WORKER_AVAILABLE = False
    return True


# ============================================================================
# v147 (v4.55) — Survivability tracker
# ============================================================================
class SurvivabilityTracker:
    """Population-level survivability metrics — designed to surface the
    "NxErs stop moving and die" pathology observed in v146 sample runs.
    
    Tracks per NxEr:
      * last_pos      — the last sampled (x, y)
      * last_pos_tick — game tick when we last saw a position change
      * birth_tick    — when registered as alive
    
    Each sampling call updates rolling counters and exposes:
    
      alive_count        — current population
      births_window      — births in last STUCK_WINDOW_TICKS
      deaths_window      — deaths in last STUCK_WINDOW_TICKS
      mean_lifespan_s    — running mean of dead-NxEr lifespans
      mean_food          — population-mean food level
      mean_motion_rate   — fraction of NxErs that moved this sample
      stuck_fraction     — fraction of NxErs that haven't moved in STUCK_TICKS
      survivability_score in [0, 1] — composite indicator (see _compute_score)
    
    The survivability_score is a single 0-100 health number that the
    dashboard's persistent HUD strip can colour-code (green/amber/red).
    
    Thread safety: this object is updated only from the main game thread (in
    _log_tick_level2); the dashboard reads its public attrs from the main
    thread too (via the renderer). No locking required.
    """
    STUCK_WINDOW_TICKS: int = 200      # window for births/deaths-rate computation
    STUCK_TICKS: int = 30              # NxEr is "stuck" if hasn't moved in this many ticks
    LIFESPAN_HISTORY: int = 200        # rolling mean over last N deaths
    
    def __init__(self):
        # per-NxEr state — keyed by id, cleared on death
        self._last_pos: Dict[int, Tuple[float, float]] = {}
        self._last_move_tick: Dict[int, int] = {}
        self._birth_tick: Dict[int, int] = {}
        # rolling counters
        self._birth_ticks_window: deque = deque(maxlen=2000)  # tick numbers
        self._death_ticks_window: deque = deque(maxlen=2000)
        self._lifespans_recent: deque = deque(maxlen=self.LIFESPAN_HISTORY)
        # latest sample
        self.alive_count: int = 0
        self.births_window: int = 0
        self.deaths_window: int = 0
        self.mean_lifespan_s: float = 0.0
        self.mean_food: float = 0.0
        self.mean_motion_rate: float = 0.0     # fraction that moved this tick
        self.stuck_fraction: float = 0.0       # fraction stuck >= STUCK_TICKS
        self.survivability_score: float = 0.5  # composite [0, 1]
        self.last_tick_seen: int = 0
        
        # v170 (v4.78) — track the founding NxEr cohort separately so users
        # can answer "how many of the original 10 survived to 30 minutes?".
        # The set is populated by the FIRST sample() call (before any birth
        # events register), so it captures whichever NxErs were alive at
        # game start. Once an original dies, its id is removed from the set
        # (preserving the invariant: |_original_ids| = originals still alive).
        self._original_ids: set = set()
        self._original_ids_locked: bool = False
        self.original_alive_count: int = 0      # how many founders are still alive
        self.original_total: int = 0             # how many founders there were
        # v192 (v5.07) — HONEST, NEVER-TRIMMED survival summary. The
        # time_series['surv_alive_count'] list is trimmed to the last
        # max_history_length (=10000) samples for memory (v185+), dropping
        # the OLDEST ~1000 samples per trim. In a multi-hour NAS trial that
        # produces far more than 10000 samples, the early high-population
        # phase is trimmed away, so peak_alive computed from the trimmed
        # series reflects only the recent (often crashed) window — the v191
        # 8h run showed peak_alive=1 for trials that had genuinely sustained
        # ~10 NxErs earlier, which the population floor then zeroed. These
        # three running scalars summarise the WHOLE trial in O(1) memory and
        # are never trimmed, so the NAS can read the TRUE peak and the
        # time-averaged sustained population.
        self.max_alive_ever: int = 0            # true historical peak population
        self.alive_sum_ever: float = 0.0        # Σ alive_count over all samples
        self.alive_samples_ever: int = 0        # # of samples (for the mean)
        self.last_alive_count: int = 0          # most recent alive_count (true final)
        # v170 — full per-NxEr lifespan log (for distribution analysis).
        # Each entry: (nxer_id, birth_tick, death_tick, was_original)
        self._lifespan_log: list = []
    
    def register_birth(self, nxer_id: int, tick: int):
        self._birth_tick[nxer_id] = tick
        self._birth_ticks_window.append(tick)
    
    def register_death(self, nxer_id: int, tick: int):
        if nxer_id in self._birth_tick:
            birth = self._birth_tick[nxer_id]
            lifespan = tick - birth
            self._lifespans_recent.append(lifespan)
            # v170 — full lifespan log for distribution analysis
            was_original = nxer_id in self._original_ids
            self._lifespan_log.append((nxer_id, birth, tick, was_original))
            del self._birth_tick[nxer_id]
        else:
            # NxEr died without a recorded birth (loaded-from-save or founder
            # before the tracker locked in). Still record the death for the
            # lifespan distribution, with birth_tick=0 as best-effort.
            was_original = nxer_id in self._original_ids
            self._lifespan_log.append((nxer_id, 0, tick, was_original))
        # v170 — if this was an original NxEr, drop it from the alive set
        self._original_ids.discard(nxer_id)
        self._last_pos.pop(nxer_id, None)
        self._last_move_tick.pop(nxer_id, None)
        self._death_ticks_window.append(tick)
    
    def sample(self, tick: int, alive_nxers: list):
        """Called once per full-analytics tick from _log_tick_level2.
        v170 — lock the founding cohort on the first call so we can
        report 'originals still alive' separately from total population."""
        self.last_tick_seen = tick
        self.alive_count = len(alive_nxers)
        # v192 (v5.07) — update the never-trimmed honest survival summary.
        if self.alive_count > self.max_alive_ever:
            self.max_alive_ever = self.alive_count
        self.alive_sum_ever += self.alive_count
        self.alive_samples_ever += 1
        self.last_alive_count = self.alive_count
        
        # v170 — lock originals on first sample
        if not self._original_ids_locked:
            self._original_ids = {a.id for a in alive_nxers}
            self._original_ids_locked = True
            self.original_total = len(self._original_ids)
        # Compute original_alive_count (intersection of still-alive set with originals)
        alive_ids = {a.id for a in alive_nxers}
        self.original_alive_count = len(alive_ids & self._original_ids)
        # ---- births/deaths within window ----
        cutoff = tick - self.STUCK_WINDOW_TICKS
        # trim deques cheaply
        while self._birth_ticks_window and self._birth_ticks_window[0] < cutoff:
            self._birth_ticks_window.popleft()
        while self._death_ticks_window and self._death_ticks_window[0] < cutoff:
            self._death_ticks_window.popleft()
        self.births_window = len(self._birth_ticks_window)
        self.deaths_window = len(self._death_ticks_window)
        # ---- mean lifespan ----
        if self._lifespans_recent:
            # lifespans are in ticks — convert to seconds via 60 fps assumption
            # for a friendly display unit. The exact factor doesn't matter
            # for trend; consumers can re-scale.
            self.mean_lifespan_s = sum(self._lifespans_recent) / (60.0 * len(self._lifespans_recent))
        # ---- food, motion, stuck ----
        if not alive_nxers:
            self.mean_food = 0.0
            self.mean_motion_rate = 0.0
            self.stuck_fraction = 0.0
        else:
            food_sum = 0.0
            moved_count = 0
            stuck_count = 0
            for a in alive_nxers:
                aid = a.id
                food_sum += float(getattr(a, 'food', 0.0))
                pos = getattr(a, 'pos', None)
                if pos is None:
                    continue
                pos_t = (float(pos[0]), float(pos[1]))
                last = self._last_pos.get(aid)
                # First time we've seen this NxEr — register its birth if
                # we missed it (load_state path doesn't always go through
                # spawn_child).
                if last is None:
                    self._last_pos[aid] = pos_t
                    self._last_move_tick[aid] = tick
                    if aid not in self._birth_tick:
                        self._birth_tick[aid] = tick
                    continue
                if pos_t != last:
                    moved_count += 1
                    self._last_pos[aid] = pos_t
                    self._last_move_tick[aid] = tick
                else:
                    if (tick - self._last_move_tick.get(aid, tick)) >= self.STUCK_TICKS:
                        stuck_count += 1
            self.mean_food = food_sum / len(alive_nxers)
            self.mean_motion_rate = moved_count / len(alive_nxers)
            self.stuck_fraction = stuck_count / len(alive_nxers)
        # ---- composite score ----
        self.survivability_score = self._compute_score()
    
    def _compute_score(self) -> float:
        """Composite 0..1 health score combining the four most diagnostic
        signals, weighted by how acutely each predicts imminent collapse."""
        # 1. Population health: scale alive_count vs a soft target of 10
        pop_factor = min(1.0, self.alive_count / 10.0)
        # 2. Movement: low motion = bad. We want at least 30% of NxErs moving.
        motion_factor = min(1.0, self.mean_motion_rate / 0.30)
        # 3. Stuck fraction: penalise heavily.
        stuck_factor = max(0.0, 1.0 - self.stuck_fraction / 0.50)
        # 4. Births vs deaths over the rolling window — death-dominated
        #    populations score lower
        if self.births_window + self.deaths_window > 0:
            bd_balance = (self.births_window /
                          max(1, self.births_window + self.deaths_window))
            bd_factor = 0.5 + bd_balance * 0.5
        else:
            bd_factor = 0.5
        # weighted sum — emphasise motion (acute predictor) and population
        score = (0.30 * pop_factor + 0.30 * motion_factor +
                 0.25 * stuck_factor + 0.15 * bd_factor)
        return max(0.0, min(1.0, score))
    
    def to_dict(self) -> dict:
        return {
            'tick': self.last_tick_seen,
            'alive_count': self.alive_count,
            'births_window': self.births_window,
            'deaths_window': self.deaths_window,
            'mean_lifespan_s': round(self.mean_lifespan_s, 3),
            'mean_food': round(self.mean_food, 3),
            'mean_motion_rate': round(self.mean_motion_rate, 4),
            'stuck_fraction': round(self.stuck_fraction, 4),
            'survivability_score': round(self.survivability_score, 4),
            'window_ticks': self.STUCK_WINDOW_TICKS,
            'stuck_ticks_threshold': self.STUCK_TICKS,
            # v170 — original-cohort survival tracking
            'original_alive_count': self.original_alive_count,
            'original_total': self.original_total,
            'lifespan_log_size': len(self._lifespan_log),
        }


class DataLogger:
    """
    Comprehensive data logger for validating the Neuraxon research paper.
    
    Level 1: Basic logging (summary statistics, final states)
    Level 2: Detailed logging (time-series of all variables, plasticity events, etc.) Default
    Level 3: Deep detailed logging

    Data is kept in memory during gameplay and only saved:
    - At the end of the game (game over / all NxErs died)
    - When user explicitly saves the game
    - When user loads a game (saves current state first)
    """
    
    def __init__(self, log_level: int = 2, max_history_length: int = 10000):
        self.log_level = max(1, min(3, log_level))
        self.max_history_length = max_history_length
        self.reset()
    
    def _compress_series(self, data_list: list) -> list:
        """
        Compresses a list into [index, value] pairs. 
        Only stores the entry when the value changes.
        Example: [0, 0, 0, 5, 5] -> [[0, 0], [3, 5]]
        """
        if not data_list:
            return []
        
        compressed = [[0, data_list[0]]]
        last_val = data_list[0]
        
        for idx, val in enumerate(data_list[1:], 1):
            if val != last_val:
                compressed.append([idx, val])
                last_val = val
        return compressed

    def reset(self):
        """Reset all logged data."""
        self.start_time = time.time()
        self.current_step_data = {}  # Add this line to fix the AttributeError
        # v145 — lazy-load research probes on first reset (avoids circular import
        # at module load — see _ensure_research_probes_loaded() docstring).
        _ensure_research_probes_loaded()
        self.game_metadata = {
            'start_timestamp': datetime.now().isoformat(),
            'log_level': self.log_level,
            'version': '5.10',                     # v184
            'internal_version': 196,               # v152
            'research_probes_available': _RESEARCH_PROBES_AVAILABLE,
            'research_probes_import_error': _RESEARCH_PROBES_IMPORT_ERROR,
        }
        # v145 — Research probe state (M1-M10 from the v145 review)
        self.research_probes = _ResearchProbeState() if _RESEARCH_PROBES_AVAILABLE else None
        self.research_metric_keys = _research_all_metric_keys() if _RESEARCH_PROBES_AVAILABLE else []
        # v148 (v4.56) — concurrency primitives + threaded-worker handle.
        # The metrics_lock serialises (a) the worker thread's writes into
        # time_series and (b) the dashboard's reads (`list(ts.get(k, []))`).
        # Held only briefly (microseconds) — never around heavy compute.
        self.metrics_lock: threading.Lock = threading.Lock()
        self.metrics_worker = None         # populated lazily on first log_tick
        self.use_threaded_metrics: bool = bool(_THREADED_METRICS_DEFAULT)
        # v147 (v4.55) — Survivability tracker. See class docstring.
        self.survivability = SurvivabilityTracker()
        # v148 (v4.56) FIX: pending counters that the neuron firing path
        # increments during simulate(), drained-and-reset inside
        # _log_tick_level2. The previous design tried to increment
        # time_series['spontaneous_firing_count'][-1] from log_spontaneous_event
        # — but that slot is overwritten with a fresh 0 by _log_tick_level2
        # at the start of every tick, so the counts were lost. M6 was stuck
        # at 0.000 every sample for the entire run as a result.
        self._spont_count_pending: int = 0
        self._driven_count_pending: int = 0
        # v148 — pause flag. When True, log_tick is short-circuited so
        # we don't pollute the time series with frozen-state samples.
        self.paused: bool = False
        
        self.summary = {
            'total_ticks': 0,
            'total_neurons_created': 0,
            'total_neurons_died': 0,
            'total_synapses_created': 0,
            'total_synapses_pruned': 0,
            'total_plasticity_events': 0,
            'total_ltp_events': 0,
            'total_ltd_events': 0,
            'peak_network_activity': 0.0,
            'average_branching_ratio': 0.0,
            'branching_ratio_samples': 0,
            'neuromodulator_peaks': {
                'dopamine': 0.0,
                'serotonin': 0.0,
                'acetylcholine': 0.0,
                'norepinephrine': 0.0
            },
            # NEW summary stats
            'total_silent_synapse_activations': 0,
            'total_spontaneous_events': 0,
            'total_dendritic_spikes': 0,
            'total_homeostatic_adjustments': 0,
            'peak_phase_coherence': 0.0,
            # NEW: Updated Save states in v 2.1
            'total_threshold_modulations': 0,
            'total_associativity_events': 0,
            'total_metabotropic_activations': 0,
            'total_ionotropic_activations': 0,
            'peak_autocorrelation_window': 0.0,
            'mean_weight_change_rate': 0.0,
            'total_subthreshold_integrations': 0,
        }
        
        self.nxer_summary = {
            'total_born': 0,
            'total_died': 0,
            'max_food_found': 0.0,
            'max_time_lived': 0.0,
            'max_mates': 0,
            'max_explored': 0
        }
        
        # --- Event Lists (Unconditional Init to prevent AttributeError) ---
        self.plasticity_events = []
        self.structural_events = []
        self.neuron_snapshots = []
        self.synapse_snapshots = []
        self.nxer_events = []
        self.itu_fitness_history = []
        self.io_patterns = []
        
        self.silent_synapse_events = []
        self.spontaneous_events = []
        self.homeostatic_events = []
        self.dendritic_spike_events = []
        self.autoreceptor_events = []
        self.neuromodulator_events = []
        self.phase_reset_events = []
        
        # NEW: Additional event lists for paper validation
        self.weight_evolution_events = []
        self.threshold_modulation_events = []
        self.associativity_events = []
        self.subthreshold_events = []
        
        self.tracked_neuron_ids = []
        self.neuron_time_series = {}
        self.tracked_synapse_ids = []
        self.synapse_time_series = {}
        
        self.snapshot_interval = 100
        self.last_snapshot_tick = -1
        self.detailed_snapshot_interval = 500
        
        if self.log_level >= 2:
            self._init_level2_data()
    
    def _init_level2_data(self):
        self.time_series = {
            'ticks': [],
            'timestamps': [],
            'network_activity': [],
            'branching_ratio': [],
            'total_energy': [],
            'average_energy': [],
            'energy_efficiency': [],
            'temporal_sync': [],
            'dopamine': [],
            'serotonin': [],
            'acetylcholine': [],
            'norepinephrine': [],
            'oscillator_drive': [],
            
            # NEW: Oscillator components for CFC analysis
            'oscillator_low': [],
            'oscillator_mid': [],
            'oscillator_high': [],
            
            # NEW: Cross-frequency coupling metrics
            'phase_coherence': [],
            'cfc_low_mid': [],
            'cfc_mid_high': [],
            
            # NEW: Trinary state distributions
            'excitatory_fraction': [],
            'inhibitory_fraction': [],
            'neutral_fraction': [],
            
            # NEW v3.0: Circadian and Temperature metrics
            'circadian_phase': [],
            'day_night_state': [],  # 0=night, 0.5=transition, 1=day
            'mean_body_temperature': [],
            'temperature_variance': [],
            'resting_fraction': [],  # Fraction of NxErs in rest mode
            'proprioceptron_forced_turns': [],
            'proprioceptron_brain_warnings': [],
            'proprioceptron_brain_turns': [],
            # NEW v3.1: Additional time series for new inputs/outputs
            'daynight_input_distribution': [],  # Distribution of day/night input signals
            'temperature_input_distribution': [],  # Distribution of temp input signals
            'proprioception_input_distribution': [],  # Distribution of proprio input signals
            'resting_output_distribution': [],  # Distribution of resting output signals
            'rock_collision_rate': [],
            
            # NEW: Autoreceptor dynamics
            'autoreceptor_mean': [],
            'autoreceptor_std': [],
            
            # NEW: Adaptation dynamics
            'adaptation_mean': [],
            
            # NEW: Spontaneous activity metrics
            'spontaneous_firing_count': [],
            'driven_firing_count': [],
            
            # NEW: Synapse health metrics
            'silent_synapse_count': [],
            'active_synapse_count': [],
            'modulatory_synapse_count': [],
            'mean_synapse_integrity': [],
            
            # NEW: Dendritic metrics (averaged across network)
            'mean_plateau_potential': [],
            'mean_branch_potential': [],
            'dendritic_spike_count': [],
            
            # NEW: Intrinsic timescale distribution
            'mean_intrinsic_timescale': [],
            'timescale_heterogeneity': [],
            
            # NEW: Membrane potential statistics
            'membrane_potential_mean': [],
            'membrane_potential_std': [],
            
            # ============================================================
            # NEW: Synaptic Weight Evolution
            # ============================================================
            # Multi-timescale weight tracking (w_fast, w_slow, w_meta)
            'mean_w_fast': [],
            'mean_w_slow': [],
            'mean_w_meta': [],
            'std_w_fast': [],
            'std_w_slow': [],
            'std_w_meta': [],
            
            # Synaptic trace dynamics
            'mean_pre_trace': [],
            'mean_post_trace': [],
            'mean_pre_trace_ltd': [],
            'std_pre_trace': [],
            
            # Weight change rates
            'mean_delta_w': [],
            'ltp_rate': [],  # LTP events per tick
            'ltd_rate': [],  # LTD events per tick
            
            # ============================================================
            # NEW: Plasticity and Associativity
            # ============================================================
            # Associativity contribution from neighboring synapses
            'mean_associativity_contribution': [],
            'associativity_event_count': [],
            
            # Learning rate modulation by neuromodulators
            'mean_learning_rate_mod': [],
            'std_learning_rate_mod': [],
            
            # ============================================================
            # NEW: Self-Generated Activity / ACW
            # ============================================================
            # Autocorrelation Window (ACW) - critical for intrinsic timescales
            'mean_autocorrelation_window': [],
            'std_autocorrelation_window': [],
            'autocorrelation_coefficient_mean': [],
            
            # ============================================================
            # NEW: Threshold Modulation
            # ============================================================
            # Effective threshold tracking (after neuromodulation + autoreceptor)
            'mean_threshold_excitatory_effective': [],
            'mean_threshold_inhibitory_effective': [],
            'threshold_modulation_by_ach': [],
            'threshold_modulation_by_autoreceptor': [],
            
            # Ionotropic vs Metabotropic channel contributions
            'ionotropic_contribution_mean': [],
            'metabotropic_contribution_mean': [],
            
            # ============================================================
            # NEW: Neuromodulator Spatial Dynamics
            # ============================================================
            # Modulator grid spatial statistics
            'modulator_grid_entropy': [],
            'modulator_grid_gradient_magnitude': [],
            'dopamine_spatial_variance': [],
            'serotonin_spatial_variance': [],
            
            # ============================================================
            # NEW: Silent Synapse Dynamics
            # ============================================================
            'silent_synapse_fraction': [],
            'silent_to_active_transitions': [],
            'active_to_silent_transitions': [],
            
            # ============================================================
            # NEW: Complex Signaling / Subthreshold
            # ============================================================
            'subthreshold_integration_count': [],
            'near_threshold_fraction': [],  # Neurons close to firing
            
            # ============================================================
            # NEW: Extended Oscillator Metrics
            # ============================================================
            # Phase-Amplitude Coupling (PAC) detailed metrics
            'pac_theta_gamma': [],
            'pac_delta_theta': [],
            'mean_phase_velocity': [],
            
            # v3.2: Resting/Circadian aggregated metrics
            'resting_fraction': [],
            'proprioceptron_forced_turns_total': [],
            'proprioceptron_successful_streak_mean': [],
            'temperature_circadian_correlation': [],
            
            # v3.2 / v4.5: I/O TIMESERIES (v4.5 adds Song input + Sing output)
            'input_0_movement_mean': [],
            'input_1_encounter_mean': [],
            'input_2_terrain_mean': [],
            'input_3_hunger_mean': [],
            'input_4_sight_mean': [],
            'input_5_smell_mean': [],
            'input_6_daynight_mean': [],
            'input_7_temperature_mean': [],
            'input_8_proprioception_mean': [],
            'input_9_song_mean': [],                # NEW v4.5 (Song / Hearing)
            'output_0_movex_mean': [],
            'output_1_movey_mean': [],
            'output_2_social_mean': [],
            'output_3_mate_mean': [],
            'output_4_givefood_mean': [],
            'output_5_resting_mean': [],
            'output_6_sing_mean': [],               # NEW v4.5 (Sing)
            # v3.2: Metabolism context
            'mean_food_level': [],
            'food_consumption_rate': [],
            'mean_body_temperature': [],
            
            # ============================================================
            # NEW:  Aigarth/ITU Metrics
            # ============================================================
            'itu_mean_fitness': [],
            'itu_fitness_variance': [],
            'itu_mutation_events': [],
            'itu_pruning_events': [],
            
            # ============================================================
            # v5.0: Multi-Sphere Architecture Metrics (Paper Sections 7-8)
            # ============================================================
            'ms_nxers_with_brain': [],         # Count of NxErs with multi-sphere brains
            'ms_mean_spheres_per_brain': [],   # Average sphere count
            'ms_mean_links_per_brain': [],     # Average link count
            'ms_mean_brain_energy': [],        # Mean multi-sphere energy
            'ms_mean_link_integrity': [],      # Mean inter-sphere link integrity
            'ms_mean_inter_coherence': [],     # Mean inter-sphere oscillatory coherence
        }
        
        # ============================================================
        # v145 (v4.53): RESEARCH-PROBE TIME SERIES — the 10 paper-fidelity
        # metrics M1-M10. See neuraxon/research_probes.py and
        # docs/LOGGING.md for the full per-metric explanation. Each key is
        # appended exactly once per full-analytics tick from
        # _log_tick_level2 → _log_research_probes_tick().
        # ============================================================
        if _RESEARCH_PROBES_AVAILABLE:
            for k in _research_all_metric_keys():
                if k not in self.time_series:
                    self.time_series[k] = []
        
        # ============================================================
        # v147 (v4.55): SURVIVABILITY TIME SERIES — population-level health
        # signals designed to surface the "NxErs stop moving and die"
        # pathology observed in v146 sample runs. Each key is appended once
        # per full-analytics tick from _log_research_probes_tick() (which
        # samples self.survivability before kicking off the metrics work).
        # ============================================================
        for k in [
            'surv_alive_count',
            'surv_births_window',
            'surv_deaths_window',
            'surv_mean_lifespan_s',
            'surv_mean_food',
            'surv_mean_motion_rate',
            'surv_stuck_fraction',
            'surv_score',
            'surv_original_count',   # v170 — founders still alive
        ]:
            if k not in self.time_series:
                self.time_series[k] = []
        
        # ============================================================
        # v149 (v4.57): NEURON-STUCK DIAGNOSTIC TIME SERIES
        # Direct measurement of the M1 lock-in pathology that v149 step 1
        # is targeting. Each neuron tracks state_streak — how many ticks
        # it has been in the same trinary state. We aggregate to:
        #   stuck15:  fraction of neurons stuck for >= 15 ticks
        #   stuck30:  fraction of neurons stuck for >= 30 ticks
        #   stuck_at_pos: same but only for neurons currently at +1
        #   mean_streak: population-mean streak length (lower is better)
        # If the v149 adaptation fix works, stuck_at_pos should drop
        # from ~0.62 (the M1 lock-in level) toward 0.20-0.30.
        # ============================================================
        for k in [
            'stuck_fraction_15',
            'stuck_fraction_30',
            'stuck_fraction_at_pos1',
            'stuck_fraction_at_neg1',
            'mean_state_streak',
        ]:
            if k not in self.time_series:
                self.time_series[k] = []
        
        # ============================================================
        # v150 (v4.58): SENSORY→MOTOR COUPLING DIAGNOSTICS
        # The v149 fix removed neuron-level lock-in (stuck@+1 < 1.5%) but
        # population still crashed because M7 inverted to 1.94 — motor
        # output became driven by oscillators, not by sensory input.
        # These metrics quantify how responsive the network actually is
        # to the environment:
        #   input_active_fraction: fraction of input neurons firing this tick
        #   input_drive_pressure:  mean abs(external_input) reaching the network
        #   sensory_motor_corr:    rolling corr(input_activity, output_activity)
        # If the v150 sensory-boost fix works, all three should be > 0.20.
        # ============================================================
        for k in [
            'input_active_fraction',
            'input_drive_pressure',
            'sensory_motor_corr',
        ]:
            if k not in self.time_series:
                self.time_series[k] = []
        
        # ============================================================
        # v151 (v4.59): IDLE/SATURATION/EXPLORATION DIAGNOSTICS
        # The v150 sample run showed exactly the failure mode v151 needs
        # to detect:
        #   input_saturation_fraction → fraction of input neurons firing
        #     EVERY tick over the last 30 ticks (= 1.0 = saturated).
        #     v150 sample reached 1.0 at tick 600 and stayed there.
        #   pop_mean_idle_seconds → mean idle time across alive NxErs
        #     in seconds. Healthy < 1s. v150 sample run had this >> 5s
        #     in the long tail when NxErs were dying.
        #   exploration_trigger_rate → fraction of NxErs that hit the
        #     v151 idle-exploration safety net last tick. Healthy ~0;
        #     elevated rate means networks aren't producing useful motor
        #     outputs and the safety net is keeping them alive.
        # ============================================================
        for k in [
            'input_saturation_fraction',
            'pop_mean_idle_seconds',
            'exploration_trigger_rate',
            'motor_neutral_fraction',  # v152 — fraction of NxErs with O1=O2=0
            'input_locked_fraction',   # v153 — TRUE pathology (locked state)
            'input_variance_mean',     # v153 — continuous variance signal
            # v179 (v4.87) — population g-factor signatures (paper methodology)
            'g_pc1_fraction',          # PC1 eigenvalue fraction (paper "PC1%")
            'g_positive_manifold',     # fraction of positive off-diagonal r
            'g_mean_offdiag_r',        # mean off-diagonal r (paper "mean r")
            'g_lambda1_over_lambda2',  # λ1/λ2 unidimensionality indicator
        ]:
            if k not in self.time_series:
                self.time_series[k] = []
        self._latest_g = {}            # v179 — last computed g signatures
        # Per-NxEr rolling buffer for input-saturation detection.
        self._sat_input_buffer: Dict[int, list] = {}
        self._SAT_BUFFER_LEN = 30
        # Counter incremented by game_loop when idle-exploration triggers
        self._exploration_trigger_pending: int = 0
        # v152 — last-tick population saturation level, read by
        # synapses to brake plasticity when inputs are saturating.
        # Kept on the logger so synapses can read it without locking.
        self._latest_input_saturation: float = 0.0
        
        # v153 (v4.61) H — MEMBRANE DYNAMICS INSTRUMENTATION
        # ----------------------------------------------------
        # The v152 investigation could not fully account for observed
        # input-neuron membrane values (saved-state mp=-1.65 with
        # external_input=-1.0 should yield mp≈-0.35 by my equation,
        # but the JSON dumps show much deeper values). This means
        # the equation as I understand it isn't capturing all dynamics
        # — possibly DSN moving-average, multi-step substeps, or
        # synaptic terms I haven't traced.
        #
        # Rather than guessing, INSTRUMENT. Capture the full membrane
        # state for a sample of input neurons across the run, write
        # to a separate diagnostic file. Then future analysis has
        # GROUND TRUTH instead of inference.
        #
        # Captured per neuron snapshot:
        #   tick, nxer_id, neuron_id, membrane_potential, adaptation,
        #   autoreceptor, trinary_state, firing_rate_avg, state_streak
        # Sampling: every 100 ticks, first 3 input neurons of first 3
        # NxErs (= 9 neurons sampled, 9 rows per 100 ticks).
        # Total rows in 30k tick run: ~2700, very cheap.
        self._membrane_diag_rows: List[dict] = []
        self._membrane_diag_enabled: bool = True
        self._membrane_diag_sample_every: int = 100
        
        # Per-NxEr rolling buffers for the sensory_motor correlation.
        # Held on the logger (not on each NxEr) to keep mutation off
        # any per-tick critical path.
        self._sm_input_buffer: Dict[int, list] = {}
        self._sm_output_buffer: Dict[int, list] = {}
        self._SM_BUFFER_LEN = 30
        
        self.per_nxer_time_series: Dict[int, Dict[str, List]] = {}
    
    def _ensure_nxer_series(self, nxer_id: int, nxer_name: str):
        """Initialize time series structure for a specific NxEr if not exists."""
        if nxer_id not in self.per_nxer_time_series:
            # v111 FIX (M16): Added 9 receptor_* keys — was causing all-NaN in parquet
            self.per_nxer_time_series[nxer_id] = {
                'name': nxer_name,
                'ticks': [],
                'alive': [],
                'food': [],
                'pos_x': [],
                'pos_y': [],
                'network_activity': [],
                'branching_ratio': [],
                'total_energy': [],
                'average_energy': [],
                'dopamine': [],
                'serotonin': [],
                'acetylcholine': [],
                'norepinephrine': [],
                'membrane_potential_mean': [],
                'membrane_potential_std': [],
                'excitatory_fraction': [],
                'inhibitory_fraction': [],
                'neutral_fraction': [],
                'mean_w_fast': [],
                'mean_w_slow': [],
                'mean_w_meta': [],
                'phase_coherence': [],
                'fitness_score': [],
                'food_found': [],
                'explored': [],
                'mates_performed': [],
                'g_factor': [],   # v181 — per-NxEr g (PC1 factor score) over time
                
                # NEW v3.0: Individual circadian/temperature
                'body_temperature': [],
                'circadian_phase': [],
                'is_resting': [],
                'proprioceptron_rock_hits': [],
                'proprioceptron_forced_turns': [],
                'proprioceptron_brain_warnings': [],
                'proprioceptron_brain_turns': [],
                'brain_movement_weight': [],

                # v111 FIX (M16): Receptor activation logging.
                # ROOT CAUSE: NeuromodulatorSystem.compute_receptor_activations()
                # computed all 9 receptors correctly, but _log_nxer_individual()
                # never read them from the NxEr's network. v109 wrote to
                # logger.current_step_data (network-level), not per-NxEr series.
                # Paper §1 Neuromodulation, §8 receptor subtypes.
                'receptor_D1': [], 'receptor_D2': [],
                'receptor_5HT1A': [], 'receptor_5HT2A': [], 'receptor_5HT4': [],
                'receptor_M1': [], 'receptor_M2': [],
                'receptor_beta1': [], 'receptor_alpha2': [],

                # v4.5: Voice / Song / Hearing time series
                'voice_harmonicity': [],   # [0,1] mean pairwise consonance of active tones
                'voice_base_freq': [],     # Hz — drifts slowly via mutation across generations
                'sing_level': [],          # -1 silent, 0 hum, 1 full voice (trinary motor output)
                'song_input': [],          # trinary Song sensory input (index 9)
            }

    def _log_nxer_individual(self, tick: int, a: 'NxEr'):
        """Log individual NxEr's data independently."""
        if not a.alive:
            return  # Don't log dead NxErs at all
        self._ensure_nxer_series(a.id, a.name)
        series = self.per_nxer_time_series[a.id]
        
        series['ticks'].append(tick)
        series['alive'].append(a.alive)
        series['food'].append(a.food)
        series['pos_x'].append(a.pos[0])
        series['pos_y'].append(a.pos[1])
        series['fitness_score'].append(a.stats.fitness_score)
        series['food_found'].append(a.stats.food_found)
        series['explored'].append(a.stats.explored)
        series['mates_performed'].append(a.stats.mates_performed)
        series.setdefault('g_factor', []).append(
            getattr(a, '_g_score', getattr(a.stats, 'g_factor', 0.0)))  # v181
        
        if not a.alive:
            # Append zeros for dead NxEr's network data
            for key in ['network_activity', 'branching_ratio', 'total_energy', 'average_energy',
                        'dopamine', 'serotonin', 'acetylcholine', 'norepinephrine',
                        'membrane_potential_mean', 'membrane_potential_std',
                        'excitatory_fraction', 'inhibitory_fraction', 'neutral_fraction',
                        'mean_w_fast', 'mean_w_slow', 'mean_w_meta', 'phase_coherence',
                        'receptor_D1', 'receptor_D2', 'receptor_5HT1A', 'receptor_5HT2A',
                        'receptor_5HT4', 'receptor_M1', 'receptor_M2', 'receptor_beta1', 'receptor_alpha2',
                        # v4.5: Voice/Song series must stay aligned when the NxEr has no active neurons
                        'voice_harmonicity', 'voice_base_freq', 'sing_level', 'song_input']:
                series[key].append(0.0)
            return

        net = a.net
        active_neurons = [n for n in net.all_neurons if n.is_active]
        active_synapses = [s for s in net.synapses if s.integrity > 0]
        
        if not active_neurons:
            for key in ['network_activity', 'branching_ratio', 'total_energy', 'average_energy',
                        'dopamine', 'serotonin', 'acetylcholine', 'norepinephrine',
                        'membrane_potential_mean', 'membrane_potential_std',
                        'excitatory_fraction', 'inhibitory_fraction', 'neutral_fraction',
                        'mean_w_fast', 'mean_w_slow', 'mean_w_meta', 'phase_coherence',
                        'receptor_D1', 'receptor_D2', 'receptor_5HT1A', 'receptor_5HT2A',
                        'receptor_5HT4', 'receptor_M1', 'receptor_M2', 'receptor_beta1', 'receptor_alpha2',
                        # v4.5: Voice/Song series must stay aligned when the NxEr has no active neurons
                        'voice_harmonicity', 'voice_base_freq', 'sing_level', 'song_input']:
                series[key].append(0.0)
            return

        # Network activity
        activity = sum(abs(n.trinary_state) for n in active_neurons) / len(active_neurons)
        series['network_activity'].append(activity)
        series['branching_ratio'].append(net.branching_ratio)
        
        # Energy
        total_energy = sum(n.energy_level for n in active_neurons)
        series['total_energy'].append(total_energy)
        series['average_energy'].append(total_energy / len(active_neurons))
        
        # Neuromodulators
        series['dopamine'].append(net.neuromodulators.get('dopamine', 0.0))
        series['serotonin'].append(net.neuromodulators.get('serotonin', 0.0))
        series['acetylcholine'].append(net.neuromodulators.get('acetylcholine', 0.0))
        series['norepinephrine'].append(net.neuromodulators.get('norepinephrine', 0.0))
        
        # Membrane potentials
        membrane_potentials = [n.membrane_potential for n in active_neurons]
        series['membrane_potential_mean'].append(np.mean(membrane_potentials))
        series['membrane_potential_std'].append(np.std(membrane_potentials))
        
        # Trinary states
        states = [n.trinary_state for n in active_neurons]
        series['excitatory_fraction'].append(sum(1 for s in states if s == 1) / len(states))
        series['inhibitory_fraction'].append(sum(1 for s in states if s == -1) / len(states))
        series['neutral_fraction'].append(sum(1 for s in states if s == 0) / len(states))
        
        # Synaptic weights
        if active_synapses:
            series['mean_w_fast'].append(np.mean([s.w_fast for s in active_synapses]))
            series['mean_w_slow'].append(np.mean([s.w_slow for s in active_synapses]))
            series['mean_w_meta'].append(np.mean([s.w_meta for s in active_synapses]))
        else:
            series['mean_w_fast'].append(0.0)
            series['mean_w_slow'].append(0.0)
            series['mean_w_meta'].append(0.0)
        
        # NEW v3.0: Individual circadian/temperature metrics
        series['body_temperature'].append(getattr(a, 'body_temperature', 37.0))
        series['circadian_phase'].append(getattr(a, 'circadian_phase', 0.0))
        series['is_resting'].append(1.0 if getattr(a, 'is_resting', False) else 0.0)
        prop = getattr(a, 'proprioceptron', None)
        series['proprioceptron_rock_hits'].append(prop.total_rock_hits if prop else 0)
        series['proprioceptron_forced_turns'].append(prop.forced_turn_count if prop else 0)
        series['proprioceptron_brain_warnings'].append(prop.brain_warning_count if prop else 0)
        series['proprioceptron_brain_turns'].append(prop.brain_avoidance_turn_count if prop else 0)
        series['brain_movement_weight'].append(getattr(a, 'brain_movement_weight', 0.5))
        
        # Phase coherence

        # v111 FIX (M16): Log actual receptor activations from the NxEr's network.
        # These are computed each tick by NeuromodulatorSystem.compute_receptor_activations()
        # and stored on the network object. Previously only written to current_step_data
        # (global), never to per-NxEr time series.
        ra = getattr(a.net, 'receptor_activations', {})
        series['receptor_D1'].append(ra.get('D1', 0.0))
        series['receptor_D2'].append(ra.get('D2', 0.0))
        series['receptor_5HT1A'].append(ra.get('5HT1A', 0.0))
        series['receptor_5HT2A'].append(ra.get('5HT2A', 0.0))
        series['receptor_5HT4'].append(ra.get('5HT4', 0.0))
        series['receptor_M1'].append(ra.get('M1', 0.0))
        series['receptor_M2'].append(ra.get('M2', 0.0))
        series['receptor_beta1'].append(ra.get('beta1', 0.0))
        series['receptor_alpha2'].append(ra.get('alpha2', 0.0))

        phases = [n.phase for n in active_neurons]
        if len(phases) >= 2:
            # v4.52 PERF (#26): cmath already imported at module scope.
            complex_phases = [cmath.exp(1j * p) for p in phases]
            phase_coherence = abs(sum(complex_phases) / len(complex_phases))
        else:
            phase_coherence = 0.0
        series['phase_coherence'].append(phase_coherence)

        # v4.5: Voice / Song per-NxEr metrics
        v = getattr(a, 'voice', None)
        series['voice_harmonicity'].append(float(getattr(v, 'harmonicity', 0.5)) if v is not None else 0.5)
        series['voice_base_freq'].append(float(getattr(v, 'base_freq', 220.0)) if v is not None else 220.0)
        series['sing_level'].append(int(getattr(a, 'last_sing_level', 0)))
        # Song sensory input sits at index 9 of last_inputs (v4.5)
        try:
            song_in = int(a.last_inputs[9]) if len(a.last_inputs) > 9 else 0
        except Exception:
            song_in = 0
        series['song_input'].append(song_in)
    
    def _log_nxer_multisphere(self, tick: int, a: 'NxEr'):
        """v5.0: Level 3 logging for Multi-Sphere brain data (Paper Sections 7-8).
        
        Logs per-sphere activity, inter-sphere link metrics, and global coherence.
        Data stored under per_nxer_time_series[nxer_id] with 'ms_' prefix keys.
        """
        brain = a.brain
        if brain is None:
            return
        
        nxer_id = a.id
        self._ensure_nxer_series(nxer_id, a.name)
        series = self.per_nxer_time_series[nxer_id]
        
        # Ensure multi-sphere keys exist
        ms_keys = [
            'ms_num_spheres', 'ms_num_links', 'ms_total_energy',
            'ms_sensory_activity', 'ms_association_activity', 'ms_motor_activity',
            'ms_sensory_exc_frac', 'ms_association_exc_frac', 'ms_motor_exc_frac',
            'ms_link_mean_weight', 'ms_link_mean_integrity',
            'ms_global_da', 'ms_global_5ht', 'ms_global_ach', 'ms_global_ne',
            'ms_inter_sphere_coherence',
        ]
        for key in ms_keys:
            if key not in series:
                series[key] = []
        
        # Global multi-sphere metrics
        series['ms_num_spheres'].append(len(brain.spheres))
        series['ms_num_links'].append(len(brain.links))
        series['ms_total_energy'].append(brain.get_energy())
        
        # Per-sphere activity
        for sphere_name in ['sensory', 'association', 'motor']:
            sphere = brain.spheres.get(sphere_name)
            if sphere:
                active_neurons = [n for n in sphere.network.all_neurons if n.is_active]
                if active_neurons:
                    activity = sum(abs(n.trinary_state) for n in active_neurons) / len(active_neurons)
                    exc_frac = sum(1 for n in active_neurons if n.trinary_state == 1) / len(active_neurons)
                else:
                    activity = 0.0
                    exc_frac = 0.0
                series[f'ms_{sphere_name}_activity'].append(activity)
                series[f'ms_{sphere_name}_exc_frac'].append(exc_frac)
            else:
                series[f'ms_{sphere_name}_activity'].append(0.0)
                series[f'ms_{sphere_name}_exc_frac'].append(0.0)
        
        # Link metrics
        if brain.links:
            mean_w = 0.0
            mean_integrity = 0.0
            count = 0
            for link in brain.links.values():
                for row in link.weight_matrix:
                    for w in row:
                        mean_w += abs(w)
                        count += 1
                mean_integrity += link.integrity
            series['ms_link_mean_weight'].append(mean_w / max(count, 1))
            series['ms_link_mean_integrity'].append(mean_integrity / len(brain.links))
        else:
            series['ms_link_mean_weight'].append(0.0)
            series['ms_link_mean_integrity'].append(0.0)
        
        # Global neuromodulators
        series['ms_global_da'].append(brain.global_neuromodulators.get('dopamine', 0.0))
        series['ms_global_5ht'].append(brain.global_neuromodulators.get('serotonin', 0.0))
        series['ms_global_ach'].append(brain.global_neuromodulators.get('acetylcholine', 0.0))
        series['ms_global_ne'].append(brain.global_neuromodulators.get('norepinephrine', 0.0))
        
        # Inter-sphere coherence (mean phase coherence across all linked pairs)
        coherence_values = []
        for link in brain.links.values():
            src = brain.spheres.get(link.source_sphere_id)
            tgt = brain.spheres.get(link.target_sphere_id)
            if src and tgt:
                gate = link._communication_gate(src.network, tgt.network)
                coherence_values.append(gate)
        series['ms_inter_sphere_coherence'].append(
            sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
        )
    
    def set_level(self, level: int):
        new_level = max(1, min(3, level))
        if new_level != self.log_level:
            self.log_level = new_level
            self.game_metadata['log_level'] = self.log_level
            if new_level >= 2 and not hasattr(self, 'time_series'):
                # v145 — make sure research probes are loaded before init
                _ensure_research_probes_loaded()
                self._init_level2_data()

    # ================================================================
    # v185 (v5.0) MEMORY FIX — unified trimming across every accumulating
    # buffer the logger owns. Called once per full-analytics tick from
    # _log_tick_level2().
    #
    # Background: prior to v185 only self.time_series (line 1182, now
    # folded in here) and self.io_patterns (line ~3063) were bounded by
    # self.max_history_length. Every other list — plasticity/structural
    # events, the 12 paper-validation event streams, neuron/synapse
    # snapshots, itu_fitness_history, and the per-NxEr time series at
    # Log Level 3 — appended forever. The menu defaults Log Level to 3
    # and "Save full game logs to disk" to False, so a long-running game
    # accumulated everything in RAM without ever flushing it. Symptom:
    # steady RSS growth over hours/days until the OS killed the process.
    #
    # The fix uses the same "drop oldest 10% when full" rule that the
    # original time_series trim used, applied uniformly. Snapshot lists
    # are trimmed against a separate snapshot budget (snapshots are heavy
    # — each holds full state for every neuron/synapse — so we keep far
    # fewer of them than per-tick samples).
    #
    # The per_nxer_time_series dict is trimmed two ways:
    #   1. Each surviving NxEr's series gets the same drop-oldest-10% rule
    #      as the global time_series.
    #   2. Entries for NxErs that have been dead for many trims (no longer
    #      receive new samples) are removed entirely once their data sits
    #      idle for longer than _DEAD_NXER_GC_GRACE trims. The summary
    #      counters (nxer_summary, summary['total_*']) keep the long-term
    #      record so nothing important is lost.
    # ================================================================
    _SNAPSHOT_CAP_FACTOR: int = 20      # snapshot lists kept to max_history/20
    _DEAD_NXER_GC_GRACE: int = 3        # # of trims an NxEr's series may sit
                                        # idle before being garbage-collected

    # v186 (v5.01) — Membrane-diag is now bounded with the same trim+stream
    # pattern. Set generous because each row is small (one neuron sample)
    # and the file is consumed as a flat table.
    _MEMBRANE_DIAG_CAP: int = 50000     # rows kept in RAM; older rows
                                        # stream to disk on overflow

    # v186 (v5.01) — Streaming-export schema. Fixed superset of columns so
    # the stream file written incrementally during the run has a stable
    # header. Custom column subsets passed to save_key_metrics() at save
    # time are projected from this superset for the in-memory tail.
    _STREAM_KMETRICS_KEYS: tuple = (
        # the 10 paper-fidelity headline metrics
        'M1_excitatory_fraction', 'M2_mean_gate', 'M3_pac_modulation_idx',
        'M4_temporal_divergence', 'M5_branching_ratio',
        'M6_spontaneous_fraction', 'M7_zero_input_mi_ratio',
        'M8_sensory_vs_association_dissociation', 'M9_transfer_ratio',
        'M10_heritability_r',
        # diagnostics
        'stuck_fraction_at_pos1', 'stuck_fraction_at_neg1',
        'stuck_fraction_15', 'mean_state_streak',
        'input_active_fraction', 'input_drive_pressure',
        'sensory_motor_corr', 'input_saturation_fraction',
        'pop_mean_idle_seconds', 'exploration_trigger_rate',
        'motor_neutral_fraction', 'input_locked_fraction',
        'input_variance_mean', 'surv_score', 'surv_alive_count',
        'surv_original_count', 'g_pc1_fraction', 'g_positive_manifold',
        'g_mean_offdiag_r', 'g_lambda1_over_lambda2',
    )

    def set_export_context(self, game_id: str, out_dir: str = ".") -> None:
        """v186 (v5.01) — set the destination context for streaming export
        files. Called by game_loop at game start, and at every round-
        restart (so the new round's stream file uses the new game_id).

        Must be called BEFORE the first trim-with-overflow fires; otherwise
        the trimmed rows have nowhere to stream and would be lost in RAM
        when the buffer rolled over. The logger constructor doesn't know
        the game_id (chicken-and-egg with config), so this is a separate
        wiring step. Safe to call multiple times — the path is recomputed
        each call, and stream files for prior game_ids are left intact.
        """
        gid = (game_id or "unknown").replace('/', '_').replace('\\', '_')
        self._export_game_id = gid
        self._export_dir = out_dir or "."
        # Reset stream cursors so the new game_id starts a fresh file.
        self._stream_kmetrics_path = os.path.join(
            self._export_dir, f"{gid}__KeyMetrics.txt")
        self._stream_kmetrics_last_tick = -1
        self._stream_kmetrics_header_written = False
        self._stream_mdiag_path = os.path.join(
            self._export_dir, f"{gid}__MembraneDiag.txt")
        self._stream_mdiag_last_row = -1
        self._stream_mdiag_header_written = False

    def _ensure_stream_attrs(self) -> None:
        """Initialise streaming attrs to defaults if set_export_context was
        never called (back-compat for headless tests that construct a
        DataLogger directly and never wire it to a game session).
        """
        if not hasattr(self, '_export_game_id'):
            self._export_game_id = None
            self._export_dir = None
            self._stream_kmetrics_path = None
            self._stream_kmetrics_last_tick = -1
            self._stream_kmetrics_header_written = False
            self._stream_mdiag_path = None
            self._stream_mdiag_last_row = -1
            self._stream_mdiag_header_written = False

    def _stream_kmetrics_rows(self, start_idx: int, end_idx: int) -> None:
        """Append time_series rows [start_idx, end_idx) to the on-disk
        KeyMetrics stream file. Writes the header on first call. Safe to
        no-op when no export context is set.
        """
        self._ensure_stream_attrs()
        if self._stream_kmetrics_path is None:
            return
        if not hasattr(self, 'time_series'):
            return
        ts = self.time_series
        ticks = ts.get('ticks') or []
        if start_idx >= end_idx or end_idx > len(ticks):
            return
        timestamps = ts.get('timestamps') or []
        try:
            new_file = not self._stream_kmetrics_header_written
            with open(self._stream_kmetrics_path,
                      'w' if new_file else 'a',
                      encoding='utf-8') as f:
                if new_file:
                    f.write("# Neuraxon Game of Life v5.10 "
                            "— Key metrics export (streaming)\n")
                    f.write(f"# game_id={self._export_game_id}\n")
                    f.write("# NOTE: this file is written incrementally "
                            "while the game runs (every trim event "
                            "flushes the oldest rows here). It IS the "
                            "complete export — no separate finalise step "
                            "needed. RAM is bounded by max_history_length "
                            "but disk records the FULL run.\n")
                    f.write("# format=tab-separated, header row, one row "
                            "per full-analytics tick\n")
                    cols = ['tick', 'wallclock_seconds'] + list(self._STREAM_KMETRICS_KEYS)
                    f.write('\t'.join(cols) + '\n')
                    self._stream_kmetrics_header_written = True
                for i in range(start_idx, end_idx):
                    cells = [str(ticks[i]),
                             f"{timestamps[i]:.3f}" if i < len(timestamps) else "0.000"]
                    for k in self._STREAM_KMETRICS_KEYS:
                        seq = ts.get(k, [])
                        v = seq[i] if i < len(seq) else 0.0
                        cells.append(f"{float(v):.6f}")
                    f.write('\t'.join(cells) + '\n')
                if end_idx > 0 and ticks:
                    self._stream_kmetrics_last_tick = ticks[end_idx - 1]
        except Exception as exc:
            print(f"[DataLogger] stream KeyMetrics append failed: {exc}")

    def _stream_mdiag_rows(self, start_idx: int, end_idx: int) -> None:
        """Append membrane-diag rows [start_idx, end_idx) to the on-disk
        stream file. Writes the header on first call.
        """
        self._ensure_stream_attrs()
        if self._stream_mdiag_path is None:
            return
        rows = getattr(self, '_membrane_diag_rows', None) or []
        if start_idx >= end_idx or end_idx > len(rows):
            return
        try:
            new_file = not self._stream_mdiag_header_written
            cols = ['tick', 'nxer_id', 'neuron_id', 'layer', 'mp', 'adapt',
                    'autoreceptor', 'trinary_state', 'firing_rate_avg',
                    'state_streak', 'energy_level']
            with open(self._stream_mdiag_path,
                      'w' if new_file else 'a',
                      encoding='utf-8') as f:
                if new_file:
                    f.write("# Neuraxon Game of Life v5.10 "
                            "— Membrane diagnostics (streaming)\n")
                    f.write(f"# game_id={self._export_game_id}\n")
                    f.write("# NOTE: this file is written incrementally "
                            "as _membrane_diag_rows rolls over (cap "
                            f"{self._MEMBRANE_DIAG_CAP} rows). RAM is "
                            "bounded; disk records the FULL run.\n")
                    f.write(f"# sampled every {self._membrane_diag_sample_every} ticks, "
                            f"first 3 hidden + first 3 input neurons of first 3 "
                            f"alive NxErs each sample (v176 — layer column)\n")
                    f.write('\t'.join(cols) + '\n')
                    self._stream_mdiag_header_written = True
                for i in range(start_idx, end_idx):
                    row = rows[i]
                    line = '\t'.join(
                        (f"{row.get(c):.6f}"
                         if isinstance(row.get(c), float)
                         else str(row.get(c, 'input' if c == 'layer' else '')))
                        for c in cols)
                    f.write(line + '\n')
                self._stream_mdiag_last_row = end_idx - 1
        except Exception as exc:
            print(f"[DataLogger] stream MembraneDiag append failed: {exc}")

    def _trim_list_in_place(self, lst, max_len: int) -> bool:
        """Drop the oldest excess of `lst` when it has reached max_len.

        Returns True if a trim happened, False otherwise. Mutates the list
        in place so other references (e.g. dashboard readers holding the
        same reference) see the trimmed view. Cheap when no trim is needed
        — single len() compare.

        Trim target: 90% of max_len. In steady-state (caller appends ~1
        sample/tick and trims every tick) this drops exactly cap//10
        per call. If the list is FAR over (catch-up after
        max_history_length is reduced at runtime, or after a logger-level
        change resurfaces a long-idle buffer) we drop all excess in one
        call instead of bleeding it out one chunk per tick over thousands
        of ticks.
        """
        if max_len <= 0:
            return False
        if len(lst) >= max_len:
            target = max(0, (max_len * 9) // 10)
            drop = len(lst) - target
            if drop > 0:
                del lst[:drop]
                return True
        return False

    def _trim_all_history(self) -> None:
        """Prune every accumulating buffer on the logger.

        Cheap O(n_buffers) compares per call; only does real work when a
        buffer has reached its cap. Safe to call every full-analytics tick.
        Held under no lock (only the metrics worker writes time_series,
        and that path is already serialised by metrics_lock — but the trim
        happens before any worker enqueue inside _log_tick_level2, so the
        ordering is naturally race-free).
        """
        cap = self.max_history_length
        snap_cap = max(10, cap // self._SNAPSHOT_CAP_FACTOR)

        # 1) Global time_series (was the only buffer trimmed prior to v185).
        #    v186: BEFORE the trim deletes the oldest slice, stream it to
        #    disk so the on-disk KeyMetrics export remains COMPLETE for
        #    the full run. Without this the file only contains whatever
        #    the bounded RAM buffer happens to hold at save time — exactly
        #    the truncation problem the user flagged after seeing
        #    "earliest retained tick = 2001" in a 7.6-hour run.
        if hasattr(self, 'time_series') and self.time_series.get('ticks'):
            if len(self.time_series['ticks']) >= cap:
                target = max(0, (cap * 9) // 10)
                drop = len(self.time_series['ticks']) - target
                if drop > 0:
                    self._stream_kmetrics_rows(0, drop)
                    for key in self.time_series:
                        v = self.time_series[key]
                        if isinstance(v, list) and len(v) > drop:
                            del v[:drop]

        # 2) Event lists (Level 2+). Each can fire many times per tick, so
        #    these are the second-worst memory hogs after per_nxer_time_series.
        for attr in (
            'plasticity_events',
            'structural_events',
            'nxer_events',
            'silent_synapse_events',
            'spontaneous_events',
            'homeostatic_events',
            'dendritic_spike_events',
            'autoreceptor_events',
            'neuromodulator_events',
            'phase_reset_events',
            'weight_evolution_events',
            'threshold_modulation_events',
            'associativity_events',
            'subthreshold_events',
        ):
            lst = getattr(self, attr, None)
            if isinstance(lst, list):
                self._trim_list_in_place(lst, cap)

        # 3) Snapshots — heavy per entry, so use the snapshot cap not the
        #    full per-tick cap. ITU fitness history goes here too because
        #    it's appended once per ITU circle per snapshot.
        for attr in ('neuron_snapshots', 'synapse_snapshots', 'itu_fitness_history'):
            lst = getattr(self, attr, None)
            if isinstance(lst, list):
                self._trim_list_in_place(lst, snap_cap)

        # 4) Per-NxEr time series — the BIGGEST leak source at Log Level 3.
        #    Each NxEr's dict has 35+ parallel lists that grew unbounded.
        pn = getattr(self, 'per_nxer_time_series', None)
        if isinstance(pn, dict) and pn:
            self._gc_counter = getattr(self, '_gc_counter', 0) + 1
            last_seen = getattr(self, '_per_nxer_last_seen', None)
            if last_seen is None:
                last_seen = {}
                self._per_nxer_last_seen = last_seen

            dead_to_remove = []
            for nxer_id, series in pn.items():
                ticks_list = series.get('ticks')
                if isinstance(ticks_list, list) and ticks_list:
                    last_tick_logged = ticks_list[-1]
                    prev = last_seen.get(nxer_id)
                    if prev is None or prev[0] != last_tick_logged:
                        last_seen[nxer_id] = (last_tick_logged, self._gc_counter)
                    elif self._gc_counter - prev[1] >= self._DEAD_NXER_GC_GRACE:
                        # No new samples for several trims → dead/inactive.
                        dead_to_remove.append(nxer_id)
                        continue
                else:
                    # Empty series — mark for cleanup if it stays empty.
                    prev = last_seen.get(nxer_id)
                    if prev is None:
                        last_seen[nxer_id] = (-1, self._gc_counter)
                    elif self._gc_counter - prev[1] >= self._DEAD_NXER_GC_GRACE:
                        dead_to_remove.append(nxer_id)
                        continue

                # In-place trim of every parallel list in this NxEr's dict.
                if isinstance(ticks_list, list) and len(ticks_list) >= cap:
                    target = max(0, (cap * 9) // 10)
                    drop = len(ticks_list) - target
                    if drop > 0:
                        for k, v in series.items():
                            if isinstance(v, list) and len(v) > drop:
                                del v[:drop]

            for nxer_id in dead_to_remove:
                pn.pop(nxer_id, None)
                last_seen.pop(nxer_id, None)
                # Also drop the matching sensory-motor correlation buffers
                # (held on the logger, not on the NxEr) so dead agents
                # don't pin their 30-sample input/output windows in RAM.
                if hasattr(self, '_sm_input_buffer'):
                    self._sm_input_buffer.pop(nxer_id, None)
                if hasattr(self, '_sm_output_buffer'):
                    self._sm_output_buffer.pop(nxer_id, None)
    
    def log_tick(self, tick: int, nxers: dict = None, full_analytics: bool = True):
        """v4.1: Added full_analytics flag — when False, skip expensive Level 2 time-series.
        
        v148 (v4.56) pause / dup-tick guards
        ------------------------------------
        Two early-exits to protect the logger from being called while the
        game is paused or for the same tick more than once per analytics
        sample.  Both conditions were observed in real runs (game_id
        105866379 had 500/764 samples with duplicate tick values, plus
        2.4 s pause-period samples being logged with stale state).
        
        1) `self.paused` — set by the game loop on K_SPACE / save / load.
           When True, log_tick returns immediately without touching
           time_series. The dashboard / HUD continues to display the
           last-known values; nothing new is sampled until unpause.
        2) `tick == self._last_tick_logged` (and analytics requested) —
           short-circuits duplicate logging from the per-frame call site
           at game_loop.py:3244 which runs every render frame regardless
           of whether the simulation actually advanced.
        """
        # v148 — pause guard
        if self.paused:
            return
        # v148 — duplicate-tick guard. We allow the FIRST call for a new
        # tick to record everything, and skip subsequent calls for the
        # same tick. The non-analytics summary update (incrementing
        # total_ticks) is also skipped — it's already correct.
        if full_analytics and tick == getattr(self, '_last_analytics_tick', -1):
            return
        if full_analytics:
            self._last_analytics_tick = tick
        self.summary['total_ticks'] = tick
        
        all_nxers = list((nxers or {}).values()) if nxers else []
        alive_nxers = [a for a in all_nxers if a.alive]
        
        if alive_nxers:
            # v4.1: Sample neurons when population is large
            _sample_nxers = alive_nxers if len(alive_nxers) <= 200 else alive_nxers[::max(1, len(alive_nxers) // 200)]
            all_active_neurons = []
            for a in _sample_nxers:
                all_active_neurons.extend([n for n in a.net.all_neurons if n.is_active])
            
            if all_active_neurons:
                activity = sum(abs(n.trinary_state) for n in all_active_neurons) / len(all_active_neurons)
                self.summary['peak_network_activity'] = max(self.summary['peak_network_activity'], activity)
            
            branching_ratios = [a.net.branching_ratio for a in alive_nxers if a.net.branching_ratio > 0]
            if branching_ratios:
                avg_br = sum(branching_ratios) / len(branching_ratios)
                self.summary['average_branching_ratio'] = (
                    (self.summary['average_branching_ratio'] * self.summary['branching_ratio_samples'] + 
                    avg_br) / (self.summary['branching_ratio_samples'] + 1)
                )
                self.summary['branching_ratio_samples'] += 1
            
            for mod in ['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']:
                levels = [a.net.neuromodulators.get(mod, 0.0) for a in alive_nxers]
                avg_level = sum(levels) / len(levels) if levels else 0.0
                self.summary['neuromodulator_peaks'][mod] = max(self.summary['neuromodulator_peaks'][mod], avg_level)
        
        if self.log_level >= 2 and full_analytics:
            # v4.1: Only run expensive Level 2 analytics on full_analytics ticks
            self._log_tick_level2(tick, alive_nxers)
            if self.log_level >= 3:
                log_nxers = alive_nxers if len(alive_nxers) <= 100 else alive_nxers[::max(1, len(alive_nxers) // 100)]
                for a in log_nxers:
                    self._log_nxer_individual(tick, a)
                    # v5.0: Multi-sphere Level 3 logging
                    if getattr(a, 'brain', None) is not None:
                        self._log_nxer_multisphere(tick, a)
    
    def _log_tick_level2(self, tick: int, alive_nxers: list):
        """Capture detailed time series data each tick from ALL alive NxErs."""
        # v4.52 PERF (#26): removed inline `import numpy as np` and `import
        # cmath` — both already at module scope.

        # v185 (v5.0) MEMORY FIX — unified pruning across ALL accumulating
        # buffers. Previously only self.time_series and self.io_patterns
        # were bounded; event lists, snapshots, itu_fitness_history, and
        # per_nxer_time_series grew without limit, so long-running games
        # (especially under the menu's default Log Level 3, with "Save full
        # game logs to disk" unchecked) leaked RAM continuously. See
        # _trim_all_history() docstring for the full list.
        self._trim_all_history()
        
        # Basic timing
        self.time_series['ticks'].append(tick)
        self.time_series['timestamps'].append(time.time() - self.start_time)
        
        if not alive_nxers:
            # Append zeros/defaults if no alive NxErs
            for key in self.time_series:
                if key not in ['ticks', 'timestamps']:
                    self.time_series[key].append(0.0)
            return
        
        # v4.1: Sample NxErs for analytics when population is large (>200)
        if len(alive_nxers) > 200:
            _step = max(1, len(alive_nxers) // 200)
            sample_nxers = alive_nxers[::_step]
        else:
            sample_nxers = alive_nxers
        
        all_active_neurons = []
        all_active_synapses = []
        all_networks = []
        
        for a in sample_nxers:
            net = a.net
            all_networks.append(net)
            all_active_neurons.extend([n for n in net.all_neurons if n.is_active])
            all_active_synapses.extend([s for s in net.synapses if s.integrity > 0])
        
        if not all_active_neurons:
            for key in self.time_series:
                if key not in ['ticks', 'timestamps']:
                    self.time_series[key].append(0.0)
            return
        
        # === EXISTING METRICS (now aggregated) ===
        activity = sum(abs(n.trinary_state) for n in all_active_neurons) / len(all_active_neurons)
        self.time_series['network_activity'].append(activity)
        
        # Branching ratio (average across networks)
        branching_ratios = [net.branching_ratio for net in all_networks]
        self.time_series['branching_ratio'].append(np.mean(branching_ratios))
        
        # Energy status (aggregate)
        total_energy = sum(n.energy_level for n in all_active_neurons)
        avg_energy = total_energy / len(all_active_neurons)
        
        # Efficiency calculation
        energy_spent = sum(max(0, n.energy_baseline - n.energy_level) for n in all_active_neurons)
        total_steps = sum(net.step_count for net in all_networks)
        energy_spent += total_steps * 0.01 * len(all_active_neurons) / max(1, len(all_networks))
        total_activation = sum(sum(net.activation_history) if net.activation_history else 0 for net in all_networks)
        efficiency = total_activation / max(1, energy_spent) if energy_spent > 0 else 0.0
        
        self.time_series['total_energy'].append(total_energy)
        self.time_series['average_energy'].append(avg_energy)
        self.time_series['energy_efficiency'].append(efficiency)
        
        # Temporal sync (phase coherence across ALL neurons)
        phases = [n.phase for n in all_active_neurons]
        if len(phases) >= 2:
            complex_phases = [cmath.exp(1j * p) for p in phases]
            temporal_sync = abs(sum(complex_phases) / len(complex_phases))
        else:
            temporal_sync = 0.0
        self.time_series['temporal_sync'].append(temporal_sync)
        
        # Neuromodulators (average across all networks)
        for mod in ['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']:
            levels = [net.neuromodulators.get(mod, 0.0) for net in all_networks]
            self.time_series[mod].append(np.mean(levels))
        
        # Oscillator drive (average)
        osc_drives = [net._global_oscillatory_drive() for net in all_networks]
        self.time_series['oscillator_drive'].append(np.mean(osc_drives))
        
        # === OSCILLATOR COMPONENTS (use first network's time as reference) ===
        ref_net = all_networks[0]
        t = ref_net.time
        low = math.sin(2.0 * math.pi * ref_net.params.oscillator_low_freq * t + ref_net.oscillator_phase_offsets[0])
        mid = math.sin(2.0 * math.pi * ref_net.params.oscillator_mid_freq * t + ref_net.oscillator_phase_offsets[1])
        high = math.sin(2.0 * math.pi * ref_net.params.oscillator_high_freq * t + ref_net.oscillator_phase_offsets[2])
        self.time_series['oscillator_low'].append(low)
        self.time_series['oscillator_mid'].append(mid)
        self.time_series['oscillator_high'].append(high)
        
        # === CROSS-FREQUENCY COUPLING ===
        cfc_low_mid = abs(low) * abs(mid)
        cfc_mid_high = abs(mid) * abs(high)
        self.time_series['cfc_low_mid'].append(cfc_low_mid)
        self.time_series['cfc_mid_high'].append(cfc_mid_high)
        
        # Phase coherence
        if len(phases) >= 2:
            complex_phases = [cmath.exp(1j * p) for p in phases]
            phase_coherence = abs(sum(complex_phases) / len(complex_phases))
            if phase_coherence > 0.4:
                prev_coherence = self.time_series['phase_coherence'][-2] if len(self.time_series['phase_coherence']) > 1 else 0.0
                if prev_coherence <= 0.4:
                    self.log_phase_event(tick, "high_synchronization", phase_coherence, {'active_count': len(all_active_neurons)})
        else:
            phase_coherence = 0.0
        self.time_series['phase_coherence'].append(phase_coherence)
        
        # === TRINARY STATE DISTRIBUTIONS ===
        states = [n.trinary_state for n in all_active_neurons]
        excitatory_frac = sum(1 for s in states if s == 1) / len(states)
        inhibitory_frac = sum(1 for s in states if s == -1) / len(states)
        neutral_frac = sum(1 for s in states if s == 0) / len(states)
        self.time_series['excitatory_fraction'].append(excitatory_frac)
        self.time_series['inhibitory_fraction'].append(inhibitory_frac)
        self.time_series['neutral_fraction'].append(neutral_frac)
        
        # === NEW v3.0: CIRCADIAN AND TEMPERATURE METRICS ===
        if alive_nxers:
            # Circadian phase (should be same for all, use first)
            circadian_phase = getattr(alive_nxers[0], 'circadian_phase', 0.0)
            self.time_series['circadian_phase'].append(circadian_phase)
            
            # Day/night state: 0=night (0.5-1.0 phase), 1=day (0.0-0.5 phase)
            # With smooth transition
            if circadian_phase < 0.25:
                day_state = 0.5 + circadian_phase * 2  # Dawn: 0.5 -> 1.0
            elif circadian_phase < 0.5:
                day_state = 1.0  # Full day
            elif circadian_phase < 0.75:
                day_state = 1.0 - (circadian_phase - 0.5) * 2  # Dusk: 1.0 -> 0.5
            else:
                day_state = 0.5 - (circadian_phase - 0.75) * 2  # Night: 0.5 -> 0.0
            self.time_series['day_night_state'].append(max(0.0, day_state))
            
            # Temperature metrics
            temps = [getattr(a, 'body_temperature', 37.0) for a in alive_nxers]
            self.time_series['mean_body_temperature'].append(np.mean(temps))
            self.time_series['temperature_variance'].append(np.var(temps))
            
            # Resting fraction
            resting_count = sum(1 for a in alive_nxers if getattr(a, 'is_resting', False))
            self.time_series['resting_fraction'].append(resting_count / len(alive_nxers))
            
            # Proprioceptron metrics
            from simulation.entities import Proprioceptron
            total_forced = sum(getattr(a, 'proprioceptron', Proprioceptron()).forced_turn_count for a in alive_nxers)
            total_warnings = sum(getattr(a, 'proprioceptron', Proprioceptron()).brain_warning_count for a in alive_nxers)
            total_brain_turns = sum(getattr(a, 'proprioceptron', Proprioceptron()).brain_avoidance_turn_count for a in alive_nxers)
            total_hits = sum(getattr(a, 'proprioceptron', Proprioceptron()).total_rock_hits for a in alive_nxers)
            self.time_series['proprioceptron_forced_turns'].append(total_forced)
            self.time_series['proprioceptron_brain_warnings'].append(total_warnings)
            self.time_series['proprioceptron_brain_turns'].append(total_brain_turns)
            self.time_series['rock_collision_rate'].append(total_hits / max(1, len(alive_nxers)))
            
            # NEW v3.1: Track new input/output distributions
            daynight_dist = {'night': 0, 'transition': 0, 'day': 0}
            temp_dist = {'cold': 0, 'normal': 0, 'hot': 0}
            proprio_dist = {'blocked': 0, 'normal': 0, 'clear': 0}
            rest_dist = {'active': 0, 'normal': 0, 'rest': 0}
            
            for a in alive_nxers:
                inputs = getattr(a, 'last_inputs', (0,)*9)
                outputs = getattr(a, 'last_outputs', (0,)*6)
                if len(inputs) >= 9:
                    if inputs[6] == -1: daynight_dist['night'] += 1
                    elif inputs[6] == 1: daynight_dist['day'] += 1
                    else: daynight_dist['transition'] += 1
                    if inputs[7] == -1: temp_dist['cold'] += 1
                    elif inputs[7] == 1: temp_dist['hot'] += 1
                    else: temp_dist['normal'] += 1
                    if inputs[8] == -1: proprio_dist['blocked'] += 1
                    elif inputs[8] == 1: proprio_dist['clear'] += 1
                    else: proprio_dist['normal'] += 1
                if len(outputs) >= 6:
                    if outputs[5] == -1: rest_dist['active'] += 1
                    elif outputs[5] == 1: rest_dist['rest'] += 1
                    else: rest_dist['normal'] += 1
            
            self.time_series['daynight_input_distribution'].append(daynight_dist)
            self.time_series['temperature_input_distribution'].append(temp_dist)
            self.time_series['proprioception_input_distribution'].append(proprio_dist)
            self.time_series['resting_output_distribution'].append(rest_dist)
        else:
            for key in ['circadian_phase', 'day_night_state', 'mean_body_temperature',
                       'temperature_variance', 'resting_fraction', 
                       'proprioceptron_forced_turns', 'proprioceptron_brain_warnings',
                       'proprioceptron_brain_turns', 'rock_collision_rate',
                       'daynight_input_distribution', 'temperature_input_distribution',
                       'proprioception_input_distribution', 'resting_output_distribution']:
                if key.endswith('_distribution'):
                    self.time_series[key].append({})
                else:
                    self.time_series[key].append(0.0)
        
        # === AUTORECEPTOR DYNAMICS ===
        autoreceptors = [n.autoreceptor for n in all_active_neurons]
        self.time_series['autoreceptor_mean'].append(np.mean(autoreceptors))
        self.time_series['autoreceptor_std'].append(np.std(autoreceptors))
        
        # === ADAPTATION DYNAMICS ===
        adaptations = [n.adaptation for n in all_active_neurons]
        self.time_series['adaptation_mean'].append(np.mean(adaptations))
        
        # === MEMBRANE POTENTIAL STATISTICS ===
        membrane_potentials = [n.membrane_potential for n in all_active_neurons]
        self.time_series['membrane_potential_mean'].append(np.mean(membrane_potentials))
        self.time_series['membrane_potential_std'].append(np.std(membrane_potentials))
        
        # === INTRINSIC TIMESCALE DISTRIBUTION ===
        timescales = [n.intrinsic_timescale for n in all_active_neurons]
        self.time_series['mean_intrinsic_timescale'].append(np.mean(timescales))
        self.time_series['timescale_heterogeneity'].append(np.std(timescales) / max(0.01, np.mean(timescales)))
        
        # === SYNAPSE STATISTICS ===
        silent_count = sum(1 for s in all_active_synapses if s.is_silent)
        modulatory_count = sum(1 for s in all_active_synapses if s.is_modulatory)
        self.time_series['silent_synapse_count'].append(silent_count)
        self.time_series['active_synapse_count'].append(len(all_active_synapses) - silent_count)
        self.time_series['modulatory_synapse_count'].append(modulatory_count)
        
        if all_active_synapses:
            self.time_series['mean_synapse_integrity'].append(np.mean([s.integrity for s in all_active_synapses]))
        else:
            self.time_series['mean_synapse_integrity'].append(0.0)
        
        # === DENDRITIC METRICS ===
        all_branch_potentials = []
        all_plateau_potentials = []
        dendritic_spike_count = 0
        for n in all_active_neurons:
            for b in n.dendritic_branches:
                all_branch_potentials.append(b.branch_potential)
                all_plateau_potentials.append(b.plateau_potential)
                if abs(b.branch_potential) > b.branch_threshold:
                    dendritic_spike_count += 1
        
        self.time_series['mean_branch_potential'].append(np.mean(all_branch_potentials) if all_branch_potentials else 0.0)
        self.time_series['mean_plateau_potential'].append(np.mean(all_plateau_potentials) if all_plateau_potentials else 0.0)
        self.time_series['dendritic_spike_count'].append(dendritic_spike_count)
        
        # === SPONTANEOUS VS DRIVEN FIRING ===
        # v148 FIX: drain the pending counters that the neuron firing path
        # has been incrementing since the previous log_tick. This is the
        # canonical event-source for M6 spontaneous fraction.
        self.time_series['spontaneous_firing_count'].append(self._spont_count_pending)
        self.time_series['driven_firing_count'].append(self._driven_count_pending)
        self._spont_count_pending = 0
        self._driven_count_pending = 0
        
        # === v149 (v4.57): NEURON-STUCK DIAGNOSTICS ===
        # Direct measurement of M1 lock-in. Each neuron carries
        # state_streak (added in neuron.py v149). We aggregate the
        # population to spot stuck-at-+1 saturation in real time.
        if all_active_neurons:
            n_pop = len(all_active_neurons)
            stuck15 = sum(1 for n in all_active_neurons
                           if getattr(n, 'state_streak', 0) >= 15) / n_pop
            stuck30 = sum(1 for n in all_active_neurons
                           if getattr(n, 'state_streak', 0) >= 30) / n_pop
            stuck_pos = sum(1 for n in all_active_neurons
                             if getattr(n, 'state_streak', 0) >= 15
                             and n.trinary_state == 1) / n_pop
            stuck_neg = sum(1 for n in all_active_neurons
                             if getattr(n, 'state_streak', 0) >= 15
                             and n.trinary_state == -1) / n_pop
            mean_streak = sum(getattr(n, 'state_streak', 0)
                               for n in all_active_neurons) / n_pop
            self.time_series['stuck_fraction_15'].append(stuck15)
            self.time_series['stuck_fraction_30'].append(stuck30)
            self.time_series['stuck_fraction_at_pos1'].append(stuck_pos)
            self.time_series['stuck_fraction_at_neg1'].append(stuck_neg)
            self.time_series['mean_state_streak'].append(mean_streak)
        else:
            for k in ('stuck_fraction_15', 'stuck_fraction_30',
                      'stuck_fraction_at_pos1', 'stuck_fraction_at_neg1',
                      'mean_state_streak'):
                self.time_series[k].append(0.0)
        
        # === v150 (v4.58): SENSORY→MOTOR COUPLING DIAGNOSTICS ===
        # Quantifies how responsive the motor output is to sensory input.
        # The "input/output decoupling" pathology that caused population
        # crash 30 → 4 in the v149 sample run will show up here as:
        #   input_active_fraction → 0 (input neurons not firing)
        #   input_drive_pressure  → 0 (no external_input arriving)
        #   sensory_motor_corr   → 0 (output uncorrelated with input)
        # If the v150 fix works, all three should rise > 0.20.
        try:
            input_active_count = 0
            input_total_count = 0
            input_drive_total = 0.0
            n_with_inputs = 0
            sm_corrs = []
            for a in alive_nxers:
                if a.net is None:
                    continue
                inputs = a.net.input_neurons or []
                outputs = a.net.output_neurons or []
                if not inputs:
                    continue
                # Activity proxies — abs trinary state
                in_act = sum(1 for n in inputs
                              if n.is_active and abs(n.trinary_state) > 0)
                input_active_count += in_act
                input_total_count += len(inputs)
                # Pressure: mean abs(last_inputs) — what's actually arriving
                last_in = getattr(a, 'last_inputs', None) or ()
                if last_in:
                    drive_mag = sum(abs(float(v)) for v in last_in) / len(last_in)
                    input_drive_total += drive_mag
                    n_with_inputs += 1
                # Per-NxEr rolling input/output activity for correlation
                in_act_norm = (in_act / max(1, len(inputs)))
                out_act = (sum(abs(n.trinary_state) for n in outputs
                                if n.is_active) /
                            max(1, len(outputs)))
                inb = self._sm_input_buffer.setdefault(a.id, [])
                outb = self._sm_output_buffer.setdefault(a.id, [])
                inb.append(in_act_norm); outb.append(out_act)
                if len(inb) > self._SM_BUFFER_LEN:
                    inb.pop(0); outb.pop(0)
                if len(inb) >= 8:
                    # Pearson correlation
                    mi = sum(inb)/len(inb); mo = sum(outb)/len(outb)
                    num = sum((inb[k]-mi)*(outb[k]-mo) for k in range(len(inb)))
                    di = sum((v-mi)**2 for v in inb)
                    do = sum((v-mo)**2 for v in outb)
                    if di > 1e-9 and do > 1e-9:
                        sm_corrs.append(num / (di**0.5 * do**0.5))
            input_active_fraction = (input_active_count / input_total_count
                                       if input_total_count > 0 else 0.0)
            input_drive_pressure = (input_drive_total / n_with_inputs
                                      if n_with_inputs > 0 else 0.0)
            sensory_motor_corr = (sum(sm_corrs) / len(sm_corrs)
                                    if sm_corrs else 0.0)
            self.time_series['input_active_fraction'].append(input_active_fraction)
            self.time_series['input_drive_pressure'].append(input_drive_pressure)
            self.time_series['sensory_motor_corr'].append(sensory_motor_corr)
            # Garbage-collect rolling buffers for dead NxErs to avoid
            # unbounded growth across long sessions.
            alive_ids = {a.id for a in alive_nxers}
            for stale_id in [k for k in self._sm_input_buffer
                              if k not in alive_ids]:
                self._sm_input_buffer.pop(stale_id, None)
                self._sm_output_buffer.pop(stale_id, None)
        except Exception:
            for k in ('input_active_fraction', 'input_drive_pressure',
                      'sensory_motor_corr'):
                self.time_series[k].append(0.0)

        # === v179 (v4.87): POPULATION g-FACTOR (paper methodology) ===
        # Read-only over the living population — never alters dynamics.
        # Mirrors the uploaded paper's pipeline: positive manifold, PC1
        # fraction, mean off-diagonal r, λ1/λ2. Also writes a transient
        # per-NxEr `_g_score` (PC1 factor score) used by the opt-in
        # g-fitness term and the lifespan/Best logs.
        try:
            from neuraxon.gfactor import compute_population_g
            _g = compute_population_g(alive_nxers, write_back=True)
            self.time_series['g_pc1_fraction'].append(_g['g_pc1_fraction'])
            self.time_series['g_positive_manifold'].append(_g['g_positive_manifold'])
            self.time_series['g_mean_offdiag_r'].append(_g['g_mean_offdiag_r'])
            self.time_series['g_lambda1_over_lambda2'].append(_g['g_lambda1_over_lambda2'])
            self._latest_g = _g
        except Exception:
            for k in ('g_pc1_fraction', 'g_positive_manifold',
                      'g_mean_offdiag_r', 'g_lambda1_over_lambda2'):
                self.time_series[k].append(0.0)
        
        # === v151 (v4.59): IDLE / SATURATION / EXPLORATION DIAGNOSTICS ===
        # The 3 metrics that would have spotted the v150 sample run's
        # collapse before population crashed. See keys init for details.
        try:
            # v153 (v4.61) G: SMARTER SATURATION METRIC
            # =========================================
            # The v152 investigation showed input_saturation_fraction
            # conflated two very different states:
            # 1. constant non-zero firing from a stable environmental
            #    signal (e.g. sustained hunger, daynight cycle) — fine
            # 2. neurons locked in a fixed state with zero variance,
            #    no longer responsive to changes — actual pathology
            # The v152 metric counted BOTH as "saturated".
            #
            # v153 splits them. The buffer now stores actual trinary
            # states (-1, 0, +1), not just fired/not-fired. We then
            # compute three things:
            #   input_saturation_fraction  — old metric (for backcompat)
            #   input_locked_fraction      — fraction of inputs in a
            #                                FIXED state for all 30
            #                                ticks (TRUE pathology)
            #   input_variance_mean        — mean std-dev of input
            #                                states (low = stuck)
            # The "locked" metric is the real smoking-gun. The
            # "variance" gives a continuous read.
            sat_count = 0
            sat_total = 0
            locked_count = 0
            variance_sum = 0.0
            variance_n = 0
            for a in alive_nxers:
                if a.net is None or not a.net.input_neurons:
                    continue
                for n in a.net.input_neurons:
                    nid = (a.id, n.id)
                    buf = self._sat_input_buffer.setdefault(nid, [])
                    # v153 — store actual trinary state instead of just
                    # fired/not-fired. Backwards-compat: a fired==1
                    # buffer entry was equivalent to abs(state)>0.
                    buf.append(int(n.trinary_state) if n.is_active else 0)
                    if len(buf) > self._SAT_BUFFER_LEN:
                        buf.pop(0)
                    if len(buf) >= self._SAT_BUFFER_LEN:
                        sat_total += 1
                        # OLD metric: all non-zero (backwards compat)
                        if all(x != 0 for x in buf):
                            sat_count += 1
                        # NEW metric — LOCKED: every entry in buf is the
                        # same value (no transitions at all). This is the
                        # actual "stuck" pathology, not just "active".
                        first = buf[0]
                        if all(x == first for x in buf):
                            locked_count += 1
                        # NEW metric — variance of the state sequence.
                        # For trinary {-1, 0, +1} a varying signal will
                        # have stdev around 0.7-1.0; a constant signal
                        # has stdev=0. Lower std = less responsive.
                        m = sum(buf) / len(buf)
                        var = sum((x - m)**2 for x in buf) / len(buf)
                        variance_sum += var**0.5
                        variance_n += 1
            input_saturation_fraction = (sat_count / sat_total
                                           if sat_total > 0 else 0.0)
            input_locked_fraction = (locked_count / sat_total
                                       if sat_total > 0 else 0.0)
            input_variance_mean = (variance_sum / variance_n
                                     if variance_n > 0 else 0.0)
            self.time_series['input_saturation_fraction'].append(
                input_saturation_fraction)
            self.time_series['input_locked_fraction'].append(
                input_locked_fraction)
            self.time_series['input_variance_mean'].append(
                input_variance_mean)
            # v154 (v4.62) K — REVERT v153's brake rewire
            # ============================================
            # v153 rewired the plasticity brake from input_saturation_fraction
            # to input_locked_fraction, on the theory that SAT was over-
            # counting "stable signals" as pathology. The v153 sample run
            # nxon2_064274592 disproved this: LOCKED also reached 1.0 in
            # the same time window as SAT did (~tick 410-880), AND
            # network metrics got WORSE (M10 collapsed 0.485→0, M6 to 0,
            # M1 below band, population crash 2× faster).
            #
            # Hypothesis why: in v152 the brake fired whenever SAT > 0.5
            # (most ticks once signals are sustained). That was doing
            # *useful preventive work* even when "wrongly" reading SAT,
            # because it slowed plasticity during the PRE-lock-in window
            # when STDP runaway was still building. By rewiring to LOCKED,
            # v153 only braked AFTER lock-in already happened — too late.
            #
            # Revert: brake reads SAT again (v152 behavior). LOCKED and
            # variance are KEPT as diagnostic metrics in the time series
            # (they're useful for understanding what's happening), they
            # just don't drive the brake. Best of both worlds: working
            # plasticity brake + new diagnostic signals visible.
            self._latest_input_saturation = input_saturation_fraction
            # garbage-collect saturation buffers for dead neurons
            valid_ids = {(a.id, n.id) for a in alive_nxers
                          if a.net is not None
                          for n in (a.net.input_neurons or [])}
            for stale_id in [k for k in self._sat_input_buffer
                              if k not in valid_ids]:
                self._sat_input_buffer.pop(stale_id, None)
            
            # v153 H — MEMBRANE DYNAMICS INSTRUMENTATION
            # v176 (v4.84) — CRITICAL DIAGNOSTIC FIX. v153-v175 sampled
            # ONLY input_neurons. Input (sensory) neurons get their state
            # from set_state(input_vector[i]) — they are clamped directly to
            # the environment's sensory reading and NEVER run the update()
            # refractory/AHP path. The refractory_period_ticks / post_spike_
            # mp_reset / symmetric_stdp machinery (v171-v175) operates on
            # HIDDEN neurons. So every "state-0 buffer didn't materialize"
            # conclusion from v170-v175 was reading the sensory input
            # distribution (~65% -1 / ~35% +1, an artefact of how the game
            # encodes sensory channels), NOT network dynamics. It was
            # structurally impossible for that metric to ever show a
            # refractory rest band.
            #
            # v176 samples HIDDEN neurons (where refractory/AHP actually
            # runs) so the trinary 1/0/-1 distribution we evaluate is the
            # one the paper's model is about. Input neurons are still
            # sampled (tagged layer='input') so prior runs remain
            # comparable and we can see the sensory drive alongside the
            # internal dynamics. New column: `layer` (input|hidden).
            if (self._membrane_diag_enabled
                    and tick % self._membrane_diag_sample_every == 0):
                sampled = 0
                for a in alive_nxers[:3]:
                    if a.net is None:
                        continue
                    # layer, neuron-list  — hidden first (the neurons the
                    # refractory/AHP mechanism actually governs)
                    _groups = (
                        ('hidden', getattr(a.net, 'hidden_neurons', None)),
                        ('input',  getattr(a.net, 'input_neurons', None)),
                    )
                    for _layer, _neurons in _groups:
                        if not _neurons:
                            continue
                        for n in _neurons[:3]:
                            self._membrane_diag_rows.append({
                                'tick': tick,
                                'nxer_id': a.id,
                                'neuron_id': n.id,
                                'layer': _layer,
                                'mp': float(n.membrane_potential),
                                'adapt': float(n.adaptation),
                                'autoreceptor': float(getattr(n, 'autoreceptor', 0.0)),
                                'trinary_state': int(n.trinary_state),
                                'firing_rate_avg': float(getattr(n, 'firing_rate_avg', 0.0)),
                                'state_streak': int(getattr(n, 'state_streak', 0)),
                                'energy_level': float(getattr(n, 'energy_level', 0.0)),
                            })
                            sampled += 1
                # v186 (v5.01) — Memory-bound + stream-to-disk. Prior to
                # v186 this discarded the oldest 8k rows (drop down to 22k
                # whenever the buffer crossed 30k), silently losing data.
                # Now we stream the dropped slice to the on-disk
                # MembraneDiag file so the export remains COMPLETE for the
                # full run while RAM stays bounded. Cap raised to 50k since
                # streaming makes drops cheap (cap is purely a RAM ceiling
                # — disk has the full record).
                if len(self._membrane_diag_rows) >= self._MEMBRANE_DIAG_CAP:
                    target = max(0, (self._MEMBRANE_DIAG_CAP * 9) // 10)
                    drop = len(self._membrane_diag_rows) - target
                    if drop > 0:
                        self._stream_mdiag_rows(0, drop)
                        del self._membrane_diag_rows[:drop]
            
            # pop_mean_idle_seconds: mean wallclock idle time across
            # alive NxErs. Computed from last_move_tick. Surfaces the
            # death-spiral pathology (v150 had this >> 5s in long tail).
            idle_secs = []
            for a in alive_nxers:
                last_mv = getattr(a, 'last_move_tick', tick)
                # We don't know FIXED_DT here, but step ticks ≈ 60/sec
                # by default. The metric is monotone in idle ticks even
                # if scale shifts.
                idle_secs.append(max(0.0, (tick - last_mv) / 60.0))
            self.time_series['pop_mean_idle_seconds'].append(
                sum(idle_secs) / len(idle_secs) if idle_secs else 0.0)
            
            # exploration_trigger_rate: fraction of NxErs that hit the
            # v151 idle-exploration safety net last tick. Drained from
            # _exploration_trigger_pending (incremented by game_loop).
            n_alive = max(1, len(alive_nxers))
            self.time_series['exploration_trigger_rate'].append(
                self._exploration_trigger_pending / n_alive)
            self._exploration_trigger_pending = 0
            
            # v152 — motor_neutral_fraction: fraction of NxErs whose
            # last_outputs O1, O2 are both 0 (the trinary "do nothing"
            # state). The v151 sample showed the safety net never fired
            # despite NxErs being idle — meaning networks were producing
            # NON-ZERO motor outputs that hit blocked targets repeatedly.
            # If motor_neutral_fraction is LOW (< 0.20) AND idle is HIGH,
            # we know NxErs are stuck on blocked moves, not on neutral
            # outputs. The dashboard shows this directly.
            mn_count = 0
            for a in alive_nxers:
                lo = getattr(a, 'last_outputs', ())
                if len(lo) >= 2 and lo[0] == 0 and lo[1] == 0:
                    mn_count += 1
            self.time_series['motor_neutral_fraction'].append(
                mn_count / n_alive if alive_nxers else 0.0)
        except Exception:
            for k in ('input_saturation_fraction', 'pop_mean_idle_seconds',
                      'exploration_trigger_rate', 'motor_neutral_fraction',
                      'input_locked_fraction', 'input_variance_mean'):
                self.time_series[k].append(0.0)
        
        # === SYNAPTIC WEIGHT EVOLUTION ===
        if all_active_synapses:
            w_fast_vals = [s.w_fast for s in all_active_synapses]
            w_slow_vals = [s.w_slow for s in all_active_synapses]
            w_meta_vals = [s.w_meta for s in all_active_synapses]
            
            self.time_series['mean_w_fast'].append(np.mean(w_fast_vals))
            self.time_series['mean_w_slow'].append(np.mean(w_slow_vals))
            self.time_series['mean_w_meta'].append(np.mean(w_meta_vals))
            self.time_series['std_w_fast'].append(np.std(w_fast_vals))
            self.time_series['std_w_slow'].append(np.std(w_slow_vals))
            self.time_series['std_w_meta'].append(np.std(w_meta_vals))
            
            pre_traces = [s.pre_trace for s in all_active_synapses]
            post_traces = [s.post_trace for s in all_active_synapses]
            pre_traces_ltd = [s.pre_trace_ltd for s in all_active_synapses]
            
            self.time_series['mean_pre_trace'].append(np.mean(pre_traces))
            self.time_series['mean_post_trace'].append(np.mean(post_traces))
            self.time_series['mean_pre_trace_ltd'].append(np.mean(pre_traces_ltd))
            self.time_series['std_pre_trace'].append(np.std(pre_traces))
            
            delta_w_vals = [abs(s.potential_delta_w) for s in all_active_synapses]
            self.time_series['mean_delta_w'].append(np.mean(delta_w_vals))
            
            lr_mods = [s.learning_rate_mod for s in all_active_synapses]
            self.time_series['mean_learning_rate_mod'].append(np.mean(lr_mods))
            self.time_series['std_learning_rate_mod'].append(np.std(lr_mods))
            
            ionotropic_contrib = [abs(s.w_fast) + abs(s.w_slow) for s in all_active_synapses if not s.is_modulatory]
            metabotropic_contrib = [abs(s.w_meta) for s in all_active_synapses if s.is_modulatory]
            
            self.time_series['ionotropic_contribution_mean'].append(
                np.mean(ionotropic_contrib) if ionotropic_contrib else 0.0)
            self.time_series['metabotropic_contribution_mean'].append(
                np.mean(metabotropic_contrib) if metabotropic_contrib else 0.0)
        else:
            for key in ['mean_w_fast', 'mean_w_slow', 'mean_w_meta', 
                    'std_w_fast', 'std_w_slow', 'std_w_meta',
                    'mean_pre_trace', 'mean_post_trace', 'mean_pre_trace_ltd', 'std_pre_trace',
                    'mean_delta_w', 'mean_learning_rate_mod', 'std_learning_rate_mod',
                    'ionotropic_contribution_mean', 'metabotropic_contribution_mean']:
                self.time_series[key].append(0.0)
        
        # === PLASTICITY AND ASSOCIATIVITY ===
        if all_active_synapses:
            associativity_contribs = []
            for s in all_active_synapses:
                if s.neighbor_synapses:
                    neighbor_deltas = [ns.potential_delta_w for ns in s.neighbor_synapses[:3]]
                    if neighbor_deltas:
                        contrib = ref_net.params.associativity_strength * sum(
                            dw / (i + 1) for i, dw in enumerate(neighbor_deltas))
                        associativity_contribs.append(abs(contrib))
            
            self.time_series['mean_associativity_contribution'].append(
                np.mean(associativity_contribs) if associativity_contribs else 0.0)
            self.time_series['associativity_event_count'].append(
                sum(1 for c in associativity_contribs if c > 0.001))
        else:
            self.time_series['mean_associativity_contribution'].append(0.0)
            self.time_series['associativity_event_count'].append(0)
        
        self.time_series['ltp_rate'].append(0)
        self.time_series['ltd_rate'].append(0)
        
        # === AUTOCORRELATION WINDOWS ===
        acw_estimates = []
        autocorr_coeffs = []
        
        for n in all_active_neurons:
            if len(n.state_history) >= 10:
                states = list(n.state_history)
                states_a = states[:-1]
                states_b = states[1:]
                if np.std(states_a) < 1e-10 or np.std(states_b) < 1e-10:
                    continue
                try:
                    autocorr = np.corrcoef(states_a, states_b)[0, 1]
                    if not np.isnan(autocorr):
                        autocorr_coeffs.append(autocorr)
                        acw = n.intrinsic_timescale * (1.0 + abs(autocorr))
                        acw_estimates.append(acw)
                except:
                    pass
        
        if acw_estimates:
            mean_acw = np.mean(acw_estimates)
            self.time_series['mean_autocorrelation_window'].append(mean_acw)
            self.time_series['std_autocorrelation_window'].append(np.std(acw_estimates))
            self.summary['peak_autocorrelation_window'] = max(
                self.summary['peak_autocorrelation_window'], mean_acw)
        else:
            self.time_series['mean_autocorrelation_window'].append(0.0)
            self.time_series['std_autocorrelation_window'].append(0.0)
        
        self.time_series['autocorrelation_coefficient_mean'].append(
            np.mean(autocorr_coeffs) if autocorr_coeffs else 0.0)
        
        # === THRESHOLD MODULATION ===
        ach_levels = [net.neuromodulators.get('acetylcholine', 0.5) for net in all_networks]
        ach = np.mean(ach_levels)
        
        theta_exc_effectives = []
        theta_inh_effectives = []
        ach_mods = []
        autoreceptor_mods = []
        
        for n in all_active_neurons:
            threshold_mod = (ach - 0.5) * 0.5
            ach_mods.append(threshold_mod)
            
            autoreceptor_mod = -0.1 * n.autoreceptor
            autoreceptor_mods.append(autoreceptor_mod)
            
            theta_exc_eff = n.firing_threshold_excitatory - threshold_mod + autoreceptor_mod
            theta_inh_eff = n.firing_threshold_inhibitory - threshold_mod - autoreceptor_mod
            
            theta_exc_effectives.append(theta_exc_eff)
            theta_inh_effectives.append(theta_inh_eff)
        
        self.time_series['mean_threshold_excitatory_effective'].append(np.mean(theta_exc_effectives))
        self.time_series['mean_threshold_inhibitory_effective'].append(np.mean(theta_inh_effectives))
        self.time_series['threshold_modulation_by_ach'].append(np.mean(ach_mods))
        self.time_series['threshold_modulation_by_autoreceptor'].append(np.mean(autoreceptor_mods))
        
        # === NEUROMODULATOR SPATIAL DYNAMICS ===
        try:
            all_grid_entropies = []
            all_grad_magnitudes = []
            all_da_variances = []
            all_ser_variances = []
            
            for net in all_networks:
                grid_flat = net.modulator_grid.flatten()
                grid_normalized = (grid_flat - grid_flat.min()) / (grid_flat.max() - grid_flat.min() + 1e-10)
                hist, _ = np.histogram(grid_normalized, bins=20, density=True)
                hist = hist[hist > 0]
                grid_entropy = -np.sum(hist * np.log(hist + 1e-10)) / np.log(20)
                all_grid_entropies.append(grid_entropy)
                
                grad_y = np.diff(net.modulator_grid, axis=0)
                grad_x = np.diff(net.modulator_grid, axis=1)
                grad_magnitude = np.sqrt(np.mean(grad_y**2) + np.mean(grad_x**2))
                all_grad_magnitudes.append(grad_magnitude)
                
                all_da_variances.append(np.var(net.modulator_grid[:, :, 0]))
                all_ser_variances.append(np.var(net.modulator_grid[:, :, 1]))
            
            self.time_series['modulator_grid_entropy'].append(np.mean(all_grid_entropies))
            self.time_series['modulator_grid_gradient_magnitude'].append(np.mean(all_grad_magnitudes))
            self.time_series['dopamine_spatial_variance'].append(np.mean(all_da_variances))
            self.time_series['serotonin_spatial_variance'].append(np.mean(all_ser_variances))
        except Exception:
            self.time_series['modulator_grid_entropy'].append(0.0)
            self.time_series['modulator_grid_gradient_magnitude'].append(0.0)
            self.time_series['dopamine_spatial_variance'].append(0.0)
            self.time_series['serotonin_spatial_variance'].append(0.0)
        
        # === SILENT SYNAPSE DYNAMICS ===
        if all_active_synapses:
            silent_count = sum(1 for s in all_active_synapses if s.is_silent)
            silent_fraction = silent_count / len(all_active_synapses)
            self.time_series['silent_synapse_fraction'].append(silent_fraction)
        else:
            self.time_series['silent_synapse_fraction'].append(0.0)
        
        self.time_series['silent_to_active_transitions'].append(0)
        self.time_series['active_to_silent_transitions'].append(0)
        
        # === SUBTHRESHOLD INTEGRATION ===
        subthreshold_count = 0
        near_threshold_count = 0
        
        for n in all_active_neurons:
            if n.trinary_state == 0:
                theta_exc = n.firing_threshold_excitatory
                theta_inh = n.firing_threshold_inhibitory
                
                if n.membrane_potential > theta_exc * 0.8 or n.membrane_potential < theta_inh * 0.8:
                    near_threshold_count += 1
                    subthreshold_count += 1
        
        self.time_series['subthreshold_integration_count'].append(subthreshold_count)
        self.time_series['near_threshold_fraction'].append(
            near_threshold_count / len(all_active_neurons) if all_active_neurons else 0.0)
        
        # === EXTENDED OSCILLATOR/PAC METRICS ===
        t = ref_net.time
        
        delta = math.sin(2.0 * math.pi * 0.02 * t)
        theta_osc = math.sin(2.0 * math.pi * 0.08 * t)
        gamma = math.sin(2.0 * math.pi * 5.0 * t)
        
        pac_theta_gamma = abs(theta_osc) * abs(gamma)
        pac_delta_theta = abs(delta) * abs(theta_osc)
        
        self.time_series['pac_theta_gamma'].append(pac_theta_gamma)
        self.time_series['pac_delta_theta'].append(pac_delta_theta)
        
        if len(all_active_neurons) >= 2:
            phase_velocities = [n.natural_frequency * 2 * math.pi for n in all_active_neurons]
            self.time_series['mean_phase_velocity'].append(np.mean(phase_velocities))
        else:
            self.time_series['mean_phase_velocity'].append(0.0)
        
        # === v4.5: I/O TIMESERIES (length-normalised, cannot crash on mixed rows) ===
        # NxErs can transiently hold last_inputs/last_outputs of mixed lengths
        # during boot, load-from-older-save, or partial updates. We pad/truncate
        # every row to the expected v4.5 width so np.array() always succeeds.
        EXPECTED_INPUTS = 10   # v4.5
        EXPECTED_OUTPUTS = 7   # v4.5

        def _normalise_row(row, width):
            r = list(row) if row is not None else []
            if len(r) < width:
                r = r + [0] * (width - len(r))
            elif len(r) > width:
                r = r[:width]
            return r

        all_inputs = []
        all_outputs = []
        for a in alive_nxers:
            if hasattr(a, 'last_inputs'):
                all_inputs.append(_normalise_row(a.last_inputs, EXPECTED_INPUTS))
            if hasattr(a, 'last_outputs'):
                all_outputs.append(_normalise_row(a.last_outputs, EXPECTED_OUTPUTS))
        
        input_keys = ['input_0_movement_mean', 'input_1_encounter_mean', 'input_2_terrain_mean',
                      'input_3_hunger_mean', 'input_4_sight_mean', 'input_5_smell_mean',
                      'input_6_daynight_mean', 'input_7_temperature_mean', 'input_8_proprioception_mean',
                      'input_9_song_mean']   # v4.5
        if all_inputs:
            try:
                inputs_arr = np.array(all_inputs, dtype=float)
                for i, key in enumerate(input_keys):
                    if i < inputs_arr.shape[1]:
                        self.time_series[key].append(float(np.mean(inputs_arr[:, i])))
                    else:
                        self.time_series[key].append(0.0)
            except Exception:
                # Absolute belt-and-braces fallback — never let logging kill the sim.
                for key in input_keys:
                    self.time_series[key].append(0.0)
        else:
            for key in input_keys:
                self.time_series[key].append(0.0)
        
        output_keys = ['output_0_movex_mean', 'output_1_movey_mean', 'output_2_social_mean',
                       'output_3_mate_mean', 'output_4_givefood_mean', 'output_5_resting_mean',
                       'output_6_sing_mean']   # v4.5
        if all_outputs:
            try:
                outputs_arr = np.array(all_outputs, dtype=float)
                for i, key in enumerate(output_keys):
                    if i < outputs_arr.shape[1]:
                        self.time_series[key].append(float(np.mean(outputs_arr[:, i])))
                    else:
                        self.time_series[key].append(0.0)
            except Exception:
                for key in output_keys:
                    self.time_series[key].append(0.0)
        else:
            for key in output_keys:
                self.time_series[key].append(0.0)
        
        # === v3.2: METABOLISM CONTEXT ===
        resting_count = 0
        total_forced_turns = 0
        successful_streak_sum = 0
        streak_count = 0
        temps = []
        phases = []
        food_levels = []
        
        for a in alive_nxers:
            if getattr(a, 'is_resting', False):
                resting_count += 1
            prop = getattr(a, 'proprioceptron', None)
            if prop:
                total_forced_turns += prop.forced_turn_count
                successful_streak_sum += prop.successful_move_streak
                streak_count += 1
            temps.append(getattr(a, 'body_temperature', 37.0))
            phases.append(getattr(a, 'circadian_phase', 0.0))
            food_levels.append(getattr(a, 'food', 0.0))
        
        n_alive = max(1, len(alive_nxers))
        self.time_series['resting_fraction'].append(resting_count / n_alive)
        self.time_series['proprioceptron_forced_turns_total'].append(total_forced_turns)
        self.time_series['proprioceptron_successful_streak_mean'].append(successful_streak_sum / max(1, streak_count))
        self.time_series['mean_food_level'].append(np.mean(food_levels) if food_levels else 0.0)
        self.time_series['mean_body_temperature'].append(np.mean(temps) if temps else 37.0)
        
        if len(self.time_series['mean_food_level']) >= 2:
            prev_food = self.time_series['mean_food_level'][-2]
            curr_food = self.time_series['mean_food_level'][-1]
            self.time_series['food_consumption_rate'].append(prev_food - curr_food)
        else:
            self.time_series['food_consumption_rate'].append(0.0)
        
        # v3.32 FIX: Temporal correlation over rolling window instead of cross-NxEr at single tick.
        # Old code compared temps vs phases across NxErs at one tick, but all NxErs share the same
        # global circadian_phase → std(phases)≈0 → always returned 0.0.
        # New: correlate the mean_body_temperature and circadian_phase time series over recent ticks.
        ts_temp = self.time_series['mean_body_temperature']
        ts_phase = self.time_series['circadian_phase']
        window = TEMP_CIRCADIAN_CORR_WINDOW
        if len(ts_temp) >= window and len(ts_phase) >= window:
            t_win = np.array(ts_temp[-window:])
            p_win = np.array(ts_phase[-window:])
            if np.std(t_win) > 1e-9 and np.std(p_win) > 1e-9:
                corr = np.corrcoef(t_win, p_win)[0, 1]
                self.time_series['temperature_circadian_correlation'].append(float(corr) if not np.isnan(corr) else 0.0)
            else:
                self.time_series['temperature_circadian_correlation'].append(0.0)
        else:
            self.time_series['temperature_circadian_correlation'].append(0.0)
        
        # === ITU/AIGARTH METRICS ===
        all_itu_circles = []
        for net in all_networks:
            all_itu_circles.extend(net.itu_circles)
        
        if all_itu_circles:
            fitness_vals = []
            for circle in all_itu_circles:
                if circle.fitness_history:
                    fitness_vals.append(circle.fitness_history[-1])
            
            if fitness_vals:
                self.time_series['itu_mean_fitness'].append(np.mean(fitness_vals))
                self.time_series['itu_fitness_variance'].append(np.var(fitness_vals))
            else:
                self.time_series['itu_mean_fitness'].append(0.0)
                self.time_series['itu_fitness_variance'].append(0.0)
        else:
            self.time_series['itu_mean_fitness'].append(0.0)
            self.time_series['itu_fitness_variance'].append(0.0)
        
        self.time_series['itu_mutation_events'].append(0)
        self.time_series['itu_pruning_events'].append(0)
        
        # v5.0: Multi-Sphere population-level metrics
        ms_nxers = [a for a in alive_nxers if getattr(a, 'brain', None) is not None]
        self.time_series['ms_nxers_with_brain'].append(len(ms_nxers))
        if ms_nxers:
            self.time_series['ms_mean_spheres_per_brain'].append(
                sum(len(a.brain.spheres) for a in ms_nxers) / len(ms_nxers))
            self.time_series['ms_mean_links_per_brain'].append(
                sum(len(a.brain.links) for a in ms_nxers) / len(ms_nxers))
            self.time_series['ms_mean_brain_energy'].append(
                sum(a.brain.get_energy() for a in ms_nxers) / len(ms_nxers))
            integrities = []
            coherences = []
            for a in ms_nxers:
                for link in a.brain.links.values():
                    integrities.append(link.integrity)
                    src = a.brain.spheres.get(link.source_sphere_id)
                    tgt = a.brain.spheres.get(link.target_sphere_id)
                    if src and tgt:
                        coherences.append(link._communication_gate(src.network, tgt.network))
            self.time_series['ms_mean_link_integrity'].append(
                sum(integrities) / len(integrities) if integrities else 0.0)
            self.time_series['ms_mean_inter_coherence'].append(
                sum(coherences) / len(coherences) if coherences else 0.0)
        else:
            for key in ['ms_mean_spheres_per_brain', 'ms_mean_links_per_brain',
                        'ms_mean_brain_energy', 'ms_mean_link_integrity', 'ms_mean_inter_coherence']:
                self.time_series[key].append(0.0)
        
        # ============================================================
        # v145 (v4.53): RESEARCH-PROBE METRICS  (M1-M10)
        # ----------------------------------------------------------------
        # See neuraxon/research_probes.py and docs/LOGGING.md.
        # All sliding-window state lives in self.research_probes (ProbeState).
        # We pass back the OBSERVATIONS already computed above so this block
        # adds no extra O(neurons*synapses) scans — just slightly heavier
        # post-processing.
        # ============================================================
        self._log_research_probes_tick(
            tick=tick,
            alive_nxers=alive_nxers,
            all_active_neurons=all_active_neurons,
            all_networks=all_networks,
        )
        
        # Take snapshots at intervals
        if tick - self.last_snapshot_tick >= self.snapshot_interval:
            self._take_snapshot_multi(tick, alive_nxers)
            self.last_snapshot_tick = tick
    
    # ============================================================
    # v145 RESEARCH PROBES — implementation of the M1-M10 dashboard.
    # ============================================================
    def _log_research_probes_tick(self, tick: int, alive_nxers: list,
                                   all_active_neurons: list, all_networks: list):
        """Compute all M1-M10 metrics for this tick and append into time_series.
        
        v147 (v4.55): when threading is available and enabled, this packs a
        payload and enqueues it onto the metrics worker — returns within
        microseconds. The worker computes the metrics on a separate thread
        and writes results back through self._metrics_result_writer under
        self.metrics_lock.
        
        When the worker is unavailable / disabled / falls behind, this falls
        back to the v146 inline behaviour. The fallback is also used when the
        queue is full and the snapshot was dropped — better an inline tick
        than a missing one.
        """
        if not _RESEARCH_PROBES_AVAILABLE or self.research_probes is None:
            return
        # ---- always-do-on-main-thread bookkeeping ----
        # Survivability is sampled here because it depends on per-NxEr
        # position (which the worker doesn't see). Cheap O(N_alive).
        try:
            self.survivability.sample(tick, alive_nxers)
            # Append to the surv_* time series under the lock so the
            # dashboard's reads stay consistent.
            sd = self.survivability.to_dict()
            with self.metrics_lock:
                ts0 = self.time_series
                for k, mapping in [
                    ('surv_alive_count',     sd['alive_count']),
                    ('surv_births_window',   sd['births_window']),
                    ('surv_deaths_window',   sd['deaths_window']),
                    ('surv_mean_lifespan_s', sd['mean_lifespan_s']),
                    ('surv_mean_food',       sd['mean_food']),
                    ('surv_mean_motion_rate',sd['mean_motion_rate']),
                    ('surv_stuck_fraction',  sd['stuck_fraction']),
                    ('surv_score',           sd['survivability_score']),
                    # v170 — original-cohort survival
                    ('surv_original_count',  sd['original_alive_count']),
                ]:
                    if k not in ts0:
                        ts0[k] = []
                    ts0[k].append(float(mapping))
        except Exception:
            pass
        # Build the payload (cheap — references, not copies).
        ts = self.time_series
        def _last(key, default=0.0):
            lst = ts.get(key)
            return float(lst[-1]) if lst else default
        sample_low  = _last('oscillator_low')
        sample_mid  = _last('oscillator_mid')
        sample_high = _last('oscillator_high')
        sample_phases = (sample_low, sample_mid, sample_high)
        spont_count  = int(_last('spontaneous_firing_count', 0))
        driven_count = int(_last('driven_firing_count', 0))
        weight_means = {
            'mean_w_fast': _last('mean_w_fast'),
            'mean_w_slow': _last('mean_w_slow'),
            'mean_w_meta': _last('mean_w_meta'),
        }
        payload = {
            'probe_state': self.research_probes,
            'tick': tick,
            'alive_nxers': alive_nxers,
            'all_active_neurons': all_active_neurons,
            'all_networks': all_networks,
            'sample_oscillator_low':  sample_low,
            'sample_oscillator_mid':  sample_mid,
            'sample_oscillator_high': sample_high,
            'sample_phases': sample_phases,
            'spont_count':  spont_count,
            'driven_count': driven_count,
            'weight_means': weight_means,
        }
        # ---- threaded path ----
        if self.use_threaded_metrics and _METRICS_WORKER_AVAILABLE:
            # Lazy-create the worker on first call so the writer / lock
            # bindings are stable.
            if self.metrics_worker is None:
                try:
                    self.metrics_worker = _get_metrics_worker(
                        compute_fn=_research_compute_all_metrics,
                        result_writer=self._metrics_result_writer,
                        logger_lock=self.metrics_lock,
                    )
                except Exception as exc:
                    print(f"[DataLogger] couldn't start metrics worker: {exc}")
                    self.use_threaded_metrics = False
            if self.metrics_worker is not None and self.metrics_worker.enabled:
                accepted = self.metrics_worker.enqueue(payload)
                if accepted:
                    # Pad the keys NOW (synchronously) with the previous
                    # tick's value so 'ticks' length and metric-series
                    # length remain perfectly aligned. The worker will
                    # OVERWRITE the last element when it's done.
                    self._pad_research_keys_for_alignment()
                    return
                # queue full / dropped — fall through to inline
        # ---- inline (synchronous) fallback ----
        try:
            metrics = _research_compute_all_metrics(**payload)
            self._metrics_result_writer(tick, metrics)
        except Exception as exc:
            # Fail soft. Append zeros so series stays aligned.
            self._pad_research_keys_for_alignment()
    
    def _metrics_result_writer(self, tick: int, metrics: Dict[str, float]):
        """Final write step: append results into time_series under
        self.metrics_lock. Called from EITHER the worker thread (after
        compute) OR the main thread (inline fallback).
        
        Idempotency note: on the threaded path, _log_research_probes_tick
        has already padded each metric key to the correct length with the
        previous tick's value. Here we OVERWRITE the last element (rather
        than append) so the worker doesn't double-extend the series.
        """
        if metrics is None:
            return
        with self.metrics_lock:
            ts = self.time_series
            target_len = len(ts.get('ticks', []))
            for k, v in metrics.items():
                if k not in ts:
                    ts[k] = []
                # ensure length matches target before write
                while len(ts[k]) < target_len:
                    ts[k].append(0.0)
                if len(ts[k]) == target_len and target_len > 0:
                    # overwrite-last (worker path) — replaces the placeholder
                    try:
                        ts[k][-1] = float(v)
                    except (TypeError, ValueError):
                        pass
                else:
                    # append (rare — only on first sample)
                    try:
                        ts[k].append(float(v))
                    except (TypeError, ValueError):
                        ts[k].append(0.0)
            # All declared keys aligned to target_len with zeros if missing
            for k in self.research_metric_keys:
                if k not in ts:
                    ts[k] = []
                while len(ts[k]) < target_len:
                    ts[k].append(0.0)
    
    def _pad_research_keys_for_alignment(self):
        """Synchronously pad every research-probe key with the previous
        tick's value (or 0.0 for new keys) so the time-series length matches
        ticks. Used on the threaded path before enqueuing — the worker will
        overwrite the last element when its compute completes."""
        with self.metrics_lock:
            ts = self.time_series
            target_len = len(ts.get('ticks', []))
            for k in self.research_metric_keys:
                if k not in ts:
                    ts[k] = []
                seq = ts[k]
                if len(seq) < target_len:
                    pad_with = seq[-1] if seq else 0.0
                    while len(seq) < target_len:
                        seq.append(pad_with)
    

    def register_birth_for_heritability(self, child_nxer, parent_a_fitness: float,
                                         parent_b_fitness: float):
        """v145 — called by genetics.spawn_child / game_loop birth path so the
        heritability tracker can lock in the parent-fitness baseline at the
        moment of conception. Safe no-op if probes are disabled."""
        if not _RESEARCH_PROBES_AVAILABLE or self.research_probes is None:
            return
        try:
            avg = 0.5 * (float(parent_a_fitness) + float(parent_b_fitness))
            self.research_probes.heritability.register_birth(child_nxer, avg)
        except Exception:
            pass
        # v147 — also register the birth with the survivability tracker so
        # we can compute lifespan when this NxEr eventually dies.
        try:
            tick = (self.time_series.get('ticks') or [0])[-1]
            self.survivability.register_birth(child_nxer.id, tick)
        except Exception:
            pass
    
    def register_death_for_survivability(self, nxer_id: int):
        """v147 — called from the game_loop's death path so we can compute
        a lifespan and add it to the rolling mean. No-op if the NxEr was
        never registered (e.g. a founder loaded from a save file)."""
        try:
            tick = (self.time_series.get('ticks') or [0])[-1]
            self.survivability.register_death(nxer_id, tick)
        except Exception:
            pass
    
    def get_survivability_dashboard(self) -> dict:
        """v147 — population health snapshot for the persistent HUD strip
        and the L-dashboard's survivability section. Cheap to call on
        every frame.
        
        v149 — extended with neuron-stuck diagnostic fields. These
        complement the agent-level stuck_fraction (= NxErs not moving)
        with a neuron-level stuck_at_pos1 (= neurons frozen at +1, the
        M1 lock-in pathology). The two together pinpoint whether the
        problem is behavioural (NxErs not moving) or sub-symbolic
        (neurons frozen)."""
        d = self.survivability.to_dict()
        # health colour — the dashboard / HUD reads this directly
        s = d['survivability_score']
        if s >= 0.65:
            d['health_colour'] = (90, 215, 130)   # green
            d['health_label']  = 'thriving'
        elif s >= 0.35:
            d['health_colour'] = (235, 200, 90)   # amber
            d['health_label']  = 'struggling'
        else:
            d['health_colour'] = (235, 95, 95)    # red
            d['health_label']  = 'collapsing'
        # v149 — pull the latest stuck-state diagnostics off the
        # time series. Cheap (just a tail read).
        ts = self.time_series
        d['stuck_fraction_15']     = (ts.get('stuck_fraction_15')     or [0.0])[-1]
        d['stuck_fraction_30']     = (ts.get('stuck_fraction_30')     or [0.0])[-1]
        d['stuck_fraction_at_pos1']= (ts.get('stuck_fraction_at_pos1')or [0.0])[-1]
        d['stuck_fraction_at_neg1']= (ts.get('stuck_fraction_at_neg1')or [0.0])[-1]
        d['mean_state_streak']     = (ts.get('mean_state_streak')     or [0.0])[-1]
        # v150 — sensory→motor coupling. The "sensory_motor_corr" is the
        # primary indicator: > 0.20 means motor output tracks sensory
        # input; near 0 (or negative) means decoupled (the v149 pathology).
        d['input_active_fraction'] = (ts.get('input_active_fraction') or [0.0])[-1]
        d['input_drive_pressure']  = (ts.get('input_drive_pressure')  or [0.0])[-1]
        d['sensory_motor_corr']    = (ts.get('sensory_motor_corr')    or [0.0])[-1]
        # v151 — input-saturation, idle-time, and safety-net trigger rate.
        # input_saturation_fraction > 0.30 means inputs are losing their
        # information channel (v150 sample run pathology).
        d['input_saturation_fraction']= (ts.get('input_saturation_fraction')or [0.0])[-1]
        d['pop_mean_idle_seconds']    = (ts.get('pop_mean_idle_seconds')    or [0.0])[-1]
        d['exploration_trigger_rate'] = (ts.get('exploration_trigger_rate') or [0.0])[-1]
        # v152 — motor_neutral_fraction: fraction of NxErs with O1=O2=0.
        # Combined with pop_mean_idle_seconds: if both are HIGH, the trap
        # is the (0,0) trinary state. If idle is high but motor_neutral
        # is low, NxErs are stuck on BLOCKED targets (the v151 pathology).
        d['motor_neutral_fraction']   = (ts.get('motor_neutral_fraction')   or [0.0])[-1]
        # v153 — input_locked_fraction is the TRUE saturation metric.
        # Distinct from input_saturation_fraction which conflates
        # "constant sustained signal" (fine) with "locked state" (bad).
        # input_locked_fraction counts neurons stuck at ONE fixed state
        # for 30 ticks (zero transitions).
        d['input_locked_fraction']    = (ts.get('input_locked_fraction')    or [0.0])[-1]
        d['input_variance_mean']      = (ts.get('input_variance_mean')      or [0.0])[-1]
        return d
    
    def get_metrics_worker_stats(self) -> dict:
        """v147 — return a lightweight stats dict from the threaded metrics
        worker. Useful for the dashboard's status strip and for debugging."""
        if self.metrics_worker is None or not self.metrics_worker.enabled:
            return {'enabled': False}
        s = self.metrics_worker.stats
        return {
            'enabled': True,
            'thread_id': self.metrics_worker.thread_id,
            'jobs_processed': s.jobs_processed,
            'jobs_dropped': s.jobs_dropped,
            'last_compute_ms': round(s.last_compute_ms, 2),
            'mean_compute_ms': round(s.mean_compute_ms, 2),
            'queue_depth': s.queue_depth,
            'last_error': s.last_error,
        }
    
    def shutdown(self):
        """v147 — call from the game-loop exit path to cleanly join the
        metrics worker thread. Safe to call multiple times."""
        if _METRICS_WORKER_AVAILABLE and _shutdown_metrics_worker is not None:
            try:
                _shutdown_metrics_worker()
            except Exception as exc:
                print(f"[DataLogger] shutdown warning: {exc}")
        self.metrics_worker = None
    
    def save_membrane_diagnostics(self, game_id: str,
                                    out_dir: str = ".") -> Optional[str]:
        """v153 H — write captured membrane dynamics snapshots to a
        tab-separated file for offline analysis.
        
        File format (one row per sampled neuron per sampled tick):
          tick  nxer_id  neuron_id  layer  mp  adapt  autoreceptor
          trinary_state  firing_rate_avg  state_streak  energy_level

        v176 (v4.84): added `layer` column (input|hidden). v153-v175
        sampled input neurons only, which are environment-clamped via
        set_state() and bypass the refractory/AHP path — the state
        distribution there was the sensory drive, not network dynamics.
        v176 samples hidden neurons (where refractory/AHP runs) too.

        v186 (v5.01): cooperates with the streaming export — if a stream
        file already exists for this game_id (because rows have rolled
        over during the run), we just APPEND the in-memory tail to it.
        Otherwise we fall back to the pre-v186 single-shot write. Either
        way the returned file is the complete record.

        Returns the saved path, or None if no rows were captured.
        """
        self._ensure_stream_attrs()
        if not self._membrane_diag_rows and not self._stream_mdiag_header_written:
            return None
        gid = (game_id or "unknown").replace('/', '_').replace('\\', '_')
        # Honour the streaming path if export context was wired and
        # rolling has already happened — otherwise pick a sensible default.
        if (self._stream_mdiag_path
                and self._stream_mdiag_header_written
                and self._export_game_id == gid):
            # Stream file is already correct; flush the in-memory tail
            # (rows index 0 .. end; trim has already moved older rows out).
            self._stream_mdiag_rows(0, len(self._membrane_diag_rows))
            # Clear the tail so subsequent save_membrane_diagnostics() calls
            # don't double-write.
            self._membrane_diag_rows = []
            return os.path.abspath(self._stream_mdiag_path)
        # Fall-back: short run, no streaming ever happened. Write once.
        filename = os.path.join(out_dir, f"{gid}__MembraneDiag.txt")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Neuraxon Game of Life v5.10 — Membrane diagnostics\n")
                f.write(f"# game_id={gid}\n")
                f.write(f"# rows={len(self._membrane_diag_rows)}\n")
                f.write(f"# sampled every {self._membrane_diag_sample_every} ticks, "
                        f"first 3 hidden + first 3 input neurons of first 3 "
                        f"alive NxErs each sample (v176 — layer column)\n")
                cols = ['tick', 'nxer_id', 'neuron_id', 'layer', 'mp', 'adapt',
                        'autoreceptor', 'trinary_state', 'firing_rate_avg',
                        'state_streak', 'energy_level']
                f.write('\t'.join(cols) + '\n')
                for row in self._membrane_diag_rows:
                    line = '\t'.join(
                        (f"{row.get(c):.6f}"
                         if isinstance(row.get(c), float)
                         else str(row.get(c, 'input' if c == 'layer' else '')))
                        for c in cols)
                    f.write(line + '\n')
            return os.path.abspath(filename)
        except Exception as exc:
            print(f"[DataLogger] membrane diag save failed: {exc}")
            return None
    
    def save_lifespan_log(self, game_id: str,
                          out_dir: str = ".") -> Optional[str]:
        """v170 (v4.78) — write the per-NxEr lifespan log to a tab-separated
        file. Each death row is:
          nxer_id  birth_tick  death_tick  age_ticks  was_original  status

        v182 (v4.90): the file is ALWAYS written, even when no NxEr died
        (a very healthy run). Previously a zero-death trial produced NO
        file at all, which looked like a bug. Now we also append the
        surviving cohort (status=ALIVE, death_tick=-1) so the file is
        always present and informative. Returns the saved path.
        """
        log = getattr(self.survivability, '_lifespan_log', None) or []
        surv = self.survivability
        # Collect still-alive NxErs. After a run, surv._birth_tick holds
        # exactly the NxErs that are still alive (register_death deletes
        # the entry), so its keys ARE the surviving cohort.
        alive_rows = []
        try:
            ts = (self.time_series or {}).get('ticks', [])
            last_tick = int(ts[-1]) if ts else 0
            births = getattr(surv, '_birth_tick', {}) or {}
            originals = getattr(surv, '_original_ids', set()) or set()
            for nid, btk in list(births.items()):
                was_o = 1 if nid in originals else 0
                alive_rows.append((nid, int(btk), last_tick,
                                   max(0, last_tick - int(btk)), was_o))
            alive_rows.sort(key=lambda r: -r[3])   # longest-lived first
        except Exception:
            alive_rows = []

        gid = (game_id or "unknown").replace('/', '_').replace('\\', '_')
        filename = os.path.join(out_dir, f"{gid}__LifespanLog.txt")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Neuraxon Game of Life v5.10 — Per-NxEr lifespan log\n")
                f.write(f"# game_id={gid}\n")
                f.write(f"# deaths={len(log)}  survivors_listed={len(alive_rows)}\n")
                f.write(f"# founders_at_start={surv.original_total}\n")
                f.write(f"# founders_still_alive_at_export={surv.original_alive_count}\n")
                if not log:
                    f.write("# NOTE: zero deaths this trial — a healthy run. "
                            "Death rows are absent; survivor rows follow.\n")
                f.write("# format: tab-separated, header row. "
                        "status=DIED|ALIVE; ALIVE rows have death_tick=-1 "
                        "and age_ticks = ticks lived so far.\n")
                f.write("nxer_id\tbirth_tick\tdeath_tick\tage_ticks\t"
                        "was_original\tstatus\n")
                for nxer_id, birth, death, was_original in log:
                    age = death - birth
                    f.write(f"{nxer_id}\t{birth}\t{death}\t{age}\t"
                            f"{1 if was_original else 0}\tDIED\n")
                for nid, birth, _lt, age, was_o in alive_rows:
                    f.write(f"{nid}\t{birth}\t-1\t{age}\t{was_o}\tALIVE\n")
            return os.path.abspath(filename)
        except Exception as exc:
            print(f"[DataLogger] lifespan log save failed: {exc}")
            return None
    
    def save_key_metrics(self, game_id: str, out_dir: str = ".",
                          headline_keys: Optional[list] = None,
                          diagnostic_keys: Optional[list] = None) -> Optional[str]:
        """v156 (v4.63) — write FULL TIME SERIES of key metrics to a
        tab-separated file. Does NOT depend on UI state — no pygame, no
        MetricsDashboard, no renderer. Safe to call from finally blocks
        where the display may already be in shutdown.
        
        BUG CONTEXT (the "snapshot" issue user observed in v154):
        --------------------------------------------------------
        The previous auto-save path created a temporary MetricsDashboard
        which required `renderer.screen` to be valid. When the game ended
        cleanly through finally, pygame might already be tearing down →
        the dashboard construction failed silently → no full file written.
        The user reported observing "only snapshot" output, which we now
        understand was likely zero output from the failed branch with a
        residual file from some other source.
        
        Fix: this method is on the logger itself, takes no UI dependencies,
        and writes the FULL time-series (all samples since logger init).
        
        Returns the absolute saved path, or None if no samples to write.
        """
        if not hasattr(self, 'time_series'):
            return None
        ts = self.time_series
        if not ts.get('ticks'):
            return None
        # Defaults match what the dashboard exports
        if headline_keys is None:
            # The 10 research-metric headline keys, in stable order.
            headline_keys = [
                'M1_excitatory_fraction', 'M2_mean_gate',
                'M3_pac_modulation_idx', 'M4_temporal_divergence',
                'M5_branching_ratio', 'M6_spontaneous_fraction',
                'M7_zero_input_mi_ratio',
                'M8_sensory_vs_association_dissociation',
                'M9_transfer_ratio', 'M10_heritability_r',
            ]
        if diagnostic_keys is None:
            diagnostic_keys = [
                'stuck_fraction_at_pos1',
                'stuck_fraction_at_neg1',
                'stuck_fraction_15',
                'mean_state_streak',
                'input_active_fraction',
                'input_drive_pressure',
                'sensory_motor_corr',
                'input_saturation_fraction',
                'pop_mean_idle_seconds',
                'exploration_trigger_rate',
                'motor_neutral_fraction',
                'input_locked_fraction',
                'input_variance_mean',
                'surv_score',
                'surv_alive_count',
                'surv_original_count',   # v170 — how many of the founding NxErs are still alive
                'g_pc1_fraction',          # v179 (v4.87) — population g signatures
                'g_positive_manifold',
                'g_mean_offdiag_r',
                'g_lambda1_over_lambda2',
            ]
        all_export_keys = headline_keys + diagnostic_keys
        n = len(ts['ticks'])
        timestamps = ts.get('timestamps', [])
        gid = (game_id or "unknown").replace('/', '_').replace('\\', '_')

        # v186 (v5.01) — stream-aware finalisation. If trim has already
        # rolled rows to disk, the streaming file IS the export. We just
        # need to append the in-memory tail (rows past
        # _stream_kmetrics_last_tick). If no streaming happened (short run),
        # fall back to the single-shot write that matches pre-v186 output
        # — same format, same headline+diagnostic columns the caller asked
        # for, no surprise schema change.
        self._ensure_stream_attrs()
        if (self._stream_kmetrics_path
                and self._stream_kmetrics_header_written
                and self._export_game_id == gid):
            # Find the in-memory rows whose tick is greater than what the
            # stream has already seen — append only those.
            last_tick = self._stream_kmetrics_last_tick
            try:
                start_idx = 0
                ticks_list = ts['ticks']
                # ticks_list is monotonically non-decreasing → linear scan
                # from end is cheaper than bisect for typical 9k samples.
                for i in range(len(ticks_list) - 1, -1, -1):
                    if ticks_list[i] <= last_tick:
                        start_idx = i + 1
                        break
                if start_idx < n:
                    self._stream_kmetrics_rows(start_idx, n)
            except Exception as exc:
                print(f"[DataLogger] save_key_metrics tail-append failed: {exc}")
            return os.path.abspath(self._stream_kmetrics_path)

        # Fall-back path: no trim has streamed yet, so the in-memory buffer
        # IS the full run. Single-shot write, same format as before.
        header = ['tick', 'wallclock_seconds'] + all_export_keys
        filename = os.path.join(out_dir, f"{gid}__KeyMetrics.txt")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Neuraxon Game of Life v5.10 — Key metrics export\n")
                f.write(f"# game_id={gid}\n")
                f.write(f"# samples={n}\n")
                f.write("# format=tab-separated, header row, one row per "
                        "full-analytics tick\n")
                f.write(f"# keys: {', '.join(all_export_keys)}\n")
                f.write('\t'.join(header) + '\n')
                for i in range(n):
                    cells = [str(ts['ticks'][i]),
                             f"{timestamps[i]:.3f}" if i < len(timestamps) else "0.000"]
                    for k in all_export_keys:
                        seq = ts.get(k, [])
                        v = seq[i] if i < len(seq) else 0.0
                        cells.append(f"{float(v):.6f}")
                    f.write('\t'.join(cells) + '\n')
            return os.path.abspath(filename)
        except Exception as exc:
            print(f"[DataLogger] save_key_metrics failed: {exc}")
            return None
    
    def get_research_dashboard(self) -> dict:
        """Return the most recent value of each M1-M10 metric plus its in-band
        status. Useful for live HUD display or end-of-game summary."""
        if not _RESEARCH_PROBES_AVAILABLE:
            return {'available': False}
        ts = self.time_series
        out = {'available': True, 'tick': ts.get('ticks', [-1])[-1] if ts.get('ticks') else -1,
               'metrics': {}, 'in_band': {}}
        for k in self.research_metric_keys:
            seq = ts.get(k, [])
            if seq:
                out['metrics'][k] = seq[-1]
        for k in _RESEARCH_HEALTHY_BANDS.keys():
            seq = ts.get(f'{k}__in_band', [])
            if seq:
                out['in_band'][k] = bool(seq[-1])
        # quick traffic-light summary
        ib = out['in_band']
        n_in = sum(1 for v in ib.values() if v)
        out['summary'] = {
            'in_band_count': n_in,
            'total': len(ib),
            'pct_in_band': (100.0 * n_in / len(ib)) if ib else 0.0,
        }
        return out
    
    def _take_snapshot_multi(self, tick: int, alive_nxers: list):
        """Take detailed snapshots of ALL alive NxErs at intervals."""
        
        # Neuron snapshot - all neurons from all NxErs
        neuron_states = {}
        for a in alive_nxers:
            for n in a.net.all_neurons:
                neuron_states[f"{a.id}_{n.id}"] = {
                    'nxer_id': a.id,
                    'nxer_name': a.name,
                    'neuron_id': n.id,
                    'trinary_state': n.trinary_state,
                    'membrane_potential': n.membrane_potential,
                    'adaptation': n.adaptation,
                    'autoreceptor': n.autoreceptor,
                    'health': n.health,
                    'energy_level': n.energy_level,
                    'phase': n.phase,
                    'is_active': n.is_active,
                    'intrinsic_timescale': n.intrinsic_timescale,
                    'dendritic_branches': [{
                        'branch_id': b.branch_id,
                        'branch_potential': b.branch_potential,
                        'plateau_potential': b.plateau_potential,
                        'branch_threshold': b.branch_threshold,
                        'local_ca_influx': b.get_local_ca_influx()
                    } for b in n.dendritic_branches]
                }
        self.neuron_snapshots.append({'tick': tick, 'neuron_states': neuron_states})
        
        # Synapse snapshot - sample from all NxErs
        synapse_weights = {}
        for a in alive_nxers:
            sample_synapses = a.net.synapses[:50] if len(a.net.synapses) > 50 else a.net.synapses
            for s in sample_synapses:
                synapse_weights[f"{a.id}_{s.pre_id}_{s.post_id}"] = {
                    'nxer_id': a.id,
                    'nxer_name': a.name,
                    'w_fast': s.w_fast,
                    'w_slow': s.w_slow,
                    'w_meta': s.w_meta,
                    'integrity': s.integrity,
                    'pre_trace': s.pre_trace,
                    'post_trace': s.post_trace,
                    'is_silent': s.is_silent,
                    'is_modulatory': s.is_modulatory,
                    'tau_fast': s.tau_fast,
                    'tau_slow': s.tau_slow,
                    'tau_meta': s.tau_meta,
                    'learning_rate': s.learning_rate,
                    'plasticity_threshold': s.plasticity_threshold,
                    'potential_delta_w': s.potential_delta_w,
                    'neighbor_count': len(s.neighbor_synapses),
                }
        self.synapse_snapshots.append({'tick': tick, 'synapse_weights': synapse_weights})
        
        # ITU fitness history from all NxErs
        for a in alive_nxers:
            for circle in a.net.itu_circles:
                if circle.fitness_history:
                    self.itu_fitness_history.append({
                        'tick': tick,
                        'nxer_id': a.id,
                        'nxer_name': a.name,
                        'circle_id': circle.circle_id,
                        'fitness': circle.fitness_history[-1]
                    })
    
    
    def _take_snapshot(self, tick: int, network: 'NeuraxonNetwork', nxers: dict):
        """Take detailed snapshots at intervals."""
        
        # Existing neuron snapshot
        neuron_states = {}
        for n in network.all_neurons:
            neuron_states[n.id] = {
                'trinary_state': n.trinary_state,
                'membrane_potential': n.membrane_potential,
                'adaptation': n.adaptation,
                'autoreceptor': n.autoreceptor,  # NEW
                'health': n.health,
                'energy_level': n.energy_level,
                'phase': n.phase,
                'is_active': n.is_active,
                'intrinsic_timescale': n.intrinsic_timescale,
                # NEW: Dendritic branch details
                'dendritic_branches': [{
                    'branch_id': b.branch_id,
                    'branch_potential': b.branch_potential,
                    'plateau_potential': b.plateau_potential,
                    'branch_threshold': b.branch_threshold,
                    'local_ca_influx': b.get_local_ca_influx()
                } for b in n.dendritic_branches]
            }
        self.neuron_snapshots.append({'tick': tick, 'neuron_states': neuron_states})
        
        # Existing synapse snapshot (sample)
        synapse_weights = {}
        sample_synapses = network.synapses[:100] if len(network.synapses) > 100 else network.synapses
        for s in sample_synapses:
            synapse_weights[(s.pre_id, s.post_id)] = {
                'w_fast': s.w_fast,
                'w_slow': s.w_slow,
                'w_meta': s.w_meta,
                'integrity': s.integrity,
                'pre_trace': s.pre_trace,
                'post_trace': s.post_trace,
                'is_silent': s.is_silent,       # NEW
                'is_modulatory': s.is_modulatory, # NEW
                'tau_fast': s.tau_fast,          # NEW
                'tau_slow': s.tau_slow,          # NEW
                'tau_meta': s.tau_meta,          # NEW
                'learning_rate': s.learning_rate,  # NEW: Individual synapse learning rate
                'plasticity_threshold': s.plasticity_threshold,  # NEW
                'potential_delta_w': s.potential_delta_w,  # NEW: Current weight change
                'neighbor_count': len(s.neighbor_synapses),  # NEW: For associativity analysis
            }
        self.synapse_snapshots.append({'tick': tick, 'synapse_weights': synapse_weights})
        for circle in network.itu_circles:
            if circle.fitness_history:
                self.itu_fitness_history.append({
                    'tick': tick,
                    'circle_id': circle.circle_id,
                    'fitness': circle.fitness_history[-1]
                })
    
    def log_plasticity_event(self, tick: int, event_type: str, pre_id: int, post_id: int, 
                             delta_w: float, details: dict = None):
        self.summary['total_plasticity_events'] += 1
        if event_type == 'LTP':
            self.summary['total_ltp_events'] += 1
            # Update tick-level LTP rate
            if self.log_level >= 2 and self.time_series['ltp_rate']:
                self.time_series['ltp_rate'][-1] += 1
        elif event_type == 'LTD':
            self.summary['total_ltd_events'] += 1
            # Update tick-level LTD rate
            if self.log_level >= 2 and self.time_series['ltd_rate']:
                self.time_series['ltd_rate'][-1] += 1
        
        if self.log_level >= 2:
            self.plasticity_events.append({
                'tick': tick,
                'type': event_type,
                'pre_id': pre_id,
                'post_id': post_id,
                'delta_w': delta_w,
                'details': details or {}
            })
    
    # NEW: Event Logging Methods
    def log_silent_synapse_event(self, tick: int, pre_id: int, post_id: int, 
                                 became_active: bool, trigger: str = "unknown"):
        """Log when a silent synapse becomes active or vice versa."""
        if self.log_level >= 2:
            self.silent_synapse_events.append({
                'tick': tick,
                'pre_id': pre_id,
                'post_id': post_id,
                'became_active': became_active,
                'trigger': trigger
            })
            # Update transition counters
            if became_active:
                if self.time_series['silent_to_active_transitions']:
                    self.time_series['silent_to_active_transitions'][-1] += 1
            else:
                if self.time_series['active_to_silent_transitions']:
                    self.time_series['active_to_silent_transitions'][-1] += 1

    def log_spontaneous_event(self, tick: int, neuron_id: int, membrane_potential: float):
        """Log spontaneous firing events (not driven by synaptic input).
        
        v148 (v4.56) FIX: increments self._spont_count_pending instead of
        the time_series tail. The pending counter is drained-and-reset by
        _log_tick_level2 once per analytics tick. Previously this method
        incremented time_series['spontaneous_firing_count'][-1] which was
        overwritten with a fresh 0 every tick, so M6 was stuck at 0.000.
        """
        # Always increment the counter — regardless of log_level — because
        # the metric system depends on this for M6. If you want events
        # disabled, use log_level=1; the counter has negligible cost.
        self._spont_count_pending += 1
        if self.log_level >= 2:
            self.spontaneous_events.append({
                'tick': tick,
                'neuron_id': neuron_id,
                'membrane_potential': membrane_potential
            })

    def log_driven_firing(self, tick: int):
        """Increment driven firing counter."""
        # v148 — pending counter pattern (see log_spontaneous_event).
        self._driven_count_pending += 1

    def log_homeostatic_event(self, tick: int, neuron_id: int, old_value: float, 
                              new_value: float, activity: float):
        """Log homeostatic plasticity threshold adjustments."""
        if self.log_level >= 2:
            self.summary['total_homeostatic_adjustments'] += 1
            self.homeostatic_events.append({
                'tick': tick,
                'neuron_id': neuron_id,
                'old_threshold': old_value,
                'new_threshold': new_value,
                'activity_level': activity,
                'direction': 'up' if new_value > old_value else 'down'
            })

    def log_dendritic_spike_event(self, tick: int, neuron_id: int, branch_id: int,
                                   branch_potential: float, plateau_potential: float,
                                   ca_influx: float):
        """Log local dendritic spike events."""
        if self.log_level >= 2:
            self.summary['total_dendritic_spikes'] += 1
            self.dendritic_spike_events.append({
                'tick': tick,
                'neuron_id': neuron_id,
                'branch_id': branch_id,
                'branch_potential': branch_potential,
                'plateau_potential': plateau_potential,
                'ca_influx': ca_influx
            })

    def log_autoreceptor_event(self, tick: int, neuron_id: int, autoreceptor_value: float,
                               threshold_effect: float):
        """Log significant autoreceptor modulation events."""
        if self.log_level >= 2:
            self.autoreceptor_events.append({
                'tick': tick,
                'neuron_id': neuron_id,
                'autoreceptor_value': autoreceptor_value,
                'threshold_effect': threshold_effect
            })

    def log_neuromodulator_event(self, tick: int, modulator: str, level: float,
                                  crossed_threshold: str, effect: str):
        """Log neuromodulator threshold crossings (high/low affinity)."""
        if self.log_level >= 2:
            self.neuromodulator_events.append({
                'tick': tick,
                'modulator': modulator,
                'level': level,
                'crossed_threshold': crossed_threshold,
                'effect': effect
            })

    def log_phase_event(self, tick: int, event_type: str, phase_coherence: float,
                        details: dict = None):
        """Log phase synchronization events."""
        if self.log_level >= 2:
            self.phase_reset_events.append({
                'tick': tick,
                'event_type': event_type,
                'phase_coherence': phase_coherence,
                'details': details or {}
            })
            # Update peak coherence
            self.summary['peak_phase_coherence'] = max(
                self.summary['peak_phase_coherence'], phase_coherence)
    
    # ============================================================
    # NEW: Additional Event Logging Methods for Paper Validation
    # ============================================================
    
    def log_weight_evolution_event(self, tick: int, synapse_pre_id: int, synapse_post_id: int,
                                    w_fast_old: float, w_fast_new: float,
                                    w_slow_old: float, w_slow_new: float,
                                    w_meta_old: float, w_meta_new: float,
                                    details: dict = None):
        """Log significant weight changes across all three timescales."""
        if self.log_level >= 2:
            evt = {
                'tick': tick,
                'pre_id': synapse_pre_id,
                'post_id': synapse_post_id,
                'w_fast_delta': w_fast_new - w_fast_old,
                'w_slow_delta': w_slow_new - w_slow_old,
                'w_meta_delta': w_meta_new - w_meta_old
            }
            if details:
                evt.update(details)
            self.weight_evolution_events.append(evt)
    
    def log_threshold_modulation_event(self, tick: int, neuron_id: int,
                                        base_threshold: float, effective_threshold: float,
                                        ach_contribution: float, autoreceptor_contribution: float):
        """Log threshold modulation events (Paper Section 1)."""
        if self.log_level >= 2:
            self.summary['total_threshold_modulations'] += 1
            self.threshold_modulation_events.append({
                'tick': tick,
                'neuron_id': neuron_id,
                'base_threshold': base_threshold,
                'effective_threshold': effective_threshold,
                'ach_contribution': ach_contribution,
                'autoreceptor_contribution': autoreceptor_contribution,
                'total_modulation': effective_threshold - base_threshold
            })
    
    def log_associativity_event(self, tick: int, synapse_pre_id: int, synapse_post_id: int,
                                 own_delta_w: float, neighbor_contribution: float,
                                 final_delta_w: float):
        """Log associativity plasticity events (Paper Section 4 equation)."""
        if self.log_level >= 2:
            self.summary['total_associativity_events'] += 1
            self.associativity_events.append({
                'tick': tick,
                'pre_id': synapse_pre_id,
                'post_id': synapse_post_id,
                'own_delta_w': own_delta_w,
                'neighbor_contribution': neighbor_contribution,
                'final_delta_w': final_delta_w,
                'amplification_factor': final_delta_w / own_delta_w if own_delta_w != 0 else 0.0
            })
    
    def log_subthreshold_event(self, tick: int, neuron_id: int, membrane_potential: float,
                                threshold: float, distance_to_threshold: float):
        """Log subthreshold integration events (Paper Section 5)."""
        if self.log_level >= 2:
            self.summary['total_subthreshold_integrations'] += 1
            self.subthreshold_events.append({
                'tick': tick,
                'neuron_id': neuron_id,
                'membrane_potential': membrane_potential,
                'threshold': threshold,
                'distance_to_threshold': distance_to_threshold,
                'fraction_of_threshold': membrane_potential / threshold if threshold != 0 else 0.0
            })
    
    def log_itu_evolution_event(self, tick: int, circle_id: int, event_type: str,
                                 fitness_before: float, fitness_after: float,
                                 neurons_affected: int):
        """Log ITU/Aigarth evolution events (Paper Section 8)."""
        if self.log_level >= 2:
            if event_type == 'mutation':
                if self.time_series['itu_mutation_events']:
                    self.time_series['itu_mutation_events'][-1] += 1
            elif event_type == 'pruning':
                if self.time_series['itu_pruning_events']:
                    self.time_series['itu_pruning_events'][-1] += 1
    
    def get_event_lists(self) -> dict:
        """Helper to return references to all event lists."""
        return {
            'plasticity_events': self.plasticity_events,  # Missing
            'silent_synapse_events': self.silent_synapse_events,
            'spontaneous_events': self.spontaneous_events,
            'homeostatic_events': self.homeostatic_events,
            'dendritic_spike_events': self.dendritic_spike_events,
            'autoreceptor_events': self.autoreceptor_events,
            'neuromodulator_events': self.neuromodulator_events,
            'phase_reset_events': self.phase_reset_events,
            # NEW event lists
            'weight_evolution_events': self.weight_evolution_events,
            'threshold_modulation_events': self.threshold_modulation_events,
            'associativity_events': self.associativity_events,
            'subthreshold_events': self.subthreshold_events,
        }

    def clear_events(self):
        """Clears transient event lists (used in worker processes)."""
        for lst in self.get_event_lists().values():
            lst.clear()

    def merge_events(self, remote_events: dict):
        """Merges events from a worker process into the main logger."""
        local_lists = self.get_event_lists()
        for key, events in remote_events.items():
            if key in local_lists and events:
                local_lists[key].extend(events)
                # ADD: Count LTP/LTD from merged events
                if key == 'plasticity_events':
                    for evt in events:
                        if evt.get('type') == 'LTP':
                            self.summary['total_ltp_events'] += 1
                            if self.time_series['ltp_rate']:
                                self.time_series['ltp_rate'][-1] += 1
                        elif evt.get('type') == 'LTD':
                            self.summary['total_ltd_events'] += 1
                            if self.time_series['ltd_rate']:
                                self.time_series['ltd_rate'][-1] += 1

    def log_structural_event(self, tick: int, event_type: str, entity_id: int, details: dict = None):
        if event_type == 'synapse_created':
            self.summary['total_synapses_created'] += 1
        elif event_type == 'synapse_pruned':
            self.summary['total_synapses_pruned'] += 1
        elif event_type == 'neuron_created':
            self.summary['total_neurons_created'] += 1
        elif event_type == 'neuron_died':
            self.summary['total_neurons_died'] += 1
        if self.log_level >= 2:
            self.structural_events.append({
                'tick': tick,
                'type': event_type,
                'entity_id': entity_id,
                'details': details or {}
            })
    
    def log_nxer_event(self, tick: int, event_type: str, nxer_id: int, details: dict = None):
        if event_type == 'born':
            self.nxer_summary['total_born'] += 1
        elif event_type == 'died':
            self.nxer_summary['total_died'] += 1
        if self.log_level >= 2:
            self.nxer_events.append({
                'tick': tick,
                'type': event_type,
                'nxer_id': nxer_id,
                'details': details or {}
            })
    
    def log_io_pattern(self, tick: int, nxer_id: int, inputs: tuple, outputs: tuple):
        if self.log_level >= 2:
            if len(self.io_patterns) >= self.max_history_length:
                self.io_patterns = self.io_patterns[self.max_history_length // 10:]
            self.io_patterns.append({
                'tick': tick,
                'nxer_id': nxer_id,
                'inputs': list(inputs),
                'outputs': list(outputs)
            })
            
            # NEW v3.1: Track specific new input/output usage
            if len(inputs) >= 9:
                self.summary.setdefault('input_6_daynight_usage', {'night': 0, 'transition': 0, 'day': 0})
                self.summary.setdefault('input_7_temperature_usage', {'cold': 0, 'normal': 0, 'hot': 0})
                self.summary.setdefault('input_8_proprioception_usage', {'blocked': 0, 'normal': 0, 'clear': 0})
                
                if inputs[6] == -1: self.summary['input_6_daynight_usage']['night'] += 1
                elif inputs[6] == 1: self.summary['input_6_daynight_usage']['day'] += 1
                else: self.summary['input_6_daynight_usage']['transition'] += 1
                
                if inputs[7] == -1: self.summary['input_7_temperature_usage']['cold'] += 1
                elif inputs[7] == 1: self.summary['input_7_temperature_usage']['hot'] += 1
                else: self.summary['input_7_temperature_usage']['normal'] += 1
                
                if inputs[8] == -1: self.summary['input_8_proprioception_usage']['blocked'] += 1
                elif inputs[8] == 1: self.summary['input_8_proprioception_usage']['clear'] += 1
                else: self.summary['input_8_proprioception_usage']['normal'] += 1
            
            if len(outputs) >= 6:
                self.summary.setdefault('output_5_resting_usage', {'force_active': 0, 'normal': 0, 'rest': 0})
                if outputs[5] == -1: self.summary['output_5_resting_usage']['force_active'] += 1
                elif outputs[5] == 1: self.summary['output_5_resting_usage']['rest'] += 1
                else: self.summary['output_5_resting_usage']['normal'] += 1
    
    def update_nxer_stats(self, nxer: 'NxEr'):
        self.nxer_summary['max_food_found'] = max(self.nxer_summary['max_food_found'], nxer.stats.food_found)
        self.nxer_summary['max_time_lived'] = max(self.nxer_summary['max_time_lived'], nxer.stats.time_lived_s)
        self.nxer_summary['max_mates'] = max(self.nxer_summary['max_mates'], nxer.stats.mates_performed)
        self.nxer_summary['max_explored'] = max(self.nxer_summary['max_explored'], nxer.stats.explored)
       

    def log_v2_metrics(self, tick, receptor_activations, msth_state=None, chrono_stats=None):
        """Log Neuraxon v2.0 specific metrics."""
        if self.log_level < 2:
            return
        entry = {
            'tick': tick,
            'receptor_activations': receptor_activations,
        }
        if msth_state:
            entry['msth'] = msth_state
        if chrono_stats:
            entry['chrono'] = chrono_stats
        self.time_series.setdefault('v2_metrics', []).append(entry)

    def to_dict(self) -> dict:
        def sanitize(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0  # Replace NaN/Inf with 0.0
                return obj
            elif isinstance(obj, list):
                return [sanitize(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            return obj
        """Serialize all logged data to a dictionary with compression."""
        self.game_metadata['end_timestamp'] = datetime.now().isoformat()
        self.game_metadata['duration_seconds'] = time.time() - self.start_time
        self.game_metadata['log_level'] = self.log_level
        
        data = {
            'metadata': self.game_metadata,
            'summary': self.summary,
            'nxer_summary': self.nxer_summary
        }
        
        if self.log_level >= 2:
            # Set limit_logs based on log level
            if self.log_level >= 3:
                limit_logs = 1000000
            else:
                limit_logs = 50000
            
            serializable_synapse_snapshots = []
            for snapshot in self.synapse_snapshots:
                serializable_weights = {}
                for key, weights in snapshot.get('synapse_weights', {}).items():
                    k_str = f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)
                    serializable_weights[k_str] = weights
                serializable_synapse_snapshots.append({
                    'tick': snapshot['tick'],
                    'synapse_weights': serializable_weights
                })

            SPARSE_KEYS = {'alive', 'food_found', 'explored', 'mates_performed' }
            
            optimized_nxer_series = {}
            # Only include per_nxer_time_series at level 3
            if self.log_level >= 3:
                for nxer_id, series in self.per_nxer_time_series.items():
                    optimized_series = {}
                    for metric, values in series.items():
                        if metric in SPARSE_KEYS:
                            optimized_series[metric] = self._compress_series(values)
                        else:
                            optimized_series[metric] = values
                    optimized_nxer_series[nxer_id] = optimized_series

            data['level2'] = {                
                'time_series':sanitize(self.time_series),
                'per_nxer_time_series': sanitize(optimized_nxer_series),
                
                'plasticity_events': self.plasticity_events[-limit_logs:],
                'structural_events': self.structural_events[-limit_logs:],
                'neuron_snapshots': self.neuron_snapshots[-limit_logs:],
                'synapse_snapshots': serializable_synapse_snapshots[-limit_logs:],
                'nxer_events': self.nxer_events[-limit_logs:],
                'itu_fitness_history': self.itu_fitness_history[-limit_logs:],
                'io_patterns': self.io_patterns[-limit_logs:],
                'silent_synapse_events': self.silent_synapse_events[-limit_logs:],
                'spontaneous_events': self.spontaneous_events[-limit_logs:],
                'homeostatic_events': self.homeostatic_events[-limit_logs:],
                'dendritic_spike_events': self.dendritic_spike_events[-limit_logs:],
                'autoreceptor_events': self.autoreceptor_events[-limit_logs:],
                'neuromodulator_events': self.neuromodulator_events[-limit_logs:],
                'phase_reset_events': self.phase_reset_events[-limit_logs:],
                'weight_evolution_events': self.weight_evolution_events[-limit_logs:],
                'threshold_modulation_events': self.threshold_modulation_events[-limit_logs:],
                'associativity_events': self.associativity_events[-limit_logs:],
                'subthreshold_events': self.subthreshold_events[-limit_logs:],
            }
        
        # v145: always export the latest research-probe dashboard, regardless
        # of log level. This is a tiny dict (the dashboard summary, NOT the
        # full series — those are already inside time_series).
        try:
            data['research_dashboard_v145'] = self.get_research_dashboard()
        except Exception:
            data['research_dashboard_v145'] = {'available': False}
        return data
    
    def save_to_file(self, filepath: str):
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[DATALOGGER] Saved to {filepath}")


# ============================================================
# NEW: Helper function to generate paper validation report
# ============================================================
def generate_paper_validation_report(logger: DataLogger) -> dict:
    """
    Generate a structured report mapping logged data to paper sections.
    Useful for validating the implementation against the Neuraxon paper.
    """
    report = {
        'paper_section_1_trinary_neuromodulation': {
            'trinary_state_distributions': {
                'excitatory_samples': len([x for x in logger.time_series.get('excitatory_fraction', []) if x > 0]),
                'inhibitory_samples': len([x for x in logger.time_series.get('inhibitory_fraction', []) if x > 0]),
                'neutral_samples': len([x for x in logger.time_series.get('neutral_fraction', []) if x > 0]),
            },
            'neuromodulator_dynamics': {
                mod: {
                    'peak': logger.summary['neuromodulator_peaks'].get(mod, 0),
                    'events_logged': len([e for e in logger.neuromodulator_events if e.get('modulator') == mod])
                } for mod in ['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']
            },
            'threshold_modulation_events': logger.summary.get('total_threshold_modulations', 0),
        },
        'paper_section_2_temporal_dynamics': {
            'total_ticks': logger.summary['total_ticks'],
            'timestamps_logged': len(logger.time_series.get('timestamps', [])),
            'oscillator_components_tracked': all(
                len(logger.time_series.get(k, [])) > 0 
                for k in ['oscillator_low', 'oscillator_mid', 'oscillator_high']
            ),
        },
        'paper_section_3_synaptic_computation': {
            'weight_timescales_tracked': {
                'w_fast': len(logger.time_series.get('mean_w_fast', [])),
                'w_slow': len(logger.time_series.get('mean_w_slow', [])),
                'w_meta': len(logger.time_series.get('mean_w_meta', [])),
            },
            'synaptic_trace_dynamics': {
                'pre_trace': len(logger.time_series.get('mean_pre_trace', [])),
                'post_trace': len(logger.time_series.get('mean_post_trace', [])),
            },
            'ionotropic_vs_metabotropic': {
                'ionotropic_samples': len(logger.time_series.get('ionotropic_contribution_mean', [])),
                'metabotropic_samples': len(logger.time_series.get('metabotropic_contribution_mean', [])),
            },
        },
        'paper_section_4_plasticity': {
            'total_plasticity_events': logger.summary['total_plasticity_events'],
            'ltp_events': logger.summary['total_ltp_events'],
            'ltd_events': logger.summary['total_ltd_events'],
            'associativity_events': logger.summary.get('total_associativity_events', 0),
            'homeostatic_adjustments': logger.summary.get('total_homeostatic_adjustments', 0),
        },
        'paper_section_5_complex_signaling': {
            'silent_synapse_activations': logger.summary.get('total_silent_synapse_activations', 0),
            'subthreshold_integrations': logger.summary.get('total_subthreshold_integrations', 0),
        },
        'paper_section_6_self_generated_activity': {
            'spontaneous_events': logger.summary.get('total_spontaneous_events', 0),
            'autocorrelation_window_tracked': len(logger.time_series.get('mean_autocorrelation_window', [])) > 0,
            'peak_acw': logger.summary.get('peak_autocorrelation_window', 0),
        },
        'paper_section_7_synchronization': {
            'phase_coherence_tracked': len(logger.time_series.get('phase_coherence', [])),
            'peak_coherence': logger.summary.get('peak_phase_coherence', 0),
            'cfc_metrics_tracked': all(
                len(logger.time_series.get(k, [])) > 0 
                for k in ['cfc_low_mid', 'cfc_mid_high', 'pac_theta_gamma']
            ),
        },
        'paper_section_8_aigarth_hybrid': {
            'itu_fitness_tracked': len(logger.time_series.get('itu_mean_fitness', [])),
            'itu_fitness_history': len(logger.itu_fitness_history),
        },
        # ============================================================
        # v145 — RESEARCH DASHBOARD (M1-M10)
        # Maps the 10 paper-fidelity metrics to the multi-Neuraxon /
        # Neuraxon-v2.0 hypotheses they test (H1/H2/H3 + bio-fidelity).
        # ============================================================
        'research_v145_M1_trinary_distribution': {
            'last_E': (logger.time_series.get('M1_excitatory_fraction') or [0])[-1],
            'last_I': (logger.time_series.get('M1_inhibitory_fraction') or [0])[-1],
            'last_N': (logger.time_series.get('M1_neutral_fraction') or [0])[-1],
            'paper_target': 'E≈0.22, I≈0.10, N≈0.68 (Neuraxon v2.0 §I/§VII)',
            'in_band_E': bool((logger.time_series.get('M1_excitatory_fraction__in_band') or [0])[-1]),
        },
        'research_v145_M2_ctc_gate': {
            'last_mean_gate':       (logger.time_series.get('M2_mean_gate') or [0])[-1],
            'last_modulation_std':  (logger.time_series.get('M2_gate_modulation_std') or [0])[-1],
            'last_ff_fb_asymmetry': (logger.time_series.get('M2_ff_fb_asymmetry') or [0])[-1],
            'paper_claim': 'Disabling CTC saturates gate at 1.0 (Multi-Neuraxon §3.2 C1).',
        },
        'research_v145_M3_pac': {
            'last_pac_modulation_idx': (logger.time_series.get('M3_pac_modulation_idx') or [0])[-1],
            'last_pac_delta_theta':    (logger.time_series.get('M3_pac_delta_theta_idx') or [0])[-1],
            'paper_claim': 'Slow rhythms modulate fast amplitudes (Neuraxon v2.0 §VI).',
        },
        'research_v145_M4_weight_separation': {
            'last_temporal_divergence': (logger.time_series.get('M4_temporal_divergence') or [0])[-1],
            'last_w_meta_active_fraction': (logger.time_series.get('M4_w_meta_active_fraction') or [0])[-1],
            'paper_claim': 'τ_fast < τ_slow < τ_meta (Neuraxon v2.0 §III).',
        },
        'research_v145_M5_criticality': {
            'last_branching_ratio':       (logger.time_series.get('M5_branching_ratio') or [0])[-1],
            'last_distance_from_critical':(logger.time_series.get('M5_distance_from_critical') or [0])[-1],
            'paper_claim': 'Cortex sits near σ ≈ 1.0 criticality (Neuraxon v2.0 §V).',
        },
        'research_v145_M6_spontaneous': {
            'last_spontaneous_fraction': (logger.time_series.get('M6_spontaneous_fraction') or [0])[-1],
            'last_acw_heterogeneity':   (logger.time_series.get('M6_acw_heterogeneity') or [0])[-1],
            'paper_claim': 'Self-generated activity is a cornerstone (Neuraxon v2.0 §V).',
        },
        'research_v145_M7_self_sustained': {
            'last_zero_input_mi_ratio': (logger.time_series.get('M7_zero_input_mi_ratio') or [0])[-1],
            'last_broadcast_index':     (logger.time_series.get('M7_broadcast_index') or [0])[-1],
            'last_n_paired_nxers':      (logger.time_series.get('M7_n_paired_nxers') or [0])[-1],
            'paper_claim': 'No-input MI ≈ 1.96 vs Nengo collapse to 0.42 (Multi-Neuraxon §3.8).',
        },
        'research_v145_M8_specialisation': {
            'last_sensory_vs_assoc_dissociation': (logger.time_series.get('M8_sensory_vs_association_dissociation') or [0])[-1],
            'paper_claim': 'Sensory spheres develop modality selectivity (Multi-Neuraxon §3.5).',
        },
        'research_v145_M9_compositional': {
            'last_transfer_ratio':           (logger.time_series.get('M9_transfer_ratio') or [0])[-1],
            'last_compositional_similarity': (logger.time_series.get('M9_compositional_similarity') or [0])[-1],
            'paper_claim': 'Novel V×A combinations achieve ≥100% trained-pair MI (Multi-Neuraxon §3.7).',
        },
        'research_v145_M10_heritability_lesion': {
            'last_heritability_r':       (logger.time_series.get('M10_heritability_r') or [0])[-1],
            'last_heritability_pairs':   (logger.time_series.get('M10_heritability_pairs') or [0])[-1],
            'last_lesion_retention_50':  (logger.time_series.get('M10_lesion_retention_50') or [0])[-1],
            'last_lesion_retention_75':  (logger.time_series.get('M10_lesion_retention_75') or [0])[-1],
            'paper_claim': '99-102% MI under 75% neuron loss (Multi-Neuraxon §3.9); heritability r > 0.3 (Aigarth §VIII).',
        },
        'research_v145_dashboard_summary': logger.get_research_dashboard().get('summary', {}),
    }
    return report


# Global data logger instance
_data_logger: Optional[DataLogger] = None

def get_data_logger() -> DataLogger:
    """v156 (v4.64) — default `log_level=2` so the time_series and
    membrane-diag attributes are initialised on first construction.
    
    Background: prior to v156 this defaulted to log_level=1, which skipped
    `_init_level2_data()` entirely. The only path that bumped the logger
    to level 2 was the menu (`ui/menus.py` line 116 calls
    `set_data_logger_level(2)`). Any programmatic entry that bypassed the
    menu (NxonArchNAS subprocesses, future research scripts, headless
    benchmark runs) ended up with a logger that had no `time_series`
    attribute → AUTO-SAVE printed "no samples to write" → NAS trials
    returned `no_time_series` fitness = -1.
    
    The menu still explicitly calls `set_data_logger_level(2)` so its
    behaviour is unchanged; this just fixes the default path."""
    global _data_logger
    if _data_logger is None:
        _data_logger = DataLogger(log_level=2)
    return _data_logger

def set_data_logger_level(level: int):
    get_data_logger().set_level(level)
