"""Neuraxon Architecture Search (NAS) — v1.5 (Game of Life v5.10 / v196)

Runs many small Game-of-Life trials in parallel (multiprocessing), each
with a different architecture sampled from a search space. Each trial is
time-limited and headless (uses the v4.52 PERF #24 headless mode of
game_loop.run_game). After all trials, the best architectures by fitness
are saved as JSON files usable directly via --architecture or NEURAXON_ARCH.

This module is INDEPENDENT of the game in the sense that it doesn't
modify any game code — it just instantiates the game programmatically.
Two separate research tracks:
  (1) Game-of-Life bio-inspired code improvements (the v144-v156 series)
  (2) NAS to find sweet-spot architectures within the v(N) game.

v196 (v5.10) — SEARCH-METHOD UPGRADE. v161-v195 grew the search SPACE and the
fitness function; the search ALGORITHM stayed a fixed-knob evolutionary loop
with a binary "escape" toggle, selecting purely on the (median-aggregated) raw
fitness. 

Usage (CLI):
  # v196 default: evolutionary + adaptive controller + guided immigrants + LCB
  python NxonArchNAS.py --trials 16 --wall-seconds 60 --workers 4

  # concentrate the budget on the 12 high-leverage genes, pinned around best:
  python NxonArchNAS.py --search-space focus --seed-archs nas_best.json

  # compound a previous run's knowledge into a fresh search:
  python NxonArchNAS.py --resume nas_runs/<prev>/nas_log.csv

Usage (programmatic):
  from NxonArchNAS import run_search
  best = run_search(num_trials=8, wall_seconds=60, workers=4)
"""

import argparse
import copy
import csv
import json
import math            # v179 — module-level (fitness g-term uses exp/tanh)
import multiprocessing as mp
import os
import queue          # v160 — saver-thread result queue
import random
import sys
import threading     # v160 — saver thread + scheduler thread coordination
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# SEARCH SPACE — v162 (v4.70) CONTINUOUS RANGES + GENETIC LOTTERY
# =============================================================================
# Each entry maps a "section.key" path (as used in architecture JSON files)
# to one of three sample-spec types:
#
#   1. CONTINUOUS RANGE   ('uniform', min, max)            float
#   2. CONTINUOUS LOGRANGE('loguniform', min, max)         float (e.g. learning rates)
#   3. CONTINUOUS INT     ('int_uniform', min, max)        int
#   4. DISCRETE           [a, b, c, ...]                   choice from list
#
# v162 changes (responding to user feedback on 743-trial run):
#   - Fitness landscape was saturated on 6 discrete params (4.88-4.92 mean
#     across ALL parameter levels — no signal). M10 heritability=0 in
#     ALL 743 trials because every NxEr was born identical.
#   - Replace most discrete lists with continuous (min, max) ranges so
#     exploration isn't limited to 4-5 grid points per axis.
#   - Add GENETIC LOTTERY section: ranges from which each NxEr samples
#     its OWN value at birth. This creates per-NxEr diversity that
#     selection can act on — the path to non-zero M10 heritability.
#   - Add more wired parameters (firing thresholds, intrinsic timescale,
#     learning rate, autoreceptor coefficient) so NAS can explore them.
SEARCH_SPACE: Dict[str, Any] = {
    # -- BIOLOGY: game-world dynamics (kept biologically plausible) --
    'biology.metabolic_ramp_per_sec':           ('uniform', 2.0, 18.0),
    'biology.max_atrophy':                      ('uniform', 1.5, 15.0),
    'biology.metabolic_rate_abs_cap_multiple':  ('uniform', 8.0, 60.0),
    'biology.idle_explore_seconds':             ('uniform', 0.3, 2.5),
    'biology.explore_probability':              ('uniform', 0.2, 0.8),
    'biology.mate_cooldown_seconds':            ('int_uniform', 6, 20),
    'biology.circadian_cycle_ticks':            ('int_uniform', 300, 1200),
    
    # -- NEURAL: network topology --
    'neural.num_hidden_neurons_default':        ('int_uniform', 6, 24),
    'neural.connection_probability':            ('uniform', 0.15, 0.45),
    'neural.afferent_synapse_strength':         ('uniform', 0.6, 1.8),
    # v195 (v5.10) — FLOOR lowered 0.40 → 0.32. The v194 8h NAS top
    # architectures pressed against the old 0.40 floor (winners at
    # thr_exc 0.42–0.49), i.e. the search WANTED lower excitatory
    # thresholds to lift M1 but couldn't. 0.32 opens the higher-M1 region
    # so excitation can stay near the paper's 0.22 even when a rest
    # mechanism is active. The M1 tent (peak 0.22) still penalises the
    # over-firing that very low thresholds would cause, so the search
    # self-limits.
    'neural.firing_threshold_excitatory':       ('uniform', 0.32, 0.70),
    # v188 (v5.03) — firing_threshold_inhibitory enters the search space.
    # Pre-v188 it was hardcoded at -0.55 across every architecture (and
    # every NAS run). The v187 8h NAS data exposed the structural cost:
    # with thr_exc searched in [0.40, 0.70] but |thr_inh| pinned at 0.55,
    # the average architecture had thr_exc > |thr_inh|, so membrane drive
    # distributed around 0 crossed the inhibitory threshold more easily
    # than the excitatory one. After v187 loosened the brakes (and the
    # network finally started firing), the trinary distribution inverted:
    # state=-1 went from 0.5% (v186) to 22.3% (v187), state=+1 went from
    # 7.6% to 0.5%. M1 (which only counts +1) couldn't climb because every
    # spike was being routed to the wrong polarity.
    #
    # Range mirrors the excitatory threshold ([-0.70, -0.40] = the
    # negative of [0.40, 0.70]) so the search can discover the symmetric
    # configuration |thr_inh| ≈ thr_exc. Asymmetric configurations
    # (intentional inhibitory bias for specific topologies) are still
    # reachable — this just stops the search being FORCED into them.
    # v195 (v5.10) — RANGE DEEPENED -0.70 → -1.20 (lower bound). v188's
    # symmetric range ([-0.70, -0.40], mirroring thr_exc) was meant to let
    # the search balance E/I, but the v194 8h NAS proved it insufficient:
    # ALL 31 trials were still inhibition-dominant (inh > M1), median
    # 0.195/0.172/0.550 (+1/0/-1). The cause is intrinsic — the network's
    # recurrent dynamics bias membrane potential NEGATIVE, so even at the
    # deepest available threshold (-0.70) negative excursions still cross
    # into state=-1 rather than resting. corr(thr_inh, inh)=+0.34 in the
    # v194 data confirms deeper thresholds reduce inhibition. To actually
    # counter the bias the search needs ASYMMETRIC thresholds
    # (|thr_inh| > thr_exc): a deep inhibitory threshold converts negative
    # mp excursions into REST (state 0) instead of -1 firing — raising the
    # rest band and lowering inhibition WITHOUT capping the firing rate
    # (the way refractory/AHP do, which is why those stay capped at 5/0.5
    # per v187). This is the primary lever for de-inverting the trinary
    # distribution toward the paper's excitation-dominant 0.22/0.68/0.10.
    # The inh tent (target 0.10, not 0) still penalises losing inhibition
    # entirely, so the search finds the balance rather than thr_inh → -∞.
    'neural.firing_threshold_inhibitory':       ('uniform', -1.20, -0.40),
    'neural.spontaneous_firing_rate':           ('loguniform', 0.005, 0.080),
    'neural.intrinsic_timescale_default':       ('uniform', 8.0, 30.0),
    'neural.resting_potential_decay':           ('uniform', 0.10, 0.35),
    # v164 — sensorimotor_coupling: multiplier on input→output edge prob.
    # 0 = pre-v164 (uniform random); 3 = 4× more sensory→motor synapses.
    # Designed to unblock the sm_corr fitness component (max 0.125 in
    # 1425 v163 trials at 900s — the only unsaturated fitness piece).
    'neural.sensorimotor_coupling':             ('uniform', 0.0, 3.0),
    # v169 (v4.77) — opt-in for MultiNeuraxon2 Bug #3 fix. Binary search:
    # False preserves v161-v168 asymmetric STDP. True enables signed traces
    # and (-1,-1)/(-1,+1) branches. If symmetric STDP improves fitness on
    # average, evolutionary search will concentrate sampling on True.
    'neural.symmetric_stdp':                    [False, True],
    # v171 (v4.79) — refractory period after firing (state 0 buffer).
    # 0 = no refractory (v161-v170 behaviour). Higher values force more
    # time at state=0 between firings, restoring the paper's intended
    # trinary dynamics.
    # v187 (v5.02) — RANGE TIGHTENED from [0, 12] to [0, 5]. The v186 8h
    # NAS picked refract=10-12 on its top architectures, which combined
    # with high AHP and slow intrinsic_timescale (~25) gave a per-spike
    # dead time of ~35 ticks. That mathematically caps the firing rate
    # at ~2.9% — making M1=0.22 unreachable. 0-5 covers the biologically
    # realistic range from the v171 comment ("biologically realistic
    # values are 1-5 (~ 1-5 game ticks at 10Hz)") and forbids the
    # extreme suppression regime the search kept converging on.
    'neural.refractory_period_ticks':           ('int_uniform', 0, 5),
    # v172 (v4.80) — after-hyperpolarization (mp reset). Pulls mp toward
    # 0 after firing. With v171 refractory, this is what makes state=0
    # an ACTUAL biological buffer (rather than just briefly forced).
    # v187 (v5.02) — RANGE TIGHTENED from [0.0, 1.0] to [0.0, 0.5]. The
    # v186 8h NAS top architectures used AHP=0.87, snapping mp to ~13%
    # of threshold after every spike. Combined with slow intrinsic_
    # timescale, recovery to threshold took ~25 ticks. 0.0-0.5 still
    # allows meaningful after-hyperpolarisation (50% snap-back is biolo-
    # gically substantial) but rules out the "near-total-reset" regime
    # that suppressed firing into quiescence. The product (refract * AHP)
    # is now bounded at 5 * 0.5 = 2.5 — well clear of the 8.7 the v186
    # best architecture used.
    'neural.post_spike_mp_reset':               ('uniform', 0.0, 0.5),
    
    # -- OPERATING RANGES: plasticity / adaptation --
    'operating_ranges.learning_rate':           ('loguniform', 0.002, 0.05),
    'operating_ranges.plasticity_threshold':    ('uniform', 0.3, 0.7),
    'operating_ranges.autoreceptor_coefficient':('uniform', 0.05, 0.25),
    'operating_ranges.adaptation_tau_ticks':    ('uniform', 10.0, 50.0),
    # v191 (v5.06) — E/I ADAPTATION-BALANCE lever enters the search space.
    # Pre-v191 these were hardcoded at 1.5 (excitatory) / 1.0 (inhibitory)
    # in neuron.py and IGNORED the values documented in default.json /
    # architecture.py (a "documented-but-not-wired" bug v191 fixes). The
    # 1.5× excitatory factor was a v149 workaround for a +1 lock-in that
    # the v171 refractory + v172 AHP machinery now handles structurally —
    # so the residual 1.5× now just suppresses +1 firing relative to -1,
    # the root cause of the inhibitory skew + sub-target M1 (~0.15 vs 0.22)
    # the v190 8h NAS exposed. Searching [0.8, 2.0] lets the optimiser
    # LOWER the excitatory brake (toward 1.0 or below) so +1 reaches the
    # paper target while refractory/AHP holds the 68% rest band, and tune
    # the inhibitory brake to pull inhibitory firing toward the paper's
    # 10%. This is the dynamics lever the v190 analysis identified as the
    # missing piece — fitness tuning alone could not resolve the M1-vs-rest
    # tension (corr -0.45 in the v190 run). Default 1.5 / 1.0 reproduces
    # v190 dynamics exactly when these keys are absent.
    'operating_ranges.adaptation_target_excitatory_multiplier': ('uniform', 0.8, 2.0),
    'operating_ranges.adaptation_target_inhibitory_multiplier': ('uniform', 0.8, 2.0),

    # -- v179 (v4.87) CHC SIX-SPHERE TOPOLOGY + g-FACTOR --
    # Adapts an upcoming paper's six-sphere design. sphere_topology
    # toggles the paper's 6-sphere build; κ/λc/βF are the paper's dominant
    # architectural levers (κ = cross-sphere coupling, λc = crystallised
    # capacity, βF = CTC coherence strength). fitness_g_weight is the
    # weight of the population g-factor in fitness.
    #
    # v190 (v5.05) SEARCH-SPACE FOCUS — two changes from v189:
    #   1. sphere_topology LOCKED to ['chc6']. The v189 8h run sampled both
    #      topologies (~50/50) but sensory_association_motor cleared the
    #      floors in only 7/67 trials (10%) vs chc6's 30/96 (31%), and its
    #      best fitness (6.003) trailed chc6's (6.422). chc6 is the paper's
    #      six-sphere build and the source of the strong positive manifold
    #      we want, so the whole trial budget now goes to it — no waste on
    #      the losing topology.
    #   2. fitness_g_weight PINNED at 1.5 (was sampled uniform 0..1). With a
    #      random per-trial weight, g was inconsistent selection pressure —
    #      a strong-g architecture that drew g_weight≈0.05 got ~0 g credit
    #      and missed the top. Fixing it at 1.5 makes g a genuine, CO-EQUAL
    #      objective alongside the two trinary components (m1 1.5 + neutral
    #      1.5): every trial is now rewarded up to 1.5 for a HEALTHY positive
    #      manifold (PC1≈0.27, positive mean-r — see the g_health block in
    #      fitness()). Trinary fidelity, survival, heritability, sm-coupling
    #      and the lock-in penalty are all unchanged and still apply.
    # See docs/G_FACTOR_METHODOLOGY.md and CHANGELOG_v179 / CHANGELOG_v190.
    'neural.sphere_topology':                   ['chc6'],
    'neural.cross_sphere_coupling':             ('uniform', 0.0, 3.0),
    'neural.cryst_capacity':                    ('uniform', 0.5, 2.5),
    'neural.free_energy_beta':                  ('uniform', 0.3, 2.0),
    'operating_ranges.fitness_g_weight':        1.5,   # v190 — pinned (was uniform 0..1)
    
    # -- GENETIC LOTTERY: per-NxEr trait diversity at birth (v162) --
    # Each NxEr at birth samples its OWN value from these ranges. This
    # creates the per-individual variation that selection can act on
    # (the missing ingredient that kept M10=0 in 743 trials of v161).
    # The NAS samples a (min, max) PAIR for each lottery dimension:
    # the architecture defines the SPREAD around the population mean.
    'genetic_lottery.metabolic_rate_multiplier_lo':  ('uniform', 0.6, 0.95),
    'genetic_lottery.metabolic_rate_multiplier_hi':  ('uniform', 1.05, 1.6),
    'genetic_lottery.intrinsic_timescale_jitter':    ('uniform', 0.0, 8.0),
    'genetic_lottery.firing_threshold_jitter':       ('uniform', 0.0, 0.15),
    'genetic_lottery.mutation_strength':             ('uniform', 0.02, 0.15),
}


# =============================================================================
# SEARCH-SPACE TIERS — v196 (v5.10) FOCUS / CORE SUBSETS
# =============================================================================
# v161-v195 always searched the FULL 35-key space. The v195 8h run (216 trials)
# plus a per-parameter signature analysis (the Pearson correlation of each
# searched gene against fitness, against the E/I-ordering margin M1−inh, and
# against the L1 distance to the paper trinary mix 0.22/0.68/0.10) showed the
# outcome is dominated by a small handful of genes; the rest are near-neutral
# noise dimensions that dilute the search. The FOCUS tier is those high-leverage
# genes — searched freely while everything else is PINNED at a known-good
# architecture (a --seed-archs JSON if given, else BEST_KNOWN_PINS below) — so
# the whole budget concentrates on the levers that actually move the metrics.
#
# Why each FOCUS gene (|r| from the v195 signature analysis, strongest first):
#   refractory_period_ticks                 fitness +0.76, paper-dist −0.94  (dominant rest-band lever)
#   firing_threshold_excitatory             E/I −0.73, fitness −0.40         (the real E/I lever)
#   num_hidden_neurons_default              fitness +0.38, E/I +0.34
#   connection_probability                  fitness +0.35, E/I +0.22
#   metabolic_rate_abs_cap_multiple         fitness +0.30                    (survival headroom)
#   resting_potential_decay                 fitness −0.33, paper-dist +0.27  (resting attractor — v196 direction)
#   intrinsic_timescale_default             fitness +0.23                    (recovery timescale)
#   adaptation_tau_ticks                    fitness +0.21, E/I +0.19
#   adaptation_target_excitatory_multiplier fitness +0.18                    (E/I balance lever)
#   adaptation_target_inhibitory_multiplier fitness −0.18, E/I −0.28         (E/I balance lever)
#   firing_threshold_inhibitory             fitness +0.20                    (v195's asymmetry lever — kept in)
#   post_spike_mp_reset                     fitness −0.14, E/I −0.13         (AHP — rest-band/firing budget)
# (circadian_cycle_ticks and metabolic_ramp_per_sec also correlate but are
# WORLD-dynamics confounds rather than brain levers, so they stay pinned in
# focus mode — see CHANGELOG_v196 "the 12 parameters". Adjust FOCUS_KEYS to
# re-scope; everything is a one-line edit.)
FOCUS_KEYS = [
    'neural.refractory_period_ticks',
    'neural.firing_threshold_excitatory',
    'neural.firing_threshold_inhibitory',
    'neural.num_hidden_neurons_default',
    'neural.connection_probability',
    'neural.resting_potential_decay',
    'neural.intrinsic_timescale_default',
    'neural.post_spike_mp_reset',
    'operating_ranges.adaptation_tau_ticks',
    'operating_ranges.adaptation_target_excitatory_multiplier',
    'operating_ranges.adaptation_target_inhibitory_multiplier',
    'biology.metabolic_rate_abs_cap_multiple',
]

# CORE = the six tightest levers (the trinary-distribution + E/I machinery only),
# for the very end-game when the survival/topology genes are already settled.
CORE_KEYS = [
    'neural.refractory_period_ticks',
    'neural.firing_threshold_excitatory',
    'neural.firing_threshold_inhibitory',
    'neural.post_spike_mp_reset',
    'operating_ranges.adaptation_target_excitatory_multiplier',
    'operating_ranges.adaptation_target_inhibitory_multiplier',
]

# Default pin values for the genes a tier does NOT search (used when no
# --seed-archs is supplied). These are the v195 leaderboard #1 (the strongest
# all-round architecture the prior search found): a healthy rest band, real
# heritability, full survival. Pinning here keeps focus-mode candidates near a
# known-viable brain so the population floor stays high.
BEST_KNOWN_PINS: Dict[str, Any] = {
    'biology.metabolic_ramp_per_sec': 18.0,
    'biology.max_atrophy': 11.236,
    'biology.metabolic_rate_abs_cap_multiple': 56.949,
    'biology.idle_explore_seconds': 1.136,
    'biology.explore_probability': 0.693,
    'biology.mate_cooldown_seconds': 9,
    'biology.circadian_cycle_ticks': 516,
    'neural.num_hidden_neurons_default': 24,
    'neural.connection_probability': 0.4345,
    'neural.afferent_synapse_strength': 0.7319,
    'neural.firing_threshold_excitatory': 0.3577,
    'neural.firing_threshold_inhibitory': -0.8835,
    'neural.spontaneous_firing_rate': 0.0631,
    'neural.intrinsic_timescale_default': 23.191,
    'neural.resting_potential_decay': 0.1662,
    'neural.sensorimotor_coupling': 2.325,
    'neural.symmetric_stdp': True,
    'neural.refractory_period_ticks': 5,
    'neural.post_spike_mp_reset': 0.2625,
    'neural.sphere_topology': 'chc6',
    'neural.cross_sphere_coupling': 1.379,
    'neural.cryst_capacity': 1.566,
    'neural.free_energy_beta': 1.229,
    'operating_ranges.learning_rate': 0.036,
    'operating_ranges.plasticity_threshold': 0.5075,
    'operating_ranges.autoreceptor_coefficient': 0.1622,
    'operating_ranges.adaptation_tau_ticks': 49.902,
    'operating_ranges.adaptation_target_excitatory_multiplier': 1.4114,
    'operating_ranges.adaptation_target_inhibitory_multiplier': 0.9091,
    'operating_ranges.fitness_g_weight': 1.5,
    'genetic_lottery.metabolic_rate_multiplier_lo': 0.838,
    'genetic_lottery.metabolic_rate_multiplier_hi': 1.378,
    'genetic_lottery.intrinsic_timescale_jitter': 5.446,
    'genetic_lottery.firing_threshold_jitter': 0.1205,
    'genetic_lottery.mutation_strength': 0.0808,
}


def _spec_bounds(spec: Any) -> Optional[Tuple[float, float]]:
    """Return (lo, hi) numeric bounds for a continuous spec, else None."""
    if (isinstance(spec, tuple) and len(spec) == 3
            and spec[0] in ('uniform', 'loguniform', 'int_uniform')):
        return float(spec[1]), float(spec[2])
    return None


# Which searched keys are continuous (gaussian-jittered) vs categorical/int.
_CATEGORICAL_KEYS = {k for k, spec in SEARCH_SPACE.items() if isinstance(spec, list)}
_INT_KEYS = {k for k, spec in SEARCH_SPACE.items()
             if isinstance(spec, tuple) and spec and spec[0] == 'int_uniform'}
_CONTINUOUS_KEYS = {k for k, spec in SEARCH_SPACE.items()
                    if _spec_bounds(spec) is not None and k not in _INT_KEYS}


def space_for_tier(tier: str) -> Dict[str, Any]:
    """Return the active search-space sub-dict for a tier name.

    'full'  (default) — the whole 35-key SEARCH_SPACE (v161-v195 behaviour).
    'focus' — only FOCUS_KEYS (the ~12 high-leverage genes); the rest pinned.
    'core'  — only CORE_KEYS (the six trinary/E-I levers); the rest pinned.

    Unknown/None tier falls back to 'full'.
    """
    tier = (tier or 'full').lower().strip()
    if tier == 'focus':
        keys = [k for k in FOCUS_KEYS if k in SEARCH_SPACE]
    elif tier == 'core':
        keys = [k for k in CORE_KEYS if k in SEARCH_SPACE]
    else:
        keys = list(SEARCH_SPACE.keys())
    return {k: SEARCH_SPACE[k] for k in keys}


def pins_for_space(space: Dict[str, Any],
                   seed_arch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fixed {full_key: value} for every SEARCH_SPACE gene NOT in `space`.

    Values come from `seed_arch` (a loaded architecture JSON) when it supplies
    them, else from BEST_KNOWN_PINS. Genes the active `space` searches are NOT
    pinned (they vary freely). The genetic_lottery lo/hi pair is handled by the
    sampler's range-collapse, so we pin the collapsed range when present.
    """
    pins: Dict[str, Any] = {}
    for full_key in SEARCH_SPACE:
        if full_key in space:
            continue
        val = None
        if seed_arch is not None:
            section, key = full_key.split('.', 1)
            val = seed_arch.get(section, {}).get(key)
        if val is None:
            val = BEST_KNOWN_PINS.get(full_key)
        if val is not None:
            pins[full_key] = val
    return pins


def _sample_one(rng: random.Random, spec: Any) -> Any:
    """v162 — sample one value from a SEARCH_SPACE entry.
    
    Supports: ('uniform', a, b), ('loguniform', a, b), ('int_uniform', a, b),
    or a plain list [a, b, c, ...].
    """
    if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], str):
        kind, lo, hi = spec
        if kind == 'uniform':
            return rng.uniform(lo, hi)
        elif kind == 'loguniform':
            import math
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))
        elif kind == 'int_uniform':
            return rng.randint(int(lo), int(hi))
        else:
            raise ValueError(f"Unknown sample kind: {kind}")
    elif isinstance(spec, (list, tuple)):
        return rng.choice(spec)
    else:
        # Scalar — return as-is (allows fixing one parameter)
        return spec


# Rounding for display purposes (the actual sampled value keeps full precision)
def _round_for_display(v: Any) -> Any:
    if isinstance(v, float):
        if abs(v) >= 100: return round(v, 1)
        if abs(v) >= 1:   return round(v, 3)
        return round(v, 5)
    return v


# =============================================================================
# WORLD CONFIG FOR NAS TRIALS
# =============================================================================
# Small world, few NxErs, long FoodRespan — designed to make trials
# fast (1-3 min) while still exercising the key dynamics. These are NOT
# search-space dimensions — they're held fixed so trials are comparable.
NAS_TRIAL_CONFIG = {
    'NxWorldSize':       60,
    'NxWorldSea':        0.4,
    'NxWorldRocks':      0.05,
    'StartingNxErs':     10,
    'MaxNxErs':          40,
    'MaxFood':           60,
    'FoodRespan':        300,
    'StartFood':         60.0,
    'MaxNeurons':        12,
    'GlobalTimeSteps':   60,
    'DayNightCycle':     2400,
    'MateCooldownSeconds': 10,
    'auto_start':        True,    # don't wait for SPACE
    'auto_save':         False,   # no auto-save side files
    'save_full_logs':    False,
    # v186 (v5.01) — pin max_rounds=1 so each trial is ONE continuous round.
    # Background: prior to v186 max_rounds was unset (default None →
    # infinite). If a trial's NxErs all died mid-trial, the game
    # auto-restarted with champions; restart_game_with_champions() calls
    # data_logger.reset() which wipes time_series. The trial's fitness
    # then reflected only whatever happened AFTER the most recent restart,
    # NOT the full 8h trial budget. This was invisible from outside
    # because there was no per-trial round count in the log.
    # With max_rounds=1 the trial ends naturally at the first extinction
    # (recorded as went_extinct=True / extinction_tick=T in the NAS log)
    # and that's an honest signal of a fragile architecture. Robust
    # architectures keep some population alive and run the full budget.
    'max_rounds':        1,
}


# v193 (v5.08) — SEED-AVERAGED TRIALS (the noise fix). The v161–v192
# longitudinal analysis showed single-trial metrics are noisy estimators of an
# architecture's behaviour: trial 4's IDENTICAL architecture read M1=0.086 in
# the v191 run and M1=0.140 in the v192 run (a 63% swing) purely from the game
# seed, and its neutral fraction moved 0.797→0.668. Selection on a single
# stochastic sample means rankings closer than ~0.5 fitness are partly luck.
# v193 evaluates each sampled architecture NAS_TRIAL_REPEATS times with
# distinct, well-separated seeds and selects on the per-metric MEDIAN (robust
# to a single crashed/outlier run), logging the per-seed spread (M1_std,
# fitness_std) so the noise — and each architecture's reliability — is visible.
# Cost: the wall budget buys NAS_TRIAL_REPEATS× fewer DISTINCT architectures,
# but each is measured robustly. Tune via --repeats or NEURAXON_NAS_REPEATS;
# set to 1 to recover exact v192 single-trial behaviour.
NAS_TRIAL_REPEATS = 3

# v194 (v5.09) — how the per-architecture wall budget relates to the repeats.
#   'subdivide' (DEFAULT): the wall the user sets is the budget PER ARCHITECTURE;
#       each of the R repeats runs wall/R. Cost-neutral vs a single v192 trial of
#       the same wall — the NAS explores the SAME number of architectures, each
#       measured as R shorter independent runs (median'd). This is the fix for
#       the v193.0 throughput collapse, where 'multiply' made each architecture
#       R× more expensive and a single long-running architecture could consume
#       the whole budget (the first v193 8h run logged exactly ONE trial: its 3
#       repeats summed to 10.3 h).
#   'multiply': each repeat runs the FULL wall (R× per-architecture cost). Use
#       only for deep single-architecture characterisation, not for search.
NAS_REPEATS_MODE = 'subdivide'


def _get_trial_repeats() -> int:
    """Number of seed-repeats per architecture (>=1). Env override:
    NEURAXON_NAS_REPEATS. Falls back to the module default."""
    import os as _os
    try:
        r = int(_os.environ.get('NEURAXON_NAS_REPEATS', NAS_TRIAL_REPEATS))
        return max(1, r)
    except (TypeError, ValueError):
        return max(1, NAS_TRIAL_REPEATS)


def _get_repeats_mode() -> str:
    """'subdivide' (default) or 'multiply'. Env: NEURAXON_NAS_REPEATS_MODE."""
    import os as _os
    m = _os.environ.get('NEURAXON_NAS_REPEATS_MODE', NAS_REPEATS_MODE)
    return m if m in ('subdivide', 'multiply') else 'subdivide'



# =============================================================================
# ARCHITECTURE GENERATION
# =============================================================================
def sample_random_architecture(rng: random.Random,
                                  trial_id: int,
                                  space: Optional[Dict[str, Any]] = None,
                                  pins: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sample one architecture from SEARCH_SPACE.
    
    v162 — supports continuous (min, max) ranges, log-uniform, integer
    uniform, and discrete lists. Also normalizes genetic_lottery key
    pairs so the `lo` value is always < the `hi` value (the NAS samples
    them independently, but downstream code expects an ordered pair).

    v196 — `space` restricts which genes are SAMPLED (a tier sub-dict from
    space_for_tier); `pins` provides fixed values for every gene NOT in
    `space` (from pins_for_space). With both None this is exactly the
    v162-v195 full-space behaviour.
    """
    space = space if space is not None else SEARCH_SPACE
    pins = pins or {}
    arch: Dict[str, Any] = {
        '_meta': {
            'source': 'NxonArchNAS',
            'trial_id': trial_id,
            'sampled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'biology': {}, 'neural': {}, 'operating_ranges': {},
        'healthy_bands': {}, 'genetic_lottery': {},
    }
    # 1) Pin the inactive genes first (so an active sample can still override).
    for full_key, val in pins.items():
        section, key = full_key.split('.', 1)
        arch.setdefault(section, {})[key] = val
    # 2) Sample the active genes.
    for full_key, spec in space.items():
        section, key = full_key.split('.', 1)
        arch.setdefault(section, {})[key] = _sample_one(rng, spec)
    
    # v162 — collapse the metabolic_rate_multiplier lo/hi pair into a
    # single [lo, hi] list so the game code sees a clean range. Same
    # for any other future lottery pairs.
    lot = arch.get('genetic_lottery', {})
    lo = lot.pop('metabolic_rate_multiplier_lo', None)
    hi = lot.pop('metabolic_rate_multiplier_hi', None)
    if lo is not None and hi is not None:
        lot['metabolic_rate_multiplier_range'] = [min(lo, hi), max(lo, hi)]
    
    return arch


def arch_summary_string(arch: Dict[str, Any]) -> str:
    """Short one-line summary of varied parameters (for logging).
    v162 — includes genetic_lottery section."""
    parts = []
    for section in ('biology', 'neural', 'operating_ranges', 'genetic_lottery'):
        for k, v in arch.get(section, {}).items():
            disp = _round_for_display(v) if not isinstance(v, list) else v
            parts.append(f"{k}={disp}")
    return " ".join(parts)


# =============================================================================
# SEARCH STRATEGIES — v164 (v4.72)
# =============================================================================
# Random search worked for v161-v163: it found M10 heritability, broke the
# 5.10 fitness ceiling, and reached 6.41 across 1425 trials. But random
# search wastes most of its budget sampling regions far from any local
# optimum. With multiple peaks now visible in the landscape (trial 394 at
# fitness 6.41 with M10=0.96 from a small high-threshold network, trial
# 840 at 6.07 from a large low-threshold network), an evolutionary search
# can exploit the discovered peaks while still exploring globally.
#
# Strategies have the same signature: given trial_id + archive of past
# results, sample one architecture. The scheduler stays unchanged.

# =============================================================================
# VARIANCE-AWARE SELECTION + E/I-ORDERING REWARD — v196 (v5.10)
# =============================================================================
# v193 made each architecture's metrics a MEDIAN over seed-repeats (robust to a
# single outlier run). v196 takes the next step: rank by a LOWER-CONFIDENCE
# BOUND on fitness, not the raw median fitness. The v195 8h leaderboard #1 (trial
# 191, fitness 9.40) had fitness_std_reps≈0.57 across its own three repeats, and
# several top-10 trials had std up to ~0.88 — selecting on the raw value rewards
# the lucky draw. The LCB central−k·stderr pulls high-variance spikes BELOW
# stable architectures with a slightly lower mean but a tight spread, so the
# search converges on a reproducible basin rather than an isolated needle. The
# logged `fitness` column is unchanged (still the median-metric fitness); only
# the RANKING uses the bound, and k=0 recovers exact v195 selection.

# Default confidence-bound strength (≈ one standard error). Tunable via
# --select-k / NEURAXON_NAS_SELECT_K.
NAS_SELECT_K = 1.0

# Default weight of the E/I-ordering selection bonus (see _ei_ordering_bonus).
# Tunable via --ei-weight / NEURAXON_NAS_EI_WEIGHT; 0 disables it.
NAS_EI_WEIGHT = 0.5


def _get_select_k() -> float:
    try:
        return float(os.environ.get('NEURAXON_NAS_SELECT_K', NAS_SELECT_K))
    except (TypeError, ValueError):
        return NAS_SELECT_K


def _get_ei_weight() -> float:
    try:
        return float(os.environ.get('NEURAXON_NAS_EI_WEIGHT', NAS_EI_WEIGHT))
    except (TypeError, ValueError):
        return NAS_EI_WEIGHT


def _ei_ordering_bonus(metrics: Optional[Dict[str, Any]], weight: float) -> float:
    """v196 — substrate-free reward for EXCITATION-DOMINANT ordering (M1 >= inh).

    The v191 trinary triad scores M1 (+1, target 0.22), neutral (0, 0.68) and
    inhibitory (−1, 0.10) with three tents centred on those targets. Those tents
    already MILDLY prefer excitation-dominance — but only weakly: the v195 search
    nailed neutral≈0.69 yet settled at M1≈inh≈0.15, inhibition-dominant in
    199/216 trials, because near that balanced sub-target operating point the two
    tents separate the orders by only ~0.14 fitness, which the substrate's
    inhibition bias plus the survival/neutral advantages of the inverted corner
    easily outweigh. This term adds an explicit, TUNABLE ordering lever (the one
    the v195 changelog recommended for v196): full `weight` when the network is
    excitation-dominant at the paper margin (M1 − inh ≥ +0.12), zero when it is
    inhibition-dominant (M1 − inh ≤ 0), linear between — strongest precisely
    where fitness discriminates least, nudging the search off the inverted corner.

    Applied ONLY to the selection score (not to fitness()), so fitness()'s 13.0
    ceiling and every v187-v195 fitness test stay byte-for-byte identical.
    """
    if not metrics or weight <= 0.0:
        return 0.0
    m1 = float(metrics.get('M1_last', 0.0))
    inh = float(metrics.get('M1_inh_last', 0.0))
    margin = m1 - inh
    PAPER_MARGIN = 0.12          # 0.22 − 0.10
    frac = max(0.0, min(1.0, margin / PAPER_MARGIN))
    return weight * frac


def selection_score(result: Dict[str, Any],
                    k: Optional[float] = None,
                    ei_weight: Optional[float] = None) -> float:
    """v196 — the value the search RANKS by (elites, global-best, top-3).

    selection_score = fitness − k · stderr(fitness over repeats)   [LCB]
                      + E/I-ordering bonus                          [v196 fix]

    `result` is a finished-trial dict (has 'fitness' and 'metrics'). Failed
    trials (fitness ≤ −1) keep their fitness so they sort last. Falls back to
    raw fitness when the per-repeat spread is unavailable (single-seed runs).
    """
    central = float(result.get('fitness', -1.0))
    if central <= -1.0:
        return central
    k = _get_select_k() if k is None else k
    ei_weight = _get_ei_weight() if ei_weight is None else ei_weight
    m = result.get('metrics') or {}
    n = max(1.0, float(m.get('n_repeats_ok', m.get('n_repeats', 1.0)) or 1.0))
    sd = float(m.get('fitness_std_reps', 0.0) or 0.0)
    lcb = central - k * sd / (n ** 0.5)
    lcb += _ei_ordering_bonus(m, ei_weight)
    return lcb


class GuidedSampler:
    """v196 — online estimation-of-distribution (EDA) layer that learns WHAT THE
    BEST ARCHITECTURES LOOK LIKE and biases the search's "random" immigrants
    toward it, so the population floor rises instead of every immigrant being
    uniform-random junk.

    Per continuous gene in the active space it fits the mean+spread of the
    values seen in the top fraction of trials so far, and draws guided
    candidates from those fitted (clamped) Gaussians; genes with no winner
    signal (too few samples) fall back to uniform, so it never over-commits.
    It is refit every time the strategy's archive grows, so the guidance
    sharpens as the run learns — and guided immigrants are MIXED with pure
    random ones (the strategy chooses the ratio), preserving exploration.

    Operates directly on architecture dicts (same shape as
    sample_random_architecture), honouring the active `space` and `pins`.
    """

    def __init__(self, space: Dict[str, Any], pins: Optional[Dict[str, Any]] = None,
                 top_frac: float = 0.25, min_winners: int = 6):
        self.space = space
        self.pins = pins or {}
        self.top_frac = top_frac
        self.min_winners = min_winners
        self.stats: Dict[str, Tuple[float, float, float, float, str]] = {}
        self.ready = False

    @staticmethod
    def _arch_get(arch: Dict[str, Any], full_key: str) -> Any:
        section, key = full_key.split('.', 1)
        return arch.get(section, {}).get(key)

    def update(self, archive: List[Dict[str, Any]]) -> None:
        """Refit the winning-gene distribution from archive entries.

        `archive` entries are {'score'|'fitness', 'arch', ...}; we rank by
        'score' (the selection score) when present, else 'fitness'.
        """
        scored = [a for a in archive if a.get('arch') is not None]
        if len(scored) < self.min_winners * 2:
            return
        scored.sort(key=lambda a: a.get('score', a.get('fitness', -1.0)),
                    reverse=True)
        n_top = max(self.min_winners, int(round(len(scored) * self.top_frac)))
        top = scored[:n_top]
        new_stats: Dict[str, Tuple[float, float, float, float, str]] = {}
        for full_key, spec in self.space.items():
            bounds = _spec_bounds(spec)
            if bounds is None or full_key in _CATEGORICAL_KEYS:
                continue  # categorical genes stay uniform
            lo, hi = bounds
            vals = []
            for a in top:
                v = self._arch_get(a['arch'], full_key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(float(v))
            if len(vals) < self.min_winners:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = max((hi - lo) * 0.06, var ** 0.5)   # floor the spread
            kind = 'int_uniform' if full_key in _INT_KEYS else 'uniform'
            new_stats[full_key] = (mean, std, lo, hi, kind)
        if new_stats:
            self.stats = new_stats
            self.ready = True

    def warm_start(self, stats: Dict[str, Tuple[float, float, float, float, str]]) -> None:
        """Seed the prior directly (e.g. from a resumed run's winners)."""
        if stats:
            self.stats = dict(stats)
            self.ready = True

    def sample(self, rng: random.Random, trial_id: int) -> Dict[str, Any]:
        """Draw one GUIDED architecture dict: modelled genes from their fitted
        Gaussian, everything else uniform / pinned (so it still explores)."""
        arch: Dict[str, Any] = {
            '_meta': {'source': 'NxonArchNAS:Guided', 'trial_id': trial_id,
                      'sampled_at': time.strftime('%Y-%m-%dT%H:%M:%S')},
            'biology': {}, 'neural': {}, 'operating_ranges': {},
            'healthy_bands': {}, 'genetic_lottery': {},
        }
        for full_key, val in self.pins.items():
            section, key = full_key.split('.', 1)
            arch.setdefault(section, {})[key] = val
        for full_key, spec in self.space.items():
            section, key = full_key.split('.', 1)
            st = self.stats.get(full_key)
            if st is None:
                arch.setdefault(section, {})[key] = _sample_one(rng, spec)
                continue
            mean, std, lo, hi, kind = st
            v = rng.gauss(mean, std)
            v = max(lo, min(hi, v))
            arch.setdefault(section, {})[key] = int(round(v)) if kind == 'int_uniform' else v
        lot = arch.get('genetic_lottery', {})
        lo = lot.pop('metabolic_rate_multiplier_lo', None)
        hi = lot.pop('metabolic_rate_multiplier_hi', None)
        if lo is not None and hi is not None:
            lot['metabolic_rate_multiplier_range'] = [min(lo, hi), max(lo, hi)]
        return arch


class _SearchStrategy:
    """Base class. Override `sample(trial_id)` to return an architecture
    dict. `update(result)` is called by the saver after each trial
    finishes — strategies that maintain state (elite pool, surrogate
    model, etc.) use this to incorporate new results."""
    
    def __init__(self, rng: random.Random,
                 space: Optional[Dict[str, Any]] = None,
                 pins: Optional[Dict[str, Any]] = None):
        self.rng = rng
        # v196 — active search sub-space + pins for the inactive genes (focus
        # tier). Default to the full space with no pins (v161-v195 behaviour).
        self.space = space if space is not None else SEARCH_SPACE
        self.pins = pins or {}
    
    def sample(self, trial_id: int) -> Dict[str, Any]:
        raise NotImplementedError
    
    def update(self, result: Dict[str, Any]) -> None:
        """Called by the saver thread after each trial completes.
        Default no-op; subclasses with state override."""
        pass
    
    def describe(self) -> str:
        return self.__class__.__name__


class RandomSearchStrategy(_SearchStrategy):
    """v161-v163 behaviour: every trial is an independent uniform sample
    from SEARCH_SPACE. Robust to noise, exhaustively explorative, never
    exploits past results. v196 — honours the active space + pins (focus tier)."""
    
    def sample(self, trial_id: int) -> Dict[str, Any]:
        return sample_random_architecture(self.rng, trial_id,
                                          space=self.space, pins=self.pins)


class EvolutionaryStrategy(_SearchStrategy):
    """v164 — (μ + λ) evolution strategy with mixed mutation/crossover/
    random operators. v196 — wrapped in a SELF-EVOLVING CONTROLLER.

    State:
      - archive: every completed trial (with fitness AND selection score),
        kept under a lock because the saver-thread updates it concurrently
        with the scheduler-thread reading it
      - random_seed_trials: first N trials always use pure random sampling so
        the elite pool gets seeded with diversity (avoids premature
        convergence)

    Operators (the base mix, adapted live by the controller):
      - p_random   immigrants (continued exploration); a `guided_frac` slice
        of these are drawn from the GuidedSampler (winning-gene EDA) instead
        of uniform-random, raising the population floor
      - p_mutation one elite, perturb a few params with Gaussian noise scaled
        to each param's range and a self-adaptive per-child sigma
      - p_crossover two elites, blend parameter-by-parameter

    v196 CONTROLLER (replaces the v169 binary escape toggle). On a sliding
    window of recent selection scores it:
      * anneals the global mutation sigma with a 1/5th-success rule (shrink
        while improving = exploit, grow while flat = explore);
      * runs a GRADED stagnation ladder keyed on trials-since-best — widen
        mutation (tier 1) → hypermutation burst (tier 2) → partial restart via
        more immigrants (tier 3) — instead of one on/off jump;
      * shifts the operator mix toward immigrants while stuck and toward
        mutation/crossover while improving.
    Set adaptive=False (or --no-adaptive) to freeze the knobs at the v169
    escape behaviour.

    Ranking everywhere (elite pool, archive) is by the LCB SELECTION SCORE
    (fitness − k·stderr + E/I bonus), not raw fitness, so the search exploits
    CONSISTENT architectures rather than lucky single draws.
    """

    def __init__(self, rng: random.Random,
                  elite_pool_size: int = 10,
                  random_seed_trials: int = 50,
                  p_random: float = 0.20,
                  p_mutation: float = 0.50,
                  p_crossover: float = 0.30,
                  mutation_n_params: Tuple[int, int] = (1, 3),
                  mutation_sigma_frac: float = 0.15,
                  escape_threshold: int = 30,
                  escape_sigma_frac: float = 0.35,
                  escape_n_params: Tuple[int, int] = (3, 6),
                  space: Optional[Dict[str, Any]] = None,
                  pins: Optional[Dict[str, Any]] = None,
                  adaptive: bool = True,
                  guided_frac: float = 0.5,
                  select_k: Optional[float] = None,
                  ei_weight: Optional[float] = None):
        super().__init__(rng, space=space, pins=pins)
        self.elite_pool_size = elite_pool_size
        self.random_seed_trials = random_seed_trials
        # Base operator mix (the controller adapts the *live* copies below).
        self.p_random_base = p_random
        self.p_mutation_base = p_mutation
        self.p_crossover_base = p_crossover
        self.p_random = p_random
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        # v169 — store the "normal" mutation knobs and the "escape" knobs separately
        # so we can switch between them as `trials_since_last_best` grows
        self.mutation_n_params_normal = mutation_n_params
        self.mutation_sigma_frac_normal = mutation_sigma_frac
        self.mutation_n_params = mutation_n_params       # current (toggled)
        self.mutation_sigma_frac = mutation_sigma_frac   # current (toggled)
        # v169 — escape mode parameters (the ladder's tier-1 widths)
        self.escape_threshold = max(1, escape_threshold)
        self.escape_sigma_frac = escape_sigma_frac
        self.escape_n_params = escape_n_params

        # v196 — selection / controller config
        self.select_k = _get_select_k() if select_k is None else select_k
        self.ei_weight = _get_ei_weight() if ei_weight is None else ei_weight
        self.adaptive = adaptive
        self.guided_frac = max(0.0, min(1.0, guided_frac))
        # absolute sigma clamp for the 1/5-rule annealing
        self._sigma_lo = max(0.04, 0.4 * mutation_sigma_frac)
        self._sigma_hi = min(0.9, 4.0 * mutation_sigma_frac)
        self._window = max(8, 3 * self.escape_threshold // 2)   # sliding window
        self._recent_improved: List[int] = []   # 1 if trial set a new best, else 0
        self.hypermutate = False                  # tier-2 burst flag
        self.restart_frac = 0.0                   # tier-3 extra-immigrant fraction
        self.tier = 0                             # current stagnation tier
        self._best_score = -1.0                   # best LCB selection score seen

        # v196 — online guided (EDA) immigrant sampler over the active space
        self._guided = GuidedSampler(self.space, pins=self.pins)

        # Thread-safe archive of all completed trials
        self._lock = threading.Lock()
        self._archive: List[Dict[str, Any]] = []
        
        # v169 (v4.77) — NAS escape state tracking (kept for status/back-compat;
        # the v196 controller generalises it into the graded ladder above).
        self.best_fitness = -1.0
        self.trials_since_last_best = 0
        self.in_escape_mode = False
        self.escape_events = []   # list of (trial_id_started, trial_id_resolved or None)
        
        # Op counters (for status reporting)
        self.op_count = {'random': 0, 'guided': 0, 'mutation': 0, 'crossover': 0}
    
    def update(self, result: Dict[str, Any]) -> None:
        """Called from saver thread. Add this trial to the archive (ranked by
        its v196 LCB selection score) and re-plan the search strategy via the
        self-evolving controller."""
        fit = result.get('fitness', -1.0)
        if fit <= -1.0:
            return
        # v196 — rank by the selection score the saver already computed
        # (fall back to computing it here for programmatic callers).
        score = result.get('selection_score')
        if score is None:
            score = selection_score(result, k=self.select_k, ei_weight=self.ei_weight)
        tid = result.get('trial_id', -1)
        with self._lock:
            self._archive.append({
                'fitness': fit,
                'score':   score,
                'arch':    result['arch'],
                'trial_id': tid,
            })
            improved = score > self._best_score + 1e-6
            if improved:
                self._best_score = score
                self.trials_since_last_best = 0
                # legacy escape bookkeeping
                if self.in_escape_mode:
                    if self.escape_events and self.escape_events[-1][1] is None:
                        self.escape_events[-1] = (self.escape_events[-1][0], tid)
                    self.in_escape_mode = False
                self.best_fitness = max(self.best_fitness, fit)
            else:
                self.trials_since_last_best += 1
            self._recent_improved.append(1 if improved else 0)
            if len(self._recent_improved) > self._window:
                self._recent_improved.pop(0)
            # Refit the guided EDA prior from the (now larger) archive.
            try:
                self._guided.update(self._archive)
            except Exception:
                pass
            # Re-plan knobs.
            if self.adaptive and len(self._archive) > self.random_seed_trials:
                self._replan(tid)
    
    def _replan(self, tid: int) -> None:
        """v196 controller — runs under self._lock. Anneal mutation sigma with a
        1/5th-success rule and escalate a graded stagnation ladder. Pure
        bookkeeping over recent selection scores; sample() reads the result."""
        # --- 1/5th success rule on the recent window -----------------------
        win = self._recent_improved
        if len(win) >= max(5, self._window // 2):
            rate = sum(win) / len(win)
            if rate > 0.2:        # improving often → exploit (shrink sigma)
                self.mutation_sigma_frac = max(self._sigma_lo,
                                               self.mutation_sigma_frac * 0.85)
            elif rate < 0.2:      # stalling → explore (grow sigma)
                self.mutation_sigma_frac = min(self._sigma_hi,
                                               self.mutation_sigma_frac * 1.25)
        # --- graded stagnation ladder --------------------------------------
        self.hypermutate = False
        self.restart_frac = 0.0
        tier = self.trials_since_last_best // self.escape_threshold
        if tier != self.tier:
            self.tier = tier
            if tier > 0:
                print(f"[NAS controller] STAGNATION tier {tier} "
                      f"({self.trials_since_last_best} trials flat, "
                      f"best_score={self._best_score:.3f}) — escalating", flush=True)
            else:
                print(f"[NAS controller] improvement at trial {tid} "
                      f"(score {self._best_score:.3f}) — de-escalating to tier 0",
                      flush=True)
        if tier == 0:
            # improving: exploit — favour mutation/crossover, few immigrants
            self.p_random = max(0.10, self.p_random_base * 0.75)
            self.mutation_n_params = self.mutation_n_params_normal
        else:
            # stuck: explore — more immigrants, broader mutation the deeper we are
            self.p_random = min(0.5, self.p_random_base + 0.12 * tier)
            self.mutation_sigma_frac = min(self._sigma_hi,
                                           max(self.mutation_sigma_frac,
                                               self.escape_sigma_frac * (1.0 + 0.3 * (tier - 1))))
            self.mutation_n_params = self.escape_n_params
            self.in_escape_mode = True
            if not self.escape_events or self.escape_events[-1][1] is not None:
                self.escape_events.append((tid, None))
            if tier >= 2:
                self.hypermutate = True          # broad burst
            if tier >= 3:
                self.restart_frac = min(0.6, 0.2 * (tier - 1))  # partial restart
        # keep the operator mix normalised (crossover takes the remainder)
        self.p_mutation = self.p_mutation_base
        self.p_crossover = max(0.0, 1.0 - self.p_random - self.p_mutation)
    
    def _get_elites(self) -> List[Dict[str, Any]]:
        """Return top-N archive entries by v196 LCB selection score. Snapshot
        under lock. Falls back to raw fitness for legacy entries lacking a score."""
        with self._lock:
            sorted_arch = sorted(
                self._archive,
                key=lambda x: -x.get('score', x.get('fitness', -1.0)))
            return sorted_arch[:self.elite_pool_size]
    
    def seed_with_archs(self, seed_archs: List[Dict[str, Any]],
                          assumed_fitness: float = 6.0) -> None:
        """v170 (v4.78) — pre-populate the elite pool with known-good
        architectures (e.g. nas_top1.json, nas_top2.json from a previous run).
        Each architecture is added to the archive with `assumed_fitness` so
        it dominates the elite-pool selection until real trials produce
        higher-fitness alternatives. Also bumps the trials_since_last_best
        counter to 0 and sets best_fitness so the escape mechanism has a
        meaningful baseline.
        
        Typical usage: cross-duration validation. Load the top-N
        architectures discovered at 1800s, then run a short search at
        2400s. The seeded architectures get evaluated first (their stored
        fitness is replaced by the real 2400s fitness on first evaluation),
        and evolutionary search refines around them.
        
        Args:
          seed_archs: list of architecture dicts (output of
                      load_architecture_from_file or sample_random_architecture)
          assumed_fitness: placeholder fitness used until real evaluation
                           (default 6.0 — high enough to dominate random seeds)
        """
        with self._lock:
            for i, arch in enumerate(seed_archs):
                self._archive.append({
                    'fitness': assumed_fitness,
                    'score': assumed_fitness,   # v196 — seeds rank by score too
                    'arch': arch,
                    'trial_id': -1 - i,   # negative trial_ids mark "seeded"
                })
            self.best_fitness = max(assumed_fitness, self.best_fitness)
            self._best_score = max(assumed_fitness, self._best_score)
            self.trials_since_last_best = 0
        print(f"[NAS] Seeded elite pool with {len(seed_archs)} pre-evaluated "
              f"architectures (assumed_fitness={assumed_fitness})", flush=True)
    
    def _empty_arch(self, trial_id: int) -> Dict[str, Any]:
        """Initialise an empty arch shell with sections matching
        sample_random_architecture's output."""
        return {
            '_meta': {
                'source': 'NxonArchNAS:Evolutionary',
                'trial_id': trial_id,
                'sampled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            },
            'biology': {}, 'neural': {}, 'operating_ranges': {},
            'healthy_bands': {}, 'genetic_lottery': {},
        }
    
    def _mutate_value(self, key: str, val: Any, spec: Any,
                      sigma_frac: Optional[float] = None) -> Any:
        """Perturb a single value within its SEARCH_SPACE bounds. For
        continuous: Gaussian noise scaled to `sigma_frac` (default the
        controller's current mutation_sigma_frac) of the range. For discrete:
        50% chance pick a neighbor in the list."""
        sf = self.mutation_sigma_frac if sigma_frac is None else sigma_frac
        if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], str):
            kind, lo, hi = spec
            span = max(1e-12, hi - lo)
            sigma = sf * span
            if kind == 'uniform':
                new_val = float(val) + self.rng.gauss(0, sigma)
                return max(lo, min(hi, new_val))
            elif kind == 'loguniform':
                import math
                log_lo, log_hi = math.log(lo), math.log(hi)
                log_span = log_hi - log_lo
                log_sigma = sf * log_span
                new_log = math.log(max(lo * 1e-6, float(val))) + self.rng.gauss(0, log_sigma)
                new_log = max(log_lo, min(log_hi, new_log))
                return math.exp(new_log)
            elif kind == 'int_uniform':
                new_val = int(round(float(val) + self.rng.gauss(0, sigma)))
                return max(int(lo), min(int(hi), new_val))
        elif isinstance(spec, (list, tuple)):
            try:
                idx = spec.index(val)
                # 50% pick a neighbor
                if self.rng.random() < 0.5 and len(spec) > 1:
                    direction = self.rng.choice([-1, 1])
                    new_idx = max(0, min(len(spec) - 1, idx + direction))
                    return spec[new_idx]
                return val
            except ValueError:
                return self.rng.choice(spec)
        return val
    
    def _mutate(self, parent: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
        """Copy parent arch, perturb a few ACTIVE-space parameters.

        v196 — only genes in the active space are mutated (focus tier leaves
        the pinned genes untouched); the per-child sigma is self-adapted
        (ES-style log-normal jitter around the controller's current sigma) so
        different children explore different scales; a tier-2 hypermutation
        burst doubles the sigma for genuine long-range jumps."""
        child = self._empty_arch(trial_id)
        # Copy all sections from parent
        for section in ('biology', 'neural', 'operating_ranges',
                        'healthy_bands', 'genetic_lottery'):
            if section in parent:
                child[section] = copy.deepcopy(parent[section])
        
        # ES-style self-adaptive per-child sigma (log-normal around current).
        sigma_eff = self.mutation_sigma_frac * math.exp(0.3 * self.rng.gauss(0.0, 1.0))
        if self.hypermutate:
            sigma_eff *= 2.0
        sigma_eff = max(self._sigma_lo, min(self._sigma_hi, sigma_eff))
        
        # Pick parameters to mutate — from the ACTIVE space only (focus tier).
        keys = list(self.space.keys())
        if not keys:
            return child
        n_lo, n_hi = self.mutation_n_params
        n_mut = self.rng.randint(n_lo, min(n_hi, len(keys)))
        mut_keys = self.rng.sample(keys, n_mut)
        
        for full_key in mut_keys:
            section, key = full_key.split('.', 1)
            spec = self.space[full_key]
            current = child.setdefault(section, {}).get(key)
            if current is None:
                # Parent doesn't have this param — sample fresh from spec
                child[section][key] = _sample_one(self.rng, spec)
            else:
                child[section][key] = self._mutate_value(full_key, current, spec,
                                                         sigma_frac=sigma_eff)
        
        # Re-collapse the metabolic_rate lottery pair (matches sample_random_architecture)
        lot = child.get('genetic_lottery', {})
        # The mutation may have hit lo or hi individually; collapse to a range
        # if both are present. Parent already has it as a single range; if mutation
        # generated both lo/hi separately, fold them.
        if 'metabolic_rate_multiplier_lo' in lot or 'metabolic_rate_multiplier_hi' in lot:
            lo = lot.pop('metabolic_rate_multiplier_lo', None)
            hi = lot.pop('metabolic_rate_multiplier_hi', None)
            existing_range = lot.get('metabolic_rate_multiplier_range')
            if lo is None and existing_range: lo = existing_range[0]
            if hi is None and existing_range: hi = existing_range[1]
            if lo is not None and hi is not None:
                lot['metabolic_rate_multiplier_range'] = [min(lo, hi), max(lo, hi)]
        return child
    
    def _crossover(self, parent_a: Dict[str, Any],
                   parent_b: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
        """Mix two parents: for each ACTIVE-space parameter, randomly pick
        from A or B (50/50). Keeps the structure but blends search-space
        coordinates. v196 — only the active genes are blended (pinned genes
        are identical in both parents in focus mode)."""
        child = self._empty_arch(trial_id)
        # Start by copying parent_a entirely
        for section in ('biology', 'neural', 'operating_ranges',
                        'healthy_bands', 'genetic_lottery'):
            if section in parent_a:
                child[section] = copy.deepcopy(parent_a[section])
        # For each ACTIVE searchable param, 50% chance to use parent_b's value
        for full_key in self.space.keys():
            section, key = full_key.split('.', 1)
            if self.rng.random() < 0.5:
                b_val = parent_b.get(section, {}).get(key)
                if b_val is not None:
                    child.setdefault(section, {})[key] = b_val
        # Re-collapse lottery range
        lot = child.get('genetic_lottery', {})
        a_range = parent_a.get('genetic_lottery', {}).get('metabolic_rate_multiplier_range')
        b_range = parent_b.get('genetic_lottery', {}).get('metabolic_rate_multiplier_range')
        if a_range and b_range:
            chosen = a_range if self.rng.random() < 0.5 else b_range
            lot['metabolic_rate_multiplier_range'] = list(chosen)
        return child

    def _immigrant(self, trial_id: int) -> Dict[str, Any]:
        """One immigrant: guided (winning-gene EDA) with prob guided_frac once
        the guided prior is ready, else a pure-random sample. Both honour the
        active space + pins."""
        if (self._guided.ready and self.guided_frac > 0.0
                and self.rng.random() < self.guided_frac):
            self.op_count['guided'] += 1
            return self._guided.sample(self.rng, trial_id)
        self.op_count['random'] += 1
        return sample_random_architecture(self.rng, trial_id,
                                          space=self.space, pins=self.pins)
    
    def sample(self, trial_id: int) -> Dict[str, Any]:
        """Generate one architecture: seed phase = random; otherwise
        pick an operator weighted by the controller's live p_random /
        p_mutation / p_crossover. v196 — immigrants may be GUIDED, and a
        tier-3 partial restart raises the effective immigrant fraction."""
        with self._lock:
            archive_size = len(self._archive)
            p_random = self.p_random
            restart_frac = self.restart_frac
        
        # Seed phase: always random until we have enough elites
        if archive_size < self.random_seed_trials:
            self.op_count['random'] += 1
            return sample_random_architecture(self.rng, trial_id,
                                              space=self.space, pins=self.pins)
        
        # v196 — partial restart (tier 3+) injects extra immigrants on top of
        # the base immigrant rate to jump to a new region of the space.
        p_imm = min(0.9, p_random + restart_frac)
        r = self.rng.random()
        if r < p_imm or archive_size < 2:
            return self._immigrant(trial_id)
        
        elites = self._get_elites()
        if not elites:
            return self._immigrant(trial_id)
        
        if r < p_imm + self.p_mutation:
            # Mutation: pick one elite, perturb a few params
            parent = self.rng.choice(elites)
            self.op_count['mutation'] += 1
            return self._mutate(parent['arch'], trial_id)
        else:
            # Crossover: pick two distinct elites
            if len(elites) >= 2:
                a, b = self.rng.sample(elites, 2)
                self.op_count['crossover'] += 1
                return self._crossover(a['arch'], b['arch'], trial_id)
            else:
                # Only one elite — fall back to mutation
                self.op_count['mutation'] += 1
                return self._mutate(elites[0]['arch'], trial_id)
    
    def describe(self) -> str:
        total = sum(self.op_count.values())
        if total == 0:
            return "EvolutionaryStrategy (no ops yet)"
        with self._lock:
            archive_n = len(self._archive)
        return (f"Evolutionary[pool={self.elite_pool_size}, "
                f"seed={self.random_seed_trials}, archive={archive_n}, "
                f"tier={self.tier} sigma={self.mutation_sigma_frac:.3f} "
                f"p_imm~{self.p_random:.2f} guided_frac={self.guided_frac:.2f}, "
                f"ops: rnd={self.op_count['random']} "
                f"gui={self.op_count['guided']} "
                f"mut={self.op_count['mutation']} "
                f"x={self.op_count['crossover']}]")


# Factory
def make_strategy(name: str, rng: random.Random, **kwargs) -> _SearchStrategy:
    """Build a strategy by name. Accepts kwargs to override defaults."""
    name = (name or 'random').lower().strip()
    if name in ('random', 'rand'):
        # RandomSearchStrategy only understands space/pins; ignore evo kwargs.
        return RandomSearchStrategy(rng,
                                    space=kwargs.get('space'),
                                    pins=kwargs.get('pins'))
    elif name in ('evolutionary', 'evo', 'evolution', 'ga'):
        return EvolutionaryStrategy(rng, **kwargs)
    else:
        raise ValueError(f"Unknown search strategy '{name}'. "
                         f"Use 'random' or 'evolutionary'.")


# =============================================================================
# WORKER — runs in a subprocess
# =============================================================================
def _run_single_trial(args: Tuple) -> Dict[str, Any]:
    """Run ONE Game-of-Life trial in headless mode with the given arch and
    seed. (v193: this is one seed-repeat; _trial_worker runs several of these
    per architecture and aggregates — see _aggregate_reps.)
    
    This runs inside a multiprocessing subprocess. All imports happen
    inside the function so each worker has fresh pygame/architecture
    state (multiprocessing.spawn starts each subprocess with a clean
    Python interpreter).
    
    Returns a dict with: trial_id, arch, metrics, error (or None),
    wall_seconds_actual, n_samples.
    """
    # v194 — accept an optional rep index + count (6/7-tuple) so seed-repeats
    # nest under one per-trial folder; stay backward-compatible with the
    # 5-tuple (single-trial) form.
    if len(args) >= 7:
        arch_dict, trial_id, wall_seconds, seed, out_dir, rep_idx, n_reps = args[:7]
    elif len(args) == 6:
        arch_dict, trial_id, wall_seconds, seed, out_dir, rep_idx = args
        n_reps = 1
    else:
        arch_dict, trial_id, wall_seconds, seed, out_dir = args
        rep_idx, n_reps = 0, 1
    result = {
        'trial_id': trial_id, 'arch': arch_dict, 'metrics': None,
        'error': None, 'wall_seconds_actual': 0.0, 'n_samples': 0,
    }
    t0 = time.time()
    
    try:
        # Force headless environment BEFORE any pygame import
        # v161 (v4.69) — NEURAXON_HEADLESS=1 also bypasses tkinter file
        # dialogs in utils._pick_save_file, which would otherwise hang
        # the subprocess forever waiting for user click after KeyMetrics
        # save (the bug that caused v158-v160 NAS to stop after one batch
        # on Windows). See utils._is_headless().
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        os.environ['NEURAXON_HEADLESS'] = '1'
        
        # Write this trial's architecture to a JSON file so the loader
        # in the game subprocess picks it up
        arch_path = os.path.join(out_dir, f'trial_{trial_id:03d}__arch.json')
        with open(arch_path, 'w', encoding='utf-8') as f:
            json.dump(arch_dict, f, indent=2)
        os.environ['NEURAXON_ARCH'] = arch_path
        
        # cd into a per-trial directory so save files don't clutter.
        # v194 — when there are multiple seed-repeats, nest them under ONE
        # per-trial folder (trial_022/rep0, rep1, rep2) instead of three
        # sibling 'trial_022_s<seed>' dirs, so the layout reads as one trial
        # with R repeats rather than several trials sharing a number. A single
        # repeat keeps the flat 'trial_022' name.
        if n_reps > 1:
            trial_dir = os.path.join(out_dir, f'trial_{trial_id:03d}', f'rep{rep_idx}')
        else:
            trial_dir = os.path.join(out_dir, f'trial_{trial_id:03d}')
        os.makedirs(trial_dir, exist_ok=True)
        os.chdir(trial_dir)
        
        # v161 — RESET WORKER-PROCESS GLOBALS for safe reuse.
        # Pool workers are spawn'd once but reused across many trials
        # (each worker subprocess may run dozens of trials). Module-level
        # globals (logger singleton, config game_id, architecture cache,
        # data_logger) persist between calls. If we don't reset them,
        # trial N+1 reads trial N's stale time_series / per_nxer data.
        try:
            import logger as logger_module
            logger_module._data_logger = None   # force fresh logger per trial
        except Exception:
            pass
        try:
            import architecture as arch_module_pre
            arch_module_pre._ARCH = None
            arch_module_pre._ARCH_PATH = None
        except Exception:
            pass
        
        # Now import pygame and game modules (per-subprocess fresh state)
        import pygame
        pygame.init()
        # Minimal display surface — game_loop requires one even in headless
        pygame.display.set_mode((320, 200))
        
        import architecture as arch_module
        arch_module._ARCH = None  # ensure reload
        arch_module._ARCH_PATH = None
        arch_module.load_architecture(arch_path, verbose=False)
        
        import config
        config._game_id = f'NAS_t{trial_id:03d}'
        config._session_id = None  # v161 — ensure fresh session id per trial
        
        from game_loop import GameOfLife
        
        # Compute time limit in minutes (run_game uses minutes)
        limit_min = max(0.1, wall_seconds / 60.0)
        
        # Trial config + computed limit + seed
        trial_args = dict(NAS_TRIAL_CONFIG)
        trial_args['limit_minutes'] = limit_min
        trial_args['random_seed'] = seed
        
        # Run the game (headless mode kicks in because limit_minutes is set)
        GameOfLife(**trial_args)
        
        # Extract metrics from the data_logger (which is per-process singleton)
        from logger import get_data_logger
        log = get_data_logger()
        ts = getattr(log, 'time_series', {})
        n_samples = len(ts.get('ticks', []))
        result['n_samples'] = n_samples
        
        if n_samples == 0:
            result['error'] = 'no_time_series'
        else:
            # Compute fitness inputs from last 20% of run (skip startup transient)
            last_n = max(1, n_samples // 5)
            
            def last_mean(key: str, default: float = 0.0) -> float:
                vals = ts.get(key, [])
                if not vals: return default
                tail = vals[-last_n:]
                return float(sum(tail) / len(tail)) if tail else default
            
            def first_mean(key: str, default: float = 0.0) -> float:
                vals = ts.get(key, [])
                if not vals: return default
                head = vals[:max(1, n_samples // 5)]
                return float(sum(head) / len(head)) if head else default
            
            # v165 (v4.73) — peak & mean over FULL trial (not just tail).
            # Needed for sm_corr which often spikes to 0.5-0.7 early then
            # collapses to 0 once input sensors saturate (input_locked_fraction
            # → 1.0). The v161-v164 last_mean measurement only captured the
            # post-saturation zero, hiding the architectural capability.
            def peak_abs(key: str, default: float = 0.0) -> float:
                vals = ts.get(key, [])
                if not vals: return default
                return float(max((abs(v) for v in vals), default=default))
            
            def mean_abs(key: str, default: float = 0.0) -> float:
                vals = ts.get(key, [])
                if not vals: return default
                return float(sum(abs(v) for v in vals) / len(vals))
            
            alive_series = ts.get('surv_alive_count', [0])
            final_alive = float(alive_series[-1]) if alive_series else 0.0
            start_alive = float(alive_series[0]) if alive_series else 0.0
            peak_alive = max((float(v) for v in alive_series), default=0.0)

            # v192 (v5.07) — HONEST, NEVER-TRIMMED survival summary. The
            # surv_alive_count time-series above is trimmed to the last
            # max_history_length (=10000) samples (v185+ memory bound),
            # which drops the early high-population phase of a multi-hour
            # trial. peak_alive / the checkpoint mean computed from the
            # trimmed series therefore reflect only the recent window —
            # the root cause of the v191 8h run scoring every trial 0.0
            # (trials that sustained ~10 NxErs early showed peak_alive=1
            # after the early samples were trimmed, then hit the population
            # floor). SurvivabilityTracker maintains O(1) un-trimmed
            # scalars over the WHOLE trial; prefer them when available and
            # fall back to the trimmed series for old result dicts / tests.
            true_peak_alive = peak_alive
            mean_alive_ever = None
            try:
                _surv = getattr(log, 'survivability', None)
                if _surv is not None and getattr(_surv, 'alive_samples_ever', 0) > 0:
                    true_peak_alive = float(max(peak_alive, _surv.max_alive_ever))
                    mean_alive_ever = float(_surv.alive_sum_ever) / float(_surv.alive_samples_ever)
                    # the tracker's last sample is the honest final population
                    final_alive = float(getattr(_surv, 'last_alive_count', final_alive))
            except Exception:
                pass
            
            # v168 (v4.76) — multi-checkpoint survival sampling.
            # Old v161-v167 fitness used `final_alive` only, which credits
            # "transient crash then recovery" the same as "stable throughout."
            # Specifically: a trial alive=10 at 25%, 5 at 50%, 5 at 75%, 10 at
            # end (recovered from a crash) scored 3.0 survival pts — same as
            # one that maintained 10 the whole time. This selects for
            # architectures that look good at trial end but may have crashed
            # mid-trial. The 1200s NAS run revealed that the v166 champions
            # (trial 38/34) had this exact pattern: they survived 900s fine
            # but couldn't sustain 1200s without mid-trial losses.
            #
            # New: sample alive count at 25%, 50%, 75%, 100% of the trial
            # and use the MEAN. Architectures that maintain population
            # THROUGHOUT score higher than those that crash and recover.
            n_alive = len(alive_series)
            if n_alive >= 4:
                q1_idx = max(0, n_alive // 4 - 1)
                q2_idx = max(0, n_alive // 2 - 1)
                q3_idx = max(0, (3 * n_alive) // 4 - 1)
                q4_idx = n_alive - 1
                alive_q1 = float(alive_series[q1_idx])
                alive_q2 = float(alive_series[q2_idx])
                alive_q3 = float(alive_series[q3_idx])
                alive_q4 = float(alive_series[q4_idx])  # == final_alive
                alive_mean_checkpoints = (alive_q1 + alive_q2 + alive_q3 + alive_q4) / 4.0
            else:
                # Short trial — degrade to final_alive
                alive_q1 = alive_q2 = alive_q3 = alive_q4 = final_alive
                alive_mean_checkpoints = final_alive

            # v192 (v5.07) — prefer the honest, never-trimmed time-average of
            # the population as the sustained-survival signal. The trimmed
            # checkpoint mean above only sees the recent window; mean_alive_ever
            # is the true Σalive/Σsamples over the WHOLE trial, so a trial that
            # held ~10 then crashed reads a moderate mean (not 1), and one that
            # never grew reads ~1. Keep the checkpoint mean as a fallback.
            if mean_alive_ever is not None:
                alive_mean_checkpoints = mean_alive_ever

            result['metrics'] = {
                'final_alive':       final_alive,
                'start_alive':       start_alive,
                # v192 — TRUE historical peak (un-trimmed), not the recent-window max
                'peak_alive':        true_peak_alive,
                'peak_alive_trimmed': peak_alive,   # what the old code saw (diagnostic)
                'mean_alive_ever':   (mean_alive_ever if mean_alive_ever is not None else alive_mean_checkpoints),
                # v168 — checkpoint alive counts for sustained-survival fitness
                'alive_q1':          alive_q1,
                'alive_q2':          alive_q2,
                'alive_q3':          alive_q3,
                'alive_q4':          alive_q4,
                'alive_mean_checkpoints': alive_mean_checkpoints,
                'M1_last':           last_mean('M1_excitatory_fraction'),
                # v189 (v5.04) — thread the buffer-state ("neutral", state=0)
                # fraction through to fitness. compute_m1_trinary_distribution
                # has logged this since v161 (line 195 of research_probes.py)
                # but it was never pulled into the metrics dict, so fitness
                # had no view of the trinary balance — only the excitatory
                # fraction (M1) and the inferred M5 branching ratio. v188
                # exposed the cost: trial 46 reached fitness 6.61 with M1 in
                # band, but the live game showed state=0 collapsed to 4.1%
                # of hidden samples (paper target: ~68%). The trinary
                # architecture had degraded to a +1/-1 binary oscillator
                # with no rest band, and 2h trials couldn't see it because
                # the lock-in develops over multiple hours. Adding this
                # column lets the fitness function reward the rest band
                # directly, and lets the NAS CSV log the value for
                # post-hoc analysis without re-running.
                'M1_neutral_last':   last_mean('M1_neutral_fraction'),
                'M1_inh_last':       last_mean('M1_inhibitory_fraction'),
                # v191 (v5.06) — the remaining healthy-band metrics are now
                # extracted so the fitness "band completeness" term (and the
                # NAS CSV) can see the FULL healthy profile, not just the
                # subset v161-v190 used. compute_*_metrics in the logger has
                # produced all of these since the early Mx series, but only
                # M1/M6/M9/M10 (+streak/sm/locked/expl) were ever threaded
                # into the trial metrics dict — so the search had no gradient
                # toward M2 gate / M3 PAC / M5 branching / M7 zero-input MI /
                # input saturation. The v190 winner t110 sat in band on
                # neutral and survival but its M5/M7 were never part of
                # selection; v191 makes the whole healthy_bands dict the
                # objective so the NAS finds the architecture where ALL
                # metrics land in band (the "right model").
                'M2_last':           last_mean('M2_mean_gate'),
                'M3_last':           last_mean('M3_pac_modulation_idx'),
                'M5_last':           last_mean('M5_branching_ratio'),
                'M7_last':           last_mean('M7_zero_input_mi_ratio'),
                'input_sat_last':    last_mean('input_saturation_fraction'),
                'M6_last':           last_mean('M6_spontaneous_fraction'),
                'M9_last':           last_mean('M9_transfer_ratio'),
                'M10_last':          last_mean('M10_heritability_r'),
                # v165 — also expose peak M10 (some trials hit high M10 briefly)
                'M10_peak':          peak_abs('M10_heritability_r'),
                'mean_streak_last':  last_mean('mean_state_streak'),
                'mean_streak_first': first_mean('mean_state_streak'),
                'locked_last':       last_mean('input_locked_fraction'),
                'surv_score_last':   last_mean('surv_score'),
                'expl_rate_last':    last_mean('exploration_trigger_rate'),
                # v165 — three sm_corr views. `peak` is now the fitness driver.
                'sm_corr_last':      last_mean('sensory_motor_corr'),
                'sm_corr_peak':      peak_abs('sensory_motor_corr'),
                'sm_corr_mean':      mean_abs('sensory_motor_corr'),
                # v179 (v4.87) — population g-factor signatures (paper method)
                'g_pc1_last':        last_mean('g_pc1_fraction'),
                'g_pc1_peak':        peak_abs('g_pc1_fraction'),
                'g_posman_last':     last_mean('g_positive_manifold'),
                'g_meanr_last':      last_mean('g_mean_offdiag_r'),
                'g_l1l2_last':       last_mean('g_lambda1_over_lambda2'),
                # v179 — surface the opt-in g-fitness weight from the arch
                # so fitness() can apply it (default 0.0 = selection-neutral).
                'fitness_g_weight':  float(
                    (arch_dict.get('operating_ranges', {}) or {})
                    .get('fitness_g_weight', 0.0) or 0.0),
                # v186 (v5.01) — round-visibility. With max_rounds=1 (the
                # default in NAS_TRIAL_CONFIG) total_rounds is 1, but we
                # query it from config so manual overrides surface in the
                # log. went_extinct = True iff surv_alive_count reached 0
                # at any sample during the trial. extinction_tick = the
                # first tick at which that happened (or -1 if never).
                'total_rounds':      int(getattr(config, '_current_round', 1) or 1),
                'went_extinct':      bool(alive_series and min(alive_series) <= 0),
                # v191 (v5.06) — min_alive makes the "limping" case visible.
                # went_extinct only fires on a LITERAL zero-population game-
                # over; the v190 8h run had 0/128 extinctions yet 70/128
                # trials were population-floored (peak_alive < 8) — they
                # limped at a tiny non-zero population for the full budget,
                # indistinguishable in went_extinct from a healthy run.
                # min_alive (the trough of the alive series) separates
                # "ran full budget healthy" from "ran full budget barely
                # alive" without re-running the trial.
                'min_alive':         (float(min(alive_series)) if alive_series else 0.0),
                # v191 — thread the architecture's healthy_bands target
                # ranges into the metrics dict so fitness() can score band
                # completeness against the SAME ranges the dashboard uses
                # (kept in sync with architectures/default.json). Falls back
                # to None → fitness uses its built-in paper defaults.
                'healthy_bands':     (arch_dict.get('healthy_bands') or None),
                'extinction_tick':   (
                    int(ts.get('ticks', [-1])[
                        next((i for i, v in enumerate(alive_series) if v <= 0),
                             len(alive_series) - 1)
                    ]) if alive_series and min(alive_series) <= 0 else -1
                ),
            }
        
        # Cleanly shut down logger so resources release before subprocess exits
        try:
            log.shutdown()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
    
    except Exception:
        result['error'] = f"trial_crashed: {traceback.format_exc().splitlines()[-1]}"
    
    result['wall_seconds_actual'] = time.time() - t0
    return result


def _aggregate_reps(reps: List[Dict[str, Any]], arch_dict: Dict[str, Any],
                    trial_id: int, n_repeats: int) -> Dict[str, Any]:
    """v193 — combine NAS_TRIAL_REPEATS seed-repeats of one architecture into a
    single robust result. Metrics are aggregated by MEDIAN (robust to one
    crashed/outlier run); per-seed spread is recorded so the noise is visible.
    Selection then runs on the median metrics (the caller computes
    fitness(result['metrics'])), making the leaderboard reflect the
    architecture's typical behaviour rather than one lucky/unlucky sample."""
    import statistics as _st
    ok = [r for r in reps if r.get('metrics')]
    agg = {
        'trial_id': trial_id, 'arch': arch_dict, 'metrics': None,
        'error': None,
        'wall_seconds_actual': sum(r.get('wall_seconds_actual', 0.0) for r in reps),
        'n_samples': sum(r.get('n_samples', 0) for r in reps),
    }
    if not ok:
        # every repeat errored/empty — surface the first error
        agg['error'] = next((r.get('error') for r in reps if r.get('error')),
                             'all_repeats_failed')
        return agg

    # Median per numeric metric across the OK repeats.
    keys = set()
    for r in ok:
        keys.update(r['metrics'].keys())
    median_metrics: Dict[str, Any] = {}
    for k in keys:
        vals = [r['metrics'].get(k) for r in ok
                if isinstance(r['metrics'].get(k), (int, float))]
        if vals:
            median_metrics[k] = _st.median(vals)
    # Carry through non-numeric metrics (e.g. healthy_bands dict) from the
    # first OK repeat so fitness() can still read them.
    for k in keys:
        if k not in median_metrics and ok[0]['metrics'].get(k) is not None:
            median_metrics[k] = ok[0]['metrics'][k]

    # Per-seed spread / reliability indicators (computed, not fatal if missing).
    def _spread(metric_key):
        vals = [r['metrics'].get(metric_key) for r in ok
                if isinstance(r['metrics'].get(metric_key), (int, float))]
        return (_st.pstdev(vals) if len(vals) > 1 else 0.0), vals

    m1_std, _ = _spread('M1_last')
    neut_std, _ = _spread('M1_neutral_last')
    streak_std, _ = _spread('mean_streak_last')
    # Per-repeat fitness spread — the headline reliability signal. fitness() is
    # defined later in this module but resolvable at call time.
    fit_vals = []
    for r in ok:
        try:
            fit_vals.append(fitness(r['metrics']))
        except Exception:
            pass
    fit_mean = (_st.mean(fit_vals) if fit_vals else 0.0)
    fit_std = (_st.pstdev(fit_vals) if len(fit_vals) > 1 else 0.0)

    median_metrics['n_repeats'] = float(n_repeats)
    median_metrics['n_repeats_ok'] = float(len(ok))
    median_metrics['M1_std'] = m1_std
    median_metrics['M1_neutral_std'] = neut_std
    median_metrics['mean_streak_std'] = streak_std
    median_metrics['fitness_mean_reps'] = fit_mean
    median_metrics['fitness_std_reps'] = fit_std

    agg['metrics'] = median_metrics
    # If any repeat errored (but not all), note it without failing the trial.
    agg['error'] = next((r.get('error') for r in reps if r.get('error')), None)
    return agg


def _trial_worker(args: Tuple) -> Dict[str, Any]:
    """v193 — evaluate ONE architecture as NAS_TRIAL_REPEATS seed-repeats and
    return the robust (median) aggregate. Drop-in for the Pool dispatch: still
    one apply_async per architecture, one result to the saver. With repeats=1
    this is exactly the v192 single-trial behaviour (median of one).
    v194 — in 'subdivide' mode (default) the per-architecture wall is split
    across the repeats (each runs wall/R), so seed-averaging is COST-NEUTRAL vs
    a single v192 trial and the search keeps its breadth. 'multiply' mode runs
    each repeat at the full wall (R× cost) for deep single-arch work.
    args = (arch_dict, trial_id, wall_seconds, seed, out_dir)."""
    arch_dict, trial_id, wall_seconds, seed, out_dir = args
    R = _get_trial_repeats()
    mode = _get_repeats_mode()
    # subdivide: each repeat gets wall/R (floor 6 s = 0.1 min, matching the
    # per-trial limit floor); multiply: each repeat gets the full wall.
    per_rep_wall = wall_seconds
    if mode == 'subdivide' and R > 1:
        per_rep_wall = max(6.0, wall_seconds / float(R))
    reps = []
    for r in range(R):
        # Well-separated seeds (distinct large stride) so repeats are
        # genuinely independent draws, not adjacent RNG states.
        rep_seed = seed + r * 2654435761 % (2 ** 31)
        reps.append(_run_single_trial(
            (arch_dict, trial_id, per_rep_wall, rep_seed, out_dir, r, R)))
    return _aggregate_reps(reps, arch_dict, trial_id, R)
def _band_score(val: float, lo: float, hi: float, margin_frac: float = 1.0) -> float:
    """v191 (v5.06) — graded in-band score for a metric.

    Returns 1.0 when lo <= val <= hi, and ramps linearly to 0 over a margin
    of `margin_frac * (hi - lo)` on each side outside the band. A gentle
    margin (default 1.0 = one band-width) means an out-of-band metric still
    contributes partial credit proportional to how close it is, giving the
    search a gradient rather than a cliff. Used by the band-completeness
    term so every healthy_bands metric pulls selection toward its target.
    """
    try:
        val = float(val); lo = float(lo); hi = float(hi)
    except (TypeError, ValueError):
        return 0.0
    if lo > hi:
        lo, hi = hi, lo
    if lo <= val <= hi:
        return 1.0
    width = max(1e-9, hi - lo)
    m = max(1e-9, margin_frac * width)
    if val < lo:
        return max(0.0, 1.0 - (lo - val) / m)
    return max(0.0, 1.0 - (val - hi) / m)


# v191 (v5.06) — paper-derived default target bands, used by the
# band-completeness term when the trial metrics dict carries no
# `healthy_bands` (e.g. legacy result dicts / unit tests). Mirrors
# architectures/default.json["healthy_bands"]. The trinary triad (M1,
# neutral, inhibitory) is scored separately by its own tent rewards;
# these are the SECONDARY health metrics whose collective in-band score
# forms the band-completeness component.
_PAPER_BANDS = {
    'M2_mean_gate':              (0.40, 0.85),
    'M3_pac_modulation_idx':     (0.005, 0.10),
    'M5_branching_ratio':        (0.92, 1.10),
    'M6_spontaneous_fraction':   (0.10, 0.45),
    'M7_zero_input_mi_ratio':    (0.40, 1.20),
    'M9_transfer_ratio':         (0.85, 1.30),
    'input_saturation_fraction': (0.0, 0.30),
    'input_locked_fraction':     (0.0, 0.20),
    'exploration_trigger_rate':  (0.01, 0.40),
}

# Maps each band-completeness metric to the trial-metrics key that carries
# its last-20%-mean value (set in run_one_trial's result['metrics']).
_BAND_METRIC_KEYS = {
    'M2_mean_gate':              'M2_last',
    'M3_pac_modulation_idx':     'M3_last',
    'M5_branching_ratio':        'M5_last',
    'M6_spontaneous_fraction':   'M6_last',
    'M7_zero_input_mi_ratio':    'M7_last',
    'M9_transfer_ratio':         'M9_last',
    'input_saturation_fraction': 'input_sat_last',
    'input_locked_fraction':     'locked_last',
    'exploration_trigger_rate':  'expl_rate_last',
}


def _compute_band_completeness(metrics: Dict[str, Any]) -> float:
    """v191 (v5.06) — mean graded in-band score over the nine SECONDARY
    healthy-band metrics (M2/M3/M5/M6/M7/M9/saturation/lock/exploration).

    Read against the architecture's own `healthy_bands` if present, else
    the paper defaults. Shared by fitness() (as the 2.0-weighted
    band-completeness component) and the CSV writer (logged per trial so
    the full healthy profile is visible without re-running). The trinary
    triad (M1, neutral, inhibitory) is NOT included here — it is scored by
    its three dedicated tent rewards. Returns 0.0 if no metric is present.
    """
    bands = metrics.get('healthy_bands') or {}
    band_scores = []
    for band_key, mkey in _BAND_METRIC_KEYS.items():
        if mkey not in metrics:
            continue
        lo_hi = None
        if isinstance(bands, dict) and band_key in bands:
            try:
                rng = bands[band_key]
                lo_hi = (float(rng[0]), float(rng[1]))
            except (TypeError, ValueError, IndexError):
                lo_hi = None
        if lo_hi is None:
            lo_hi = _PAPER_BANDS.get(band_key)
        if lo_hi is None:
            continue
        band_scores.append(_band_score(metrics.get(mkey, 0.0), lo_hi[0], lo_hi[1]))
    return (sum(band_scores) / len(band_scores)) if band_scores else 0.0


def fitness(metrics: Optional[Dict[str, float]]) -> float:
    """Combine trial metrics into a single fitness score (higher = better).
    
    Components (v191 / v5.06 — ceiling 13.0):
      surv               2.0   population survival (mean-of-checkpoints)
      M10 heritability   1.5   selection signal (abs(M10_peak))
      streak             1.0   lock-in penalty (1 - mean_streak/100)
      --- TRINARY TRIAD (the paper's defining +1/0/-1 property) = 4.5 ---
      M1   (+1) tent     1.5   peak 0.22
      neutral (0) tent   1.5   peak 0.68 (upper side → 0 at 0.85)
      inh  (-1) tent     1.5   peak 0.10 (v191 — completes the triad)
      ------------------------------------------------------------------
      sm_corr            0.5   sensory→motor coupling
      band_completeness  2.0   mean in-band score over the nine SECONDARY
                               healthy_bands metrics (M2/M3/M5/M6/M7/M9/
                               saturation/lock/exploration) — v191, the
                               "all the metrics found in NAS" objective
      g_health         gw≤1.5  population g (gated on a healthy positive
                               manifold near PC1≈0.27); pinned 1.5 in v190

    A model that is paper-faithful on the full trinary triad AND lands
    every secondary band AND survives AND shows a healthy positive
    manifold reaches the 13.0 ceiling (verified test_v191). The whole
    healthy_bands dict is now the objective, so the NAS selects the
    "right model" where ALL metrics — not just M1/neutral — sit in band.

    v191 (v5.06) over v190:
      1. TRINARY TRIAD completed — explicit inhibitory tent (target 0.10),
         weight 1.5, matching M1 and neutral. The joint optimum of the
         three tents is exactly the paper mix 0.22 / 0.68 / 0.10, and the
         binary-oscillator regime (inhibitory≈0.65) now earns ~0 on it.
      2. BAND COMPLETENESS (2.0) — every secondary healthy_bands metric
         now drives selection (was: M2/M3/M5/M7/saturation invisible).
      3. The E/I ADAPTATION-BALANCE lever (adaptation_target_*_multiplier)
         is wired into the dynamics and added to SEARCH_SPACE — the
         mechanism that lets the search reach 22% +1 without collapsing
         the rest band (the v190 M1-vs-rest tension the prior fitness
         could not resolve). See CHANGELOG_v191.

    Returns float; -1.0 for failed trials, 0.0 for degenerate trials
    (peak_alive < 8, or M1 < 0.08 — the silence floor).
    """
    if metrics is None:
        return -1.0

    # v187 (v5.02) — POPULATION FLOOR. A "population" of one isolated
    # NxEr cannot demonstrate heritability, sensorimotor coupling, or any
    # of the multi-individual metrics; its sm_corr=1.0 and M1-by-luck are
    # mathematical artefacts of a sample of one. The v186 NAS log showed
    # 11/39 trials collapsing to peak_alive=1 yet earning fitness 4.5-5.5
    # via these spurious signals.
    #
    # v188 (v5.03) — RAISED from 3 to 8. The v187 floor at 3 still let
    # trial 28 (peak_alive=5) reach rank 3 with fitness 5.46 — a 5-NxEr
    # "population" can't sustain generational evolution. Raising to 8
    # forces the search toward architectures that simultaneously hit M1
    # AND sustain a real reproducing population. 8 is the minimum where
    # there's room for multiple breeding pairs + their offspring lineage
    # to be statistically distinguishable. Re-scoring the v187 NAS log
    # against the new floor: of the 56 v187 healthy trials, 8 trials had
    # peak_alive in [3, 7] (now floored to 0); the top-of-list churn is
    # mild but the trial-28 family (peak=5, M1 in band but fragile) no
    # longer competes against trials 93/105 (peak 41/46, the genuinely
    # robust architectures).
    #
    # v192 (v5.07) — GRADED POPULATION VIABILITY (was: hard floor peak<8 → 0).
    # Two compounding bugs made the v191 8h NAS score EVERY trial 0.0:
    #   (1) peak_alive was read from the in-RAM time_series, which is trimmed
    #       to the last max_history_length (=10000) samples (v185+). A
    #       multi-hour trial produces far more than 10000 samples, so the
    #       early high-population phase is trimmed away and peak_alive reads
    #       the recent (crashed) window — ~1 even for trials that sustained
    #       ~10 NxErs for hours. v192 fixes this at the source: the NAS now
    #       reads SurvivabilityTracker.max_alive_ever (never trimmed) into
    #       metrics['peak_alive'], so the value below is the TRUE peak.
    #   (2) the floor zeroed the ENTIRE fitness, so once every trial's
    #       (artefactual) peak was < 8 the leaderboard was all-zero and the
    #       optimiser had no gradient. Even with the honest peak, a hard
    #       floor still throws away the trinary distribution + all band info
    #       the moment a population is small — antithetical to "find the
    #       model where ALL metrics are in band."
    # Fix: a graded viability MULTIPLIER. peak ≥ 8 → 1.0 (IDENTICAL to v191,
    # so viable trials and the 13.0/14.0 ceiling are unchanged); peak ≤ 1 →
    # 0.10 (heavily penalised but not zeroed, so a tiny-population trial is
    # still RANKED by its trinary fidelity); linear ramp between. Combined
    # with the honest survival signal (mean_alive_ever) this gives the search
    # a continuous gradient toward sustainable populations instead of a cliff.
    peak_alive = metrics.get('peak_alive', 0.0)
    if peak_alive >= 8.0:
        pop_viability = 1.0
    elif peak_alive <= 1.0:
        pop_viability = 0.10
    else:
        pop_viability = 0.10 + 0.90 * (peak_alive - 1.0) / 7.0

    # v190 (v5.05) — M1 FLOOR (excitatory-firing floor). The v189 8h
    # extended NAS (163 trials) exposed a "quiescent trap": the four
    # highest-fitness trials all fired < 5% excitatory (M1 0.009-0.050,
    # paper target 0.22). They won on survival + heritability + g-factor
    # while contributing ~0 from the M1 component — a near-silent network
    # maxes survival (less metabolic cost = less death) and accumulates
    # clean heritability, so the 1.5-weight M1 reward couldn't overcome
    # the survival advantage of silence. A network firing < 8% excitatory
    # is not demonstrating the paper's trinary dynamics regardless of how
    # well it survives; it's a quiet binary (rest/inhibit) oscillator.
    # Same pattern as the population floor: below the threshold, the
    # architecture is disqualified (fitness 0), not merely penalised.
    # 0.08 is well below the band floor (0.18) — it only excludes genuine
    # silence, not architectures that are merely below-target. Re-scoring
    # the v189 8h log: this floors trials 154/119/5/132 (the silent
    # top-4), promoting trial 120 (M1=0.181, L1-to-paper=0.115) to the
    # top — exactly the paper-faithful architecture we want to select.
    # v193 (v5.08) — GRADED M1 silence floor (was a hard cliff: M1<0.08 → 0).
    # The longitudinal v161–v192 analysis showed the hard cutoff sat exactly
    # where the best architectures cluster: in the v191 8h run trial 4
    # (M1=0.086) scored 6.99 while its near-twin trial 5 (M1=0.077, and a
    # MORE dynamic rest band) scored 0 — a 0.009 M1 margin producing the whole
    # fitness. And the single closest-to-paper trinary ever found (a v189 trial
    # at 0.224/0.658/0.118, L1=0.044) was zeroed too. The cliff both inverts
    # selection at the boundary and destroys the gradient there. v193 makes it
    # a graded multiplier (applied with pop_viability at the return): M1≥0.08 →
    # ×1.0 (identical to v192 for every non-silent trial, so the ceiling and
    # all viable rankings are unchanged), M1≤0.03 → ×0.05 (genuine silence,
    # heavily suppressed but still RANKED), linear between. Silence stays
    # uncompetitive; the boundary cliff is gone.
    m1_floor_val = metrics.get('M1_last', 0.0)
    if m1_floor_val >= 0.08:
        m1_viability = 1.0
    elif m1_floor_val <= 0.03:
        m1_viability = 0.05
    else:
        m1_viability = 0.05 + 0.95 * (m1_floor_val - 0.03) / (0.08 - 0.03)

    # Component 1 — population survival (normalised to 0-1)
    # v168 (v4.76) — use mean of 4 checkpoint samples (25/50/75/100% of trial)
    # instead of just final_alive. Old fitness credited a "transient crash and
    # recover" trial the same as "stable throughout" — a critical blind spot
    # exposed by the 1200s NAS run, where v166 champions (trial 38/34) scored
    # 6.97 at 900s but dropped to 6.17 at 1200s because they had been winning
    # the sprint while crashing mid-marathon. Mean-of-checkpoints rewards
    # sustained population maintenance throughout the trial duration.
    # Backward compat: falls back to final_alive if checkpoint metrics absent
    # (legacy v161-v167 result dicts).
    # v171 (v4.79) — normalise by the ACTUAL starting population
    # (metrics['start_alive']) rather than the hardcoded NAS_TRIAL_CONFIG
    # value. The v170 @2400s run used start_alive=20 NxErs but fitness still
    # divided by 10, so survival was trivially saturated for any trial
    # maintaining >10 NxErs (essentially all top trials). With this fix the
    # metric is duration- AND population-agnostic: maintaining 99% of starting
    # population scores 0.99 whether you start with 10 or 20 NxErs.
    alive_signal = metrics.get('alive_mean_checkpoints',
                                metrics.get('final_alive', 0.0))
    # Use the actual founder count as the denominator. Fall back through
    # start_alive → peak_alive → hardcoded 10 for backward compatibility.
    start_pop = metrics.get('start_alive', 0.0)
    if start_pop <= 0:
        # Legacy result dicts may lack start_alive; try peak as a sensible
        # upper bound, otherwise fall back to the hardcoded config value.
        start_pop = metrics.get('peak_alive', NAS_TRIAL_CONFIG['StartingNxErs'])
    if start_pop <= 0:
        start_pop = float(NAS_TRIAL_CONFIG['StartingNxErs'])
    surv = min(1.0, alive_signal / start_pop)
    
    # Component 2 — heritability (clamp + normalise)
    # v166 (v4.74) — use M10_peak (max |M10| over full trial) instead of
    # M10_last (mean of last 20%). Same diagnostic pattern as v165's
    # sm_corr fix: across 1333 v165 trials, 98 (7.4%) had M10_peak - M10_last > 0.3,
    # with some hitting |M10_peak|=1.0 mid-trial but M10_last drifting to -0.99
    # by trial end due to small-sample noise (heritability is computed on
    # only 4-8 parent-child pairs, so a single late birth can flip the
    # Pearson r dramatically). The architectural capability is in the peak;
    # the last-mean was noise. abs() is used because M10_peak is logged as
    # max(|r|) — a strong negative correlation is still evidence the
    # architecture has heritability structure, just measured at small N.
    # Hypothetical re-scoring of v165 data showed best fitness rising
    # from 6.875 → 6.970 (+0.10) with this change alone.
    m10 = max(0.0, min(1.0, abs(metrics.get('M10_peak',
                                              metrics.get('M10_last', 0.0)))))
    
    # Component 3 — lock-in penalty (mean_state_streak)
    # streak ~2-5 is healthy, > 100 is severe lock-in
    streak = metrics.get('mean_streak_last', 0.0)
    streak_score = max(0.0, 1.0 - streak / 100.0)
    
    # Component 4 — M1 excitatory fraction, target 0.22 (paper-derived).
    # v187 (v5.02) GRADED REWARD. Pre-v187 was binary:
    #     m1_in_band = 1.0 if 0.18 <= m1 <= 0.30 else 0.0
    # This had no gradient: search couldn't tell whether moving from
    # m1=0.05 toward 0.18 was progress, nor whether m1=0.32 was close to
    # the band. Across 743 v161 + 30 v185 + 39 v186 trials, the only
    # configurations reaching the band were the degenerate N=1 ones
    # (where M1 is whatever the single NxEr's neurons happen to do).
    # The tent function below peaks at 1.0 at m1=0.22 (paper target)
    # and falls linearly to 0 at m1=0 or m1=0.44. Combined with the
    # population floor above, this directs search toward architectures
    # that genuinely sustain ~22% excitatory firing across a real
    # population, not just lucky single-NxEr quiescence.
    m1 = metrics.get('M1_last', 0.0)
    M1_TARGET = 0.22
    m1_score = max(0.0, 1.0 - abs(m1 - M1_TARGET) / M1_TARGET)

    # Component 4b — M1_neutral_fraction (state=0, "buffer"), target 0.68
    # (paper-derived). v189 (v5.04). Added to fix the v188 "rest band
    # collapse" pathology: trial 46 (v188 best, fitness 6.61) had M1 in
    # band (0.22) but the live game showed state=0 occupying only 4.1%
    # of hidden samples — the paper target is 68%. The trinary
    # architecture degenerates into a +1/-1 binary oscillator with no
    # rest band; M5 branching ratio doesn't catch this, mean_state_streak
    # catches partial lock-in but not the alternation regime, and M1
    # alone is satisfied by ANY 22% excitatory firing rate including
    # 22% +1 / 78% -1 / 0% neutral.
    #
    # Tent peaks at 1.0 at neutral=0.68 (paper). Asymmetric in
    # consequence: the lower side falls 0.68 wide (zero at 0%), the
    # upper side falls only 0.32 wide (zero at 100%) — a quiescent
    # network (all neurons in rest) gets harshly penalised but
    # progressively, while collapse-to-binary (neutral≈0) hits the
    # floor immediately. Combined with the M1 component (which rewards
    # +1 firing), the joint optimum is exactly the paper trinary
    # distribution: ~22% +1 / ~68% 0 / ~10% −1 (inhibitory implied by
    # the remainder). Weight 1.5 matches the M1 component because the
    # rest-band collapse is as severe a paper-fidelity failure as M1
    # being out of band.
    neutral = metrics.get('M1_neutral_last', 0.0)
    NEUTRAL_TARGET = 0.68
    if neutral <= NEUTRAL_TARGET:
        # Below target: linear ramp 0 → 1 over [0, 0.68]
        neutral_score = neutral / NEUTRAL_TARGET
    else:
        # Above target: v190 (v5.05) — upper side TIGHTENED. Was
        # `(neutral - 0.68) / (1.0 - 0.68)` (fall to 0 at neutral=1.0),
        # which let an over-quiet network at neutral=0.88 still score
        # 0.375. That's how the v189 quiescent winners (neutral 0.86-0.88)
        # kept meaningful neutral credit. v190 falls to 0 at neutral=0.85
        # — a network resting >85% of the time is too quiet for paper
        # trinary (target rest band 68%, with 22% excitatory + 10%
        # inhibitory firing). Combined with the M1 floor above, the
        # over-quiescent corner of the search is now doubly penalised.
        NEUTRAL_UPPER_ZERO = 0.85
        neutral_score = max(0.0, 1.0 - (neutral - NEUTRAL_TARGET) / (NEUTRAL_UPPER_ZERO - NEUTRAL_TARGET))

    # v193 (v5.08) — DYNAMISM GATE on the rest-band reward. This is the single
    # highest-leverage finding of the v161–v192 longitudinal analysis. The
    # neutral fraction alone cannot distinguish a DYNAMIC rest band (neurons
    # cycling through state 0 between firings — the paper's intent) from a
    # FROZEN one (neurons parked at 0 for hundreds of ticks). Since the v189
    # rest-band reward was added, the search has been won by progressively
    # FREEZING networks that park neurons at 0 to farm neutral credit while
    # their live +1 firing decays toward silence: the v191 champion froze to
    # mean_state_streak≈307, the v192 champion to ≈67, and the best survivors
    # sat at streak 250–14000. Frozen rest is not the paper's trinary dynamics;
    # it is the quiescent/lock-in trap in disguise. So the neutral reward is
    # now multiplied by a dynamism factor: full credit when the rest band
    # actually cycles (streak ≤ STREAK_DYNAMIC), zero when it is frozen
    # (streak ≥ STREAK_FROZEN), linear between. A genuinely dynamic paper-
    # faithful network (streak ~2–15, neutral ~0.68) is unaffected — the 13.0
    # ceiling is preserved — while the freeze-for-rest-band exploit is closed.
    # On the v192 run this widens the dynamic winner's (trial 14, streak 67)
    # lead over the frozen runner-up (trial 4, streak 300) from 0.53 to ~1.3.
    STREAK_DYNAMIC = 15.0
    STREAK_FROZEN = 120.0
    neutral_dynamism = max(0.0, min(1.0,
                           (STREAK_FROZEN - streak) / (STREAK_FROZEN - STREAK_DYNAMIC)))
    neutral_score *= neutral_dynamism

    # Component 5 — sensory→motor coupling (v165 redesigned)
    # Why peak instead of last: v164 added neural.sensorimotor_coupling
    # which biases input→output edge probability. The architectural fix
    # WORKS — KeyMetrics traces show sm_corr reaching 0.5-0.7+ early in
    # trials. But by trial end, input sensors saturate (input_locked_fraction
    # → 1.0), zeroing the rolling-window Pearson correlation. v161-v164
    # used `sm_corr_last` which captured only the post-saturation zero,
    # hiding the actual capability. v165 uses `sm_corr_peak` (max |sm_corr|
    # over the full trial) — rewards architectures that CAN couple
    # sensors to motors, even if saturation later kills the measurement.
    # Falls back to sm_corr_last if peak isn't in metrics (for backward
    # compat with old result dicts).
    sm = abs(metrics.get('sm_corr_peak',
                          metrics.get('sm_corr_last', 0.0)))
    sm_score = min(1.0, sm * 5.0)  # 0.2 corr = full points

    # Component 4c — M1_inhibitory_fraction (state=-1), target 0.10
    # (paper-derived). v191 (v5.06). COMPLETES THE TRINARY TRIAD. Pre-v191
    # the fitness rewarded only +1 (M1) and the rest band (neutral); the
    # inhibitory fraction was implicit (≈ 1 - M1 - neutral) and never scored.
    # The cost showed in the v190 8h run: the binary-oscillator regime
    # (M1≈0.24, neutral≈0.12, INHIBITORY≈0.65) was penalised only via its
    # collapsed neutral — its runaway inhibition went unpunished, so it
    # still scored ~3.7-5.0. And even the rest-band-healthy winner t110
    # fired 23% inhibitory (vs the paper's 10%) with no fitness pressure to
    # correct it. The trinary distribution is the paper's DEFINING property
    # (Neuraxon v2.0 Eq. 2: +1 / 0 / -1), so all three states must be
    # rewarded explicitly. Tent peaks at 1.0 at inhibitory=0.10, falls to 0
    # at inhibitory=0 (lower) and inhibitory=0.30 (upper). Weight 1.5 ==
    # the M1 and neutral weights: the joint optimum of the three tents is
    # now EXACTLY the paper trinary mix 0.22 / 0.68 / 0.10, and the binary
    # regime (inhibitory≈0.65) earns 0 here on top of ~0 neutral.
    inh = float(metrics.get('M1_inh_last', 0.0))
    INH_TARGET = 0.10
    INH_UPPER_ZERO = 0.30
    if inh <= INH_TARGET:
        inh_score = max(0.0, inh / INH_TARGET)
    else:
        inh_score = max(0.0, 1.0 - (inh - INH_TARGET) / (INH_UPPER_ZERO - INH_TARGET))

    # Component 7 — BAND COMPLETENESS (v191, v5.06). The headline change for
    # "all the metrics found in NAS so we find the right model." v161-v190
    # threaded only M1/neutral/M10/streak/sm (+g) into fitness, so the
    # search had NO gradient toward the other healthy_bands metrics — M2
    # gate, M3 PAC, M5 branching ratio, M6 spontaneous, M7 zero-input MI,
    # M9 transfer ratio, input saturation, input lock, exploration. An
    # architecture could top the leaderboard while sitting out of band on
    # most of them (the v190 winner was never selected on its M5/M7 at
    # all). This term is the MEAN graded in-band score over those nine
    # secondary metrics (the trinary triad M1/0/-1 is scored separately
    # above), read against the architecture's own healthy_bands (falling
    # back to the paper defaults). Weight 2.0: a model healthy on
    # EVERYTHING earns the full 2.0; one good only on the triad + survival
    # tops out ~2.0 lower. This makes "land every metric in band" the
    # explicit optimisation target without letting any single secondary
    # metric dominate the trinary triad.
    band_completeness = _compute_band_completeness(metrics)

    score = (
        2.0 * surv +         # 0-2   survival (v190: down from 3.0)
        1.5 * m10 +          # 0-1.5 (heritability — abs(M10_peak))
        1.0 * streak_score + # 0-1.0 (lock-in penalty)
        1.5 * m1_score +     # 0-1.5 trinary triad: +1 fraction, target 0.22
        1.5 * neutral_score +# 0-1.5 trinary triad: rest band, target 0.68
        1.5 * inh_score +    # 0-1.5 trinary triad: -1 fraction, target 0.10 (v191)
        0.5 * sm_score +     # 0-0.5 (sensory→motor coupling)
        2.0 * band_completeness  # 0-2.0 all secondary healthy bands (v191)
    )

    # Component 6 (v179, v4.87) — OPT-IN population g-factor term.
    # Default weight 0.0 ⇒ this is identically inert and v179 fitness is
    # bit-identical to v178 unless the architecture sets
    # operating_ranges.fitness_g_weight > 0. Design (applying the lesson
    # from the v178 M10 analysis): reward *healthy* g, NOT raw magnitude —
    # a positive manifold whose strength sits near the paper's
    # human-comparable PC1 band (~0.27) AND whose mean off-diagonal r is
    # positive (a genuine positive manifold, not an anti-correlated
    # artefact). g_health ∈ [0,1]; a strong negative mean-r contributes 0.
    gw = max(0.0, float(metrics.get('fitness_g_weight', 0.0)))
    if gw > 0.0:
        pc1 = max(0.0, float(metrics.get('g_pc1_peak',
                                         metrics.get('g_pc1_last', 0.0))))
        meanr = float(metrics.get('g_meanr_last', 0.0))
        posman = max(0.0, min(1.0, float(metrics.get('g_posman_last', 0.0))))
        # Bell-shaped reward peaking at the paper's PC1≈0.27 (human-comparable),
        # zero at PC1=0 or PC1≥~0.55 (degenerate single-factor collapse).
        pc1_health = math.exp(-((pc1 - 0.27) ** 2) / (2 * 0.12 ** 2)) if pc1 > 0 else 0.0
        # Gate on a genuinely POSITIVE manifold (mean-r > 0 and majority of
        # off-diagonal correlations positive). Negative mean-r ⇒ no credit.
        manifold_gate = max(0.0, math.tanh(5.0 * meanr)) * posman
        g_health = pc1_health * manifold_gate          # ∈ [0,1]
        score += gw * g_health                          # 0 .. gw

    # v192 (v5.07) — apply the graded population-viability multiplier. For a
    # viable trial (true peak ≥ 8) this is ×1.0, so the score and the ceiling
    # are IDENTICAL to v191. For a small/crashed population it scales the score
    # down toward 0.10× (peak=1) rather than zeroing it, so the leaderboard
    # keeps a continuous gradient over the trinary distribution + bands instead
    # of collapsing to all-zeros the way the v191 8h run did.
    # v192 (v5.07) — apply the graded population-viability multiplier; v193
    # (v5.08) — also apply the graded M1 silence multiplier. Both are ×1.0 for
    # a viable, non-silent trial (true peak ≥ 8 and M1 ≥ 0.08), so the score and
    # the 13.0 ceiling are IDENTICAL to v191 for every architecture that clears
    # both bars. Small/crashed populations and near-silent networks are scaled
    # down toward 0.10×/0.05× rather than zeroed, so the leaderboard keeps a
    # continuous gradient over the trinary distribution + bands instead of
    # collapsing to all-zeros or cliff-edged ties.
    return float(score * pop_viability * m1_viability)



# =============================================================================
# ORCHESTRATION — v160 (v4.68) STREAMING + THREADED-SAVER REDESIGN
# =============================================================================
# v159 used pool.map() per batch. Failure modes the user reported:
#   1. One slow worker blocks pool.map for the whole batch
#   2. Pool teardown between batches hangs (pygame subprocess threads
#      don't always exit cleanly; pool.join() waits forever)
#   3. The "best" never gets written if pool.map never returns
#
# v160 replaces this with a streaming design:
#   - ONE pool, created once, reused for all trials (no per-batch
#     teardown)
#   - Trials submitted via apply_async with a callback
#   - The callback drops finished results onto a Queue (thread-safe)
#   - A dedicated SAVER THREAD reads from the queue, compares fitness
#     against the running global best, and writes nas_best.json /
#     nas_top1-3.json / nas_log.csv IMMEDIATELY when a new winner appears
#   - The main scheduler loop submits new trials as soon as in-flight
#     count drops below `workers` — no "batch" boundary
#   - Ctrl-C signals graceful shutdown: stop submitting, wait for
#     in-flight, write final best
# =============================================================================

# Default scheduler poll interval. Small enough for snappy submission,
# large enough to not burn CPU on the scheduler thread.
_SCHEDULER_POLL_SEC = 0.5


def _write_csv_header(csv_path: Path):
    """Write a fresh CSV header. v160 simplified — one row per finished
    trial. v165 added sm_corr_peak, sm_corr_mean, M10_peak columns.
    v168 adds alive_q1, alive_q2, alive_q3, alive_mean checkpoint columns
    (alive_mean is the new survival fitness driver — see fitness() docstring).
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial_id', 'completed_at_iso', 'fitness', 'selection_score',
            'final_alive', 'peak_alive',
            'alive_q1', 'alive_q2', 'alive_q3', 'alive_mean',
            'M1', 'M1_neutral', 'M1_inh',   # v189 — trinary balance visibility
            # v191 (v5.06) — the remaining healthy_bands metrics + the
            # band-completeness composite, so the full healthy profile is
            # in the log (the "all metrics in NAS" deliverable).
            'M2', 'M3', 'M5', 'M7', 'input_sat', 'band_complete',
            'M6', 'M9',
            'M10', 'M10_peak',
            'mean_streak_last', 'mean_streak_first',
            'locked', 'surv_score', 'expl_rate',
            'sm_corr', 'sm_corr_peak', 'sm_corr_mean',
            'g_pc1', 'g_pc1_peak', 'g_posman', 'g_meanr', 'g_l1l2',
            'n_samples', 'wall_actual_s', 'error',
            # v186 (v5.01) — round-visibility. With NAS_TRIAL_CONFIG's
            # max_rounds=1, total_rounds is always 1 and went_extinct
            # captures whether the trial hit game-over before the budget.
            'total_rounds', 'went_extinct', 'extinction_tick', 'min_alive',
            # v192 (v5.07) — honest-peak diagnostic columns (the value the old
            # trimmed code saw, and the honest time-average), finally persisted.
            'peak_alive_trimmed', 'mean_alive_ever',
            # v193 (v5.08) — seed-repeat reliability: how many repeats, and the
            # per-seed spread of M1 and of fitness. Large fitness_std ⇒ the
            # architecture's score is noisy / unreliable.
            'n_repeats', 'n_repeats_ok', 'M1_std', 'fitness_std_reps',
            'arch_summary', 'is_global_best',
        ])


def _append_csv_row(csv_path: Path, result: Dict, is_global_best: bool):
    """Append a single finished trial's row. Called from the saver
    thread, never from the worker callback (avoids file lock contention
    on Windows). v165 — extra columns for M10_peak + sm_corr peak/mean.
    v168 — extra columns for checkpoint alive counts (q1/q2/q3/mean)."""
    m = result.get('metrics') or {}
    row = [
        result['trial_id'],
        time.strftime('%Y-%m-%dT%H:%M:%S'),
        f"{result['fitness']:.4f}",
        # v196 — the LCB selection score the search actually ranks by (falls
        # back to fitness when not yet computed, e.g. a failed/legacy row).
        f"{result.get('selection_score', result['fitness']):.4f}",
        m.get('final_alive', ''), m.get('peak_alive', ''),
        # v168 — checkpoint columns
        m.get('alive_q1', ''), m.get('alive_q2', ''),
        m.get('alive_q3', ''),
        f"{m.get('alive_mean_checkpoints', 0):.2f}",
        f"{m.get('M1_last', 0):.4f}",
        # v189 (v5.04) — emit the full trinary balance to the log so it's
        # post-hoc analysable. M1_inh + M1_neutral + M1 ≈ 1 by construction;
        # we log two for legibility and the third is implicit.
        f"{m.get('M1_neutral_last', 0):.4f}",
        f"{m.get('M1_inh_last', 0):.4f}",
        # v191 (v5.06) — secondary healthy_bands metrics + band-completeness
        f"{m.get('M2_last', 0):.4f}",
        f"{m.get('M3_last', 0):.4f}",
        f"{m.get('M5_last', 0):.4f}",
        f"{m.get('M7_last', 0):.4f}",
        f"{m.get('input_sat_last', 0):.4f}",
        f"{_compute_band_completeness(m):.4f}",
        f"{m.get('M6_last', 0):.4f}",
        f"{m.get('M9_last', 0):.4f}", f"{m.get('M10_last', 0):.4f}",
        f"{m.get('M10_peak', 0):.4f}",
        f"{m.get('mean_streak_last', 0):.2f}",
        f"{m.get('mean_streak_first', 0):.2f}",
        f"{m.get('locked_last', 0):.4f}",
        f"{m.get('surv_score_last', 0):.4f}",
        f"{m.get('expl_rate_last', 0):.4f}",
        f"{m.get('sm_corr_last', 0):.4f}",
        f"{m.get('sm_corr_peak', 0):.4f}",
        f"{m.get('sm_corr_mean', 0):.4f}",
        f"{m.get('g_pc1_last', 0):.4f}",
        f"{m.get('g_pc1_peak', 0):.4f}",
        f"{m.get('g_posman_last', 0):.4f}",
        f"{m.get('g_meanr_last', 0):.4f}",
        f"{m.get('g_l1l2_last', 0):.4f}",
        result['n_samples'], f"{result['wall_seconds_actual']:.1f}",
        result['error'] or '',
        # v186 (v5.01) — round-visibility columns. With max_rounds=1
        # (NAS_TRIAL_CONFIG default) total_rounds is always 1; the trial
        # ends at first extinction (went_extinct=1, extinction_tick set)
        # or at wall budget (went_extinct=0, extinction_tick=-1).
        m.get('total_rounds', 1),
        1 if m.get('went_extinct', False) else 0,
        m.get('extinction_tick', -1),
        m.get('min_alive', ''),   # v191 — separates "full-budget limping" from healthy
        # v192 — honest-peak diagnostics
        m.get('peak_alive_trimmed', ''),
        f"{m.get('mean_alive_ever', ''):.3f}" if isinstance(m.get('mean_alive_ever'), (int, float)) else '',
        # v193 — seed-repeat reliability
        int(m.get('n_repeats', 1)),
        int(m.get('n_repeats_ok', 1)),
        f"{m.get('M1_std', 0):.4f}",
        f"{m.get('fitness_std_reps', 0):.4f}",
        arch_summary_string(result['arch']),
        1 if is_global_best else 0,
    ]
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_default_template() -> Dict[str, Any]:
    """v162 — load architectures/default.json as the canonical template.
    NAS best-arch outputs are merged into this so they're plug-and-play
    swappable for default.json. Returns empty dict if not found.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / 'architectures' / 'default.json',
        Path.cwd() / 'architectures' / 'default.json',
    ]
    for p in candidates:
        try:
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
    return {}


_DEFAULT_TEMPLATE: Optional[Dict[str, Any]] = None
def _get_default_template() -> Dict[str, Any]:
    """Lazy cached loader for the default template."""
    global _DEFAULT_TEMPLATE
    if _DEFAULT_TEMPLATE is None:
        _DEFAULT_TEMPLATE = _load_default_template()
    return _DEFAULT_TEMPLATE


def _save_arch_json(arch: Dict[str, Any], path: Path, rank: int,
                     trial_id: int, fitness_val: float):
    """Persist one architecture as a directly-loadable JSON.
    
    v162 — Output is PLUG-AND-PLAY compatible with architectures/default.json:
    contains all sections (_meta, biology, neural, operating_ranges,
    healthy_bands, genetic_lottery), with values from the NAS trial
    OVERRIDING the corresponding defaults. Any key the NAS didn't sample
    falls through to the default-template value, so the resulting file
    can be dropped in as a direct replacement.
    
    Atomic write via .tmp + os.replace.
    """
    # Start from the default-template (deep copy so we don't mutate it)
    template = copy.deepcopy(_get_default_template())
    
    # Strip non-section keys (like the original _meta from sampling) from arch
    sampled = {k: v for k, v in arch.items()
                if not (isinstance(k, str) and k.startswith('_'))}
    
    # Merge sampled values INTO template (sampled overrides defaults)
    for section, kv in sampled.items():
        if not isinstance(kv, dict):
            continue
        # Ensure the section exists in output
        if section not in template:
            template[section] = {}
        # Drop _doc fields from the template's section ONLY for keys
        # the NAS overrode (we want NAS-found values to be naked, but
        # we keep the rest of the docs for unchanged keys)
        for k, v in kv.items():
            template[section][k] = v
    
    # New _meta — completely replaces template _meta with NAS-run info
    template['_meta'] = {
        'name': f'nas_best_t{trial_id:03d}',
        'version': 'NxonArchNAS v1.5 (v196)',
        'description': (
            f'Architecture found by NAS — trial {trial_id}, fitness {fitness_val:.4f}. '
            'Plug-and-play compatible with architectures/default.json: drop this file '
            'into architectures/ or load with NEURAXON_ARCH=path/to/this.json python main.py'
        ),
        'source':    'NxonArchNAS',
        'rank':      rank,
        'trial_id':  trial_id,
        'fitness':   fitness_val,
        'saved_at':  time.strftime('%Y-%m-%dT%H:%M:%S'),
        'notes': [
            'Sections inherited from default.json:  healthy_bands (unchanged target ranges)',
            'Sections overridden by NAS:  biology, neural, operating_ranges, genetic_lottery',
            'Load with NEURAXON_ARCH=path/to/this.json python main.py',
        ],
    }
    
    # v160 — atomic write to avoid readers seeing partial JSON
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2)
    os.replace(tmp_path, path)


class _NASScheduler:
    """v160 — Streaming NAS scheduler with threaded saver.
    
    Architecture:
    
      ┌─────────────────────┐         ┌──────────────────┐
      │  Main thread:       │         │  Saver thread:   │
      │  scheduler loop     │         │  reads from queue│
      │                     │         │  writes JSON+CSV │
      │  pool.apply_async() │         │  updates best    │
      │  ↓                  │         └──────────────────┘
      │  worker subprocess  │              ↑
      │  ↓ (callback)       │              │
      │  results_queue ─────┼──────────────┘
      └─────────────────────┘
    
    Three threads in the parent process:
    - Main: scheduler loop submits new trials when slots free up
    - Pool result handler (created by mp.Pool): runs callbacks when
      worker subprocesses return
    - Saver: reads from queue, does all I/O serially
    """
    
    def __init__(self, num_trials: int, wall_seconds: int,
                  workers: int, seed: int, out_path: Path,
                  max_trials: Optional[int], verbose: bool,
                  strategy: Optional[_SearchStrategy] = None):
        self.num_trials = num_trials  # legacy: trials per batch (display only)
        self.wall_seconds = wall_seconds
        self.workers = workers
        self.seed = seed
        self.out_path = out_path
        self.max_trials = max_trials  # None = forever
        self.verbose = verbose
        
        # Shared state (lock-protected)
        self.lock = threading.Lock()
        self.global_best: Optional[Dict[str, Any]] = None
        self.global_top3: List[Dict[str, Any]] = []
        self.completed = 0
        self.submitted = 0
        self.errors = 0
        
        # Queue carries finished results from pool-callback thread → saver
        self.results_queue: 'queue.Queue[Optional[Dict]]' = queue.Queue()
        
        # Stop signals
        self.stop_submitting = threading.Event()  # set on Ctrl-C
        self.saver_done = threading.Event()        # set when saver exits
        
        self.csv_path = out_path / 'nas_log.csv'
        self.rng = random.Random(seed)
        # v164 — pluggable search strategy. Defaults to RandomSearchStrategy
        # for backward compatibility with v161-v163 behavior.
        self.strategy = strategy if strategy is not None else RandomSearchStrategy(self.rng)
        self.start_time = time.time()
    
    def _on_trial_complete(self, result: Dict[str, Any]):
        """Pool callback — runs in the Pool's result-handler thread when
        a worker subprocess returns. Computes fitness and puts the result
        on the queue. Must NEVER block — the pool handler is shared with
        other workers."""
        try:
            result['fitness'] = fitness(result.get('metrics'))
        except Exception:
            result['fitness'] = -1.0
        self.results_queue.put(result)
    
    def _on_trial_error(self, exc):
        """Pool error_callback — fires when a worker subprocess raises
        before returning. Build a synthetic failure record so the saver
        still sees something."""
        err_str = f"worker_exception: {type(exc).__name__}: {exc}"
        self.results_queue.put({
            'trial_id': -1, 'arch': {'biology': {}, 'neural': {},
                                       'operating_ranges': {}, 'healthy_bands': {}},
            'metrics': None, 'error': err_str,
            'n_samples': 0, 'wall_seconds_actual': 0.0,
            'fitness': -1.0,
        })
    
    def _saver_loop(self):
        """Dedicated thread — reads finished results, updates best,
        writes files. Runs serially so file I/O never races. Exits when
        it receives the sentinel `None`."""
        while True:
            try:
                r = self.results_queue.get(timeout=1.0)
            except queue.Empty:
                # Periodic check: if we've stopped submitting AND no
                # in-flight remain, we may be done
                continue
            if r is None:  # sentinel
                break
            
            try:
                with self.lock:
                    self.completed += 1
                    if r.get('error'):
                        self.errors += 1
                    
                    new_best = False
                    if r['fitness'] > -1.0:
                        # v196 — compute the LCB selection score ONCE (here in
                        # the saver) and attach it to the result. Selection of
                        # the global best, the top-3, and the strategy's elite
                        # pool all rank by THIS, not the raw median fitness, so
                        # a lucky single draw can't out-rank a consistent
                        # architecture with a tighter seed-repeat spread.
                        r['selection_score'] = selection_score(r)
                        best_key = (self.global_best.get('selection_score',
                                    self.global_best['fitness'])
                                    if self.global_best else None)
                        if (self.global_best is None
                            or r['selection_score'] > best_key):
                            self.global_best = r
                            new_best = True
                        # Update top-3 (by selection score)
                        self.global_top3 = sorted(
                            self.global_top3 + [r],
                            key=lambda x: -x.get('selection_score', x['fitness']))[:3]
                    
                    # v164 — feed every completed trial to the search
                    # strategy so it can update its archive (evolutionary
                    # uses this to maintain the elite pool). Done inside
                    # the saver thread so there's no race with sample().
                    try:
                        self.strategy.update(r)
                    except Exception as exc:
                        print(f"[NAS] strategy.update warning: {exc}", flush=True)
                    
                    # Write outputs IMMEDIATELY when a new best appears.
                    # If no new best, still update CSV but skip JSONs.
                    if new_best and self.global_best is not None:
                        _save_arch_json(self.global_best['arch'],
                                          self.out_path / 'nas_best.json',
                                          rank=1,
                                          trial_id=self.global_best['trial_id'],
                                          fitness_val=self.global_best['fitness'])
                        for i, top in enumerate(self.global_top3):
                            _save_arch_json(top['arch'],
                                              self.out_path / f'nas_top{i+1}.json',
                                              rank=i + 1,
                                              trial_id=top['trial_id'],
                                              fitness_val=top['fitness'])
                    elif self.global_top3:
                        # Top-3 might have rotated even if #1 didn't change
                        for i, top in enumerate(self.global_top3):
                            _save_arch_json(top['arch'],
                                              self.out_path / f'nas_top{i+1}.json',
                                              rank=i + 1,
                                              trial_id=top['trial_id'],
                                              fitness_val=top['fitness'])
                    
                    # Append CSV (per-trial)
                    is_best = new_best
                    _append_csv_row(self.csv_path, r, is_best)
                    
                    # Snapshot for progress print (outside lock)
                    best_fit = (self.global_best['fitness']
                                if self.global_best else -1.0)
                    completed = self.completed
                    submitted = self.submitted
                    errors = self.errors
                
                if self.verbose:
                    m = r.get('metrics') or {}
                    # v168 — show alive_mean (the new fitness driver) alongside final
                    alive = m.get('final_alive', '?')
                    alive_mean = m.get('alive_mean_checkpoints', alive)
                    streak = m.get('mean_streak_last', 0)
                    # v166 — show M10_pk (the new fitness driver) and sm_peak.
                    # M10_last was 0.027 mean across 1333 v165 trials while
                    # M10_peak averaged much higher; the small-N (4-8 pairs)
                    # Pearson computation produced noisy last-window values
                    # that hid the architectural heritability capability.
                    M10_pk = abs(m.get('M10_peak', m.get('M10_last', 0)))
                    sm_peak = m.get('sm_corr_peak', 0)
                    marker = " ★ NEW BEST" if new_best else ""
                    # v169 — show escape mode indicator
                    escape_marker = ""
                    if hasattr(self.strategy, 'in_escape_mode') and self.strategy.in_escape_mode:
                        escape_marker = " [escape]"
                    err_short = (' err' if r.get('error') else '')
                    # v168 — `alive` now shows mean checkpoints (sustained
                    # population) rather than just final count
                    alive_disp = f"{alive_mean:.1f}" if isinstance(alive_mean, (int, float)) else str(alive_mean)
                    print(f"[NAS] trial {r['trial_id']:>4}{err_short:>4} "
                          f"fit={r['fitness']:6.3f}  aliveQ={alive_disp:>4} "
                          f"streak={streak:>5.1f}  M10_pk={M10_pk:5.2f}  "
                          f"sm_peak={sm_peak:5.2f}  "
                          f"| done {completed:>4}/{submitted}  "
                          f"errs={errors:>3}  best={best_fit:6.3f}"
                          f"{marker}{escape_marker}", flush=True)
            except Exception as exc:
                # Saver thread must never die — catch and log
                print(f"[NAS] saver-thread warning: {exc}", flush=True)
                traceback.print_exc()
        
        self.saver_done.set()
    
    def _submit_one(self, pool):
        """Submit one new trial to the pool, increment submitted counter.
        v164 — architecture comes from self.strategy (pluggable: random,
        evolutionary)."""
        with self.lock:
            tid = self.submitted
            self.submitted += 1
        arch = self.strategy.sample(tid)
        args = (arch, tid, self.wall_seconds,
                self.seed + tid * 7919, str(self.out_path))
        pool.apply_async(_trial_worker, (args,),
                         callback=self._on_trial_complete,
                         error_callback=self._on_trial_error)
    
    def run(self):
        """Main entry — start saver thread, fill the pool, scheduler
        loop. Returns when max_trials is reached or Ctrl-C."""
        # Initialise CSV
        _write_csv_header(self.csv_path)
        
        # Start saver
        saver = threading.Thread(target=self._saver_loop,
                                   name='NAS-Saver', daemon=False)
        saver.start()
        
        # Create the pool ONCE, reuse for all trials
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(self.workers)
        
        if self.verbose:
            mode = ("continuous (Ctrl-C to stop)"
                    if self.max_trials is None
                    else f"{self.max_trials} trials")
            print(f"[NAS] Mode: {mode}")
            print(f"[NAS] Strategy: {self.strategy.describe()}")  # v164
            print(f"[NAS] wall_seconds/trial={self.wall_seconds}  "
                  f"workers={self.workers}")
            # v194 — make the seed-repeat budget math explicit at startup.
            _R = _get_trial_repeats()
            _mode = _get_repeats_mode()
            if _R > 1:
                if _mode == 'subdivide':
                    _per = max(6.0, self.wall_seconds / float(_R))
                    print(f"[NAS] repeats={_R} ({_mode}): wall PER ARCHITECTURE "
                          f"={self.wall_seconds}s → {_per:.0f}s/repeat × {_R} "
                          f"(cost-neutral vs single-trial; selection on the median)")
                else:
                    print(f"[NAS] repeats={_R} ({_mode}): {self.wall_seconds}s/repeat "
                          f"× {_R} = {self.wall_seconds * _R}s PER ARCHITECTURE "
                          f"({_R}× cost — deep single-arch mode, NOT for broad search)")
            print(f"[NAS] Output dir: {self.out_path}")
            print(f"[NAS] CSV log:    {self.csv_path}")
            print(f"[NAS] Best arch:  {self.out_path / 'nas_best.json'}")
            print(f"[NAS] Search dims: {len(SEARCH_SPACE)}")
            for k, vs in SEARCH_SPACE.items():
                print(f"[NAS]   {k}: {vs}")
            print()
        
        try:
            # Initial fill — submit `workers` trials right away
            for _ in range(self.workers):
                if (self.max_trials is not None
                    and self.submitted >= self.max_trials):
                    break
                self._submit_one(pool)
            
            if self.verbose:
                print(f"[NAS] Initial fill: {self.submitted} trials submitted, "
                      f"waiting for results...\n", flush=True)
            
            # Scheduler loop — submit a new trial whenever in-flight count
            # drops below workers. We measure in-flight as submitted - completed
            # (this is approximate but fine for scheduling).
            while not self.stop_submitting.is_set():
                # Cap check
                if (self.max_trials is not None
                    and self.submitted >= self.max_trials):
                    break
                
                with self.lock:
                    in_flight = self.submitted - self.completed
                
                # Submit replacements
                slots_free = max(0, self.workers - in_flight)
                for _ in range(slots_free):
                    if (self.max_trials is not None
                        and self.submitted >= self.max_trials):
                        break
                    self._submit_one(pool)
                
                time.sleep(_SCHEDULER_POLL_SEC)
        
        except KeyboardInterrupt:
            if self.verbose:
                print(f"\n[NAS] Ctrl-C received — stopping new submissions, "
                      f"waiting for {self.submitted - self.completed} in-flight trials...",
                      flush=True)
            self.stop_submitting.set()
        
        # Wait for all in-flight to finish
        if self.verbose:
            print(f"[NAS] All trials submitted ({self.submitted}). "
                  f"Waiting for completion...", flush=True)
        
        # Periodic progress while draining
        drain_start = time.time()
        while True:
            with self.lock:
                in_flight = self.submitted - self.completed
            if in_flight == 0:
                break
            if self.verbose and (time.time() - drain_start) % 10 < 1:
                print(f"[NAS] still draining: {in_flight} in flight, "
                      f"{self.completed}/{self.submitted} done",
                      flush=True)
            time.sleep(1.0)
        
        # Close pool — all callbacks should have fired by now
        try:
            pool.close()
            pool.join()
        except Exception as exc:
            print(f"[NAS] pool teardown warning: {exc}", flush=True)
            try: pool.terminate()
            except Exception: pass
        
        # Signal saver and wait
        self.results_queue.put(None)
        self.saver_done.wait(timeout=10)
        try:
            saver.join(timeout=5)
        except Exception:
            pass
        
        # Final summary
        total_elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"\n[NAS] === FINAL ===", flush=True)
            print(f"[NAS] Trials: completed={self.completed}/"
                  f"submitted={self.submitted}  errors={self.errors}")
            print(f"[NAS] Wall time: {total_elapsed:.0f}s "
                  f"({total_elapsed/60:.1f} min)")
            if self.global_best is not None:
                print(f"[NAS] Global best fitness: "
                      f"{self.global_best['fitness']:.3f}")
                print(f"[NAS] Best arch summary: "
                      f"{arch_summary_string(self.global_best['arch'])}")
                print(f"[NAS] Run a full game with:")
                print(f"      NEURAXON_ARCH={self.out_path / 'nas_best.json'} "
                      f"python main.py")
            else:
                print(f"[NAS] No successful trials.")
        
        return {
            'best_arch': (self.global_best['arch']
                          if self.global_best else None),
            'best_fitness': (self.global_best['fitness']
                              if self.global_best else -1.0),
            'completed': self.completed,
            'submitted': self.submitted,
            'errors': self.errors,
            'interrupted': self.stop_submitting.is_set(),
            'out_dir': str(self.out_path),
        }


def run_continuous(num_trials: int = 8,
                     wall_seconds: int = 60,
                     workers: Optional[int] = None,
                     seed: int = 42,
                     out_dir: Optional[str] = None,
                     batches: Optional[int] = None,
                     max_trials: Optional[int] = None,
                     verbose: bool = True,
                     search_strategy: str = 'evolutionary',
                     evo_elite_pool: int = 10,
                     evo_random_seed_trials: int = 50,
                     evo_p_random: float = 0.20,
                     evo_p_mutation: float = 0.50,
                     evo_p_crossover: float = 0.30,
                     evo_mutation_sigma: float = 0.15,
                     evo_escape_threshold: int = 30,
                     evo_escape_sigma: float = 0.35,
                     evo_escape_n_params: Tuple[int, int] = (3, 6),
                     seed_archs: Optional[List[Dict[str, Any]]] = None,
                     seed_archs_fitness: float = 6.0,
                     search_space_tier: str = 'full',
                     adaptive: bool = True,
                     guided_frac: float = 0.5,
                     select_k: Optional[float] = None,
                     ei_weight: Optional[float] = None) -> Dict[str, Any]:
    """v160 — Streaming NAS. v164 adds pluggable search_strategy.
    v169 — adds escape mechanism for evolutionary strategy when stuck.
    v196 — default strategy is 'evolutionary'; adds the focus-tier search space,
    the self-evolving controller (adaptive + guided immigrants), and LCB
    variance-aware selection (select_k / ei_weight).
    
    Args:
      num_trials: legacy display-batch size
      wall_seconds: per-trial time limit
      workers: parallel processes (default cpu_count - 1)
      seed: RNG seed for architecture sampling
      out_dir: output directory (default nas_runs/<timestamp>)
      batches: legacy v159 flag (max_trials = batches * num_trials)
      max_trials: total trial cap (None = forever until Ctrl-C)
      verbose: per-trial console output
      search_strategy: 'evolutionary' (v196 default) or 'random' (v161-v163)
      evo_*: evolutionary parameters when search_strategy='evolutionary'
      evo_escape_*: v169 escape parameters — the controller's tier-1 widths.
      search_space_tier: 'full' (35 genes), 'focus' (~12 high-leverage, rest
                         pinned) or 'core' (6 trinary/E-I levers, rest pinned).
      adaptive: v196 self-evolving controller on/off.
      guided_frac: fraction of immigrants drawn from the guided EDA prior.
      select_k: LCB strength (fitness − k·stderr); None → env/default (1.0).
      ei_weight: E/I-ordering selection bonus weight; None → env/default (0.5).
    """
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)
    
    if out_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_dir = f'nas_runs/{timestamp}'
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Translate batches → max_trials for backward compat
    if max_trials is None and batches is not None:
        max_trials = batches * num_trials

    # v196 — route the LCB knobs through the environment so the saver thread's
    # selection_score() (which reads env) matches the strategy's ranking, AND
    # so spawned workers inherit them. Explicit args win over any prior env.
    if select_k is not None:
        os.environ['NEURAXON_NAS_SELECT_K'] = str(select_k)
    if ei_weight is not None:
        os.environ['NEURAXON_NAS_EI_WEIGHT'] = str(ei_weight)
    eff_k = _get_select_k()
    eff_ei = _get_ei_weight()

    # v196 — build the active search space + pins for the chosen tier. In a
    # pinned tier the FIRST seed arch (if any) supplies the pin values; else
    # BEST_KNOWN_PINS. Seeds also pre-populate the elite pool below.
    active_space = space_for_tier(search_space_tier)
    pin_seed = seed_archs[0] if seed_archs else None
    pins = pins_for_space(active_space, seed_arch=pin_seed) \
        if len(active_space) < len(SEARCH_SPACE) else {}
    if verbose and pins:
        print(f"[NAS] search-space tier '{search_space_tier}': searching "
              f"{len(active_space)} genes, pinning {len(pins)} "
              f"({'from --seed-archs' if pin_seed else 'from BEST_KNOWN_PINS'})",
              flush=True)
    
    # v164/v196 — build the strategy (evolutionary by default)
    strategy_rng = random.Random(seed)
    strategy = make_strategy(
        search_strategy, strategy_rng,
        elite_pool_size=evo_elite_pool,
        random_seed_trials=evo_random_seed_trials,
        p_random=evo_p_random,
        p_mutation=evo_p_mutation,
        p_crossover=evo_p_crossover,
        mutation_sigma_frac=evo_mutation_sigma,
        # v169 escape knobs (ignored by random strategy)
        escape_threshold=evo_escape_threshold,
        escape_sigma_frac=evo_escape_sigma,
        escape_n_params=evo_escape_n_params,
        # v196 — tier space/pins + controller + LCB knobs
        space=active_space,
        pins=pins,
        adaptive=adaptive,
        guided_frac=guided_frac,
        select_k=eff_k,
        ei_weight=eff_ei,
    )
    
    # v170 (v4.78) — pre-populate elite pool with known-good architectures.
    # Only supported by EvolutionaryStrategy; random strategy ignores seeds.
    if seed_archs and hasattr(strategy, 'seed_with_archs'):
        strategy.seed_with_archs(seed_archs, assumed_fitness=seed_archs_fitness)
    elif seed_archs:
        print(f"[NAS] WARNING: --seed-archs/--resume requires evolutionary "
              f"strategy; ignoring {len(seed_archs)} seed archs", flush=True)
    
    scheduler = _NASScheduler(
        num_trials=num_trials, wall_seconds=wall_seconds,
        workers=workers, seed=seed, out_path=out_path,
        max_trials=max_trials, verbose=verbose,
        strategy=strategy,
    )
    return scheduler.run()


# v156-v158 backward-compat — single batch
# =============================================================================
# RESUME / WARM-START — v196 (v5.10)
# =============================================================================
# Compute should COMPOUND across runs rather than every search starting cold.
# --resume takes a previous run (a run directory, a nas_best/nas_top*.json, or a
# nas_log.csv) and returns its best architectures as seed archs: these
# pre-populate the elite pool AND — once they enter the archive — warm the
# guided EDA prior, so a fresh search starts from the prior run's winners and
# refines them instead of re-discovering them.

# Short-key → full dotted-key map, for reconstructing an arch from a CSV
# arch_summary string (the summary uses the bare key, e.g. "num_hidden_..").
_SHORT_TO_FULL = {full.split('.', 1)[1]: full for full in SEARCH_SPACE}
_SHORT_TO_FULL['metabolic_rate_multiplier_range'] = \
    'genetic_lottery.metabolic_rate_multiplier_range'


def _coerce_scalar(tok: str) -> Any:
    """Parse a value token from an arch_summary into bool/int/float/list/str."""
    tok = tok.strip()
    if tok in ('True', 'False'):
        return tok == 'True'
    if tok.startswith('[') and tok.endswith(']'):
        try:
            return [float(x) for x in tok[1:-1].split(',') if x.strip()]
        except ValueError:
            return tok
    try:
        f = float(tok)
        return int(f) if f.is_integer() and 'e' not in tok.lower() and '.' not in tok else f
    except ValueError:
        return tok


def _arch_from_summary(summary: str) -> Dict[str, Any]:
    """Reconstruct an architecture dict from a CSV arch_summary string.

    Display precision only (the summary rounds floats), which is fine for a
    SEED — it is re-evaluated and refined. Tokens are `key=value` space-joined;
    list values like metabolic_rate_multiplier_range=[a, b] contain a space, so
    we parse by scanning for `key=` boundaries rather than splitting on spaces.
    """
    arch: Dict[str, Any] = {
        '_meta': {'source': 'NxonArchNAS:resume'},
        'biology': {}, 'neural': {}, 'operating_ranges': {},
        'healthy_bands': {}, 'genetic_lottery': {},
    }
    import re as _re
    # split into key=value chunks: a key is a run of [a-z0-9_] immediately
    # followed by '='; the value runs until the next such key or end-of-string.
    pattern = _re.compile(r"([a-zA-Z0-9_]+)=(.*?)(?=\s+[a-zA-Z0-9_]+=|$)")
    for m in pattern.finditer(summary or ''):
        short, raw = m.group(1), m.group(2)
        full = _SHORT_TO_FULL.get(short)
        if not full:
            continue
        section, key = full.split('.', 1)
        arch.setdefault(section, {})[key] = _coerce_scalar(raw)
    return arch


def load_resume_seeds(path: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Load the best architectures from a previous run for warm-starting.

    `path` may be: a run directory (uses nas_best/nas_top*.json inside, else
    nas_log.csv), a single arch JSON, or a nas_log.csv. Returns up to `top_n`
    architecture dicts (de-duplicated), best first.
    """
    p = Path(path).expanduser().resolve()
    json_candidates: List[Path] = []
    csv_path: Optional[Path] = None
    if p.is_dir():
        for name in ('nas_best.json', 'nas_top1.json', 'nas_top2.json', 'nas_top3.json'):
            if (p / name).is_file():
                json_candidates.append(p / name)
        if (p / 'nas_log.csv').is_file():
            csv_path = p / 'nas_log.csv'
    elif p.suffix == '.json' and p.is_file():
        json_candidates.append(p)
    elif p.suffix == '.csv' and p.is_file():
        csv_path = p
        # prefer sibling JSONs if present (full precision)
        for name in ('nas_best.json', 'nas_top1.json', 'nas_top2.json', 'nas_top3.json'):
            if (p.parent / name).is_file():
                json_candidates.append(p.parent / name)

    seeds: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(arch: Dict[str, Any]):
        if not isinstance(arch, dict) or 'biology' not in arch:
            return
        # de-dup on the searched-gene signature
        sig = arch_summary_string(arch)
        if sig in seen:
            return
        seen.add(sig)
        seeds.append(arch)

    for jp in json_candidates:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                _add(json.load(f))
        except Exception as exc:
            print(f"[NAS] --resume: could not read {jp}: {exc}", flush=True)

    # Fall back to (or supplement with) the CSV's top rows.
    if csv_path is not None and len(seeds) < top_n:
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            def _key(row):
                try:
                    return float(row.get('selection_score') or row.get('fitness') or -1)
                except (TypeError, ValueError):
                    return -1.0
            rows.sort(key=_key, reverse=True)
            for row in rows:
                if len(seeds) >= top_n:
                    break
                _add(_arch_from_summary(row.get('arch_summary', '')))
        except Exception as exc:
            print(f"[NAS] --resume: could not parse {csv_path}: {exc}", flush=True)

    seeds = seeds[:top_n]
    print(f"[NAS] --resume: loaded {len(seeds)} seed architecture(s) from {p}",
          flush=True)
    return seeds


def run_search(num_trials: int = 8,
                 wall_seconds: int = 60,
                 workers: Optional[int] = None,
                 seed: int = 42,
                 out_dir: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, Any]:
    """v156-v158 API. v160 implements this as a one-shot streaming run
    with max_trials = num_trials."""
    return run_continuous(
        num_trials=num_trials, wall_seconds=wall_seconds,
        workers=workers, seed=seed, out_dir=out_dir,
        max_trials=num_trials, verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Neuraxon Architecture Search v1.0 — streaming scheduler '
                    'with threaded saver and pluggable search strategy. '
                    'Continuous by default.')
    parser.add_argument('--trials', type=int, default=8,
                        help='Initial pool size (also display-batch size) (default: 8). '
                             'Actual parallelism is set by --workers.')
    parser.add_argument('--wall-seconds', type=int, default=60,
                        help='Wall-clock seconds per trial (default: 60)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel processes (default: cpu_count - 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for architecture sampling (default: 42)')
    parser.add_argument('--repeats', type=int, default=None,
                        help='v193 — seed-repeats per architecture; selection uses '
                             'the per-metric median to fight measurement noise '
                             f'(default: {NAS_TRIAL_REPEATS}; 1 = v192 single-trial). '
                             'See --repeats-mode for how the wall budget is shared.')
    parser.add_argument('--repeats-mode', choices=['subdivide', 'multiply'], default=None,
                        help="v194 — how --wall-seconds relates to repeats. "
                             "'subdivide' (default): wall is PER ARCHITECTURE, each "
                             "repeat runs wall/repeats (cost-neutral vs v192, keeps "
                             "search breadth). 'multiply': each repeat runs the full "
                             "wall (repeats× cost; deep single-arch characterisation).")
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: nas_runs/<timestamp>)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-trial console output')
    parser.add_argument('--max-trials', type=int, default=None,
                        help='Total trial cap (default: run forever until Ctrl-C)')
    # v159 legacy — translated to max_trials = batches * trials
    parser.add_argument('--batches', type=int, default=None,
                        help='[v159 legacy] Run N batches of --trials each, then stop. '
                             'Equivalent to --max-trials (batches * trials).')
    parser.add_argument('--single-batch', action='store_true',
                        help='[v158 legacy] Equivalent to --max-trials --trials.')
    # v164 — pluggable search strategy
    parser.add_argument('--search-strategy', choices=['random', 'evolutionary'],
                        default='evolutionary',
                        help='Sampling strategy. "evolutionary" (v196 default) maintains '
                             'an elite pool ranked by the LCB selection score and generates '
                             'new trials via the self-evolving controller (adaptive mutation, '
                             'guided immigrants, mutation/crossover of elites). "random" '
                             '(v161-v163 behavior) samples each trial independently.')
    # v196 — search-space tier, controller, guided immigrants, LCB selection
    parser.add_argument('--search-space', choices=['full', 'focus', 'core'],
                        default='full',
                        help="v196 — which genes to search. 'full' (default, 35 genes); "
                             "'focus' (~12 high-leverage genes, the rest pinned at "
                             "--seed-archs or the best-known arch); 'core' (6 trinary/E-I "
                             "levers). Use 'focus'/'core' WITH --seed-archs to refine a "
                             "known-good architecture along only the levers that matter.")
    parser.add_argument('--no-adaptive', action='store_true',
                        help='v196 — freeze the self-evolving controller (constant '
                             'mutation/operator knobs + v169 binary escape).')
    parser.add_argument('--guided-frac', type=float, default=0.5,
                        help='v196 — fraction of immigrants drawn from the guided '
                             '(winning-gene EDA) prior vs pure-random (default 0.5; '
                             '0 disables guided immigrants).')
    parser.add_argument('--select-k', type=float, default=None,
                        help='v196 — LCB selection strength: rank by fitness − k·stderr '
                             f'over the seed-repeats (default {NAS_SELECT_K}). 0 recovers '
                             'v195 raw-fitness selection.')
    parser.add_argument('--ei-weight', type=float, default=None,
                        help='v196 — weight of the E/I-ordering selection bonus that '
                             f'rewards excitation-dominant (M1≥inh) trinary (default '
                             f'{NAS_EI_WEIGHT}; 0 disables). Selection-only — fitness() '
                             'and its 13.0 ceiling are unchanged.')
    parser.add_argument('--resume', type=str, default=None,
                        help='v196 — warm-start from a previous run: pass a run directory, '
                             'a nas_best/nas_top*.json, or a nas_log.csv. Its best '
                             'architectures seed the elite pool AND the guided prior so '
                             'compute compounds across runs. Implies a small seed phase.')
    parser.add_argument('--evo-elite-pool', type=int, default=10,
                        help='[evolutionary] Number of top trials kept as parents (default: 10)')
    parser.add_argument('--evo-random-seed-trials', type=int, default=50,
                        help='[evolutionary] Initial random trials before switching to evo '
                             '(default: 50). Resume runs can set this low.')
    parser.add_argument('--evo-p-random', type=float, default=0.20,
                        help='[evolutionary] Probability of pure random sample (default: 0.20)')
    parser.add_argument('--evo-p-mutation', type=float, default=0.50,
                        help='[evolutionary] Probability of mutation operator (default: 0.50)')
    parser.add_argument('--evo-p-crossover', type=float, default=0.30,
                        help='[evolutionary] Probability of crossover operator (default: 0.30)')
    parser.add_argument('--evo-mutation-sigma', type=float, default=0.15,
                        help='[evolutionary] Gaussian noise stddev as fraction of param range '
                             '(default: 0.15)')
    # v169 (v4.77) — NAS escape mechanism CLI
    parser.add_argument('--evo-escape-threshold', type=int, default=30,
                        help='[evolutionary] Trials without improvement before switching to '
                             'escape mode (broad mutation). Set to a very large number to '
                             'disable. Default: 30.')
    parser.add_argument('--evo-escape-sigma', type=float, default=0.35,
                        help='[evolutionary] Mutation sigma during escape mode '
                             '(default: 0.35, vs normal 0.15)')
    parser.add_argument('--evo-escape-n-params-lo', type=int, default=3,
                        help='[evolutionary] Min params mutated per child in escape mode '
                             '(default: 3, vs normal 1)')
    parser.add_argument('--evo-escape-n-params-hi', type=int, default=6,
                        help='[evolutionary] Max params mutated per child in escape mode '
                             '(default: 6, vs normal 3)')
    # v170 (v4.78) — seed-architecture bootstrapping for cross-duration validation
    parser.add_argument('--seed-archs', type=str, default=None,
                        help='[evolutionary] Comma-separated list of JSON architecture '
                             'files to pre-populate the elite pool with. Use this for '
                             'cross-duration validation (e.g. take nas_top1.json from a '
                             '1800s run and re-evaluate at 2400s). The seeded archs are '
                             'evaluated first and then mutated. Recommend setting '
                             '--evo-random-seed-trials to a small number (5-10) so '
                             'evolutionary search starts immediately from the seeds.')
    parser.add_argument('--seed-archs-fitness', type=float, default=6.0,
                        help='[evolutionary] Placeholder fitness for seeded archs '
                             '(default 6.0). Higher → more likely to dominate the '
                             'elite pool until real evaluations complete.')
    args = parser.parse_args()

    # v193 — propagate --repeats. mp uses 'spawn', so a mutated module global
    # would NOT reach the worker subprocesses; the environment IS copied at
    # spawn time, and _get_trial_repeats() reads NEURAXON_NAS_REPEATS first
    # (falling back to the module default), so setting the env var is the one
    # mechanism that reaches both the parent and every worker.
    if args.repeats is not None:
        os.environ['NEURAXON_NAS_REPEATS'] = str(max(1, args.repeats))
    elif 'NEURAXON_NAS_REPEATS' not in os.environ:
        os.environ['NEURAXON_NAS_REPEATS'] = str(NAS_TRIAL_REPEATS)
    # v194 — repeats budget mode (subdivide | multiply), env-routed for spawn.
    if args.repeats_mode is not None:
        os.environ['NEURAXON_NAS_REPEATS_MODE'] = args.repeats_mode
    elif 'NEURAXON_NAS_REPEATS_MODE' not in os.environ:
        os.environ['NEURAXON_NAS_REPEATS_MODE'] = NAS_REPEATS_MODE
    
    # v170 — load seed architectures if specified
    seed_archs = None
    if args.seed_archs:
        from pathlib import Path
        seed_archs = []
        for path_str in args.seed_archs.split(','):
            path = Path(path_str.strip()).expanduser().resolve()
            if not path.is_file():
                print(f"[NAS] --seed-archs: file not found: {path}", flush=True)
                return 1
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # Strip _meta wrapper if present (saved arch files have it)
                if isinstance(raw, dict) and 'biology' in raw:
                    seed_archs.append(raw)
                else:
                    print(f"[NAS] --seed-archs: {path} doesn't look like an arch JSON "
                          f"(no 'biology' section)", flush=True)
                    return 1
                print(f"[NAS] Loaded seed arch from {path}", flush=True)
            except Exception as exc:
                print(f"[NAS] --seed-archs: failed to load {path}: {exc}", flush=True)
                return 1

    # v196 — --resume: warm-start from a previous run (dir / json / csv). The
    # discovered archs are prepended to any explicit --seed-archs and, because
    # they pre-populate the elite pool, also warm the guided EDA prior. Resuming
    # implies a small seed phase (so the search starts from the seeds, not 50
    # fresh randoms) unless the user set --evo-random-seed-trials explicitly.
    resume_used = False
    if args.resume:
        resume_seeds = load_resume_seeds(args.resume)
        if resume_seeds:
            seed_archs = (seed_archs or []) + resume_seeds
            resume_used = True
        else:
            print(f"[NAS] --resume: no usable architectures found in {args.resume}",
                  flush=True)
    if resume_used and args.evo_random_seed_trials == 50:  # untouched default
        args.evo_random_seed_trials = max(5, min(10, len(seed_archs)))
        print(f"[NAS] --resume: seed phase set to "
              f"{args.evo_random_seed_trials} trials (override with "
              f"--evo-random-seed-trials)", flush=True)
    
    max_trials = args.max_trials
    if args.single_batch:
        max_trials = args.trials
    elif args.batches is not None:
        max_trials = args.batches * args.trials
    
    summary = run_continuous(
        num_trials=args.trials,
        wall_seconds=args.wall_seconds,
        workers=args.workers,
        seed=args.seed,
        out_dir=args.out_dir,
        max_trials=max_trials,
        verbose=not args.quiet,
        search_strategy=args.search_strategy,
        evo_elite_pool=args.evo_elite_pool,
        evo_random_seed_trials=args.evo_random_seed_trials,
        evo_p_random=args.evo_p_random,
        evo_p_mutation=args.evo_p_mutation,
        evo_p_crossover=args.evo_p_crossover,
        evo_mutation_sigma=args.evo_mutation_sigma,
        evo_escape_threshold=args.evo_escape_threshold,
        evo_escape_sigma=args.evo_escape_sigma,
        evo_escape_n_params=(args.evo_escape_n_params_lo, args.evo_escape_n_params_hi),
        seed_archs=seed_archs,
        seed_archs_fitness=args.seed_archs_fitness,
        # v196 — tier / controller / guided / LCB
        search_space_tier=args.search_space,
        adaptive=not args.no_adaptive,
        guided_frac=args.guided_frac,
        select_k=args.select_k,
        ei_weight=args.ei_weight,
    )
    
    if summary['best_arch'] is not None:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
