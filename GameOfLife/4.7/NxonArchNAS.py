"""Neuraxon Architecture Search (NAS) — v0.1 (Game of Life v4.64 / v156)

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

Usage (CLI):
  python NxonArchNAS.py --trials 16 --wall-seconds 60 --workers 4

Usage (programmatic):
  from NxonArchNAS import run_search
  best = run_search(num_trials=8, wall_seconds=60, workers=4)
"""

import argparse
import copy
import csv
import json
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
    'neural.firing_threshold_excitatory':       ('uniform', 0.40, 0.70),
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
    # trinary dynamics. Range chosen to cover both "minimal" (1-3 ticks)
    # and "substantial buffer" (8-12 ticks) regimes.
    'neural.refractory_period_ticks':           ('int_uniform', 0, 12),
    
    # -- OPERATING RANGES: plasticity / adaptation --
    'operating_ranges.learning_rate':           ('loguniform', 0.002, 0.05),
    'operating_ranges.plasticity_threshold':    ('uniform', 0.3, 0.7),
    'operating_ranges.autoreceptor_coefficient':('uniform', 0.05, 0.25),
    'operating_ranges.adaptation_tau_ticks':    ('uniform', 10.0, 50.0),
    
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
}


# =============================================================================
# ARCHITECTURE GENERATION
# =============================================================================
def sample_random_architecture(rng: random.Random,
                                  trial_id: int) -> Dict[str, Any]:
    """Sample one architecture from SEARCH_SPACE.
    
    v162 — supports continuous (min, max) ranges, log-uniform, integer
    uniform, and discrete lists. Also normalizes genetic_lottery key
    pairs so the `lo` value is always < the `hi` value (the NAS samples
    them independently, but downstream code expects an ordered pair).
    """
    arch: Dict[str, Any] = {
        '_meta': {
            'source': 'NxonArchNAS',
            'trial_id': trial_id,
            'sampled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'biology': {}, 'neural': {}, 'operating_ranges': {},
        'healthy_bands': {}, 'genetic_lottery': {},
    }
    for full_key, spec in SEARCH_SPACE.items():
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

class _SearchStrategy:
    """Base class. Override `sample(trial_id)` to return an architecture
    dict. `update(result)` is called by the saver after each trial
    finishes — strategies that maintain state (elite pool, surrogate
    model, etc.) use this to incorporate new results."""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
    
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
    exploits past results."""
    
    def sample(self, trial_id: int) -> Dict[str, Any]:
        return sample_random_architecture(self.rng, trial_id)


class EvolutionaryStrategy(_SearchStrategy):
    """v164 — (μ + λ) evolution strategy with mixed mutation/crossover/
    random operators.
    
    State:
      - archive: every completed trial (with fitness), kept under a lock
        because the saver-thread updates it concurrently with the
        scheduler-thread reading it
      - n_random_seed: first N trials always use pure random sampling so
        the elite pool gets seeded with diversity (avoids premature
        convergence)
    
    Each sample after seeding picks one of three operators:
      - 20%  random (continued exploration; ensures we don't get stuck)
      - 50%  mutation (one elite, perturb 1-3 parameters with Gaussian
                       noise scaled to each param's range)
      - 30%  crossover (two elites, blend parameter-by-parameter)
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
                  escape_n_params: Tuple[int, int] = (3, 6)):
        super().__init__(rng)
        self.elite_pool_size = elite_pool_size
        self.random_seed_trials = random_seed_trials
        self.p_random = p_random
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        # v169 — store the "normal" mutation knobs and the "escape" knobs separately
        # so we can switch between them as `trials_since_last_best` grows
        self.mutation_n_params_normal = mutation_n_params
        self.mutation_sigma_frac_normal = mutation_sigma_frac
        self.mutation_n_params = mutation_n_params       # current (toggled)
        self.mutation_sigma_frac = mutation_sigma_frac   # current (toggled)
        # v169 — escape mode parameters
        self.escape_threshold = escape_threshold
        self.escape_sigma_frac = escape_sigma_frac
        self.escape_n_params = escape_n_params
        
        # Thread-safe archive of all completed trials
        self._lock = threading.Lock()
        self._archive: List[Dict[str, Any]] = []
        
        # v169 (v4.77) — NAS escape state tracking
        # =========================================
        # The v168 @1800s run revealed an evolutionary search failure mode:
        # the best architecture was found at trial 9 (seed phase) and the
        # next 115 trials produced ZERO improvements. Evolutionary search
        # got trapped exploiting trial 9's neighborhood without ever
        # exploring far enough to discover better regions.
        # 
        # Fix: track trials_since_last_best. When it exceeds escape_threshold
        # (default 30), switch to escape mode:
        #   - mutation_sigma_frac: 0.15 → 0.35 (broader Gaussian per-param)
        #   - mutation_n_params:   (1,3) → (3,6) (more params mutated per child)
        # This produces "long-range" mutations that genuinely explore distant
        # regions of the search space. When a new best is found, revert to
        # normal mode and reset the counter.
        self.best_fitness = -1.0
        self.trials_since_last_best = 0
        self.in_escape_mode = False
        self.escape_events = []   # list of (trial_id_started, trial_id_resolved or None)
        
        # Op counters (for status reporting)
        self.op_count = {'random': 0, 'mutation': 0, 'crossover': 0}
    
    def update(self, result: Dict[str, Any]) -> None:
        """Called from saver thread. Add this trial to the archive if it
        has a usable fitness. v169 — also tracks the trials-since-last-best
        counter that drives escape mode."""
        fit = result.get('fitness', -1.0)
        if fit <= -1.0:
            return
        tid = result.get('trial_id', -1)
        with self._lock:
            self._archive.append({
                'fitness': fit,
                'arch':    result['arch'],
                'trial_id': tid,
            })
            # v169 — escape state machine
            if fit > self.best_fitness:
                # New global best — exit escape if active, reset counter
                if self.in_escape_mode:
                    # Mark the escape event as resolved
                    if self.escape_events and self.escape_events[-1][1] is None:
                        self.escape_events[-1] = (self.escape_events[-1][0], tid)
                    self.in_escape_mode = False
                    self.mutation_sigma_frac = self.mutation_sigma_frac_normal
                    self.mutation_n_params = self.mutation_n_params_normal
                    print(f"[NAS escape] RESOLVED at trial {tid} "
                          f"(new best {fit:.3f}) — reverting to normal mutation",
                          flush=True)
                self.best_fitness = fit
                self.trials_since_last_best = 0
            else:
                self.trials_since_last_best += 1
                # Enter escape mode if we've been stuck too long
                if (not self.in_escape_mode and
                    self.trials_since_last_best >= self.escape_threshold and
                    len(self._archive) > self.random_seed_trials):
                    self.in_escape_mode = True
                    self.mutation_sigma_frac = self.escape_sigma_frac
                    self.mutation_n_params = self.escape_n_params
                    self.escape_events.append((tid, None))
                    print(f"[NAS escape] TRIGGERED at trial {tid} "
                          f"({self.trials_since_last_best} trials without improvement, "
                          f"best={self.best_fitness:.3f}) — switching to broad mutation "
                          f"(sigma={self.escape_sigma_frac}, n_params={self.escape_n_params})",
                          flush=True)
    
    def _get_elites(self) -> List[Dict[str, Any]]:
        """Return top-N archive entries by fitness. Snapshot under lock."""
        with self._lock:
            sorted_arch = sorted(self._archive, key=lambda x: -x['fitness'])
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
                    'arch': arch,
                    'trial_id': -1 - i,   # negative trial_ids mark "seeded"
                })
            self.best_fitness = max(assumed_fitness, self.best_fitness)
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
    
    def _mutate_value(self, key: str, val: Any, spec: Any) -> Any:
        """Perturb a single value within its SEARCH_SPACE bounds. For
        continuous: Gaussian noise scaled to `mutation_sigma_frac` of
        the range. For discrete: 50% chance pick a neighbor in the list."""
        if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], str):
            kind, lo, hi = spec
            span = max(1e-12, hi - lo)
            sigma = self.mutation_sigma_frac * span
            if kind == 'uniform':
                new_val = float(val) + self.rng.gauss(0, sigma)
                return max(lo, min(hi, new_val))
            elif kind == 'loguniform':
                import math
                log_lo, log_hi = math.log(lo), math.log(hi)
                log_span = log_hi - log_lo
                log_sigma = self.mutation_sigma_frac * log_span
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
        """Copy parent arch, perturb 1-3 random parameters."""
        child = self._empty_arch(trial_id)
        # Copy all sections from parent
        for section in ('biology', 'neural', 'operating_ranges',
                        'healthy_bands', 'genetic_lottery'):
            if section in parent:
                child[section] = copy.deepcopy(parent[section])
        
        # Pick 1-3 parameters to mutate
        keys = list(SEARCH_SPACE.keys())
        n_lo, n_hi = self.mutation_n_params
        n_mut = self.rng.randint(n_lo, min(n_hi, len(keys)))
        mut_keys = self.rng.sample(keys, n_mut)
        
        for full_key in mut_keys:
            section, key = full_key.split('.', 1)
            spec = SEARCH_SPACE[full_key]
            current = child.setdefault(section, {}).get(key)
            if current is None:
                # Parent doesn't have this param — sample fresh from spec
                child[section][key] = _sample_one(self.rng, spec)
            else:
                child[section][key] = self._mutate_value(full_key, current, spec)
        
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
        """Mix two parents: for each searchable parameter, randomly pick
        from A or B (50/50). Keeps the structure but blends search-space
        coordinates."""
        child = self._empty_arch(trial_id)
        # Start by copying parent_a entirely
        for section in ('biology', 'neural', 'operating_ranges',
                        'healthy_bands', 'genetic_lottery'):
            if section in parent_a:
                child[section] = copy.deepcopy(parent_a[section])
        # For each searchable param, 50% chance to use parent_b's value
        for full_key in SEARCH_SPACE.keys():
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
    
    def sample(self, trial_id: int) -> Dict[str, Any]:
        """Generate one architecture: seed phase = random; otherwise
        pick an operator weighted by p_random / p_mutation / p_crossover."""
        with self._lock:
            archive_size = len(self._archive)
        
        # Seed phase: always random until we have enough elites
        if archive_size < self.random_seed_trials:
            self.op_count['random'] += 1
            return sample_random_architecture(self.rng, trial_id)
        
        # Choose operator
        r = self.rng.random()
        if r < self.p_random or archive_size < 2:
            self.op_count['random'] += 1
            return sample_random_architecture(self.rng, trial_id)
        
        elites = self._get_elites()
        if not elites:
            self.op_count['random'] += 1
            return sample_random_architecture(self.rng, trial_id)
        
        if r < self.p_random + self.p_mutation:
            # Mutation: pick one elite, perturb 1-3 params
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
                f"ops: rnd={self.op_count['random']} "
                f"mut={self.op_count['mutation']} "
                f"x={self.op_count['crossover']}]")


# Factory
def make_strategy(name: str, rng: random.Random, **kwargs) -> _SearchStrategy:
    """Build a strategy by name. Accepts kwargs to override defaults."""
    name = (name or 'random').lower().strip()
    if name in ('random', 'rand'):
        return RandomSearchStrategy(rng)
    elif name in ('evolutionary', 'evo', 'evolution', 'ga'):
        return EvolutionaryStrategy(rng, **kwargs)
    else:
        raise ValueError(f"Unknown search strategy '{name}'. "
                         f"Use 'random' or 'evolutionary'.")


# =============================================================================
# WORKER — runs in a subprocess
# =============================================================================
def _trial_worker(args: Tuple) -> Dict[str, Any]:
    """Run one Game-of-Life trial in headless mode with the given arch.
    
    This runs inside a multiprocessing subprocess. All imports happen
    inside the function so each worker has fresh pygame/architecture
    state (multiprocessing.spawn starts each subprocess with a clean
    Python interpreter).
    
    Returns a dict with: trial_id, arch, metrics, error (or None),
    wall_seconds_actual, n_samples.
    """
    arch_dict, trial_id, wall_seconds, seed, out_dir = args
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
        
        # cd into a per-trial directory so save files don't clutter
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
            
            result['metrics'] = {
                'final_alive':       final_alive,
                'start_alive':       start_alive,
                'peak_alive':        peak_alive,
                # v168 — checkpoint alive counts for sustained-survival fitness
                'alive_q1':          alive_q1,
                'alive_q2':          alive_q2,
                'alive_q3':          alive_q3,
                'alive_q4':          alive_q4,
                'alive_mean_checkpoints': alive_mean_checkpoints,
                'M1_last':           last_mean('M1_excitatory_fraction'),
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


# =============================================================================
# FITNESS FUNCTION
# =============================================================================
def fitness(metrics: Optional[Dict[str, float]]) -> float:
    """Combine trial metrics into a single fitness score (higher = better).
    
    Components (weights tunable):
    - Final alive count (high weight — robust populations matter most)
    - M10 heritability (medium — selection signal)
    - Inverse mean_state_streak (medium — lock-in penalty — the v155 finding)
    - M1 in-band bonus (low — keeps networks in operating regime)
    - sensory_motor_corr non-zero (low — input-output coupling)
    
    Returns float; -1.0 for failed trials.
    """
    if metrics is None:
        return -1.0
    
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
    
    # Component 4 — M1 excitatory fraction in band [0.18, 0.28]
    m1 = metrics.get('M1_last', 0.0)
    m1_in_band = 1.0 if 0.18 <= m1 <= 0.30 else 0.0
    
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
    
    score = (
        3.0 * surv +         # 0-3
        1.5 * m10 +          # 0-1.5
        1.5 * streak_score + # 0-1.5
        0.5 * m1_in_band +   # 0-0.5
        0.5 * sm_score       # 0-0.5
    )
    return float(score)



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
            'trial_id', 'completed_at_iso', 'fitness',
            'final_alive', 'peak_alive',
            'alive_q1', 'alive_q2', 'alive_q3', 'alive_mean',
            'M1', 'M6', 'M9',
            'M10', 'M10_peak',
            'mean_streak_last', 'mean_streak_first',
            'locked', 'surv_score', 'expl_rate',
            'sm_corr', 'sm_corr_peak', 'sm_corr_mean',
            'n_samples', 'wall_actual_s', 'error',
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
        m.get('final_alive', ''), m.get('peak_alive', ''),
        # v168 — checkpoint columns
        m.get('alive_q1', ''), m.get('alive_q2', ''),
        m.get('alive_q3', ''),
        f"{m.get('alive_mean_checkpoints', 0):.2f}",
        f"{m.get('M1_last', 0):.4f}", f"{m.get('M6_last', 0):.4f}",
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
        result['n_samples'], f"{result['wall_seconds_actual']:.1f}",
        result['error'] or '',
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
        'version': 'NxonArchNAS v0.4 (v162)',
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
                        if (self.global_best is None
                            or r['fitness'] > self.global_best['fitness']):
                            self.global_best = r
                            new_best = True
                        # Update top-3
                        self.global_top3 = sorted(
                            self.global_top3 + [r],
                            key=lambda x: -x['fitness'])[:3]
                    
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
                     search_strategy: str = 'random',
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
                     seed_archs_fitness: float = 6.0) -> Dict[str, Any]:
    """v160 — Streaming NAS. v164 adds pluggable search_strategy.
    v169 — adds escape mechanism for evolutionary strategy when stuck.
    
    Args:
      num_trials: legacy display-batch size
      wall_seconds: per-trial time limit
      workers: parallel processes (default cpu_count - 1)
      seed: RNG seed for architecture sampling
      out_dir: output directory (default nas_runs/<timestamp>)
      batches: legacy v159 flag (max_trials = batches * num_trials)
      max_trials: total trial cap (None = forever until Ctrl-C)
      verbose: per-trial console output
      search_strategy: 'random' (v161-v163) or 'evolutionary' (v164)
      evo_*: evolutionary parameters when search_strategy='evolutionary'
      evo_escape_*: v169 escape parameters — trigger broad mutation when
                    no improvement for evo_escape_threshold trials.
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
    
    # v164 — build the strategy
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
    ) if search_strategy != 'random' else RandomSearchStrategy(strategy_rng)
    
    # v170 (v4.78) — pre-populate elite pool with known-good architectures.
    # Only supported by EvolutionaryStrategy; random strategy ignores seeds.
    if seed_archs and hasattr(strategy, 'seed_with_archs'):
        strategy.seed_with_archs(seed_archs, assumed_fitness=seed_archs_fitness)
    elif seed_archs:
        print(f"[NAS] WARNING: --seed-archs requires --search-strategy evolutionary; "
              f"ignoring {len(seed_archs)} seed archs", flush=True)
    
    scheduler = _NASScheduler(
        num_trials=num_trials, wall_seconds=wall_seconds,
        workers=workers, seed=seed, out_path=out_path,
        max_trials=max_trials, verbose=verbose,
        strategy=strategy,
    )
    return scheduler.run()


# v156-v158 backward-compat — single batch
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
        description='Neuraxon Architecture Search v0.5 — streaming scheduler '
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
                        default='random',
                        help='Sampling strategy. "random" (default, v161-v163 behavior) '
                             'samples each trial independently from SEARCH_SPACE. '
                             '"evolutionary" maintains an elite pool of top-K trials and '
                             'generates new trials via mutation/crossover of elites '
                             '(after seeding random_seed_trials with random samples).')
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
    )
    
    if summary['best_arch'] is not None:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
