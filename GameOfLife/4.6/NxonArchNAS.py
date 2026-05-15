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
# SEARCH SPACE
# =============================================================================
# Each entry maps a "section.key" path (as used in architecture JSON files)
# to a list of candidate values. Random search picks one from each list.
#
# Keep this list short for now — the curse of dimensionality means
# random search needs ~100s of trials to cover 10+ dimensions well.
# Future versions can add Bayesian optimization or evolutionary search.
SEARCH_SPACE: Dict[str, List[Any]] = {
    # BIOLOGY — game-world dynamics
    'biology.metabolic_ramp_per_sec':       [3.0, 5.0, 7.0, 10.0, 15.0],
    'biology.max_atrophy':                  [2.0, 3.0, 5.0, 8.0, 12.0],
    'biology.metabolic_rate_abs_cap_multiple': [10.0, 20.0, 50.0],
    'biology.idle_explore_seconds':         [0.5, 1.0, 1.5, 2.0],
    'biology.explore_probability':          [0.3, 0.5, 0.7],
    # NEURAL — network topology
    'neural.num_hidden_neurons_default':    [8, 12, 16, 20],
    # OPERATING_RANGES — these aren't wired into v155/v156 code yet
    # (see CHANGELOG_v155.md migration roadmap) — leave commented until
    # they're actually consumed:
    # 'operating_ranges.learning_rate':     [0.005, 0.01, 0.02],
    # 'operating_ranges.autoreceptor_coefficient': [0.05, 0.1, 0.2],
}


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
    """Sample one architecture from SEARCH_SPACE. Returns a dict in the
    same format as architectures/default.json — directly loadable by
    architecture.load_architecture()."""
    arch: Dict[str, Any] = {
        '_meta': {
            'source': 'NxonArchNAS',
            'trial_id': trial_id,
            'sampled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'biology': {}, 'neural': {}, 'operating_ranges': {}, 'healthy_bands': {},
    }
    for full_key, choices in SEARCH_SPACE.items():
        section, key = full_key.split('.', 1)
        arch.setdefault(section, {})[key] = rng.choice(choices)
    return arch


def arch_summary_string(arch: Dict[str, Any]) -> str:
    """Short one-line summary of the varied parameters (for logging)."""
    parts = []
    for section in ('biology', 'neural', 'operating_ranges'):
        for k, v in arch.get(section, {}).items():
            parts.append(f"{k}={v}")
    return " ".join(parts)


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
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        
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
            
            alive_series = ts.get('surv_alive_count', [0])
            final_alive = float(alive_series[-1]) if alive_series else 0.0
            start_alive = float(alive_series[0]) if alive_series else 0.0
            peak_alive = max((float(v) for v in alive_series), default=0.0)
            
            result['metrics'] = {
                'final_alive':       final_alive,
                'start_alive':       start_alive,
                'peak_alive':        peak_alive,
                'M1_last':           last_mean('M1_excitatory_fraction'),
                'M6_last':           last_mean('M6_spontaneous_fraction'),
                'M9_last':           last_mean('M9_transfer_ratio'),
                'M10_last':          last_mean('M10_heritability_r'),
                'mean_streak_last':  last_mean('mean_state_streak'),
                'mean_streak_first': first_mean('mean_state_streak'),
                'locked_last':       last_mean('input_locked_fraction'),
                'surv_score_last':   last_mean('surv_score'),
                'expl_rate_last':    last_mean('exploration_trigger_rate'),
                'sm_corr_last':      last_mean('sensory_motor_corr'),
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
    surv = min(1.0, metrics.get('final_alive', 0.0) / NAS_TRIAL_CONFIG['StartingNxErs'])
    
    # Component 2 — heritability (clamp + normalise)
    m10 = max(0.0, min(1.0, metrics.get('M10_last', 0.0)))
    
    # Component 3 — lock-in penalty (mean_state_streak)
    # streak ~2-5 is healthy, > 100 is severe lock-in
    streak = metrics.get('mean_streak_last', 0.0)
    streak_score = max(0.0, 1.0 - streak / 100.0)
    
    # Component 4 — M1 excitatory fraction in band [0.18, 0.28]
    m1 = metrics.get('M1_last', 0.0)
    m1_in_band = 1.0 if 0.18 <= m1 <= 0.30 else 0.0
    
    # Component 5 — sensory→motor coupling restored
    sm = abs(metrics.get('sm_corr_last', 0.0))
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
    trial, no batches. Columns:
      trial_id, completed_at_iso, fitness, final_alive, M1, M6, M9, M10,
      mean_streak_last, mean_streak_first, locked, surv_score, expl_rate,
      sm_corr, n_samples, wall_actual_s, error, arch_summary, is_global_best
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial_id', 'completed_at_iso', 'fitness',
            'final_alive', 'peak_alive', 'M1', 'M6', 'M9', 'M10',
            'mean_streak_last', 'mean_streak_first',
            'locked', 'surv_score', 'expl_rate', 'sm_corr',
            'n_samples', 'wall_actual_s', 'error',
            'arch_summary', 'is_global_best',
        ])


def _append_csv_row(csv_path: Path, result: Dict, is_global_best: bool):
    """Append a single finished trial's row. Called from the saver
    thread, never from the worker callback (avoids file lock contention
    on Windows)."""
    m = result.get('metrics') or {}
    row = [
        result['trial_id'],
        time.strftime('%Y-%m-%dT%H:%M:%S'),
        f"{result['fitness']:.4f}",
        m.get('final_alive', ''), m.get('peak_alive', ''),
        f"{m.get('M1_last', 0):.4f}", f"{m.get('M6_last', 0):.4f}",
        f"{m.get('M9_last', 0):.4f}", f"{m.get('M10_last', 0):.4f}",
        f"{m.get('mean_streak_last', 0):.2f}",
        f"{m.get('mean_streak_first', 0):.2f}",
        f"{m.get('locked_last', 0):.4f}",
        f"{m.get('surv_score_last', 0):.4f}",
        f"{m.get('expl_rate_last', 0):.4f}",
        f"{m.get('sm_corr_last', 0):.4f}",
        result['n_samples'], f"{result['wall_seconds_actual']:.1f}",
        result['error'] or '',
        arch_summary_string(result['arch']),
        1 if is_global_best else 0,
    ]
    # Use 'a' mode with line buffering so partial writes flush even if
    # the script is killed before clean exit
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _save_arch_json(arch: Dict[str, Any], path: Path, rank: int,
                     trial_id: int, fitness_val: float):
    """Persist one architecture as a directly-loadable JSON. Atomic write
    (write to temp, rename) so readers in another shell never see a
    half-written file."""
    clean = {k: v for k, v in arch.items()
             if not (isinstance(k, str) and k.startswith('_'))}
    clean['_meta'] = {
        'source': 'NxonArchNAS',
        'rank': rank,
        'trial_id': trial_id,
        'fitness': fitness_val,
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'note': 'load with NEURAXON_ARCH=path/to/this.json python main.py',
    }
    # v160 — atomic write to avoid readers seeing partial JSON
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2)
    # On Windows os.replace is atomic; on POSIX it's also atomic (renames)
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
                  max_trials: Optional[int], verbose: bool):
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
                    alive = m.get('final_alive', '?')
                    streak = m.get('mean_streak_last', 0)
                    M10 = m.get('M10_last', 0)
                    marker = " ★ NEW BEST" if new_best else ""
                    err_short = (' err' if r.get('error') else '')
                    print(f"[NAS] trial {r['trial_id']:>4}{err_short:>4} "
                          f"fit={r['fitness']:6.3f}  alive={alive:>4} "
                          f"streak={streak:>5.1f}  M10={M10:5.2f}  "
                          f"| done {completed:>4}/{submitted}  "
                          f"errs={errors:>3}  best={best_fit:6.3f}"
                          f"{marker}", flush=True)
            except Exception as exc:
                # Saver thread must never die — catch and log
                print(f"[NAS] saver-thread warning: {exc}", flush=True)
                traceback.print_exc()
        
        self.saver_done.set()
    
    def _submit_one(self, pool):
        """Submit one new trial to the pool, increment submitted counter."""
        with self.lock:
            tid = self.submitted
            self.submitted += 1
        arch = sample_random_architecture(self.rng, tid)
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
                     verbose: bool = True) -> Dict[str, Any]:
    """v160 — Streaming NAS. Trials run continuously, replaced as they
    finish. A dedicated saver thread updates nas_best.json + CSV as each
    result arrives, so the latest best is on disk within seconds.
    
    Args:
      num_trials: legacy — number of trials per "logical batch" (display
        only; in v160 the actual scheduling is per-trial not per-batch)
      wall_seconds: per-trial time limit
      workers: parallel processes (default cpu_count - 1)
      seed: RNG seed for architecture sampling
      out_dir: output directory (default nas_runs/<timestamp>)
      batches: legacy v159 flag — converted to max_trials = batches * num_trials
      max_trials: total trial cap (None = forever until Ctrl-C)
      verbose: per-trial console output
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
    
    scheduler = _NASScheduler(
        num_trials=num_trials, wall_seconds=wall_seconds,
        workers=workers, seed=seed, out_path=out_path,
        max_trials=max_trials, verbose=verbose,
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
        description='Neuraxon Architecture Search v0.3 — streaming '
                    'scheduler with threaded saver. Continuous by default.')
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
    args = parser.parse_args()
    
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
    )
    
    if summary['best_arch'] is not None:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
