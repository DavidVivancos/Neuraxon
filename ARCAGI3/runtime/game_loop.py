"""
MultiNxonArcAGI3 — play loop.

``play_game`` runs ONE ARC-AGI-3 game (real or offline) with a Neuraxon
Multi-Sphere brain built from an architecture genome, for up to ``max_steps``
environment steps, and returns a metrics dict used by the NAS fitness.

The loop is deliberately tiny and engine-agnostic — it only uses the unified
adapter surface (reset / step / available_actions / state / score).
"""

from __future__ import annotations

import os
import time
import random
from typing import Any, Dict, List, Optional

from arc_env import make_env, GameState, GameAction, MAX_SCORE
from arc_brain import ArcBrain, build_params_from_arch


def _board_changed(cur, prev, border: int = 0) -> bool:
    """Did the BOARD change, ignoring a `border`-wide frame?

    Several games draw an action counter into the frame edge and it ticks on every action,
    including ones the game otherwise rejects. r11l's transcript is explicit: "The click was a
    pure no-op (only the counter ticked)." A plain grid comparison calls that a change.

    Measured over 120 random actions per game, "any cell changed" vs "any cell >= 2 from the
    edge changed":  vc33 100.0% -> 0.0% · cd82 79.0% -> 33.0% · m0r0 71.7% -> 51.7% ·
    r11l 100.0% -> 65.0%. On vc33 the contamination was total.
    """
    if cur is None or prev is None:
        return False
    if not border:
        return cur != prev
    H = len(cur)
    if H != len(prev) or H == 0:
        return cur != prev
    W = len(cur[0]) if cur[0] else 0
    if 2 * border >= H or 2 * border >= W:
        return cur != prev
    for y in range(border, H - border):
        cr, pr = cur[y], prev[y]
        for x in range(border, W - border):
            if cr[x] != pr[x]:
                return True
    return False


def _frame_signature(grid: List[List[int]]) -> int:
    """Cheap hash of a grid for distinct-state counting."""
    h = 1469598103934665603  # FNV-1a 64-bit
    for row in grid:
        for c in row:
            h ^= (c & 0xFF)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        h ^= 0xFF
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


# ============ v1.098 E — DETERMINISM: SEED THE GLOBAL RNG ============
# MEASURED, not theorised. Same brain, same game, same `base_seed`, five identical calls to
# play_learned_brain('sp80', base_seed=3847, max_steps=400) returned:
#
#       actions_to_l1:  4, 176, 142, 272, 7        -- a 68x spread inside ONE fixed seed
#
# Cause: ArcBrain seeds a PRIVATE generator (`self.rng = random.Random(seed)`), but the
# neuraxon internals call the STDLIB GLOBAL `random` module for neuron dynamics --
# homeostatic threshold drift, intrinsic timescales, oscillator phase init
# (neuraxon/components.py lines ~683-690, 709, 842). Nothing ever seeded the global, so the
# seed was a label, not a control.
#
# Consequences of the bug, all now explained by it:
#   * "20 seeds" was 20 draws from one distribution, not 20 controlled conditions, so the
#     min/median/max profile and every `spread_ratio` measured the brain's own run-to-run
#     noise rather than layout difficulty. hard_seeds()/easy_seeds() were mining noise.
#   * universality.py and test_coldpass_stability.py use the IDENTICAL 20-seed list and still
#     disagreed per-seed on the same brain and game (t192/sp80: stability seed-1 = 10 actions,
#     universality seed-1 = 15, and four universality seeds recorded 0 where stability's max
#     was 386 -- under the 500-step cap, so the cap could not explain it).
#   * A package scoring 0.25 then 0.12 on identical code is this variance, drawn once per game
#     across 55 games.
#
# After the fix, verified: 5/5 identical runs at seed 3847 -> 12 actions; 3/3 at seed 1928 -> 37.
#
# NOTE ON INTERPRETATION. Kaggle runs with the global RNG unseeded, so the STOCHASTIC regime
# is the deployed one. Seeding here does not remove the variance -- it makes it measurable and
# makes A/B comparisons valid. To ESTIMATE what a brain will score you still need repeats: at
# the measured CV (~1.0-1.25 on sp80) a 10-draw mean of E[RHAE] carries ~39% relative standard
# error and reverses the t192/t36 ordering 26% of the time; 20 draws ~28%, 100 draws ~12%.
# Set MULTINXON_DETERMINISTIC=0 to restore the 1097 behaviour for an explicit noise study.
def _v1098_seed_everything(seed: int) -> None:
    """Seed EVERY generator the brain can reach, so a play is reproducible from `seed` alone."""
    if os.environ.get('MULTINXON_DETERMINISTIC', '1') != '1':
        return
    try:
        s = int(seed) & 0x7FFFFFFF
    except Exception:
        s = 0
    random.seed(s)
    try:
        import numpy as _np
        _np.random.seed(s)
    except Exception:
        pass


def play_game(game_id: str = "off_reach",
              arch_path: Optional[str] = None,
              *,
              params=None,
              max_steps: int = 300,
              seed: int = 0,
              micro_steps: Optional[int] = None,
              prefer_real: bool = True,
              force_offline: Optional[bool] = None,
              render: bool = False,
              verbose: bool = False) -> Dict[str, Any]:
    """Play one game; return metrics.

    Either pass ``arch_path`` (a genome JSON) or a prebuilt ``params``.
    """
    rng = random.Random(seed)
    if params is None:
        params = build_params_from_arch(arch_path, verbose=verbose)

    _v1098_seed_everything(seed)   # v1.098 E: reproducible plays
    env = make_env(game_id, seed=seed, prefer_real=prefer_real,
                   render_mode=("terminal" if render else None),
                   force_offline=force_offline)

    # exploration knobs from the genome (biology section), with defaults
    import architecture as arch_mod
    explore_p = float(arch_mod.get_param("biology", "explore_probability", 0.5) or 0.5)
    idle_secs = float(arch_mod.get_param("biology", "idle_explore_seconds", 1.0) or 1.0)
    idle_steps = max(2, int(round(idle_secs * 6)))   # ~6 macro-steps ≈ 1s

    brain = ArcBrain(params, seed=seed, micro_steps=micro_steps,
                     explore_probability=explore_p, idle_explore_steps=idle_steps)

    # --- metrics accumulators ---
    peak_score = 0
    final_score = 0
    max_levels = 0
    reached_win = False
    action_steps = 0          # steps where we issued a play action (not RESET)
    frames_changed = 0        # play steps that changed the grid
    distinct = set()
    resets = 0
    game_overs = 0
    # v1.052: action count at first clear of each level (0 = never reached).
    actions_to_l1 = 0; actions_to_l2 = 0; actions_to_l3 = 0
    steps_to_l1 = 0; steps_to_l2 = 0; steps_to_l3 = 0  # v1.091 KAGGLE-FAITHFUL (resets counted)
    _prev_levels = 0

    # v1.015 within-level progress tracking — the gradient the NAS was blind to.
    # `level_progress` is exact for offline games and -1 (unknown) for real ones.
    has_exact_progress = False
    best_progress = 0.0       # peak within-level progress reached (exact)
    final_progress = 0.0      # progress at the last observed frame
    progress_sum = 0.0        # Σ progress over action steps (→ time-near-goal AUC)
    progress_gain_sum = 0.0   # Σ max(0, Δprogress) — total forward progress made
    prev_progress = 0.0
    # game-agnostic proxy (real games): reward reaching a grid state and HOLDING
    # / returning to it deliberately, vs random churn. Tracked as the fraction
    # of action steps whose grid had been seen before AND was re-reached after
    # leaving (a weak controllability signal). Plus best structural compression.
    revisit_hits = 0
    prev_sig = None

    obs = env.observation_space
    if obs is None or obs.state == GameState.NOT_PLAYED:
        obs = env.reset()
    brain.reset(obs)
    prev_grid = [row[:] for row in obs.grid]
    distinct.add(_frame_signature(obs.grid))

    step = 0
    while step < max_steps:
        state = obs.state
        if state == GameState.WIN:
            reached_win = True
            break
        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if state == GameState.GAME_OVER:
                game_overs += 1
                if hasattr(brain, 'v77_death'):
                    brain.v77_death(obs)                    # v1.177: the death frame, diagnosed (budget / fatal)
                if hasattr(brain, 'v55_note_death'):
                    brain.v55_note_death()                  # v1.155: the fatal (state, action) is remembered
            obs = env.reset()
            brain.reset(obs)
            prev_grid = [row[:] for row in obs.grid]
            resets += 1
            step += 1
            continue

        action, data = brain.play_step(obs)
        if action is GameAction.RESET:
            obs = env.reset()
            brain.reset(obs)
            prev_grid = [row[:] for row in obs.grid]
            resets += 1
            step += 1
            continue

        nxt = env.step(action, data=data)
        action_steps += 1

        changed = nxt.grid != prev_grid
        if changed:
            frames_changed += 1
        distinct.add(_frame_signature(nxt.grid))

        peak_score = max(peak_score, int(nxt.score))
        final_score = int(nxt.score)
        max_levels = max(max_levels, int(nxt.levels_completed))
        # v1.052: stamp action count the first time each level is cleared.
        if max_levels > _prev_levels:
            if max_levels >= 1 and actions_to_l1 == 0:
                actions_to_l1 = action_steps
                steps_to_l1 = action_steps + resets  # v1.091: cumulative incl resets (Kaggle-exact)
            if max_levels >= 2 and actions_to_l2 == 0:
                actions_to_l2 = action_steps
                steps_to_l2 = action_steps + resets
            if max_levels >= 3 and actions_to_l3 == 0:
                actions_to_l3 = action_steps
                steps_to_l3 = action_steps + resets
            _prev_levels = max_levels

        # v1.015 within-level progress. Use the exact signal when the engine
        # provides it (offline, or a future real engine); otherwise accumulate
        # the grid-derived proxy below.
        lp = float(getattr(nxt, "level_progress", -1.0))
        if lp >= 0.0:
            has_exact_progress = True
            best_progress = max(best_progress, lp)
            final_progress = lp
            progress_sum += lp
            if lp > prev_progress:
                progress_gain_sum += (lp - prev_progress)
            prev_progress = lp
        else:
            # proxy: did the agent RE-REACH a previously seen state right after
            # leaving it (deliberate return → controllability), not just churn?
            sig = _frame_signature(nxt.grid)
            if changed and prev_sig is not None and sig in distinct and sig != prev_sig:
                revisit_hits += 1
            prev_sig = sig

        if render and (step % 10 == 0):
            print(f"step {step:4d} | act {action.name:8s} | score {nxt.score:3d} "
                  f"| lvl {nxt.levels_completed} | prog {lp:+.2f} "
                  f"| state {nxt.state.name}")

        prev_grid = [row[:] for row in nxt.grid]
        obs = nxt
        step += 1
        if obs.state == GameState.WIN:
            reached_win = True
            break

    eff_rate = frames_changed / max(1, action_steps)
    # v1.011 — the old fixed explore_cap=32 saturated to 1.0 for EVERY viable
    # brain on a multi-thousand-step episode (analysis of the two v1.010 runs:
    # all 973 trials had exploration==1.000 exactly), so the term carried zero
    # search gradient while distinct_frames — the 2nd-strongest behavioural
    # correlate of fitness (r≈+0.83) — was thrown away. Scale the cap with the
    # episode length so breadth-of-states is graded, not pinned: at 5000 steps
    # the cap is 2000 distinct frames (top brains ~1.0, thrashers ~0.1), which
    # restores a real gradient to the exploration fitness term. The floor keeps
    # short smoke runs sensible.
    explore_cap = max(64.0, 0.4 * float(max_steps))
    exploration = min(1.0, len(distinct) / explore_cap)

    # v1.015 within-level progress signals. `level_progress_best` is the headline
    # gradient: how deep into a level the brain got (exact for offline games).
    # When the engine gives no exact signal, fall back to the controllability
    # proxy so real-game runs still have a (weaker) within-level gradient.
    if has_exact_progress:
        level_progress_best = best_progress
        level_progress_final = final_progress
        level_progress_auc = progress_sum / max(1, action_steps)
        level_progress_gain = min(1.0, progress_gain_sum)
        progress_source = 1.0  # exact
    else:
        proxy = revisit_hits / max(1, action_steps)   # 0..1 controllability
        level_progress_best = proxy
        level_progress_final = proxy
        level_progress_auc = proxy
        level_progress_gain = proxy
        progress_source = 0.0  # proxy

    metrics: Dict[str, Any] = {
        "game_id": getattr(env, "game_id", game_id),
        "backend": getattr(env, "backend", "offline"),
        "peak_score": float(peak_score),
        "final_score": float(final_score),
        "score_norm": float(peak_score) / float(MAX_SCORE),
        "levels_completed": float(max_levels),
        "reached_win": 1.0 if reached_win else 0.0,
        "effective_action_rate": float(eff_rate),
        "exploration": float(exploration),
        "distinct_frames": float(len(distinct)),
        # v1.016 mastery signals — the reframed objective is reliable level-1
        # then the level-1→2 crossing on a SINGLE game, not breadth across games.
        # cleared_l1: did this run clear at least one level (binary, per run).
        # reached_l2: did it reach level 2 — the first real capability gain.
        "cleared_l1": 1.0 if max_levels >= 1 else 0.0,
        "reached_l2": 1.0 if max_levels >= 2 else 0.0,
        "max_levels_raw": float(max_levels),
        # v1.017: per-run flags that aggregate by MAX (did it EVER happen on any
        # seed), so a lucky-seed level-2 crossing is not erased by the median-
        # over-seeds aggregation — the v1.016 runs showed cd82 brains reaching
        # level 2 on ~1 of 4-8 seeds (peak_score 20), which the median hid.
        "ever_cleared_l1": 1.0 if max_levels >= 1 else 0.0,
        "ever_reached_l2": 1.0 if max_levels >= 2 else 0.0,
        "ever_reached_l3": 1.0 if max_levels >= 3 else 0.0,
        # v1.117: how many cells the click mask considered live on the last frame.
        # 0 means the mask was off. On r11l the transcripts predict ~3.
        "click_candidates": float(getattr(brain, "_cm_n", 0) or 0),
        # v1.015 within-level progress (the previously-missing gradient)
        "level_progress_best": float(level_progress_best),
        "level_progress_final": float(level_progress_final),
        "level_progress_auc": float(level_progress_auc),
        "level_progress_gain": float(level_progress_gain),
        "progress_source": float(progress_source),
        "sm_corr": float(brain.telemetry.sensorimotor_corr()),
        "motor_active_rate": float(brain.telemetry.motor_active_rate()),
        "explore_rate": float(brain.telemetry.explore_rate()),
        "steps": float(step),
        "action_steps": float(action_steps),
        "resets": float(resets),
        "game_overs": float(game_overs),
        # v1.052: per-level action efficiency (0 if the level wasn't reached).
        "actions_to_l1": float(actions_to_l1),
        "actions_to_l2": float(actions_to_l2),
        "actions_to_l3": float(actions_to_l3),
        'is_real': bool(getattr(env, 'is_real', getattr(env, 'backend', '') == 'real')),   # v1.150: first-try trials run through play_game
        "steps_to_l1": float(steps_to_l1),   # v1.091 KAGGLE-FAITHFUL cumulative (resets counted)
        "steps_to_l2": float(steps_to_l2),
        "steps_to_l3": float(steps_to_l3),
    }
    return metrics


def _v1153_replay(env, prefix):
    """env.reset() then the verified prefix, WITHOUT the brain: the start of level L"""
    from arc_env.structs import GameAction as _GA
    obs = env.reset()
    for a in prefix:
        if a == 'RESET':
            obs = env.reset()
        elif str(a).startswith('ACTION6:'):
            x, y = map(int, str(a).split(':')[1].split(',')); obs = env.step(_GA.ACTION6, {'x': x, 'y': y})
        else:
            obs = env.step(_GA[str(a)])
    return obs


def _v1153_play_level(brain, env, prefix, max_steps, first_reset):
    """v1.153: ONE try at level L. Every game over or brain RESET restarts level L (the prefix replayed, not counted;
    the reset itself counts one action, Kaggle-style). Cleared = levels_completed rises above its start."""
    obs = _v1153_replay(env, prefix); start = int(obs.levels_completed or 0)
    try:
        brain.reset(obs) if first_reset else brain.soft_reset(obs)
    except Exception:
        pass
    steps = 0; resets = 0
    while steps < max_steps:
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            obs = _v1153_replay(env, prefix); brain.soft_reset(obs); resets += 1; steps += 1; continue
        action, data = brain.play_step(obs)
        if action is GameAction.RESET:
            obs = _v1153_replay(env, prefix); brain.soft_reset(obs); resets += 1; steps += 1; continue
        obs = env.step(action, data=data); steps += 1
        if obs.state == GameState.GAME_OVER and hasattr(brain, 'v77_death'):
            brain.v77_death(obs)                            # v1.177: the death frame, diagnosed (budget / fatal)
        if obs.state == GameState.GAME_OVER and hasattr(brain, 'v55_note_death'):
            brain.v55_note_death()                          # v1.155
        if int(obs.levels_completed or 0) > start:
            return {'cleared': 1, 'actions': steps, 'resets': resets, 'beh': dict(getattr(brain, '_v55_beh', {}) or {}),
                    'is_real': bool(getattr(env, 'is_real', getattr(env, 'backend', '') == 'real'))}       # v1.156
    return {'cleared': 0, 'actions': 0, 'resets': resets, 'beh': dict(getattr(brain, '_v55_beh', {}) or {}),
            'is_real': bool(getattr(env, 'is_real', getattr(env, 'backend', '') == 'real'))}


def train_level_life(game_id, arch_path, prefix, n_episodes, max_steps, seed, prefer_real=True, force_offline=False, micro_steps=3):
    """v1.153: a newborn built EXACTLY as play_lifelong builds it, living n_episodes at level L (plasticity on)"""
    from arc_brain import ArcBrain, build_params_from_arch
    _v1098_seed_everything(seed)
    params = build_params_from_arch(arch_path, verbose=False)
    import architecture as arch_mod
    explore_p = float(arch_mod.get_param("biology", "explore_probability", 0.5) or 0.5)
    idle_steps = max(2, int(round(float(arch_mod.get_param("biology", "idle_explore_seconds", 1.0) or 1.0) * 6)))
    brain = ArcBrain(params, seed=seed, micro_steps=micro_steps, explore_probability=explore_p, idle_explore_steps=idle_steps)
    env = make_env(game_id, seed=seed, prefer_real=prefer_real, force_offline=force_offline)
    eps = []
    for i in range(int(n_episodes)):
        r = _v1153_play_level(brain, env, prefix, max_steps, first_reset=(i == 0))
        eps.append(float(r['actions']) if r['cleared'] else 0.0)
    return brain, {'episodes': int(n_episodes), 'ep_actions': eps}


def cold_level_try(game_id, brain_path, arch_path, prefix, max_steps, seed, prefer_real=True, force_offline=False, micro_steps=3,
                   caps=None):
    """v1.153: the saved brain loaded EXACTLY as play_learned_brain loads it; ONE cold try at level L.
    v1.167: caps = one budget per level (>= 2): the try continues into the next levels after each clear."""
    import gzip as _gz
    import json
    from arc_brain import ArcBrain, build_params_from_arch
    from neuraxon.multisphere import load_multisphere_from_dict
    try:
        from neuraxon.network import _rebuild_net_from_dict as _rebuild
    except Exception:
        _rebuild = None
    _fbs = os.environ.get('MULTINXON_FIXED_BRAIN_SEED', '')
    _brain_seed = int(_fbs) if _fbs.strip() != '' else int(seed)
    _v1098_seed_everything(seed)
    params = build_params_from_arch(arch_path, verbose=False)
    with _gz.open(brain_path, 'rt', encoding='utf-8') as fh:
        payload = json.load(fh)
    prebuilt = load_multisphere_from_dict(payload.get('brain', payload), rebuild_net_fn=_rebuild)
    env = make_env(game_id, seed=seed, prefer_real=prefer_real, force_offline=force_offline)
    brain = ArcBrain(params, seed=_brain_seed, micro_steps=micro_steps, prebuilt_brain=prebuilt)
    if caps is not None and len(caps) >= 2:
        return _v1153_play_level(brain, env, prefix, max_steps, first_reset=True, caps=list(caps))
    return _v1153_play_level(brain, env, prefix, max_steps, first_reset=True)


def train_from_prefix(brain, game_id, prefix, max_steps, seed, prefer_real=True, force_offline=False):
    """v1.152: one episode that STARTS after `prefix` (a verified solution replayed WITHOUT the brain): the same brain,
    plasticity on, plays from there. Returns _run_one_episode's metrics."""
    from arc_env.adapter import make_env
    from arc_env.structs import GameAction as _GA
    env = make_env(game_id, seed=seed, prefer_real=prefer_real, force_offline=force_offline)
    obs = env.reset()
    for a in prefix:
        if a == 'RESET':
            obs = env.reset()
        elif str(a).startswith('ACTION6:'):
            x, y = map(int, str(a).split(':')[1].split(',')); obs = env.step(_GA.ACTION6, {'x': x, 'y': y})
        else:
            obs = env.step(_GA[str(a)])
    brain.soft_reset(obs)                 # keep what the life learned; the board jumped to level 2
    return _run_one_episode(brain, env, max_steps)


def _run_one_episode(brain, env, max_steps: int) -> Dict[str, Any]:
    """Run ONE episode with an already-built brain (used by play_lifelong). The
    brain is NOT re-created; on in-episode GAME_OVER it soft-resets (keeps
    learned weights). Returns the same per-episode metric fields play_game does.

    v1.043 WARM-RESTART: once the brain has cleared level 1 within this episode it
    is at the L2 boundary; on a subsequent GAME_OVER it does NOT always reset back
    to level 1 — with warm_restart probability it perturbs exploration and keeps
    practising from the frontier, so it gets many warm L2 attempts per episode."""
    peak_score = 0; final_score = 0; max_levels = 0; reached_win = False
    action_steps = 0; frames_changed = 0; resets = 0; game_overs = 0
    # v1.122: border frame ignored when deciding whether the BOARD changed.
    _HUD_B = int(os.environ.get('MULTINXON_HUD_BORDER', '0') or 0)
    # v1.121 signature counters. See the block in the loop for what each one buys.
    _sig_act: Dict[str, int] = {}
    _sig_prev_a = None
    _sig_repeat = 0
    _sig_cells = 0
    _sig_cellmax = 0
    _sig_clicks = 0
    _sig_onlive = 0
    _sig_bg_cache: Dict[int, int] = {}

    def _sig_bg(grid):
        """Modal colour, cached per frame identity. Cheap enough to run every click."""
        if not grid or not grid[0]:
            return 0
        k = id(grid)
        v = _sig_bg_cache.get(k)
        if v is not None:
            return v
        c: Dict[int, int] = {}
        for row in grid:
            for z in row:
                c[z] = c.get(z, 0) + 1
        v = max(c.items(), key=lambda kv: kv[1])[0] if c else 0
        if len(_sig_bg_cache) > 8:
            _sig_bg_cache.clear()
        _sig_bg_cache[k] = v
        return v
    _ep_game_id = getattr(env, 'game_id', None) or getattr(env, 'game', '') or ''
    warm_attempts = 0
    distinct = set()
    # v1.066 CLEAR CAPTURE (opt-in via env MULTINXON_CAPTURE=1). For each level the
    # brain clears, record the LAYOUT it faced and the ACTION TRAJECTORY that solved
    # it, so lucky-vs-general can finally be analysed. The real engine won't
    # regenerate a layout (it ignores the seed), so this must be captured live. Off by
    # default => zero cost in normal search.
    _capture = os.environ.get('MULTINXON_CAPTURE', '') == '1'
    _clears: List[Dict[str, Any]] = []
    _act_log: List[str] = []
    _init_grid = None
    # v1.066: optional per-episode wall-clock cap (seconds) to kill runaway episodes
    # (the VPS still had a ~24,000s tail in 1065). 0 = disabled.
    _time_cap = float(os.environ.get('MULTINXON_EPISODE_TIME_CAP', '0') or 0.0)
    _ep_start = time.time()
    # v1.052: record the action count at the moment each level is FIRST cleared,
    # so we have true per-level action efficiency (what RHAE scores), not just the
    # episode total. actions_to_lN stays 0 until level N is reached.
    actions_to_l1 = 0; actions_to_l2 = 0; actions_to_l3 = 0
    steps_to_l1 = 0; steps_to_l2 = 0; steps_to_l3 = 0  # v1.091 KAGGLE-FAITHFUL (resets counted)
    _prev_levels = 0
    _p = getattr(brain, 'params', None)
    _warm = float(getattr(_p, 'warm_restart', 0.0) or 0.0)
    obs = env.observation_space
    if obs is None or obs.state == GameState.NOT_PLAYED:
        obs = env.reset()
    prev_grid = [row[:] for row in obs.grid]
    distinct.add(_frame_signature(obs.grid))
    if _capture:
        _init_grid = [row[:] for row in obs.grid]
    lp_best = 0.0; lp_final = 0.0; lp_sum = 0.0
    step = 0
    while step < max_steps:
        if _time_cap > 0.0 and (step & 63) == 0 and (time.time() - _ep_start) > _time_cap:
            break
        state = obs.state
        if state == GameState.WIN:
            reached_win = True; break
        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if state == GameState.GAME_OVER:
                game_overs += 1
                if hasattr(brain, 'v77_death'):
                    brain.v77_death(obs)                    # v1.177: the death frame, diagnosed (budget / fatal)
                if hasattr(brain, 'v55_note_death'):
                    brain.v55_note_death()                  # v1.155: the fatal (state, action) is remembered
            obs = env.reset()
            # v1.043 WARM-RESTART: if L1 already cleared this episode, keep the
            # brain's frontier-directed state instead of a full reset, so it
            # resumes practising the L2 crossing rather than re-grinding L1.
            if _warm > 0.0 and max_levels >= 1 and brain.rng.random() < _warm:
                warm_attempts += 1
                if hasattr(brain, 'warm_perturb'):
                    try:
                        brain.warm_perturb(obs)
                    except Exception:
                        brain.soft_reset(obs)
                else:
                    brain.soft_reset(obs)
            else:
                brain.soft_reset(obs)          # keep learning across in-episode resets
            prev_grid = [row[:] for row in obs.grid]
            resets += 1; step += 1; continue
        action, data = brain.play_step(obs)
        if action is GameAction.RESET:
            obs = env.reset(); brain.soft_reset(obs)
            prev_grid = [row[:] for row in obs.grid]
            resets += 1; step += 1; continue
        nxt = env.step(action, data=data)
        action_steps += 1
        if _capture and len(_act_log) < 600:
            _act_log.append(getattr(action, 'name', str(action)))
        # v1.122: the HUD counter is not the board. See _board_changed.
        changed = _board_changed(nxt.grid, prev_grid, _HUD_B)
        if changed:
            frames_changed += 1
        # ============ v1.121 SIGNATURE COUNTERS ============
        # The offline probe on 100 brains x 25 games showed these separate a brain that is
        # SOLVING from one that is thrashing, and no aggregate the NAS produced contained them:
        #   lp85 and ft09 run at 0.90-0.98 no-op -- almost every action is inert, so the whole
        #   budget is spent on clicks the game ignores
        #   tr87 runs at moved = 1.000 and clears 1 in 100 -- maximum change, no progress
        #   a failing r11l episode causes ~4x MORE board change per action than a succeeding one
        # They cost one dict lookup and one comparison per action, and they ride in the metrics
        # dict the NAS actually aggregates -- the one at line ~436, not the other one. The
        # click_candidates metric in v1117 went into the wrong dict and was missing for two runs.
        _sig_act[getattr(action, 'name', str(action))] = \
            _sig_act.get(getattr(action, 'name', str(action)), 0) + 1
        if _sig_prev_a is not None and getattr(action, 'name', '') == _sig_prev_a:
            _sig_repeat += 1
        _sig_prev_a = getattr(action, 'name', str(action))
        if changed:
            _n = 0
            _cur, _pv = nxt.grid, prev_grid
            if _cur and _pv and len(_cur) == len(_pv):
                for _y in range(len(_cur)):
                    _cr, _pr = _cur[_y], _pv[_y]
                    for _x in range(len(_cr)):
                        if _cr[_x] != _pr[_x]:
                            _n += 1
            _sig_cells += _n
            _sig_cellmax = max(_sig_cellmax, _n)
        # did the click land on a non-background cell?
        if isinstance(data, dict) and 'x' in data and prev_grid and prev_grid[0]:
            _H, _W = len(prev_grid), len(prev_grid[0])
            _gy = min(_H - 1, int(data['y'] * _H / 64))
            _gx = min(_W - 1, int(data['x'] * _W / 64))
            _sig_clicks += 1
            if prev_grid[_gy][_gx] != _sig_bg(prev_grid):
                _sig_onlive += 1
        distinct.add(_frame_signature(nxt.grid))
        peak_score = max(peak_score, int(nxt.score))
        final_score = int(nxt.score)
        max_levels = max(max_levels, int(nxt.levels_completed))
        # v1.052: stamp the action count the first time each level is cleared.
        if max_levels > _prev_levels:
            if max_levels >= 1 and actions_to_l1 == 0:
                actions_to_l1 = action_steps
                steps_to_l1 = action_steps + resets  # v1.091: cumulative incl resets (Kaggle-exact)
            if max_levels >= 2 and actions_to_l2 == 0:
                actions_to_l2 = action_steps
                steps_to_l2 = action_steps + resets
            if max_levels >= 3 and actions_to_l3 == 0:
                actions_to_l3 = action_steps
                steps_to_l3 = action_steps + resets
            if _capture:
                _clears.append({
                    'level': int(max_levels),
                    'actions_to_clear': int(action_steps),
                    'init_sig': _frame_signature(_init_grid) if _init_grid else None,
                    'init_grid': _init_grid,
                    'clear_grid': [row[:] for row in nxt.grid],
                    'action_traj': list(_act_log),
                })
            _prev_levels = max_levels
        lp = float(getattr(nxt, "level_progress", -1.0))
        if lp >= 0.0:
            lp_best = max(lp_best, lp); lp_final = lp; lp_sum += lp
        prev_grid = [row[:] for row in nxt.grid]
        obs = nxt; step += 1
        if obs.state == GameState.WIN:
            reached_win = True; break
    eff_rate = (frames_changed / action_steps) if action_steps else 0.0
    explore_cap = max(64.0, 0.4 * float(max_steps))
    return {
        'game_id': getattr(env, 'game_id', '?'),
        'backend': getattr(env, 'backend', 'offline'),
        'is_real': bool(getattr(env, 'is_real', getattr(env, 'backend', '') == 'real')),
        '_clears': _clears,
        'peak_score': float(peak_score), 'final_score': float(final_score),
        'score_norm': float(peak_score) / float(MAX_SCORE),
        'max_levels_raw': float(max_levels),
        # ============ v1.121 SIGNATURE, in the dict the NAS ACTUALLY AGGREGATES ============
        # Derived from the offline probe over 100 brains x 25 games. Calibration from that
        # study, so these are readable against known behaviour:
        #   sig_no_op       r11l 0.000 · lp85 0.975 · ft09 0.896 · sb26 0.629
        #   sig_repeat      a uniform six-action random walk gives 0.167; sp80 measured 0.174
        #                   across four runs, r11l 1.000 (one action, repeated -- and it solves)
        #   sig_cells_med   a FAILING r11l episode changes ~4x more board per action than a
        #                   succeeding one. Big change without progress is thrashing.
        #   sig_click_hit   lp85 hit 100% live cells and still changed NOTHING -- the mask can
        #                   be aiming perfectly at decoration
        'sig_no_op': float(1.0 - (frames_changed / action_steps)) if action_steps else 0.0,
        'sig_repeat': float(_sig_repeat / max(action_steps - 1, 1)),
        'sig_actions_used': float(len(_sig_act)),
        'sig_cells_mean': float(_sig_cells / max(frames_changed, 1)),
        'sig_cells_max': float(_sig_cellmax),
        'sig_click_hit': float(_sig_onlive / _sig_clicks) if _sig_clicks else -1.0,
        'sig_clicks': float(_sig_clicks),
        'sig_distinct': float(len(distinct)),
        'sig_distinct_per_action': float(len(distinct) / max(action_steps, 1)),
        'cleared_l1': 1.0 if max_levels >= 1 else 0.0,
        'reached_l2': 1.0 if max_levels >= 2 else 0.0,
        'ever_reached_l3': 1.0 if max_levels >= 3 else 0.0,
        'reached_win': 1.0 if reached_win else 0.0,
        'effective_action_rate': float(eff_rate),
        'exploration': min(1.0, len(distinct) / explore_cap),
        'distinct_frames': float(len(distinct)),
        'level_progress_best': lp_best, 'level_progress_final': lp_final,
        'level_progress_auc': lp_sum / max(1, action_steps),
        'level_progress_gain': lp_best,
        'progress_source': 1.0 if lp_best > 0 else 0.0,
        'sm_corr': float(brain.telemetry.sensorimotor_corr()),
        'motor_active_rate': float(brain.telemetry.motor_active_rate()),
        'explore_rate': float(brain.telemetry.explore_rate()),
        'steps': float(step), 'action_steps': float(action_steps),
        'resets': float(resets), 'game_overs': float(game_overs),
        # v1.052: per-level action efficiency (0 if the level wasn't reached).
        'actions_to_l1': float(actions_to_l1),
        'actions_to_l2': float(actions_to_l2),
        'actions_to_l3': float(actions_to_l3),
        'steps_to_l1': float(steps_to_l1),   # v1.091 KAGGLE-FAITHFUL cumulative (resets counted)
        'steps_to_l2': float(steps_to_l2),
        'steps_to_l3': float(steps_to_l3),
        'warm_attempts': float(warm_attempts),
        # v1.044 mid-game brain-region behaviour (which spheres/pathways did what)
        **{k: float(v) for k, v in brain.telemetry.region_summary().items()},
        # v1.076 ACTION-GENES: per-action-type usage (answers "is the brain pressing X?")
        # + entropy (how spread across actions) + the ordered action trace for motif
        # mining. The trace is only kept for clears at <=1.5x human so the dataset is
        # solution sequences, not flailing.
        **_action_gene_metrics(brain, actions_to_l1, _ep_game_id),
    }


def _action_gene_metrics(brain, actions_to_l1, game_id):
    """v1.076 — extract the action histogram, per-action fractions, action entropy,
    and (for efficient clears) the action sequence from the brain's captured trace."""
    import math as _math
    hist = dict(getattr(brain, '_action_hist', {}) or {})
    seq = list(getattr(brain, '_action_seq', []) or [])
    tot = sum(hist.values()) or 1
    out = {}
    # per-action-type FRACTION (ACTION1..ACTION6, RESET) — the diagnostic signal
    for a, c in hist.items():
        out['act_frac_' + str(a)] = c / tot
    # how many distinct action types were used, and the entropy of the distribution
    out['action_types_used'] = float(len([c for c in hist.values() if c > 0]))
    # v1.077 ACTION COVERAGE: of the actions AVAILABLE this episode, what fraction did the
    # brain actually ISSUE at least once? <1.0 means it's ignoring an available action
    # (wa30: ACTION5 available, coverage would show it unused). This is the signal the
    # coverage bonus uses to push the brain to try every available action.
    # v1.078: REAL planning activation fraction (was dead-gene 0; now genuinely counted)
    _ps = int(getattr(brain, '_plan_steps', 0) or 0)
    out['mech_frac_planning'] = (int(getattr(brain, '_plan_total', 0) or 0) / _ps) if _ps else 0.0
    avail = set(getattr(brain, '_avail_actions', set()) or set())
    used = set(a for a, c in hist.items() if c > 0)
    if avail:
        out['action_coverage'] = len(avail & used) / len(avail)
        out['action_avail_n'] = float(len(avail))
        # list the AVAILABLE-BUT-UNUSED actions (the diagnostic) as a compact string
        unused = sorted(avail - used)
        if unused:
            out['action_unused'] = ','.join(unused)
    ent = -sum((c / tot) * _math.log((c / tot) + 1e-12) for c in hist.values() if c > 0)
    out['action_entropy'] = float(ent)
    # the ordered trace, kept ONLY for efficient clears (solution sequences). Stored as
    # a compact comma string so it rides in the metrics/brain json.
    try:
        from arc_env.adapter import _KAGGLE_HUMAN_L1 as _HB
    except Exception:
        _HB = {}
    human = float(_HB.get(str(game_id or ''), 0.0)) or 40.0
    if actions_to_l1 and 0 < actions_to_l1 <= 1.5 * human and seq:
        out['action_seq_l1'] = ','.join(seq[:int(actions_to_l1) + 5])
        out['action_seq_len'] = float(len(seq[:int(actions_to_l1) + 5]))
    return out


# ============ v1.149: LEARNING, MEASURED AGAINST A FROZEN TWIN ============
# A trial is one life of n_episodes (weights kept between episodes when MULTINXON_ZERO_SHOT=0).
# Improvement across episodes is NOT learning by itself: a newborn whose weights are frozen still
# went 20 -> 3 -> 3 actions on vc33 (state carried by soft resets). So a trial can replay the same
# life from the same newborn with every synapse frozen in place (the TWIN); learning = plastic - twin.
_V1149_CONTAINERS = ('synapses', '_syn_by_pre', '_syn_by_post', '_afferent_by_pre')


def _v1149_networks(obj, depth=0, seen=None, out=None):
    seen = seen if seen is not None else set(); out = out if out is not None else []
    if id(obj) in seen or depth > 6:
        return out
    seen.add(id(obj))
    if type(obj).__name__ == 'NeuraxonNetwork':
        out.append(obj); return out
    it = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else \
        (vars(obj).values() if hasattr(obj, '__dict__') else [])
    for v in it:
        if not isinstance(v, (int, float, str, bytes, bool)) and v is not None:
            _v1149_networks(v, depth + 1, seen, out)
    return out


def _v1149_synapse_count(brain) -> int:
    try:
        return int(sum(len(getattr(n, 'synapses', []) or []) for n in _v1149_networks(brain)))
    except Exception:
        return -1


def _v1149_inplace_freezer(brain):
    """HARD freeze that keeps object identity (learntest's deep-copy restore left three copies of each
    synapse and never restored _afferent_by_pre). Restores every container to the ORIGINAL synapse
    objects (undoing synaptogenesis and pruning) and writes every synapse's numeric state back in place.
    Verified: weights unchanged after a frozen life, trace-identical repeats."""
    import collections as _col
    snaps = []
    for n in _v1149_networks(brain):
        cont = {}
        for k in _V1149_CONTAINERS:
            v = getattr(n, k, None)
            if isinstance(v, list):
                cont[k] = ('list', None, list(v))
            elif isinstance(v, dict):
                cont[k] = ('dict', type(v), {kk: list(vv) for kk, vv in v.items()})
        objs = {}
        for kind, typ, c in cont.values():
            for x in (c if kind == 'list' else [y for vv in c.values() for y in vv]):
                objs[id(x)] = x
        state = {i: {a: val for a, val in vars(x).items()
                     if isinstance(val, (int, float, bool)) or type(val).__name__ == 'SynapseType'}
                 for i, x in objs.items()}
        for x in objs.values():
            try:
                x.learning_rate = 0.0
            except Exception:
                pass
        snaps.append((n, cont, objs, state))

    def restore():
        for n, cont, objs, state in snaps:
            for k, (kind, typ, c) in cont.items():
                if kind == 'list':
                    setattr(n, k, list(c))
                else:
                    d = _col.defaultdict(list) if typ is _col.defaultdict else {}
                    for kk, vv in c.items():
                        d[kk] = list(vv)
                    setattr(n, k, d)
            for i, x in objs.items():
                for a, val in state[i].items():
                    setattr(x, a, val)
    return restore


_V1149_TEACHER_CACHE = {}


def _v1149_teacher_map(game_id):
    """(Space, {board key: exact distance to the level-1 goal}) from solve_levels' BFS, cached on disk
    (teacher_cache/<game>.pkl) so a trial pays for it once per machine, not once per trial."""
    g = str(game_id).split('-')[0]
    if g in _V1149_TEACHER_CACHE:
        return _V1149_TEACHER_CACHE[g]
    import pickle
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teacher_cache')
    p = os.path.join(d, f'{g}.pkl')
    if os.path.exists(p):
        try:
            _V1149_TEACHER_CACHE[g] = pickle.load(open(p, 'rb'))
            return _V1149_TEACHER_CACHE[g]
        except Exception:
            pass
    from learntest import distance_map
    sp, dist = distance_map(g)
    os.makedirs(d, exist_ok=True)
    tmp = p + f'.{os.getpid()}.tmp'
    pickle.dump((sp, dist), open(tmp, 'wb')); os.replace(tmp, p)
    _V1149_TEACHER_CACHE[g] = (sp, dist)
    return sp, dist


def _v1149_install_teacher(brain, game_id, mode):
    """mode 'best': dopamine only for a NEW BEST exact distance to the level-1 goal in this episode
    (+1) and for the clear (+2). The +1 closer / -0.2 further teacher paid for oscillation (seed 0:
    +1 x46, -0.2 x47 in one episode) and think() clamps negative reward to zero dopamine."""
    sp, dist = _v1149_teacher_map(game_id)
    brain._teacher_only = True; brain._gg_scale = 1.0; brain._ir_scale = 0.0
    brain._progress_reward_scale = 0.0; brain._progress_penalty_scale = 0.0
    brain._t_lv = 0; brain._t_best = 10 ** 9; brain._t_pos = 0

    def _teach(grid, frame, _b=brain):
        if int(getattr(frame, 'levels_completed', 0) or 0) > _b._t_lv:
            _b._t_lv += 1; return 2.0
        d = dist.get(sp.key(grid))
        if d is not None and d < _b._t_best:
            _b._t_best = d; _b._t_pos += 1; return 1.0
        return 0.0
    brain._goal_gradient_reward = _teach

    def episode_start(obs, _b=brain):
        _b._t_lv = 0
        d0 = dist.get(sp.key(getattr(obs, 'grid', None) or [[0]]))
        _b._t_best = d0 if d0 is not None else 10 ** 9
    return episode_start


def play_lifelong(game_id: str = "off_reach",
                  arch_path: Optional[str] = None,
                  *,
                  params=None,
                  n_episodes: int = 8,
                  max_steps: int = 5000,
                  base_seed: int = 0,
                  micro_steps: Optional[int] = None,
                  prefer_real: bool = True,
                  force_offline: Optional[bool] = None,
                  verbose: bool = False,
                  return_brain: bool = False,
                  frozen: bool = False) -> Dict[str, Any]:
    """v1.024 CONTINUOUS-LEARNING evaluation. The same brain lives ONE life
    across ``n_episodes`` consecutive episodes — its learned synaptic and
    modulatory state is NOT reset between them (only the environment resets).

    This embodies the "no training/inference, always learning" principle and
    directly attacks the reliability wall: in the episodic frame every seed is a
    newborn brain that must rediscover the level-2 solution from scratch, so a
    crossing is a per-seed lottery. Here competence ACCUMULATES — once the brain
    learns to clear level 1 fast it spends later episodes working on level 2, and
    a level-2 discovery persists into subsequent episodes. Reliability can then
    emerge from accumulation rather than luck.

    Returns metrics in the SAME schema as play_game (so the NAS fitness is
    unchanged), but the per-level "reliable" rates are computed over the LATER
    half of the lifetime (after the brain has had time to learn), and extra
    learning-curve fields record whether capability rose across the lifetime.
    """
    _v1098_seed_everything(base_seed)   # v1.098 E: FIRST statement — see play_learned_brain
    # build the persistent brain ONCE; it is never re-created or reset below
    if params is None:
        params = build_params_from_arch(arch_path, verbose=verbose)
    import architecture as arch_mod
    explore_p = float(arch_mod.get_param("biology", "explore_probability", 0.5) or 0.5)
    idle_secs = float(arch_mod.get_param("biology", "idle_explore_seconds", 1.0) or 1.0)
    idle_steps = max(2, int(round(idle_secs * 6)))
    brain = ArcBrain(params, seed=base_seed, micro_steps=micro_steps,
                     explore_probability=explore_p, idle_explore_steps=idle_steps)
    # ---- v1.149: synapses at birth, the frozen twin, the teacher ----
    _syn_start = _v1149_synapse_count(brain)
    if frozen:
        _restore = _v1149_inplace_freezer(brain)
        _orig_step = brain.play_step

        def _frozen_step(frame, _o=_orig_step, _r=_restore):
            out = _o(frame)
            _r()
            return out
        brain.play_step = _frozen_step
    _teacher_mode = os.environ.get('MULTINXON_TEACHER', '').strip()
    _teacher_ep = None
    if _teacher_mode:
        try:
            _teacher_ep = _v1149_install_teacher(brain, game_id, _teacher_mode)
        except Exception as _te:
            print(f"  [v1.149] teacher '{_teacher_mode}' NOT installed on {game_id}: {type(_te).__name__}: {_te}", flush=True)
            _teacher_ep = None
    try:
        _abort_frac = float(os.environ.get('MULTINXON_LIFE_ABORT_FRAC', '') or 0.0)
    except Exception:
        _abort_frac = 0.0

    ep_metrics: List[Dict[str, Any]] = []
    first_env = True
    # v1.042 FRONTIER-FOCUSED scheduling. Read the frontier genes off the brain's
    # params. Once the brain's recent L1 clear-rate exceeds the mastery threshold,
    # grant EXTRA episodes (more shots at the L1→L2 boundary) up to a bonus budget,
    # and flag the brain as "at frontier" so it concentrates exploration there —
    # converting the ~1 lucky L2 exposure per life into many.
    _p = getattr(brain, 'params', None)
    _frontier = float(getattr(_p, 'frontier_focus', 0.0) or 0.0)
    _mastery = float(getattr(_p, 'frontier_mastery', 0.7) or 0.7)
    _bonus_frac = float(getattr(_p, 'frontier_bonus_frac', 1.0) or 1.0)
    base_episodes = n_episodes
    max_episodes = n_episodes
    if _frontier > 0.0:
        max_episodes = n_episodes + int(round(n_episodes * _bonus_frac * min(1.0, _frontier)))
    l2_exposures = 0                     # how many episodes touched level 2 (signal count)

    ep = 0
    while ep < max_episodes:
        seed = base_seed + 101 * ep
        _v1098_seed_everything(seed)   # v1.098 E: reproducible per-episode
        env = make_env(game_id, seed=seed, prefer_real=prefer_real,
                       render_mode=None, force_offline=force_offline)
        # IMPORTANT: do NOT rebuild the brain. We only point it at the new env's
        # first observation. The brain keeps all learned weights/modulator state.
        obs = env.reset()
        if first_env:
            brain.reset(obs)          # initialise buffers once, at the very start
            first_env = False
        else:
            # ============ v1.134: ZERO-SHOT TRIALS ============
            # THE ROOT CAUSE OF WARM != COLD. The trial ran 10 episodes with soft_reset between
            # them -- learning KEPT across all ten -- and reported the MINIMUM actions and the
            # late-episode levels. The cold pass is ONE fresh episode. So the fitness rewarded
            # "best of ten attempts after learning" while the cold pass measured "first attempt,
            # fresh". r11l holds 20/20 because its solution needs no learning; lp85 sat at 19/20
            # and sp80 at <=10/20 for four runs under three different objectives because every
            # one of them scored the learned brain. MULTINXON_ZERO_SHOT=1 makes every episode a
            # FULL reset: ten independent first attempts, and the aggregate below reports their
            # median. The trial then measures exactly what the cold pass measures.
            if os.environ.get('MULTINXON_ZERO_SHOT', '0') == '1':
                brain.reset(obs)      # full reset: a fresh brain for every episode
            else:
                brain.soft_reset(obs) # new episode, KEEP learning; reset only the
                                      # per-episode bookkeeping (cursor, traces)
        if _teacher_ep is not None:
            _teacher_ep(obs)
        m = _run_one_episode(brain, env, max_steps)
        m['episode'] = ep
        ep_metrics.append(m)
        if float(m.get('reached_l2', 0.0)) > 0 or float(m.get('max_levels_raw', 0.0)) >= 2.0:
            l2_exposures += 1
        if verbose:
            print(f"  [lifelong ep {ep}] peak={m['peak_score']:.0f} "
                  f"levels={m['max_levels_raw']:.0f} "
                  f"cleared_l1={m['cleared_l1']:.0f} reached_l2={m['reached_l2']:.0f}")
        # v1.030 SPEED: early-abort a clearly-dead lifetime. If after the first
        # few episodes the brain has not cleared level 1 even once, it will not
        # reach level 2; stop wasting the remaining (expensive) episodes on it.
        # v1.149: MULTINXON_LIFE_ABORT_FRAC=f -> abort a dead life only after f x base episodes
        # (1.0 = never abort: a learning brain may need several episodes before its first clear).
        _abort_after = max(2, int(round(_abort_frac * base_episodes))) if _abort_frac > 0 else max(2, base_episodes // 3)
        if ep + 1 >= _abort_after and not (_abort_frac >= 1.0):
            best_lvl = max(float(x.get('max_levels_raw', 0)) for x in ep_metrics)
            if best_lvl < 1.0:
                break

        # v1.042 FRONTIER: decide whether to grant bonus episodes beyond the base.
        # Only extend a life that has MASTERED L1 (so the extra episodes practise
        # the L1→L2 boundary, not re-prove L1) and is therefore in the regime where
        # more L2 shots actually help. Tell the brain it is at the frontier.
        if ep + 1 >= base_episodes and _frontier > 0.0 and ep + 1 < max_episodes:
            recent = ep_metrics[-max(2, base_episodes // 2):]
            l1_rate = sum(1 for x in recent
                          if float(x.get('cleared_l1', 0)) > 0) / len(recent)
            reached_frontier = max(float(x.get('max_levels_raw', 0))
                                   for x in ep_metrics) >= 1.0
            if l1_rate >= _mastery and reached_frontier:
                if hasattr(brain, 'set_frontier'):
                    try:
                        brain.set_frontier(True)
                    except Exception:
                        pass
            else:
                break  # not mastered / not progressing → don't waste bonus episodes
        ep += 1

    agg = _aggregate_lifelong(ep_metrics)
    agg['l2_exposures'] = float(l2_exposures)       # v1.042 training-signal count
    agg['life_episodes'] = float(len(ep_metrics))
    # ---- v1.149: the learning curve, episode by episode (0 = not cleared within max_steps) ----
    agg['ep_actions_to_l1'] = [float(r.get('actions_to_l1', 0.0) or 0.0) for r in ep_metrics]
    agg['ep_actions_to_l2'] = [float(r.get('actions_to_l2', 0.0) or 0.0) for r in ep_metrics]
    agg['ep_max_levels'] = [float(r.get('max_levels_raw', 0.0) or 0.0) for r in ep_metrics]
    agg['ep_effective_action_rate'] = [round(float(r.get('effective_action_rate', 0.0) or 0.0), 4) for r in ep_metrics]
    agg['ep_steps'] = [float(r.get('steps', 0.0) or 0.0) for r in ep_metrics]
    agg['syn_start'] = float(_syn_start)
    agg['syn_end'] = float(_v1149_synapse_count(brain))
    agg['net_default_steps'] = float(getattr(brain, '_net_default_n', 0) or 0)
    agg['frozen'] = 1.0 if frozen else 0.0
    agg['teacher_rewards'] = float(getattr(brain, '_t_pos', 0) or 0)
    if return_brain:
        # v1.054: hand back the LEARNED brain so the caller can analyze its
        # synaptic structure (brain_analyzer) and/or snapshot it if it is a
        # champion. The brain holds everything it learned across the lifetime.
        agg['_brain'] = getattr(brain, 'brain', brain)
        agg['_arc_brain'] = brain
    return agg


def play_learned_brain(game_id: str,
                       brain_snapshot_path: str,
                       arch_path: str,
                       *,
                       n_episodes: int = 4,
                       max_steps: int = 5000,
                       base_seed: int = 0,
                       micro_steps: Optional[int] = None,
                       prefer_real: bool = True,
                       force_offline: Optional[bool] = None,
                       restore_episodic: bool = True,
                       warmup_steps: int = 0) -> Dict[str, Any]:
    """v1.054 — load a snapshotted LEARNED brain and play it WARMED-UP.

    Loads the gzipped brain snapshot (synaptic weights the brain learned during
    the NAS), rebuilds it, injects it into an ArcBrain on top of the genome's
    params, and plays a lifelong run. Plasticity stays on, so the brain starts
    with its learned competence but still adapts to the (possibly different) real
    game. Returns the same metric schema as play_lifelong, so RHAE scoring is
    identical. Falls back to a newborn brain if the snapshot can't be loaded.

    v1.060: ``restore_episodic`` (default True) controls whether the 2a episodic
    working state (visit counts, goal, play-steps, ever-scored) is restored on top
    of the synaptic weights — the "warm start". Set False to measure how much of
    the warm-up is storable 2a memory vs irreducible live re-settling."""
    # v1.098 E: FIRST statement in the body. It must precede build_params_from_arch() and
    # load_multisphere_from_dict(): rebuilding the network consumes global randomness (neuron
    # parameter jitter), so seeding after that point leaves the brain's own construction
    # unseeded and the play only PARTLY reproducible (measured: 13/12/172/234 instead of
    # 12/12/12/12).
    # v1.147: THE SEED THAT SHIPS. Every verification in the project paired the brain seed with
    # the env seed, so '40/40' meant 'some brain seed works for each board', never 'the shipped
    # brain seed works on every board' -- r1146's seed scan: 40/40 lp85 brains with seeds that
    # never clear. MULTINXON_FIXED_BRAIN_SEED fixes the brain's seed across every env seed; the
    # env keeps base_seed. A 40/40 brain is then reliable at the seed it will ship with.
    _fbs = os.environ.get('MULTINXON_FIXED_BRAIN_SEED', '')
    _brain_seed = int(_fbs) if _fbs.strip() != '' else int(base_seed)
    _v1098_seed_everything(base_seed)      # globals and env vary per env seed; only the BRAIN's seed is fixed -- the Kaggle condition
    import gzip
    import json
    from arc_brain import ArcBrain, build_params_from_arch
    from neuraxon.multisphere import load_multisphere_from_dict
    try:
        from neuraxon.network import _rebuild_net_from_dict as _rebuild
    except Exception:
        _rebuild = None

    params = build_params_from_arch(arch_path, verbose=False)
    prebuilt = None
    episodic_state = {}
    try:
        with gzip.open(brain_snapshot_path, 'rt', encoding='utf-8') as fh:
            payload = json.load(fh)
        snap = payload.get('brain', payload)
        episodic_state = payload.get('episodic_state', {}) or {}
        prebuilt = load_multisphere_from_dict(snap, rebuild_net_fn=_rebuild)
    except Exception:
        prebuilt = None   # fall back to newborn

    env = make_env(game_id, seed=base_seed, prefer_real=prefer_real, force_offline=force_offline)
    brain = ArcBrain(params, seed=_brain_seed, micro_steps=micro_steps,
                     prebuilt_brain=prebuilt)
    # v1.054: initialise the brain's per-trial perception state once (sets
    # _ever_scored, visit counts, etc.) before the episode loop — _run_one_episode
    # only soft-resets on GAME_OVER and assumes this first reset has happened.
    _obs0 = env.observation_space
    if _obs0 is None or _obs0.state == GameState.NOT_PLAYED:
        _obs0 = env.reset()
    try:
        brain.reset(_obs0)
    except Exception:
        pass
    # v1.059/v1.060: restore the 2a episodic working state (visit counts, goal,
    # play-steps, ever-scored) AFTER reset, so the reloaded brain starts with its
    # game-memory and needs far less live warm-up. Toggleable for A/B measurement.
    _epi_restored = False
    if episodic_state and restore_episodic:
        try:
            brain.import_episodic_state(episodic_state)
            _epi_restored = True
        except Exception:
            pass

    # v1.060 WARMUP PHASE — re-settle the brain's fast dynamical ("Layer-3") state
    # BEFORE the scored trials, so the scored episodes measure the brain's WARM
    # competence (the few-action clear it reaches in training) rather than charging
    # RHAE for the re-settling time. The reload restores synaptic weights and the 2a
    # game-map instantly, but the live neuron potentials / eligibility traces / action
    # loop have to re-enter their working groove by PLAYING — that's what this does.
    # These steps are NOT scored. Plasticity stays on so the warmup genuinely settles
    # the brain. After warmup we reset onto a FRESH env for the clean scored attempt.
    _warm_done = 0
    if warmup_steps and warmup_steps > 0:
        try:
            _wenv = make_env(game_id, seed=base_seed + 991,
                             prefer_real=prefer_real, force_offline=force_offline)
            _wobs = _wenv.observation_space
            if _wobs is None or _wobs.state == GameState.NOT_PLAYED:
                _wobs = _wenv.reset()
            try:
                brain.reset(_wobs)
            except Exception:
                pass
            # re-apply episodic after the warmup reset (reset clears some 2a state)
            if episodic_state and restore_episodic:
                try:
                    brain.import_episodic_state(episodic_state)
                except Exception:
                    pass
            for _ in range(int(warmup_steps)):
                try:
                    action, data = brain.play_step(_wobs)
                    # FIX: pass the click x/y data through, exactly like the scored
                    # loop — ACTION6 (complex) needs {"x","y"} or the real engine
                    # raises KeyError('x') and the warmup step is lost.
                    _wobs = _wenv.step(action, data=data)
                    _warm_done += 1
                    if _wobs.state == GameState.GAME_OVER:
                        try:
                            brain.soft_reset(_wobs)
                        except Exception:
                            _wobs = _wenv.reset()
                    elif _wobs.state == GameState.NOT_PLAYED:
                        _wobs = _wenv.reset()
                        try:
                            brain.soft_reset(_wobs)
                        except Exception:
                            pass
                except Exception:
                    break
            # v1.060b: close the warmup env to release its scorecard — otherwise
            # each game leaks an extra open scorecard and parallel runs hit the
            # engine's open-scorecard limit (-> silent offline fallback).
            try:
                _wenv.close()
            except Exception:
                pass
        except Exception:
            pass

    ep_metrics: List[Dict[str, Any]] = []
    # if we warmed up, give the (now-warm) brain a FRESH env + reset for the clean
    # scored attempt — the brain keeps its settled weights/dynamics, but the game
    # starts from the top so actions-to-clear is measured honestly.
    if _warm_done > 0:
        try:
            env = make_env(game_id, seed=base_seed,
                           prefer_real=prefer_real, force_offline=force_offline)
            _o = env.reset()
            brain.reset(_o)
            if episodic_state and restore_episodic:
                try:
                    brain.import_episodic_state(episodic_state)
                except Exception:
                    pass
        except Exception:
            pass
    for ep in range(max(1, n_episodes)):
        if ep > 0:
            # close the previous episode's env before opening the next, so open
            # scorecards don't accumulate across episodes (v1.060b).
            try:
                env.close()
            except Exception:
                pass
            try:
                env = make_env(game_id, seed=base_seed + ep,
                               prefer_real=prefer_real,
                               force_offline=force_offline)
            except Exception:
                pass
        m = _run_one_episode(brain, env, max_steps)
        ep_metrics.append(m)
    # close the final env (release its scorecard)
    try:
        env.close()
    except Exception:
        pass
    agg = _aggregate_lifelong(ep_metrics)
    agg['life_episodes'] = float(len(ep_metrics))
    agg['warmed_up'] = 1.0 if prebuilt is not None else 0.0
    agg['episodic_restored'] = 1.0 if _epi_restored else 0.0
    agg['had_episodic'] = 1.0 if episodic_state else 0.0
    agg['warmup_steps_done'] = float(_warm_done)
    return agg


def _aggregate_lifelong(ep_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode metrics from a continuous lifetime. Reliability is
    measured over the LATER half (post-learning); 'ever' flags over the whole
    life; plus learning-curve fields (did capability rise from first to second
    half — evidence the brain actually learned across episodes)."""
    n = len(ep_metrics)
    if n == 0:
        return {}
    half = max(1, n // 2)
    late = ep_metrics[half:] or ep_metrics[-1:]
    early = ep_metrics[:half] or ep_metrics[:1]

    def mean(rows, key):
        vs = [float(r.get(key, 0.0)) for r in rows]
        return sum(vs) / len(vs) if vs else 0.0

    def any_pos(rows, key):
        return 1.0 if any(float(r.get(key, 0.0)) > 0 for r in rows) else 0.0

    _zs = os.environ.get('MULTINXON_ZERO_SHOT', '0') == '1'

    def _best_actions(rows, key):
        if _zs:
            # v1.134: ten independent first attempts -> report the MEDIAN, which is what the
            # 20-seed cold pass reports. The min of ten fresh attempts would still be a lucky
            # draw; the median is the brain.
            vs = sorted(float(r.get(key, 0) or 0) for r in rows if float(r.get(key, 0) or 0) > 0)
            return vs[len(vs) // 2] if vs else 0.0
        # fewest (non-zero) actions any episode took to reach the level; 0 = never.
        vs = [float(r.get(key, 0.0)) for r in rows if float(r.get(key, 0.0)) > 0]
        return min(vs) if vs else 0.0

    # reliability over the LATER half (the brain has had time to learn)
    cl1_late = mean(late, 'cleared_l1')
    rl2_late = mean(late, 'reached_l2')
    # learning curve: did clearing/reaching improve from early to late?
    cl1_gain = mean(late, 'cleared_l1') - mean(early, 'cleared_l1')
    l2_gain = mean(late, 'max_levels_raw') - mean(early, 'max_levels_raw')

    agg: Dict[str, Any] = {
        'game_id': ep_metrics[0].get('game_id'),
        'backend': ep_metrics[0].get('backend', 'offline'),
        'peak_score': max(float(r.get('peak_score', 0.0)) for r in ep_metrics),
        'final_score': float(ep_metrics[-1].get('final_score', 0.0)),
        'score_norm': max(float(r.get('score_norm', 0.0)) for r in ep_metrics),
        'levels_completed': mean(late, 'max_levels_raw'),
        'reached_win': any_pos(ep_metrics, 'reached_win'),
        'effective_action_rate': mean(ep_metrics, 'effective_action_rate'),
        'exploration': mean(ep_metrics, 'exploration'),
        'distinct_frames': mean(ep_metrics, 'distinct_frames'),
        # reliability over the later half — the continuous-learning payoff
        'cleared_l1': cl1_late,
        'reached_l2': rl2_late,
        'max_levels_raw': max(float(r.get('max_levels_raw', 0.0)) for r in ep_metrics),
        # 'ever' over the whole lifetime
        'ever_cleared_l1': any_pos(ep_metrics, 'cleared_l1'),
        'ever_reached_l2': any_pos(ep_metrics, 'reached_l2'),
        'ever_reached_l3': any_pos(ep_metrics, 'ever_reached_l3'),
        # learning-curve evidence (did the brain improve across its life?)
        'lifelong_cl1_gain': cl1_gain,
        'lifelong_l2_gain': l2_gain,
        'level_progress_best': max(float(r.get('level_progress_best', 0.0)) for r in ep_metrics),
        'level_progress_final': float(ep_metrics[-1].get('level_progress_final', 0.0)),
        'level_progress_auc': mean(ep_metrics, 'level_progress_auc'),
        'level_progress_gain': mean(ep_metrics, 'level_progress_gain'),
        'progress_source': ep_metrics[0].get('progress_source', 0.0),
        'sm_corr': mean(ep_metrics, 'sm_corr'),
        'motor_active_rate': mean(ep_metrics, 'motor_active_rate'),
        'explore_rate': mean(ep_metrics, 'explore_rate'),
        'steps': mean(ep_metrics, 'steps'),
        'action_steps': mean(ep_metrics, 'action_steps'),
        'resets': mean(ep_metrics, 'resets'),
        'game_overs': mean(ep_metrics, 'game_overs'),
        # v1.052: per-level action efficiency — the BEST (fewest) actions the brain
        # took to clear each level across its episodes (0 if never reached). Kept for
        # backward compatibility, but NO LONGER the reliability signal (see below).
        'actions_to_l1': _best_actions(ep_metrics, 'actions_to_l1'),
        'actions_to_l2': _best_actions(ep_metrics, 'actions_to_l2'),
        'actions_to_l3': _best_actions(ep_metrics, 'actions_to_l3'),
        # v1.091 KAGGLE-FAITHFUL cumulative (resets counted) — the number Kaggle actually
        # charges. _best_actions takes the fewest across episodes, matching the fast-tail RHAE.
        'steps_to_l1': _best_actions(ep_metrics, 'steps_to_l1'),
        'steps_to_l2': _best_actions(ep_metrics, 'steps_to_l2'),
        'steps_to_l3': _best_actions(ep_metrics, 'steps_to_l3'),
    }

    # v1.066 GENERALISATION SIGNAL. The real engine randomises the layout every
    # episode (it ignores the seed), so each episode here is a DIFFERENT random
    # layout. The old aggregation collapsed the per-episode step counts to their
    # minimum (the luckiest layout), which is exactly why the search bred layout
    # memorisers. Instead, expose the MEAN and coefficient-of-variation of the
    # step counts over the episodes that cleared, plus the FRACTION of all episodes
    # (=layouts) that cleared. A genome that clears many random layouts in a tight,
    # low step band generalises; one that clears one lucky layout does not. These are
    # what the reliability fitness now reads (see NxonArcNAS.fitness).
    def _ep_stats(rows, key):
        vs = [float(r.get(key, 0.0)) for r in rows if float(r.get(key, 0.0)) > 0]
        if not vs:
            return 0.0, 0.0, 0.0, 0
        mu = sum(vs) / len(vs)
        sd = (sum((v - mu) ** 2 for v in vs) / len(vs)) ** 0.5
        return mu, sd, (sd / mu if mu > 0 else 0.0), len(vs)
    n_ep = max(1, len(ep_metrics))
    for lvl_key, tag in (('actions_to_l1', 'l1'), ('actions_to_l2', 'l2'),
                         ('actions_to_l3', 'l3')):
        mu, sd, cv, nz = _ep_stats(ep_metrics, lvl_key)
        agg[f'{lvl_key}_mean'] = mu
        agg[f'{lvl_key}_std'] = sd
        agg[f'{lvl_key}_cv'] = cv
        # fraction of the whole lifetime's episodes (random layouts) that cleared
        agg[f'{tag}_layouts_cleared'] = nz
        agg[f'{tag}_generalize_frac'] = nz / n_ep
    agg['n_layouts'] = n_ep
    # v1.066: surface the captured clears (layout + trajectory) and whether the real
    # engine was actually used (guards silent offline fallback).
    _all_clears = []
    for r in ep_metrics:
        _all_clears.extend(r.get('_clears') or [])
    agg['_clears'] = _all_clears
    agg['is_real'] = any(bool(r.get('is_real')) for r in ep_metrics)

    # v1.044: pass mid-game brain-region behaviour through (averaged over the
    # whole life), so the analyzer can see which spheres/pathways did what.
    region_keys = set()
    for r in ep_metrics:
        for k in r:
            if (k.startswith('sphere_act_') or k.startswith('mech_frac_')
                    or k.startswith('busiest')):
                region_keys.add(k)
    for k in region_keys:
        agg[k] = mean(ep_metrics, k)

    # v1.031: expose the spread of the lottery metrics across the later-half
    # episodes, so the variance-aware fitness (task6lcb) can compute a lower-
    # confidence bound from WITHIN one life too (not only across repeats). A
    # brain that reaches level 2 on a fluke episode has high spread here; one
    # that does it consistently has low spread. n_runs is set to the later-half
    # episode count so the stderr is scaled correctly.
    def _std(rows, key):
        vs = [float(r.get(key, 0.0)) for r in rows]
        if len(vs) < 2:
            return 0.0
        mu = sum(vs) / len(vs)
        return (sum((v - mu) ** 2 for v in vs) / len(vs)) ** 0.5
    agg['cleared_l1_std'] = _std(late, 'cleared_l1')
    agg['reached_l2_std'] = _std(late, 'reached_l2')
    agg['lifelong_l2_gain_std'] = 0.0
    agg['n_runs'] = float(len(late))
    # v1.076 ACTION-GENES aggregation: average the per-action fractions + entropy across
    # the lifetime, and carry the action SEQUENCE from the FASTEST-clearing episode (the
    # best solution trace), so both the diagnostic distribution and the motif source ship.
    _afk = set()
    for r in ep_metrics:
        _afk.update(k for k in r if k.startswith('act_frac_'))
    for k in _afk:
        agg[k] = mean(ep_metrics, k)
    agg['action_entropy'] = mean(ep_metrics, 'action_entropy')
    agg['action_types_used'] = mean(ep_metrics, 'action_types_used')
    agg['action_coverage'] = mean(ep_metrics, 'action_coverage')   # v1.077
    # carry the worst-case unused-action list (diagnostic) from any episode
    for r in ep_metrics:
        if r.get('action_unused'):
            agg['action_unused'] = r['action_unused']; break
    # best solution sequence = the episode with the smallest positive actions_to_l1 that
    # actually stored a trace
    _seq_eps = [r for r in ep_metrics if r.get('action_seq_l1') and float(r.get('actions_to_l1', 0) or 0) > 0]
    if _seq_eps:
        best_ep = min(_seq_eps, key=lambda r: float(r.get('actions_to_l1')))
        agg['action_seq_l1'] = best_ep['action_seq_l1']
        agg['action_seq_len'] = float(best_ep.get('action_seq_len', 0.0))
    # ============ v1.121: CARRY THE SIGNATURE THROUGH ============
    # This aggregator builds `agg` from an explicit key list, so a new per-episode metric is
    # silently dropped unless it is named here. That is exactly the class of defect that lost
    # the v1117 `click_candidates` metric for two entire runs -- it was emitted into a dict
    # nothing aggregated, and every analysis afterwards had to re-derive it offline.
    #
    # Averaged over the LATER half, matching every other reliability figure in this dict: the
    # brain has had time to learn by then. sig_click_hit uses -1 for "no clicks issued", so
    # episodes that never clicked are excluded rather than dragging the mean to zero.
    for _k in ('sig_no_op', 'sig_repeat', 'sig_actions_used', 'sig_cells_mean',
               'sig_cells_max', 'sig_clicks', 'sig_distinct', 'sig_distinct_per_action'):
        agg[_k] = mean(late, _k)
    _hits = [float(r.get('sig_click_hit', -1.0)) for r in late
             if float(r.get('sig_click_hit', -1.0)) >= 0.0]
    agg['sig_click_hit'] = (sum(_hits) / len(_hits)) if _hits else -1.0
    return agg


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Play one ARC-AGI-3 game with a Neuraxon brain")
    ap.add_argument("--game", default="off_reach",
                    help="game id (offline: reach/switch/match or off_*; real: e.g. ls20)")
    ap.add_argument("--architecture", default=None, help="path to architecture genome JSON")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-steps", type=int, default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--offline", action="store_true", help="force the offline engine")
    args = ap.parse_args()

    m = play_game(args.game, args.architecture, max_steps=args.steps, seed=args.seed,
                  micro_steps=args.micro_steps, render=args.render,
                  force_offline=True if args.offline else None, verbose=True)
    print("\n=== RESULT ===")
    for k in ("game_id", "backend", "peak_score", "final_score", "levels_completed",
              "reached_win", "effective_action_rate", "exploration", "sm_corr",
              "motor_active_rate", "explore_rate", "resets", "steps"):
        print(f"  {k:22s}: {m[k]}")


if __name__ == "__main__":
    _main()


# ============ v1.160: the verified-prefix replay is not the brain's play: discard it from the per-try tally ============
_v1160_replay_orig = _v1153_replay


def _v1153_replay(env, prefix):
    obs = _v1160_replay_orig(env, prefix)
    try:
        from arc_env import adapter as _ad160
        _ad160.v1160_take()
    except Exception:
        pass
    return obs


# ============ v1.163: the expert prefix's COMPLETED LEVELS teach the steering's goal map (nas_1162_report §19, change 3) ===
# A next-level line starts at level L after the verified prefix is replayed WITHOUT the brain. Those levels completed in
# front of it: their start and finished boards are exactly what a brain that had played them itself would have seen. The
# replay records them (and the within-level transitions, for the research's HUD mask) and the play hands them to the
# steering once, after the brain's reset (Steer4.prior_levels). Inert for every steering without prior_levels.
_V1163_PRIOR = {'key': None, 'pairs': [], 'hud': None}
_v1163_replay_plain = _v1153_replay                      # the v1160 replay (the prefix's tally discarded)


def _v1163_grid(obs):
    import numpy as _np
    fr = getattr(obs, 'frame', None) or []
    try:
        g = _np.asarray(fr[-1], dtype=_np.int16) if fr else None
        return g if (g is not None and g.ndim == 2) else None
    except Exception:
        return None


def _v1153_replay(env, prefix):
    """v1.153's replay, v1.163-fixed: env.reset() restarts the CURRENT level (measured on the real engine: after a game
    over or a mid-level reset on level L, levels_completed stays L-1 and the board is level L's start; a second reset
    restarts the game). 1153-1162 replayed the FULL prefix after a single reset, i.e. level 1's moves on level L's board:
    every try that died or reset continued from a corrupted board. Now the game is restarted (reset until
    levels_completed is 0) before the prefix is replayed, so a reset gives the brain level L again, k moves in."""
    import hashlib as _hl
    if not prefix:
        _V1163_PRIOR.update(key=None, pairs=[], hud=None, warm=[], warm_game=[])  # level 1 from its real start: no level before it
        return _v1163_replay_plain(env, prefix)
    from arc_env.structs import GameAction as _GA
    obs = env.reset(); _n = 0
    while int(getattr(obs, 'levels_completed', 0) or 0) > 0 and _n < 3:
        obs = env.reset(); _n += 1
    if _n:
        _V1163_PRIOR['restarts'] = int(_V1163_PRIOR.get('restarts', 0)) + 1
    g = _v1163_grid(obs)
    key = (_hl.md5(g.tobytes()).hexdigest() if g is not None else None, tuple(prefix))
    if key == _V1163_PRIOR.get('key'):                   # the same prefix again (a reset inside the play): replay only
        for a in prefix:
            if a == 'RESET':
                obs = env.reset()
            elif str(a).startswith('ACTION6:'):
                x, y = map(int, str(a).split(':')[1].split(',')); obs = env.step(_GA.ACTION6, {'x': x, 'y': y})
            else:
                obs = env.step(_GA[str(a)])
    else:
        import goalmap1163 as _gm
        start = g; lv = int(obs.levels_completed or 0); pairs = []; trans = []
        warm = []                                        # v1.173: this level's prefix moves (obs before, action, boards) -- effect1173
        warm_game = []                                   # v1.174: the EARLIER levels' moves (action, boards; no obs) -- effect1174's game memory
        for a in prefix:
            pre = _v1163_grid(obs); ob0 = obs
            if a == 'RESET':
                obs = env.reset()
            elif str(a).startswith('ACTION6:'):
                x, y = map(int, str(a).split(':')[1].split(',')); obs = env.step(_GA.ACTION6, {'x': x, 'y': y})
            else:
                obs = env.step(_GA[str(a)])
            cur = _v1163_grid(obs); lv2 = int(obs.levels_completed or 0)
            if lv2 > lv:
                fin = _gm.finished_board(pre, getattr(obs, 'frame', None)) if pre is not None else None
                pairs.append((start, fin)); start = cur; lv = lv2
                warm_game.extend((None, w[1], w[2], w[3]) for w in warm); warm = []
                if len(warm_game) > 480:
                    del warm_game[:-400]
            elif a == 'RESET':
                start = cur; warm = []
            elif pre is not None and cur is not None and pre.shape == cur.shape:
                trans.append((pre, cur))
                warm.append((ob0, a, pre, cur))
                if len(warm) > 240:
                    del warm[:-200]
        hud = _gm.hud_from_transitions(trans, g.shape) if (g is not None and trans) else None
        _V1163_PRIOR.update(key=key, pairs=[(s, f) for s, f in pairs if s is not None and f is not None], hud=hud, warm=warm,
                            warm_game=warm_game)
    try:
        from arc_env import adapter as _ad163
        _ad163.v1160_take()                              # the prefix is not the brain's play (v1.160)
    except Exception:
        pass
    return obs


def _v1163_feed_prior(brain):
    st = getattr(brain, '_steer', None)
    if st is not None and hasattr(st, 'prior_levels') and _V1163_PRIOR.get('pairs'):
        try:
            st.prior_levels(_V1163_PRIOR['pairs'], _V1163_PRIOR.get('hud'))
        except Exception:
            pass


def _v1167_play_levels(brain, env, prefix, caps, first_reset):
    """v1.167 CONTINUATION (nas_1166_report.md §23.5): the play starts at level L (the prefix replayed, as in 1153) and goes ON
    after each clear into the next len(caps) - 1 levels, the SAME brain and steering (its goal map and graph learned on L carry
    over). Level j has its own budget caps[j] (k x its human count; the start level's scaled to the moves left).
    Resets, measured on the real engine (vc33, sp80):
      * a game over or a brain RESET on the START level replays the prefix (1153's rule: the game restarts, the prefix again);
      * on a CONTINUATION level a RESET after >= 1 action restarts that level (levels_completed stays) -- one action counted;
      * a RESET with NO action since that level began (or since the last reset) would restart the GAME at level 1: it is not
        sent; it is counted as one action (the brain asked for it; on Kaggle it would have cost the whole game).
    Returns cleared = the number of levels cleared in a row, steps_to = the cumulative action count at each clear (every
    action counts, RESETs included: Kaggle's count), actions = steps_to[0]."""
    obs = _v1153_replay(env, prefix); start = int(obs.levels_completed or 0)
    try:
        brain.reset(obs) if first_reset else brain.soft_reset(obs)
    except Exception:
        pass
    _v1163_feed_prior(brain)
    caps = [max(1, int(c)) for c in caps]
    total = 0; resets = 0; steps_to = []; j = 0; lv_steps = 0; acted = False; held = 0
    while j < len(caps) and lv_steps < caps[j]:
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            obs = _v1153_replay(env, prefix) if j == 0 else env.reset()
            brain.soft_reset(obs); resets += 1; total += 1; lv_steps += 1; acted = False; continue
        brain._v78_left = caps[j] - lv_steps                # v1.178: the actions this level has left (the planner never plans past them)
        action, data = brain.play_step(obs)
        if action is GameAction.RESET:
            if j == 0:
                obs = _v1153_replay(env, prefix)
            elif acted:
                obs = env.reset()
            else:
                held += 1                                          # would restart the game: not sent, still counted
            brain.soft_reset(obs); resets += 1; total += 1; lv_steps += 1; acted = False; continue
        obs = env.step(action, data=data); total += 1; lv_steps += 1; acted = True
        if obs.state == GameState.GAME_OVER and hasattr(brain, 'v77_death'):
            brain.v77_death(obs)                            # v1.177: the death frame, diagnosed (budget / fatal)
        if obs.state == GameState.GAME_OVER and hasattr(brain, 'v55_note_death'):
            brain.v55_note_death()
        if int(obs.levels_completed or 0) >= start + j + 1:
            steps_to.append(total); j += 1; lv_steps = 0; acted = False
            if obs.state == GameState.WIN:
                break
    return {'cleared': len(steps_to), 'actions': (steps_to[0] if steps_to else 0), 'steps_to': steps_to, 'resets': resets,
            'held_resets': held, 'levels_played': min(len(caps), len(steps_to) + 1),
            'beh': dict(getattr(brain, '_v55_beh', {}) or {}),
            'is_real': bool(getattr(env, 'is_real', getattr(env, 'backend', '') == 'real'))}


def _v1153_play_level(brain, env, prefix, max_steps, first_reset, caps=None):
    """v1.153 (unchanged) + v1.163: the prefix's completed levels are handed to the steering once, after the reset.
    v1.167: with caps (one budget per level, >= 2 levels) the play continues after the clear (_v1167_play_levels)."""
    if caps is not None and len(caps) >= 2:
        return _v1167_play_levels(brain, env, prefix, caps, first_reset)
    obs = _v1153_replay(env, prefix); start = int(obs.levels_completed or 0)
    try:
        brain.reset(obs) if first_reset else brain.soft_reset(obs)
    except Exception:
        pass
    _v1163_feed_prior(brain)
    steps = 0; resets = 0
    while steps < max_steps:
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            obs = _v1153_replay(env, prefix); brain.soft_reset(obs); resets += 1; steps += 1; continue
        brain._v78_left = max_steps - steps                 # v1.178: the actions this try has left (the planner never plans past them)
        action, data = brain.play_step(obs)
        if action is GameAction.RESET:
            obs = _v1153_replay(env, prefix); brain.soft_reset(obs); resets += 1; steps += 1; continue
        obs = env.step(action, data=data); steps += 1
        if obs.state == GameState.GAME_OVER and hasattr(brain, 'v77_death'):
            brain.v77_death(obs)                            # v1.177: the death frame, diagnosed (budget / fatal)
        if obs.state == GameState.GAME_OVER and hasattr(brain, 'v55_note_death'):
            brain.v55_note_death()                          # v1.155
        if int(obs.levels_completed or 0) > start:
            return {'cleared': 1, 'actions': steps, 'resets': resets, 'beh': dict(getattr(brain, '_v55_beh', {}) or {}),
                    'is_real': bool(getattr(env, 'is_real', getattr(env, 'backend', '') == 'real'))}       # v1.156
    return {'cleared': 0, 'actions': 0, 'resets': resets, 'beh': dict(getattr(brain, '_v55_beh', {}) or {}),
            'is_real': bool(getattr(env, 'is_real', getattr(env, 'backend', '') == 'real'))}


# ============ v1.164: per-try PROGRESS toward the level's finished board (prog1164.py; a teacher's score for the gate, never
# shown to the brain) and the try's tally kept across in-try replays (1160-1163 kept only the last life). ============
try:
    import prog1164 as _p164
    _p164.install()
except Exception as _e164:                                                      # pragma: no cover
    print(f'  ! v1.164 prog1164 not installed: {type(_e164).__name__}: {_e164}', flush=True)


# ============ v1.173: THE WARM PREFIX AND THE GAME SENSE (effect1173.py). The prefix replay above records this level's prefix moves;
# after the steering's prior levels, effect1173.feed_warm hands them to the steering's memories (warm_prefix >= 1) and lets the
# network watch the last of them (warm_prefix 2). cold_level_try tells the sense which game is played -- ONLY for its hit / miss
# tally (MULTINXON_SENSE_TRUTH; the steering never decides with it). Inert at the defaults. ============
_v1173_feed_prior0 = _v1163_feed_prior


def _v1163_feed_prior(brain):
    _v1173_feed_prior0(brain)
    try:
        import effect1173 as _e173
        _e173.feed_warm(brain, _V1163_PRIOR)
    except Exception as _e:
        try:
            brain._v1173_error = f'feed_warm: {type(_e).__name__}: {_e}'[:160]
        except Exception:
            pass


_v1173_cold0 = cold_level_try


def cold_level_try(game_id, brain_path, arch_path, prefix, max_steps, seed, prefer_real=True, force_offline=False, micro_steps=3,
                   caps=None):
    os.environ['MULTINXON_SENSE_TRUTH'] = str(game_id).split('-')[0]
    return _v1173_cold0(game_id, brain_path, arch_path, prefix, max_steps, seed, prefer_real=prefer_real,
                        force_offline=force_offline, micro_steps=micro_steps, caps=caps)
