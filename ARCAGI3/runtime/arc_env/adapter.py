"""
MultiNxonArcAGI3 — environment adapter.

One factory, two backends:

  * REAL    — the official local engine, if ``arc_agi`` + ``arcengine`` are
              importable.  Used automatically.  (``pip install arc-agi`` /
              the ARC-AGI Toolkit; see README.)
  * OFFLINE — the built-in games in ``arc_env.mock_games`` otherwise.

Both are presented to the rest of the project through the SAME thin surface:

    env.reset() -> FrameData
    env.step(action: GameAction, data: dict|None) -> FrameData
    env.action_space -> list[GameAction]
    env.observation_space -> FrameData
    env.info -> EnvironmentInfo
    env.backend -> "real" | "offline"

The real engine already exposes almost exactly this API; the shim below just
normalises its return objects into our ``FrameData``/``GameAction`` types
(by enum *name*, so it is robust to minor version drift) and translates our
``GameAction`` back to the engine's before calling ``step``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

# v1.060b: serialize REAL-engine Arcade creation. The official engine's
# Arcade()/ScorecardManager init is not safe to run from many threads at once
# (BestTest runs one thread per game). Without this lock, ~all but one of N
# concurrent _RealEnvShim constructions raise during init and silently fall back
# to the offline mock — which then can't find the real (hashed) game id and
# errors "Unknown offline game 'cd82-fb555c5d'". The lock guards only the brief
# CREATION; subsequent reset()/step() play stays fully parallel.
_ARCADE_CREATE_LOCK = threading.Lock()

from .structs import FrameData, GameState, GameAction, EnvironmentInfo
from .mock_games import make_offline_env, list_offline_games


def _default_env_dir() -> str:
    """Resolve the ARC-AGI-3 environment-files directory.

    Priority:
      1. ``MULTINXON_ARC_ENV_DIR`` if set (user override);
      2. ``<project_root>/environment_files`` if it exists — an ABSOLUTE path
         so the bundled games are found no matter what the current working
         directory is (important: the engine otherwise resolves
         'environment_files' relative to CWD, and multiprocessing 'spawn'
         workers may not share the launch directory);
      3. the bare relative ``'environment_files'`` as a last resort.
    """
    override = os.environ.get("MULTINXON_ARC_ENV_DIR")
    if override:
        return override
    # adapter.py lives at <project_root>/arc_env/adapter.py
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundled = os.path.join(root, "environment_files")
    if os.path.isdir(bundled):
        return bundled
    return "environment_files"


def _grids_to_lists(raw) -> List[List[List[int]]]:
    """Normalise a real-engine ``frame`` into a list of 2D int grids.

    The official engine returns ``frame`` as a list of 2D NUMPY arrays (one per
    visible grid). The brain expects plain Python ``list[list[int]]`` grids, so
    we convert here. Done by duck-typing (``ndim`` / ``tolist``) to avoid
    importing numpy into the otherwise dependency-free environment layer.
    Handles: a 3D ndarray (G,H,W), a single 2D ndarray, or a Python list whose
    elements are ndarrays or lists.
    """
    if raw is None:
        return []
    ndim = getattr(raw, "ndim", None)
    if ndim == 3:                      # ndarray (grids, rows, cols)
        return [g.tolist() for g in raw]
    if ndim == 2:                      # a single 2D ndarray
        return [raw.tolist()]
    out: List[List[List[int]]] = []
    try:
        for g in raw:                  # python list of grids
            if hasattr(g, "tolist"):
                out.append(g.tolist())
            else:
                out.append([list(row) for row in g])
    except TypeError:
        return []
    return out


# --------------------------------------------------------------------------- #
#  Real-engine detection
# --------------------------------------------------------------------------- #
def _quiet_arc_logging() -> None:
    """Silence the engine's verbose INFO logging (scorecard chatter, etc.).

    The official engine logs a lot at INFO via the standard ``logging`` module
    (and attaches its own StreamHandler), which floods NAS output across many
    worker processes. Drop every arc/engine logger AND its handlers to WARNING
    unless the user asked for debug output. Safe to call repeatedly — the
    engine creates more loggers as it runs, so callers re-invoke this right
    before constructing/using an environment.
    """
    if os.environ.get("MULTINXON_DEBUG"):
        return
    import logging
    names = set(logging.root.manager.loggerDict.keys())
    names |= {"arc_agi", "arc_agi.scorecard", "arc_agi.base", "arcengine"}
    for name in names:
        low = name.lower()
        if not (low.startswith("arc") or "arcengine" in low
                or "scorecard" in low):
            continue
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        for h in list(lg.handlers):
            try:
                h.setLevel(logging.WARNING)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
#  Real-engine detection
# --------------------------------------------------------------------------- #
def real_engine_available() -> bool:
    """True iff the official local engine can be imported."""
    try:
        import arc_agi  # noqa: F401
        try:
            import arcengine  # noqa: F401
        except Exception:
            # some versions re-export structs from arc_agi itself
            pass
        _quiet_arc_logging()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Real-engine shim
# --------------------------------------------------------------------------- #
class _RealEnvShim:
    """Wraps an official ``env`` so it speaks our structs."""

    backend = "real"

    def __init__(self, game_id: str, render_mode: Optional[str] = None,
                 offline: bool = True):
        import arc_agi
        # resolve the engine's GameAction / GameState enums
        try:
            from arcengine import GameAction as RGA, GameState as RGS  # type: ignore
        except Exception:  # pragma: no cover - version fallback
            from arc_agi import GameAction as RGA, GameState as RGS  # type: ignore
        self._RGA = RGA
        self._RGS = RGS

        # operation mode (OFFLINE local play when supported)
        arcade_kwargs = {}
        try:
            from arc_agi import OperationMode  # type: ignore
            arcade_kwargs["operation_mode"] = (
                OperationMode.OFFLINE if offline else OperationMode.ONLINE
            )
        except Exception:
            pass
        # The official engine ships WITHOUT game environment files; it reads
        # them from ``environments_dir``. Resolve to an ABSOLUTE path (bundled
        # under the project root) so it works regardless of CWD.
        env_dir = _default_env_dir()
        if env_dir:
            arcade_kwargs["environments_dir"] = env_dir
        # serialize Arcade creation across threads (see _ARCADE_CREATE_LOCK above)
        with _ARCADE_CREATE_LOCK:
            self._arc = arc_agi.Arcade(**arcade_kwargs)
            # The engine (re)configures logging when the Arcade is built; quiet it
            # again before make() so per-env scorecard/loader INFO lines stay off.
            _quiet_arc_logging()
            self._env = self._arc.make(game_id, render_mode=render_mode)
        if self._env is None:
            raise RuntimeError(f"arc_agi failed to create environment {game_id!r}")
        self.game_id = game_id

        # name -> our enum  /  our enum -> engine member
        self._state_by_name = {s.name: s for s in GameState}
        self._our_action_by_name = {a.name: a for a in GameAction}
        self._engine_action_by_name = {a.name: a for a in self._RGA}

    # -- conversions --------------------------------------------------------
    def _to_our_action(self, raw) -> Optional[GameAction]:
        # The real engine reports actions sometimes as enum members (with a
        # ``.name``) and sometimes as plain ints (their value, e.g. 1..7).
        # Handle both. Our GameAction is an IntEnum whose values match the
        # engine's, so an int maps directly.
        name = getattr(raw, "name", None)
        if name is not None:
            a = self._our_action_by_name.get(name)
            if a is not None:
                return a
        try:
            return GameAction(int(raw))
        except (ValueError, TypeError):
            return None

    def _to_engine_action(self, action: GameAction):
        return self._engine_action_by_name.get(action.name)

    def _to_our_state(self, raw_state) -> GameState:
        name = getattr(raw_state, "name", str(raw_state))
        return self._state_by_name.get(name, GameState.NOT_FINISHED)

    def _wrap(self, obs) -> FrameData:
        if obs is None:
            return FrameData(game_id=self.game_id, state=GameState.GAME_OVER)
        # Real frames are a list of 2D NUMPY grids; convert to plain Python
        # lists of ints so the brain's list-based grid logic (and frame-change
        # comparison) works without a numpy dependency and without hitting
        # numpy's ambiguous-truth-value error on ``row != row``.
        frame = _grids_to_lists(getattr(obs, "frame", None))
        avail = []
        for raw in getattr(obs, "available_actions", []) or []:
            a = self._to_our_action(raw)
            if a is not None:
                avail.append(a)
        state = self._to_our_state(getattr(obs, "state", None))
        levels = int(getattr(obs, "levels_completed", 0) or 0)
        # The real ARC-AGI-3 FrameData has NO per-frame ``score`` (score lives
        # in the scorecard / EnvironmentScore). Derive a monotone reward proxy
        # from level progress so the brain's score-delta -> dopamine signal
        # still fires when it clears a level, with a bonus for a WIN. If a
        # future engine version *does* expose a frame score, prefer it.
        raw_score = getattr(obs, "score", None)
        if raw_score is not None:
            score = int(raw_score)
        else:
            score = 10 * levels + (40 if state == GameState.WIN else 0)
        return FrameData(
            game_id=getattr(obs, "game_id", self.game_id),
            frame=frame,
            state=state,
            score=score,
            available_actions=avail,
            levels_completed=levels,
            # v1.015: the real engine exposes no within-level progress; mark it
            # unknown (-1.0) so the loop falls back to its grid-derived proxy.
            # If a future engine version adds a progress/fraction field, read it
            # here in preference to the proxy.
            level_progress=float(getattr(obs, "level_progress", -1.0)),
        )

    # -- public API ---------------------------------------------------------
    def reset(self) -> FrameData:
        return self._wrap(self._env.reset())

    def step(self, action: GameAction, data: Optional[Dict[str, Any]] = None,
             reasoning: Optional[Any] = None) -> FrameData:
        engine_action = self._to_engine_action(action)
        kwargs: Dict[str, Any] = {}
        if action.is_complex() and data:
            kwargs["data"] = {"x": int(data.get("x", 0)), "y": int(data.get("y", 0))}
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
        return self._wrap(self._env.step(engine_action, **kwargs))

    @property
    def action_space(self) -> List[GameAction]:
        out = []
        for raw in getattr(self._env, "action_space", []) or []:
            a = self._to_our_action(raw)
            if a is not None:
                out.append(a)
        return out

    @property
    def observation_space(self) -> FrameData:
        return self._wrap(getattr(self._env, "observation_space", None))

    @property
    def info(self) -> EnvironmentInfo:
        i = getattr(self._env, "info", None)
        if i is None:
            return EnvironmentInfo(game_id=self.game_id, title=self.game_id)
        return EnvironmentInfo(
            game_id=getattr(i, "game_id", self.game_id),
            title=getattr(i, "title", self.game_id),
            tags=list(getattr(i, "tags", []) or []),
        )

    def close(self) -> None:
        """v1.060b — release the engine env AND its open scorecard.

        The official engine opens a ScorecardManager + scorecard per Arcade and
        does NOT auto-close them. When BestTest runs one thread per game (and each
        game creates several envs: scored + warmup + re-grow), dozens of scorecards
        open near-simultaneously and the engine's open-scorecard limit is exceeded —
        new Arcade creations then fail and silently fall back to the offline mock
        ("Unknown offline game 'cd82-<hash>'"). Closing each env when we're done with
        it keeps the open count low so parallel runs work. Best-effort: tries the
        env's and the Arcade's close/cleanup/scorecard-close, whatever the installed
        engine version exposes, and never raises."""
        for obj in (getattr(self, "_env", None), getattr(self, "_arc", None)):
            if obj is None:
                continue
            for meth in ("close", "cleanup", "shutdown", "stop",
                         "close_scorecard", "end_scorecard"):
                fn = getattr(obj, meth, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
            # some engines expose a scorecard manager with a close-all
            sm = getattr(obj, "scorecard_manager", None) or \
                getattr(obj, "_scorecard_manager", None)
            if sm is not None:
                for meth in ("close_all", "close", "cleanup"):
                    fn = getattr(sm, meth, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
        self._env = None
        self._arc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------- #
#  Offline shim (mock games already speak our structs; add `.backend`)
# --------------------------------------------------------------------------- #
class _OfflineEnvShim:
    backend = "offline"

    def __init__(self, game_id: str, seed: int = 0):
        self._env = make_offline_env(game_id, seed=seed)
        self.game_id = self._env.game_id

    def reset(self) -> FrameData:
        return self._env.reset()

    def step(self, action: GameAction, data: Optional[Dict[str, Any]] = None,
             reasoning: Optional[Any] = None) -> FrameData:
        return self._env.step(action, data=data, reasoning=reasoning)

    @property
    def action_space(self) -> List[GameAction]:
        return self._env.action_space

    @property
    def observation_space(self) -> FrameData:
        return self._env.observation_space

    @property
    def info(self) -> EnvironmentInfo:
        return self._env.info

    def close(self) -> None:
        self._env = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def make_env(game_id: str = "off_reach", *, seed: int = 0,
             prefer_real: bool = True, render_mode: Optional[str] = None,
             force_offline: Optional[bool] = None):
    """Create an environment behind the unified interface.

    Parameters
    ----------
    game_id       : official game id (e.g. "ls20") or offline id/alias
                    ("reach"/"switch"/"match" or "off_reach"/...).
    seed          : RNG seed for the offline games (ignored by the real engine).
    prefer_real   : if True and the real engine is importable, use it.
    render_mode   : passed to the real engine (e.g. "terminal"); None = headless.
    force_offline : override; if True always use the offline games, if False
                    always try the real engine.  Also honoured via the env var
                    ``MULTINXON_FORCE_OFFLINE=1``.
    """
    if force_offline is None:
        force_offline = os.environ.get("MULTINXON_FORCE_OFFLINE", "") == "1"

    want_real = prefer_real and not force_offline
    if want_real and real_engine_available():
        # try the real engine. If the metadata-derived hashed id (e.g.
        # 'sp80-589a99af') doesn't match the version the engine actually has
        # (e.g. 'sp80-0ee2d095'), creation fails — so on failure we RETRY with the
        # BARE game name ('sp80'), which the engine resolves to its current
        # version. This makes BestTest robust to stale environment_files metadata
        # hashes (the real cause of the "Unknown offline game" fallback). Only
        # after both the hashed AND bare attempts fail do we drop to offline.
        bare = str(game_id).split('-')[0]
        candidates = [game_id]
        if bare != game_id:
            candidates.append(bare)
        last_exc = None
        for cand in candidates:
            for _attempt in range(2):
                try:
                    shim = _RealEnvShim(cand, render_mode=render_mode, offline=True)
                    if cand != game_id and os.environ.get("MULTINXON_DEBUG"):
                        print(f"[adapter] {game_id!r} version not found; "
                              f"using bare id {cand!r} (engine's current version)")
                    return shim
                except Exception as exc:
                    last_exc = exc
                    if os.environ.get("MULTINXON_DEBUG"):
                        print(f"[adapter] real engine create failed for {cand!r} "
                              f"(attempt {_attempt+1}): {exc}")
                    import time as _t
                    _t.sleep(0.15)
        if os.environ.get("MULTINXON_DEBUG"):
            print(f"[adapter] real engine giving up for {game_id!r}: {last_exc}; "
                  f"falling back to offline")
    return _OfflineEnvShim(game_id, seed=seed)


def real_game_ids(env_dir: Optional[str] = None) -> List[str]:
    """Return the official ARC-AGI-3 game ids the engine can see locally.

    The official engine ships the *engine* but not the game environment files;
    it discovers them under its ``environments_dir`` (default
    'environment_files', overridable via ``MULTINXON_ARC_ENV_DIR``). With no
    files present this returns ``[]``.
    """
    if not real_engine_available():
        return []
    try:
        import arc_agi
        kwargs: Dict[str, Any] = {}
        try:
            from arc_agi import OperationMode  # type: ignore
            kwargs["operation_mode"] = OperationMode.OFFLINE
        except Exception:
            pass
        env_dir = env_dir or _default_env_dir()
        if env_dir:
            kwargs["environments_dir"] = env_dir
        arc = arc_agi.Arcade(**kwargs)
        _quiet_arc_logging()
        get_envs = getattr(arc, "get_environments", None)
        if callable(get_envs):
            out = []
            for g in (get_envs() or []):
                gid = getattr(g, "game_id", None)
                out.append(str(gid) if gid is not None else str(g))
            return out
    except Exception:
        pass
    return []


def real_env_status(env_dir: Optional[str] = None) -> Dict[str, Any]:
    """Diagnostic summary of the real-engine backend (for --list)."""
    available = real_engine_available()
    ids = real_game_ids(env_dir) if available else []
    return {
        "engine_installed": available,
        "game_ids": ids,
        "env_dir": env_dir or _default_env_dir(),
    }


def list_games(prefer_real: bool = True) -> List[str]:
    """List available game ids for the active backend."""
    if prefer_real and real_engine_available():
        ids = real_game_ids()
        if ids:
            return ids
    return list_offline_games()


# ============ v1.160: THE PER-TRY TALLY, counted where every action passes (the engine boundary) ============
_V1160 = {}


def v1160_take():
    """the tally since the last call (and a fresh one): steps, resets, per-action counts, frames that did not change,
    distinct frames, game overs, level-ups -- the per-try action flow of the 1160 record"""
    global _V1160
    t = _V1160; _V1160 = {}
    return {k: v for k, v in t.items() if not k.startswith('_')}


def _v1160_note(fd, kind):
    t = _V1160
    fr = getattr(fd, 'frame', None) or []
    try:
        h = hash(tuple(tuple(r) for r in fr[-1])) if fr else None
    except Exception:
        h = None
    if kind != 'reset':
        t['steps'] = t.get('steps', 0) + 1
        t['a_' + kind] = t.get('a_' + kind, 0) + 1
        if h is not None and h == t.get('_last'):
            t['nochange'] = t.get('nochange', 0) + 1
    else:
        t['resets'] = t.get('resets', 0) + 1
    if h is not None:
        seen = t.setdefault('_seen', set()); seen.add(h); t['distinct'] = len(seen)
    t['_last'] = h
    st = getattr(fd, 'state', None)
    if getattr(st, 'name', str(st)) == 'GAME_OVER':
        t['game_over'] = t.get('game_over', 0) + 1
    lv = int(getattr(fd, 'levels_completed', 0) or 0)
    if lv > t.get('_lv', lv):
        t['levelups'] = t.get('levelups', 0) + (lv - t.get('_lv', lv))
    t['_lv'] = lv


_v1160_step_orig = _RealEnvShim.step
_v1160_reset_orig = _RealEnvShim.reset


def _v1160_step(self, action, data=None, reasoning=None):
    fd = _v1160_step_orig(self, action, data, reasoning)
    try:
        _v1160_note(fd, getattr(action, 'name', str(action)))
    except Exception:
        pass
    return fd


def _v1160_reset(self):
    fd = _v1160_reset_orig(self)
    try:
        _v1160_note(fd, 'reset')
    except Exception:
        pass
    return fd


_RealEnvShim.step = _v1160_step
_RealEnvShim.reset = _v1160_reset
