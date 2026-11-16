"""prog1164.py -- NAS 1164: a GRADED, per-try PROGRESS measure for selection (nas_1163_report.md §20, change 2), and the
per-try tally made whole.

WHAT IT MEASURES. For the level a try plays (game, L), boards1164.npz holds the level's START board and its FINISHED
board (the board at the moment the level completes, from the verified 1158 solutions) and the game's HUD mask. The goal
cells D are the cells where start != finished, HUD excluded. During the try, every frame on level L is scored

    m(frame) = share of D cells that hold their FINISHED colour

and the try's progress is   prog = min(0.999, max(0, (best m - m0) / max(ref - m0, (1 - m0) / 2)))   (1.0 only when the level
completes), where m0 is
m at the moment the brain takes over (0 at a level's start, by construction; > 0 after a curriculum prefix) and ref is the
best m the verified expert solution itself reaches BEFORE its completing move (prog_ref1164.json; measured on the real
engine: 165 level starts identical to the boards; ref median 0.87 over 156 levels, below 0.3 on 10 (tn36 L3 0.06, lp85 L8 0.08, su15 L2 0.12, lp85 L1 0.19) -- the
completing move changes much of the board). ~1.0 = as close as the expert got one move before clearing; the span is never
less than half the remaining distance (at a start a few moves from the end the expert's own remaining gain is tiny, and a
brain that changes more cells would saturate it -- ka59 L1 at 3 moves left: 999 without a clear). The tally carries
'prog' (0-1000), 'prog_m0' and 'prog_d' (the goal-cell count).

WHAT IT IS NOT. A TEACHER'S SCORE used by the selection gate (nas1164.py) and by nothing else: the brain, its steering
and its sense never see the finished board (the lab's labelled oracle arm, steer_goal_oracle, is the one exception and
the preflight refuses it in any fleet cell). It rewards brains whose plays move the board toward the level's goal even
when they do not complete it -- a dense signal where a clear is a rare binary (1163: brain effect = null on 22 lines).

THE TALLY FIX. Since 1160, game_loop's prefix replay called adapter.v1160_take() to discard the replay's own frames --
which also discarded everything the try had counted so far whenever a game over or a brain RESET replayed the prefix
mid-try. 1160-1163 per-try tallies held only the LAST life of a try. install() replaces that with snapshot / restore:
the replay's frames are dropped, the try's counts are kept.

    import prog1164; prog1164.install()          # game_loop does it at import (workers included)
    prog1164.set_target(game, level)             # cold_level_try / play_learned_brain / play_game / train_level_life do it
                                                 # per try; _v1153_play_level marks the try's FIRST replay (a fresh tally)"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_B = None
_T = {'key': None, 'D': None, 'fin': None, 'L': 1, 'n': 0, 'ref': 1.0}
_R = None


def _refs():
    global _R
    if _R is None:
        try:
            import json
            _R = json.load(open(os.path.join(HERE, 'prog_ref1164.json'), encoding='utf-8')).get('ref') or {}
        except Exception:
            _R = {}
    return _R
_INSTALLED = {'done': False}


def _boards():
    global _B
    if _B is None:
        p = os.path.join(HERE, 'boards1164.npz')
        try:
            z = np.load(p)
            _B = {k: z[k] for k in z.files}
        except Exception:
            _B = {}
    return _B


def goal_cells(game, L):
    """(D, finished) for (game, level L): D = start != finished, the game's HUD excluded (all of it when that leaves < 4)"""
    B = _boards()
    s = B.get(f'{game}:{int(L)}:start'); f = B.get(f'{game}:{int(L)}:finished')
    if s is None or f is None or s.shape != f.shape:
        return None, None
    D = s != f
    hud = B.get(f'{game}:hud')
    if hud is not None and hud.shape == D.shape:
        Dm = D & ~hud
        if int(Dm.sum()) >= 4:
            D = Dm
    if int(D.sum()) < 1:
        return None, None
    return D, f


def set_target(game, L=None):
    """the level the next try plays: (game, L); L defaults to MULTINXON_START_LEVEL (a level line) or 1"""
    if L is None:
        try:
            L = int(os.environ.get('MULTINXON_START_LEVEL', '0') or 0)
        except Exception:
            L = 0
        L = L if L >= 1 else 1
    g = str(game or '').split('-')[0]
    key = (g, int(L))
    if _T['key'] != key:
        D, f = goal_cells(g, L)
        r = (_refs().get(g) or {}).get(str(int(L)))
        _T.update(key=key, D=D, fin=f, L=int(L), n=int(D.sum()) if D is not None else 0, ref=float(r) if r else 1.0)
    return _T['n']


def clear_target():
    _T.update(key=None, D=None, fin=None, n=0)


def match(grid):
    """m(frame): the share of the goal cells holding their finished colour, or None"""
    D = _T['D']
    if D is None or grid is None:
        return None
    try:
        a = np.asarray(grid)
        if a.ndim == 3:
            a = a[-1]
        if a.shape != D.shape:
            return None
        return float((a[D] == _T['fin'][D]).mean())
    except Exception:
        return None


def _track(fd, t):
    """called for every frame at the engine boundary (after the tally's own bookkeeping)"""
    if _T['D'] is None:
        return
    lv = int(getattr(fd, 'levels_completed', 0) or 0)
    if lv >= _T['L']:
        t['_pdone'] = 1                                   # the target level completed in this try
        return
    if lv != _T['L'] - 1:
        return
    fr = getattr(fd, 'frame', None)
    if not fr:
        return
    m = match(fr[-1] if isinstance(fr, list) else fr)
    if m is None:
        return
    if '_pm0' not in t:
        t['_pm0'] = m
    t['_pcur'] = m
    if m > t.get('_pbest', -1.0):
        t['_pbest'] = m


def prog_of(t):
    if t.get('_pdone'):
        return 1000
    if '_pm0' not in t or '_pbest' not in t:
        return None
    m0 = float(t['_pm0']); b = float(t['_pbest'])
    if m0 >= 1.0:
        return 0
    ref = float(_T.get('ref') or 1.0)
    span = max(ref - m0, 0.5 * (1.0 - m0))                          # never less than half the remaining distance: near a
    return int(round(1000.0 * min(0.999, max(0.0, (b - m0) / span))))   # level's end the expert's own remaining gain is tiny


def _adapter():
    from arc_env import adapter as AD
    return AD


def _patch_adapter(AD):
    """the adapter's tally: progress per frame, and 'prog' in every take (idempotent per module object: a re-imported
    adapter is patched again)"""
    if getattr(AD._v1160_note, '_v1164', False) and getattr(AD.v1160_take, '_v1164', False):
        return                                            # markers on the FUNCTIONS: a reload redefines them, unpatched
    AD._v1164_prog = True
    _note0 = AD._v1160_note

    def _note(fd, kind):
        _note0(fd, kind)
        try:
            _track(fd, AD._V1160)
        except Exception:
            pass
    _note._v1164 = True
    AD._v1160_note = _note
    _take0 = AD.v1160_take

    def v1160_take():
        t = AD._V1160
        try:
            p = prog_of(t)
            if p is not None:
                t['prog'] = p; t['prog_m0'] = int(round(1000 * float(t.get('_pm0', 0.0)))); t['prog_d'] = int(_T['n'])
        except Exception:
            pass
        return _take0()
    v1160_take._v1164 = True
    AD.v1160_take = v1160_take
    AD.v1164_snapshot = lambda: _snapshot(AD)
    AD.v1164_restore = lambda s, obs=None: _restore(AD, s, obs)


def install():
    """patch the adapter's tally (progress per frame; 'prog' in every take) and game_loop's prefix replay (snapshot /
    restore instead of discarding the try's counts); set the target at the start of every try. Idempotent PER MODULE
    OBJECT (markers on the modules, not a global flag): preflight re-imports game_loop / arc_brain, and a re-imported
    module must be patched again."""
    try:
        AD = _adapter()
    except Exception:
        return False
    _patch_adapter(AD)
    try:
        import game_loop as GL
        _patch_game_loop(GL)
    except Exception:
        pass
    _INSTALLED['done'] = True
    return True


def _snapshot(AD):
    t = AD._V1160
    s = dict(t)
    if isinstance(t.get('_seen'), set):
        s['_seen'] = set(t['_seen'])
    return s


def _restore(AD, s, obs=None):
    """the replay's frames are dropped and the try's counts come back. The progress baseline is the board the brain
    takes over: set on the first replay of a try (from the replayed board), kept on later ones"""
    first = bool(_T.pop('fresh', False))                  # the try's first replay (set by _v1153_play_level's wrapper)
    new = {} if first else dict(s)                        # (1160-1163 semantics for the first replay: a fresh tally)
    if not first:
        new['replays'] = int(new.get('replays', 0)) + 1  # in-try restarts (game over / brain RESET), counted
    if '_pm0' not in new and obs is not None:
        try:
            if int(getattr(obs, 'levels_completed', 0) or 0) == _T['L'] - 1:
                fr = getattr(obs, 'frame', None)
                m = match(fr[-1]) if fr else None
                if m is not None:
                    new['_pm0'] = m; new['_pbest'] = m; new['_pcur'] = m
        except Exception:
            pass
    AD._V1160.clear(); AD._V1160.update(new)


def _patch_game_loop(GL, AD=None):
    if getattr(getattr(GL, '_v1153_replay', None), '_v1164', False):
        return                                            # markers on the FUNCTIONS: importlib.reload redefines them
    GL._v1164_prog = True
    _replay0 = GL._v1153_replay                           # the 1163 replay (restart the game, replay, discard its tally)

    def _v1153_replay(env, prefix):
        AD = _adapter(); _patch_adapter(AD)
        s = _snapshot(AD)
        obs = _replay0(env, prefix)
        _restore(AD, s, obs)
        return obs
    _v1153_replay.__doc__ = (_replay0.__doc__ or '') + '\n    v1.164: the try\'s tally is kept across in-try replays.'
    _v1153_replay._v1164 = True
    GL._v1153_replay = _v1153_replay

    _pl0 = GL._v1153_play_level                           # one call = one try (or one training episode) at level L

    def _v1153_play_level(*a, **k):
        _T['fresh'] = True
        return _pl0(*a, **k)
    _v1153_play_level.__doc__ = _pl0.__doc__
    _v1153_play_level._v1164 = True
    if not getattr(_pl0, '_v1164', False):
        GL._v1153_play_level = _v1153_play_level

    for name in ('cold_level_try', 'play_learned_brain', 'play_game', 'train_level_life'):
        f = getattr(GL, name, None)
        if f is None or getattr(f, '_v1164', False):
            continue

        def _wrap(f=f, name=name):
            def w(*a, **k):
                gid = a[0] if a else k.get('game_id')
                try:
                    AD = _adapter(); _patch_adapter(AD)
                    set_target(gid)
                    for kk in ('_pm0', '_pbest', '_pcur', '_pdone'):   # a try measures its own progress
                        AD._V1160.pop(kk, None)
                except Exception:
                    pass
                return f(*a, **k)
            w._v1164 = True; w.__name__ = name; w.__doc__ = f.__doc__
            return w
        setattr(GL, name, _wrap())
