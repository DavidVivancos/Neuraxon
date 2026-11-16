"""wall1172.py -- NAS 1172 (nas_1171_report.md section 20, change 4; FrameLab 15's movecheck): two genes for the walls. Both are OFF
by default (an absent gene plays exactly as 1171), both are bandit arms (evo1172), and the lab only sets their priors per level
(gene_priors_1172.json): the brain still makes every move.

cand_tier 0 / 1 / 2 -- THE CANDIDATE COVERAGE TIER
  0  today's candidates: the steering's keys + its 48 objects (rarest / smallest first), and the brain's click mask (its size band);
  1  + EVERY OBJECT: the steering offers every object of the board (no 48 cut), and the brain's click mask adds the centroid of every
     object outside its band;
  2  + A 4-PX GRID: tier 1 and the centre of every 4 x 4 cell of the board outside the HUD, in the steering and in the mask.
  FL15 (4,014 verified moves within 34 of a level's end): 6.9% are not among the candidates (21.7% of the clicks); the last moves of
  tn36 (L2-L7), ft09 L5, lp85 and su15 are uncovered; tier 1 covers tn36 and ft09, su15 needs the grid. A random proposer over every
  object lost 0.2 points overall (n.s.), so the tier is a gene each line selects, never a blanket switch.
commit_sweep 0 / 1 -- THE PER-LEVEL KEY SWEEP
  after commit_stall (8; MULTINXON_COMMIT_STALL) steps without a NEW SCREEN in this level, each key action the game offers (ACTION1-5,
  7) that has not been pressed IN THIS LEVEL is pressed once (one per stalled step, in action order), then the brain leads again; the
  set restarts at every level (FL15: 'per level' worked where 'per board' failed). FL15: committing once per level lifted the levels
  whose last move comes after the goal is already satisfied (g50t, sc25, cd82 L5 / L6, m0r0 L6, tr87) from 21% to 45% one move from
  the end; it cost wa30 (69 -> 38 owned moves): a gene. No goal is read: the trigger is the stall.
The commit sweep wraps ArcBrain.play_step OUTERMOST (after the 1167 frontier); an override goes through the frontier's _follow, so the
brain's books, the steering's last step and the capability's live set record the executed key."""
from __future__ import annotations

import os

import numpy as np

KEYS = (1, 2, 3, 4, 5, 7)
GRID = 4


def _gene(params, name, default):
    v = getattr(params, name, None)
    return default if v in (None, '') else v


def _int(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def _val(a):
    try:
        return int(getattr(a, 'value', a))
    except Exception:
        return 0


def tier_of(params):
    e = os.environ.get('MULTINXON_PIN_CAND_TIER', '')
    if e.strip() != '':
        return max(0, min(2, _int(e)))
    return max(0, min(2, _int(_gene(params, 'cand_tier', 0))))


def sweep_of(params):
    e = os.environ.get('MULTINXON_PIN_COMMIT_SWEEP', '')
    if e.strip() != '':
        return _int(e) == 1
    return _int(_gene(params, 'commit_sweep', 0)) == 1


def stall_of(params):
    s = _int(_gene(params, 'commit_stall', 0))
    if s <= 0:
        s = _int(os.environ.get('MULTINXON_COMMIT_STALL', '8'), 8)
    return max(1, s)


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


# ======================================================================================================== cand_tier: the extra points
def extra_points(grid, tier, bg=None, ignore=None):
    """[(y, x)]: tier >= 1 every object's centre cell (the cell of the object nearest its centroid, so a click lands ON it), tier 2 also
    the centre of every 4 x 4 cell outside the ignore mask. grid: 2-D numpy int array"""
    out = []
    if tier <= 0 or grid is None or grid.ndim != 2 or grid.size == 0:
        return out
    if bg is None:
        bg = int(np.bincount(np.clip(grid.ravel(), 0, 15), minlength=16).argmax())
    if ignore is None:
        ignore = np.zeros(grid.shape, dtype=bool)
    try:
        from agent1160 import objects
        for col, s, y, x in objects(grid, ignore, bg):
            out.append((int(y), int(x)))
    except Exception:
        pass
    if tier >= 2:
        H, W = grid.shape
        for y in range(GRID // 2, H, GRID):
            for x in range(GRID // 2, W, GRID):
                if not ignore[y, x]:
                    out.append((y, x))
    return list(dict.fromkeys(out))


def _install_steering():
    """Agent.candidates (the base of every steering): + every object (tier 1) + the 4-px grid (tier 2); Steer.__init__ reads the gene"""
    import agent1160 as AG
    import steer1160 as S0
    if getattr(AG.Agent.candidates, '_v1172', False):
        return
    _cand0 = AG.Agent.candidates

    def candidates(self, g, avail):
        out = _cand0(self, g, avail)
        t = int(getattr(self, '_cand_tier', 0) or 0)
        if t <= 0 or 6 not in avail:
            return out
        try:
            seen = {a for a, _ in out}; ign = self.ignore(); n = 0
            for col, s, y, x in AG.objects(g, ign, self.bg):
                if ('C', y, x) not in seen:
                    seen.add(('C', y, x)); out.append((('C', y, x), ('C', col, AG.size_bucket(s)))); n += 1
            if t >= 2:
                H, W = g.shape
                for y in range(GRID // 2, H, GRID):
                    for x in range(GRID // 2, W, GRID):
                        if ign[y, x] or ('C', y, x) in seen:
                            continue
                        seen.add(('C', y, x)); out.append((('C', y, x), ('G', int(g[y, x])))); n += 1
            if n:
                _tally('tier_cands', n)
        except Exception as e:
            self._v1172_error = f'candidates: {type(e).__name__}: {e}'[:160]
        return out
    candidates._v1172 = True
    candidates.__doc__ = _cand0.__doc__
    AG.Agent.candidates = candidates

    _sinit0 = S0.Steer.__init__

    def __init__(self, params, seed=0):
        _sinit0(self, params, seed)
        self._cand_tier = tier_of(params)
    __init__._v1172 = True
    S0.Steer.__init__ = __init__


# ======================================================================================================== the brain hooks
def install(ArcBrain):
    """wrap ArcBrain.__init__, _click_candidates, play_step (outermost), reset; Agent.candidates; idempotent; inert at the defaults"""
    if getattr(ArcBrain, '_v1172_wall', False):
        return True
    ArcBrain._v1172_wall = True
    _install_steering()
    _init0 = ArcBrain.__init__; _cc0 = ArcBrain._click_candidates; _ps0 = ArcBrain.play_step; _rs0 = ArcBrain.reset

    def __init__(self, params, *a, **k):
        _init0(self, params, *a, **k)
        self._v1172_tier = tier_of(params)
        self._v1172_cc = None
        self._v1172_cs = ({'lv': None, 'pressed': set(), 'seen': set(), 'stall': 0, 'stall_at': stall_of(params), 'n': 0}
                          if sweep_of(params) else None)

    def _click_candidates(self, grid):
        out = _cc0(self, grid)
        t = int(getattr(self, '_v1172_tier', 0) or 0)
        if t <= 0 or not grid or not grid[0]:
            return out
        try:
            import arc_brain as AB
            sig = AB._frame_signature(grid)
            c = getattr(self, '_v1172_cc', None)
            if c is None or c[0] != sig:
                g = np.asarray(grid, dtype=np.int16)
                c = (sig, extra_points(g, t))
                self._v1172_cc = c
            have = set(out or [])
            add = [p for p in c[1] if p not in have]
            return (list(out or []) + add) if add else out
        except Exception as e:
            self._v1172_error = f'mask: {type(e).__name__}: {e}'[:160]
            return out

    def play_step(self, frame):
        a, d = _ps0(self, frame)                                                # the brain, its steering and the frontier always run
        cs = getattr(self, '_v1172_cs', None)
        if cs is None:
            return a, d
        try:
            lv = int(getattr(frame, 'levels_completed', 0) or 0)
            if cs['lv'] != lv:
                cs.update(lv=lv, pressed=set(), seen=set(), stall=0)
            fr = getattr(frame, 'frame', None)
            key = None
            if fr:
                g = np.asarray(fr[-1], dtype=np.int16)
                key = hash(g.tobytes())
            if key is None or key in cs['seen']:
                cs['stall'] += 1
            else:
                cs['seen'].add(key); cs['stall'] = 0
            v = _val(a)
            if v in KEYS:
                cs['pressed'].add(v)
            if cs['stall'] < cs['stall_at']:
                return a, d
            av = sorted({_val(x) for x in (getattr(frame, 'available_actions', None) or [])} & set(KEYS))
            todo = [x for x in av if x not in cs['pressed']]
            if not todo or v in todo:
                return a, d
            k = todo[0]
            cs['pressed'].add(k); cs['n'] += 1
            from arc_env.structs import GameAction
            a2, d2 = GameAction(k), {}
            try:
                import frontier1167 as F7
                F7._follow(self, frame, a2, d2)
            except Exception:
                self.last_action = a2
            _tally('commit_sweep')
            return a2, d2
        except Exception as e:
            self._v1172_error = f'commit: {type(e).__name__}: {e}'[:160]
            return a, d

    def reset(self, frame):
        out = _rs0(self, frame)
        cs = getattr(self, '_v1172_cs', None)
        if cs is not None:
            cs.update(lv=None, pressed=set(), seen=set(), stall=0)
        return out

    for f in (__init__, _click_candidates, play_step, reset):
        f._v1172 = True
    ArcBrain.__init__ = __init__; ArcBrain._click_candidates = _click_candidates
    ArcBrain.play_step = play_step; ArcBrain.reset = reset
    return True
