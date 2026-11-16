"""effect1175.py -- NAS 1175: the brain side of nas_1174_report.md section 25, change 3 (a goal whose wrong cells cannot steer:
FrameLab 19 in the NAS form). Every brain still proposes every action; nothing here knows a game's solution. All of it is behind a
gene (goal_what) or a new value of a gene (goal_strict 3) with environment pins for A/B tests; a brain without them plays exactly
as in 1174.

goal_what 0 / 1 (MULTINXON_PIN_GOAL_WHAT; MULTINXON_GOAL_WHAT_DEFAULT for a brain without the gene) -- WHAT, NOT ONLY WHERE
  The 1163 goal map names WHERE something must end up (cells that change and end non-background on the levels seen); a named cell
  "arrives" by ANY change that is not background, so a candidate that disturbs a wrong cell lowers h: the map's wrong cells steer
  (FL19's stated limitation; the 1174 audit: 19.5% of the named open cells are right). With goal_what 1 every named cell may carry
  a TARGET COLOUR learned from the same levels the map learned from: on each level seen, the arrival cells' (start colour -> finished
  colour) transitions are counted; a named cell whose start colour has a decisive finished colour (>= 2 arrivals, >= 60% share; else
  the levels' dominant finished colour at >= 60%) gets it as its target. A cell with a target arrives ONLY at that colour (h counts
  the named cells not at their colour); a cell without one keeps the 1163 rule. A wrong cell that no action paints its colour adds
  the same to every candidate: inert. The model lives with the brain's play of ONE game (as the goal map). Level 1 from its start
  has no level before it: no goal, nothing changes.
goal_strict 3 -- STRICTNESS BY THE MOVES LEFT (report change 3)
  2 (the top 25% of the named cells with p >= 0.7) on a play that starts 1-2 moves from the end of its level, 1 (50%, p >= 0.5) at
  3-8 moves, 0 beyond (far from the end recall matters); a continuation level of the same play starts at its own beginning (0).
THE AUDIT'S COLOUR COLUMN (measurement only; effect1174's audit is the where-column)
  at the first goal fit of a play: st_goal_what (named cells with a target), st_goal_what_tp (those that are true arrivals),
  st_goal_what_ok (those whose target is the true finished colour). what-precision = ok / tp.
Tallies: st_goal_what_on (plays with goal_what 1 and a fitted table), st_goal_strict3 (plays at goal_strict 3), st_what_h (h
computed with targets)."""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import effect1173 as E3
import effect1174 as E4

WHAT_MIN_N = 2               # arrivals of a start colour before its finished colour is trusted
WHAT_SHARE = 0.6             # the finished colour's share among those arrivals
_LAST = {'start': None, 'fin': None}
_PFX = {'doc': None}


def _gene(params, name):
    return getattr(params, name, None) if params is not None else None


def _int_raw(e, default=0):
    """an int from an environment variable's raw text (the caller reads it with os.environ.get, so the wiring audit sees the reader)"""
    try:
        e = '' if e is None else e
        return int(float(e)) if str(e).strip() != '' else int(default)
    except Exception:
        return int(default)


def what_of(params):
    """0 / 1: the pin, else the brain's gene, else the cell's default for a brain without the gene"""
    e = os.environ.get('MULTINXON_PIN_GOAL_WHAT', '')
    try:
        if e.strip() != '':
            return 1 if int(float(e)) >= 1 else 0
        g = _gene(params, 'goal_what')
        if g not in (None, ''):
            return 1 if int(float(g)) >= 1 else 0
    except Exception:
        pass
    return 1 if _int_raw(os.environ.get('MULTINXON_GOAL_WHAT_DEFAULT'), 0) >= 1 else 0


def strict_gene(params):
    """0 / 1 / 2 / 3: the pin, else the gene, else MULTINXON_STRICT_DEFAULT (a brain without the gene)"""
    e = os.environ.get('MULTINXON_PIN_GOAL_STRICT', '')
    try:
        if e.strip() != '':
            return max(0, min(3, int(float(e))))
        g = _gene(params, 'goal_strict')
        if g not in (None, ''):
            return max(0, min(3, int(float(g))))
    except Exception:
        pass
    return max(0, min(3, _int_raw(os.environ.get('MULTINXON_STRICT_DEFAULT'), 0)))


def strict_by_moves(left):
    if left is None:
        return 0
    return 2 if left <= 2 else (1 if left <= 8 else 0)


def moves_left_at_start():
    """the moves the expert still needs from the play's start (level MULTINXON_START_LEVEL, k = MULTINXON_START_STEP), or None"""
    try:
        L = int(os.environ.get('MULTINXON_START_LEVEL', '1') or 1); k = int(os.environ.get('MULTINXON_START_STEP', '0') or 0)
        game = os.environ.get('MULTINXON_SENSE_TRUTH', '').strip()
        if not game:
            return None
        if _PFX['doc'] is None:
            _PFX['doc'] = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'level_prefix.json'),
                                         encoding='utf-8'))['prefix']
        P = _PFX['doc'].get(game) or {}
        start = [] if L <= 1 else P.get(str(L)) or []
        nxt = P.get(str(L + 1))
        if nxt is None:
            return None
        return max(0, len(nxt) - len(start) - k)
    except Exception:
        return None


# ================================================================================================ the target-colour mask
class GoalMask(np.ndarray):
    """the goal map's bool cells with the target colours attached (.what: int16, -1 = no target) -- h_of reads them"""
    def __array_finalize__(self, obj):
        self.what = getattr(obj, 'what', None)


def _with_what(goal, T):
    m = np.asarray(goal, bool).view(GoalMask)
    m.what = T
    return m


# ================================================================================================ the goal map hooks
def _install_goalmap():
    import goalmap1163 as GM
    if getattr(GM.GoalModel, '_v1175_what', False):
        return
    GM.GoalModel._v1175_what = True
    _arr0 = GM.arrivals

    def arrivals(start, fin, ignore=None):
        m = _arr0(start, fin, ignore)
        _LAST['start'] = start; _LAST['fin'] = fin                     # the pair the next GoalModel.add learns its colours from
        return m
    GM.arrivals = arrivals
    _add0 = GM.GoalModel.add

    def add(self, start, arrive):
        ok = _add0(self, start, arrive)
        if ok:
            try:
                s0 = _LAST['start']; f0 = _LAST['fin']
                if s0 is not None and f0 is not None:
                    s = np.asarray(start); s1 = np.asarray(s0); f = np.asarray(f0); a = np.asarray(arrive, bool)
                    if s.shape == s1.shape == f.shape == a.shape and (s is s0 or np.array_equal(s, s1)):
                        pairs = getattr(self, 'what_pairs', None)
                        if pairs is None:
                            pairs = self.what_pairs = []
                        pairs.append((s[a].astype(np.int16), f[a].astype(np.int16)))
                        self._what_tab = None
            except Exception:
                pass
        return ok
    GM.GoalModel.add = add

    def what_table(self):
        tab = getattr(self, '_what_tab', None)
        if tab is not None:
            return tab
        by = collections.defaultdict(collections.Counter); allf = collections.Counter()
        for sc, fc in (getattr(self, 'what_pairs', None) or []):
            for a, b in zip(sc.tolist(), fc.tolist()):
                by[int(a)][int(b)] += 1; allf[int(b)] += 1
        dec = {}
        for a, c in by.items():
            n = sum(c.values()); col, k = c.most_common(1)[0]
            if n >= WHAT_MIN_N and k >= WHAT_SHARE * n:
                dec[a] = col
        glob_ = None
        if allf:
            col, k = allf.most_common(1)[0]
            if k >= WHAT_SHARE * sum(allf.values()):
                glob_ = col
        self._what_tab = tab = {'by': dec, 'global': glob_, 'n': int(sum(allf.values()))}
        return tab
    GM.GoalModel.what_table = what_table

    def targets(self, board, goal):
        """int16 target colours for the goal's cells (-1: no target; never the cell's own start colour)"""
        tab = self.what_table()
        b = np.asarray(board, np.int16); g = np.asarray(goal, bool)
        T = np.full(b.shape, -1, np.int16)
        if not tab['n']:
            return T
        for col in np.unique(b[g]).tolist():
            t = tab['by'].get(int(col), tab['global'])
            if t is None or int(t) == int(col):
                continue
            T[g & (b == col)] = int(t)
        return T
    GM.GoalModel.targets = targets
    _pred0 = GM.GoalModel.predict                                           # effect1174's (goal_strict 0 / 1 / 2)

    def predict(self, board, ignore=None, n=None):
        s3 = int(getattr(self, 'strict3', 0) or 0)
        keep = getattr(self, 'strict', 0)
        if s3:
            self.strict = int(getattr(self, 'strict_eff', 0) or 0)          # goal_strict 3: the level's strictness by the moves left
        try:
            goal, p = _pred0(self, board, ignore, n)
        finally:
            if s3:
                self.strict = keep
        if goal is None or not int(getattr(self, 'what', 0) or 0):
            return goal, p
        try:
            T = self.targets(board, goal)
            if (T >= 0).any():
                return _with_what(goal, T), p
        except Exception:
            pass
        return goal, p
    GM.GoalModel.predict = predict
    _h0 = GM.h_of                                                           # steer1164's (the oracle reference) over 1163's

    def h_of(board, start, goal):
        T = getattr(goal, 'what', None)
        if T is None:
            return _h0(board, start, goal)
        b = np.asarray(board); s = np.asarray(start)
        if b.shape != goal.shape or s.shape != goal.shape or T.shape != goal.shape:
            return _h0(board, start, goal)
        bg = GM.bgc(s)
        arrived = np.where(T >= 0, b == T, (b != s) & (b != bg))
        return int((np.asarray(goal, bool) & ~arrived).sum())
    GM.h_of = h_of


# ================================================================================================ the audit's colour column
def audit_what(st):
    """once per play, with effect1174's audit: the targets of the named open cells against the verified finished board"""
    if getattr(st, '_v1175_audited', False) or getattr(st, 'goal', None) is None or int(getattr(st, '_v1174_nstart', 0) or 0) > 1:
        return
    st._v1175_audited = True
    T = getattr(st.goal, 'what', None)
    if T is None:
        return
    game = os.environ.get('MULTINXON_SENSE_TRUTH', '').strip()
    try:
        L = int(os.environ.get('MULTINXON_START_LEVEL', '1') or 1)
    except Exception:
        L = 1
    if not game:
        return
    try:
        import goalmap1163 as GM
        F = E4._truth(game, L)
        cur = st._cur[0] if getattr(st, '_cur', None) else st.start_board
        if F is None or cur is None:
            return
        cur = np.asarray(cur, np.int16); goal = np.asarray(st.goal, bool)
        if F.shape != cur.shape or goal.shape != cur.shape:
            return
        tr = np.asarray(GM.arrivals(cur, F, None), bool)
        has = goal & (T >= 0)
        E3._tally('goal_what', int(has.sum())); E3._tally('goal_what_tp', int((has & tr).sum()))
        E3._tally('goal_what_ok', int((has & tr & (F == T)).sum()))
    except Exception as e:
        st._v1175_error = f'audit what: {type(e).__name__}: {e}'[:160]


def _what_on(st):
    """once per play: goal_what 1 and a colour table fitted from levels seen (st_goal_what_on)"""
    try:
        if getattr(st, '_v1175_what_on', False) or not getattr(st, 'goal_what', 0) or not getattr(st.gm, 'what_pairs', None):
            return
        st._v1175_what_on = True
        E3._tally('goal_what_on')
    except Exception:
        pass


# ================================================================================================ install
def install(Steer4):
    """idempotent; after effect1174.install"""
    if getattr(Steer4, '_v1175_effect', False):
        return True
    Steer4._v1175_effect = True
    _install_goalmap()
    _init0 = Steer4.__init__; _start0 = Steer4.start_level; _goal_only0 = getattr(Steer4, 'start_level_goal_only', None)
    _stats0 = Steer4.stats

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        sg = strict_gene(params)
        self.goal_strict3 = 1 if sg == 3 else 0
        self.goal_strict = sg                                               # effect1174 read 0..2; a default without the gene lands here
        self.goal_what = what_of(params)
        try:
            self.gm.strict = 0 if sg == 3 else sg
            self.gm.strict3 = self.goal_strict3; self.gm.what = self.goal_what
            self.gm.strict_eff = strict_by_moves(moves_left_at_start()) if sg == 3 else 0
        except Exception:
            pass
        self._v1175_audited = False; self._v1175_what_on = False

    def start_level(self, g):
        n_before = int(getattr(self, '_v1174_nstart', 0) or 0)
        if getattr(self, 'goal_strict3', 0):
            try:                                                            # the play's first level: its moves left; a continuation: 0
                self.gm.strict_eff = strict_by_moves(moves_left_at_start()) if n_before == 0 else 0
            except Exception:
                pass
        _start0(self, g)
        if n_before == 0 and getattr(self, 'goal_strict3', 0):
            E3._tally('goal_strict3')
        _what_on(self)
        audit_what(self)

    def stats(self):
        s = _stats0(self)
        s.update(goal_what=getattr(self, 'goal_what', 0), goal_strict3=getattr(self, 'goal_strict3', 0),
                 v1175_error=getattr(self, '_v1175_error', None))
        return s

    Steer4.__init__ = __init__; Steer4.start_level = start_level; Steer4.stats = stats
    if _goal_only0 is not None:
        def start_level_goal_only(self):                                    # an end-first play: the prior levels arrive after the start
            _goal_only0(self)
            _what_on(self)
            audit_what(self)
        Steer4.start_level_goal_only = start_level_goal_only
    return True
