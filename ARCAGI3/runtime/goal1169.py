"""goal1169.py -- NAS 1169 change 7 (LAB ONLY): the goal chooses by its MODEL'S PREDICTION, and h counts what must STAY.

WHY (nas_1168_report.md §16d, §19.7; FrameLab 9, 183 levels x 3 seeds, the TRUE goal in every arm):
  goal1168's rule -- WHEN the brain's proposal has an effect signature that was tried on this level and never moved h, HOW the
  candidate whose signature has the best OBSERVED mean drop of h -- cleared 7.0 levels per seed, no better than no goal (6.7);
  FrameLab's rule -- WHEN the dynamics model does not predict that the proposal lowers h, HOW the candidate with the best
  PREDICTED drop of h in the current board -- 20.0. Keeping goal1168's WHEN and choosing by the prediction doubled it (14.0,
  +7.3 per seed against no goal, 26 / 4); the eager WHEN with the observed HOW was the worst arm (3.3: it replaced 74.5 of 100
  proposals and got stuck). The 1168 lab's goal2 (goal1168's rule, learned and oracle) was null at all four starts.
  FrameLab 8: the cells that must stay unchanged are a third of the true goal's value.

WHAT IT DOES (genes, all inert at 0; it needs the structured goal: steer_goal_struct 1, or the lab-only oracle
steer_goal_struct_oracle 1, and steer_goal_choose 1):
  steer_goal_how 1   HOW = the PREDICTED drop of h: for every candidate of this board, M1 (Steer3's online effect memory: the
                     remembered effects of the candidate's action signature -- move, to the click, recolour, vanish -- applied to
                     the objects they touch) predicts the next board; h is measured on it; the candidate with the most negative
                     predicted change of h is taken (< 0 only). A candidate whose signature M1 has never seen has no prediction.
  steer_goal_when 1  WHEN = M1 does not predict that the brain's proposal lowers h (its predicted change of h >= 0); an
                     unknown proposal is a probe and stands. With 0 the WHEN is goal1168's (the proposal's signature tried on this
                     level and never moved h). steer_goal_when 1 always runs with the predicted HOW (FrameLab 9's 3.3 arm is never
                     played).
  steer_goal_stay 1  h also counts STAY predicates: a colour whose cells were identical on the start and the finished board of
                     every earlier level of the game (learned), or of the level's true finished board (oracle), must stay; a board
                     that changes it is one predicate further from the goal.
The brain still proposes every step; the goal replaces a proposal only under the WHEN, and only by a candidate predicted to
lower h. TALLY: st_sg_pchoose (a predicted choice made), st_sg_pwhen (the WHEN fired), st_sg_pnone (no candidate predicted
to lower h), st_sg_unknown (candidates without a prediction), st_sg_stay (stay predicates in the goal)."""
from __future__ import annotations

import collections

import numpy as np

import goal1167 as G7
from steer1161 import _gene


# ================================================================================================ the stay predicates
def stay_truth(S, F, ignore=None):
    """{('stay', (c,)): held} for the colours of a level's start board: held when every cell of colour c is unchanged at the end"""
    S = G7._masked(S, ignore); F = G7._masked(F, ignore)
    if S.shape != F.shape:
        return {}, {}
    bg = G7.bg_of(S); out = {}; app = {}
    for c in np.unique(S):
        c = int(c)
        if c < 0 or c == bg:
            continue
        m = S == c
        app[('stay', (c,))] = True
        out[('stay', (c,))] = bool(((F == c) == m).all())
    return out, app


class StructGoal1169(G7.StructGoal):
    """goal1167's structured goal + STAY predicates (learned within the game, predicted where applicable and held every time)"""

    def __init__(self, mode='all', min_levels=1, stay=True):
        super().__init__(mode=mode, min_levels=min_levels)
        self.stay = bool(stay)

    def add(self, S, F, ignore=None):
        ok = super().add(S, F, ignore)
        if ok and self.stay:
            try:
                t, a = stay_truth(S, F, ignore)
                self.levels[-1][0].update(t); self.levels[-1][1].update(a)
            except Exception:
                pass
        return ok


_sat0 = G7.satisfied


def satisfied(p, board, start, ignore=None):
    """goal1167's predicates + 'stay': every cell of the colour is where it was on the level's start board"""
    if p and p[0] == 'stay':
        B = G7._masked(board, ignore); S0 = G7._masked(start, ignore)
        if B.shape != S0.shape:
            return False
        c = p[1][0]
        return bool(((B == c) == (S0 == c)).all())
    return _sat0(p, board, start, ignore)


def oracle_stays(g, ignore):
    """LAB ONLY: the stay predicates of the level's TRUE finished board (boards1164.npz)"""
    import prog1164 as P
    key = P._T.get('key')
    if not key:
        return []
    game, L = key
    Bd = P._boards()
    s = Bd.get(f'{game}:{int(L)}:start'); f = Bd.get(f'{game}:{int(L)}:finished'); hud = Bd.get(f'{game}:hud')
    if s is None or f is None:
        return []
    t, _ = stay_truth(s, f, hud if (hud is not None and hud.shape == s.shape) else None)
    return [p for p, ok in t.items() if ok]


# ================================================================================================ M1's predicted board
def predict_board(st, g, O, akey):
    """the board M1 predicts after action `akey` on board g (objects O): the remembered effects of the action's signature applied
    to the objects they touch (as Steer3.predict applies them to the features). None = an unknown signature."""
    sig = st.sig_of(akey, O)
    n = st.sig_n.get(sig, 0)
    if not n:
        return None
    from steer1162 import sbk
    B = np.array(g, copy=True); bg = int(st.bg); H, W = B.shape
    for (ocls, e), c in st.mem[sig].items():
        if c < 0.5 * n:
            continue
        tg = [o for o in O if (o['col'], sbk(o['n'])) == ocls]
        if akey[0] == 'C' and tg:
            tg = [min(tg, key=lambda o: abs(o['cy'] - akey[1]) + abs(o['cx'] - akey[2]))]
        for o in tg[:4]:
            y0, x0, h, w = int(o['y0']), int(o['x0']), int(o['h']), int(o['w'])
            box = np.zeros_like(B, dtype=bool); box[y0:y0 + h, x0:x0 + w] = True
            m = box & (np.asarray(g) == o['col'])
            if not m.any():
                continue
            if e[0] in ('move', 'to_click'):
                if e[0] == 'move':
                    dy, dx = int(e[1]), int(e[2])
                elif akey[0] == 'C':
                    dy, dx = int(round(akey[1] - o['cy'])), int(round(akey[2] - o['cx']))
                else:
                    continue
                ys, xs = np.nonzero(m)
                ny, nx = ys + dy, xs + dx
                ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
                B[m] = bg
                B[ny[ok], nx[ok]] = o['col']
            elif e[0] == 'vanish':
                B[m] = bg
            elif e[0] == 'recolour':
                B[m] = int(e[1])
    return B


def predicted_dh(st, g, O, akey):
    """M1's predicted change of h for `akey` on board g: None without a prediction (an unknown signature); 0 for a signature known
    to change nothing"""
    if not st.sg_pred or st.start_board is None:
        return None
    sig = st.sig_of(akey, O)
    if not st.sig_n.get(sig, 0):
        return None
    if st.sig_chg.get(sig, 0) == 0:
        return 0.0
    Bp = predict_board(st, g, O, akey)
    if Bp is None:
        return None
    h0 = G7.h_of(st.sg_pred, g, st.start_board, st.ignore())
    h1 = G7.h_of(st.sg_pred, Bp, st.start_board, st.ignore())
    if h0 is None or h1 is None:
        return None
    return float(h1 - h0)


def _neutral1168(st, sig):
    """goal1168's WHEN: the signature tried on this level and never moved h, or known to change nothing"""
    s_ = st.sg_dh.get(sig)
    if s_ and s_[1] >= 1 and abs(s_[0]) < 1e-12:
        return True
    return st.sig_n.get(sig, 0) >= 1 and st.sig_chg.get(sig, 0) == 0


# ================================================================================================ install
def install(Steer4):
    """idempotent; wraps goal1168's decide: with steer_goal_how / steer_goal_when the 1169 rule replaces goal1168's choice"""
    if getattr(Steer4, '_v1169_choose', False):
        return True
    Steer4._v1169_choose = True
    G7.satisfied = satisfied                                   # 'stay' is a new kind: the 1167 predicates are scored as before
    _init0 = Steer4.__init__; _dec0 = Steer4.decide; _stats0 = Steer4.stats; _start0 = Steer4.start_level

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)

        def gi(name):
            try:
                return int(float(_gene(params, name, 0) or 0))
            except Exception:
                return 0
        self.sg_when69 = gi('steer_goal_when') == 1
        self.sg_how69 = gi('steer_goal_how') == 1 or self.sg_when69         # the eager WHEN never runs with the observed HOW
        self.sg_stay = gi('steer_goal_stay') == 1
        if self.sg_stay and not getattr(self, 'sg_oracle', False):
            self.sg = StructGoal1169(mode=getattr(self, 'sg_mode', 'all'), stay=True)
        self.n_pchoose = 0

    def start_level(self, g):
        _start0(self, g)
        if getattr(self, 'sg_stay', False) and getattr(self, 'sg_oracle', False) and g is not None:
            try:
                extra = [p for p in oracle_stays(g, self.ignore()) if p not in (self.sg_pred or [])]
                if extra:
                    self.sg_pred = list(self.sg_pred or []) + extra
                    self.sg_h = G7.h_of(self.sg_pred, g, g, self.ignore())
                    self._tally('sg_stay', len(extra))
            except Exception as _e:
                self._v1169_error = f'stay: {type(_e).__name__}: {_e}'[:160]
        elif getattr(self, 'sg_stay', False) and self.sg_pred:
            self._tally('sg_stay', sum(1 for p in self.sg_pred if p[0] == 'stay'))

    def decide(self, brain, frame, a, d):
        rule69 = getattr(self, 'sg_choose', False) and (getattr(self, 'sg_how69', False) or getattr(self, 'sg_when69', False))
        if not rule69:
            return _dec0(self, brain, frame, a, d)
        self.sg_choose = False                                   # goal1168's observed-HOW choice is not played under the 1169 rule
        try:
            out = _dec0(self, brain, frame, a, d)
        finally:
            self.sg_choose = True
        if not getattr(self, 'sg_pred', None) or not getattr(self, '_from_brain', False) or not self._cur or self.prev is None:
            return out
        try:
            g, lv = self._cur
            k = self.key(g); nd = self.graph.get(k); O = self.objs.get(k) or []
            if nd is None:
                return out
            bkey = self.prev[1]
            if self.sg_when69:
                dh_b = predicted_dh(self, g, O, bkey)
                fire = dh_b is not None and dh_b >= 0
            else:
                fire = _neutral1168(self, self.sig_of(bkey, O))
            if not fire:
                return out
            self._tally('sg_pwhen')
            best = None; bm = 0.0; unknown = 0
            for a2, cl in nd['cands']:
                if a2 == bkey or self.dead(cl):
                    continue
                t = nd['tried'].get(a2)
                if t is not None and t[2]:
                    continue                                                    # it ended the game here before
                m2 = predicted_dh(self, g, O, a2)
                if m2 is None:
                    unknown += 1
                    continue
                if m2 < bm:
                    best, bm = (a2, cl), m2
            if unknown:
                self._tally('sg_unknown', unknown)
            if best is None:
                self._tally('sg_pnone')
                return out
            self.prev = (self.prev[0], best[0], best[1], self.prev[3], self.prev[4]); self.pred = None
            self._tally('sg_pchoose'); self._from_brain = False; self.n_pchoose += 1
            return self.to_action(best[0])
        except Exception as _e:
            self._v1169_error = f'predicted choose: {type(_e).__name__}: {_e}'[:160]
            return out

    def stats(self):
        s = _stats0(self)
        s.update(sg_how69=int(bool(getattr(self, 'sg_how69', False))), sg_when69=int(bool(getattr(self, 'sg_when69', False))),
                 sg_stay=int(bool(getattr(self, 'sg_stay', False))), sg_pchosen=getattr(self, 'n_pchoose', 0),
                 v1169_error=getattr(self, '_v1169_error', None))
        return s

    Steer4.__init__ = __init__; Steer4.decide = decide; Steer4.stats = stats; Steer4.start_level = start_level
    return True
