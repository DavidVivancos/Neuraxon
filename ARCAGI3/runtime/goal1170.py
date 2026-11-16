"""goal1170.py -- NAS 1170: THE FRAMELAB GOAL IN THE FLEET (gene steer_goal_fl, a bandit arm; nas_1169_report.md §19.7 + the user's
"be sure the FrameLab learnings are implemented in the NAS").

WHAT FRAMELAB MEASURED (FrameLab 6-11, the random proposer + v5 harness, 183 levels x 3 seeds; claude/FrameLab_2026-10-08.md):
  FL7  the goal that ONLY CHOOSES among the candidates (it never plans, never vetoes) is where a goal pays: true goal +13.3 levels
       per seed against no goal;
  FL8  h counted only on the cells / predicates the goal names (the NAS form) keeps most of it (15.7 vs 20.0); the cells that
       must STAY are about a third of the true goal's value; precision matters (p 0.74 cost 1.6 per seed at recall 0.43);
  FL9  the RULE: WHEN = the model does not predict that the brain's proposal lowers h, HOW = the candidate with the best
       PREDICTED drop of h (20.0) -- the OBSERVED mean drop per signature as the HOW is harmful (3.3, below no goal);
  FL10 the model behind the goal decides most of it: NAS form with the learned model 9.3, with a perfect model 24.3;
  FL11 a model that predicts only for SEEN signatures (the key, or the clicked colour + size) but predicts them EXACTLY scores
       17 against perfect 19 and learned 11; an unpredictable proposal stands (known_probe, 17): accuracy where the model
       speaks matters, coverage less; recall of the goal's predicates (0.43 -> 0.8) is what limits the goal itself.
WHAT 1169 MEASURED IN THE NAS: lab goal3 (goal1169's rule with M1's effect grammar) was null even with the oracle structure; M1 had
no prediction for 10-67 candidates per step and chose 0.1-1.5 times per 100 steps.

WHAT THIS MODULE PUTS IN THE PLAYED BRAIN'S STEERING (all generic; nothing per game; the brain still proposes every action):
  steer_goal_fl 1      turns on, together: the learned structured goal (goal1167, learned only from the levels this play has
                       seen complete -- the verified prefix's levels and its own clears; never an oracle), the FrameLab rule
                       (goal1169's predicted WHEN / HOW), the STAY predicates (goal1169), and switches OFF the observed-mean
                       value term in the plan (steer_goal_w -> 0: FL9's harmful half) and the goal's veto (override 0: FL7);
  THE SEEN-SIGNATURE MODEL (FL11, replaces M1's effect grammar as the goal's model; M1 stays the fallback):
     1. EXACT: a (board, action) the play has executed before predicts exactly the board it produced (deterministic games);
     2. REPLAY: an action whose SIGNATURE (Steer3.sig_of: the key, or the clicked object's colour, size class and on / beside)
        was executed before replays that execution's cell changes -- the changes near the clicked object relative to it
        (moved with the new target), the others where they were -- when >= 80% of the cells it would change still hold the
        colour they had then (a replay that does not fit the board predicts nothing);
     3. M1 (goal1169.predict_board): the remembered effects of the signature on the objects they touch;
     a signature known to change nothing predicts dh 0; anything else is UNKNOWN -- an unknown proposal is a probe and stands
     (known_probe), an unknown candidate is not chosen;
  steer_goal_support 1 | 2  the PRECISION gene: a predicate enters h only if it held in >= support earlier levels (FL8).
RECALL is logged every step the rule runs: st_sg_cand (candidates scored), st_sg_known (with a prediction), st_sg_pexact /
st_sg_preplay / st_sg_pm1 / st_sg_psame (which part of the model predicted), st_sg_budget (scored past the per-step budget).
recall = st_sg_known / st_sg_cand. goal1169's st_sg_pwhen / st_sg_pchoose / st_sg_pnone / st_sg_unknown / st_sg_stay stay.
COST: at most FL_MAX_H predicted boards are scored per step (h on each); exact / same predictions are free.

In the fleet the gene is an ARM of the paired bandit (evo1170: steer_goal_fl 0 / 1, on the lines that play a level >= 2) and the
gate makes a goal copy of the top champions at generation 0 (nas1170: paired on the anchors with their originals), so the fleet
measures the FrameLab goal on its own brains, seed by seed. A brain without the gene plays exactly as in 1169."""
from __future__ import annotations

import collections

import numpy as np

import goal1167 as G7
import goal1169 as G9
from steer1161 import _gene

FL_MAX_H = 48            # predicted boards scored per step (h on each); the rest of the step's candidates count as unknown
EXACT_MAX = 6000         # (board, action) -> its observed change, FIFO
REPLAY_FIT = 0.8         # the share of a replay's cells that must still hold their old colour
LOCAL_R = 6              # cells within this many of the clicked object's box move with the click target
MAX_CHANGE = 0.5         # a change of more than half the board (a level transition, a reset) is not a replay


# ================================================================================================ precision (FL8)
class StructGoal1170(G9.StructGoal1169):
    """goal1169's structured goal with stays + a SUPPORT floor: a predicate is predicted only if it held in >= support levels"""

    def __init__(self, mode='all', min_levels=1, stay=True, support=1):
        super().__init__(mode=mode, min_levels=min_levels, stay=stay)
        self.support = max(1, int(support or 1))

    def predict(self, S, ignore=None):
        return [x for x in super().predict(S, ignore) if int(x[1]) >= self.support]


# ================================================================================================ the seen-signature model (FL11)
def _anchor(akey, O):
    """the click's anchor: the object Steer3.sig_of picks ((y0, x0, h, w)) or, on floor, the click itself; None for a key"""
    if akey[0] != 'C':
        return None
    y, x = akey[1], akey[2]
    best = None; bd = 1e9
    for o in O or []:
        d = max(abs(o['cy'] - y), abs(o['cx'] - x))
        if d < bd:
            bd, best = d, o
    if best is None or bd > 12:
        return (int(y), int(x), 1, 1)
    return (int(best['y0']), int(best['x0']), int(best['h']), int(best['w']))


def record(st, ak, g0, g1):
    """after an executed action: the exact memo and the signature's replay"""
    if ak is None or g0 is None or g1 is None:
        return
    g0 = np.asarray(g0); g1 = np.asarray(g1)
    if g0.shape != g1.shape:
        return
    ch = np.flatnonzero(g0 != g1)
    if len(ch) > MAX_CHANGE * g0.size:
        return
    try:
        k0 = st.key(g0)
    except Exception:
        k0 = None
    if k0 is not None:
        st.fl_exact[(k0, ak)] = (ch.astype(np.int32), g1.ravel()[ch].copy()) if len(ch) else 'same'
        st.fl_exact.move_to_end((k0, ak))
        while len(st.fl_exact) > EXACT_MAX:
            st.fl_exact.popitem(last=False)
    if not len(ch):
        return
    O0 = st.objs.get(k0) if k0 is not None else None
    if O0 is None:
        from steer1162 import components
        O0 = components(g0, st.bg, st.ignore())
    sig = st.sig_of(ak, O0)
    H, W = g0.shape
    ys, xs = np.divmod(ch, W)
    old = g0.ravel()[ch]; new = g1.ravel()[ch]
    an = _anchor(ak, O0)
    if an is None:
        loc = np.zeros(len(ch), dtype=bool)
        ay = ax = 0
    else:
        ay, ax, h, w = an
        loc = (ys >= ay - LOCAL_R) & (ys < ay + h + LOCAL_R) & (xs >= ax - LOCAL_R) & (xs < ax + w + LOCAL_R)
    st.fl_sig[sig] = {'rel': (ys[loc] - ay, xs[loc] - ax, old[loc], new[loc]), 'abs': (ys[~loc], xs[~loc], old[~loc], new[~loc])}


def _replay(st, g, O, akey, rp):
    B = np.array(g, copy=True); H, W = B.shape
    an = _anchor(akey, O)
    ys = []; xs = []; olds = []; news = []
    dy, dx, o, n = rp['rel']
    if len(dy):
        if an is None:
            return None
        ys.append(dy + an[0]); xs.append(dx + an[1]); olds.append(o); news.append(n)
    y2, x2, o2, n2 = rp['abs']
    if len(y2):
        ys.append(y2); xs.append(x2); olds.append(o2); news.append(n2)
    if not ys:
        return None
    Y = np.concatenate(ys); X = np.concatenate(xs); Od = np.concatenate(olds); Nw = np.concatenate(news)
    ok = (Y >= 0) & (Y < H) & (X >= 0) & (X < W)
    if not ok.any():
        return None
    Y, X, Od, Nw = Y[ok], X[ok], Od[ok], Nw[ok]
    fit = B[Y, X] == Od
    if fit.mean() < REPLAY_FIT:
        return None
    B[Y[fit], X[fit]] = Nw[fit]
    return B


def fl_predict(st, g, O, akey, k=None):
    """(predicted board or None, source): exact, same, replay, m1 -- or (None, None) for an unknown action; k = st.key(g)"""
    if k is None:
        try:
            k = st.key(g)
        except Exception:
            k = None
    d = st.fl_exact.get((k, akey)) if k is not None else None
    if d is not None:
        if isinstance(d, str):
            return g, 'same'
        B = np.array(g, copy=True); idx, vals = d
        B.ravel()[idx] = vals
        return B, 'exact'
    sig = st.sig_of(akey, O)
    if st.sig_n.get(sig, 0) and st.sig_chg.get(sig, 0) == 0:
        return g, 'same'
    rp = st.fl_sig.get(sig)
    if rp is not None:
        B = _replay(st, g, O, akey, rp)
        if B is not None:
            return B, 'replay'
    B = G9.predict_board(st, g, O, akey) if st.sig_n.get(sig, 0) else None
    if B is not None:
        return B, 'm1'
    return None, None


_pdh0 = G9.predicted_dh


def predicted_dh(st, g, O, akey):
    """goal1169's predicted change of h, with the seen-signature model where steer_goal_fl is on (cached per step)"""
    if not getattr(st, 'fl_on', False):
        return _pdh0(st, g, O, akey)
    if not st.sg_pred or st.start_board is None:
        return None
    if st._fl_t != st._fl_step or st._fl_g is not g:
        st._fl_t = st._fl_step; st._fl_g = g; st._fl_cache = {}; st._fl_h0 = None; st._fl_nh = 0
        try:
            st._fl_k = st.key(g)
        except Exception:
            st._fl_k = None
    if akey in st._fl_cache:
        return st._fl_cache[akey]
    st._tally('sg_cand')
    B, src = fl_predict(st, g, O, akey, st._fl_k)
    res = None
    if B is not None:
        if src == 'same':
            res = 0.0
        elif st._fl_nh >= FL_MAX_H:
            st._tally('sg_budget')
        else:
            ign = st.ignore()
            if st._fl_h0 is None:
                st._fl_h0 = G7.h_of(st.sg_pred, g, st.start_board, ign)
            h1 = G7.h_of(st.sg_pred, B, st.start_board, ign); st._fl_nh += 1
            if st._fl_h0 is not None and h1 is not None:
                res = float(h1 - st._fl_h0)
    if res is not None:
        st._tally('sg_known'); st._tally('sg_p' + src)
    st._fl_cache[akey] = res
    return res


# ================================================================================================ install
def install(Steer4):
    """idempotent; after goal1167 / goal1168 / goal1169: steer_goal_fl turns the FrameLab goal on (inert at 0)"""
    if getattr(Steer4, '_v1170_fl', False):
        return True
    Steer4._v1170_fl = True
    G9.predicted_dh = predicted_dh                      # goal1169's rule reads its module's predicted_dh at call time
    _init0 = Steer4.__init__; _after0 = Steer4._after_action; _stats0 = Steer4.stats

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)

        def gi(name, d=0):
            try:
                return int(float(_gene(params, name, d) or d))
            except Exception:
                return d
        import os
        self.fl_on = gi('steer_goal_fl') == 1 or str(os.environ.get('MULTINXON_STEER_GOAL_FL', '0')) == '1'
        self.fl_support = max(1, gi('steer_goal_support', 1))
        self.fl_exact = collections.OrderedDict(); self.fl_sig = {}
        self._fl_step = 0; self._fl_t = None; self._fl_g = None; self._fl_k = None; self._fl_cache = {}; self._fl_h0 = None; self._fl_nh = 0
        if self.fl_on:
            self.sg_on = True; self.sg_choose = True; self.sg_how69 = True; self.sg_when69 = True; self.sg_stay = True
            self.sg_w = 0.0; self.sg_override = False                      # FL9: no observed-mean term; FL7: the goal only chooses
            if not getattr(self, 'sg_oracle', False):
                self.sg = StructGoal1170(mode=getattr(self, 'sg_mode', 'all'), stay=True, support=self.fl_support)

    def _after_action(self, ak, g0, g1, ch, trans):
        _after0(self, ak, g0, g1, ch, trans)
        if not getattr(self, 'fl_on', False):
            return
        self._fl_step += 1
        try:
            record(self, ak, g0, g1)
        except Exception as _e:
            self._v1170_error = f'record: {type(_e).__name__}: {_e}'[:160]

    def stats(self):
        s = _stats0(self)
        s.update(fl_on=int(bool(getattr(self, 'fl_on', False))), fl_support=getattr(self, 'fl_support', 1),
                 fl_exact=len(getattr(self, 'fl_exact', {}) or {}), fl_sigs=len(getattr(self, 'fl_sig', {}) or {}),
                 v1170_error=getattr(self, '_v1170_error', None))
        return s

    Steer4.__init__ = __init__; Steer4._after_action = _after_action; Steer4.stats = stats
    return True
