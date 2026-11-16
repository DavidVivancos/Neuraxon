"""goal1168.py -- NAS 1168 change 5 (LAB FIRST): the structured goal CHOOSES among the alternatives; it does not veto.

WHY (nas_1167_report.md §16c, §23.5): the structured goal of goal1167 passes its precision gate (0.738) and h falls along 111 of
129 expert paths, but in play it was null at four starts over 800 plays, even with the TRUE goal: it could only re-rank the
plan's alternatives (a value term) and replace proposals that RAISED h (override) -- and a brain's proposals rarely raise h,
they do nothing for it. FrameLab 6 (outFramelab6/REPORT.md) measured the same question outside the NAS: a goal acting on
proposals at the structured goal's precision (0.74, recall 0.43) gave -1.0 clears per seed against random play, the learned
goal +1.3 (n.s.), only the TRUE finished board +11.7. So this is a LAB arm: it enters the fleet's bandit only if the learned arm
wins at one start in lab job goal2 (lab1168.py).

WHAT IT DOES (gene steer_goal_choose, inert at 0; needs the structured goal: steer_goal_struct 1, or the lab-only oracle
steer_goal_struct_oracle 1): when the brain's proposal is H-NEUTRAL -- its action signature has been tried on this level and
never moved h (the summed change of h is 0), or the effect model knows it changes nothing -- the steering looks at the
candidate actions of this board and takes the one whose signature has the best observed mean DROP of h (< 0). The brain still
proposes every step; the goal only chooses among alternatives when the proposal does nothing for h; an unknown proposal is a
probe and stands. TALLY: st_sg_choose (a choice made), st_sg_neutral (an h-neutral proposal seen), st_sg_nochoice (none of
the candidates has lowered h yet)."""
from __future__ import annotations

from steer1161 import _gene


def install(Steer4):
    """idempotent; wraps the decide that goal1167 installed (so it runs after the 1167 override, before Steer5's v5 logic)"""
    if getattr(Steer4, '_v1168_choose', False):
        return True
    Steer4._v1168_choose = True
    _init0 = Steer4.__init__; _dec0 = Steer4.decide; _stats0 = Steer4.stats

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        try:
            self.sg_choose = int(float(_gene(params, 'steer_goal_choose', 0) or 0)) == 1
        except Exception:
            self.sg_choose = False
        self.n_choose = 0

    def _mean_dh(self, sig):
        st = self.sg_dh.get(sig)
        if not st or st[1] <= 0:
            return None
        return st[0] / float(st[1])

    def _neutral(self, sig):
        st = self.sg_dh.get(sig)
        if st and st[1] >= 1 and abs(st[0]) < 1e-12:
            return True
        return self.sig_n.get(sig, 0) >= 1 and self.sig_chg.get(sig, 0) == 0

    def decide(self, brain, frame, a, d):
        out = _dec0(self, brain, frame, a, d)
        if not getattr(self, 'sg_choose', False) or not getattr(self, 'sg_pred', None):
            return out
        if not getattr(self, '_from_brain', False) or not self._cur or self.prev is None:
            return out
        try:
            g, lv = self._cur
            k = self.key(g); nd = self.graph.get(k); O = self.objs.get(k) or []
            if nd is None:
                return out
            bkey = self.prev[1]
            if not _neutral(self, self.sig_of(bkey, O)):
                return out
            self._tally('sg_neutral')
            best = None; bm = 0.0
            for a2, cl in nd['cands']:
                if a2 == bkey or self.dead(cl):
                    continue
                t = nd['tried'].get(a2)
                if t is not None and t[2]:
                    continue                                                    # it ended the game here before
                m2 = _mean_dh(self, self.sig_of(a2, O))
                if m2 is not None and m2 < bm:
                    best, bm = (a2, cl), m2
            if best is None:
                self._tally('sg_nochoice')
                return out
            self.prev = (self.prev[0], best[0], best[1], self.prev[3], self.prev[4]); self.pred = None
            self._tally('sg_choose'); self._from_brain = False; self.n_choose += 1
            return self.to_action(best[0])
        except Exception as _e:
            self._v1168_error = f'sg choose: {type(_e).__name__}: {_e}'[:160]
            return out

    def stats(self):
        s = _stats0(self)
        s.update(sg_choose=int(bool(getattr(self, 'sg_choose', False))), sg_chosen=getattr(self, 'n_choose', 0),
                 v1168_error=getattr(self, '_v1168_error', None))
        return s

    Steer4.__init__ = __init__; Steer4.decide = decide; Steer4.stats = stats
    return True
