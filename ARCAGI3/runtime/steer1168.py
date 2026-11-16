"""steer1168.py -- NAS 1168 change 7: the steering's VETO and PLAN become genes the line's bandit tunes per game.

WHY (nas_1167_report.md §11, §23.7): the veto acts per game in opposite directions -- on lp85 and m0r0 the clears are the plays
vetoed MORE (35 vs 11, 21 vs 15 per 100 steps), on lf52, ft09, su15, sp80 and s5i5 the plays vetoed LESS (28 vs 52, 16 vs 50,
7 vs 23, 3.7 vs 10.5, 26 vs 35). One fixed rule cannot be right on both; a line can learn its own from its children.

THE TWO GENES (read by Steer3.decide, the v3 / v4 / v5 decision; v1 / v2 are unchanged):
  steer_veto_th  how many no-change tries of the brain's proposal in this board before the steering vetoes it:
                   0 = never (the memory veto off: as steer_no_memory 1; deaths are not vetoed either),
                   1 = one (the 1160-1167 rule on boards whose outcome is consistent; DEFAULT),
                   2 = two (the hidden-state rule of 1160, now on every board: a proposal gets a second chance).
  steer_plan_w   the plan's weight against the brain: the plan replaces a proposal it predicts useless only when its best
                   alternative is predicted better by margin / plan_w (margin = 1 - steer_trust, as 1162):
                   0 = the plan never replaces a proposal, 0.5 = twice the margin, 1 = as 1162-1167 (DEFAULT), 2 = half.
At the defaults the decision is the 1162 code exactly (this module re-states Steer3.decide with the two numbers in place);
preflight 116 checks that a default brain plays move for move as before.
TALLY: st_veto (unchanged), st_veto_spared (a proposal the 1167 rule would have vetoed, kept by steer_veto_th 0 / 2),
st_plan_held (a plan the 1167 margin would have taken, held by steer_plan_w < 1), st_plan_extra (taken only because of
steer_plan_w > 1)."""
from __future__ import annotations

import os

import steer1162 as S3
from steer1161 import _gene

DEFAULT_VETO_TH = 1
DEFAULT_PLAN_W = 1.0


def _num(v, d):
    try:
        return float(v)
    except Exception:
        return d


def install():
    """idempotent: Steer3 (and so Steer4 / Steer5) read steer_veto_th and steer_plan_w"""
    St = S3.Steer3
    if getattr(St, '_v1168_steer', False):
        return True
    St._v1168_steer = True
    St._decide1167 = St.decide                     # kept: preflight 115 replays a play with it (the defaults must not change a move)
    _init0 = St.__init__

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        vt = _num(_gene(params, 'steer_veto_th', DEFAULT_VETO_TH), DEFAULT_VETO_TH)
        pin = os.environ.get('MULTINXON_PIN_VETO_TH', '')
        if pin not in ('', None):
            vt = _num(pin, vt)
        self.veto_th = int(max(0, min(4, round(vt))))
        pw = _num(_gene(params, 'steer_plan_w', DEFAULT_PLAN_W), DEFAULT_PLAN_W)
        pin = os.environ.get('MULTINXON_PIN_PLAN_W', '')
        if pin not in ('', None):
            pw = _num(pin, pw)
        self.plan_w = max(0.0, min(8.0, pw))

    def decide(self, brain, frame, a, d):
        if not self.v3:
            return S3.Steer2.decide(self, brain, frame, a, d)
        if not self.pre:
            self.observe_frame(frame)
        self.pre = False
        if not self._cur:
            return a, d
        g, lv = self._cur
        v = int(getattr(a, 'value', a) or 0) if a is not None else 0
        if v == 0:
            self.prev = None
            return a, d
        try:
            avail = [int(getattr(x, 'value', x)) for x in brain._masked_avail(frame)]
        except Exception:
            avail = [int(getattr(x, 'value', x)) for x in (getattr(frame, 'available_actions', None) or [])]
        k, nd = self.node(g, avail)
        O = self.objects_of(k, g)
        bkey, bcls = self.brain_key(g, a, d, nd)
        t = nd['tried'].get(bkey)
        vt = getattr(self, 'veto_th', DEFAULT_VETO_TH)
        if self.markov:
            wasted67 = t is not None and ((not t[1]) or t[2])
        else:
            rec = self.obs.get((k, bkey), [0, 0]); wasted67 = t is not None and (t[2] or (rec[0] >= 2 and rec[1] == 0))
        if vt == 1:
            wasted = wasted67
        elif vt <= 0:
            wasted = False
        else:                                                                   # a second chance: vt no-change tries (a death still vetoes)
            rec = self.obs.get((k, bkey), [0, 0])
            wasted = t is not None and (t[2] or (rec[0] >= vt and rec[1] == 0))
        bsig = self.sig_of(bkey, O)
        bv, bp = self.value(bsig, O, bkey) if (self.model_on and not wasted) else (-1.0, None)
        self.last_pv = bv
        choice = None; why = None; cp = None
        veto_on = self.cfg['memory'] and vt > 0
        if veto_on and (wasted or (t is None and self.deadly(bcls))):
            choice, cv = self.best3(nd, bkey, O); why = 'veto'
            cp = cv[1] if cv else None
            if choice is None and self.cfg['nav']:
                step = self.bfs(k)
                if step is not None:
                    choice = (step, next((cl for a2, cl in nd['cands'] if a2 == step), ('NAV',))); why = 'nav'
        elif self.dead(bcls):
            if self.cfg['memory'] and wasted67 and not wasted:
                self._tally('veto_spared')
            alt, cv = self.best3(nd, bkey, O)
            if alt is not None and not self.dead(alt[1]):
                choice, why, cp = alt, 'classveto', (cv[1] if cv else None)
        elif self.model_on and ((self.level and self.level >= 1) or self.classveto_l1) and \
                (bv < 0 or (bp is None and self.t > self.miss_until and len(self.sig_n) >= 4)):
            if self.cfg['memory'] and wasted67 and not wasted:
                self._tally('veto_spared')
            alt, cv = self.best3(nd, bkey, O)
            margin = 1.0 - min(0.9, max(0.0, self.trust))                       # trust lowers the bar for the plan
            pw = getattr(self, 'plan_w', DEFAULT_PLAN_W)
            if alt is not None and cv is not None and alt[0] != bkey:
                take67 = cv[0] > bv + margin
                take = (pw > 0) and (cv[0] > bv + (margin / pw if pw != 1.0 else margin))
                if take:
                    choice, why, cp = alt, 'plan', cv[1]
                    if not take67:
                        self._tally('plan_extra')
                elif take67:
                    self._tally('plan_held')
        elif self.cfg['memory'] and wasted67 and not wasted:
            self._tally('veto_spared')
        if choice is None:
            akey, cls, out = bkey, bcls, (a, d)
            self.pred = (bsig, bp) if bp is not None else None
            if t is None and not self.dead(bcls):
                self._tally('agree')
                if self.teach:
                    brain._steer_teach_pending = float(getattr(brain, '_steer_teach_pending', 0.0) or 0.0) + 0.1
        else:
            akey, cls = choice
            out = self.to_action(akey)
            self.pred = (self.sig_of(akey, O), cp) if cp is not None else None
            self._tally(why)
            if cls and cls[0] in ('D', 'L'):
                self._tally('dest' if cls[0] == 'D' else 'lattice')
            if self.teach:
                brain._steer_teach_pending = float(getattr(brain, '_steer_teach_pending', 0.0) or 0.0) - 0.05
            if why in ('veto', 'nav'):
                self.n_veto += 1
        if self.t <= self.miss_until:
            self._tally('probe')
        self.prev_cls_kind = cls[0] if cls else None
        self.prev = (k, akey, cls, g, lv); self.n_steps += 1; self._tally('steps')
        return out

    _stats0 = St.stats

    def stats(self):
        s = _stats0(self)
        s.update(veto_th=getattr(self, 'veto_th', DEFAULT_VETO_TH), plan_w=getattr(self, 'plan_w', DEFAULT_PLAN_W))
        return s

    St.__init__ = __init__; St.decide = decide; St.stats = stats
    return True
