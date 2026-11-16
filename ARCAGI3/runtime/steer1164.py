"""steer1164.py -- NAS 1164: STEERING v5 = v4 + a goal that is ALLOWED TO ACT (nas_1163_report.md §20, change 3).

1163 measured why v4's goal map never decided anything: its term lived inside a plan that runs only when M1 predicts the
brain's proposal useless -- 0.2-1.6 times per 100 steps on the next-level lines (st_goal 0.00-0.09). v5 gives the goal
two direct routes, both genes, both counted:

  1. AUTHORITY (gene steer_goal_auth, a probability, default 0.5). At every step with a goal map, the brain's proposal is
     kept when its action signature has lowered h on this level, or when it is UNKNOWN (novelty first: an untried
     signature is a probe -- the research's 'wh'). Otherwise, with probability steer_goal_auth, it is replaced by the
     candidate whose signature has the best observed mean drop of h (st_v5goal).
  2. STALL NAVIGATION (gene steer_stall, steps, default 10). After steer_stall steps without any drop of h, the steering
     navigates through the known state graph to the frontier state with the LOWEST h (v4's bfs) (st_v5nav).

THE ORACLE ARM (gene steer_goal_oracle; LAB ONLY -- preflight check 87 refuses it in any fleet cell). The goal map is
replaced by the level's TRUE goal cells (prog1164: start != finished, HUD excluded, ending non-background) and h is
measured from the level's true start. It measures the CEILING of authority: FrameLab v3 found that the true goal is
worth +8 levels and 2-4 x the RHAE while its learned map was worth nothing; this arm asks the same question inside the
NAS's steering, so a learned goal is never trusted before its ceiling is known.

The brain still proposes every action; v5 changes which proposals survive when a goal is known. GENES: steer_v5 (or
MULTINXON_STEER_V5=1) | steer_goal_auth | steer_stall | steer_goal_oracle (lab) + every v1-v4 gene."""
from __future__ import annotations

import os

import numpy as np

import goalmap1163 as GM
import steer1163 as S4
from steer1161 import _gene

_ORACLE_REF = {'start': None}
_h_of0 = GM.h_of


def _h_of(board, start, goal):
    ref = _ORACLE_REF['start']
    return _h_of0(board, ref if (ref is not None and np.asarray(ref).shape == np.asarray(board).shape) else start, goal)


GM.h_of = _h_of                                   # steer1163 calls GM.h_of through the module: the oracle can swap the reference


class Steer5(S4.Steer4):
    def __init__(self, params, seed=0):
        super().__init__(params, seed=seed)
        g = lambda n, d: _gene(params, n, d)
        self.v5 = 1 if (int(g('steer_v5', 0) or 0) or str(os.environ.get('MULTINXON_STEER_V5', '0')) == '1') else 0
        if self.v5:
            self.v4 = 1; self.v3 = 1; self.v2 = 1
            self.goal_on = not int(g('steer_no_goalmap', 0) or 0)
        self.auth = max(0.0, min(1.0, float(g('steer_goal_auth', 0.5) or 0.0)))
        self.stall = max(0, int(float(g('steer_stall', 10) or 0)))
        self.oracle = int(g('steer_goal_oracle', 0) or 0) if self.v5 else 0
        self.since_drop = 0; self._ostart = None; self._olv = None

    # ------------------------------------------------------------------ the oracle goal (lab only)
    def _oracle_goal(self):
        try:
            import prog1164 as P
            T = P._T
            if T.get('D') is None or self.level is None or int(self.level) != int(T['L']) - 1:
                return None, None
            B = P._boards(); gk = T['key'][0]
            s = B.get(f"{gk}:{int(T['L'])}:start"); f = B.get(f"{gk}:{int(T['L'])}:finished")
            if s is None or f is None:
                return None, None
            goal = GM.arrivals(s, f, None) & T['D']
            return (goal if goal.any() else None), s
        except Exception:
            return None, None

    def start_level(self, g):
        super().start_level(g)
        self.since_drop = 0
        self._olv = None                                                   # the oracle is applied once the level is known

    def _apply_oracle(self):
        """the oracle goal for the level being played (called once self.level is set: after on_reset / observe_frame)"""
        if not self.oracle or self.level is None or self._olv == self.level or self.start_board is None:
            return
        self._olv = self.level
        goal, s = self._oracle_goal()
        self._ostart = s
        if goal is None or goal.shape != self.start_board.shape:
            return
        _ORACLE_REF['start'] = s
        try:
            cur = self._cur[0] if self._cur else self.start_board
            self.goal = goal; self.goal_n = int(goal.sum()); self.unit = max(1, -(-self.goal_n // 6))
            self.h = GM.h_of(cur, self.start_board, self.goal); self.hq = -(-self.h // self.unit) if self.h is not None else None
            self._tally('oracle')
        finally:
            _ORACLE_REF['start'] = None

    def on_reset(self, frame):
        super().on_reset(frame)
        self._apply_oracle()

    def observe_frame(self, frame):
        super().observe_frame(frame)
        self._apply_oracle()

    def start_level_goal_only(self):
        if self.oracle:
            return                                                         # the oracle's goal is not replaced by a learned one
        return super().start_level_goal_only()

    def _after_action(self, ak, g0, g1, ch, trans):
        if self.oracle and self._ostart is not None:
            _ORACLE_REF['start'] = self._ostart
        dh0 = self.dh_last
        try:
            super()._after_action(ak, g0, g1, ch, trans)
        finally:
            _ORACLE_REF['start'] = None
        if self.v5 and self.goal is not None:
            if self.dh_last < 0 and (self.dh_last != dh0 or True):
                self.since_drop = 0
            else:
                self.since_drop += 1

    # ------------------------------------------------------------------ the goal acts
    def _dh_mean(self, sig):
        st = self.dh_sig.get(sig)
        if not st or st[1] <= 0:
            return None
        return st[0] / float(st[1])

    def decide(self, brain, frame, a, d):
        out = super().decide(brain, frame, a, d)
        if not (self.v5 and self.goal is not None) or not self._cur or self.prev is None:
            return out
        try:
            g, lv = self._cur
            k = self.key(g); nd = self.graph.get(k); O = self.objs.get(k) or []
            if nd is None:
                return out
            akey = self.prev[1]
            # 2. stall navigation: no drop of h for steer_stall steps -> the frontier state with the lowest h
            if self.stall and self.since_drop >= self.stall:
                step = self.bfs(k)
                if step is not None and step != akey:
                    cls = next((cl for a2, cl in nd['cands'] if a2 == step), ('NAV',))
                    self.prev = (self.prev[0], step, cls, self.prev[3], self.prev[4]); self.pred = None
                    self.since_drop = 0; self._tally('v5nav'); self._from_brain = False
                    return self.to_action(step)
            # 1. authority: a known proposal that never lowered h yields to the best h-lowering candidate
            if not self._from_brain or self.auth <= 0 or self.rng.random() >= self.auth:
                return out
            bsig = self.sig_of(akey, O)
            m = self._dh_mean(bsig)
            if m is None or m < 0:
                return out                                                 # unknown (a probe: novelty first) or already lowering h
            best = None; bm = 0.0
            for a2, cl in nd['cands']:
                if a2 == akey or self.dead(cl):
                    continue
                t = nd['tried'].get(a2)
                if t is not None and t[2]:
                    continue                                               # it killed us here before
                m2 = self._dh_mean(self.sig_of(a2, O))
                if m2 is not None and m2 < bm:
                    best, bm = (a2, cl), m2
            if best is None:
                return out
            self.prev = (self.prev[0], best[0], best[1], self.prev[3], self.prev[4]); self.pred = None
            self._tally('v5goal'); self._from_brain = False
            return self.to_action(best[0])
        except Exception as _e:
            self._v5_error = f'v5: {type(_e).__name__}: {_e}'[:160]
            return out

    def stats(self):
        s = super().stats()
        s.update(v5=self.v5, auth=self.auth, stall=self.stall, oracle=self.oracle, v5_error=getattr(self, '_v5_error', None))
        return s
