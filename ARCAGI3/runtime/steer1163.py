"""steer1163.py -- NAS 1163: STEERING v4 = steering v3 (the research's M1 + novelty + predict-act-compare) + a GOAL learned
from the boards the play has seen + the research's steering hygiene + node clicks. (nas_1162_report.md §19, changes 3-5.)

THE BRAIN STILL PROPOSES EVERY ACTION. v4 changes what the steering knows when it corrects a proposal, and what the brain
can sense; it never plays a scripted solution and never knows the game.

  1. THE GOAL MAP (change 3; goalmap1163.py). When a level completes -- in play, or inside a next-level line's expert
     prefix (game_loop hands the prefix's completed levels to the steering) -- its start board and its finished board
     teach a small per-game cell model which cells must END non-background ("arrivals"). On the next level the model
     predicts that level's goal map from its start board. The guide h = predicted goal cells not yet arrived, ROUNDED to
     6 buckets so novelty stays meaningful (the research's 'wh': novelty first, then h).
       * the plan's value of a candidate = v3's value + steer_goal_w x (the observed mean drop of h for that action
         signature on this level, clipped);
       * navigation, when a state is exhausted, goes to the frontier state with the LOWEST h (ties: the nearest);
       * gene steer_goal_override (0/1): the plan may also replace a proposal whose signature RAISES h;
       * the brain learns it: sense channels 13-16 (goal active, h, the last action's change of h, a clock/wake flag) when
         steer_sense = 3; teaching (steer_teach) rewards the brain's OWN proposals that lowered h, after the outcome.
  2. CLOCKS (change 4). A thin strip where ONE cell at a time flips with the same colour change, at positions that
     progress along a row or a column (wa30's step bar on row 63, re86's step counter on row 63, sb26's attempts counter on
     row 53 -- measured on the real games), is a counter: it is masked from the state keys, M1's objects, the novelty
     atoms, M1's hit test and the goal's arrivals. (The 1160 HUD mask -- cells flipping on > 35% of transitions after 16 of
     them, border lines on > 20% -- stays.)
  3. WAKE (change 4). A level's first action after its start (or after a reset) that changes nothing is NOT learned as a
     no-op (sc25 L1 ignores its first action): no graph entry, no class statistics, no M1 signature count.
  4. RE-PROBING (change 4). A signature or click class known to change nothing becomes worth one more try once the board
     has shown 3 new feature atoms since its last no-op (the research's re-probing at novel nodes); on a new level every
     known no-op is re-testable (the expert needs ACTION1 / 2 later even when they do nothing at the start).
  5. NODE CLICKS (change 5). The click candidates of a state always include the objects that APPEARED or MOVED since the
     level began (dc22's mid-level button, m0r0 / ka59 "click the object where it is now"), even past steer_max_clicks.
GENES: steer_v4 (or MULTINXON_STEER_V4=1) | steer_goal_w (0.4) | steer_goal_override (0) | steer_no_goalmap |
       steer_no_clock | steer_no_wake | steer_no_reprobe | steer_no_nodeclick | steer_sense 3 = 16 channels |
       + every v1 / v2 / v3 gene. TALLIES (per try): st_goalfit, st_goal, st_hdown, st_hup, st_clock, st_wake, st_reprobe,
       st_node, st_navh."""
from __future__ import annotations

import collections
import os

import numpy as np

import goalmap1163 as GM
from agent1160 import first_frame, objects, settled, size_bucket
from steer1161 import _gene
from steer1162 import Steer3, components


class Clocks:
    """counter strips: single-cell flips with one colour change, progressing along a row (or a column)"""

    def __init__(self):
        self.mask = None; self.ev = collections.defaultdict(list); self.found = []

    def update(self, g0, g1):
        """returns True when the mask grew (state keys must be rebuilt)"""
        if g0 is None or g1 is None or g0.shape != g1.shape:
            return False
        d = g0 != g1
        n = int(d.sum())
        if n == 0 or n > 400:
            return False
        if self.mask is None or self.mask.shape != d.shape:
            self.mask = np.zeros(d.shape, bool)
        ys, xs = np.nonzero(d & ~self.mask)
        if len(ys) == 0 or len(ys) > 200:
            return False
        H, W = d.shape; grew = False
        pts = set(zip(ys.tolist(), xs.tolist()))
        for y, x in pts:
            if any((y + dy, x + dx) in pts for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx):
                continue                                             # part of a larger blob: not a counter tick
            tr = (int(g0[y, x]), int(g1[y, x]))
            for axis, line, pos in (('r', y, x), ('c', x, y)):
                ev = self.ev[(axis, line, tr)]
                ev.append(pos)
                del ev[:-6]
                if len(ev) >= 3:
                    steps = [b - a for a, b in zip(ev[-3:], ev[-3:][1:])]
                    if all(1 <= s <= 2 for s in steps) or all(-2 <= s <= -1 for s in steps):
                        grew |= self._mask_strip(g1, axis, line, tr)
        return grew

    def _mask_strip(self, g, axis, line, tr):
        row = g[line, :] if axis == 'r' else g[:, line]
        ev = self.ev[(axis, line, tr)]
        cols = set(tr)
        lo = min(ev); hi = max(ev)
        while lo - 1 >= 0 and int(row[lo - 1]) in cols:
            lo -= 1
        while hi + 1 < len(row) and int(row[hi + 1]) in cols:
            hi += 1
        before = int(self.mask.sum())
        if axis == 'r':
            self.mask[line, lo:hi + 1] = True
        else:
            self.mask[lo:hi + 1, line] = True
        if int(self.mask.sum()) > before:
            self.found.append((axis, line, lo, hi, tr))
            return True
        return False


class Steer4(Steer3):
    def __init__(self, params, seed=0):
        super().__init__(params, seed=seed)
        g = lambda n, d: _gene(params, n, d)
        self.v4 = 1 if (int(g('steer_v4', 0) or 0) or str(os.environ.get('MULTINXON_STEER_V4', '0')) == '1') else 0
        if self.v4:
            self.v3 = 1; self.v2 = 1                                             # v4 includes v3 and v2
        on = lambda n: bool(self.v4) and not int(g(n, 0) or 0)
        self.goal_on = on('steer_no_goalmap'); self.clock_on = on('steer_no_clock'); self.wake_on = on('steer_no_wake')
        self.reprobe_on = on('steer_no_reprobe'); self.node_on = on('steer_no_nodeclick')
        self.goal_w = float(g('steer_goal_w', 0.4) or 0.4)
        self.goal_override = int(g('steer_goal_override', 0) or 0)
        ss = int(g('steer_sense', 0) or 0)
        self.sense_w = 16 if ss >= 3 else (12 if ss >= 2 else 8)
        self.gm = GM.GoalModel(); self.goal = None; self.goal_n = 0; self.h = None; self.hq = None; self.unit = 1
        self.dh_last = 0; self.dh_sig = collections.defaultdict(lambda: [0.0, 0]); self.h_state = {}
        self.clocks = Clocks(); self.noop_at = {}; self.cls_try_at = {}
        self._wake = True; self._wake_skip = False; self._brain = None; self._from_brain = False
        self._prior_done = False; self.start_objs = set(); self._kb = {}
        try:
            self._k0 = int(os.environ.get('MULTINXON_START_STEP', '0') or 0)     # a curriculum start k moves INTO its level
        except Exception:
            self._k0 = 0
        self._first_lv = None; self._started = False

    # ------------------------------------------------------------------ masks: the HUD (1160) + clocks (1163)
    def ignore(self):
        m = self.hud.mask
        if self.clock_on and self.clocks.mask is not None and self.clocks.mask.shape == m.shape:
            return m | self.clocks.mask
        return m

    def node(self, g, avail):
        k, nd = super().node(g, avail)
        if self.clock_on and k not in self._kb and len(self._kb) < 3000:
            self._kb[k] = g
        return k, nd

    def _rekey(self):
        """the clock mask grew: every known state gets its new key (the counter no longer tells states apart) and the
        tried memory is KEPT -- a transition that only moved the counter becomes a no-op self-loop"""
        mp = {k: self.key(b) for k, b in self._kb.items()}
        G2 = {}
        for k, nd in self.graph.items():
            nk = mp.get(k, k)
            n2 = G2.get(nk)
            if n2 is None:
                G2[nk] = n2 = {'cands': nd['cands'], 'tried': {}}
            for a, (v, chg, died) in nd['tried'].items():
                v2 = mp.get(v, v) if v is not None else None
                n2['tried'].setdefault(a, (v2, bool(chg) and v2 != nk, died))
        self.graph = G2
        self._kb = {mp.get(k, k): b for k, b in self._kb.items()}
        self.obs = {(mp.get(k, k), a): r for (k, a), r in self.obs.items()}
        self.h_state = {mp.get(k, k): h for k, h in self.h_state.items()}
        self.objs = {}
        if self.prev is not None:
            self.prev = (mp.get(self.prev[0], self.prev[0]),) + tuple(self.prev[1:])

    # ------------------------------------------------------------------ level lifecycle
    def _wake_ok(self, lv):
        """the wake rule applies when the level REALLY starts: in play after a level-up, at a play's real level start,
        after a reset of such a level -- never at a curriculum start k > 0 moves into the level (the prefix already
        made the level's first moves; tn36 L2 at 2 moves left lost a clear to a deferred no-op veto)"""
        return not (self._k0 > 0 and (self._first_lv is None or lv == self._first_lv))

    def start_level(self, g):
        super().start_level(g)
        first = not self._started; self._started = True
        self._wake = (not first) or self._k0 <= 0                              # a level-up in play always really starts a level
        self.dh_last = 0; self.h_state = {}; self._kb = {}
        self.dh_sig = collections.defaultdict(lambda: [0.0, 0])
        self.noop_at = {s: -10 for s in self.noop_at}                           # known no-ops: re-testable on a new level
        self.cls_try_at = {}
        try:
            self.start_objs = {(o[0], o[2], o[3]) for o in objects(g, self.ignore(), self.bg)} if self.node_on else set()
        except Exception:
            self.start_objs = set()
        self.goal = None; self.h = None; self.hq = None
        if self.goal_on and self.gm.levels:
            try:
                goal, _ = self.gm.predict(g, ignore=self.ignore())
                if goal is not None and goal.any():
                    self.goal = goal; self.goal_n = int(goal.sum()); self.unit = max(1, -(-self.goal_n // 6))
                    self.h = GM.h_of(g, self.start_board, self.goal); self.hq = -(-self.h // self.unit)
                    self._tally('goalfit')
            except Exception as _e:
                self._v4_error = f'goal: {type(_e).__name__}: {_e}'[:160]

    def on_reset(self, frame):
        super().on_reset(frame)
        if self._first_lv is None:
            self._first_lv = self.level
        self._wake = self._wake_ok(self.level)

    def prior_levels(self, pairs, hud=None):
        """the levels an expert prefix completed before this play's level (game_loop hands them over once per play):
        each (start board, finished board) teaches the goal map, exactly as a level completed in play would"""
        if not self.goal_on or self._prior_done:
            return 0
        self._prior_done = True; n = 0
        for start, fin in pairs or []:
            try:
                if start is not None and fin is not None and np.asarray(start).shape == np.asarray(fin).shape:
                    m = hud if (hud is not None and np.asarray(hud).shape == np.asarray(start).shape) else None
                    n += int(self.gm.add(start, GM.arrivals(start, fin, m)))
            except Exception:
                pass
        if n and self.start_board is not None:
            self.start_level_goal_only()
        return n

    def start_level_goal_only(self):
        """predict the current level's goal map without resetting the level's state (prior levels arrived late)"""
        try:
            goal, _ = self.gm.predict(self.start_board, ignore=self.ignore())
            if goal is not None and goal.any():
                self.goal = goal; self.goal_n = int(goal.sum()); self.unit = max(1, -(-self.goal_n // 6))
                cur = self._cur[0] if self._cur else self.start_board
                self.h = GM.h_of(cur, self.start_board, self.goal); self.hq = -(-self.h // self.unit)
                self._tally('goalfit')
        except Exception as _e:
            self._v4_error = f'goal: {type(_e).__name__}: {_e}'[:160]

    # ------------------------------------------------------------------ learning
    def observe(self, k0, akey, cls, g0, g1, died, up, frames=None):
        if self.wake_on and self._wake and not died and not up and g0 is not None and g1 is not None and g0.shape == g1.shape \
                and not bool(((g0 != g1) & ~self.ignore()).any()):
            self._wake = False; self._wake_skip = True; self._tally('wake')
            return False                                                       # the level's first no-op is not a no-op
        self._wake = False; self._wake_skip = False
        if cls and cls[0] in ('C', 'D', 'L', 'R'):
            self.cls_try_at[cls] = len(self.seen)
        return super().observe(k0, akey, cls, g0, g1, died, up, frames=frames)

    def observe_frame(self, frame):
        if self.goal_on and self.prev is not None:
            lv = int(getattr(frame, 'levels_completed', 0) or 0)
            k0, ak, cl, g0, lv0 = self.prev
            if lv > lv0 and self.start_board is not None and g0 is not None:
                fin = GM.finished_board(g0, getattr(frame, 'frame', None))
                if fin is not None and fin.shape == self.start_board.shape:
                    try:
                        if self.gm.add(self.start_board, GM.arrivals(self.start_board, fin, self.ignore())):
                            self._tally('goallearn')
                    except Exception:
                        pass
        super().observe_frame(frame)

    def _after_action(self, ak, g0, g1, ch, trans):
        if self.clock_on and g0 is not None and g1 is not None:
            try:
                if self.clocks.update(g0, g1):
                    self._rekey(); self._tally('clock')                        # state keys exclude the counter now
            except Exception:
                pass
        if self._wake_skip:
            self._wake_skip = False
            from steer1161 import Steer2
            Steer2._after_action(self, ak, g0, g1, ch, trans)                 # the mover bookkeeping only; M1 learns nothing
            self.pred = None
            return
        sig = None; was_noop = False
        if self.v4 and ak is not None and g0 is not None and g1 is not None and g0.shape == g1.shape:
            try:
                k0 = self.key(g0); O0 = self.objs.get(k0)
                if O0 is None:
                    O0 = components(g0, self.bg, self.ignore()); self.objs[k0] = O0
                sig = self.sig_of(ak, O0)
                was_noop = self.sig_n.get(sig, 0) >= 3 and self.sig_chg.get(sig, 0) == 0
            except Exception:
                sig = None
        super()._after_action(ak, g0, g1, ch, trans)
        if not self.v4 or sig is None:
            return
        try:
            if was_noop and self.reprobe_on:
                self._tally('reprobe')
                if ch or trans:
                    self._tally('reprobe_hit')
            if self.reprobe_on and not (ch or trans):
                self.noop_at[sig] = len(self.seen)
            if self.goal is not None and self.start_board is not None:
                h1 = GM.h_of(g1, self.start_board, self.goal)
                if h1 is not None:
                    hq1 = -(-h1 // self.unit); dh = (hq1 - self.hq) if self.hq is not None else 0
                    self.h, self.hq = h1, hq1
                    self.dh_last = int(np.sign(dh))
                    st = self.dh_sig[sig]; st[0] += dh; st[1] += 1
                    self.h_state[self.key(g1)] = hq1
                    if dh < 0:
                        self._tally('hdown')
                    elif dh > 0:
                        self._tally('hup')
                    if self.teach and self._from_brain and self._brain is not None and dh != 0:
                        b = self._brain
                        b._steer_teach_pending = float(getattr(b, '_steer_teach_pending', 0.0) or 0.0) + (0.2 if dh < 0 else -0.1)
        except Exception as _e:
            self._v4_error = f'after: {type(_e).__name__}: {_e}'[:160]

    # ------------------------------------------------------------------ candidates: + node clicks (appeared / moved objects)
    def candidates(self, g, avail):
        out = super().candidates(g, avail)
        if not (self.v4 and self.node_on) or 6 not in avail:
            return out
        try:
            seen = {a for a, _ in out}; front = []
            for col, s, y, x in objects(g, self.ignore(), self.bg):
                if (col, y, x) in self.start_objs or ('C', y, x) in seen:
                    continue
                if self.start_board is not None and self.start_board.shape == g.shape and int(self.start_board[y, x]) == col:
                    continue                                                   # it was there at the start (another cell of it)
                front.append((('C', y, x), ('C', col, size_bucket(s)))); seen.add(('C', y, x))
                if len(front) >= 16:
                    break
            if front:
                self._tally('node', len(front))
            return front + out
        except Exception:
            return out

    # ------------------------------------------------------------------ the class veto and the plan's value, re-probing
    def dead(self, cls):
        if self.reprobe_on and cls in self.cls_try_at and len(self.seen) >= self.cls_try_at[cls] + 3:
            return False                                                       # the board changed since: try it again
        return super().dead(cls)

    def value(self, sig, O, akey):
        if self.reprobe_on and self.model_on and self.sig_n.get(sig, 0) >= 3 and self.sig_chg.get(sig, 0) == 0:
            at = self.noop_at.get(sig)
            if at is not None and len(self.seen) >= at + 3:
                self._reprobe_pending = sig
                return 0.45, None                                              # worth one more try (below an unknown's 0.8)
        v, p = super().value(sig, O, akey)
        if self.goal is not None and self.goal_w > 0:
            st = self.dh_sig.get(sig)
            if st and st[1] > 0:
                v += self.goal_w * max(-1.0, min(1.0, -st[0] / float(st[1])))
        return v, p

    def raises_h(self, sig):
        st = self.dh_sig.get(sig)
        return bool(st and st[1] >= 2 and st[0] / float(st[1]) > 0.5)

    # ------------------------------------------------------------------ navigation to the frontier with the lowest h
    def bfs(self, start):
        if self.goal is None or not self.h_state:
            return super().bfs(start)
        prev = {start: None}; q = collections.deque([(start, 0)]); best = None
        while q and len(prev) < int(getattr(self, '_bfs_max', 5000) or 5000):          # v1.177 change 9: 20,000 on a 1177 brain
            u, dep = q.popleft()
            nd = self.graph.get(u)
            if nd is None:
                continue
            if u != start and self.frontier(nd):
                cand = (self.h_state.get(u, 10 ** 6), dep, u)
                if best is None or cand < best:
                    best = cand
                if dep > (best[1] + 6):
                    break
            for a, (v, chg, died) in nd['tried'].items():
                if v is None or died or not chg or v in prev:
                    continue
                prev[v] = (u, a); q.append((v, dep + 1))
        if best is None:
            return None
        path = []; v = best[2]
        while prev[v] is not None:
            pu, a = prev[v]; path.append(a); v = pu
        if path:
            self._tally('navh')
        return path[-1] if path else None

    # ------------------------------------------------------------------ the sense (16 channels with steer_sense = 3)
    def sense(self):
        v = super().sense()
        if self.sense_w <= 12:
            return v
        v = (v + [0.0] * 12)[:12]
        g_on = 1.0 if self.goal is not None else 0.0
        hn = (self.hq / 6.0) if (self.goal is not None and self.hq is not None) else 0.0
        clk = 1.0 if (self.clocks.mask is not None and self.clocks.mask.any()) else 0.0
        return v + [g_on, max(0.0, min(1.0, hn)), float(self.dh_last), max(clk, 1.0 if self._wake else 0.0)]

    # ------------------------------------------------------------------ choosing
    def decide(self, brain, frame, a, d):
        self._brain = brain
        out = super().decide(brain, frame, a, d)
        try:
            self._from_brain = (out[0] is a and (out[1] or {}) == (d or {}))
        except Exception:
            self._from_brain = False
        if not (self.v4 and self.goal is not None and self.goal_override) or not self._from_brain or not self._cur:
            return out
        try:                                                                    # gene steer_goal_override: a proposal that
            g, lv = self._cur                                                   # RAISES h may be replaced by the plan
            k = self.key(g); nd = self.graph.get(k); O = self.objs.get(k) or []
            if nd is None or self.prev is None:
                return out
            bkey = self.prev[1]
            if not self.raises_h(self.sig_of(bkey, O)):
                return out
            alt, cv = self.best3(nd, bkey, O)
            if alt is None or alt[0] == bkey:
                return out
            akey, cls = alt
            self.prev = (self.prev[0], akey, cls, self.prev[3], self.prev[4])
            self.pred = (self.sig_of(akey, O), cv[1]) if cv and cv[1] is not None else None
            self._tally('goal'); self._from_brain = False
            return self.to_action(akey)
        except Exception as _e:
            self._v4_error = f'override: {type(_e).__name__}: {_e}'[:160]
            return out

    def stats(self):
        s = super().stats()
        s.update(v4=self.v4, goal_levels=len(self.gm.levels), goal_cells=int(self.goal_n or 0), h=self.h,
                 clocks=len(self.clocks.found), v4_error=getattr(self, '_v4_error', None))
        return s


Steer4._reprobe_pending = None
