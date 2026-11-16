"""steer1161.py -- NAS 1161: STEERING v2. The 1158 research, game by game, turned into GENERIC board primitives the
brain + steering can learn from what they see (no per-game code, no simulator copies, every action charged).

WHAT EACH PRIMITIVE COMES FROM (V1158_STATUS, Status8) -> what it does here
  * CLASS-LEVEL NO-OPS (lp85 82%, ft09 89%, sb26 53% of steps changed nothing in 1160, on NEW cells each time; the
    1158 sweep: su15 / sc25 "actions as tried produce no new frames"). A click class (colour, size) that changed nothing
    in steer_classveto tries, in any state, is vetoed while anything else is untried.
  * DESTINATIONS (r11l: the expert's clicks land on EMPTY floor, select a piece then click where it goes; lf52 carts and
    pegs, cn04 selection, sp80 bars: select then move). Empty-floor targets at +-r and +-r/2 around objects; after a
    click that SELECTED something (a small change at the clicked object) destinations are preferred.
  * MOVERS (tu93, m0r0, ar25, ls20, wa30, cn04, sk48: "movers learned by probing keys"; tr87's cursor). After a key, the
    cells vacated vs occupied give the controlled object's displacement; blocked keys (no change) are counted. The last
    key that moved the mover is preferred when the steering must replace an action (directional commitment: the
    expert's planned sequences missed 4.8%, its single probes 39%).
  * LATTICES (sp80 "infer the cell size from the common divisor of colour-run lengths"; tr87's 7-px pitch; lp85 / ft09 /
    sb26 boards). The level's cell size and offset are inferred once; contiguous same-colour regions get one candidate
    per lattice cell (a bar of 5 cells is 5 targets, not 1).
  * GOAL CUES from finished boards (stale frame; "colour gone" for sp80 / g50t / tr87 / cn04 is necessary not
    sufficient; ar25's 1/9 rule; vc33 alignment): the colour-change signature (1160) + the alignment of every effect
    with it, exposed to the brain as a SENSE.
  * HIDDEN STATE (g50t) and TRANSIENT EFFECTS (sp80 pour, tn36 run): kept from 1160b.
THE BRAIN LEARNS IT TOO
  * SENSES (gene steer_sense, newborn genomes): 8 input channels from the steering -- is this state new, the untried
    fraction here, was the last action a no-op, a transient, the mover's last dy / dx, the last effect's alignment with
    the goal signature, hidden state -- appended LAST to the encoder (inherited brains, whose width is fixed, trim it).
  * TEACHING (gene steer_teach): a small dopamine bonus when the brain's own proposal was the one executed and was a
    sensible untried action, a small penalty when the steering had to override it -> the network's DA-gated STDP
    learns what the steering knows (the 1160 masters' networks were near-silent).
  * INSTRUMENTATION: every decision is counted into the per-try engine tally (st_steps, st_veto, st_classveto,
    st_trust, st_nav, st_agree, st_dest, st_lattice, st_commit).
GENES (absent = default; NAS forces steer_v2 on the v2 lines and flips the components):
  steer_v2 0/1 | steer_classveto N (6) | steer_no_classveto 0/1 | steer_dest r (6) | steer_no_dest 0/1 |
  steer_no_lattice 0/1 | steer_no_commit 0/1 | steer_teach 0/1 | steer_sense 0/1 (the brain's input channels)"""
from __future__ import annotations

import collections
import os

import numpy as np

from agent1160 import first_frame, objects, settled
from steer1160 import Steer


def _gene(params, name, default):
    v = getattr(params, name, None)
    return default if v in (None, '') else v


def infer_lattice(g, bg):
    """(cell, off_y, off_x) from the colour-run lengths of non-background runs (sp80's rule), or None"""
    H, W = g.shape
    runs = collections.Counter(); sx = collections.defaultdict(collections.Counter); sy = collections.defaultdict(collections.Counter)
    for axis, step_lines, store in ((0, range(0, H, 2), sx), (1, range(0, W, 2), sy)):
        for i in step_lines:
            row = (g[i] if axis == 0 else g[:, i]).tolist(); x0 = 0
            for x in range(1, len(row) + 1):
                if x == len(row) or row[x] != row[x0]:
                    L = x - x0
                    if row[x0] != bg and 2 <= L <= 32:
                        runs[L] += 1
                        for c in range(3, 9):
                            if L % c == 0:
                                store[c][x0 % c] += 1
                    x0 = x
    if not runs:
        return None
    best = max(range(3, 9), key=lambda c: (sum(v for L, v in runs.items() if L % c == 0) * c, c))
    if sum(v for L, v in runs.items() if L % best == 0) < 8:
        return None
    ox = sx[best].most_common(1)[0][0] if sx[best] else 0
    oy = sy[best].most_common(1)[0][0] if sy[best] else 0
    return (best, oy, ox)


class Steer2(Steer):
    def __init__(self, params, seed=0):
        super().__init__(params, seed=seed)
        g = lambda n, d: _gene(params, n, d)
        self.v2 = 1 if (int(g('steer_v2', 0) or 0) or str(os.environ.get('MULTINXON_STEER_V2', '0')) == '1') else 0
        # v1.161 (A/B before launch: on tn36 the class veto + new targets turned 9 / 7 actions into 40 / 38): the class
        # veto is OPT-IN (gene steer_classveto_on, flippable; forced on lp85-v2 only), with stronger evidence (12 tries)
        self.classveto = int(g('steer_classveto', 12) or 12) if int(g('steer_classveto_on', 0) or 0) else 0
        self.dest = 0 if int(g('steer_no_dest', 0) or 0) else int(g('steer_dest', 6) or 6)
        self.lattice = 0 if int(g('steer_no_lattice', 0) or 0) else 1
        self.commit = 0 if int(g('steer_no_commit', 0) or 0) else 1
        self.teach = int(g('steer_teach', 0) or 0)
        self.cell = None; self.mover = {}; self.last_move = (0.0, 0.0); self.last_key = None
        self.last_changed = True; self.last_trans = False; self.last_align = 0.0; self.selected = None
        self.cnt = collections.Counter(); self.pre = False; self._cur = None; self._novel = 1.0; self._untried = 1.0

    # ------------------------------------------------------------------ bookkeeping
    def _tally(self, k, n=1):
        self.cnt[k] += n
        try:
            from arc_env import adapter as AD
            t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
        except Exception:
            pass

    def dead(self, cls):
        if not self.v2 or self.classveto <= 0 or not cls or cls[0] not in ('C', 'D', 'L', 'R'):
            return False
        st = self.eff.get(cls)
        return bool(st and st[0] >= self.classveto and st[1] == 0)

    def start_level(self, g):
        super().start_level(g)
        self.cell = infer_lattice(g, self.bg) if (self.v2 and self.lattice) else None
        self.selected = None

    # ------------------------------------------------------------------ candidates: + destinations + lattice cells
    def candidates(self, g, avail):
        out = super().candidates(g, avail)
        if not self.v2 or 6 not in avail:
            return out
        seen = {a for a, _ in out}; ign = self.ignore(); H, W = g.shape
        if self.dest > 0:
            objs = objects(g, ign, self.bg)
            cnt = np.bincount(np.clip(g.ravel(), 0, 15), minlength=16)
            objs.sort(key=lambda o: -(64.0 / (1.0 + cnt[o[0]]) + 8.0 / (1.0 + o[1])))
            for col, s, y, x in objs[:8]:
                for r in sorted({self.dest, max(2, self.dest // 2)}):
                    for dy, dx in ((-r, 0), (r, 0), (0, -r), (0, r)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and g[yy, xx] == self.bg and not ign[yy, xx] and ('C', yy, xx) not in seen:
                            seen.add(('C', yy, xx)); out.append((('C', yy, xx), ('D', r)))
        if self.cell:
            c, oy, ox = self.cell; n = 0
            for cy in range(oy + c // 2, H, c):
                for cx in range(ox + c // 2, W, c):
                    col = int(g[cy, cx])
                    if col != self.bg and not ign[cy, cx] and ('C', cy, cx) not in seen:
                        seen.add(('C', cy, cx)); out.append((('C', cy, cx), ('L', col))); n += 1
                        if n >= 64:
                            break
                if n >= 64:
                    break
        return out

    # ------------------------------------------------------------------ learning from the last action (before the brain encodes)
    def _after_action(self, ak, g0, g1, ch, trans):
        self.last_changed = bool(ch); self.last_trans = bool(trans)
        if g0 is None or g1 is None or g0.shape != g1.shape:
            return
        d = (g0 != g1) & ~self.ignore()
        if ak and ak[0] == 'K':
            m = self.mover.setdefault(ak[1], [0, 0, 0.0, 0.0]); m[0] += 1
            if ch:
                ys, xs = np.nonzero(d)
                a = [(y, x) for y, x in zip(ys.tolist(), xs.tolist()) if g1[y, x] != self.bg]
                b = [(y, x) for y, x in zip(ys.tolist(), xs.tolist()) if g0[y, x] != self.bg]
                if a and b:
                    dy = sum(p[0] for p in a) / len(a) - sum(p[0] for p in b) / len(b)
                    dx = sum(p[1] for p in a) / len(a) - sum(p[1] for p in b) / len(b)
                    m[1] += 1; m[2] += dy; m[3] += dx; self.last_move = (dy, dx); self.last_key = ak[1]
                else:
                    self.last_move = (0.0, 0.0)
            else:
                self.last_move = (0.0, 0.0)
            self.selected = None
        elif ak and ak[0] == 'C':
            self.selected = None
            if ch:
                ys, xs = np.nonzero(d)
                if len(ys) and (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) <= 0.2 * d.size:
                    y, x = ak[1], ak[2]
                    if ys.min() - 3 <= y <= ys.max() + 3 and xs.min() - 3 <= x <= xs.max() + 3:
                        self.selected = (y, x)                       # a local change at the clicked object: a SELECTION
        if ch and self.sig is not None:
            dh = self.hist(g1) - self.hist(g0); n = float(np.linalg.norm(dh))
            self.last_align = float(dh @ self.sig) / n if n else 0.0
        else:
            self.last_align = 0.0

    def observe_frame(self, frame):
        g = settled(frame, None)
        if g is None:
            return
        lv = int(getattr(frame, 'levels_completed', 0) or 0)
        if self.prev is not None:
            k0, ak, cl, g0, lv0 = self.prev
            up = lv > lv0; t0 = self.n_transient
            ch = self.observe(k0, ak, cl, g0, g, False, up, frames=getattr(frame, 'frame', None))
            self._after_action(ak, g0, g, ch, self.n_transient > t0)
            if up:
                fin = first_frame(frame); self.level_done(fin if fin is not None else g0)
            self.prev = None
        if self.level is None or lv != self.level:
            self.start_level(g); self.level = lv
        self._cur = (g, lv)
        nd = self.graph.get(self.key(g))
        self._novel = 1.0 if nd is None else 0.0
        self._untried = (sum(1 for a, _ in nd['cands'] if a not in nd['tried']) / max(1, len(nd['cands']))) if nd else 1.0
        self.pre = True

    def sense(self):
        """8 channels for the brain's encoder (gene steer_sense)"""
        dy, dx = self.last_move
        return [float(self._novel), float(self._untried), 0.0 if self.last_changed else 1.0, 1.0 if self.last_trans else 0.0,
                max(-1.0, min(1.0, dy / 8.0)), max(-1.0, min(1.0, dx / 8.0)), max(-1.0, min(1.0, self.last_align)),
                0.0 if self.markov else 1.0]

    # ------------------------------------------------------------------ choosing
    def score2(self, cls, akey=None):
        s = self.score(cls)
        if self.v2:
            if self.selected is not None and cls and cls[0] == 'D':
                s += 1.0                                             # select -> place
            if self.commit and akey is not None and akey[0] == 'K' and akey[1] == self.last_key and self.last_move != (0.0, 0.0):
                s += 0.5                                             # keep moving the mover
        return s

    def best2(self, nd, bkey, kinds=None):
        pool = [x for x in nd['cands'] if x[0] not in nd['tried'] and not self.deadly(x[1]) and (kinds is None or x[1][0] in kinds)]
        live = [x for x in pool if not self.dead(x[1])]
        pool = live or pool
        if not pool:
            return None
        return max(pool, key=lambda x: self.score2(x[1], x[0]) + self.w_brain * self.near(x[0], bkey))

    def decide(self, brain, frame, a, d):
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
        bkey, bcls = self.brain_key(g, a, d, nd)
        t = nd['tried'].get(bkey)
        if self.markov:
            wasted = t is not None and ((not t[1]) or t[2])
        else:
            rec = self.obs.get((k, bkey), [0, 0])
            wasted = t is not None and (t[2] or (rec[0] >= 2 and rec[1] == 0))
        choice = None; why = None
        if self.cfg['memory'] and (wasted or (t is None and self.deadly(bcls))):
            choice = self.best2(nd, bkey); why = 'veto'
            if choice is None and self.cfg['nav']:
                step = self.bfs(k)
                if step is not None:
                    choice = (step, next((cl for a2, cl in nd['cands'] if a2 == step), ('NAV',))); why = 'nav'
        elif self.dead(bcls):
            alt = self.best2(nd, bkey)
            if alt is not None and not self.dead(alt[1]):
                choice = alt; why = 'classveto'
        elif self.trust > 0 and self.rng.random() < self.trust:
            alt = self.best2(nd, bkey, kinds=('K', 'C', 'R'))     # trust picks only what v1 offered: new targets replace WASTE only
            if alt is not None and alt[0] != bkey and self.score2(alt[1], alt[0]) > self.score2(bcls, bkey) + 1e-9:
                choice = alt; why = 'trust'
        if choice is None:
            akey, cls, out = bkey, bcls, (a, d)
            if t is None and not self.dead(bcls):
                self._tally('agree')
                if self.teach:
                    brain._steer_teach_pending = float(getattr(brain, '_steer_teach_pending', 0.0) or 0.0) + 0.1
        else:
            akey, cls = choice
            out = self.to_action(akey)
            self._tally(why)
            if cls and cls[0] == 'D':
                self._tally('dest')
            elif cls and cls[0] == 'L':
                self._tally('lattice')
            if akey[0] == 'K' and akey[1] == self.last_key and self.commit:
                self._tally('commit')
            if self.teach:
                brain._steer_teach_pending = float(getattr(brain, '_steer_teach_pending', 0.0) or 0.0) - 0.05
            if why == 'veto':
                self.n_veto += 1
            elif why == 'nav':
                self.n_nav += 1; self.n_veto += 1
            elif why == 'trust':
                self.n_trust += 1
        self.prev = (k, akey, cls, g, lv); self.n_steps += 1; self._tally('steps')
        return out

    def stats(self):
        s = super().stats(); s.update({k: v for k, v in self.cnt.items()}); s['cell'] = self.cell; s['v2'] = self.v2
        return s
