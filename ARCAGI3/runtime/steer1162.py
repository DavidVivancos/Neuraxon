"""steer1162.py -- NAS 1162: STEERING v3 = the 1158 research's M1 + M3 loop, ONLINE, CHARGED, generic, around the brain.

WHAT THE RESEARCH FOUND (V1158_STATUS Status9, planner.py, fm_frame.py, fm_adapt.py, gbfws.py, nxon_policy.py)
  * M1 FORWARD MODEL is learnable: objects = 4-connected components; a transition = objects MATCHED between frames, each
    with an EFFECT from a small grammar (same / move k along the key / to the click / recolour / vanish / resize /
    appear); effects remembered per ACTION SIGNATURE (a key, or the clicked object's class) and replayed. Predicted-frame
    IoU on UNSEEN levels 0.54-0.77; within-level adaptation lp85 72 -> 94% after 3 observations.
  * M3 PLANNER = a one-to-few-step lookahead: predict each candidate's effect with M1, score the predicted frame, act,
    OBSERVE, COMPARE, update M1 (predict-act-compare; the transcripts' probe-then-commit: single probes missed 39%,
    planned steps 4.8%).
  * NOVELTY over board FEATURES (gbfws / IW(k)): features = each colour's pixel count and centroid on a 3-px grid; a
    state is worth expanding if it shows a never-seen feature. Room-making moves are neutral to any distance guide but
    NOVEL -- this cleared s5i5 L3 / L4 / L8, cd82 L2, ar25 L5 where guided search failed.
  * Root-pruning no-ops is HARMFUL (ACTION1/2 do nothing at the start, are needed later) -> the class veto is GATED.
  * Clicks are best grouped by EFFECT, not by object (probe a grid, keep one click per distinct effect).
  * M2 (the goal) is the crux; until a reference-match goal exists, novelty + the finished-board signature guide.
WHAT v3 DOES (on top of steer1161.Steer2; the brain still proposes every action)
  1. ONLINE M1: after every action, objects of the two frames are matched and the effects recorded under the action's
     signature; memory persists across levels of the game (cross-level memory).
  2. PREDICTION: for a candidate action, the remembered effects of its signature are applied to the current objects'
     FEATURES (counts, centroids) -> the predicted feature set; unknown signatures are flagged.
  3. SCORING = feature NOVELTY of the prediction (IW(1) atoms never seen this level) + goal-signature alignment
     + a PROBE bonus for unknown signatures in the probe phase - a penalty for signatures known to change nothing.
  4. THE BRAIN'S PROPOSAL is scored the same way; it is kept unless a candidate is predicted better by a margin
     (1 - steer_trust): the brain leads, the model corrects. Wasted / dead proposals are replaced by the best-scored
     candidate, V1 TARGETS FIRST (the 1161 bug: destinations / lattice cells only after objects are exhausted or once
     such a class has produced an effect in this game).
  5. PROBE / COMMIT / HALT-AT-MISS: every executed action with a prediction is compared with the outcome (the changed
     object classes); a miss switches to the probe phase for 3 steps (novelty + unknown signatures first).
  6. THE CLASS VETO is ON but GATED: from level 2 on (or after this play's first level-up); on level 1 only with the gene
     steer_classveto_on (lp85 1161: the veto opened L2, cost L1).
  7. THE BRAIN LEARNS: 12 sense channels (v2's 8 + new atoms last step, a recent miss, the predicted value of the brain's
     own proposal, the phase) when steer_sense = 2; teaching (steer_teach) rewards the brain when its own proposal is the
     plan's choice.
  8. INSTRUMENTATION: st_plan, st_hit, st_miss, st_novel, st_probe join the per-try tally.
GENES: steer_v3 (or MULTINXON_STEER_V3=1) | steer_no_model | steer_no_novelty | steer_no_classveto | steer_classveto_on
       (the veto on level 1 too) | + every 1160 / 1161 steering gene."""
from __future__ import annotations

import collections
import os

import numpy as np

from agent1160 import label, settled
from steer1161 import Steer2, _gene


def sbk(n):
    return 0 if n <= 1 else 1 if n <= 4 else 2 if n <= 16 else 3 if n <= 64 else 4 if n <= 256 else 5


def components(g, bg, ignore, maxsize=400, cap=80):
    """4-connected components of every non-background colour (<= maxsize px), as in the research's nxon_policy"""
    out = []
    for c in np.unique(g):
        c = int(c)
        if c == bg:
            continue
        m = (g == c) & ~ignore
        if not m.any():
            continue
        lab, n = label(m)
        if not n:
            continue
        idx = np.arange(1, n + 1)
        sl = __import__('scipy.ndimage', fromlist=['find_objects']).find_objects(lab) if _HAS_SCIPY else None
        flat = lab.ravel(); sizes = np.bincount(flat, minlength=n + 1)
        yy, xx = np.indices(lab.shape)
        sy = np.bincount(flat, weights=yy.ravel(), minlength=n + 1); sx = np.bincount(flat, weights=xx.ravel(), minlength=n + 1)
        for i in idx:
            s = int(sizes[i])
            if s == 0 or s > maxsize:
                continue
            if sl is not None:
                ys, xs = sl[i - 1]; y0, x0, h, w = ys.start, xs.start, ys.stop - ys.start, xs.stop - xs.start
            else:
                p = np.argwhere(lab == i); y0, x0 = p.min(0); h, w = p.max(0) - p.min(0) + 1
            out.append(dict(col=c, n=s, cy=sy[i] / s, cx=sx[i] / s, y0=int(y0), x0=int(x0), h=int(h), w=int(w)))
            if len(out) >= cap:
                return out
    return out


try:
    import scipy.ndimage  # noqa: F401
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def atoms_of(counts, cents):
    """gbfws features: ('n', colour, count bucket) and ('p', colour, cy // 3, cx // 3)"""
    out = set()
    for c, n in counts.items():
        if n <= 0:
            continue
        out.add(('n', c, n if n < 8 else 8 + int(np.log2(n))))
        sy, sx = cents[c]
        out.add(('p', c, int(sy / n) // 3, int(sx / n) // 3))
    return out


class Steer3(Steer2):
    def __init__(self, params, seed=0):
        super().__init__(params, seed=seed)
        g = lambda n, d: _gene(params, n, d)
        self.v3 = 1 if (int(g('steer_v3', 0) or 0) or str(os.environ.get('MULTINXON_STEER_V3', '0')) == '1') else 0
        self.v2 = 1 if self.v3 else self.v2                                     # v3 includes v2's primitives
        self.model_on = 0 if int(g('steer_no_model', 0) or 0) else 1
        self.novelty_on = 0 if int(g('steer_no_novelty', 0) or 0) else 1
        self.classveto = 0 if int(g('steer_no_classveto', 0) or 0) else int(g('steer_classveto', 12) or 12)
        self.classveto_l1 = int(g('steer_classveto_on', 0) or 0)
        self.sense_w = 12 if int(g('steer_sense', 0) or 0) >= 2 else 8
        self.mem = collections.defaultdict(collections.Counter); self.sig_n = collections.Counter(); self.sig_chg = collections.Counter()
        self.seen = set(); self.objs = {}; self.pred = None; self.miss_until = -1; self.t = 0
        self.last_new = 0; self.last_miss = 0; self.last_pv = 0.0; self.dl_effect = {'D': 0, 'L': 0}

    # ------------------------------------------------------------------ level / gating
    def start_level(self, g):
        super().start_level(g)
        self.seen = set(); self.objs = {}; self.pred = None

    def dead(self, cls):
        if not (self.level and self.level >= 1) and not self.classveto_l1:
            return False                                                        # GATED: not on level 1 (lp85 1161)
        return super().dead(cls)

    # ------------------------------------------------------------------ M1: objects, signatures, effects
    def objects_of(self, k, g):
        o = self.objs.get(k)
        if o is None:
            o = components(g, self.bg, self.ignore()); self.objs[k] = o
        return o

    def feats(self, O):
        counts = collections.Counter(); cents = collections.defaultdict(lambda: [0.0, 0.0])
        for o in O:
            counts[o['col']] += o['n']; cents[o['col']][0] += o['cy'] * o['n']; cents[o['col']][1] += o['cx'] * o['n']
        return counts, cents

    def sig_of(self, akey, O):
        if akey[0] == 'K':
            return ('K', akey[1])
        y, x = akey[1], akey[2]
        best = None; bd = 1e9
        for o in O:
            d = max(abs(o['cy'] - y), abs(o['cx'] - x))
            if d < bd:
                bd, best = d, o
        if best is None or bd > 12:
            return ('P', -1)                                                     # a click on empty floor
        return ('C', best['col'], sbk(best['n']), 0 if bd <= 1.5 else 1)       # on the object / beside it

    def effects(self, O0, O1, akey):
        used = set(); out = []
        for o in O0:
            cands = [j for j, p in enumerate(O1) if j not in used and p['col'] == o['col'] and p['n'] == o['n'] and p['h'] == o['h'] and p['w'] == o['w']]
            if cands:
                j = min(cands, key=lambda j: abs(O1[j]['cy'] - o['cy']) + abs(O1[j]['cx'] - o['cx'])); used.add(j)
                dy, dx = O1[j]['cy'] - o['cy'], O1[j]['cx'] - o['cx']
                if abs(dy) < 0.5 and abs(dx) < 0.5:
                    continue
                if akey[0] == 'C' and abs(O1[j]['cy'] - akey[1]) <= 2.5 and abs(O1[j]['cx'] - akey[2]) <= 2.5:
                    e = ('to_click',)
                else:
                    e = ('move', int(round(dy)), int(round(dx)))
            else:
                near = [j for j, p in enumerate(O1) if j not in used and abs(p['cy'] - o['cy']) <= 2.5 and abs(p['cx'] - o['cx']) <= 2.5]
                if near:
                    j = near[0]; used.add(j); p = O1[j]
                    e = ('recolour', p['col']) if p['col'] != o['col'] else ('resize', int(np.sign(p['n'] - o['n'])))
                else:
                    e = ('vanish',)
            out.append(((o['col'], sbk(o['n'])), e))
        out += [((p['col'], sbk(p['n'])), ('appear',)) for j, p in enumerate(O1) if j not in used][:4]
        return out

    def predict(self, sig, O, akey):
        """the predicted feature atoms after the action (None = unknown signature)"""
        n = self.sig_n.get(sig, 0)
        if not n:
            return None
        counts, cents = self.feats(O)
        for (ocls, e), c in self.mem[sig].items():
            if c < 0.5 * n:
                continue
            tg = [o for o in O if (o['col'], sbk(o['n'])) == ocls]
            if akey[0] == 'C' and tg:
                tg = [min(tg, key=lambda o: abs(o['cy'] - akey[1]) + abs(o['cx'] - akey[2]))]
            for o in tg[:4]:
                col, m = o['col'], o['n']
                if e[0] == 'move':
                    cents[col][0] += e[1] * m; cents[col][1] += e[2] * m
                elif e[0] == 'to_click' and akey[0] == 'C':
                    cents[col][0] += (akey[1] - o['cy']) * m; cents[col][1] += (akey[2] - o['cx']) * m
                elif e[0] == 'vanish':
                    counts[col] -= m; cents[col][0] -= o['cy'] * m; cents[col][1] -= o['cx'] * m
                elif e[0] == 'recolour':
                    nc = e[1]; counts[col] -= m; counts[nc] += m
                    cents[col][0] -= o['cy'] * m; cents[col][1] -= o['cx'] * m
                    cents[nc][0] += o['cy'] * m; cents[nc][1] += o['cx'] * m
                elif e[0] == 'resize':
                    counts[col] += e[1] * max(1, m // 4)
        return atoms_of(counts, cents)

    def value(self, sig, O, akey):
        """the plan's value of an action: novelty of its predicted features + goal alignment + probe bonus"""
        p = self.predict(sig, O, akey) if self.model_on else None
        if p is None:
            probing = self.t <= self.miss_until or len(self.sig_n) < 4
            return (0.8 if probing else 0.3), None
        if self.sig_chg.get(sig, 0) == 0 and self.sig_n[sig] >= 3:
            return -1.0, p                                                        # known to change nothing here
        nov = len(p - self.seen) if self.novelty_on else 0
        v = min(3, nov) * 0.5
        if self.sig is not None:
            st = self.eff.get(('K', sig[1])) if sig[0] == 'K' else None
            if st and st[1] > 0:
                d = st[4] / float(st[1]); nd = float(np.linalg.norm(d))
                if nd > 0:
                    v += 0.5 * float(d @ self.sig) / nd
        return v, p

    # ------------------------------------------------------------------ learning after every action (before the brain encodes)
    def _after_action(self, ak, g0, g1, ch, trans):
        super()._after_action(ak, g0, g1, ch, trans)
        if not self.v3 or ak is None or g0 is None or g1 is None or g0.shape != g1.shape:
            return
        self.t += 1
        try:
            k0 = self.key(g0); O0 = self.objs.get(k0) or components(g0, self.bg, self.ignore())
            O1 = components(g1, self.bg, self.ignore()); self.objs[self.key(g1)] = O1
            sig = self.sig_of(ak, O0); eff = self.effects(O0, O1, ak)
            self.sig_n[sig] += 1; self.sig_chg[sig] += 1 if (eff or ch or trans) else 0
            for e in eff:
                self.mem[sig][e] += 1
            if ak[0] == 'C':
                for kind in ('D', 'L'):
                    if self.prev_cls_kind == kind and (eff or ch):
                        self.dl_effect[kind] += 1
            c1, s1 = self.feats(O1); a1 = atoms_of(c1, s1)
            new = len(a1 - self.seen); self.seen |= a1; self.last_new = new
            if new:
                self._tally('novel', new)
            if self.pred is not None:                                           # COMPARE: the predicted vs the actual atoms
                ps, pa = self.pred
                if ps == sig:
                    hit = len(pa & a1) / float(max(1, len(pa | a1)))
                    if hit >= 0.6:
                        self._tally('hit'); self.last_miss = 0
                    else:
                        self._tally('miss'); self.last_miss = 1; self.miss_until = self.t + 3     # HALT AT MISS -> probe
            self.pred = None
        except Exception as _e:
            self._v3_error = f'{type(_e).__name__}: {_e}'[:160]

    def sense(self):
        v = super().sense()
        if self.sense_w <= 8:
            return v
        return v + [min(1.0, self.last_new / 4.0), float(self.last_miss), max(-1.0, min(1.0, self.last_pv)),
                    0.0 if self.t <= self.miss_until else 1.0]

    # ------------------------------------------------------------------ choosing: v1 targets first; the plan corrects the brain
    def tiers(self, nd, kinds_ok=None):
        pool = [x for x in nd['cands'] if x[0] not in nd['tried'] and not self.deadly(x[1])]
        live = [x for x in pool if not self.dead(x[1])] or pool
        t1 = [x for x in live if x[1][0] in ('K', 'C', 'R')]
        t2 = [x for x in live if x[1][0] in ('D', 'L') and (not t1 or self.dl_effect.get(x[1][0], 0) > 0)]
        return t1 + t2

    def best3(self, nd, bkey, O, limit=24):
        pool = self.tiers(nd)
        if not pool:
            return None, None
        pool = sorted(pool, key=lambda x: -(self.score2(x[1], x[0]) + self.w_brain * self.near(x[0], bkey)))[:limit]
        if not (self.v3 and self.model_on):
            return pool[0], None
        best = None; bv = -1e9; bp = None
        for a, cl in pool:
            v, p = self.value(self.sig_of(a, O), O, a)
            v += 0.05 * (self.score2(cl, a) + self.w_brain * self.near(a, bkey))
            if v > bv:
                best, bv, bp = (a, cl), v, p
        return best, (bv, bp)

    def decide(self, brain, frame, a, d):
        if not self.v3:
            return super().decide(brain, frame, a, d)
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
        if self.markov:
            wasted = t is not None and ((not t[1]) or t[2])
        else:
            rec = self.obs.get((k, bkey), [0, 0]); wasted = t is not None and (t[2] or (rec[0] >= 2 and rec[1] == 0))
        bsig = self.sig_of(bkey, O)
        bv, bp = self.value(bsig, O, bkey) if (self.model_on and not wasted) else (-1.0, None)
        self.last_pv = bv
        choice = None; why = None; cp = None
        if self.cfg['memory'] and (wasted or (t is None and self.deadly(bcls))):
            choice, cv = self.best3(nd, bkey, O); why = 'veto'
            cp = cv[1] if cv else None
            if choice is None and self.cfg['nav']:
                step = self.bfs(k)
                if step is not None:
                    choice = (step, next((cl for a2, cl in nd['cands'] if a2 == step), ('NAV',))); why = 'nav'
        elif self.dead(bcls):
            alt, cv = self.best3(nd, bkey, O)
            if alt is not None and not self.dead(alt[1]):
                choice, why, cp = alt, 'classveto', (cv[1] if cv else None)
        elif self.model_on and ((self.level and self.level >= 1) or self.classveto_l1) and \
                (bv < 0 or (bp is None and self.t > self.miss_until and len(self.sig_n) >= 4)):
            # ... and, like the class veto, only from level 2 on (the research: pruning no-ops at the ROOT is harmful;
            # the pre-launch A/B: on level 1 these overrides cost lp85 37 -> 49 and sp80 45 -> fail)
            # v1.162 (pre-launch A/B: the tn36 master's 9 / 7 actions became 34 / 18 when the plan overrode EFFECTIVE
            # proposals): the plan replaces only a proposal it predicts USELESS (a known no-op, or an unknown signature
            # once probing is over) -- never an effective action of a competent brain
            alt, cv = self.best3(nd, bkey, O)
            margin = 1.0 - min(0.9, max(0.0, self.trust))                       # trust lowers the bar for the plan
            if alt is not None and cv is not None and alt[0] != bkey and cv[0] > bv + margin:
                choice, why, cp = alt, 'plan', cv[1]
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

    def stats(self):
        s = super().stats()
        s.update(v3=self.v3, sigs=len(self.sig_n), atoms=len(self.seen), error=getattr(self, '_v3_error', None))
        return s


Steer3.prev_cls_kind = None
