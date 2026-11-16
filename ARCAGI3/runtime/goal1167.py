"""goal1167.py -- NAS 1167, research R3: THE STRUCTURED GOAL (the 1165 report §21.10, the 1166 report §19d).

WHY. The research's per-game guides were STRUCTURAL counts (wa30: uncovered target pixels / 4; ka59: enclosed floor / 9;
vc33: marker gaps rounded to thirds), not a cell-wise match. FrameLab 4 measured the cell-wise goal learned within a game at
WHERE 0.534 (precision at the true count of goal cells) and WHAT 0.494 (colour accuracy on the true goal cells), and a goal
at that precision SUBTRACTED clears; the true goal added +40%. The rule since 1165: keep the structured per-game goal as
research until its precision is above ~0.7.

WHAT IT IS. From the (start, finished) boards of the levels a play has seen complete (the prefix's levels and its own clears,
exactly what Steer4.prior_levels / observe_frame already collect), learn which STRUCTURAL PREDICATES held at the end of every
level of this game, and predict them for the next level from its start board:
  per colour c (the background and the HUD excluded):   vanish, appear, grow, shrink, merge (fewer components), split,
                                                         moved (same size, other cells), align (components in one row / column)
  per colour pair (c, d):                                onto (c arrives on d's start cells), inside (c ends enclosed by d)
A predicate is predicted for level L when it is APPLICABLE there (its colours are on L's start board) and it held in every
earlier level where it was applicable (mode 'all'; 'majority' = at least half). Colours are this game's own colours: the
predicates are learned within a game and never shared between games (the research: goals transfer within a game, 0.36 IoU,
not across games, 0.12).

THE GUIDE. h(board) = the predicted predicates not yet satisfied on `board`, against the level's start board. The steering
prior (gene steer_goal_struct, Steer4's goal path; install() below) adds steer_goal_w x the observed mean drop of h for an
action signature to the plan's value, exactly like the 1163 goal map's term, and keeps the counters st_sg_* in the tally.
Off unless the gene is 1. The ORACLE (steer_goal_struct_oracle, LAB ONLY: the preflight refuses it in any cell) takes the
predicates of the level's TRUE finished board (boards1164.npz) -- the ceiling of the mechanism.

THE STUDY (the precision the gate asks for), on the 183 verified (start, finished) pairs of boards1164.npz:
    python3 goal1167.py study [--out research/goal1167_precision.json]
  per game and level L >= 2: the predicates predicted from levels 1..L-1, their precision and recall at L; and the same
  corpus scored the FrameLab way (WHERE at the true count, WHAT on the true goal cells) with a per-colour change / transition
  model learned from the earlier levels, so the two numbers sit next to FrameLab's 0.534 / 0.494."""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:                                      # pragma: no cover - the fleet pins scipy; a fallback keeps import safe
    _ndi = None

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = 0.7                                             # the precision the 1165 report asked for before any use in the fleet
FRAMELAB = {'where_within': 0.534, 'what_within': 0.494, 'where_cross': 0.412, 'what_cross': 0.330, 'levels': 158}
KINDS1 = ('vanish', 'appear', 'grow', 'shrink', 'merge', 'split', 'moved', 'align')
KINDS2 = ('onto', 'inside')


# ================================================================================================ board primitives
def _label(m):
    if _ndi is not None:
        return _ndi.label(m)
    import goalmap1163 as GM
    return GM._label(m)


def _fill(m):
    if _ndi is not None:
        return _ndi.binary_fill_holes(m)
    import goalmap1163 as GM
    return GM._fill_holes(m)


def _masked(a, ignore):
    a = np.asarray(a, np.int16).copy()
    if ignore is not None and np.asarray(ignore).shape == a.shape:
        a[np.asarray(ignore, bool)] = -1
    return a


def bg_of(a):
    v = a[a >= 0]
    return int(np.bincount(np.clip(v, 0, 15), minlength=16).argmax()) if v.size else 0


def _comps(m):
    lab, n = _label(m)
    if not n:
        return []
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        out.append((float(ys.mean()), float(xs.mean()), len(ys)))
    return out


def _aligned(cs):
    """>= 2 components whose centroids all share one row or one column (within 1 cell)"""
    if len(cs) < 2:
        return False
    ys = [c[0] for c in cs]; xs = [c[1] for c in cs]
    return (max(ys) - min(ys) <= 1.0) or (max(xs) - min(xs) <= 1.0)


class Stats:
    """per-colour counts / components of one board (HUD cells = -1)"""

    def __init__(self, a, bg):
        self.a = a; self.bg = bg
        self.cols = sorted(int(c) for c in np.unique(a) if c >= 0 and c != bg)
        self.n = {c: int((a == c).sum()) for c in self.cols}
        self._k = {}; self._cs = {}

    def count(self, c):
        return self.n.get(c, 0) if c in self.n else int((self.a == c).sum())

    def comps(self, c):
        if c not in self._cs:
            self._cs[c] = _comps(self.a == c)
        return self._cs[c]

    def k(self, c):
        return len(self.comps(c))


def _near(a, b):
    return abs(a - b) <= max(1, 0.1 * max(a, b))


# ================================================================================================ predicates of one level
def predicates(S, F, ignore=None):
    """{(kind, colours): True/False} of the CHANGE predicates of a level whose start board is S and finished board F, plus
    {(kind, colours): applicable at S} -- the truth is measured on (S, F), applicability on S alone"""
    S = _masked(S, ignore); F = _masked(F, ignore)
    if S.shape != F.shape:
        return {}, {}
    bg = bg_of(S)
    sS = Stats(S, bg); sF = Stats(F, bg)
    cols = sorted(set(sS.cols) | set(sF.cols))
    truth = {}; app = {}
    for c in cols:
        nS, nF = sS.count(c), sF.count(c)
        app[('appear', (c,))] = nS == 0
        for kd in ('vanish', 'grow', 'shrink', 'merge', 'split', 'moved', 'align'):
            app[(kd, (c,))] = nS > 0
        truth[('vanish', (c,))] = nS > 0 and nF == 0
        truth[('appear', (c,))] = nS == 0 and nF > 0
        truth[('grow', (c,))] = nS > 0 and nF > nS and not _near(nS, nF)
        truth[('shrink', (c,))] = nF > 0 and nF < nS and not _near(nS, nF)
        kS, kF = (sS.k(c) if nS else 0), (sF.k(c) if nF else 0)
        truth[('merge', (c,))] = kS >= 2 and 0 < kF < kS
        truth[('split', (c,))] = kS >= 1 and kF > kS
        truth[('moved', (c,))] = nS > 0 and nF > 0 and _near(nS, nF) and bool(((S == c) != (F == c)).any())
        truth[('align', (c,))] = nF > 0 and _aligned(sF.comps(c)) and not (nS > 0 and _aligned(sS.comps(c)))
    for c in cols:
        arr = (F == c) & (S != c)
        na = int(arr.sum())
        for d in cols:
            if d == c:
                continue
            app[('onto', (c, d))] = sS.count(d) > 0
            truth[('onto', (c, d))] = na >= 1 and int((arr & (S == d)).sum()) >= 0.5 * na
            app[('inside', (c, d))] = sS.count(d) > 0
            nF_c = sF.count(c)
            if nF_c and sF.count(d):
                holes = _fill(F == d) & (F != d)
                insF = int((holes & (F == c)).sum()) >= 0.5 * nF_c
            else:
                insF = False
            if insF and sS.count(c) and sS.count(d):
                holesS = _fill(S == d) & (S != d)
                insS = int((holesS & (S == c)).sum()) >= 0.5 * sS.count(c)
            else:
                insS = False
            truth[('inside', (c, d))] = insF and not insS
    return truth, app


def applicable(S, ignore=None):
    """{predicate: applicable} for a start board alone (the next level's, before it is played)"""
    S = _masked(S, ignore)
    bg = bg_of(S); sS = Stats(S, bg)
    present = set(sS.cols)
    return present, bg


# ================================================================================================ the within-game model
class StructGoal:
    """the structured goal of ONE game, learned from the levels this play has seen complete"""

    def __init__(self, mode='all', min_levels=1):
        self.levels = []                                # (truth, app) per completed level
        self.mode = mode; self.min_levels = int(min_levels)
        self.trans = collections.defaultdict(collections.Counter)    # start colour -> finished colour of changed cells
        self.rate = collections.defaultdict(lambda: [0, 0])          # start colour -> [changed, cells]

    def add(self, S, F, ignore=None):
        try:
            t, a = predicates(S, F, ignore)
        except Exception:
            return False
        if not t:
            return False
        self.levels.append((t, a))
        Sm = _masked(S, ignore); Fm = _masked(F, ignore)
        if Sm.shape == Fm.shape:
            ok = Sm >= 0
            ch = (Sm != Fm) & ok
            for c in np.unique(Sm[ok]):
                m = (Sm == c) & ok
                r = self.rate[int(c)]; r[0] += int((m & ch).sum()); r[1] += int(m.sum())
                if (m & ch).any():
                    vals, cnt = np.unique(Fm[m & ch], return_counts=True)
                    for v, n in zip(vals, cnt):
                        self.trans[int(c)][int(v)] += int(n)
        return True

    def predict(self, S, ignore=None):
        """[(predicate, support, applicable_levels)] predicted for the level whose start board is S"""
        if len(self.levels) < self.min_levels:
            return []
        present, bg = applicable(S, ignore)
        held = collections.Counter(); seen = collections.Counter()
        for t, a in self.levels:
            for p, ok in a.items():
                if ok:
                    seen[p] += 1
                    if t.get(p):
                        held[p] += 1
        out = []
        for p, n in held.items():
            kind, cs = p
            if kind == 'appear':
                if cs[0] in present or cs[0] == bg:
                    continue
            elif kind in ('onto', 'inside'):
                if cs[1] not in present:
                    continue
            elif cs[0] not in present:
                continue
            m = seen[p]
            if (self.mode == 'all' and n == m) or (self.mode == 'majority' and n >= 0.5 * m):
                out.append((p, n, m))
        return out

    # ------------------------------------------------------------------ the cell-wise view (FrameLab's yardstick)
    def where_scores(self, S, ignore=None):
        Sm = _masked(S, ignore)
        tot = [sum(v[0] for v in self.rate.values()), sum(v[1] for v in self.rate.values())]
        base = tot[0] / max(1, tot[1])
        sc = np.full(Sm.shape, -1.0)
        for c in np.unique(Sm[Sm >= 0]):
            r = self.rate.get(int(c))
            sc[Sm == c] = (r[0] / r[1]) if (r and r[1]) else base
        return sc

    def what_of(self, c, bg):
        t = self.trans.get(int(c))
        if t:
            return t.most_common(1)[0][0]
        allc = collections.Counter()
        for v in self.trans.values():
            allc.update(v)
        return allc.most_common(1)[0][0] if allc else bg


# ================================================================================================ the guide h
def satisfied(p, board, start, ignore=None):
    """is predicate p satisfied by `board` against the level's start board?"""
    B = _masked(board, ignore); S = _masked(start, ignore)
    if B.shape != S.shape:
        return False
    kind, cs = p
    c = cs[0]
    nS, nB = int((S == c).sum()), int((B == c).sum())
    if kind == 'vanish':
        return nB == 0
    if kind == 'appear':
        return nB > 0
    if kind == 'grow':
        return nB > nS and not _near(nS, nB)
    if kind == 'shrink':
        return 0 < nB < nS and not _near(nS, nB)
    if kind == 'merge':
        kS = len(_comps(S == c)); kB = len(_comps(B == c)) if nB else 0
        return 0 < kB < kS
    if kind == 'split':
        return len(_comps(B == c)) > len(_comps(S == c)) if nB else False
    if kind == 'moved':
        return nB > 0 and _near(nS, nB) and bool(((S == c) != (B == c)).any())
    if kind == 'align':
        return nB > 0 and _aligned(_comps(B == c))
    d = cs[1]
    if kind == 'onto':
        arr = (B == c) & (S != c); na = int(arr.sum())
        return na >= 1 and int((arr & (S == d)).sum()) >= 0.5 * na
    if kind == 'inside':
        if not nB or not int((B == d).sum()):
            return False
        holes = _fill(B == d) & (B != d)
        return int((holes & (B == c)).sum()) >= 0.5 * nB
    return False


def h_of(preds, board, start, ignore=None):
    """the predicted predicates NOT yet satisfied (0 = every structural goal met)"""
    if not preds:
        return None
    return sum(0 if satisfied(p, board, start, ignore) else 1 for p in preds)


# ================================================================================================ the steering prior (gene)
def _gene(params, name, default):
    v = getattr(params, name, None)
    return default if v in (None, '') else v


def install(Steer4):
    """steer_goal_struct = 1: the structured guide joins Steer4's goal path (idempotent; inert at 0)"""
    if getattr(Steer4, '_v1167_struct', False):
        return
    Steer4._v1167_struct = True
    _init0 = Steer4.__init__; _start0 = Steer4.start_level; _prior0 = Steer4.prior_levels
    _obs0 = Steer4.observe_frame; _after0 = Steer4._after_action; _val0 = Steer4.value; _stats0 = Steer4.stats
    _dec0 = Steer4.decide

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        self.sg_on = bool(int(_gene(params, 'steer_goal_struct', 0) or 0)) or os.environ.get('MULTINXON_STEER_GOAL_STRUCT', '0') == '1'
        self.sg_oracle = bool(int(_gene(params, 'steer_goal_struct_oracle', 0) or 0))
        self.sg_mode = 'majority' if int(_gene(params, 'steer_goal_struct_mode', 0) or 0) == 1 else 'all'
        # the goal ACTS (research R3c): a brain proposal whose action signature has raised h (its summed dh > 0 on this level)
        # is replaced by the plan's best alternative (which carries the goal's value term). Without it the goal
        # only re-ranks the plan's alternatives, and the plan replaces only proposals it predicts useless (v1.162)
        self.sg_override = bool(int(_gene(params, 'steer_goal_struct_override', 0) or 0))
        self.sg = StructGoal(mode=self.sg_mode); self.sg_pred = []; self.sg_h = None
        self.sg_dh = collections.defaultdict(lambda: [0.0, 0]); self.sg_w = float(_gene(params, 'steer_goal_w', 0.4) or 0.4)

    def _sg_level(self, g):
        self.sg_pred = []; self.sg_h = None; self.sg_dh = collections.defaultdict(lambda: [0.0, 0])
        if not (self.sg_on or self.sg_oracle) or g is None:
            return
        try:
            if self.sg_oracle:
                self.sg_pred = _oracle_preds(g, self.ignore())
            else:
                self.sg_pred = [p for p, _, _ in self.sg.predict(g, self.ignore())]
            if self.sg_pred:
                self.sg_h = h_of(self.sg_pred, g, g, self.ignore()); self._tally('sg_fit'); self._tally('sg_preds', len(self.sg_pred))
        except Exception as _e:
            self._v1167_error = f'sg: {type(_e).__name__}: {_e}'[:160]

    def start_level(self, g):
        _start0(self, g)
        _sg_level(self, g)

    def prior_levels(self, pairs, hud=None):
        n = _prior0(self, pairs, hud)
        if self.sg_on and not self.sg_oracle:
            for s, f in pairs or []:
                try:
                    if s is not None and f is not None and np.asarray(s).shape == np.asarray(f).shape:
                        self.sg.add(s, f, hud if (hud is not None and np.asarray(hud).shape == np.asarray(s).shape) else None)
                except Exception:
                    pass
            if self.start_board is not None:
                _sg_level(self, self.start_board)
        return n

    def observe_frame(self, frame):
        if self.sg_on and not self.sg_oracle and self.prev is not None:
            try:
                import goalmap1163 as GM
                lv = int(getattr(frame, 'levels_completed', 0) or 0)
                k0, ak, cl, g0, lv0 = self.prev
                if lv > lv0 and self.start_board is not None and g0 is not None:
                    fin = GM.finished_board(g0, getattr(frame, 'frame', None))
                    if fin is not None and fin.shape == self.start_board.shape and self.sg.add(self.start_board, fin, self.ignore()):
                        self._tally('sg_learn')
            except Exception:
                pass
        return _obs0(self, frame)

    def _after_action(self, ak, g0, g1, ch, trans):
        sig = None
        if self.sg_pred and ak is not None and g0 is not None and g1 is not None and g0.shape == g1.shape:
            try:
                from steer1162 import components
                k0 = self.key(g0); O0 = self.objs.get(k0)
                if O0 is None:
                    O0 = components(g0, self.bg, self.ignore()); self.objs[k0] = O0
                sig = self.sig_of(ak, O0)
            except Exception:
                sig = None
        _after0(self, ak, g0, g1, ch, trans)
        if sig is None or not self.sg_pred or self.start_board is None:
            return
        try:
            h1 = h_of(self.sg_pred, g1, self.start_board, self.ignore())
            if h1 is not None:
                dh = h1 - (self.sg_h if self.sg_h is not None else h1)
                self.sg_h = h1
                st = self.sg_dh[sig]; st[0] += dh; st[1] += 1
                if dh < 0:
                    self._tally('sg_down')
                elif dh > 0:
                    self._tally('sg_up')
        except Exception as _e:
            self._v1167_error = f'sg after: {type(_e).__name__}: {_e}'[:160]

    def value(self, sig, O, akey):
        v, p = _val0(self, sig, O, akey)
        if self.sg_pred and self.sg_w > 0:
            st = self.sg_dh.get(sig)
            if st and st[1] > 0:
                v += self.sg_w * max(-1.0, min(1.0, -st[0] / float(st[1])))
        return v, p

    def raises_sg(self, sig):
        # h is sparse (a few changes a level), so one try that raised h with nothing lowering it since marks the signature
        # (v4's rule, >= 2 tries at mean > 0.5, never fired on vc33 L2: 0 overrides in 4 plays)
        st = self.sg_dh.get(sig)
        return bool(st and st[1] >= 1 and st[0] > 0)

    def decide(self, brain, frame, a, d):
        out = _dec0(self, brain, frame, a, d)
        if not (self.sg_override and self.sg_pred) or not getattr(self, '_from_brain', False) or not self._cur:
            return out
        try:                                                                    # the brain's own proposal, kept so far
            g, lv = self._cur
            k = self.key(g); nd = self.graph.get(k); O = self.objs.get(k) or []
            if nd is None or self.prev is None:
                return out
            bkey = self.prev[1]
            if not raises_sg(self, self.sig_of(bkey, O)):
                return out
            alt, cv = self.best3(nd, bkey, O)
            if alt is None or alt[0] == bkey or raises_sg(self, self.sig_of(alt[0], O)):
                return out
            akey, cls = alt
            self.prev = (self.prev[0], akey, cls, self.prev[3], self.prev[4])
            self.pred = (self.sig_of(akey, O), cv[1]) if cv and cv[1] is not None else None
            self._tally('sg_override'); self._from_brain = False
            return self.to_action(akey)
        except Exception as _e:
            self._v1167_error = f'sg override: {type(_e).__name__}: {_e}'[:160]
            return out

    def stats(self):
        s = _stats0(self)
        s.update(sg_on=int(bool(self.sg_on)), sg_oracle=int(bool(self.sg_oracle)), sg_levels=len(self.sg.levels),
                 sg_preds=len(self.sg_pred or []), sg_h=self.sg_h, sg_override=int(bool(self.sg_override)),
                 v1167_error=getattr(self, '_v1167_error', None))
        return s

    Steer4.__init__ = __init__; Steer4.start_level = start_level; Steer4.prior_levels = prior_levels
    Steer4.observe_frame = observe_frame; Steer4._after_action = _after_action; Steer4.value = value; Steer4.stats = stats
    Steer4.decide = decide


_ORACLE = {'key': None}


def _oracle_preds(g, ignore):
    """LAB ONLY: the predicates of the level's TRUE finished board (boards1164.npz, the level the play is on)"""
    import prog1164 as P
    key = P._T.get('key')
    if not key:
        return []
    game, L = key
    B = P._boards()
    s = B.get(f'{game}:{int(L)}:start'); f = B.get(f'{game}:{int(L)}:finished'); hud = B.get(f'{game}:hud')
    if s is None or f is None:
        return []
    t, a = predicates(s, f, hud if (hud is not None and hud.shape == s.shape) else None)
    return [p for p, ok in t.items() if ok and a.get(p)]


# ================================================================================================ the study
def _corpus(npz):
    z = np.load(npz)
    games = collections.defaultdict(dict); hud = {}
    for k in z.files:
        parts = k.split(':')
        if len(parts) == 2 and parts[1] == 'hud':
            hud[parts[0]] = z[k].astype(bool)
        elif len(parts) == 3:
            games[parts[0]].setdefault(int(parts[1]), {})[parts[2]] = z[k]
    return games, hud


def study(npz=None, out=None, seed=1167):
    npz = npz or os.path.join(HERE, 'boards1164.npz')
    games, hud = _corpus(npz)
    rng = np.random.default_rng(seed)
    rows = []
    for g in sorted(games):
        lv = games[g]; H = hud.get(g)
        Ls = sorted(L for L in lv if 'start' in lv[L] and 'finished' in lv[L])
        models = {m: StructGoal(mode=m) for m in ('all', 'majority')}
        for L in Ls:
            S, F = lv[L]['start'], lv[L]['finished']
            ign = H if (H is not None and H.shape == S.shape) else None
            if L >= 2 and models['all'].levels:
                truth, app = predicates(S, F, ign)
                true_app = {p for p, ok in truth.items() if ok and app.get(p)}
                r = {'game': g, 'L': L, 'prior_levels': len(models['all'].levels), 'true_predicates': len(true_app)}
                # baselines: (1) every applicable predicate of the kinds the model predicted (the base rate);
                # (2) the predicates true in the PREVIOUS level only (copy the last level)
                pr_all = [p for p, _, _ in models['all'].predict(S, ign)]
                kinds_here = {p[0] for p in pr_all}
                base = [p for p, ok in app.items() if ok and p[0] in kinds_here]
                r['base_pred'] = len(base); r['base_hit'] = sum(1 for p in base if truth.get(p))
                lt, la = models['all'].levels[-1]
                last = [p for p, ok in lt.items() if ok and app.get(p)]
                r['last_pred'] = len(last); r['last_hit'] = sum(1 for p in last if truth.get(p))
                r['nomoved_pred'] = sum(1 for p in pr_all if p[0] != 'moved'); r['nomoved_hit'] = sum(1 for p in pr_all if p[0] != 'moved' and truth.get(p))
                for m, M in models.items():
                    pr = [p for p, _, _ in M.predict(S, ign)]
                    hit = sum(1 for p in pr if truth.get(p))
                    r[f'{m}_pred'] = len(pr); r[f'{m}_hit'] = hit
                    r[f'{m}_precision'] = round(hit / len(pr), 4) if pr else None
                    r[f'{m}_recall'] = round(hit / len(true_app), 4) if true_app else None
                    r[f'{m}_h_start'] = h_of(pr, S, S, ign) if pr else None
                    r[f'{m}_h_finished'] = h_of(pr, F, S, ign) if pr else None
                    if m == 'all':
                        r['kinds_pred'] = dict(collections.Counter(p[0] for p in pr))
                        r['kinds_hit'] = dict(collections.Counter(p[0] for p in pr if truth.get(p)))
                # FrameLab's yardstick: WHERE at the true count, WHAT on the true goal cells (the per-colour model)
                M = models['all']
                Sm = _masked(S, ign); Fm = _masked(F, ign); ok = Sm >= 0
                goal = (Sm != Fm) & ok; n = int(goal.sum())
                if n > 0:
                    sc = M.where_scores(S, ign) + 1e-6 * rng.random(S.shape)
                    sc[~ok] = -9.0
                    top = np.argsort(-sc.ravel())[:n]
                    r['goal_cells'] = n
                    r['where_precision'] = round(float(goal.ravel()[top].mean()), 4)
                    r['where_random'] = round(n / float(ok.sum()), 4)
                    bg = bg_of(Sm)
                    pred_col = np.vectorize(lambda c: M.what_of(int(c), bg))(Sm[goal])
                    r['what_accuracy'] = round(float((pred_col == Fm[goal]).mean()), 4)
                rows.append(r)
            for M in models.values():
                M.add(S, F, ign)
    res = {'levels': len(rows), 'framelab_within': FRAMELAB, 'gate': GATE, 'rows': rows}
    for m in ('all', 'majority'):
        P = [r for r in rows if r.get(f'{m}_precision') is not None]
        hit = sum(r[f'{m}_hit'] for r in P); n = sum(r[f'{m}_pred'] for r in P)
        res[f'{m}_levels_with_predictions'] = len(P)
        res[f'{m}_precision_pooled'] = round(hit / n, 4) if n else None
        res[f'{m}_precision_mean'] = round(float(np.mean([r[f'{m}_precision'] for r in P])), 4) if P else None
        R = [r for r in rows if r.get(f'{m}_recall') is not None]
        res[f'{m}_recall_mean'] = round(float(np.mean([r[f'{m}_recall'] for r in R])), 4) if R else None
        res[f'{m}_h_zero_at_finish'] = sum(1 for r in P if r.get(f'{m}_h_finished') == 0)
    for b in ('base', 'last', 'nomoved'):
        hit = sum(r.get(f'{b}_hit', 0) for r in rows); n = sum(r.get(f'{b}_pred', 0) for r in rows)
        res[f'{b}_precision_pooled'] = round(hit / n, 4) if n else None; res[f'{b}_predicted'] = n
    W = [r for r in rows if 'where_precision' in r]
    res['where_precision_mean'] = round(float(np.mean([r['where_precision'] for r in W])), 4) if W else None
    res['where_random_mean'] = round(float(np.mean([r['where_random'] for r in W])), 4) if W else None
    res['what_accuracy_mean'] = round(float(np.mean([r['what_accuracy'] for r in W])), 4) if W else None
    per_game = {}
    for g in sorted({r['game'] for r in rows}):
        G = [r for r in rows if r['game'] == g and r.get('all_precision') is not None]
        hit = sum(r['all_hit'] for r in G); n = sum(r['all_pred'] for r in G)
        per_game[g] = {'levels': sum(1 for r in rows if r['game'] == g), 'levels_with_predictions': len(G),
                       'precision_pooled': round(hit / n, 3) if n else None, 'predicted': n,
                       'where': round(float(np.mean([r['where_precision'] for r in rows if r['game'] == g and 'where_precision' in r])), 3)
                       if any('where_precision' in r for r in rows if r['game'] == g) else None,
                       'what': round(float(np.mean([r['what_accuracy'] for r in rows if r['game'] == g and 'what_accuracy' in r])), 3)
                       if any('what_accuracy' in r for r in rows if r['game'] == g) else None}
        per_game[g]['passes_gate'] = bool(per_game[g]['precision_pooled'] is not None and per_game[g]['precision_pooled'] >= GATE
                                          and per_game[g]['levels_with_predictions'] >= 2)
    res['per_game'] = per_game
    kinds = collections.Counter(); khit = collections.Counter()
    for r in rows:
        kinds.update(r.get('kinds_pred') or {}); khit.update(r.get('kinds_hit') or {})
    res['per_kind'] = {k: {'predicted': kinds[k], 'precision': round(khit[k] / kinds[k], 3)} for k in sorted(kinds)}
    res['gate_passed_overall'] = bool(res.get('all_precision_pooled') is not None and res['all_precision_pooled'] >= GATE)
    res['games_passing_gate'] = sorted(g for g, v in per_game.items() if v['passes_gate'])
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1)
    return res


def _spearman(x, y):
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def pathcheck(npz=None, out=None, games=None):
    """does h FALL along the verified expert path? For every level L >= 2 with a prediction: replay the prefix to L's start on
    the REAL engine, then the expert's moves of L; h after every move on level L (the completing move excluded)"""
    os.environ.setdefault('MULTINXON_ARC_ENV_DIR', os.path.join(HERE, 'environment_files'))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from arc_env.adapter import make_env
    from arc_env.structs import GameAction as GA
    P = json.load(open(os.path.join(HERE, 'level_prefix.json'), encoding='utf-8'))['prefix']
    gl, hud = _corpus(npz or os.path.join(HERE, 'boards1164.npz'))
    rows = []

    def step(env, a):
        if a == 'RESET':
            return env.reset()
        if str(a).startswith('ACTION6:'):
            x, y = map(int, str(a).split(':')[1].split(','))
            return env.step(GA.ACTION6, {'x': x, 'y': y})
        return env.step(GA[str(a)])

    def grid(obs):
        fr = getattr(obs, 'frame', None) or []
        return np.asarray(fr[-1], np.int16) if fr else None
    for g in sorted(games or gl):
        lv = gl[g]; H = hud.get(g); M = StructGoal('all')
        for L in sorted(lv):
            S, F = lv[L].get('start'), lv[L].get('finished')
            if S is None or F is None:
                continue
            ign = H if (H is not None and H.shape == S.shape) else None
            pr = [p for p, _, _ in M.predict(S, ign)] if (L >= 2 and M.levels) else []
            if pr and str(L) in P.get(g, {}) and str(L + 1) in P.get(g, {}):
                try:
                    env = make_env(g, seed=0, prefer_real=True, force_offline=False)
                    obs = env.reset()
                    for a in P[g][str(L)]:
                        obs = step(env, a)
                    b0 = grid(obs); same = b0 is not None and b0.shape == S.shape and bool((b0 == S).all())
                    moves = P[g][str(L + 1)][len(P[g][str(L)]):]
                    seq = [h_of(pr, b0, b0, ign)]; start = b0; lv0 = int(obs.levels_completed or 0)
                    for a in moves[:-1]:
                        obs = step(env, a)
                        if int(obs.levels_completed or 0) > lv0:
                            break
                        b = grid(obs)
                        if b is None or b.shape != start.shape:
                            break
                        seq.append(h_of(pr, b, start, ign))
                    d = [b - a for a, b in zip(seq, seq[1:])]
                    rows.append({'game': g, 'L': L, 'preds': len(pr), 'start_matches_corpus': same, 'moves': len(moves),
                                 'h_start': seq[0], 'h_min': min(seq), 'h_before_last': seq[-1], 'h_finished': h_of(pr, F, S, ign),
                                 'down_steps': sum(1 for x in d if x < 0), 'up_steps': sum(1 for x in d if x > 0),
                                 'spearman_step_h': _spearman(list(range(len(seq))), seq)})
                except Exception as e:
                    rows.append({'game': g, 'L': L, 'error': f'{type(e).__name__}: {e}'[:160]})
            M.add(S, F, ign)
    ok = [r for r in rows if 'error' not in r]
    res = {'levels': len(ok), 'errors': len(rows) - len(ok), 'rows': rows,
           'h_falls_start_to_before_last': sum(1 for r in ok if r['h_before_last'] < r['h_start']),
           'h_rises': sum(1 for r in ok if r['h_before_last'] > r['h_start']),
           'h_flat': sum(1 for r in ok if r['h_before_last'] == r['h_start']),
           'down_steps': sum(r['down_steps'] for r in ok), 'up_steps': sum(r['up_steps'] for r in ok),
           'spearman_mean': round(float(np.mean([r['spearman_step_h'] for r in ok if r['spearman_step_h'] is not None])), 3)
           if any(r['spearman_step_h'] is not None for r in ok) else None,
           'start_matches_corpus': sum(1 for r in ok if r['start_matches_corpus'])}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1)
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'pathcheck':
        out = os.path.join(HERE, 'research', 'goal1167_pathcheck.json')
        r = pathcheck(out=out)
        print(f"h along the expert path, {r['levels']} levels ({r['errors']} errors; start board == corpus on {r['start_matches_corpus']}): "
              f"falls start -> before the completing move on {r['h_falls_start_to_before_last']}, rises on {r['h_rises']}, flat on "
              f"{r['h_flat']}; steps down {r['down_steps']} vs up {r['up_steps']}; mean Spearman(step, h) {r['spearman_mean']}")
        return 0
    if not argv or argv[0] != 'study':
        print(__doc__); return 0
    out = os.path.join(HERE, 'research', 'goal1167_precision.json'); npz = None
    if '--out' in argv:
        out = argv[argv.index('--out') + 1]
    if '--npz' in argv:
        npz = argv[argv.index('--npz') + 1]
    r = study(npz, out)
    print(f"structured goal, {r['levels']} levels (L >= 2) of {len(r['per_game'])} games:")
    for m in ('all', 'majority'):
        print(f"  mode {m:8s}: precision pooled {r[f'{m}_precision_pooled']} (mean over levels {r[f'{m}_precision_mean']}, "
              f"{r[f'{m}_levels_with_predictions']} levels with a prediction), recall {r[f'{m}_recall_mean']}, "
              f"h = 0 at the finished board on {r[f'{m}_h_zero_at_finish']} levels")
    print(f"  baselines: every applicable predicate of the predicted kinds {r['base_precision_pooled']} ({r['base_predicted']}); "
          f"the previous level's predicates {r['last_precision_pooled']} ({r['last_predicted']}); the model without 'moved' "
          f"{r['nomoved_precision_pooled']} ({r['nomoved_predicted']})")
    print(f"  FrameLab yardstick: WHERE {r['where_precision_mean']} (random {r['where_random_mean']}; FrameLab within 0.534), "
          f"WHAT {r['what_accuracy_mean']} (FrameLab within 0.494)")
    print(f"  per kind: {r['per_kind']}")
    print(f"  games passing the {GATE} gate: {r['games_passing_gate']}  | overall gate passed: {r['gate_passed_overall']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
