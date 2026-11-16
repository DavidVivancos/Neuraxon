"""frontier1167.py -- NAS 1167 change 6: FRONTIER STEERING. Remember the boards of the play, return to the least-explored one.

WHY (nas_1166_report.md §16, §19d, §23.6). The keys games cleared 0 / 120 at 20 x human with 16-99% of steps on new boards:
the brains wander and never return. su15 L1 was cleared 0 of 1,520 away plays. FrameLab 4's frames-only agent, which keeps a
graph of the states it has seen and RESETs to its frontier, clears su15 L1 (3 / 3 at 1.7 x human), g50t L1 (3.04 x), ls20 L1
(15 x), sc25 L1-L2 and tu93 L2-L9. The winners of both ARC milestones share the same two things: memory and directed
exploration. This is their analogue that a Neuraxon brain can carry: no search engine, no copies of the game, only the
boards the brain has seen.

WHAT IT DOES (FrameLab v3's graph, mask and routing, ported; fl_agent3.py):
  * a GRAPH of the play's boards on the current level: key = a hash of the board with the clock cells and the ticking
    border lines masked (learned on the first 8-32 transitions, then frozen; the graph is rebuilt from the stored transitions
    whenever the mask changes); edges state --action--> state, the most common outcome kept; an edge that ended the game is
    never routed through;
  * a STALL: the brain's own proposals have reached no new board for frontier_stall steps (gene, 8-32);
  * then it ROUTES: the shortest known action path to the best FRONTIER board (a board with untried actions: the keys the game
    offers + one click per object, at the object's centre; more objects and a 4-pixel grid once those are spent), ranked
    least-visited first, nearest first; when no frontier is reachable from here but one is from the level's start, it
    RESETs first (never two RESETs in a row and never before an action on the level: the engine would restart the GAME);
  * at the frontier it HANDS CONTROL BACK: the brain proposes; if its proposal was already tried there, the frontier takes ONE
    untried action instead (the probe, gene frontier_probe, default on), then the brain plays on;
  * the brain is asked every step (play_step always runs: its network keeps perceiving and learning); while routing, the
    executed action replaces the proposal, and the brain's, the steering's and the capability's bookkeeping follow the
    executed action (as the 1160 steering override does).
GENES: steer_frontier (0 / 1) | frontier_stall (8 / 12 / 16 / 24 / 32, default 16) | frontier_probe (0 / 1, default 1).
MULTINXON_FRONTIER=1 forces it on (the keys-frontier cell, the lab's on arms); MULTINXON_FRONTIER=0 forces it off.
TALLY (per try): st_fr_route, st_fr_reset, st_fr_step, st_fr_probe, st_fr_off, st_fr_states, st_fr_stall, st_fr_none."""
from __future__ import annotations

import collections
import hashlib
import math
import os

import numpy as np

KEYS = (1, 2, 3, 4, 5, 7)


def _gene(params, name, default):
    v = getattr(params, name, None)
    return default if v in (None, '') else v


def _val(a):
    try:
        return int(getattr(a, 'value', a))
    except Exception:
        return 0


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def _grid(frame):
    fr = getattr(frame, 'frame', None)
    if not fr:
        return None
    try:
        a = np.asarray(fr[-1], dtype=np.int16)
        return a if a.ndim == 2 else None
    except Exception:
        return None


def _label(m):
    try:
        from scipy import ndimage as _ndi
        return _ndi.label(m)
    except Exception:                                       # pragma: no cover
        import goalmap1163 as GM
        return GM._label(m)


def comps(f):
    """every same-colour 4-component except the background: (y, x) of the cell nearest its centre, colour, size"""
    vals, cnt = np.unique(f, return_counts=True); bg = int(vals[np.argmax(cnt)])
    out = []
    for c in vals:
        c = int(c)
        if c == bg or c < 0:
            continue
        lab, n = _label(f == c)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if not len(ys):
                continue
            cy, cx = ys.mean(), xs.mean(); j = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            out.append((int(ys[j]), int(xs[j]), c, int(len(ys))))
    return out, bg


class Frontier:
    def __init__(self, stall=16, probe=1, clock_frac=0.7, line_frac=0.5, mask_after=32, max_clicks=48):
        self.stall_n = max(2, int(stall)); self.probe_on = int(probe)
        self.clock_frac, self.line_frac, self.mask_after, self.max_clicks = clock_frac, line_frac, mask_after, max_clicks
        self.chg = None; self.n_tr = 0; self.frozen = False; self.mask = None
        self.cnt = collections.Counter()
        self.level = None; self.prev = None; self.acted = 0; self.last_reset = True
        self.plan = []; self.expect = None; self.probe_here = None; self.since_new = 0
        self._new_level(None)

    # ------------------------------------------------------------------ bookkeeping
    def _t(self, k, n=1):
        self.cnt[k] += n; _tally('fr_' + k, n)

    def _new_level(self, f):
        self.FR = []; self.fid = {}; self.T = []; self.start_idx = set(); self.dead_idx = set()
        self.kprior = collections.Counter(); self.kseen = collections.Counter()
        self.plan = []; self.expect = None; self.probe_here = None; self.since_new = 0; self.prev = None
        self._rebuild()
        if f is not None:
            self._start(f)

    def _fidx(self, f):
        b = np.ascontiguousarray(f.astype(np.int16)).tobytes()
        i = self.fid.get(b)
        if i is None:
            i = len(self.FR); self.fid[b] = i; self.FR.append(f.astype(np.int16).copy())
        return i

    def key(self, f):
        g = f if self.mask is None or self.mask.shape != f.shape else np.where(self.mask, -1, f)
        return hashlib.blake2b(np.ascontiguousarray(g.astype(np.int16)).tobytes(), digest_size=12).digest()

    def _rebuild(self):
        self.edges = collections.defaultdict(dict); self.eout = collections.defaultdict(collections.Counter)
        self.tried = collections.defaultdict(set); self.visits = collections.Counter(); self.bad = set()
        self.frames = {}; self.tier = {}; self.ccache = {}
        keys = [self.key(F) for F in self.FR]
        for i, k in enumerate(keys):
            self.frames.setdefault(k, self.FR[i])
        self.start_keys = {keys[i] for i in self.start_idx}
        for k in self.start_keys:
            self.visits[k] += 1
        normal = set(self.start_keys) | {keys[j] for (i, a, j, ov) in self.T if not ov}
        self.dead = {keys[j] for j in self.dead_idx} - normal
        overs = collections.Counter()
        for (i, a, j, ov) in self.T:
            self._edge(keys[i], a, keys[j])
            if ov:
                overs[(keys[i], a)] += 1
                if keys[j] in self.dead or overs[(keys[i], a)] >= 2:
                    self.bad.add((keys[i], a))

    def _edge(self, k, a, k2):
        self.tried[k].add(a); self.visits[k2] += 1
        c = self.eout[(k, a)]; c[k2] += 1
        top, n = c.most_common(1)[0]
        self.edges[k][a] = top
        if len(c) > 1 and n < 0.6 * sum(c.values()):
            self.bad.add((k, a))                                          # a varying outcome: not a route

    def _start(self, f):
        i = self._fidx(f); self.start_idx.add(i); k = self.key(f)
        self.frames.setdefault(k, self.FR[i]); self.start_keys.add(k); self.visits[k] += 1
        return k

    def _mask_update(self, f0, f1):
        """FrameLab v3: cells changing on >= 70% of transitions + border lines (4 px) changing on >= 50%, or sparsely on
        >= 20% (a step bar); learned on the first transitions, frozen after mask_after"""
        if self.frozen or f0.shape != f1.shape:
            return False
        d = f0 != f1
        if self.chg is None or self.chg.shape != d.shape:
            H, W = d.shape
            self.chg = np.zeros(d.shape, np.int32); self.rows = np.zeros(H, np.int32); self.cols = np.zeros(W, np.int32)
            self.rowc = np.zeros(H, np.int32); self.colc = np.zeros(W, np.int32)
        self.chg += d; self.n_tr += 1
        self.rows += d.any(1); self.cols += d.any(0); self.rowc += d.sum(1); self.colc += d.sum(0)
        if self.n_tr < 8:
            return False
        n = self.n_tr; m = self.chg >= self.clock_frac * n; H, W = m.shape
        for y in list(range(0, 4)) + list(range(H - 4, H)):
            if self.rows[y] >= self.line_frac * n or (self.rows[y] >= 0.2 * n and self.rowc[y] <= 4 * self.rows[y]):
                m[y, :] = True
        for x in list(range(0, 4)) + list(range(W - 4, W)):
            if self.cols[x] >= self.line_frac * n or (self.cols[x] >= 0.2 * n and self.colc[x] <= 4 * self.cols[x]):
                m[:, x] = True
        if self.n_tr >= self.mask_after:
            self.frozen = True
        if not m.any():
            m = None
        old = self.mask; self.mask = m
        if (old is None) != (m is None):
            return True
        return old is not None and not np.array_equal(old, m)

    # ------------------------------------------------------------------ candidates
    @staticmethod
    def akey_of(a, d):
        v = _val(a)
        if v == 0:
            return 'RESET'
        if v == 6:
            return ('C', int((d or {}).get('y', 0)), int((d or {}).get('x', 0)))
        return ('K', v)

    def _comps(self, k, f):
        if k not in self.ccache:
            self.ccache[k] = comps(f)
        return self.ccache[k]

    def canon(self, k, f, ak):
        """a click on an object counts as a click on its centre (the brain's points differ by a pixel; the object is what
        was tried)"""
        if ak == 'RESET' or ak[0] != 'C':
            return ak
        y, x = ak[1], ak[2]
        if not (0 <= y < f.shape[0] and 0 <= x < f.shape[1]):
            return ak
        C, bg = self._comps(k, f)
        col = int(f[y, x])
        if col == bg:
            return ak
        best = None
        for (cy, cx, c, n) in C:
            if c == col:
                dd = abs(cy - y) + abs(cx - x)
                if best is None or dd < best[0]:
                    best = (dd, cy, cx)
        return ('C', best[1], best[2]) if best else ak

    def kind(self, f, ak):
        if ak[0] == 'C':
            return ('C', int(f[ak[1], ak[2]]))
        return ak

    def prior(self, kd):
        t, c = self.kseen[kd], self.kprior[kd]
        return (c + 1.0) / (t + 2.0)

    def candidates(self, k, f, avail, tier):
        out = [('K', v) for v in KEYS if v in avail]
        if 6 not in avail:
            return out
        if tier >= 2:
            return out + [('C', y, x) for y in range(2, f.shape[0], 4) for x in range(2, f.shape[1], 4)]
        C, bg = self._comps(k, f)
        C = sorted(C, key=lambda d: (-self.prior(('C', d[2])), d[3]))
        if tier == 0:
            C = C[:self.max_clicks]
        seen = set(); res = []
        for (y, x, c, n) in C:
            if (y, x) not in seen:
                seen.add((y, x)); res.append(('C', y, x))
        return out + res

    def open_actions(self, k, avail=None):
        f = self.frames.get(k)
        if f is None or k in self.dead:
            return []
        av = avail if avail is not None else self.avail
        t = self.tier.get(k, 0)
        while t <= (2 if 6 in av else 0):
            c = [a for a in self.candidates(k, f, av, t) if a not in self.tried[k]]
            if c:
                self.tier[k] = t
                return c
            t += 1
        self.tier[k] = 99
        return []

    # ------------------------------------------------------------------ routing
    def _bfs(self, k0):
        prev = {k0: None}; dist = {k0: 0}; q = collections.deque([k0]); order = []
        while q and len(prev) < 6000:
            k = q.popleft(); order.append(k)
            for a, k2 in self.edges.get(k, {}).items():
                if (k, a) in self.bad or k2 in prev or k2 in self.dead:
                    continue
                prev[k2] = (k, a); dist[k2] = dist[k] + 1; q.append(k2)
        return prev, dist, order

    @staticmethod
    def _path(prev, k):
        p = []
        while prev[k] is not None:
            k, a = prev[k]; p.append(a)
        return p[::-1]

    def route(self, k0):
        """the shortest known path from k0 to the best frontier: least visited, then nearest (FrameLab's frontier value
        without its learned goal) -> (path, frontier key) or (None, None)"""
        prev, dist, order = self._bfs(k0)
        fronts = []
        for k in order[1:]:
            if k in self.frames and self.open_actions(k):
                fronts.append(k)
                if len(fronts) >= 48:
                    break
        if not fronts:
            return None, None
        k = min(fronts, key=lambda x: (math.log1p(self.visits[x]) + 0.1 * dist[x], dist[x]))
        return self._path(prev, k), k

    # ------------------------------------------------------------------ the step
    def observe(self, f, lv, over=False):
        """learn the transition of the last executed action into board f (level lv)"""
        if self.prev is None:
            return
        i0, ak, lv0 = self.prev; self.prev = None
        if ak == 'RESET' or lv != lv0:
            return
        f0 = self.FR[i0]
        k0 = self.key(f0)
        j = self._fidx(f)
        if over:
            self.dead_idx.add(j)
        ch = bool(((f0 != f) if self.mask is None or self.mask.shape != f.shape else ((f0 != f) & ~self.mask)).any()) if f0.shape == f.shape else True
        kd = self.kind(f0, ak); self.kseen[kd] += 1; self.kprior[kd] += int(ch)
        self.T.append((i0, ak, j, bool(over)))
        if self._mask_update(f0, f) if f0.shape == f.shape else False:
            self._rebuild(); self._t('mask')
        else:
            k2 = self.key(f); self.frames.setdefault(k2, self.FR[j]); self._edge(k0, ak, k2)
            if over:
                self.bad.add((k0, ak))

    def on_reset(self, frame, by_us):
        """a reset happened: the brain's own (RESET returned by us or by the brain) or the loop's (a game over)"""
        f = _grid(frame)
        lv = int(getattr(frame, 'levels_completed', 0) or 0)
        if self.prev is not None and not by_us:                                # the loop reset after our last action: it
            i0, ak, lv0 = self.prev                                            # ended the game -- never route through it
            k0 = self.key(self.FR[i0]); self.bad.add((k0, ak)); self.tried[k0].add(ak); self._t('death')
        self.prev = None
        if f is None:
            return
        if self.level is None or lv != self.level:
            self.level = lv; self._new_level(f)
        else:
            self._start(f)
        self.acted = 0; self.last_reset = True
        if self.plan and self.plan[0] == 'RESET':
            self.plan.pop(0)
        if self.plan:
            self.expect = self.key(f)

    def decide(self, frame, a, d, avail_vals):
        """-> None (the brain's action stands) or the akey to execute instead"""
        f = _grid(frame)
        if f is None:
            return None
        lv = int(getattr(frame, 'levels_completed', 0) or 0)
        self.avail = set(avail_vals)
        if self.level is None or lv != self.level:
            self.prev = None
            self.level = lv; self._new_level(f); self.acted = 0; self.last_reset = True
            new = False
        else:
            k_pre = self.key(f)                                                # novelty BEFORE this transition is recorded
            new = self.visits.get(k_pre, 0) == 0 and k_pre not in self.start_keys
            self.observe(f, lv)
        k = self.key(f)
        if k not in self.frames:
            self.frames[k] = self.FR[self._fidx(f)]
        bkey = self.akey_of(a, d)
        choice = None
        if self.plan:                                                          # routing: stay on the route or drop it
            if self.expect is not None and k != self.expect:
                self.plan = []; self.expect = None; self._t('off')
            else:
                nxt = self.plan.pop(0)
                if nxt == 'RESET':
                    if self.acted > 0 and not self.last_reset:
                        choice = 'RESET'; self._t('reset')
                    else:
                        self.plan = []
                else:
                    choice = nxt; self._t('step')
                    self.expect = self.edges.get(k, {}).get(nxt)
                    if not self.plan:
                        self.probe_here = self.expect                          # the frontier: the brain proposes there first
        elif self.probe_here is not None and k == self.probe_here:            # at the frontier: the brain's proposal, or a probe
            self.probe_here = None
            cb = self.canon(k, f, bkey) if bkey != 'RESET' else bkey
            if self.probe_on and (bkey == 'RESET' or cb in self.tried[k]):
                oa = self.open_actions(k)
                if oa:
                    choice = oa[0]; self._t('probe')
            self.since_new = 0
        else:
            self.probe_here = None
            self.since_new = 0 if new else self.since_new + 1
            if self.since_new >= self.stall_n:
                self._t('stall')
                oa = self.open_actions(k)
                path = None
                if not oa:
                    path, fk = self.route(k)
                if oa:                                                         # untried actions HERE: the brain's, or the probe
                    cb = self.canon(k, f, bkey) if bkey != 'RESET' else bkey
                    if self.probe_on and (bkey == 'RESET' or cb in self.tried[k]):
                        choice = oa[0]; self._t('probe')
                elif path:                                                     # this board is spent: the nearest least-visited frontier
                    self.plan = list(path); self.expect = None; self._t('route')
                    choice = self.plan.pop(0); self.expect = self.edges.get(k, {}).get(choice); self._t('step')
                    if not self.plan:
                        self.probe_here = self.expect
                else:
                    sk = next((s for s in self.start_keys if s != k and (self.open_actions(s) or self.route(s)[0])), None)
                    if sk is not None and self.acted > 0 and not self.last_reset:
                        p2 = [] if self.open_actions(sk) else (self.route(sk)[0] or [])
                        self.plan = ['RESET'] + list(p2); choice = self.plan.pop(0); self._t('route'); self._t('reset')
                        if not self.plan:
                            self.probe_here = sk
                    else:
                        self._t('none')
                self.since_new = 0
        ex = choice if choice is not None else bkey
        if ex == 'RESET':
            self.prev = None; self.last_reset = True
        else:
            cx = self.canon(k, f, ex)
            self.prev = (self._fidx(f), cx, lv); self.acted += 1; self.last_reset = False
        self.cnt['states'] = len(self.frames)
        return choice


def to_action(akey):
    from arc_env.structs import GameAction
    if akey == 'RESET':
        return GameAction.RESET, {}
    if akey[0] == 'K':
        return GameAction(int(akey[1])), {}
    return GameAction.ACTION6, {'x': int(akey[2]), 'y': int(akey[1])}


def frontier_on(params):
    e = os.environ.get('MULTINXON_FRONTIER', '')
    if e == '0':
        return False
    if e == '1':
        return True
    try:
        return int(float(_gene(params, 'steer_frontier', 0) or 0)) == 1
    except Exception:
        return False


# ======================================================================================================== the brain hooks
def install(ArcBrain):
    """wrap ArcBrain.__init__, play_step, reset, soft_reset (outermost); idempotent per class object; inert unless on"""
    if getattr(ArcBrain, '_v1167_frontier', False):
        return True
    ArcBrain._v1167_frontier = True
    _init0 = ArcBrain.__init__; _ps0 = ArcBrain.play_step; _rs0 = ArcBrain.reset; _srs0 = ArcBrain.soft_reset

    def __init__(self, params, *a, **k):
        _init0(self, params, *a, **k)
        self._fr = None
        if frontier_on(params):
            try:
                st = int(float(os.environ.get('MULTINXON_PIN_FRONTIER_STALL', '') or _gene(params, 'frontier_stall', 16) or 16))
            except Exception:
                st = 16
            pr = int(float(_gene(params, 'frontier_probe', 1) if _gene(params, 'frontier_probe', 1) is not None else 1))
            self._fr = Frontier(stall=st, probe=pr)
        self._fr_ours = False

    def play_step(self, frame):
        a, d = _ps0(self, frame)                                              # the brain (and its steering) always runs
        fr = getattr(self, '_fr', None)
        if fr is None:
            return a, d
        try:
            av = [_val(x) for x in (getattr(frame, 'available_actions', None) or []) if _val(x) != 0]
            ch = fr.decide(frame, a, d, av)
        except Exception as e:
            self._fr_error = f'{type(e).__name__}: {e}'[:200]
            return a, d
        if ch is None:
            self._fr_ours = _val(a) == 0
            return a, d
        a2, d2 = to_action(ch)
        self._fr_ours = ch == 'RESET'
        _follow(self, frame, a2, d2)
        return a2, d2

    def reset(self, frame):
        out = _rs0(self, frame)
        fr = getattr(self, '_fr', None)
        if fr is not None:
            try:
                fr.on_reset(frame, bool(getattr(self, '_fr_ours', False)))
            except Exception as e:
                self._fr_error = f'reset: {type(e).__name__}: {e}'[:200]
        self._fr_ours = False
        return out

    def soft_reset(self, frame):
        out = _srs0(self, frame)
        fr = getattr(self, '_fr', None)
        if fr is not None:
            try:
                fr.on_reset(frame, bool(getattr(self, '_fr_ours', False)))
            except Exception as e:
                self._fr_error = f'soft_reset: {type(e).__name__}: {e}'[:200]
        self._fr_ours = False
        return out

    for f in (__init__, play_step, reset, soft_reset):
        f._v1167 = True
    ArcBrain.__init__ = __init__; ArcBrain.play_step = play_step; ArcBrain.reset = reset; ArcBrain.soft_reset = soft_reset
    return True


def _follow(brain, frame, a2, d2):
    """the executed action replaces the proposal everywhere the brain keeps books (the 1160 override's rule), the steering's
    record of its last step and the capability's live-set record included"""
    v = _val(a2)
    brain.last_action = a2
    try:
        brain._eff_prev_idx = v - 1 if 1 <= v <= 6 else -1
        if v == 6 and d2:
            brain._last_click_cand = (int(d2.get('y', 0)), int(d2.get('x', 0)))
    except Exception:
        pass
    st = getattr(brain, '_steer', None)
    if st is not None:
        try:
            if v == 0:
                st.prev = None
            elif st.prev is not None:
                k, _, _, g, lv = st.prev
                nd = st.graph.get(k) or {'cands': []}
                akey, cls = st.brain_key(g, a2, d2, nd)
                st.prev = (k, akey, cls, g, lv)
            if hasattr(st, 'pred'):
                st.pred = None
            if hasattr(st, '_from_brain'):
                st._from_brain = False
        except Exception:
            pass
    if getattr(brain, '_cap_on', False):
        try:
            g = _grid(frame)
            lv = int(getattr(frame, 'levels_completed', 0) or 0)
            brain._cap_prev = (g.tolist() if g is not None else None, v, lv) if v > 0 else None
        except Exception:
            pass
