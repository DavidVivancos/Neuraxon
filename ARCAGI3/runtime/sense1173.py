"""sense1173.py -- NAS 1173: GAME SENSE. Which game is this, from its pixels and from what a couple of actions do to them?

The user's goal for the NAS (21 Oct 2026): every brain can play every game through the same input / output setup, and a brain should be
able to DETECT the game it is in from the board alone, or with a couple of exploration actions -- not from the list of actions the game
offers, but from the pixels that make the board and the feedback the board gives when an action is made. This module is the measurable
core of that, and the steering's gene game_sense (effect1173.py) uses it.

WHAT IT COMPUTES (numpy only; no game knowledge, no engine during play):
  board_desc(grid)    42 numbers from the pixels: the colour histogram (16), the number of objects of each colour (16; 4-connected
                      components, log scale), and the layout -- the share of the largest object, the number of colours and objects, the
                      left-right and up-down symmetry, the border's fill and colours, the board's size.
  effect_desc(g0, g1) 39 numbers from what an action did: the share of the board that changed, the colours the changed cells had before
                      (16) and after (16), where the change is (centroid, extent) and how many pieces it has.
  Index               every known level of the 25 games (the boards at its start, at a quarter, half and three quarters of its verified
                      moves and just before its end) and, per game and kind of action (each key, a click), how often a GENERIC action
                      changes the board (the build's random probes) and what the changes look like (the probes' and the verified moves'
                      effect descriptors). Built from the verified solutions on the real engine by build(); shipped as
                      sense_index_1173.json. classify() finds the nearest known game: the board alone, or with the feedback.
  lolo() / live()     the measures, every level held out in turn (its boards and its effects left out; its game's OTHER levels stay
                      in): lolo with the feedback of the level's first two verified moves; live with two GENERIC probes (a random
                      key or a click on a random object -- what any brain can do first) made on the real engine at the level start.

THE GENE (effect1173.py): game_sense 1 -- at a level's first frame the steering asks the Index which game this looks like; after its own
first two actions in the level it asks again with their feedback; when the answer is a game other than the brain's home game it plays with
that game's priors (game_priors_1173.json: the book champion's click band, candidate tier and key sweep). A brain at home keeps its own
genes. Inert at 0.

    python3 sense1173.py build [--out sense_index_1173.json]     the index (the real engine; ~3 min) + both measures
    python3 sense1173.py lolo  [--index sense_index_1173.json]   the leave-one-level-out accuracies again (live from its cache)"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NC = 16
BOARD_DIM = 42
FX_DIM = 39
INDEX_FILE = 'sense_index_1173.json'
FXV_MAX = 24             # effect descriptors kept per game, action kind and source (sampled evenly)
PROBE_SEEDS = 3          # build: generic probes at every known level start, seeds 101.. (the live measure uses 0..2)
PROBE_N = 3              # build: probes per level start and seed
LIVE_SEEDS = (0, 1, 2)
W_FX = 0.0               # weight of the 'did the board change' log-likelihood (pooled counts, clipped). 0: WHETHER an action changes
                         # the board depends on who acts (the expert, a random probe, a brain) -- the build's scan of 16 weight pairs
                         # (1173.3): any w_fx > 0 lowered the live (generic-probe) accuracy (0.25: 96.5 -> 95.8%, 1.0: 86.4%)
W_EFX = 8.0              # weight of the nearest effect descriptor of the same action kind: WHAT changed (the colours before and
                         # after, where, how much) -- 0 / 2 / 4 / 8: live 93.8 / 95.8 / 96.8 / 97.1%, held-out verified 93.8 -> 97.1%
P_CLIP = (0.1, 0.9)      # P(change | kind, game) is clipped: a brain's own actions are neither the expert's nor uniform
MID_FRACS = (0.25, 0.5, 0.75, 1.0)   # boards indexed inside each level (fractions of its verified moves; 1.0 = just before its end)
PRIORS_FILE = 'game_priors_1173.json'

try:
    from scipy.ndimage import label as _label
except Exception:                                                   # pragma: no cover
    _label = None


def _lab(m):
    if _label is not None:
        return _label(m)
    # a tiny BFS labeller (4-connected) if scipy is missing
    H, W = m.shape; lab = np.zeros((H, W), dtype=np.int32); n = 0
    for y in range(H):
        for x in range(W):
            if m[y, x] and not lab[y, x]:
                n += 1; st = [(y, x)]; lab[y, x] = n
                while st:
                    a, b = st.pop()
                    for c, d in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= c < H and 0 <= d < W and m[c, d] and not lab[c, d]:
                            lab[c, d] = n; st.append((c, d))
    return lab, n


def _grid(g):
    a = np.asarray(g)
    if a.ndim == 3:
        a = a[-1]
    return np.clip(a.astype(np.int16), 0, NC - 1)


# ======================================================================================================== descriptors
def board_desc(g):
    """42 numbers from the pixels of a board (see the module docstring)"""
    g = _grid(g); H, W = g.shape; N = float(g.size)
    hist = np.bincount(g.ravel(), minlength=NC)[:NC] / N
    bg = int(np.argmax(hist))
    objs = np.zeros(NC); sizes = []
    for c in range(NC):
        if c == bg or hist[c] <= 0:
            continue
        lab, n = _lab(g == c)
        objs[c] = math.log1p(n) / 4.0
        if n:
            s = np.bincount(lab.ravel())[1:]
            sizes.extend(int(x) for x in s)
    sizes = sorted(sizes)
    n_obj = len(sizes)
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    lay = [math.log1p(n_obj) / 5.0, (sizes[-1] / N) if sizes else 0.0, (math.log1p(sizes[len(sizes) // 2]) / 6.0) if sizes else 0.0,
           float((hist > 0).sum()) / NC, float(np.mean(g == g[:, ::-1])), float(np.mean(g == g[::-1, :])),
           float(np.mean(border != bg)), float(len(np.unique(border))) / NC, H / 64.0, W / 64.0]
    return np.concatenate([hist, objs, np.asarray(lay, dtype=float)])


def effect_desc(g0, g1):
    """39 numbers from what an action did to the board (all zero but the last for no change)"""
    g0 = _grid(g0); g1 = _grid(g1)
    if g0.shape != g1.shape:
        out = np.zeros(FX_DIM); out[0] = 1.0; out[-1] = 1.0
        return out
    d = g0 != g1; N = float(g0.size)
    if not d.any():
        return np.zeros(FX_DIM)
    H, W = g0.shape
    before = np.bincount(g0[d], minlength=NC)[:NC] / float(d.sum())
    after = np.bincount(g1[d], minlength=NC)[:NC] / float(d.sum())
    ys, xs = np.nonzero(d)
    _, n = _lab(d)
    geo = [float(d.mean()), float(ys.mean()) / H, float(xs.mean()) / W, float(ys.max() - ys.min() + 1) / H, float(xs.max() - xs.min() + 1) / W,
           math.log1p(n) / 4.0]
    return np.concatenate([before, after, np.asarray(geo, dtype=float), [1.0]])


def action_kind(a):
    """'K1'..'K7' for a key, 'C' for a click (ACTION6), None for a reset"""
    s = str(getattr(a, 'name', a))
    if s.startswith('ACTION6') or s == 'C' or (isinstance(a, tuple) and a and a[0] == 'C'):
        return 'C'
    if isinstance(a, tuple) and a and a[0] == 'K':
        return f'K{int(a[1])}'
    if s.startswith('ACTION'):
        try:
            return f'K{int(s[6:7])}'
        except Exception:
            return None
    try:
        v = int(getattr(a, 'value', a))
        return 'C' if v == 6 else (f'K{v}' if 1 <= v <= 7 else None)
    except Exception:
        return None


# ======================================================================================================== the index
class Index:
    """the known games: boards of every known level (start + inside) and the feedback of actions, standardised; nearest game"""

    def __init__(self, doc):
        self.doc = doc
        self.entries = doc.get('entries') or []
        self.B = np.asarray([e['board'] for e in self.entries], dtype=float) if self.entries else np.zeros((0, BOARD_DIM))
        self.mu = np.asarray(doc['mu'], dtype=float) if doc.get('mu') is not None else (self.B.mean(0) if len(self.B) else np.zeros(BOARD_DIM))
        sd = np.asarray(doc['sd'], dtype=float) if doc.get('sd') is not None else (self.B.std(0) if len(self.B) else np.ones(BOARD_DIM))
        self.sd = np.where(sd > 1e-6, sd, 1.0)
        self.Z = (self.B - self.mu) / self.sd if len(self.B) else self.B
        self.games = [e['game'] for e in self.entries]
        self.levels = np.asarray([int(e['L']) for e in self.entries], dtype=int) if self.entries else np.zeros(0, dtype=int)
        self.cnt = {}                                    # game -> kind -> [changed, n]  (verified moves + generic probes)
        for src in ('game_fx', 'probe_fx'):
            for g, kv in (doc.get(src) or {}).items():
                for k, c in kv.items():
                    a = self.cnt.setdefault(g, {}).setdefault(k, [0, 0]); a[0] += int(c[0]); a[1] += int(c[1])
        self.fxv = {}                                    # game -> kind -> array [[L, fx...]]
        for src in ('game_fxv', 'probe_fxv'):
            for g, kv in (doc.get(src) or {}).items():
                for k, v in kv.items():
                    if not v:
                        continue
                    a = np.asarray(v, dtype=float); d = self.fxv.setdefault(g, {})
                    d[k] = a if k not in d else np.concatenate([d[k], a])
        self.lcnt = {}                                   # (game, L) -> kind -> [changed, n]: the held-out level's own share
        for g, kv in (doc.get('level_fx') or {}).items():
            for Lk, c in kv.items():
                L, k = Lk.split(':')
                self.lcnt.setdefault((g, int(L)), {})[k] = [int(c[0]), int(c[1])]

    @classmethod
    def load(cls, path=None):
        p = path or os.path.join(HERE, INDEX_FILE)
        return cls(json.load(open(p, encoding='utf-8')))

    def board_dist(self, desc, exclude=None):
        z = (np.asarray(desc, dtype=float) - self.mu) / self.sd
        d = ((self.Z - z) ** 2).mean(1)
        if exclude is not None:
            g, L = exclude
            m = np.asarray([x == g for x in self.games]) & (self.levels == int(L))
            d = np.where(m, np.inf, d)
        return d

    def p_change(self, game, kind, exclude_L=None):
        c = (self.cnt.get(game) or {}).get(kind)
        if c is None:
            return None
        ch, n = c
        if exclude_L is not None:
            lc = (self.lcnt.get((game, int(exclude_L))) or {}).get(kind)
            if lc:
                ch -= lc[0]; n -= lc[1]
        if n <= 0:
            return None
        p = (ch + 1.0) / (n + 2.0)
        return min(P_CLIP[1], max(P_CLIP[0], p))

    def fx_loglik(self, game, obs, exclude_L=None):
        """log P(the observed changes / no changes | game), per kind of action (pooled counts, clipped); a kind the game was
        never seen to use costs log 0.3"""
        ll = 0.0
        for o in obs:
            kind, changed = o[0], o[1]
            if kind is None:
                continue
            p = self.p_change(game, kind, exclude_L)
            if p is None:
                ll += math.log(0.3)
                continue
            ll += math.log(p if changed else 1.0 - p)
        return ll

    def fx_near(self, game, obs, exclude_L=None):
        """minus the mean nearest same-kind effect distance over the observed CHANGES (0 when nothing changed)"""
        st = self.fxv.get(game) or {}
        tot = 0.0; n = 0
        for o in obs:
            kind, changed = o[0], o[1]
            fxd = o[2] if len(o) > 2 else None
            if kind is None or not changed or fxd is None:
                continue
            M = st.get(kind)
            if M is not None and exclude_L is not None:
                M = M[M[:, 0] != exclude_L]
            if M is None or not len(M):
                tot += 0.25; n += 1
                continue
            dd = ((M[:, 1:] - np.asarray(fxd, dtype=float)[None, :]) ** 2).mean(1)
            tot += float(dd.min()); n += 1
        return -tot / n if n else 0.0

    def classify(self, board, obs=None, exclude=None, w_fx=W_FX, w_efx=W_EFX):
        """(game, margin, ranking [(game, score)]): the nearest game by its nearest indexed board; with obs (a list of (kind, changed[,
        effect descriptor]) of the actions made so far) the feedback joins the score. exclude = (game, L): that level held out"""
        if not len(self.entries):
            return None, 0.0, []
        d = self.board_dist(board, exclude)
        best = {}
        for i, g in enumerate(self.games):
            if d[i] < best.get(g, np.inf):
                best[g] = float(d[i])
        sc = {g: -v for g, v in best.items() if np.isfinite(v)}
        if obs:
            for g in list(sc):
                exL = int(exclude[1]) if (exclude is not None and exclude[0] == g) else None
                if w_fx:
                    sc[g] += w_fx * self.fx_loglik(g, obs, exL)
                if w_efx:
                    sc[g] += w_efx * self.fx_near(g, obs, exL)
        rank = sorted(sc.items(), key=lambda x: -x[1])
        if not rank:
            return None, 0.0, []
        margin = rank[0][1] - rank[1][1] if len(rank) > 1 else float('inf')
        return rank[0][0], margin, rank


def _obs_of(e):
    fxs = e.get('fx') or []
    return [(k, bool(c), (fxs[i] if i < len(fxs) else None)) for i, (k, c) in enumerate(zip(e.get('kinds') or [], e.get('changed') or []))]


def evaluate(index, records, w_fx=W_FX, w_efx=W_EFX):
    """records [(game, L, board desc, obs)]: accuracy with the board alone and with the feedback, each level held out"""
    n = ok_b = ok_f = 0; per = {}; margins = []
    for g, L, bd, obs in records:
        gb, mb, _ = index.classify(bd, exclude=(g, L))
        gf, mf, _ = index.classify(bd, obs=obs, exclude=(g, L), w_fx=w_fx, w_efx=w_efx) if obs else (gb, mb, [])
        n += 1; ok_b += int(gb == g); ok_f += int(gf == g); margins.append((round(float(mb), 3), int(gb == g)))
        p = per.setdefault(g, [0, 0, 0]); p[0] += 1; p[1] += int(gb == g); p[2] += int(gf == g)
    return {'n': n, 'board_only': round(ok_b / max(1, n), 3), 'with_feedback': round(ok_f / max(1, n), 3),
            'per_game': {g: {'n': v[0], 'board_only': v[1], 'with_feedback': v[2]} for g, v in sorted(per.items())}, 'margins': margins}


def lolo_records(index, level1=False):
    return [(e['game'], int(e['L']), e['board'], _obs_of(e)) for e in index.entries
            if e.get('kind', 'start') == 'start' and (not level1 or int(e['L']) == 1)]


def sure_table(margins):
    """board-only accuracy at margin >= m (the steering decides from the board alone above SENSE_SURE)"""
    out = {}
    for m in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        sel = [ok for mg, ok in margins if mg >= m]
        out[str(m)] = {'share': round(len(sel) / max(1, len(margins)), 3), 'accuracy': round(sum(sel) / max(1, len(sel)), 3)}
    return out


_IDX = {'idx': None, 'pri': None}


def index_cached():
    if _IDX['idx'] is None:
        try:
            _IDX['idx'] = Index.load()
        except Exception:
            _IDX['idx'] = Index({'entries': []})
    return _IDX['idx']


def priors_cached():
    if _IDX['pri'] is None:
        try:
            _IDX['pri'] = json.load(open(os.path.join(HERE, PRIORS_FILE), encoding='utf-8')).get('games') or {}
        except Exception:
            _IDX['pri'] = {}
    return _IDX['pri']


# ======================================================================================================== build (the real engine)
def _step(env, a):
    from arc_env.structs import GameAction as _GA
    if a == 'RESET':
        return env.reset()
    if str(a).startswith('ACTION6:'):
        x, y = map(int, str(a).split(':')[1].split(','))
        return env.step(_GA.ACTION6, {'x': x, 'y': y})
    return env.step(_GA[str(a)])


def _restart(env):
    """env.reset() restarts the CURRENT level on the real engine; reset again until the game is back at level 1"""
    obs = env.reset(); n = 0
    while int(getattr(obs, 'levels_completed', 0) or 0) > 0 and n < 4:
        obs = env.reset(); n += 1
    return obs


def _to_level(env, P, g, L):
    """the game restarted and the verified prefix replayed to level L's start; None when it does not arrive there"""
    obs = _restart(env)
    for a in (P[g].get(str(L)) or []) if L > 1 else []:
        obs = _step(env, a)
    return obs if int(getattr(obs, 'levels_completed', 0) or 0) == L - 1 else None


def _settled(obs):
    fr = getattr(obs, 'frame', None) or []
    try:
        g = np.asarray(fr[-1], dtype=np.int16) if fr else None
        return g if (g is not None and g.ndim == 2) else None
    except Exception:
        return None


def _probe_actions(env, obs, rng, n=2):
    """n generic exploration actions from the current state: a random available key, or a click on a random cell of a non-background
    colour (a random cell when there is none) -- what any brain can do on its first steps"""
    from arc_env.structs import GameAction as _GA
    out = []
    for _ in range(n):
        g0 = _settled(obs)
        if g0 is None:
            break
        av = [int(getattr(a, 'value', a)) for a in (getattr(obs, 'available_actions', None) or [])]
        keys = [v for v in av if 1 <= v <= 7 and v != 6]
        click = 6 in av
        if keys and (not click or rng.random() < 0.5):
            v = int(rng.choice(keys)); obs2 = env.step(_GA(v)); kind = f'K{v}'
        elif click:
            hist = np.bincount(g0.ravel(), minlength=NC); bg = int(np.argmax(hist))
            ys, xs = np.nonzero(g0 != bg)
            if len(ys):
                i = int(rng.integers(len(ys))); y, x = int(ys[i]), int(xs[i])
            else:
                y, x = int(rng.integers(g0.shape[0])), int(rng.integers(g0.shape[1]))
            obs2 = env.step(_GA.ACTION6, {'x': x, 'y': y}); kind = 'C'
        else:
            break
        g1 = _settled(obs2)
        if g1 is None:
            break
        ch = bool((g0 != g1).any()) if g0.shape == g1.shape else True
        out.append((kind, ch, [round(float(x), 4) for x in effect_desc(g0, g1)] if ch else None))
        obs = obs2
        if str(getattr(obs, 'state', '')).endswith(('GAME_OVER', 'WIN')):
            break
    return out


def _sample(v, n):
    return v if len(v) <= n else [v[int(i * (len(v) - 1) / (n - 1))] for i in range(n)]


def build(out=None, games=None, log=print, live_seeds=LIVE_SEEDS):
    """the index from the verified solutions + the generic probes, then both measures (lolo; live, cached next to the index)"""
    sys.path.insert(0, HERE)
    from arc_env.adapter import make_env
    P = json.load(open(os.path.join(HERE, 'level_prefix.json'), encoding='utf-8'))['prefix']
    entries = []; gfx = {}; gfv = {}; lfx = {}
    for g in sorted(games or P):
        Ls = sorted(int(x) for x in P[g])
        full = P[g][str(max(Ls))]
        env = make_env(g, seed=0, prefer_real=True, force_offline=False)
        obs = _restart(env)
        lv = int(obs.levels_completed or 0)
        boards = {lv + 1: [_settled(obs)]}; moves = {}; fx = {}; fxv = {}
        for a in full:
            pre = _settled(obs)
            obs = _step(env, a)
            post = _settled(obs)
            lv2 = int(obs.levels_completed or 0)
            kind = action_kind(str(a).split(':')[0]) if a != 'RESET' else None
            if kind and pre is not None and post is not None and lv2 == lv:
                ch = bool((pre != post).any()) if pre.shape == post.shape else True
                c = fx.setdefault(kind, [0, 0]); c[0] += int(ch); c[1] += 1
                lc = lfx.setdefault(g, {}).setdefault(f'{lv + 1}:{kind}', [0, 0]); lc[0] += int(ch); lc[1] += 1
                moves.setdefault(lv + 1, []).append((kind, ch, pre, post))
                boards.setdefault(lv + 1, []).append(post)
                if ch:
                    fxv.setdefault(kind, []).append([lv + 1] + [round(float(x), 4) for x in effect_desc(pre, post)])
            if lv2 > lv:
                lv = lv2; boards[lv + 1] = [post]
        gfx[g] = fx
        gfv[g] = {k: _sample(v, FXV_MAX) for k, v in fxv.items()}
        for L, bl in sorted(boards.items()):
            if L != 1 and str(L) not in P[g]:
                continue
            bl = [b for b in bl if b is not None]
            if not bl:
                continue
            mv = (moves.get(L) or [])[:2]
            entries.append({'game': g, 'L': int(L), 'kind': 'start', 'board': [round(float(x), 5) for x in board_desc(bl[0])],
                            'kinds': [m[0] for m in mv], 'changed': [int(m[1]) for m in mv],
                            'fx': [[round(float(x), 5) for x in effect_desc(m[2], m[3])] for m in mv]})
            if L in moves and len(bl) > 2:                          # boards inside the level (a solved level only)
                for f in MID_FRACS:
                    i = min(len(bl) - 2, max(1, int(round(f * (len(bl) - 2)))))
                    entries.append({'game': g, 'L': int(L), 'kind': 'mid', 'board': [round(float(x), 5) for x in board_desc(bl[i])]})
        log(f'  {g}: {len([e for e in entries if e["game"] == g])} boards, verified action kinds {sorted(fx)}')
    pfx = {}; pfv = {}
    for g in sorted(games or P):
        env = make_env(g, seed=0, prefer_real=True, force_offline=False)
        cnt = {}; vv = {}
        for L in [1] + sorted(int(x) for x in P[g]):
            for sd in range(PROBE_SEEDS):
                obs = _to_level(env, P, g, L)
                if obs is None:
                    continue
                rng = np.random.default_rng(100 + 1000 * sd + 17 * L)
                for kind, ch, fxd in _probe_actions(env, obs, rng, PROBE_N):
                    c = cnt.setdefault(kind, [0, 0]); c[0] += int(ch); c[1] += 1
                    lc = lfx.setdefault(g, {}).setdefault(f'{L}:{kind}', [0, 0]); lc[0] += int(ch); lc[1] += 1
                    if ch and fxd is not None:
                        vv.setdefault(kind, []).append([L] + list(fxd))
        pfx[g] = cnt; pfv[g] = {k: _sample(v, FXV_MAX) for k, v in vv.items()}
        log(f'  {g}: generic probes changed the board {dict((k, f"{v[0]}/{v[1]}") for k, v in sorted(cnt.items()))}')
    B = np.asarray([e['board'] for e in entries], dtype=float)
    doc = {'version': '1173.3', 'entries': entries, 'game_fx': gfx, 'game_fxv': gfv, 'probe_fx': pfx, 'probe_fxv': pfv, 'level_fx': lfx,
           'mu': [round(float(x), 6) for x in B.mean(0)], 'sd': [round(float(x), 6) for x in B.std(0)]}
    idx = Index(doc)
    recs = live_records(idx, seeds=live_seeds, games=games, log=lambda *_: None)
    doc['measures'] = measures(idx, recs)
    p = out or os.path.join(HERE, INDEX_FILE)
    json.dump(doc, open(p, 'w', encoding='utf-8'))
    json.dump([[g, L, list(map(float, bd)), [[k, int(c), f] for k, c, f in obs]] for g, L, bd, obs in recs],
              open(os.path.splitext(p)[0] + '_live.json', 'w', encoding='utf-8'))
    m = doc['measures']
    log(f"  index: {len(entries)} boards ({sum(1 for e in entries if e['kind'] == 'start')} level starts) of {len(gfx)} games -> {p}")
    log(f"  held-out level, board only / + feedback: verified first moves {m['lolo']['board_only']:.1%} / {m['lolo']['with_feedback']:.1%}; "
        f"level 1 {m['lolo_level1']['board_only']:.1%} / {m['lolo_level1']['with_feedback']:.1%}; two GENERIC probes (live, "
        f"{m['live']['n']} level starts x seeds) {m['live']['board_only']:.1%} / {m['live']['with_feedback']:.1%}")
    return doc


def live_records(index, seeds=LIVE_SEEDS, n_probe=2, games=None, log=print):
    """the runtime question's data: at every known level start the generic probe makes n_probe actions on the real engine"""
    sys.path.insert(0, HERE)
    from arc_env.adapter import make_env
    P = json.load(open(os.path.join(HERE, 'level_prefix.json'), encoding='utf-8'))['prefix']
    recs = []
    for g in sorted(games or P):
        env = make_env(g, seed=0, prefer_real=True, force_offline=False)
        for L in [1] + sorted(int(x) for x in P[g]):
            for sd in seeds:
                obs = _to_level(env, P, g, L)
                b = _settled(obs) if obs is not None else None
                if b is None:
                    continue
                rng = np.random.default_rng(1000 * sd + 17 * L + 1173)
                recs.append((g, L, [float(x) for x in board_desc(b)], _probe_actions(env, obs, rng, n_probe)))
        log(f'  {g}: {sum(1 for r in recs if r[0] == g)} live level starts')
    return recs


def measures(idx, recs, w_fx=W_FX, w_efx=W_EFX):
    lo = evaluate(idx, lolo_records(idx), w_fx, w_efx); l1 = evaluate(idx, lolo_records(idx, True), w_fx, w_efx)
    lv = evaluate(idx, recs, w_fx, w_efx)
    return {'lolo': {k: v for k, v in lo.items() if k != 'margins'}, 'lolo_level1': {k: v for k, v in l1.items() if k not in ('margins', 'per_game')},
            'live': {k: v for k, v in lv.items() if k != 'margins'}, 'sure_board_only': sure_table(lo['margins'] + lv['margins']),
            'weights': {'w_fx': w_fx, 'w_efx': w_efx}}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('cmd', choices=('build', 'lolo'))
    ap.add_argument('--out', default=None); ap.add_argument('--index', default=None); ap.add_argument('--games', default='')
    ap.add_argument('--w-fx', type=float, default=W_FX); ap.add_argument('--w-efx', type=float, default=W_EFX)
    ap.add_argument('--write', action='store_true', help='lolo: store the measures in the index')
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        build(a.out, [g for g in a.games.split(',') if g] or None)
        return 0
    p = a.index or os.path.join(HERE, INDEX_FILE)
    idx = Index.load(p)
    recs = [(g, L, bd, [(k, bool(c), f) for k, c, f in obs]) for g, L, bd, obs in
            json.load(open(os.path.splitext(p)[0] + '_live.json', encoding='utf-8'))]
    m = measures(idx, recs, a.w_fx, a.w_efx)
    if a.write:
        idx.doc['measures'] = m
        json.dump(idx.doc, open(p, 'w', encoding='utf-8'))
    print(json.dumps(m, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
