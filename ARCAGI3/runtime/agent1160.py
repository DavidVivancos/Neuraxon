#!/usr/bin/env python3
"""agent1160.py -- NAS 1160: THE GENERIC AGENT, measured LEAVE-ONE-GAME-OUT, every action charged.

WHY (nas_1159_report.md, section 17, changes 1-5 and 10)
  * Five NAS runs mastered 2 levels; game-specific planners that PROBE, read the GOAL from finished boards and SEARCH
    cleared 47 levels at or below human (V1158_STATUS) -- but on free branched copies of the simulator, with
    perception written per game. Kaggle gives neither: one play, every action counted, unseen games.
  * This file is the honest version of that loop: ONE agent, no per-game code, no simulator copies, every probe a
    real action, measured on a game it was never tuned on.

THE AGENT (one episode = one play of one game from level 1)
  perception   the SETTLED frame (frame[-1]); objects = 4-connected components of each non-background colour;
               a HUD mask learned online (cells changing on > hud_rate of transitions; border LINES changing on
               > 20%), frozen after 16 transitions; until then the 2-px border is ignored (budget bars live there)
  actions      the game's available actions: keys 1-5 (+ UNDO 7 when allowed) and clicks on one representative
               cell of every object (rarer colours and smaller objects first, at most max_clicks), optionally
               ring points at +-ring around the 8 first objects (1158: 98% of expert clicks lie within +-8 of one)
  memory       state graph per level: hash of the board without the HUD -> node; per node the actions tried and
               their outcome (no change / change -> node / death); never repeat a no-change action at a node
               (Blind Squirrel). Boards are deterministic, so the graph survives deaths and RESETs.
  effects      per action CLASS (a key; a click on an object of (colour, size bucket)): P(change), P(death), the
               mean colour-histogram change, learned online across states and levels (StochasticGoose, counted)
  navigation   when the node has nothing untried, BFS over observed edges to the nearest node that has, and walk
               there; else UNDO, else RESET (both charged)
  goal         after a level is cleared, its start board and its FINISHED board (frame[0] of the level-up response:
               the stale-frame discovery) give a colour-change signature; later levels prefer action classes whose
               observed effects move the board along it; classes on the winning path are boosted (cross-level
               memory)
  budget       per level ceil(cap_k x human) actions, RESETs included (cap_k = 3: beyond 2x human the level is
               worth <= 0.25 and beyond 3x <= 0.11 of its points); a level not cleared within it ends the episode

THE MEASUREMENT (one machine = one fold)
  * the machine's game is HELD OUT: the configuration search never sees its score
  * seeds: the agent's RNG seed (boards are fixed per game). Screening: 10 seeds; if the outcome differs between
    seeds or the action counts of a cleared level have CV > screen_cv, 10 more (20 = the gold standard). Every new
    best is topped up to 20 seeds on every training game and played 20 times on the held-out game
  * the search: (mu + lambda) over the agent's knobs, fitness = the mean official game score over the 24 training
    games (level score min(1, human / actions)^2, game score sum L * s_L / (1 + ... + levels)), gen 0 = the
    ABLATION LADDER A0 random -> A6 full, so every fold also measures what each component adds
  * outputs in arc_runs/arc_nas_<date>_agent/: episodes.jsonl (EVERY play: seed, actions per level, cleared,
    resets, deaths, undos, the per-try action flow), configs.jsonl, best.json, summary.json (the best config: per
    game, per seed, actions per level -- train AND held-out), progress.txt

    python3 agent1160.py --games vc33,lp85 --seeds 2 --config A6        # quick local test
    python3 NxonArcNAS.py --vps agent-vc33                               # a fleet machine (cell in V1097_MACHINE)
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    from scipy import ndimage as _NDI
except Exception:                                   # the boxes guarantee numpy (arcengine), not scipy
    _NDI = None

VERSION = '1.160.0'
GAMES = ['ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59', 'lf52', 'lp85', 'ls20', 'm0r0', 'r11l',
         're86', 's5i5', 'sb26', 'sc25', 'sk48', 'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30']
SEEDS = [7, 1013, 2027, 3049, 4051, 5059, 6067, 7079, 8081, 9091,
         10093, 11113, 12119, 13127, 14149, 15161, 16183, 17189, 18199, 19207]

# ----------------------------------------------------------------------------------------------- baselines
_H = {}


def human(game):
    """the human action count of every level (the game's metadata: the denominator of RHAE)"""
    if not _H:
        d = os.environ.get('MULTINXON_ARC_ENV_DIR') or os.path.join(HERE, 'environment_files')
        for md in glob.glob(os.path.join(d, '**', 'metadata.json'), recursive=True):
            try:
                m = json.load(open(md, encoding='utf-8'))
                _H[str(m['game_id']).split('-')[0]] = [float(x) for x in m.get('baseline_actions') or []]
            except Exception:
                pass
    return _H.get(game) or []


def game_score(levels, H):
    """the official score of one play: sum L * min(1, h_L / a_L)^2 over cleared levels / (1 + ... + n)"""
    W = sum(range(1, len(H) + 1)) or 1
    return sum(l['L'] * min(1.0, l['h'] / max(1.0, l['actions'])) ** 2 for l in levels if l['cleared']) / W

# ----------------------------------------------------------------------------------------------- perception


def label(mask):
    """4-connected components (scipy if present, else a flood fill)"""
    if _NDI is not None:
        return _NDI.label(mask)
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); n = 0
    m = mask.tolist()
    lb = [[0] * W for _ in range(H)]
    for y0 in range(H):
        row = m[y0]
        for x0 in range(W):
            if not row[x0] or lb[y0][x0]:
                continue
            n += 1; lb[y0][x0] = n; stack = [(y0, x0)]
            while stack:
                y, x = stack.pop()
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= yy < H and 0 <= xx < W and m[yy][xx] and not lb[yy][xx]:
                        lb[yy][xx] = n; stack.append((yy, xx))
    if n:
        lab[:, :] = np.asarray(lb, np.int32)
    return lab, n


def objects(g, ignore, bg):
    """(colour, size, y, x) per component of every non-background colour outside the ignore mask; (y, x) = the
    component's cell nearest its centroid (inside the object, so a click lands ON it)"""
    out = []
    Hh, Ww = g.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww]
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
        flat = lab.ravel(); sel = flat > 0
        Lb = flat[sel]; Y = yy.ravel()[sel]; X = xx.ravel()[sel]
        size = np.bincount(Lb, minlength=n + 1)[1:]
        cy = np.bincount(Lb, weights=Y, minlength=n + 1)[1:] / np.maximum(size, 1)
        cx = np.bincount(Lb, weights=X, minlength=n + 1)[1:] / np.maximum(size, 1)
        d = (Y - cy[Lb - 1]) ** 2 + (X - cx[Lb - 1]) ** 2
        order = np.lexsort((d, Lb))
        Ls = Lb[order]
        first = np.r_[True, Ls[1:] != Ls[:-1]]
        for i in order[first]:
            lid = int(Lb[i])
            out.append((c, int(size[lid - 1]), int(Y[i]), int(X[i])))
    return out


def size_bucket(s):
    return 0 if s <= 0 else 1 if s == 1 else 2 if s <= 4 else 3 if s <= 16 else 4 if s <= 64 else 5 if s <= 256 else 6


class Hud:
    """cells / border lines that change on most transitions (counters, budget bars): never a 'change', never state"""

    def __init__(self, rate, shape=(64, 64), freeze_at=16):
        self.rate = float(rate); self.freeze_at = int(freeze_at); self.n = 0; self.frozen = False
        self.cnt = np.zeros(shape, np.int32)
        self.lines = np.zeros(12, np.int32)                       # rows 0,1,2,-3,-2,-1 then the same columns
        self.border = np.zeros(shape, bool)
        self.border[:2, :] = self.border[-2:, :] = True; self.border[:, :2] = self.border[:, -2:] = True
        self.mask = self.border.copy()

    def update(self, g0, g1):
        if self.frozen or g0 is None or g1 is None or g0.shape != g1.shape:
            return False
        if self.cnt.shape != g0.shape:
            self.__init__(self.rate, g0.shape, self.freeze_at)
        d = g0 != g1
        self.cnt += d; self.n += 1
        for i, r in enumerate((0, 1, 2, -3, -2, -1)):
            self.lines[i] += bool(d[r, :].any()); self.lines[6 + i] += bool(d[:, r].any())
        if self.n >= self.freeze_at:
            m = (self.cnt / float(self.n)) > self.rate
            for i, r in enumerate((0, 1, 2, -3, -2, -1)):
                if self.lines[i] / float(self.n) > 0.2:
                    m[r, :] = True
                if self.lines[6 + i] / float(self.n) > 0.2:
                    m[:, r] = True
            self.mask = m; self.frozen = True
            return True                                           # the state key changes: the caller rebuilds keys
        return False


def settled(obs, prev):
    fr = getattr(obs, 'frame', None) or []
    if fr:
        try:
            g = np.asarray(fr[-1], dtype=np.int16)
            if g.ndim == 2 and g.size:
                return g
        except Exception:
            pass
    return prev


def first_frame(obs):
    fr = getattr(obs, 'frame', None) or []
    if len(fr) > 1:
        try:
            g = np.asarray(fr[0], dtype=np.int16)
            return g if g.ndim == 2 else None
        except Exception:
            return None
    return None

# ----------------------------------------------------------------------------------------------- configurations


DEFAULT = dict(memory=1, effect=1, nav=1, goal=1, undo=1, reset=1, w_change=1.2, w_novel=0.6, w_death=4.0,
               w_goal=0.8, w_path=0.5, w_small=0.4, w_rare=0.6, noise=0.05, eps=0.03, max_clicks=48, ring=0,
               hud_rate=0.35)
_OFF = dict(memory=0, effect=0, nav=0, goal=0, w_change=0.0, w_novel=0.0, w_goal=0.0, w_path=0.0, w_small=0.0,
            w_rare=0.0, noise=1.0, eps=0.0)
LADDER = collections.OrderedDict([
    ('A0', dict(DEFAULT, **_OFF)),                                                     # random over the candidates
    ('A1', dict(DEFAULT, **dict(_OFF, memory=1))),                                     # + state memory
    ('A2', dict(DEFAULT, **dict(_OFF, memory=1, effect=1, w_change=1.2, noise=0.05))),  # + effect model
    ('A3', dict(DEFAULT, **dict(_OFF, memory=1, effect=1, nav=1, w_change=1.2, w_novel=0.6, noise=0.05))),
    ('A4', dict(DEFAULT, **dict(_OFF, memory=1, effect=1, nav=1, goal=1, w_change=1.2, w_novel=0.6, w_goal=0.8,
                                noise=0.05))),                                         # + goal signature
    ('A5', dict(DEFAULT, w_small=0.0, w_rare=0.0)),                                    # + path memory, no click order
    ('A6', dict(DEFAULT)),                                                             # everything
])
SPACE = {'w_change': (0.0, 3.0), 'w_novel': (0.0, 2.0), 'w_death': (0.0, 10.0), 'w_goal': (0.0, 3.0),
         'w_path': (0.0, 2.0), 'w_small': (0.0, 2.0), 'w_rare': (0.0, 2.0), 'noise': (0.0, 0.5), 'eps': (0.0, 0.3),
         'hud_rate': (0.15, 0.7), 'max_clicks': (8, 128)}
CHOICES = {'ring': [0, 2, 4, 8], 'undo': [0, 1], 'reset': [0, 1], 'nav': [0, 1], 'goal': [0, 1], 'memory': [0, 1],
           'effect': [0, 1]}


def cfg_id(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def mutate(cfg, rng, p=0.3):
    c = dict(cfg)
    for k, (lo, hi) in SPACE.items():
        if rng.random() < p:
            v = float(c.get(k, lo)) + rng.gauss(0, 0.15 * (hi - lo))
            v = min(hi, max(lo, v))
            c[k] = int(round(v)) if k == 'max_clicks' else round(v, 4)
    for k, opts in CHOICES.items():
        if rng.random() < p * 0.5:
            c[k] = rng.choice(opts)
    return c

# ----------------------------------------------------------------------------------------------- the agent


class Agent:
    def __init__(self, cfg, rng):
        self.cfg = dict(DEFAULT, **(cfg or {})); self.rng = rng
        self.hud = Hud(self.cfg['hud_rate'])
        self.eff = {}                    # class -> [n, changed, died, levelups, delta(16)]
        self.path = collections.Counter()  # classes on winning paths
        self.sig = None                  # the goal signature (normalised colour change start -> finished)
        self.graph = {}; self.start_board = None; self.bg = 0; self.level_path = []
        self.n_nochange = 0

    # -- state
    def ignore(self):
        return self.hud.mask

    def key(self, g):
        h = g.copy(); h[self.ignore()] = -1
        return hashlib.md5(h.tobytes()).hexdigest()[:16]

    def hist(self, g):
        return np.bincount(np.clip(g[~self.ignore()], 0, 15).ravel(), minlength=16)[:16].astype(float)

    def start_level(self, g):
        self.graph = {}; self.start_board = g.copy(); self.level_path = []
        self.bg = int(np.bincount(np.clip(g.ravel(), 0, 15), minlength=16).argmax())

    def node(self, g, avail):
        k = self.key(g)
        nd = self.graph.get(k)
        if nd is None:
            nd = {'cands': self.candidates(g, avail), 'tried': {}}
            self.graph[k] = nd
        return k, nd

    def candidates(self, g, avail):
        c = self.cfg; out = []
        for a in avail:
            if a in (1, 2, 3, 4, 5) or (a == 7 and c['undo']):
                out.append((('K', a), ('K', a)))
        if 6 in avail:
            objs = objects(g, self.ignore(), self.bg)
            cnt = np.bincount(np.clip(g.ravel(), 0, 15), minlength=16)
            objs.sort(key=lambda o: -(c['w_rare'] * 64.0 / (1.0 + cnt[o[0]]) + c['w_small'] * 8.0 / (1.0 + o[1]))
                      + 1e-3 * self.rng.random())
            objs = objs[:max(1, int(c['max_clicks']))]
            seen = set()
            for col, s, y, x in objs:
                if (y, x) not in seen:
                    seen.add((y, x)); out.append((('C', y, x), ('C', col, size_bucket(s))))
            r = int(c.get('ring') or 0)
            if r > 0:
                H, W = g.shape
                for col, s, y, x in objs[:8]:
                    for dy, dx in ((-r, 0), (r, 0), (0, -r), (0, r)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and (yy, xx) not in seen:
                            seen.add((yy, xx)); out.append((('C', yy, xx), ('R', int(g[yy, xx]), r)))
        return out

    # -- choice
    def score(self, cls):
        c = self.cfg; st = self.eff.get(cls)
        n = st[0] if st else 0
        pc = ((st[1] + 1.0) / (n + 2.0)) if (st and c['effect']) else 0.5
        pd = (st[2] / (n + 1.0)) if st else 0.0
        nov = 1.0 / math.sqrt(1.0 + n)
        goal = 0.0
        if c['goal'] and self.sig is not None and st and st[1] > 0:
            d = st[4] / float(st[1]); nd = float(np.linalg.norm(d))
            if nd > 0:
                goal = float(d @ self.sig) / nd
        path = math.log1p(self.path.get(cls, 0))
        return (c['w_change'] * pc + c['w_novel'] * nov - c['w_death'] * pd + c['w_goal'] * goal + c['w_path'] * path
                + c['noise'] * self.rng.random())

    def deadly(self, cls):
        st = self.eff.get(cls)
        return bool(st and st[2] >= 2 and st[2] > 0.5 * st[0])

    def frontier(self, nd):
        return any(a not in nd['tried'] and not self.deadly(cl) for a, cl in nd['cands'])

    def bfs(self, start):
        prev = {start: None}; q = collections.deque([start])
        while q and len(prev) < int(getattr(self, '_bfs_max', 5000) or 5000):          # v1.177 change 9: 20,000 on a 1177 brain
            u = q.popleft()
            nd = self.graph.get(u)
            if nd is None:
                continue
            if u != start and self.frontier(nd):
                path = []; v = u
                while prev[v] is not None:
                    pu, a = prev[v]; path.append(a); v = pu
                return path[-1] if path else None
            for a, (v, chg, died) in nd['tried'].items():
                if v is None or died or not chg or v in prev:
                    continue
                prev[v] = (u, a); q.append(v)
        return None

    def choose(self, k, nd, avail):
        c = self.cfg; cands = nd['cands']
        if not cands:
            return ('RESET',), ('RESET',)
        if c['eps'] > 0 and self.rng.random() < c['eps']:
            return self.rng.choice(cands)
        pool = [x for x in cands if (x[0] not in nd['tried'] or not c['memory'])]
        pool = [x for x in pool if not self.deadly(x[1])] or (pool if not c['memory'] else [])
        if pool:
            return max(pool, key=lambda x: self.score(x[1]))
        if c['nav']:
            a = self.bfs(k)
            if a is not None:
                return a, next((cl for aa, cl in cands if aa == a), ('NAV',))
        if c['undo'] and 7 in avail:
            return ('K', 7), ('K', 7)
        if c['reset']:
            return ('RESET',), ('RESET',)
        return self.rng.choice(cands)

    # -- learning
    def observe(self, k0, akey, cls, g0, g1, died, up):
        if self.hud.update(g0, g1):                     # the HUD froze: state keys change -> a fresh graph
            self.graph = {}
        ch = bool(((g0 != g1) & ~self.ignore()).any()) if (g0 is not None and g1 is not None and g0.shape == g1.shape) else True
        nd = self.graph.get(k0)
        nk = None if died or up else self.key(g1)
        if nd is not None:
            nd['tried'][akey] = (nk, ch, died)
        if cls and cls[0] in ('K', 'C', 'R'):
            st = self.eff.setdefault(cls, [0, 0, 0, 0, np.zeros(16)])
            st[0] += 1; st[1] += int(ch); st[2] += int(died); st[3] += int(up)
            if ch and not died and g0 is not None and g1 is not None and g0.shape == g1.shape:
                st[4] = st[4] + (self.hist(g1) - self.hist(g0))
            if ch:
                self.level_path.append(cls)
        if not ch:
            self.n_nochange += 1
        return ch

    def level_done(self, finished):
        if self.start_board is not None and finished is not None and finished.shape == self.start_board.shape:
            d = self.hist(finished) - self.hist(self.start_board); nd = float(np.linalg.norm(d))
            if nd > 0:
                d = d / nd
                self.sig = d if self.sig is None else (self.sig + d) / max(1e-9, float(np.linalg.norm(self.sig + d)))
        for cl in self.level_path:
            self.path[cl] += 1

# ----------------------------------------------------------------------------------------------- one episode


def _act(env, akey):
    from arc_env.structs import GameAction
    if akey[0] == 'RESET':
        return env.reset()
    if akey[0] == 'K':
        return env.step(GameAction(int(akey[1])))
    return env.step(GameAction.ACTION6, {'x': int(akey[2]), 'y': int(akey[1])})


def run_episode(game, cfg, seed, cap_k=3.0, max_wall=300.0):
    """one play of `game` from level 1 by a fresh agent; every action (RESETs included) counted"""
    import game_loop as G
    rng = random.Random(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    H = human(game)
    if not H:
        raise RuntimeError(f'no human baseline for {game}')
    t0 = time.time()
    try:
        from arc_env import adapter as AD
    except Exception:
        AD = None
    env = G.make_env(game, seed=seed, prefer_real=True, force_offline=False)
    if AD is not None and hasattr(AD, 'v1160_take'):
        AD.v1160_take()
    obs = env.reset()
    if AD is not None and hasattr(AD, 'v1160_take'):
        AD.v1160_take()                                            # the start is not the agent's action
    ag = Agent(cfg, rng)
    g = settled(obs, None)
    if g is None:
        raise RuntimeError('no frame at reset')
    ag.start_level(g)
    level = int(getattr(obs, 'levels_completed', 0) or 0)
    nlev = len(H)
    levels = []; screens = []; la = 0; total = 0; resets = deaths = undos = keys = clicks = 0
    cap = int(math.ceil(cap_k * H[min(level, nlev - 1)]))
    status = 'cap'
    while True:
        if time.time() - t0 > max_wall:
            status = 'wall'; break
        if la >= cap:
            status = 'cap'; break
        avail = [int(a) for a in (getattr(obs, 'available_actions', None) or [])]
        k, nd = ag.node(g, avail)
        akey, cls = ag.choose(k, nd, avail)
        g0 = g
        obs = _act(env, akey); total += 1; la += 1
        if akey[0] == 'RESET':
            resets += 1
        elif akey[0] == 'K':
            keys += 1; undos += int(akey[1] == 7)
        else:
            clicks += 1
        st = int(getattr(obs, 'state', 1) or 1)
        g1 = settled(obs, g0)
        newlv = int(getattr(obs, 'levels_completed', 0) or 0)
        died = (st == 3); up = newlv > level
        ag.observe(k, akey, cls, g0, g1, died, up)
        if up:
            fin = first_frame(obs)
            levels.append(dict(L=level + 1, actions=la, cleared=True, h=H[level], cap=cap))
            fb = fin if fin is not None else g0                          # the FINISHED board (stale-frame discovery)
            screens.append(dict(L=level + 1, start=hashlib.md5(ag.start_board.tobytes()).hexdigest()[:12],
                                finished=hashlib.md5(fb.tobytes()).hexdigest()[:12],
                                delta=[int(x) for x in (ag.hist(fb) - ag.hist(ag.start_board))]))
            ag.level_done(fb)
            level = newlv; la = 0
            if st == 2 or level >= nlev:
                status = 'win'; break
            cap = int(math.ceil(cap_k * H[level])); g = g1; ag.start_level(g)
            continue
        if st == 2:
            status = 'win'; break
        if died:
            deaths += 1
            obs = env.reset(); total += 1; la += 1; resets += 1   # Kaggle counts the RESET
            g = settled(obs, g1)
            continue
        g = g1
    if status != 'win' and level < nlev:
        levels.append(dict(L=level + 1, actions=la, cleared=False, h=H[level], cap=cap))
    tally = AD.v1160_take() if (AD is not None and hasattr(AD, 'v1160_take')) else None
    cl = sum(1 for l in levels if l['cleared'])
    return dict(game=game, seed=seed, levels=levels, cleared=cl, n_levels=nlev, actions=total, resets=resets,
                deaths=deaths, undos=undos, keys=keys, clicks=clicks, nochange=ag.n_nochange,
                states=len(ag.graph), status=status, score=round(game_score(levels, H), 6),
                wall=round(time.time() - t0, 2), tally=tally, screens=screens)

# ----------------------------------------------------------------------------------------------- evaluation


def _task(args):
    game, cfg, seed, cap_k, cid, set_ = args
    try:
        row = run_episode(game, cfg, seed, cap_k=cap_k)
    except Exception as e:
        row = dict(game=game, seed=seed, levels=[], cleared=0, actions=0, score=0.0, status='error',
                   error=f'{type(e).__name__}: {e}'[:300], trace=traceback.format_exc()[-600:])
    row['cid'] = cid; row['set'] = set_
    return row


def stable(rows, cv):
    """10 seeds are enough when every seed cleared the same number of levels and, on every cleared level, the action
    counts are similar (coefficient of variation <= cv); otherwise play 10 more"""
    if not rows or any(r.get('error') for r in rows):
        return False
    ns = {r['cleared'] for r in rows}
    if len(ns) > 1:
        return False
    for L in range(1, next(iter(ns)) + 1):
        a = [r['levels'][L - 1]['actions'] for r in rows]
        m = sum(a) / float(len(a))
        if m > 0 and statistics.pstdev(a) / m > cv:
            return False
    return True


def summarise(rows, game):
    """per game: seeds, mean score, levels cleared, L1 rate, actions per level of cleared seeds, the per-seed table"""
    H = human(game); rows = sorted(rows, key=lambda r: SEEDS.index(r['seed']) if r['seed'] in SEEDS else 99)
    per_level = {}
    for r in rows:
        for l in r.get('levels') or []:
            if l['cleared']:
                per_level.setdefault(l['L'], []).append(l['actions'])
    lv = {}
    for L, a in sorted(per_level.items()):
        lv[f'L{L}'] = dict(cleared=len(a), mean=round(sum(a) / len(a), 2), median=statistics.median(a), min=min(a),
                           max=max(a), human=H[L - 1] if L - 1 < len(H) else None)
    n = len(rows) or 1
    return dict(n=len(rows), score=round(sum(r['score'] for r in rows) / n, 6),
                levels=round(sum(r['cleared'] for r in rows) / n, 3),
                l1_rate=round(sum(1 for r in rows if r['cleared'] >= 1) / n, 3), per_level=lv,
                errors=sum(1 for r in rows if r.get('error')),
                per_seed=[[r['seed'], r['cleared'], [l['actions'] for l in r.get('levels') or []]] for r in rows])


class Evaluator:
    def __init__(self, out_dir, workers, cap_k, n_screen, n_full, cv):
        import multiprocessing as mp
        self.out = out_dir; self.cap_k = cap_k; self.ns = n_screen; self.nf = n_full; self.cv = cv
        self.workers = max(1, int(workers))
        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = mp.get_context('spawn')
        self.pool = ctx.Pool(self.workers) if self.workers > 1 else None
        self.cache = {}                                      # (cid, game) -> rows
        self.f_ep = open(os.path.join(out_dir, 'episodes.jsonl'), 'a', encoding='utf-8')

    def _run(self, tasks):
        it = self.pool.imap_unordered(_task, tasks) if self.pool else map(_task, tasks)
        out = []
        for r in it:
            out.append(r)
            self.f_ep.write(json.dumps(r, default=str) + '\n')
        self.f_ep.flush()
        return out

    def games(self, cid, cfg, games, set_, full=False):
        """every game at n_screen seeds (topped up to n_full when unstable), or straight at n_full"""
        want = {}
        for g in games:
            have = self.cache.get((cid, g), [])
            n = self.nf if full else self.ns
            if len(have) < n:
                want[g] = SEEDS[len(have):n]
        tasks = [(g, cfg, s, self.cap_k, cid, set_) for g, ss in want.items() for s in ss]
        for r in self._run(tasks):
            self.cache.setdefault((cid, r['game']), []).append(r)
        if not full and self.nf > self.ns:
            more = [g for g in games if len(self.cache.get((cid, g), [])) < self.nf and not stable(self.cache.get((cid, g), []), self.cv)]
            tasks = [(g, cfg, s, self.cap_k, cid, set_) for g in more for s in SEEDS[len(self.cache.get((cid, g), [])):self.nf]]
            for r in self._run(tasks):
                self.cache.setdefault((cid, r['game']), []).append(r)
        return {g: summarise(self.cache.get((cid, g), []), g) for g in games}

    def close(self):
        try:
            self.f_ep.close()
        except Exception:
            pass
        if self.pool:
            self.pool.terminate()


def fitness(summ):
    """the mean official game score over the training games; ties -> mean levels cleared"""
    if not summ:
        return (0.0, 0.0)
    return (sum(v['score'] for v in summ.values()) / len(summ), sum(v['levels'] for v in summ.values()) / len(summ))

# ----------------------------------------------------------------------------------------------- the machine


def available_games():
    d = os.environ.get('MULTINXON_ARC_ENV_DIR') or os.path.join(HERE, 'environment_files')
    return [g for g in GAMES if glob.glob(os.path.join(d, g, '*', 'metadata.json')) or glob.glob(os.path.join(d, g, 'metadata.json'))]


def run_machine(mid, cell, out_root=None):
    """one fleet machine = one leave-one-game-out fold: search on 24 games, report the held-out one"""
    smoke = bool(int(os.environ.get('MULTINXON_SMOKE', '0') or 0))
    hold = str(cell.get('agent_holdout') or cell.get('game'))
    av = available_games()
    train = [g for g in av if g != hold]
    cap_k = float(cell.get('agent_cap_k') or 3.0)
    ns = int(cell.get('agent_screen') or 10); nf = int(cell.get('agent_stable') or 20); cv = float(cell.get('agent_cv') or 0.25)
    mu = int(cell.get('agent_mu') or 6); lam = int(cell.get('agent_lambda') or 12)
    hours = float(cell.get('agent_hours') or 24.0)
    seed = int(cell.get('agent_seed') or 1160)
    workers = int(cell.get('agent_workers') or 0) or (os.cpu_count() or 1)
    if smoke:
        train = train[:2]; ns = nf = 2; mu = 1; lam = 1; hours = 0.05; workers = min(2, workers)
    rng = random.Random(seed)
    out_root = out_root or os.path.join(HERE, 'arc_runs')
    out = os.path.join(out_root, time.strftime('arc_nas_%Y%m%d_%H%M%S') + '_agent'); os.makedirs(out, exist_ok=True)
    meta = dict(machine=mid, version=VERSION, holdout=hold, train=train, cap_k=cap_k, screen=ns, stable=nf, cv=cv,
                mu=mu, lam=lam, hours=hours, seed=seed, workers=workers, smoke=smoke, seeds=SEEDS[:nf], cell=cell,
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(meta, open(os.path.join(out, 'machine.json'), 'w'), indent=1, default=str)
    prog = open(os.path.join(out, 'progress.txt'), 'a', encoding='utf-8')

    def say(s):
        print(s, flush=True); prog.write(s + '\n'); prog.flush()

    say(f"[agent1160] {mid}: held-out {hold}; train {len(train)} games; cap {cap_k}x human per level; seeds {ns} -> {nf} "
        f"(cv {cv}); mu {mu} + lambda {lam}; {workers} workers; {hours} h; out {out}")
    ev = Evaluator(out, workers, cap_k, ns, nf, cv)
    f_cfg = open(os.path.join(out, 'configs.jsonl'), 'a', encoding='utf-8')
    pop = []                                                       # (fitness, cid, cfg, summary)
    best = None; t_start = time.time(); gen = 0; seen = set()

    def evaluate(cfg, name, parent, gen_):
        cid = cfg_id(cfg)
        if cid in seen:
            return None
        seen.add(cid)
        t1 = time.time()
        summ = ev.games(cid, cfg, train, 'train')
        fit = fitness(summ)
        rec = dict(gen=gen_, cid=cid, name=name, parent=parent, cfg=cfg, fitness=fit[0], levels=fit[1],
                   train={g: {k: v[k] for k in ('n', 'score', 'levels', 'l1_rate', 'per_level')} for g, v in summ.items()},
                   wall=round(time.time() - t1, 1))
        hold_s = None
        if gen_ == 0 or best is None or fit > best[0]:
            hold_s = ev.games(cid, cfg, [hold], 'holdout', full=True)[hold]      # 20 seeds, NEVER used to select
            rec['holdout'] = {k: hold_s[k] for k in ('n', 'score', 'levels', 'l1_rate', 'per_level', 'per_seed')}
        f_cfg.write(json.dumps(rec, default=str) + '\n'); f_cfg.flush()
        say(f"  g{gen_} {name:>5s} {cid}: train score {fit[0]:.4f} levels {fit[1]:.2f}"
            + (f" | held-out {hold}: score {hold_s['score']:.4f} levels {hold_s['levels']:.2f} L1 {hold_s['l1_rate']:.2f}" if hold_s else '')
            + f" ({rec['wall']} s)")
        return (fit, cid, cfg, summ, hold_s)

    try:
        names = list(LADDER.items())
        if smoke:
            names = names[:2]
        for name, cfg in names:                                       # generation 0: the ablation ladder
            r = evaluate(dict(cfg), name, None, 0)
            if r:
                pop.append(r)
                if best is None or r[0] > best[0]:
                    best = r
        while not smoke and (time.time() - t_start) < hours * 3600.0:
            gen += 1
            pop.sort(key=lambda x: x[0], reverse=True)
            parents = pop[:mu]
            kids = []
            for i in range(lam):
                p = parents[i % len(parents)] if rng.random() < 0.7 else rng.choice(parents)
                kids.append((mutate(p[2], rng), p[1]))
            for cfg, pid in kids:
                r = evaluate(cfg, 'mut', pid, gen)
                if not r:
                    continue
                pop.append(r)
                if r[0] > best[0]:
                    best = r
                    ev.games(r[1], r[2], train, 'train', full=True)         # the new best at 20 seeds everywhere
            pop.sort(key=lambda x: x[0], reverse=True); pop = pop[:max(mu * 3, 12)]
            write_best(out, best, train, hold, ev, say)
        write_best(out, best, train, hold, ev, say)
    finally:
        ev.close(); f_cfg.close()
    say(f"[agent1160] {mid}: done after {gen} generations, {len(seen)} configurations, {(time.time() - t_start) / 3600.0:.2f} h")
    prog.close()
    return 0


def write_best(out, best, train, hold, ev, say):
    if not best:
        return
    fit, cid, cfg, summ, hold_s = best
    tr = {g: summarise(ev.cache.get((cid, g), []), g) for g in train}
    ho = summarise(ev.cache.get((cid, hold), []), hold) if ev.cache.get((cid, hold)) else None
    doc = dict(cid=cid, cfg=cfg, train_fitness=fit[0], train_levels=fit[1], train=tr, holdout_game=hold, holdout=ho,
               note='per_seed = [seed, levels cleared, [actions on each level played]]; the last level of an '
                    'uncleared play is the one that hit the cap (3 x human)')
    json.dump(doc, open(os.path.join(out, 'best.json'), 'w'), indent=1, default=str)
    json.dump(dict(best=doc, written=time.strftime('%Y-%m-%d %H:%M:%S')), open(os.path.join(out, 'summary.json'), 'w'),
              indent=1, default=str)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='the 1160 generic agent: local test / one fold')
    ap.add_argument('--games', default='vc33,lp85')
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--config', default='A6', help='A0..A6 or a JSON dict')
    ap.add_argument('--cap-k', type=float, default=3.0)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args(argv)
    cfg = LADDER.get(a.config) or json.loads(a.config)
    tasks = [(g, cfg, s, a.cap_k, cfg_id(cfg), 'test') for g in a.games.split(',') for s in SEEDS[:a.seeds]]
    t0 = time.time()
    for r in map(_task, tasks):
        lv = ' '.join(f"L{l['L']}:{l['actions']}{'' if l['cleared'] else 'x'}" for l in r['levels'])
        print(f"{r['game']} seed {r['seed']}: cleared {r['cleared']} | {lv} | actions {r['actions']} resets {r.get('resets')} "
              f"deaths {r.get('deaths')} states {r.get('states')} score {r['score']:.4f} {r.get('status')} {r.get('wall')} s"
              + (f" ERROR {r['error']}" if r.get('error') else ''), flush=True)
    print(f'{time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
