"""goalmap1163.py -- NAS 1163: the GOAL MAP a brain learns from the boards it has seen (change 3 of nas_1162_report §19).

WHAT IT IS (the 1158 research, goal_cells.py / goal_cells_chain.py, ported to numpy so every box can run it)
  * When a level completes, its START board and its FINISHED board are known (the finished board is the first frame of
    the completing response when that frame is local -- the research's goal_data rule). The cells that CHANGED and END
    non-background are the level's "arrivals": something had to end up there.
  * A small per-game cell model learns, from the start board alone, which cells will be arrivals. Features per cell:
    colour (one-hot), 3x3 and 7x7 colour mixes, the colour's rarity, the distance to the rarest colour and to any object,
    inside a hollow outline, the log size of its component, position, background. No GPU, no sklearn: a histogram
    gradient-boosted tree model (numpy), ~1 s per level on one CPU.
  * On the next level, the model predicts that level's goal map from its start board. Measured on the 25 games' verified
    levels (level 1 alone -> level 2): mean IoU 0.28 (sklearn HistGradientBoosting 0.26; distance-to-rarest baseline 0.06);
    sb26 0.97, wa30 0.58, tn36 0.53, tu93 0.54, sc25 0.55, ar25 0.42, lf52 0.41, g50t 0.36, ka59 0.33, ft09 0.32, tr87 0.33.
    From two levels -> level 3: 0.36.
  * Goals transfer WITHIN a game, not across games (the research: 0.12 across) -- the model lives as long as the brain's
    play of ONE game and is never shared between games or plays.
The guide the steering uses (steer1163.Steer4): h = the predicted goal cells not yet arrived, ROUNDED (6 buckets) so
novelty stays meaningful -- the research's 'wh' order: novelty first, then h."""
from __future__ import annotations

import hashlib
import math

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:                                   # a box without a working scipy: pure-numpy fallbacks below
    _ndi = None


# ------------------------------------------------------------------------------------------------- numpy fallbacks
def _uniform(a, k):
    """mean filter of size k (odd) over axes 0, 1 of a (H, W, C) array, zero padding like ndimage 'reflect' is not needed
    for a ranking feature; edges are averaged over the cells that exist"""
    if _ndi is not None:
        return _ndi.uniform_filter(a, size=(k, k, 1))
    r = k // 2; H, W = a.shape[:2]
    p = np.pad(a, ((r + 1, r), (r + 1, r), (0, 0)))
    c = p.cumsum(0).cumsum(1)
    s = c[k:k + H, k:k + W] - c[0:H, k:k + W] - c[k:k + H, 0:W] + c[0:H, 0:W]
    ones = np.pad(np.ones((H, W, 1), a.dtype), ((r + 1, r), (r + 1, r), (0, 0))).cumsum(0).cumsum(1)
    n = ones[k:k + H, k:k + W] - ones[0:H, k:k + W] - ones[k:k + H, 0:W] + ones[0:H, 0:W]
    return s / np.maximum(n, 1)


def _cheb_dist(mask_false_is_target, cap=20):
    """chessboard distance to the nearest cell where the mask is False (ndimage.distance_transform_cdt semantics)"""
    if _ndi is not None:
        return _ndi.distance_transform_cdt(mask_false_is_target, metric='chessboard').astype(np.float32)
    m = mask_false_is_target.copy(); d = np.zeros(m.shape, np.float32); cur = ~m
    for i in range(1, cap + 1):
        p = np.pad(cur, 1)
        grown = np.zeros_like(cur)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= p[1 + dy:1 + dy + cur.shape[0], 1 + dx:1 + dx + cur.shape[1]]
        new = grown & ~cur
        d[new] = i; cur = grown
        if cur.all():
            break
    d[~cur] = cap
    return d


def _label(mask):
    if _ndi is not None:
        return _ndi.label(mask)
    from agent1160 import label
    return label(mask)


def _fill_holes(m):
    if _ndi is not None:
        return _ndi.binary_fill_holes(m)
    H, W = m.shape; out = np.ones((H, W), bool); seen = np.zeros((H, W), bool); stack = []
    for y in range(H):
        for x in (0, W - 1):
            if not m[y, x] and not seen[y, x]:
                seen[y, x] = True; stack.append((y, x))
    for x in range(W):
        for y in (0, H - 1):
            if not m[y, x] and not seen[y, x]:
                seen[y, x] = True; stack.append((y, x))
    while stack:
        y, x = stack.pop(); out[y, x] = False
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < H and 0 <= xx < W and not m[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True; stack.append((yy, xx))
    return out


def bgc(f):
    return int(np.bincount(np.clip(f.ravel(), 0, 15), minlength=16).argmax())


def cell_features(f):
    """the research's goal-cell features of one board: (H*W, 56) float32"""
    f = np.clip(np.asarray(f, np.int64), 0, 15)
    H, W = f.shape; bg = bgc(f)
    oh = np.zeros((H, W, 16), np.float32)
    for c in range(16):
        oh[:, :, c] = (f == c)
    n3 = _uniform(oh, 3); n7 = _uniform(oh, 7)
    freq = np.bincount(f.ravel(), minlength=16)[:16] / float(f.size)
    own = freq[f]
    rare = [c for c in np.argsort(freq) if freq[c] > 0 and c != bg]
    d_rare = _cheb_dist(f != rare[0]) if rare else np.zeros((H, W), np.float32)
    nonbg = f != bg
    d_obj = _cheb_dist(~nonbg) if (nonbg.any() and (~nonbg).any()) else np.zeros((H, W), np.float32)
    inside = np.zeros((H, W), np.float32)
    for c in np.unique(f):
        if c == bg:
            continue
        m = f == c; inside[_fill_holes(m) & ~m] = 1.0
    lab_size = np.zeros((H, W), np.float32)
    for c in np.unique(f):
        lab, n = _label(f == c)
        if n:
            sizes = np.bincount(lab.ravel()); sel = f == c
            lab_size[sel] = np.log1p(sizes[lab[sel]])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = np.concatenate([oh, n3, n7, own[:, :, None], np.minimum(d_rare, 20)[:, :, None] / 20, np.minimum(d_obj, 20)[:, :, None] / 20,
                        inside[:, :, None], lab_size[:, :, None] / 8, (yy / H)[:, :, None], (xx / W)[:, :, None],
                        (f == bg)[:, :, None]], axis=2)
    return X.reshape(H * W, -1).astype(np.float32)


def arrivals(start, fin, ignore=None):
    """cells that changed between the level's start and finished boards and END non-background (HUD / clocks masked)"""
    start = np.asarray(start); fin = np.asarray(fin)
    bg = bgc(start)
    m = (start != fin) & (fin != bg)
    if ignore is not None and ignore.shape == m.shape:
        m &= ~ignore
    return m


def finished_board(pre, frames):
    """the research's goal_data rule: the first frame of the completing response that differs from the board before the
    completing action LOCALLY (< 25% of what the next level's board differs); None when the game jumps straight on"""
    try:
        pre = np.asarray(pre)
        frs = [np.asarray(x, np.int16) for x in (frames or [])]
        frs = [x for x in frs if x.shape == pre.shape]
    except Exception:
        return None
    if len(frs) < 2:
        return None
    dn = int((frs[-1] != pre).sum())
    for x in frs[:-1]:
        d = int((x != pre).sum())
        if 0 < d < 0.25 * max(dn, 1):
            return x
    return None


def hud_from_transitions(pairs, shape):
    """the research's hud_mask: cells that change on more than 30% of within-level transitions + border LINES (rows /
    columns 0-2 and the last 3) that change on more than 20% of them"""
    D = [np.asarray(a) != np.asarray(b) for a, b in pairs if np.asarray(a).shape == shape and np.asarray(b).shape == shape]
    if not D:
        return np.zeros(shape, bool)
    D = np.stack(D); m = D.mean(axis=0) > 0.3
    H, W = shape
    for y in list(range(0, 3)) + list(range(H - 3, H)):
        if D[:, y, :].any(axis=1).mean() > 0.2:
            m[y, :] = True
    for x in list(range(0, 3)) + list(range(W - 3, W)):
        if D[:, :, x].any(axis=1).mean() > 0.2:
            m[:, x] = True
    return m


# ------------------------------------------------------------------------------------------------- the model
class GBDT:
    """histogram gradient-boosted trees, binary logistic loss, Newton leaves, balanced classes -- numpy only"""

    def __init__(self, n_iter=60, depth=3, lr=0.15, bins=24, min_leaf=12, l2=1.0):
        self.n_iter, self.depth, self.lr, self.bins, self.min_leaf, self.l2 = n_iter, depth, lr, bins, min_leaf, l2
        self.trees = []; self.f0 = 0.0; self.edges = []

    def _bin(self, X):
        B = np.empty(X.shape, np.uint8)
        for j in range(X.shape[1]):
            B[:, j] = np.searchsorted(self.edges[j], X[:, j], side='right')
        return B

    def fit(self, X, y):
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float64)
        self.edges = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            e = (u[:-1] + u[1:]) / 2.0 if len(u) <= self.bins else np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.bins + 1)[1:-1]))
            self.edges.append(np.asarray(e, np.float32))
        B = self._bin(X); n = len(y)
        pos = max(1.0, y.sum()); neg = max(1.0, n - y.sum())
        w = np.where(y > 0, 0.5 / pos, 0.5 / neg) * n
        p0 = min(1 - 1e-4, max(1e-4, (w * y).sum() / w.sum()))
        self.f0 = float(math.log(p0 / (1 - p0))); F = np.full(n, self.f0); nb = self.bins + 1
        self.trees = []
        for _ in range(self.n_iter):
            p = 1.0 / (1.0 + np.exp(-F)); g = w * (p - y); h = w * p * (1 - p) + 1e-9
            tree = []

            def build(idx, d):
                G = g[idx].sum(); Hs = h[idx].sum()
                if d >= self.depth or len(idx) < 2 * self.min_leaf:
                    tree.append(('leaf', -G / (Hs + self.l2))); return len(tree) - 1
                best = None; base = G * G / (Hs + self.l2)
                Bi = B[idx]; gi = g[idx]; hi = h[idx]
                for j in range(B.shape[1]):
                    hg = np.bincount(Bi[:, j], weights=gi, minlength=nb); hh = np.bincount(Bi[:, j], weights=hi, minlength=nb)
                    hc = np.bincount(Bi[:, j], minlength=nb)
                    cg = np.cumsum(hg)[:-1]; ch = np.cumsum(hh)[:-1]; cc = np.cumsum(hc)[:-1]
                    ok = (cc >= self.min_leaf) & ((len(idx) - cc) >= self.min_leaf)
                    if not ok.any():
                        continue
                    gain = cg ** 2 / (ch + self.l2) + (G - cg) ** 2 / (Hs - ch + self.l2) - base
                    gain[~ok] = -1e18
                    t = int(np.argmax(gain))
                    if best is None or gain[t] > best[0]:
                        best = (gain[t], j, t)
                if best is None or best[0] <= 1e-9:
                    tree.append(('leaf', -G / (Hs + self.l2))); return len(tree) - 1
                _, j, t = best
                m = B[idx, j] <= t
                me = len(tree); tree.append(None)
                left = build(idx[m], d + 1); right = build(idx[~m], d + 1)
                tree[me] = (j, t, left, right); return me
            build(np.arange(n), 0)
            self.trees.append(tree)
            F += self.lr * self._tree_pred(tree, B)
        return self

    @staticmethod
    def _tree_pred(tree, B):
        out = np.zeros(len(B)); stack = [(0, np.arange(len(B)))]
        while stack:
            node, idx = stack.pop()
            nd = tree[node]
            if nd[0] == 'leaf':
                out[idx] = nd[1]; continue
            j, t, left, right = nd; m = B[idx, j] <= t
            stack.append((left, idx[m])); stack.append((right, idx[~m]))
        return out

    def predict_proba(self, X):
        B = self._bin(np.asarray(X, np.float32)); F = np.full(len(B), self.f0)
        for tr in self.trees:
            F += self.lr * self._tree_pred(tr, B)
        return 1.0 / (1.0 + np.exp(-F))


_CACHE = {}                                   # training-set digest -> fitted model (per process; plays repeat prefixes)


class GoalModel:
    """the goal map of ONE game, learned from the levels this play has seen complete"""

    def __init__(self, max_neg_ratio=4.0, seed=1163):
        self.levels = []                      # (start board, arrival mask)
        self.model = None; self.dirty = False; self.max_neg_ratio = max_neg_ratio; self.seed = seed

    def add(self, start, arrive):
        arrive = np.asarray(arrive, bool)
        if arrive.sum() < 3 or arrive.sum() > 0.6 * arrive.size:
            return False
        self.levels.append((np.asarray(start, np.int16).copy(), arrive.copy())); self.dirty = True
        return True

    def n_goal(self):
        """how many goal cells to expect on a new level: the mean arrival count of the levels seen"""
        return int(round(float(np.mean([a.sum() for _, a in self.levels])))) if self.levels else 0

    def fit(self):
        if not self.levels:
            return None
        if not self.dirty and self.model is not None:
            return self.model
        h = hashlib.md5()
        for s, a in self.levels:
            h.update(s.tobytes()); h.update(np.packbits(a).tobytes())
        key = h.hexdigest()
        m = _CACHE.get(key)
        if m is None:
            rng = np.random.default_rng(self.seed)
            Xs, ys = [], []
            for s, a in self.levels:
                X = cell_features(s); y = a.ravel()
                neg = np.nonzero(~y)[0]; posn = int(y.sum())
                keep_neg = neg if len(neg) <= self.max_neg_ratio * max(posn, 25) else rng.choice(neg, int(self.max_neg_ratio * max(posn, 25)), replace=False)
                idx = np.concatenate([np.nonzero(y)[0], np.sort(keep_neg)])
                Xs.append(X[idx]); ys.append(y[idx])
            m = GBDT().fit(np.concatenate(Xs), np.concatenate(ys))
            _CACHE[key] = m
            while len(_CACHE) > 8:
                _CACHE.pop(next(iter(_CACHE)))
        self.model = m; self.dirty = False
        return m

    def predict(self, board, ignore=None, n=None):
        """(goal cells: bool HxW, probabilities HxW) for a level whose start board is `board`"""
        m = self.fit()
        if m is None:
            return None, None
        b = np.asarray(board)
        p = m.predict_proba(cell_features(b)).reshape(b.shape)
        if ignore is not None and ignore.shape == p.shape:
            p = np.where(ignore, -1.0, p)
        n = int(n if n is not None else self.n_goal())
        n = max(1, min(n, int(0.5 * p.size)))
        flat = p.ravel(); top = np.argsort(-flat)[:n]
        goal = np.zeros(p.size, bool); goal[top] = True; goal &= flat > 0.0
        return goal.reshape(p.shape), p


def h_of(board, start, goal):
    """predicted goal cells NOT yet arrived (still as at the level start, or background)"""
    if goal is None:
        return None
    b = np.asarray(board); s = np.asarray(start)
    if b.shape != goal.shape or s.shape != goal.shape:
        return None
    bg = bgc(s)
    arrived = (b != s) & (b != bg)
    return int((goal & ~arrived).sum())
