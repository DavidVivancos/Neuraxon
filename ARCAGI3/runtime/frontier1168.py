"""frontier1168.py -- NAS 1168: the frontier acts EARLY (FrameLab 6) and survives boards that move faster than its mask.

WHY (FrameLab 6, `outFramelab6/REPORT.md`, 549 paired level runs x 3 seeds per variant):
  * the 1167 frontier (a proposer + one plan after 16 steps without a new board) is no better than random play: +0.7 clears per
    seed (p 0.75); after 4 steps +2.7 (p 0.077); after ONE step +8.0 (p < 0.001), the frontier then taking 40% of the actions
    (631 takeovers per level run), 23 levels by any seed (random 9): lp85 L1-L3, L5, L7, L8, tu93 L2-L4, L7, L9, su15 L1, L6,
    ar25, bp35, cd82, ft09, lf52 L1;
  * a goal acting on proposals at the structured goal's precision (0.74, recall 0.43) does not help (-1.0 per seed vs random;
    the learned goal +1.3, n.s.); only the TRUE finished board does (+11.7). The goal stays a lab arm; the frontier is the lever.
  * NAS 1167: on g50t 1.2 stalls per 100 steps found no frontier at all (the board keys move faster than the mask settles; 59%
    of the quick-lab stalls).

WHAT CHANGES (patches on frontier1167; inert unless the frontier is on):
  * frontier_stall may be 1 (1167 clamped it at >= 2): the frontier takes over at the FIRST step without a new board;
  * MULTINXON_FRONTIER_STALL_DEFAULT: the stall of a brain whose genome has no frontier_stall (the cells where the frontier is
    forced: the keys cells, su15-l1x) -- 2 in the 1168 fleet;
  * COARSE BOARDS: when, after >= 8 stalls, more than half of the stalls found no frontier to route to, the graph keys the
    boards at half resolution (every second row and column, the clock mask applied) and is rebuilt from the stored
    transitions; tally st_fr_coarse."""
from __future__ import annotations

import os

import numpy as np

import frontier1167 as F7


def _default_stall():
    try:
        v = os.environ.get('MULTINXON_FRONTIER_STALL_DEFAULT', '')
        return int(float(v)) if v not in ('', None) else None
    except Exception:
        return None


def install():
    """idempotent: patch frontier1167 (the module the brain hooks call) in this process"""
    if getattr(F7, '_v1168', False):
        return True
    F7._v1168 = True
    _gene0 = F7._gene

    def _gene(params, name, default):
        v = getattr(params, name, None)
        if name == 'frontier_stall' and v in (None, ''):
            d = _default_stall()
            if d is not None:
                return d
        return _gene0(params, name, default)
    F7._gene = _gene

    Fr = F7.Frontier
    _init0 = Fr.__init__; _key0 = Fr.key; _decide0 = Fr.decide

    def __init__(self, stall=16, probe=1, *a, **k):
        _init0(self, stall, probe, *a, **k)
        self.stall_n = max(1, int(stall))                     # 1167: max(2, .); FrameLab 6: one step pays most
        self.coarse = False

    def key(self, f):
        if not getattr(self, 'coarse', False):
            return _key0(self, f)
        g = f if self.mask is None or self.mask.shape != f.shape else np.where(self.mask, -1, f)
        g = np.ascontiguousarray(g[::2, ::2].astype(np.int16))
        import hashlib
        return hashlib.blake2b(g.tobytes(), digest_size=12).digest()

    def decide(self, frame, a, d, avail_vals):
        out = _decide0(self, frame, a, d, avail_vals)
        try:
            if not self.coarse and self.cnt.get('stall', 0) >= 8 and self.cnt.get('none', 0) > 0.5 * self.cnt.get('stall', 0):
                self.coarse = True
                self._rebuild(); self._t('coarse')
        except Exception:
            pass
        return out

    Fr.__init__ = __init__; Fr.key = key; Fr.decide = decide
    return True
