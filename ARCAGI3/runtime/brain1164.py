"""brain1164.py -- NAS 1164: BRAIN-DRIVEN EXPLORATION (gene net_explore; nas_1164_plan.md §4 item 4, priority P1).

WHY. Phase 0 played 20 different cd82 brains on the same seed: 12 / 20 made the same 5-action r11l clear, 10 / 20 the same
28-action lp85 clear. 1163's brain share of the clear variance was its permutation null on 22 of 25 lines. When the motor
readout selects nothing (most steps of a silent network) or the idle rule fires, ArcBrain._explore draws the action from
self.rng UNIFORMLY -- the seed decides, and the brain does not. Evolution cannot select what the brain does not do.

WHAT. With gene net_explore = beta > 0 (or MULTINXON_PIN_NET_EXPLORE), the exploratory action is drawn from
softmax(beta * s) over the playable actions, where s is the motor output neurons' MEMBRANE POTENTIAL read the way the
trinary readout reads it (sub-threshold: the network's lean, not its decision):

    ACTION1 = -p0   ACTION2 = +p0   ACTION3 = -p1   ACTION4 = +p1   ACTION5 = p2   ACTION6 = p3   ACTION7 = p4

normalised by the largest |s| so beta is a temperature on [-1, 1] scores. A click goes to the network's own cursor (the
cand_cursor selection when that gene is on), not a random cell. self.rng draws (deterministic per seed). beta = 0 is the
1163 behaviour bit for bit (the original _explore runs). Counted per try as st_netx (the adapter tally).

The steering still decides afterwards (veto, plan, v5 authority): this changes only WHO proposes when the brain is silent."""
from __future__ import annotations

import math
import os

_INSTALLED = {'done': False}
_ORDER = ('ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6', 'ACTION7')


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def _beta_of(params):
    v = os.environ.get('MULTINXON_PIN_NET_EXPLORE')
    try:
        if v not in (None, ''):
            return max(0.0, float(v))
    except Exception:
        pass
    try:
        return max(0.0, float(getattr(params, 'net_explore', 0.0) or 0.0))
    except Exception:
        return 0.0


def potentials(brain):
    """the motor output neurons' membrane potentials (7 values, 0 where missing)"""
    try:
        ns = list(brain.motor.network.output_neurons)
        p = [float(getattr(n, 'membrane_potential', 0.0) or 0.0) for n in ns]
    except Exception:
        p = []
    return (p + [0.0] * 7)[:7]


def scores(p):
    p0, p1, p2, p3, p4 = p[0], p[1], p[2], p[3], p[4]
    return {'ACTION1': -p0, 'ACTION2': p0, 'ACTION3': -p1, 'ACTION4': p1, 'ACTION5': p2, 'ACTION6': p3, 'ACTION7': p4}


def choose(brain, playable, beta):
    """softmax(beta * normalised score) over the playable actions, drawn with the brain's own rng"""
    S = scores(potentials(brain))
    acts = [a for a in playable if getattr(a, 'name', '') in S]
    if not acts:
        return None
    raw = [S[a.name] for a in acts]
    m = max(1e-9, max(abs(x) for x in raw))
    z = [beta * x / m for x in raw]
    mx = max(z)
    w = [math.exp(v - mx) for v in z]
    tot = sum(w)
    r = brain.rng.random() * tot
    acc = 0.0
    for a, wi in zip(acts, w):
        acc += wi
        if r <= acc:
            return a
    return acts[-1]


def install(ArcBrain):
    """wrap ArcBrain.__init__ (the gene) and ArcBrain._explore (the choice); idempotent"""
    if getattr(ArcBrain, '_v1164_netx', False):                      # per class object: a re-imported arc_brain is patched again
        return True
    ArcBrain._v1164_netx = True
    ArcBrain._v1164_explore0 = ArcBrain._explore                    # the 1163 exploration (preflight 83 compares against it)
    _init0 = ArcBrain.__init__
    _explore0 = ArcBrain._explore

    def __init__(self, params, *a, **k):
        _init0(self, params, *a, **k)
        self._net_explore = _beta_of(params)
        self._netx_n = 0

    def _explore(self, frame, h, w):
        beta = float(getattr(self, '_net_explore', 0.0) or 0.0)
        if beta <= 0.0:
            return _explore0(self, frame, h, w)
        try:
            from arc_env.structs import GameAction
            playable = [a for a in self._masked_avail(frame) if a is not GameAction.RESET]
            if not playable:
                return GameAction.RESET, {}
            a = choose(self, playable, beta)
            if a is None:
                return _explore0(self, frame, h, w)
            data = {}
            if a.is_complex():
                data = self._cursor_click(h, w)                       # the network's cursor / selected candidate
            self._netx_n += 1
            _tally('netx')
            return a, data
        except Exception as _e:
            self._netx_error = f'{type(_e).__name__}: {_e}'[:160]
            return _explore0(self, frame, h, w)

    __init__.__doc__ = _init0.__doc__
    ArcBrain.__init__ = __init__
    ArcBrain._explore = _explore
    _INSTALLED['done'] = True
    return True
