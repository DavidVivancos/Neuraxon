"""cap1166.py -- NAS 1166: the LIVE SET DECAYS (nas_1165_report.md §21.6). Installed on ArcBrain after cap1165.

WHY. 1165 made every brain universal: an action outside the bred preference that once changed the board joins the play's
LIVE set and stays there. The action flow showed the cost: on ls20 the opened key 2 was 8% of the actions of cleared plays
and 21% of failed ones; on su15 the opened action 7 was 3% vs 8%. An action that worked once kept being used after it
stopped working.

WHAT (on when the capability is on and cap_decay > 0):
  * an action in the live set that is NOT in the bred preference and is executed N times in a row without changing the
    board's interior (nor completing a level) leaves the live set (st_cap_decay counts it); it can come back the next time
    it changes the board (cap1165 re-adds it);
  * N = the gene cap_decay (0 = never, 1165's behaviour); the default when a brain does not carry it is the cell's knob
    (cap_decay -> MULTINXON_CAP_DECAY); MULTINXON_PIN_CAP_DECAY overrides (the lab).
The preferred actions never decay: they are the family's habit, not a discovery."""
from __future__ import annotations

import collections
import os

import cap1165 as C5

_STATE = {'brain': False}


def decay_n(params=None):
    for v in (os.environ.get('MULTINXON_PIN_CAP_DECAY'), getattr(params, 'cap_decay', None) if params is not None else None,
              os.environ.get('MULTINXON_CAP_DECAY')):
        if v not in (None, ''):
            try:
                return max(0, int(float(v)))
            except Exception:
                continue
    return 0


def install(ArcBrain):
    if getattr(ArcBrain, '_v1166_cap', False):
        return True
    if not getattr(ArcBrain, '_v1165_cap', False):
        C5.install(ArcBrain)
    ArcBrain._v1166_cap = True
    _init0 = ArcBrain.__init__
    _ps0 = ArcBrain.play_step

    def __init__(self, params, *a, **k):
        _init0(self, params, *a, **k)
        self._cap_decay = decay_n(params) if getattr(self, '_cap_on', False) else 0
        self._cap_nc = collections.Counter()

    def play_step(self, frame):
        n = getattr(self, '_cap_decay', 0)
        prev = getattr(self, '_cap_prev', None)
        if n > 0 and prev is not None:
            g0, v0, lv0 = prev
            pref = getattr(self, '_cap_pref', None)
            if v0 > 0 and v0 in self._cap_live and (pref is None or v0 not in pref):
                lv = int(getattr(frame, 'levels_completed', 0) or 0)
                if lv > lv0 or C5._interior_changed(g0, C5._grid_of(frame)):
                    self._cap_nc[v0] = 0
                else:
                    self._cap_nc[v0] += 1
                    if self._cap_nc[v0] >= n:
                        self._cap_live.discard(v0); self._cap_nc[v0] = 0
                        self._cap_n['decay'] += 1; C5._tally('cap_decay')
        return _ps0(self, frame)

    __init__.__doc__ = _init0.__doc__
    ArcBrain.__init__ = __init__; ArcBrain.play_step = play_step
    _STATE['brain'] = True
    return True
