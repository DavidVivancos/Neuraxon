"""prior1178.py -- NAS 1178 change 3: AN EFFECT MODEL THAT PREDICTS FROM THE FIRST STEP OF LEVEL 1.

WHAT 1177 MEASURED (nas_1177_report.md section 10): effect_ctx 3 repaired the model on levels 2+ and predicted NOTHING on level 1 (0 of
577,902 level-1 steps): a level's first board has no tables to back off to. Every untried action is worth the same flat 0.8 (while
probing) or 0.3 to the steering, so the failing level-1 plays spend their actions on cells that do nothing (ft09 76%, lp85 50%, sb26
39% no-change).

WHAT 1178 DOES (gene prior78, pin MULTINXON_PIN_PRIOR78; l1probe78 0-8, pin MULTINXON_PIN_L1PROBE78):
  * A CROSS-GAME PRIOR OF OBJECT TYPES (prior_1178.json, tools/prior1178_build.py): what clicking an object of a type, pressing a key or
    clicking the floor did when the games were probed at their level starts -- the type is the object's colour, its colour's rarity on
    the board (unique / few / many), its size class, its shape (dot, line, filled box, sparse frame, blob) and whether it touches the edge,
    backed off to coarser types where the counts are thin. Read LEAVE-ONE-GAME-OUT: a game's own probes never inform its prior.
  * THE VALUE OF AN UNTRIED ACTION is ORDERED by the prior around the steering's own novelty value u0 (0.8 while probing, 0.3 otherwise):
    u0 + 0.5 (P - Pbar) while probing, u0 + 0.2 (P - Pbar) otherwise, Pbar = the table's mean for that kind (keys / clicks) -- so when the
    steering replaces a wasted move (the veto, on level 1 too) it goes to the type most likely to do something, the types that never
    changed anything anywhere come last (Reki's dead signatures), and the brains' balance of exploring vs exploiting stays (the first
    design, 0.3 + 0.7 P, lowered every untried click: 11 level-1 k0 lines x 8 seeds, 10 clears -> 4 with the probes; VALUE_MODE 'abs').
    The game's own observations override it as they arrive (a signature seen once is the steering's own business again).
  * THE PROBES (l1probe78 = N): during the first 2N + 4 decisions of a level the play ENTERED (a later level of the same play -- never the
    level it starts on, at k0 or an end start: the line's brain is trained there; measured on 11 level-1 k0 lines x 8 seeds, probes there
    took the clears from 10 to 3), an untried action of the brain whose type the prior rates >= 0.3 below the best untried type is
    swapped for that type, N times at most (each swap a probe, the rest of the level the brain's).
  * PROBE, THEN TAKE BACK: a probe that changed the board is taken back with ACTION7 at the next step where the game undoes it (undo77
    on and its undo confirmed in this play) -- the effect is learnt, the level goes on from the board it had (st_p8_back); a probe that
    changed NOTHING ends the probes of that level (the prior is wrong there: st_p8_miss) -- one wasted action at most.
  * THE GAME'S EVIDENCE OVERRULES THE PRIOR: when the types the prior rated likely (P >= 0.4) changed nothing on their first two tries,
    or two types it rated dead (P < 0.1) changed the board, the prior is off for the rest of the play (flat values, no probes;
    st_p8_off) -- a cross-game prior can be wrong for one game (ft09's effective cells are the 'many large' type it rates 0.03).
  * the bus's obj block (bus1178) reads the same prior for the objects whose type the level has not tried.
  * THE CALIBRATION (the level-1 drop bench of change 3): every first try of a type in a level is tallied against the P the prior gave it
    (st_p8_cal<bin>_n / _c, bins P < 0.1 / 0.3 / 0.6 / 1; level 1 apart as st_p8_cal1<bin>): the run measures the prior where it acts.
Inert unless prior78 / l1probe78 is on. TALLY: st_p8_val (values from the prior), st_p8_probe (level-1 probe swaps), st_p8_back (probes
taken back), st_p8_miss (a probe that changed nothing: no more probes on the level), st_p8_off (the prior off for the play), st_p8_dead (a type
the prior rates < 0.05 given its last place), st_p8_cal*."""
from __future__ import annotations

import json
import os

import numpy as np

VERSION = 1
M_SHRINK = 3.0
VALUE_MODE = os.environ.get('MULTINXON_PRIOR78_VALUE', 'shift')   # 'shift' (mean-preserving order) | 'abs' (the first design: 0.3 + 0.7 P)
_TAB = {'doc': None, 'path': None, 'tot': None, 'logo': {}}


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def _pin(name):
    return os.environ.get('MULTINXON_PIN_' + name.upper(), '')


def gene(params, name):
    e = _pin(name)
    try:
        if str(e).strip() != '':
            return int(float(e))
        v = getattr(params, name, None)
        return int(float(v)) if v not in (None, '') else 0
    except Exception:
        return 0


# ======================================================================================================== the type keys (builder and play)
def _sbk(n):
    return 0 if n <= 1 else 1 if n <= 4 else 2 if n <= 16 else 3 if n <= 64 else 4 if n <= 256 else 5


def _rar(cnt):
    return 'u' if cnt <= 1 else ('f' if cnt <= 3 else 'm')


def _shape(n, h, w):
    if n <= 1:
        return 'dot'
    if h == 1 or w == 1:
        return 'line'
    if n >= h * w:
        return 'box'
    if n < 0.5 * h * w:
        return 'frame'
    return 'blob'


def _edge(y0, x0, h, w, H, W, m=2):
    return 1 if (y0 < m or x0 < m or y0 + h > H - m or x0 + w > W - m) else 0


def click_keys(col, n, cnt, h, w, y0, x0, H, W):
    """the four keys of a clicked object, most specific first"""
    r = _rar(cnt); s = _sbk(n); sh = _shape(n, h, w); e = _edge(y0, x0, h, w, H, W)
    return [f'C|{int(col)}|{r}|{s}|{sh}|{e}', f'C|{r}|{s}|{sh}|{e}', f'C|{r}|{s}', 'C']


def key_keys(a):
    return [f'K|{int(a)}', 'K']


def floor_keys():
    return ['P', 'C']


# ======================================================================================================== the table (leave one game out)
def _path():
    return os.environ.get('MULTINXON_PRIOR78_PATH') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prior_1178.json')


def table(game=None):
    """{key: [n, changed, died, level]} summed over every game but `game`"""
    p = _path()
    if _TAB['path'] != p:
        try:
            _TAB['doc'] = json.load(open(p, encoding='utf-8'))
        except Exception:
            _TAB['doc'] = {'counts': {}}
        _TAB['path'] = p; _TAB['tot'] = None; _TAB['logo'] = {}
    C = (_TAB['doc'] or {}).get('counts') or {}
    if _TAB['tot'] is None:
        tot = {}
        for g, t in C.items():
            for k, r in t.items():
                x = tot.setdefault(k, [0, 0, 0, 0])
                for i in range(4):
                    x[i] += int(r[i]) if i < len(r) else 0
        _TAB['tot'] = tot
    if not game or game not in C:
        return _TAB['tot']
    if game not in _TAB['logo']:
        own = C[game]
        out = {}
        for k, r in _TAB['tot'].items():
            o = own.get(k)
            out[k] = [r[i] - (int(o[i]) if (o and i < len(o)) else 0) for i in range(4)]
        _TAB['logo'][game] = out
    return _TAB['logo'][game]


def p_of(keys, T):
    """P(change) for the keys (most specific first), each level shrunk to the coarser one: p = (c + m p_parent) / (n + m)"""
    p = 0.5
    for k in reversed(keys):
        r = T.get(k)
        if r and r[0] > 0:
            p = (r[1] + M_SHRINK * p) / (r[0] + M_SHRINK)
    return float(p)


def n_of(keys, T):
    r = T.get(keys[0])
    return int(r[0]) if r else 0


class Prior:
    """the prior as one brain reads it: the table without its game, P(change) per object / key / floor"""

    def __init__(self, game=None):
        self.game = game
        self.T = table(game)

    def p_click(self, o, cnt, H, W):
        return p_of(click_keys(o['col'], o['n'], cnt, o.get('h', 1), o.get('w', 1), o.get('y0', 0), o.get('x0', 0), H, W), self.T)

    def p_key(self, a):
        return p_of(key_keys(a), self.T)

    def p_floor(self):
        return p_of(floor_keys(), self.T)

    def pbar(self, kind):
        """the table's mean P(change) of a kind of action ('K' keys, 'C' clicks and the floor): the level the shift keeps"""
        c = self.__dict__.setdefault('_pbar', {})
        if kind not in c:
            c[kind] = p_of(['K'] if kind == 'K' else ['C'], self.T)
        return c[kind]

    def p_effect(self, col, n, cnt, o, H=64, W=64):
        try:
            return self.p_click(o, cnt, H, W)
        except Exception:
            return None


def _game_of(st):
    for k in ('MULTINXON_SENSE_TRUTH', 'MULTINXON_HOME_GAME'):
        v = str(os.environ.get(k, '') or '').split(',')[0].strip()
        if v:
            return v
    g = getattr(st, 'home_game', None)
    if isinstance(g, str) and g:
        return g
    if getattr(st, 'sense_final', False) and getattr(st, 'sense_game', None):
        return st.sense_game
    return None


def prior_of(st):
    P = getattr(st, '_p78_P', None)
    if P is None:
        P = Prior(_game_of(st))
        st._p78_P = P
    return P


def p_action(st, akey, O, g=None):
    """the prior's P(change) for an action on objects O (a click: the object sig_of would name)"""
    P = prior_of(st)
    if akey[0] == 'K':
        return P.p_key(akey[1])
    y, x = akey[1], akey[2]
    best = None; bd = 1e9
    for o in O or []:
        d = max(abs(o['cy'] - y), abs(o['cx'] - x))
        if d < bd:
            bd, best = d, o
    if best is None or bd > 12:
        return P.p_floor()
    cnt = sum(1 for o in O if o['col'] == best['col'])
    H, W = (np.asarray(g).shape if g is not None else (64, 64))
    return P.p_click(best, cnt, H, W)


# ======================================================================================================== the steering
def install(Steer4, ArcBrain, Steer5=None):
    if getattr(Steer4, '_v1178_prior', False):
        return True
    Steer4._v1178_prior = True
    _init0 = Steer4.__init__
    _value0 = Steer4.value
    _start0 = Steer4.start_level

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        self.prior78 = gene(params, 'prior78'); self.l1probe78 = max(0, min(8, gene(params, 'l1probe78')))
        self._p78_P = None; self._p78_n = 0; self._p78_probes = 0; self._p78_in = False; self._p78_last = None
        self._p78_ev = [0, 0, 0, 0]; self._p78_off = False                # the play's evidence: [hi tried, hi changed, lo tried, lo changed]

    def start_level(self, g):
        first = not getattr(self, '_p78_started', False)
        _start0(self, g)
        self._p78_started = True
        self._p78_n = 0; self._p78_probes = 0; self._p78_last = None
        # probes only on a level the play ENTERED (a later level of the same play): never on the level the play starts on -- k0 or an
        # end start, the line's brain is trained there (measured: 11 level-1 k0 lines x 8 seeds, the probes took 10 clears to 3)
        self._p78_fresh = not first

    def value(self, sig, O, akey):
        v, p = _value0(self, sig, O, akey)
        if not getattr(self, 'prior78', 0) or p is not None or getattr(self, '_p78_off', False):
            return v, p
        try:
            if self.sig_n.get(sig, 0):
                return v, p                                               # the game's own observations rule
            probing = self.t <= self.miss_until or len(self.sig_n) < 4
            u0 = 0.8 if probing else 0.3
            g = self._cur[0] if getattr(self, '_cur', None) else None
            P = p_action(self, akey, O, g)
            if VALUE_MODE == 'abs':
                u1 = (0.3 + 0.7 * P) if probing else (0.1 + 0.4 * P)
            else:                                                         # 'shift': the prior ORDERS the untried actions of a kind
                pb = prior_of(self).pbar('K' if akey[0] == 'K' else 'C')  # around the steering's own novelty value, never lowers
                u1 = max(0.05, u0 + (0.5 if probing else 0.2) * (P - pb))  # the kind's mean (the exploration balance stays)
            if P < 0.05:
                _tally('p8_dead')
            _tally('p8_val')
            return v - u0 + u1, p
        except Exception as e:
            self._v1178p_error = f'value: {type(e).__name__}: {e}'[:160]
            return v, p

    def decide_outer(self, orig, brain, frame, a, d):
        if self._p78_in or not getattr(self, 'l1probe78', 0):
            return orig(self, brain, frame, a, d)
        self._p78_in = True
        try:
            out = orig(self, brain, frame, a, d)
        finally:
            self._p78_in = False
        try:
            lp = getattr(self, '_p78_last', None)
            self._p78_last = None
            if lp is not None and self._cur and self.prev is not None and self.prev[0] == lp:
                # the probe CHANGED NOTHING: the cross-game prior is wrong on this level (e.g. ft09, whose effective cells are the
                # 'many large' type the prior rates dead) -- no more probes here; the brain plays (st_p8_miss)
                self._p78_probes = max(self._p78_probes, self.l1probe78)
                _tally('p8_miss')
            elif lp is not None and self._cur and self.prev is not None and getattr(self, '_from_brain', True):
                # PROBE, THEN TAKE BACK (change 3): the probe of the last step changed the board -> ACTION7 at once where the game undoes
                # (undo77 on, its undo confirmed this play); the effect stays in the steering's tables, the level goes on from its board
                k1 = self.prev[0]; akey1 = self.prev[1]
                e77 = getattr(self, '_e77', None) or {}
                if k1 != lp and akey1 != ('K', 7) and (getattr(self, 'g77', None) or {}).get('undo77') and e77.get('undo_state') == 1:
                    try:
                        avail = {int(getattr(x, 'value', x)) for x in brain._masked_avail(frame)}
                    except Exception:
                        avail = set()
                    if 7 in avail:
                        _tally('p8_back')
                        return self._swap(('K', 7), ('K', 7), 'p8_back')
            if not self._cur or self.prev is None or self._p78_probes >= self.l1probe78 or not getattr(self, '_p78_fresh', False) \
                    or getattr(self, '_p78_off', False):
                return out
            self._p78_n += 1
            if self._p78_n > 2 * self.l1probe78 + 4:
                return out
            k, akey, cls = self.prev[0], self.prev[1], self.prev[2]
            nd = self.graph.get(k)
            if nd is None or not getattr(self, '_from_brain', True):
                return out                                                # the steering already chose this step
            O = self.objs.get(k) or []
            g = self._cur[0]
            if self.sig_n.get(self.sig_of(akey, O), 0):
                return out
            pb = p_action(self, akey, O, g)
            best = None; bp = pb; seen = {self.sig_of(akey, O)}
            for a2, cl in nd['cands']:
                if a2 in nd['tried'] or self.deadly(cl):
                    continue
                s2 = self.sig_of(a2, O)
                if s2 in seen or self.sig_n.get(s2, 0):
                    continue
                seen.add(s2)
                p2 = p_action(self, a2, O, g)
                if p2 > bp:
                    best, bp = (a2, cl), p2
            if best is not None and bp >= pb + 0.3:
                self._p78_probes += 1
                self._p78_last = k                                        # the board the probe leaves (its take-back is checked next step)
                return self._swap(best[0], best[1], 'p8_probe')
        except Exception as e:
            self._v1178p_error = f'probe: {type(e).__name__}: {e}'[:160]
        return out

    _after0 = Steer4._after_action

    def _after_action(self, ak, g0, g1, ch, trans):
        """the prior's calibration (the level-1 drop bench of change 3): an action whose type this level had not tried -- did it change
        anything, against the P the prior gave it (tallies p8_cal{bin}_n / _c, level 1 apart as p8_cal1{bin})"""
        if not getattr(self, 'prior78', 0) or ak is None or g0 is None:
            return _after0(self, ak, g0, g1, ch, trans)
        P = sig = None
        try:
            k0 = self.key(g0); O0 = self.objs.get(k0)
            if O0 is None:
                from steer1162 import components
                O0 = components(g0, self.bg, self.ignore())
            sig = self.sig_of(ak, O0)
            if not self.sig_n.get(sig, 0):
                P = p_action(self, ak, O0, g0)
        except Exception:
            P = None
        r = _after0(self, ak, g0, g1, ch, trans)
        if P is not None:
            try:
                b = 0 if P < 0.1 else (1 if P < 0.3 else (2 if P < 0.6 else 3))
                c = 1 if self.sig_chg.get(sig, 0) else 0
                lv = '1' if not (self.level and self.level >= 1) else ''
                _tally(f'p8_cal{lv}{b}_n'); _tally(f'p8_cal{lv}{b}_c', c)
                # THE GAME'S EVIDENCE OVERRULES THE PRIOR: the types it rated likely (P >= 0.4) changed nothing twice in a row of first
                # tries, or the types it rated dead (P < 0.1) changed the board twice -- the cross-game prior is wrong for this game:
                # off for the rest of the play (the flat values, no probes; st_p8_off)
                ev = self._p78_ev
                if P >= 0.4:
                    ev[0] += 1; ev[1] += c
                elif P < 0.1:
                    ev[2] += 1; ev[3] += c
                if not self._p78_off and ((ev[0] >= 2 and ev[1] == 0) or ev[3] >= 2):
                    self._p78_off = True
                    _tally('p8_off')
            except Exception:
                pass
        return r

    Steer4.__init__ = __init__; Steer4.value = value; Steer4.start_level = start_level; Steer4._after_action = _after_action
    _dec4 = Steer4.decide

    def decide4(self, brain, frame, a, d):
        return decide_outer(self, _dec4, brain, frame, a, d)
    Steer4.decide = decide4
    if Steer5 is not None and not getattr(Steer5, '_v1178_prior', False) and Steer5.decide is not decide4:
        Steer5._v1178_prior = True
        _dec5 = Steer5.decide

        def decide5(self, brain, frame, a, d):
            return decide_outer(self, _dec5, brain, frame, a, d)
        Steer5.decide = decide5
    _binit0 = ArcBrain.__init__

    def binit(self, params, *a, **k):
        _binit0(self, params, *a, **k)
        try:
            st = getattr(self, '_steer', None)
            if st is not None and (getattr(st, 'prior78', 0) or (getattr(self, '_b78_act', None) or {}).get('obj')):
                self._p78_prior = prior_of(st)                            # the bus's obj block reads it too
        except Exception:
            pass
    ArcBrain.__init__ = binit
    return True
