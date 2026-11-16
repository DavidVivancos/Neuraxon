"""cap1165.py -- NAS 1165: UNIVERSAL CAPABILITY. Every brain can play every game: every action the game offers, keys and
clicks, anywhere on the board. Families, inheritance and evolution are kept; the capability is ADDED, never bred in or out.

WHY (nas_1164_report.md, the 1164 capability audit of the 129 champion brains):
  * the action mask was a HARD filter. A family bred on a click game ('6': lp85, r11l, s5i5, su15, tn36, vc33, ft09) can
    never press a key on a mixed game; a family bred on a keys game ('12345': ar25, g50t, re86, wa30) can never click.
    Kaggle measured it: re86 e_t256 (mask 12345) scored 0.05 because it never clicks in mixed games (12 of 25 public
    games offer both keys and clicks; the hidden set too).
  * the motor network could not even WANT some actions: of the 7 motor outputs (ACTION1/2 = -/+ o0, ACTION3/4 = -/+ o1,
    ACTION5 = o2, ACTION6 = o3, ACTION7 = o4), the 246-synapse family (sp80, dc22, ft09, ka59, lf52, wa30) has o0, o2, o5
    unwired; the lp85 family wires only o2 and o6; the ls20 family wires none; the r1162 vc33 family leaves o1, o4, o6
    unwired. Weight mutation perturbs existing synapses only, so an unwired output stays unwired forever.
  * the click band / colour (click_mask_min / max, click_colour) returned NO candidate on a board of other sizes and
    colours, and the click fell back to the cursor's pixel.

WHAT (all on when cap_univ = 1 / MULTINXON_CAP_UNIV = 1; nothing changes otherwise):
  1. THE MASK BECOMES A PREFERENCE. The bred mask is kept (self._cap_pref, inherited, the family's habit); the hard filter
     is removed (self._action_mask = None). Where the brain chooses:
       * its NETWORK's action: a non-preferred action is executed with probability cap_open (gene, 0..1, evolvable),
         or always once that action is known LIVE in this play;
       * EXPLORATION: drawn from the preferred + live actions, from every offered action with probability cap_explore;
       * the STEERING sees every offered action; a candidate whose action is neither preferred nor live scores cap_pen
         lower, so it is probed when the preferred ones are exhausted in a state (the steering's own effect model then
         keeps it if it works).
  2. THE LIVE SET IS LEARNED FROM THE BOARD, per play: an action that changed the board (outside a 2-cell HUD border) or
     completed a level joins the live set for the rest of the play (st_cap_live counts discoveries). This is the mask
     re-derived online for each game -- what the 1158 transcripts did by reading the action set of each game.
  3. CLICKS ANYWHERE: when the bred band / colour finds no candidate, every component of the board is a candidate.
  4. A WIRED MOTOR: every motor output gets >= cap_wire incoming synapses (default 3) from the motor sphere's hidden
     neurons (its inputs if it has none), weights +-cap_wire_w x U(0.5, 1.5) (default 0.03: below the firing threshold
     0.3, so the inherited behaviour barely moves). Applied deterministically (seeded by the motor's own synapse list) when
     a brain is loaded (MULTINXON_CAP_WIRE > 0) and saved into every child, so mutation and selection can grow them.

TALLIES (per try, the adapter's): st_cap_np (non-preferred actions executed), st_cap_live (actions found live),
st_cap_open (network actions let through by cap_open), st_cap_wired (synapses added at load).
REPORT: report(neural, brain) -> per game, the offered actions the brain can and cannot reach (with and without the layer),
and its unwired outputs. preflight1165 and the fleet audit refuse a 1165 cell whose champions are not universal."""
from __future__ import annotations

import collections
import hashlib
import os
import random

# the actions each public game offers on its first frame (the real engine, 12 Oct 2026)
GAME_ACTIONS = {'ar25': [1, 2, 3, 4, 5, 6, 7], 'bp35': [3, 4, 6, 7], 'cd82': [1, 2, 3, 4, 5, 6], 'cn04': [1, 2, 3, 4, 5, 6],
                'dc22': [1, 2, 3, 4, 6], 'ft09': [6], 'g50t': [1, 2, 3, 4, 5], 'ka59': [1, 2, 3, 4, 6], 'lf52': [1, 2, 3, 4, 6, 7],
                'lp85': [6], 'ls20': [1, 2, 3, 4], 'm0r0': [1, 2, 3, 4, 5, 6], 'r11l': [6], 're86': [1, 2, 3, 4, 5],
                's5i5': [6], 'sb26': [5, 6, 7], 'sc25': [1, 2, 3, 4, 6], 'sk48': [1, 2, 3, 4, 6, 7], 'sp80': [1, 2, 3, 4, 5, 6],
                'su15': [6, 7], 'tn36': [6], 'tr87': [1, 2, 3, 4], 'tu93': [1, 2, 3, 4], 'vc33': [6], 'wa30': [1, 2, 3, 4, 5]}
GROUP = {g: ('click' if set(a) <= {6, 7} else ('keys' if 6 not in a else 'mixed')) for g, a in GAME_ACTIONS.items()}
DEFAULTS = {'cap_open': 0.1, 'cap_explore': 0.1, 'cap_pen': 0.5, 'cap_wire': 3, 'cap_wire_w': 0.03}
HUD = 2                                             # cells of border ignored when deciding whether an action changed the board
_STATE = {'brain': False, 'loader': False, 'wired': 0, 'loads': 0, 'error': None}


# ======================================================================================================== genes
def _env(name):
    v = os.environ.get(name)
    return None if v in (None, '') else v


_PIN = {'cap_open': lambda: os.environ.get('MULTINXON_PIN_CAP_OPEN'), 'cap_explore': lambda: os.environ.get('MULTINXON_PIN_CAP_EXPLORE'),
        'cap_pen': lambda: os.environ.get('MULTINXON_PIN_CAP_PEN')}


def gene(params, name, default=None):
    """a capability gene: MULTINXON_PIN_<NAME> overrides (the lab), then the brain's gene, then the default"""
    v = (_PIN.get(name) or (lambda: None))()
    v = None if v in (None, '') else v
    if v is None:
        v = getattr(params, name, None) if params is not None else None
    if v in (None, ''):
        v = DEFAULTS.get(name, default) if default is None else default
    try:
        return float(v)
    except Exception:
        return float(DEFAULTS.get(name, 0.0))


def cap_on(params=None):
    v = os.environ.get('MULTINXON_CAP_UNIV')
    v = None if v in (None, '') else v
    if v is not None:
        try:
            return int(float(v)) == 1
        except Exception:
            return False
    try:
        return int(float(getattr(params, 'cap_univ', 0) or 0)) == 1
    except Exception:
        return False


def mask_set(mask):
    s = str(mask if mask is not None else '').strip()
    return set(int(c) for c in s if c.isdigit()) or None


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def _val(a):
    try:
        return int(getattr(a, 'value', a))
    except Exception:
        return -1


# ======================================================================================================== 4. the wired motor
_SYN_KEYS = ('associative_strength', 'axonal_delay', 'integrity', 'is_modulatory', 'is_silent', 'learning_rate', 'learning_rate_mod',
             'plasticity_threshold', 'post_trace', 'potential_delta_w', 'pre_trace', 'pre_trace_ltd', 'synapse_type', 'tau_fast',
             'tau_ltd', 'tau_ltp', 'tau_meta', 'tau_slow')


def _template(brain, net):
    for s in (net.get('synapses') or []):
        if isinstance(s, dict):
            return s
    sph = brain.get('spheres') or {}
    for sp in (sph.values() if isinstance(sph, dict) else sph):
        for s in ((sp or {}).get('network') or {}).get('synapses') or []:
            if isinstance(s, dict):
                return s
    p = net.get('parameters') or {}
    return {'associative_strength': 0.0, 'axonal_delay': 1.0, 'integrity': 1.0, 'is_modulatory': False, 'is_silent': False,
            'learning_rate': float(p.get('learning_rate', 0.007)), 'learning_rate_mod': 1.0,
            'plasticity_threshold': float(p.get('plasticity_threshold', 0.3)), 'post_trace': 0.0, 'potential_delta_w': 0.0,
            'pre_trace': 0.0, 'pre_trace_ltd': 0.0, 'synapse_type': 'ionotropic_fast', 'tau_fast': float(p.get('tau_fast', 5.0)),
            'tau_ltd': float(p.get('tau_ltd', 55.0)), 'tau_ltp': float(p.get('tau_ltp', 8.0)), 'tau_meta': float(p.get('tau_meta', 300.0)),
            'tau_slow': float(p.get('tau_slow', 50.0))}


def new_synapse(tpl, pre, post, w):
    s = {k: tpl.get(k) for k in _SYN_KEYS if k in tpl}
    s.update(pre_id=int(pre), post_id=int(post), w_fast=float(w), w_slow=float(w) * 0.5, w_meta=0.0, is_silent=False,
             is_modulatory=False, integrity=1.0, potential_delta_w=0.0, pre_trace=0.0, post_trace=0.0, pre_trace_ltd=0.0,
             associative_strength=0.0, neighbor_synapse_ids=[])
    s.setdefault('axonal_delay', 1.0)
    return s


def motor_net(brain):
    sph = (brain or {}).get('spheres') or {}
    sp = sph.get('motor') if isinstance(sph, dict) else None
    return ((sp or {}).get('network') or None) if sp else None


def unwired(brain, m=1):
    """the motor outputs (first 7, the action readout) with fewer than m incoming synapses"""
    net = motor_net(brain)
    if not net:
        return None
    outs = [n['id'] for n in ((net.get('neurons') or {}).get('output') or [])][:7]
    inc = collections.Counter(s.get('post_id') for s in (net.get('synapses') or []) if isinstance(s, dict))
    return [i for i, o in enumerate(outs) if inc.get(o, 0) < m]


def wire_motor(brain, m=3, w=0.03):
    """every motor output gets >= m incoming synapses (tiny weights). Deterministic, idempotent. Returns the number added."""
    net = motor_net(brain)
    if not net or m <= 0:
        return 0
    neu = net.get('neurons') or {}
    outs = [n['id'] for n in (neu.get('output') or [])][:7]
    pre = [n['id'] for n in (neu.get('hidden') or []) if n.get('is_active', True)] or [n['id'] for n in (neu.get('input') or [])]
    if not outs or not pre:
        return 0
    syns = net.setdefault('synapses', [])
    inc = collections.Counter(s.get('post_id') for s in syns if isinstance(s, dict))
    if all(inc.get(o, 0) >= m for o in outs):
        return 0
    h = hashlib.md5(repr(sorted((s.get('pre_id'), s.get('post_id')) for s in syns if isinstance(s, dict))).encode()
                    + repr(outs).encode()).hexdigest()
    rng = random.Random(int(h[:12], 16))
    have = {(s.get('pre_id'), s.get('post_id')) for s in syns if isinstance(s, dict)}
    tpl = _template(brain, net)
    added = 0
    for o in outs:
        need = m - inc.get(o, 0)
        if need <= 0:
            continue
        cands = [p for p in pre if (p, o) not in have and p != o]
        rng.shuffle(cands)
        for p in cands[:need]:
            wt = (1.0 if rng.random() < 0.5 else -1.0) * float(w) * (0.5 + rng.random())
            syns.append(new_synapse(tpl, p, o, wt)); have.add((p, o)); added += 1
    return added


def wire_m():
    try:
        return int(float(os.environ.get('MULTINXON_CAP_WIRE') or 0))
    except Exception:
        return 0


def wire_w():
    try:
        return float(os.environ.get('MULTINXON_CAP_WIRE_W') or DEFAULTS['cap_wire_w'])
    except Exception:
        return DEFAULTS['cap_wire_w']


def install_loader():
    """every brain loaded from a dict (cold tries, learned plays, the lab) gets its motor wired when MULTINXON_CAP_WIRE > 0"""
    try:
        import neuraxon.multisphere as MS
    except Exception as e:                                                       # pragma: no cover
        _STATE['error'] = f'loader: {type(e).__name__}: {e}'
        return False
    f0 = MS.load_multisphere_from_dict
    if getattr(f0, '_v1165', False):
        return True

    def load_multisphere_from_dict(data, rebuild_net_fn=None, *a, **k):
        m = wire_m()
        if m > 0 and isinstance(data, dict) and data.get('spheres'):
            try:
                n = wire_motor(data, m, wire_w())
                _STATE['loads'] += 1; _STATE['wired'] += n
                if n:
                    _tally('cap_wired', n)
            except Exception as e:
                _STATE['error'] = f'wire: {type(e).__name__}: {e}'[:160]
        return f0(data, rebuild_net_fn, *a, **k)

    load_multisphere_from_dict._v1165 = True
    load_multisphere_from_dict.__wrapped__ = f0
    MS.load_multisphere_from_dict = load_multisphere_from_dict
    _STATE['loader'] = True
    return True


# ======================================================================================================== 1-3. the brain
def _interior_changed(g0, g1, b=HUD):
    if g0 is None or g1 is None:
        return False
    try:
        H = len(g1); W = len(g1[0]) if H else 0
        if len(g0) != H or (H and len(g0[0]) != W):
            return True
        for y in range(b, H - b):
            r0 = g0[y]; r1 = g1[y]
            for x in range(b, W - b):
                if r0[x] != r1[x]:
                    return True
    except Exception:
        return False
    return False


def _grid_of(frame):
    try:
        fr = getattr(frame, 'frame', None)
        if isinstance(fr, list) and fr:
            g = fr[-1]
            return [list(r) for r in g] if g is not None else None
        g = getattr(frame, 'grid', None)
        return [list(r) for r in g] if g is not None else None
    except Exception:
        return None


def _act_of_cls(cls):
    if not cls:
        return None
    if cls[0] == 'K':
        try:
            return int(cls[1])
        except Exception:
            return None
    return 6


def install(ArcBrain):
    """wrap ArcBrain.__init__, _masked_avail, act, play_step, _click_candidates; idempotent per class object"""
    if getattr(ArcBrain, '_v1165_cap', False):
        return True
    ArcBrain._v1165_cap = True
    _init0 = ArcBrain.__init__
    _mav0 = ArcBrain._masked_avail
    _act0 = ArcBrain.act
    _ps0 = ArcBrain.play_step
    _cc0 = ArcBrain._click_candidates

    def __init__(self, params, *a, **k):
        _init0(self, params, *a, **k)
        self._cap_on = cap_on(params)
        if not self._cap_on:
            return
        self._cap_pref = set(self._action_mask) if getattr(self, '_action_mask', None) else None
        self._action_mask = None                                             # 1. the hard filter is gone
        self._cap_open = max(0.0, min(1.0, gene(params, 'cap_open')))
        self._cap_explore = max(0.0, min(1.0, gene(params, 'cap_explore')))
        self._cap_pen = max(0.0, gene(params, 'cap_pen'))
        seed = k.get('seed', a[0] if a else 0)
        self._cap_rng = random.Random(f'cap1165|{seed}')                    # its own stream: the brain's rng is untouched
        self._cap_live = set(); self._cap_full = False; self._cap_prev = None
        self._cap_n = collections.Counter()
        st = getattr(self, '_steer', None)
        if st is not None:
            brain = self
            d0 = st.decide; s0 = st.score

            def decide(b_, frame, a_, d_):
                b_._cap_full = True                                              # the steering sees every offered action
                try:
                    return d0(b_, frame, a_, d_)
                finally:
                    b_._cap_full = False

            def score(cls, *aa, **kk):
                v = s0(cls, *aa, **kk)
                pr = brain._cap_pref
                if pr is not None:
                    act = _act_of_cls(cls)
                    if act is not None and act not in pr and act not in brain._cap_live:
                        v -= brain._cap_pen
                return v
            st.decide = decide; st.score = score

    def _masked_avail(self, frame):
        if not getattr(self, '_cap_on', False):
            return _mav0(self, frame)
        avail = list(getattr(frame, 'available_actions', None) or [])
        pr = self._cap_pref
        if pr is None or self._cap_full or self._cap_rng.random() < self._cap_explore:
            return avail
        allowed = pr | self._cap_live
        keep = [x for x in avail if _val(x) == 0 or _val(x) in allowed]
        return keep if any(_val(x) != 0 for x in keep) else avail

    def act(self, outs, frame):
        if not getattr(self, '_cap_on', False) or self._cap_pref is None:
            return _act0(self, outs, frame)
        opened = self._cap_rng.random() < self._cap_open
        self._action_mask = None if opened else (self._cap_pref | self._cap_live)
        try:
            a, d = _act0(self, outs, frame)
        finally:
            self._action_mask = None
        if opened and _val(a) not in self._cap_pref and _val(a) not in self._cap_live and _val(a) > 0:
            self._cap_n['open'] += 1; _tally('cap_open')
        return a, d

    def play_step(self, frame):
        if not getattr(self, '_cap_on', False):
            return _ps0(self, frame)
        g1 = _grid_of(frame)
        lv = int(getattr(frame, 'levels_completed', 0) or 0)
        if self._cap_prev is not None:                                         # 2. what did the last executed action do?
            g0, v0, lv0 = self._cap_prev
            if v0 > 0 and (lv > lv0 or _interior_changed(g0, g1)):
                if v0 not in self._cap_live and (self._cap_pref is None or v0 not in self._cap_pref):
                    self._cap_n['live'] += 1; _tally('cap_live')
                self._cap_live.add(v0)
        a, d = _ps0(self, frame)
        v = _val(a)
        self._cap_prev = (g1, v, lv) if v > 0 else None
        if v > 0 and self._cap_pref is not None and v not in self._cap_pref:
            self._cap_n['np'] += 1; _tally('cap_np')
        if v > 0:
            self._cap_n[f'a{v}'] += 1
        return a, d

    def _click_candidates(self, grid):
        out = _cc0(self, grid)
        if out or not getattr(self, '_cap_on', False) or not grid:
            return out
        lo, hi, cc = getattr(self, '_cm_lo', 1), getattr(self, '_cm_hi', 64), getattr(self, '_click_colour', -1)
        mode = getattr(self, '_click_mask', 'off')
        if lo <= 1 and hi >= 4096 and cc in (None, -1) and mode != 'auto':
            return out
        self._cm_lo, self._cm_hi, self._click_colour, self._cm_sig = 1, 4096, -1, None   # 3. any component of the board
        if mode == 'auto':
            self._click_mask = 'centroid'
        try:
            out = _cc0(self, grid)
            self._cap_n['click_open'] += 1
        finally:
            self._cm_lo, self._cm_hi, self._click_colour, self._click_mask = lo, hi, cc, mode
        return out

    for f in (__init__, _masked_avail, act, play_step, _click_candidates):
        f._v1165 = True
    __init__.__doc__ = _init0.__doc__
    ArcBrain.__init__ = __init__; ArcBrain._masked_avail = _masked_avail; ArcBrain.act = act
    ArcBrain.play_step = play_step; ArcBrain._click_candidates = _click_candidates
    _STATE['brain'] = True
    return True


def stats(brain):
    """the capability counters of one brain object (a play)"""
    if not getattr(brain, '_cap_on', False):
        return {'on': False}
    n = dict(brain._cap_n)
    return {'on': True, 'pref': sorted(brain._cap_pref) if brain._cap_pref else None, 'live': sorted(brain._cap_live),
            'np': n.get('np', 0), 'live_found': n.get('live', 0), 'open': n.get('open', 0), 'click_open': n.get('click_open', 0),
            'actions': {k: v for k, v in n.items() if k.startswith('a')}}


# ======================================================================================================== the report
def report(neural, brain=None, cap=True):
    """per game: offered actions, the ones the brain can reach (with the capability layer or without it), the missing;
    plus its unwired motor outputs (before wiring)"""
    neural = neural or {}
    pref = mask_set(neural.get('action_mask'))
    out = {}
    for g, offered in GAME_ACTIONS.items():
        o = set(offered)
        if cap:
            reach = set(o)
        else:
            k = (pref & o) if pref else set(o)
            reach = k if k else set(o)                                        # the engine wins when the mask empties
        out[g] = {'group': GROUP[g], 'offered': sorted(o), 'reach': sorted(reach), 'missing': sorted(o - reach)}
    uw = unwired(brain, 1) if brain else None
    band = (neural.get('click_mask_min'), neural.get('click_mask_max'), neural.get('click_colour'))
    return {'pref': sorted(pref) if pref else None, 'games': out,
            'games_missing': sorted(g for g, v in out.items() if v['missing']),
            'unwired': uw, 'click_band': band,
            'universal': (not any(v['missing'] for v in out.values())) and (cap or not uw)}
