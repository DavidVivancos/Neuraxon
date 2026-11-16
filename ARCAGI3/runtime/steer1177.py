"""steer1177.py -- NAS 1177 changes 2-7: THE MILESTONE-2 LOOP IN THE STEERING (the part of the brain that decides the executed action).
Every brain still proposes every action and learns only from effects; nothing here knows a game's solution or replays an expert move.
Each mechanism is a gene (absent = off = the 1176 steering, preflight 177 checks identical traces); every 1177 cell forces them on.

WHAT MILESTONE 2 MEASURED (dfranzen's write-up; the Duck harness) and what each becomes here:
  anim77 (change 2)  ANIMATIONS. "Some games play animations which contain important information ... exposing animations showed the
                     strongest benefit"; a hint to look only when TRANSIENT PIXELS (a pixel that changes more than once on the way
                     from the first to the settled frame). Here: every response's frames are read -- per action signature: animated,
                     transient pixels, and the lowest goal distance h the animation passed through. An action whose animation went
                     CLOSER to the goal than its settled board ("it almost worked") gets that drop as value; an action with transient
                     pixels is never a no-op for the model (it acts on hidden state) and is worth re-probing in a new context.
  effect_ctx 3       FRAME_DIFF, OBJECT BY OBJECT, NEVER KNOWING LESS THAN THE SIGNATURE (change 3). 1176's context model abstained on
                     31% and lost recall (15.6% vs 37.4%). Mode 3 backs off instead of abstaining: the context when it has >= 2
                     observations at >= 70% agreement, else the 1175 signature mean (any observation) -- by construction it predicts
                     wherever 1175 did. goal_what is NOT coupled to it (the 1176 twin with colour targets lost +526 / -979). The
                     object grammar gains 'rotate' (same colour and size, height and width swapped).
  death77 (change 4) THE GAME-OVER FRAME AND ITS DIAGNOSIS. dfranzen's BAR RULE: "the budget caused the death ONLY if the bar is fully
                     (or almost fully) depleted in the death frame ... if the bar clearly had budget remaining: the fatal action itself
                     caused the death -- do NOT retry the same path; a DIFFERENT approach must be tried". The play loop hands the death
                     frame to the brain; a death is a BUDGET death when the border band that moves with the actions is >= 85% used in
                     the death frame, or when the attempt died at the same action count (+-1) as an earlier death. A budget death is
                     NOT a fatal (state, action): no veto, no deadly class, no death_memory entry (1155 measured r11l's deaths: budget
                     exhaustion every ~30 actions, recorded as fatal actions) -- it sets the PACE: past 60% of the budget without a drop
                     of h the steering navigates to its best-known state and stops probing. A FATAL death keeps the 1160 veto and adds
                     the fatal CONTEXT (the action's signature and the cell ahead of the mover / the clicked object's class): a
                     candidate in that context is penalised on every board of the level; the last 3 moves before a fatal death are
                     SUSPECT and, once two deaths passed through one, it yields to an untried alternative ("change the plan BEFORE the
                     fatal point").
  undo77 (change 5)  UNDO AS A PROBE. "Making the UNDO action available helped substantially" (su15: undo used repeatedly). ACTION7 is
                     offered on ar25, bp35, lf52, sb26, sk48, su15 (measured: it restores the interior exactly on ar25, bp35, sk48).
                     The steering learns whether ACTION7 undoes (the board returns to the previous state's key); when it does, a probe
                     that RAISED h, or that left a state with untried actions for a state with none, is TAKEN BACK at once (one action
                     instead of RESET + walk). 1177 adds ACTION7 to every action mask (the masks were the solutions' vocabularies); where
                     it was added, ACTION7 chosen as an ordinary move is allowed UNDO_PROBES times a level or at a dead end, and once it
                     undoes only the take-back uses it.
  guard77 (change 6) NO-OP GUARDS. "Stop the remaining actions after an action leaves the interior board unchanged"; "after an
                     ineffective action, block a later action call" (from level 2); "stop after a game over or a level completion".
                     Here: from level 2 the action after an ineffective one may not repeat its signature (an untried one of another
                     signature is taken); on level 1 the same after 3 ineffective actions in a row; every macro (below) stops at its
                     first step that changes nothing or raises h; a level-up or a death ends every running macro.
  transfer77 (ch. 7) LEVEL TRANSFER. "Mechanics from the preceding level often transfer; new elements may introduce mechanics that must be
                     understood -- prioritize small, informative tests when their behaviour is unclear; re-check anything contradicted."
                     At a level-up the object types never seen on the earlier levels are NEW ELEMENTS: their clicks get a probe bonus
                     until tried twice; a signature carried from earlier levels whose first prediction on the new level MISSES has its
                     old evidence halved.
  macro77 (ch. 7)    RETENTION (dfranzen kept the model's own Python functions for the whole game). The analogue: a MACRO = the classes of
                     the brain's OWN last effective actions (<= 6) that lowered h by >= 2 buckets, or that completed a level; on later
                     boards a macro whose first class is available is offered as one choice and replayed class by class under the
                     guard. Never from the expert prefix (the warm feed is excluded): effects, never taught moves.
THE EYES' STATE (eyes1177): every step the steering leaves in self._e77 what the brain's eyes block reads -- frame_diff of the last
action, the mover's position and colour, animation, prediction hit / miss, guard, new elements, undo state, budget pace.
TALLIES (st_*): e7_anim, e7_trans, e7_hbelow, e7_death_fatal, e7_death_budget, e7_fatal_pen, e7_suspect, e7_pace, e7_un_ok, e7_un_fail,
e7_un_back, e7_un_veto, e7_gd_stale, e7_gd_macro, e7_new, e7_newbonus, e7_recheck, e7_mac_new, e7_mac_start, e7_mac_step, e7_mac_done, e7_ctx3."""
from __future__ import annotations

import collections
import os

import numpy as np

UNDO_MIN = 1
UNDO_PROBES = 2           # ACTION7 chosen as an ordinary move: at most 2 probes a level (to learn whether it undoes), then only take-backs
BUDGET_USED = 0.85
PACE_AT = 0.6
SUSPECT_N = 3
MACRO_MAX = 6
MACRO_DROP = 2
NEW_BONUS = 0.4
ANIM_W = 0.4
FATAL_PEN = 1.0
GENES = ('anim77', 'death77', 'undo77', 'guard77', 'transfer77', 'macro77')
# change 9 -- THE CONTEXT UPSCALED (dfranzen kept the whole history in context; the analogue: the steering's memories and lookahead):
UPSCALE77 = {'_bfs_max': 20000,      # states the navigation searches (1160-1176: 5,000)
             '_warm_max': 120,       # moves of the level's prefix the steering's memories learn (60)
             '_watch_n': 16,         # warm_prefix 2: frames the network watches (8)
             '_em_max_h': 144,       # boards the effect model scores per step (96)
             '_em_beam': 8,          # first-step boards looked past (6)
             '_em_second': 32}       # second-step actions per looked-past board (24)
_INFO = {'installed': False, 'error': None}


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def _forced_env():
    """a 1177 cell (MULTINXON_EYES77 = 1) forces the capabilities: a brain whose arch lacks a 1177 gene plays with it on
    (never in the bench's control)"""
    return str(os.environ.get('MULTINXON_EYES77', '0')) == '1' and str(os.environ.get('MULTINXON_V1177_CTRL', '0')) != '1'


def _pin77(name):
    """the pins, read literally (wiring_audit.py follows only literal names)"""
    if name == 'anim77':
        return os.environ.get('MULTINXON_PIN_ANIM77', '')
    if name == 'death77':
        return os.environ.get('MULTINXON_PIN_DEATH77', '')
    if name == 'undo77':
        return os.environ.get('MULTINXON_PIN_UNDO77', '')
    if name == 'guard77':
        return os.environ.get('MULTINXON_PIN_GUARD77', '')
    if name == 'transfer77':
        return os.environ.get('MULTINXON_PIN_TRANSFER77', '')
    if name == 'macro77':
        return os.environ.get('MULTINXON_PIN_MACRO77', '')
    return os.environ.get('MULTINXON_PIN_' + name.upper(), '')


def _gene(params, name):
    try:
        v = getattr(params, name, None)
        e = _pin77(name)
        if str(e).strip() != '':
            return int(float(e))
        if v in (None, ''):
            return 1 if (name in GENES and _forced_env()) else 0
        return int(float(v))
    except Exception:
        return 0


def _interior_changed(g0, g1, m=4):
    try:
        a = np.asarray(g0); b = np.asarray(g1)
        if a.shape != b.shape:
            return True
        return bool((a[m:-m, m:-m] != b[m:-m, m:-m]).any())
    except Exception:
        return True


def _transient(frames, m=4):
    """(animated, transient pixels): some interior pixel changes >= 2 times along the response frames (sampled to <= 12)"""
    try:
        fr = [np.asarray(f, dtype=np.int16) for f in frames]
        if len(fr) < 2:
            return False, False
        if len(fr) > 12:
            idx = sorted(set(int(round(i * (len(fr) - 1) / 11.0)) for i in range(12)))
            fr = [fr[i] for i in idx]
        A = np.stack(fr)[:, m:-m, m:-m]
        cnt = (A[1:] != A[:-1]).sum(0)
        return True, bool((cnt >= 2).any())
    except Exception:
        return False, False


def _rotate_effects(O0, O1, eff):
    """frame_diff's 'rotate': an unmatched object whose colour and pixel count reappear with height and width swapped"""
    try:
        out = []
        for (ocls, e) in eff:
            if e[0] in ('vanish', 'appear'):
                continue
        rot = 0
        used = set()
        for o in O0:
            if o['h'] == o['w']:
                continue
            for j, p in enumerate(O1):
                if j in used:
                    continue
                if p['col'] == o['col'] and p['n'] == o['n'] and p['h'] == o['w'] and p['w'] == o['h'] \
                        and abs(p['cy'] - o['cy']) <= max(o['h'], o['w']) and abs(p['cx'] - o['cx']) <= max(o['h'], o['w']):
                    used.add(j); rot += 1
                    break
        return rot
    except Exception:
        return 0


# ======================================================================================================== effect_ctx 3 (back-off)
def install_effect3():
    import effect1176 as E6
    if getattr(E6, '_v1177_ctx3', False):
        return True
    E6._v1177_ctx3 = True
    _pred0 = E6.ctx_pred
    _ctxp0 = E6.ctx_of_params

    def ctx_of_params(params):
        try:
            e = os.environ.get('MULTINXON_PIN_EFFECT_CTX', '')
            if str(e).strip() != '':
                return max(0, min(3, int(float(e))))
            if str(os.environ.get('MULTINXON_V1177_CTRL', '0')) == '1':      # the bench's control: the brain's own 1176 value
                b0 = getattr(params, 'eyes77_b0', None) or {}
                if isinstance(b0, dict) and 'effect_ctx' in b0:
                    v0 = b0.get('effect_ctx')
                    return max(0, min(2, int(float(v0)))) if v0 not in (None, '') else _ctxp0(type('P0', (), {})())
            g = getattr(params, 'effect_ctx', None)
            if g not in (None, ''):
                return max(0, min(3, int(float(g))))
            if _forced_env():
                return 3
        except Exception:
            pass
        return _ctxp0(params)

    def ctx_pred(st, sig, ctx):
        if int(getattr(st, 'ectx', 0) or 0) != 3:
            return _pred0(st, sig, ctx)
        if sig is None:
            return None, 'unknown'
        mn = st.ctx_min_n; ag = st.ctx_agree
        if ctx is not None:
            s = st.ctx_dh.get((sig, ctx))
            if s and s[1] >= mn and E6._agree(s) >= ag:
                return s[0] / float(s[1]), 'ctx'
            e = st.ctx_eff.get((sig, ctx))
            if e and e[0] >= mn and e[1] == 0:
                return 0.0, 'noeff'
        b = (getattr(st, 'dh_sig', None) or {}).get(sig)                 # the 1175 signature mean: any observation (v5's own rule)
        if b and b[1] > 0:
            return b[0] / float(b[1]), 'sig'
        return None, 'unknown'
    E6.ctx_of_params = ctx_of_params
    E6.ctx_pred = ctx_pred
    return True


# ======================================================================================================== the steering mixin
def install(Steer, Steer4, Steer5=None):
    """idempotent; after effect1176.install. Steer = steer1160.Steer (deaths are observed in its on_reset)"""
    if getattr(Steer4, '_v1177', False):
        return True
    Steer4._v1177 = True
    install_effect3()
    import effect1173 as E3
    import goalmap1163 as GM
    from steer1162 import components, sbk

    _init0 = Steer4.__init__; _start0 = Steer4.start_level; _after0 = Steer4._after_action; _value0 = Steer4.value
    _stats0 = Steer4.stats; _of4 = Steer4.observe_frame
    _onreset0 = Steer.on_reset
    _feed0 = E3._feed_steer

    def _feed(brain, st, items):
        try:
            st._e77_feeding = True
            return _feed0(brain, st, items)
        finally:
            st._e77_feeding = False
    E3._feed_steer = _feed

    # ------------------------------------------------------------------ init / level
    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        self.g77 = {g: _gene(params, g) for g in GENES}
        self.e77_on = (any(self.g77.values()) or str(os.environ.get('MULTINXON_EYES77', '0')) == '1'
                       or int(float(getattr(params, 'eyes77', 0) or 0)) == 1) and str(os.environ.get('MULTINXON_V1177_CTRL', '0')) != '1'
        self._e77 = {}; self._e77_feeding = False
        self._e77_anim = {}; self._e77_hbelow = {}
        self._e77_fatal = collections.Counter(); self._e77_suspect = collections.Counter()
        self._e77_trace = collections.deque(maxlen=SUSPECT_N); self._e77_steps = 0; self._e77_budget_n = None; self._e77_death_steps = []
        self._e77_verdict = None
        self._e77_hist = collections.deque(maxlen=8); self._e77_un = [0, 0]; self._e77_back_at = -1
        self._e77_noop_run = 0; self._e77_last_noop_sig = None
        self._e77_types_seen = set(); self._e77_new = set(); self._e77_newtry = collections.Counter()
        self._e77_carried = set(); self._e77_rechecked = set()
        self._e77_macros = []; self._e77_seg = []; self._e77_mac = None; self._e77_mac_from = set()
        self._e77_un7 = 0
        try:                                                               # ACTION7 added by 1177 (absent from the brain's 1176 mask)
            b0 = getattr(params, 'eyes77_b0', None) or {}
            self._e77_add7 = isinstance(b0, dict) and b0.get('action_mask') not in (None, '') and '7' not in str(b0.get('action_mask'))
        except Exception:
            self._e77_add7 = False
        try:
            if int(getattr(self, 'ectx', 0) or 0) == 3 and getattr(self, '_what_coupled', 0):
                self.goal_what = 0; self._what_coupled = 0                  # mode 3 is never coupled to colour targets
                try:
                    self.gm.what = 0
                except Exception:
                    pass
        except Exception:
            pass
        if self.e77_on:
            self.sense_w = 16                                              # the eyes read the steering's full sense
            for k_, v_ in UPSCALE77.items():                               # change 9: the context upscaled
                setattr(self, k_, v_)

    def start_level(self, g):
        first = not getattr(self, '_e77_started', False)
        _start0(self, g)
        self._e77_started = True
        if not getattr(self, 'e77_on', False):
            return
        self._e77_steps = 0; self._e77_budget_n = None; self._e77_death_steps = []
        self._e77_fatal = collections.Counter(); self._e77_suspect = collections.Counter(); self._e77_trace.clear()
        self._e77_hist.clear(); self._e77_noop_run = 0; self._e77_last_noop_sig = None
        self._e77_mac = None; self._e77_mac_from = set(); self._e77_seg = []; self._e77_un7 = 0
        self._e77.pop('pace', None)
        try:
            types = {(int(o['col']), sbk(o['n'])) for o in components(np.asarray(g), self.bg, self.ignore())}
        except Exception:
            types = set()
        if self.g77.get('transfer77') and not first and self._e77_types_seen:
            self._e77_new = types - self._e77_types_seen
            self._e77_newtry = collections.Counter()
            self._e77['newelem'] = bool(self._e77_new)
            if self._e77_new:
                _tally('e7_new', len(self._e77_new))
            self._e77_carried = set(getattr(self, 'sig_n', {}) or {}); self._e77_rechecked = set()
        else:
            self._e77_new = set(); self._e77['newelem'] = False
        self._e77_types_seen |= types

    # ------------------------------------------------------------------ the death: frame, diagnosis, pace
    def diagnose(self, brain, obs):
        """BUDGET or FATAL from the death frame (the band that moves with the actions used up) and the action count"""
        used = None
        try:
            fr = getattr(obs, 'frame', None) or []
            dg = fr[-1] if fr else None
            if dg is not None and hasattr(brain, '_v55_budget_channels') and getattr(brain, '_v55_border0', None) is not None:
                v = brain._v55_budget_channels(dg)
                used = (float(v[0]) + 1.0) / 2.0
        except Exception:
            used = None
        n = int(self._e77_steps)
        regular = any(abs(n - m) <= 1 for m in self._e77_death_steps) and n >= 3
        budget = (used is not None and used >= BUDGET_USED) or regular
        self._e77_death_steps.append(n)
        self._e77_verdict = 'budget' if budget else 'fatal'
        if budget:
            self._e77_budget_n = n if self._e77_budget_n is None else min(self._e77_budget_n, n)
            _tally('e7_death_budget')
        else:
            _tally('e7_death_fatal')
        try:
            brain._e77_died = -1 if budget else 1; brain._e77_died_t = int(getattr(brain, '_play_steps', 0))
            brain._e77_verdict = self._e77_verdict
        except Exception:
            pass
        return self._e77_verdict

    def on_reset(self, frame):
        """steer1160.Steer.on_reset with the 1177 diagnosis: a BUDGET death is not a fatal (state, action)"""
        _dg = self.g77.get('death77') if getattr(self, 'e77_on', False) else 0
        _fg = self.g77.get('fatal78', _dg) if getattr(self, 'e77_on', False) else 0      # v1.178: the fatal contexts apart from the BAR rule
        if (_dg or _fg) and self.prev is not None and self.died:
            verdict = (self._e77_verdict or 'fatal') if _dg else 'fatal'
            k0, ak, cl, g0, lv0 = self.prev
            if verdict == 'budget':
                self.died = False                                          # observed as an ordinary transition below (none: no frame)
                self.prev = None
            elif _fg:
                try:
                    O0 = self.objs.get(k0) or components(g0, self.bg, self.ignore())
                    import effect1176 as E6
                    sig, ctx = E6.sig_ctx(self, ak, O0, g0)
                    self._e77_fatal[(sig, ctx)] += 1
                    for kk in list(self._e77_trace):
                        self._e77_suspect[kk] += 1
                except Exception:
                    pass
        self._e77_verdict = None
        self._e77_steps = 0; self._e77_trace.clear(); self._e77_hist.clear(); self._e77_mac = None; self._e77_noop_run = 0
        return _onreset0(self, frame)

    # ------------------------------------------------------------------ animations (observe_frame sees the response frames)
    def observe_frame(self, frame):
        pre = self.prev if (getattr(self, 'e77_on', False) and not getattr(self, '_e77_feeding', False)) else None
        h0 = self.h if getattr(self, 'goal', None) is not None else None
        _of4(self, frame)
        if pre is None:
            return
        try:
            fr = getattr(frame, 'frame', None) or []
            anim, trans = _transient(fr) if len(fr) >= 2 else (False, False)
            self._e77['anim'] = 1 if trans else (0 if anim else -1)
            if not anim:
                return
            k0, ak, cl, g0, lv0 = pre
            O0 = self.objs.get(k0) or []
            sig = self.sig_of(ak, O0) if O0 else (('K', ak[1]) if ak and ak[0] == 'K' else None)
            a = self._e77_anim.setdefault(sig, [0, 0]); a[0] += 1; a[1] += int(trans)
            _tally('e7_anim')
            if trans:
                _tally('e7_trans')
            if self.g77.get('anim77') and self.goal is not None and self.start_board is not None and h0 is not None and len(fr) >= 3:
                hs = []
                step = max(1, (len(fr) - 1) // 10)
                for f in fr[:-1:step]:
                    hh = GM.h_of(np.asarray(f, dtype=np.int16), self.start_board, self.goal)
                    if hh is not None:
                        hs.append(hh)
                h1 = GM.h_of(np.asarray(fr[-1], dtype=np.int16), self.start_board, self.goal)
                if hs and h1 is not None and min(hs) < min(h0, h1):
                    d = (min(h0, h1) - min(hs)) / float(max(1, self.unit))
                    b = self._e77_hbelow.setdefault(sig, [0.0, 0]); b[0] += d; b[1] += 1
                    _tally('e7_hbelow')
        except Exception as e:
            self._v1177_error = f'anim: {type(e).__name__}: {e}'[:160]

    # ------------------------------------------------------------------ after every action: frame_diff, mover, undo, guard, macros
    def _after_action(self, ak, g0, g1, ch, trans):
        if not getattr(self, 'e77_on', False) or getattr(self, '_e77_feeding', False):
            return _after0(self, ak, g0, g1, ch, trans)
        k0 = None; O0 = None; sig = None; hq0 = getattr(self, 'hq', None)
        try:
            if g0 is not None and ak is not None:
                k0 = self.key(g0); O0 = self.objs.get(k0)
                if O0 is None:
                    O0 = components(g0, self.bg, self.ignore()); self.objs[k0] = O0
                sig = self.sig_of(ak, O0)
        except Exception:
            pass
        miss0 = getattr(self, 'last_miss', 0); pred0 = self.pred
        _after0(self, ak, g0, g1, ch, trans)
        E = self._e77
        try:
            E['guard'] = False
            compared = pred0 is not None and sig is not None and pred0[0] == sig
            missed = bool(compared and getattr(self, 'last_miss', 0))
            E['pred'] = 0 if not compared else (-1 if missed else 1)
            if g0 is None or g1 is None or ak is None or g0.shape != g1.shape:
                return
            k1 = self.key(g1)
            O1 = self.objs.get(k1)
            if O1 is None:
                O1 = components(g1, self.bg, self.ignore()); self.objs[k1] = O1
            eff = self.effects(O0 or [], O1, ak) if hasattr(self, 'effects') else []
            fd = collections.Counter(e[1][0] for e in eff)
            rot = _rotate_effects(O0 or [], O1, eff)
            if rot:
                fd['rotate'] += rot
                if sig is not None:
                    self.mem[sig][(('rot',), ('rotate',))] += 1
            E['fdiff'] = dict(fd)
            d = (g0 != g1) & ~self.ignore()
            E['nchg'] = int(d.sum())
            inter = _interior_changed(g0, g1)
            if ak[0] == 'K' and ch:                                        # the mover: where the moved thing is now
                dd = d & (g1 != self.bg)
                ys, xs = np.nonzero(dd)
                if 0 < len(ys) <= max(4, int(0.05 * d.size)):
                    E['mover'] = (float(ys.mean()), float(xs.mean()))
                    E['mover_col'] = int(np.bincount(np.clip(g1[ys, xs].ravel(), 0, 15)).argmax())
            # undo: did ACTION7 bring back the previous state? (the attempt's history of state keys)
            H = self._e77_hist
            if not H or H[-1] != k0:
                H.append(k0)
            if ak == ('K', 7):
                if len(H) >= 2 and self.g77.get('undo77'):
                    if k1 == H[-2]:
                        self._e77_un[0] += 1; _tally('e7_un_ok')
                    else:
                        self._e77_un[1] += 1; _tally('e7_un_fail')
                if len(H) >= 2 and k1 == H[-2]:
                    H.pop()                                                # the undone state leaves the history
                elif k1 != k0:
                    H.append(k1)
            elif k1 != k0:
                H.append(k1)
            ok = self._e77_un[0] >= UNDO_MIN and self._e77_un[0] >= 2 * self._e77_un[1]
            E['undo_state'] = 1 if ok else (0 if (self._e77_un[0] + self._e77_un[1]) else -1)
            # the guard's memory: consecutive ineffective actions and the last ineffective signature
            if inter or trans:
                self._e77_noop_run = 0; self._e77_last_noop_sig = None
            else:
                self._e77_noop_run += 1; self._e77_last_noop_sig = sig
            # transfer: a carried signature whose first prediction on the new level misses -> its old evidence halves
            if self.g77.get('transfer77') and sig is not None and sig in self._e77_carried and sig not in self._e77_rechecked \
                    and compared:
                self._e77_rechecked.add(sig)
                if missed:
                    n = int(self.sig_n.get(sig, 0))
                    if n >= 2:
                        self.sig_n[sig] = max(1, n // 2); self.sig_chg[sig] = int(self.sig_chg.get(sig, 0)) // 2
                        for ek in list(self.mem[sig]):
                            self.mem[sig][ek] = self.mem[sig][ek] // 2
                        _tally('e7_recheck')
            # macros: the segment of the brain's own effective actions
            if self.g77.get('macro77'):
                cl = None
                try:
                    nd = self.graph.get(k0)
                    cl = next((c for a2, c in (nd or {}).get('cands', []) if a2 == ak), None) if nd else None
                except Exception:
                    cl = None
                if cl is None:
                    cl = ('K', ak[1]) if ak[0] == 'K' else ('C', int(g0[ak[1], ak[2]]), 0)
                if inter:
                    self._e77_seg.append((cl, hq0))
                    del self._e77_seg[:-MACRO_MAX]
                    hq1 = getattr(self, 'hq', None)
                    if hq1 is not None:
                        lo = [i for i, (c_, h_) in enumerate(self._e77_seg) if h_ is not None and h_ - hq1 >= MACRO_DROP]
                        if lo:
                            self._add_macro([c_ for c_, _ in self._e77_seg[lo[0]:]])
                else:
                    self._e77_seg = []
            if self._e77_budget_n:
                E['pace'] = min(1.0, self._e77_steps / float(max(1, self._e77_budget_n)))
        except Exception as e:
            self._v1177_error = f'after: {type(e).__name__}: {e}'[:160]

    def _add_macro(self, seq):
        seq = [tuple(c) for c in seq if c]
        if len(seq) < 2 or any(c[0] not in ('K', 'C', 'R', 'L', 'D') for c in seq):
            return
        t = tuple(seq)
        if t in [m[0] for m in self._e77_macros]:
            return
        self._e77_macros.append([t, 0, 0])
        del self._e77_macros[:-8]
        _tally('e7_mac_new')

    def level_done(self, finished):
        r = Steer4._v1177_level_done0(self, finished)
        try:
            if getattr(self, 'e77_on', False) and self.g77.get('macro77') and len(self._e77_seg) >= 2:
                self._add_macro([c_ for c_, _ in self._e77_seg])          # the segment that completed the level
            self._e77_seg = []; self._e77_mac = None                       # terminal guard: a level-up ends every running macro
        except Exception:
            pass
        return r

    # ------------------------------------------------------------------ value: animation, new elements, fatal contexts, pace
    def value(self, sig, O, akey):
        v, p = _value0(self, sig, O, akey)
        if not getattr(self, 'e77_on', False):
            return v, p
        try:
            if self.g77.get('anim77'):
                b = self._e77_hbelow.get(sig)
                if b and b[1] > 0:
                    v += ANIM_W * min(2.0, b[0] / b[1])
                a = self._e77_anim.get(sig)
                if a and a[1] > 0 and p is None:
                    v += 0.2                                              # transient pixels: hidden mechanics, worth a probe
            if self.g77.get('transfer77') and self._e77_new and sig and sig[0] == 'C' and (sig[1], sig[2]) in self._e77_new \
                    and self._e77_newtry[(sig[1], sig[2])] < 2:
                v += NEW_BONUS; _tally('e7_newbonus')
            if self.g77.get('fatal78', self.g77.get('death77')) and self._e77_fatal:     # v1.178: the fatal penalty is its own gene
                import effect1176 as E6
                g = self._cur[0] if getattr(self, '_cur', None) else None
                s2, c2 = E6.sig_ctx(self, akey, O, g)
                n = self._e77_fatal.get((s2, c2), 0)
                if n:
                    v -= FATAL_PEN * min(2, n); _tally('e7_fatal_pen')
            if self.g77.get('death77') and self._e77.get('pace') is not None and self._e77['pace'] > PACE_AT and p is None:
                v -= 0.3                                                  # the budget is running out: no new probes
        except Exception as e:
            self._v1177_error = f'value: {type(e).__name__}: {e}'[:160]
        return v, p

    # ------------------------------------------------------------------ the decision: suspects, undo take-back, guard, macros, pace
    def _cands(self, nd):
        return [x for x in nd['cands'] if x[0] not in nd['tried'] and not self.deadly(x[1])]

    def _swap(self, akey, cls, why):
        self.prev = (self.prev[0], akey, cls, self.prev[3], self.prev[4]); self.pred = None
        self._from_brain = False
        _tally(why)
        return self.to_action(akey)

    def decide_outer(self, orig, brain, frame, a, d):
        if getattr(self, '_e77_in', False) or not getattr(self, 'e77_on', False):
            return orig(self, brain, frame, a, d)
        self._e77_in = True
        try:
            out = orig(self, brain, frame, a, d)
        finally:
            self._e77_in = False
        if not self._cur or self.prev is None:
            return out
        self._e77_steps += 1
        try:
            g, lv = self._cur
            k = self.prev[0]; akey = self.prev[1]; cls = self.prev[2]
            nd = self.graph.get(k)
            if nd is None:
                return out
            O = self.objs.get(k) or []
            avail = set()
            try:
                avail = {int(getattr(x, 'value', x)) for x in brain._masked_avail(frame)}
            except Exception:
                pass
            # 0. ACTION7 where 1177 added it to the mask is a PROBE and a TAKE-BACK, never an ordinary move: chosen by the steering it is
            #    allowed UNDO_PROBES times a level (to learn whether it undoes) or at a dead end; once it undoes, only the take-back (2.)
            if self.g77.get('undo77') and getattr(self, '_e77_add7', False) and akey == ('K', 7):
                alt = [x for x in self._cands(nd) if x[0] != ('K', 7)]
                if alt and (self._e77.get('undo_state') == 1 or self._e77_un7 >= UNDO_PROBES):
                    best = max(alt, key=lambda x: self.value(self.sig_of(x[0], O), O, x[0])[0])
                    return self._swap(best[0], best[1], 'e7_un_veto')
                self._e77_un7 += 1
            # 1. a running macro: the next class, under the guard
            if self.g77.get('macro77') and self._e77_mac is not None:
                seq, i, h_at, mref = self._e77_mac
                ok = i < len(seq) and self._e77_noop_run == 0 and (h_at is None or getattr(self, 'hq', None) is None or self.hq <= h_at)
                if not ok:
                    self._e77_mac = None; _tally('e7_gd_macro')
                else:
                    want = seq[i]
                    pick = next((x for x in nd['cands'] if tuple(x[1]) == want and not self.deadly(x[1])
                                 and not (nd['tried'].get(x[0]) is not None and not nd['tried'][x[0]][1])), None)
                    if pick is None:
                        self._e77_mac = None; _tally('e7_gd_macro')
                    else:
                        self._e77_mac = (seq, i + 1, getattr(self, 'hq', None), mref) if i + 1 < len(seq) else None
                        if self._e77_mac is None:
                            mref[2] += 1; _tally('e7_mac_done')
                        _tally('e7_mac_step')
                        self._e77_trace.append((k, pick[0]))
                        if pick[0] != akey:
                            return self._swap(pick[0], pick[1], 'e7_mac_swap')
                        return out
            # 2. undo take-back: the last probe raised h, or left untried actions behind for a dead end
            if self.g77.get('undo77') and 7 in avail and self._e77.get('undo_state') == 1 and len(self._e77_hist) >= 2 \
                    and self._e77_back_at != self._e77_steps - 1 and akey != ('K', 7):
                prev_k = self._e77_hist[-2]
                pn = self.graph.get(prev_k)
                worse = int(getattr(self, 'dh_last', 0) or 0) > 0
                stuck = not self._cands(nd) and pn is not None and bool(self._cands(pn))
                if worse or stuck:
                    self._e77_back_at = self._e77_steps
                    return self._swap(('K', 7), ('K', 7), 'e7_un_back')
            # 3. the stale-state guard: after an ineffective action the same signature is not tried again (level 2+; level 1 after 3)
            if self.g77.get('guard77') and self._e77_last_noop_sig is not None:
                lim = 1 if (self.level or 0) >= 1 else 3
                if self._e77_noop_run >= lim and self.sig_of(akey, O) == self._e77_last_noop_sig:
                    alt = [x for x in self._cands(nd) if self.sig_of(x[0], O) != self._e77_last_noop_sig]
                    if alt:
                        best = max(alt, key=lambda x: self.value(self.sig_of(x[0], O), O, x[0])[0])
                        self._e77['guard'] = True
                        return self._swap(best[0], best[1], 'e7_gd_stale')
            # 4. suspects: a move two fatal deaths passed through yields to an untried alternative
            if self.g77.get('fatal78', self.g77.get('death77')) and self._e77_suspect.get((k, akey), 0) >= 2:
                alt = [x for x in self._cands(nd) if x[0] != akey and self._e77_suspect.get((k, x[0]), 0) < 2]
                if alt:
                    best = max(alt, key=lambda x: self.value(self.sig_of(x[0], O), O, x[0])[0])
                    return self._swap(best[0], best[1], 'e7_suspect')
            # 5. pace: past PACE_AT of the budget without a drop of h -> the stall navigation of v5 (to the best-known state)
            if self.g77.get('death77') and self._e77.get('pace') is not None and self._e77['pace'] > PACE_AT \
                    and getattr(self, 'stall', 0) and getattr(self, 'since_drop', 0) >= 3 and getattr(self, 'goal', None) is not None:
                step = self.bfs(k)
                if step is not None and step != akey:
                    self.since_drop = 0
                    cl2 = next((c for a2, c in nd['cands'] if a2 == step), ('NAV',))
                    return self._swap(step, cl2, 'e7_pace')
            # 6. a macro CONTINUES the brain's own choice: the executed action's class starts a known productive sequence (once per
            #    state; a macro started 3 times without completing is dropped)
            if self.g77.get('macro77') and self._e77_macros and k not in self._e77_mac_from and cls:
                self._e77_macros = [m for m in self._e77_macros if not (m[1] >= 3 and m[2] == 0)]
                for m in sorted(self._e77_macros, key=lambda m: -(m[2] - 0.5 * m[1])):
                    if tuple(cls) == m[0][0]:
                        self._e77_mac_from.add(k); m[1] += 1
                        self._e77_mac = (m[0], 1, getattr(self, 'hq', None), m); _tally('e7_mac_start')
                        break
            # bookkeeping for suspects and new elements
            self._e77_trace.append((k, akey))
            if self._e77_new and cls and cls[0] == 'C':
                self._e77_newtry[(cls[1], cls[2])] += 1
        except Exception as e:
            self._v1177_error = f'decide: {type(e).__name__}: {e}'[:160]
        return out

    def stats(self):
        s = _stats0(self)
        try:
            s.update(e77=dict(getattr(self, 'g77', {}) or {}), macros=len(getattr(self, '_e77_macros', []) or []),
                     undo=list(getattr(self, '_e77_un', [0, 0])), fatal_ctx=len(getattr(self, '_e77_fatal', {}) or {}),
                     v1177_error=getattr(self, '_v1177_error', None))
        except Exception:
            pass
        return s

    Steer4.__init__ = __init__; Steer4.start_level = start_level; Steer4._after_action = _after_action; Steer4.value = value
    Steer4.stats = stats; Steer4.observe_frame = observe_frame; Steer4._add_macro = _add_macro; Steer4._cands = _cands
    Steer4._swap = _swap; Steer4.diagnose = diagnose
    Steer4._v1177_level_done0 = Steer4.level_done
    Steer4.level_done = level_done
    Steer.on_reset = on_reset
    _obs0 = Steer.observe

    def observe(self, k0, akey, cls, g0, g1, died, up, frames=None):
        """an undo's outcome depends on the play's history, not on the board: it never marks the game non-Markov (undo77)"""
        if akey == ('K', 7) and getattr(self, 'e77_on', False) and (getattr(self, 'g77', None) or {}).get('undo77'):
            mk = self.markov; ni = self.n_inconsistent
            r = _obs0(self, k0, akey, cls, g0, g1, died, up, frames=frames)
            self.markov = mk; self.n_inconsistent = ni
            return r
        return _obs0(self, k0, akey, cls, g0, g1, died, up, frames=frames)
    Steer.observe = observe
    _dec4 = Steer4.decide

    def decide4(self, brain, frame, a, d):
        return decide_outer(self, _dec4, brain, frame, a, d)
    Steer4.decide = decide4
    if Steer5 is not None and not getattr(Steer5, '_v1177', False):
        Steer5._v1177 = True
        _dec5 = Steer5.decide

        def decide5(self, brain, frame, a, d):
            return decide_outer(self, _dec5, brain, frame, a, d)
        Steer5.decide = decide5
    _INFO['installed'] = True
    return True


# ======================================================================================================== the brain: the death frame
def install_brain(ArcBrain):
    """ArcBrain.v77_death(obs): the play loop hands over the death frame; the steering diagnoses it; a BUDGET death is not written into
    the brain's death memory (v55_note_death)"""
    if getattr(ArcBrain, '_v1177_death', False):
        return True
    ArcBrain._v1177_death = True
    _nd0 = ArcBrain.v55_note_death

    def v77_death(self, obs):
        self._e77_verdict = None
        st = getattr(self, '_steer', None)
        if st is not None and getattr(st, 'e77_on', False) and (st.g77 or {}).get('death77') and hasattr(st, 'diagnose'):
            try:
                st.diagnose(self, obs)
            except Exception as e:
                self._v1177_error = f'diagnose: {type(e).__name__}: {e}'[:160]

    def v55_note_death(self, *a, **k):
        if getattr(self, '_e77_verdict', None) == 'budget':
            try:
                self._v55_beh['deaths'] += 1
            except Exception:
                pass
            st = getattr(self, '_steer', None)
            if st is not None:
                try:
                    st.on_death()                                          # the steering knows the attempt ended (its verdict: budget)
                except Exception:
                    pass
            return None
        return _nd0(self, *a, **k)
    ArcBrain.v77_death = v77_death
    ArcBrain.v55_note_death = v55_note_death
    return True
