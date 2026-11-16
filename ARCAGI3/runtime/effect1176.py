"""effect1176.py -- NAS 1176: the brain side of nas_1175_report.md section 25, changes 3 and 4 (a context effect model measured on a
drop bench; the goal as the twins and FrameLab say). Every brain still proposes every action; nothing here knows a game's solution or
replays a move: the tables below are filled only by EFFECTS the play observed (its own moves, the taught prefix's moves as effects,
the earlier levels of the same play). All of it is behind genes with environment pins; a brain without them plays exactly as in 1175.

effect_ctx 0 / 1 / 2 (MULTINXON_PIN_EFFECT_CTX; MULTINXON_EFFECT_CTX_DEFAULT for a brain without the gene) -- EFFECTS IN CONTEXT
  1175's steering (v5) judged an action by its SIGNATURE's mean change of the goal distance h on this level: ('K', key), or a click's
  nearest object (colour, size bin, on / beside). It is context-blind: tn36 L2's submit click taught "raises h" in one state and was
  avoided in the state where it finishes the level (the 1175 regression). effect_ctx keys every observed effect by (signature, local
  context):
    * a click: the clicked object's SHAPE class (square / wide / tall x solid / mid / hollow) and whether that kind of object has acted
      as a CONTROL in this play (its click changed cells away from it: a submit, a reset, a switch); a control's context adds the
      current goal distance bucket hq (the same click in a state far from the goal and in one near it are different contexts);
    * a key: the colour class of the cell one step ahead of the avatar (its mean move per key, from the play's own observations):
      background, or the colour there, or the grid edge;
  and holds, per (signature, context), the level's dh counts [sum, n, down, up, zero] and, across the play's levels, the effect
  counts [n, changed] (goal-free). PREDICTION WITH ABSTENTION (FrameLab 21): a context predicts only with >= ctx_min_n (2)
  observations and >= ctx_agree (0.7) agreement on the sign of dh; a context with enough observations that disagree ABSTAINS (the
  brain's proposal stands); a context never seen falls back to the signature only when the signature itself agrees >= ctx_agree; a
  context in which the action never changed anything (>= ctx_min_n tries, any level) predicts "no drop". effect_ctx 1 uses the model
  where v5 uses its signature mean (the goal authority: a proposal that is known not to lower h yields to the candidate predicted to
  lower it most); effect_ctx 2 also uses it in the plan's one-step value and in the "raises h" veto (the report's two-step look is
  NOT built: the one-step value is where 1175's steering reads the prediction; a look through predicted boards needs a board model
  the table does not have).
THE DROP BENCH IN PLAY (measurement only; with effect_ctx >= 1 or MULTINXON_DROPBENCH76 = 1 the tables are kept for every brain)
  at every action the steering takes, BEFORE learning from it: did the context model predict a drop (dh < 0), did it abstain, did the
  signature baseline (1175: any observation) predict a drop -- against what happened: the play's own goal map h (st_db_*) and, on the
  play's first level when the verified finished board is known (MULTINXON_SENSE_TRUTH), the TRUE goal h (cells still different from
  the finished board; st_dbt_*). drop recall = tp / down, precision = tp / predicted drops. FrameLab's bench scores every candidate of
  a board; this one scores the action taken (the NAS cannot replay boards in play): it is the same definition on the moves played.
goal_what COUPLED (report change 4): a brain with effect_ctx >= 1 and NO goal_what gene plays with goal_what 1 (colour targets pay in
  proportion to the model, FL21); an explicit goal_what 0 / 1 (the 2 x 2 twins) wins.
goal_recall 0 / 1 (MULTINXON_PIN_GOAL_RECALL) -- RECALL WHERE COLOUR HOLDS PRECISION
  at goal strictness 0 with goal_what targets fitted, the goal map names 1.5 x its usual count, keeping each extra cell only when it has
  a target colour (a cell with a target arrives only at its colour, so it cannot steer by a wrong disturbance).
goal_prior 0 / 1 (MULTINXON_PIN_GOAL_PRIOR) -- A COLOUR PRIOR FROM OTHER GAMES
  goal_what's (start colour -> finished colour) table is learned from the play's own completed levels; a start colour those levels
  have not decided takes the decision of the OTHER games' level-1 finished boards (color_prior_1176.json, built from boards1164.npz;
  the game being played is left out). It acts only where a goal map exists: level 1 from its start has no goal map (nothing named), so
  the prior acts on level >= 2 and on continuation levels of whole games, never on level 1 itself.
No goal timeout (FL21: it costs 0.04-0.07 per send with good models)."""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import effect1173 as E3

CTX_MIN_N = 2
CTX_AGREE = 0.7
RECALL_X = 1.5
PRIOR_MIN_N = 5
PRIOR_SHARE = 0.6
_PRIOR = {'doc': None, 'cache': {}}


def _gene(params, name):
    return getattr(params, name, None) if params is not None else None


def _int_raw(e, default=0):
    try:
        e = '' if e is None else e
        return int(float(e)) if str(e).strip() != '' else int(default)
    except Exception:
        return int(default)


def _pin_int(env_raw, params, gene, default_raw, lo, hi):
    """the pin, else the brain's gene, else the default for a brain without the gene (each env read by its caller)"""
    try:
        if str(env_raw or '').strip() != '':
            return max(lo, min(hi, int(float(env_raw))))
        g = _gene(params, gene)
        if g not in (None, ''):
            return max(lo, min(hi, int(float(g))))
    except Exception:
        pass
    return max(lo, min(hi, _int_raw(default_raw, 0)))


def ctx_of_params(params):
    return _pin_int(os.environ.get('MULTINXON_PIN_EFFECT_CTX', ''), params, 'effect_ctx', os.environ.get('MULTINXON_EFFECT_CTX_DEFAULT'), 0, 2)


def recall_of(params):
    return _pin_int(os.environ.get('MULTINXON_PIN_GOAL_RECALL', ''), params, 'goal_recall', os.environ.get('MULTINXON_GOAL_RECALL_DEFAULT'), 0, 1)


def prior_of(params):
    return _pin_int(os.environ.get('MULTINXON_PIN_GOAL_PRIOR', ''), params, 'goal_prior', os.environ.get('MULTINXON_GOAL_PRIOR_DEFAULT'), 0, 1)


def _float_gene(params, name, default, lo, hi):
    try:
        g = _gene(params, name)
        v = float(g) if g not in (None, '') else float(default)
    except Exception:
        v = float(default)
    return max(lo, min(hi, v))


# ================================================================================================ the colour prior (other games)
def prior_for(game):
    """{start colour: Counter(finished colour)} from the other games' level-1 finished boards (all games when the game is unknown)"""
    game = str(game or '').strip()
    if game in _PRIOR['cache']:
        return _PRIOR['cache'][game]
    if _PRIOR['doc'] is None:
        try:
            _PRIOR['doc'] = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'color_prior_1176.json'), encoding='utf-8'))
        except Exception:
            _PRIOR['doc'] = {}
    by = collections.defaultdict(collections.Counter)
    for g, v in ((_PRIOR['doc'] or {}).get('games') or {}).items():
        if g == game:
            continue
        for sc, fcs in (v.get('by') or {}).items():
            for fc, n in (fcs or {}).items():
                by[int(sc)][int(fc)] += int(n)
    _PRIOR['cache'][game] = dict(by)
    return _PRIOR['cache'][game]


def prior_decisions(prior):
    dec = {}
    for a, c in (prior or {}).items():
        n = sum(c.values())
        if n < PRIOR_MIN_N:
            continue
        col, k = c.most_common(1)[0]
        if k >= PRIOR_SHARE * n and int(col) != int(a):
            dec[int(a)] = int(col)
    return dec


# ================================================================================================ the goal map hooks (recall, prior)
def _install_goalmap():
    import goalmap1163 as GM
    if getattr(GM.GoalModel, '_v1176_hooks', False):
        return
    GM.GoalModel._v1176_hooks = True
    _wt0 = getattr(GM.GoalModel, 'what_table', None)

    def what_table(self):
        tab = _wt0(self) if _wt0 is not None else {'by': {}, 'global': None, 'n': 0}
        pr = getattr(self, 'prior76', None)
        if not pr or tab.get('_v1176'):
            return tab
        dec = prior_decisions(pr)
        add = {a: c for a, c in dec.items() if a not in tab['by']}
        tab = dict(tab, by=dict(tab['by'], **add), prior_used=len(add), _v1176=True)
        if add and not tab.get('n'):
            tab['n'] = 1                                                    # targets() needs a fitted table: the prior is one
        self._what_tab = tab
        return tab
    if _wt0 is not None:
        GM.GoalModel.what_table = what_table
    _pred0 = GM.GoalModel.predict                                           # effect1175's (strict, strict 3, colour targets)

    def predict(self, board, ignore=None, n=None):
        goal, p = _pred0(self, board, ignore, n)
        if not int(getattr(self, 'recall76', 0) or 0) or goal is None or not int(getattr(self, 'what', 0) or 0):
            return goal, p
        s_eff = int(getattr(self, 'strict_eff', 0) or 0) if int(getattr(self, 'strict3', 0) or 0) else int(getattr(self, 'strict', 0) or 0)
        if s_eff != 0 or not hasattr(self, 'targets'):
            return goal, p
        try:
            n0 = int(n if n is not None else self.n_goal())
            g2, _ = _pred0(self, board, ignore, max(1, int(round(RECALL_X * max(1, n0)))))
            if g2 is None:
                return goal, p
            base = np.asarray(goal, bool); wide = np.asarray(g2, bool)
            T2 = self.targets(board, wide)
            extra = wide & ~base & (T2 >= 0)
            if not extra.any():
                return goal, p
            m = base | extra
            T = np.where(m, T2, -1).astype(np.int16)
            try:
                import effect1175 as E5
                out = E5._with_what(m, T)
            except Exception:
                out = m
            self.recall_extra = int(extra.sum())
            return out, p
        except Exception:
            return goal, p
    GM.GoalModel.predict = predict


# ================================================================================================ the context of an action
def _shape(o):
    h, w, n = int(o.get('h') or 1), int(o.get('w') or 1), int(o.get('n') or 1)
    asp = 'q' if abs(h - w) <= 1 else ('w' if w > h else 't')
    f = n / float(max(1, h * w))
    return asp + ('s' if f >= 0.8 else ('h' if f <= 0.5 else 'm'))


def sig_ctx(st, akey, O, g):
    """(signature exactly as steer1162.sig_of, context) of an action on board g with objects O"""
    from steer1162 import sbk
    if akey is None:
        return None, None
    if akey[0] == 'K':
        sig = ('K', akey[1])
        return sig, _key_ctx(st, akey[1], g)
    y, x = akey[1], akey[2]
    best = None; bd = 1e9
    for o in O or []:
        d = max(abs(o['cy'] - y), abs(o['cx'] - x))
        if d < bd:
            bd, best = d, o
    if best is None or bd > 12:
        return ('P', -1), None
    sig = ('C', best['col'], sbk(best['n']), 0 if bd <= 1.5 else 1)
    shp = _shape(best)
    if (int(best['col']), sbk(best['n']), shp) in getattr(st, '_ctl76', ()):
        return sig, ('c', shp, 1, getattr(st, 'hq', None))
    return sig, ('c', shp, 0)


def _key_ctx(st, key, g):
    m = (getattr(st, 'mover', None) or {}).get(key)
    pos = getattr(st, '_av76', None)
    if not m or int(m[1]) < 1 or pos is None or g is None:
        return None
    dy, dx = m[2] / float(m[1]), m[3] / float(m[1])
    ty, tx = int(round(pos[0] + dy)), int(round(pos[1] + dx))
    H, W = np.asarray(g).shape[:2]
    if not (0 <= ty < H and 0 <= tx < W):
        return ('k', 'edge')
    col = int(np.asarray(g)[ty, tx])
    return ('k', 'bg' if col == int(getattr(st, 'bg', 0)) else col)


# ================================================================================================ prediction with abstention
def _agree(s):
    return max(s[2], s[3], s[4]) / float(max(1, s[1]))


def ctx_pred(st, sig, ctx):
    """(predicted mean dh or None, source): 'ctx', 'noeff', 'sig' predict; 'mixed', 'mixed_sig', 'unknown' abstain"""
    mn = st.ctx_min_n; ag = st.ctx_agree
    if sig is None:
        return None, 'unknown'
    if ctx is not None:
        s = st.ctx_dh.get((sig, ctx))
        if s and s[1] >= mn:
            if _agree(s) >= ag:
                return s[0] / float(s[1]), 'ctx'
            return None, 'mixed'
        e = st.ctx_eff.get((sig, ctx))
        if e and e[0] >= mn and e[1] == 0:
            return 0.0, 'noeff'
    s = st.sig_dhx.get(sig)
    if s and s[1] >= mn:
        if _agree(s) >= ag:
            return s[0] / float(s[1]), 'sig'
        return None, 'mixed_sig'
    return None, 'unknown'


def _bump(tab, key, dh):
    s = tab.get(key)
    if s is None:
        s = tab[key] = [0.0, 0, 0, 0, 0]
    s[0] += dh; s[1] += 1
    s[2 if dh < 0 else (3 if dh > 0 else 4)] += 1


def _truth_goal(st):
    """(finished board F, true goal mask) of the play's first level, or (None, None)"""
    if getattr(st, '_tg76', None) is not None:
        return st._tg76
    st._tg76 = (None, None)
    try:
        game = os.environ.get('MULTINXON_SENSE_TRUTH', '').strip()
        L = int(os.environ.get('MULTINXON_START_LEVEL', '1') or 1)
        if not game or st.start_board is None:
            return st._tg76
        import effect1174 as E4
        F = E4._truth(game, L)
        S = np.asarray(st.start_board, np.int16)
        if F is None or F.shape != S.shape:
            return st._tg76
        T = (S != F) & ~np.asarray(st.ignore(), bool)
        if T.any():
            st._tg76 = (F, T)
    except Exception:
        pass
    return st._tg76


def _h_true(F, T, g):
    return int((T & (np.asarray(g, np.int16) != F)).sum())


# ================================================================================================ install
def install(Steer4, Steer5=None):
    """idempotent; after effect1175.install. Steer5 (steer1164) gets the context-aware _dh_mean."""
    if getattr(Steer4, '_v1176_effect', False):
        return True
    Steer4._v1176_effect = True
    _install_goalmap()
    _init0 = Steer4.__init__; _start0 = Steer4.start_level; _after0 = Steer4._after_action
    _value0 = Steer4.value; _raises0 = Steer4.raises_h; _stats0 = Steer4.stats
    _sig0 = Steer4.sig_of

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        self.ectx = ctx_of_params(params)
        self.ctx_min_n = max(1, int(_float_gene(params, 'ctx_min_n', CTX_MIN_N, 1, 10)))
        self.ctx_agree = _float_gene(params, 'ctx_agree', CTX_AGREE, 0.5, 1.0)
        self.bench76 = bool(self.ectx >= 1 or str(os.environ.get('MULTINXON_DROPBENCH76', '0')) == '1')
        self.ctx_dh = {}; self.sig_dhx = {}; self.ctx_eff = {}; self._ctl76 = set(); self._av76 = None; self._ctx_cur = None
        self._tg76 = None; self._v1176_on_tallied = False
        self.recall76 = recall_of(params); self.prior76 = prior_of(params)
        coupled = 0
        try:
            if self.ectx >= 1 and str(os.environ.get('MULTINXON_PIN_GOAL_WHAT', '')).strip() == '' and _gene(params, 'goal_what') in (None, ''):
                self.goal_what = 1; coupled = 1
        except Exception:
            pass
        self._what_coupled = coupled
        try:
            self.gm.recall76 = self.recall76
            if coupled:
                self.gm.what = 1
            if self.prior76:
                self.gm.prior76 = prior_for(os.environ.get('MULTINXON_SENSE_TRUTH', ''))
        except Exception:
            pass

    def start_level(self, g):
        _start0(self, g)
        self.sig_dhx = {}; self.ctx_dh = {}                                  # dh is relative to THIS level's goal; ctx_eff stays
        self._av76 = None; self._ctx_cur = None
        if not getattr(self, '_v1176_on_tallied', True):
            self._v1176_on_tallied = True
            if self.ectx >= 1:
                E3._tally('ctx_on')
            if getattr(self, '_what_coupled', 0):
                E3._tally('what_coupled')
            if self.recall76:
                E3._tally('recall76')
            if self.prior76:
                E3._tally('prior76')
        try:
            if self.recall76 and int(getattr(self.gm, 'recall_extra', 0) or 0):
                E3._tally('recall_extra', int(self.gm.recall_extra)); self.gm.recall_extra = 0
            tab = self.gm.what_table() if (self.prior76 and hasattr(self.gm, 'what_table') and getattr(self.gm, 'what_pairs', None) is not None) else None
            if tab and tab.get('prior_used'):
                E3._tally('prior_used', int(tab['prior_used']))
        except Exception:
            pass

    def sig_of(self, akey, O):
        if not (getattr(self, 'ectx', 0) or getattr(self, 'bench76', False)):
            return _sig0(self, akey, O)
        try:
            g = self._cur[0] if getattr(self, '_cur', None) else None
            s, c = sig_ctx(self, akey, O, g)
            self._ctx_cur = (s, c)
            return s
        except Exception:
            return _sig0(self, akey, O)

    def _after_action(self, ak, g0, g1, ch, trans):
        if not (getattr(self, 'ectx', 0) or getattr(self, 'bench76', False)):
            return _after0(self, ak, g0, g1, ch, trans)
        pre = None
        try:
            if ak is not None and g0 is not None and g1 is not None and g0.shape == g1.shape and getattr(self, 'v4', 0) \
                    and not getattr(self, '_wake_skip', False):
                from steer1162 import components
                k0 = self.key(g0); O0 = self.objs.get(k0)
                if O0 is None:
                    O0 = components(g0, self.bg, self.ignore()); self.objs[k0] = O0
                sig, ctx = sig_ctx(self, ak, O0, g0)
                pm, src = ctx_pred(self, sig, ctx)
                bs = self.dh_sig.get(sig)
                bm = (bs[0] / float(bs[1])) if (bs and bs[1] > 0) else None
                F, T = _truth_goal(self) if int(getattr(self, '_v1174_nstart', 0) or 0) <= 1 else (None, None)
                ht0 = _h_true(F, T, g0) if F is not None else None
                pre = (sig, ctx, pm, src, bm, self.hq if self.goal is not None else None, ht0, F, T, O0)
        except Exception as _e:
            self._v1176_error = f'pre: {type(_e).__name__}: {_e}'[:160]
            pre = None
        _after0(self, ak, g0, g1, ch, trans)
        if pre is None:
            return
        try:
            sig, ctx, pm, src, bm, hq0, ht0, F, T, O0 = pre
            e = self.ctx_eff.get((sig, ctx))
            if e is None:
                e = self.ctx_eff[(sig, ctx)] = [0, 0]
            e[0] += 1; e[1] += int(bool(ch))
            ign = self.ignore()
            if ch and ak[0] == 'C' and sig and sig[0] == 'C' and sig[3] == 0:          # a click ON an object: did it act elsewhere?
                d = (g0 != g1) & ~ign
                o = min(O0, key=lambda o: max(abs(o['cy'] - ak[1]), abs(o['cx'] - ak[2])))
                y0, x0 = max(0, o['y0'] - 2), max(0, o['x0'] - 2)
                d2 = d.copy(); d2[y0:o['y0'] + o['h'] + 2, x0:o['x0'] + o['w'] + 2] = False
                if d2.any():
                    from steer1162 import sbk
                    self._ctl76.add((int(o['col']), sbk(o['n']), _shape(o)))
            if ak[0] == 'K' and ch:
                d = (g0 != g1) & ~ign & (g1 != self.bg)
                ys, xs = np.nonzero(d)
                if 0 < len(ys) <= max(4, int(0.05 * d.size)):
                    self._av76 = (float(ys.mean()), float(xs.mean()))
            dh = None
            if hq0 is not None and self.goal is not None and self.hq is not None:
                dh = float(self.hq - hq0)
                _bump(self.ctx_dh, (sig, ctx), dh); _bump(self.sig_dhx, sig, dh)
            if src in ('ctx', 'noeff', 'sig'):
                E3._tally('ctx_' + src)
            else:
                E3._tally('ctx_abst' if src.startswith('mixed') else 'ctx_unk')
            if dh is not None:                                                # the bench against the play's own goal map
                E3._tally('db_n')
                down = dh < 0
                if down:
                    E3._tally('db_down')
                if pm is None:
                    E3._tally('db_abst')
                elif pm < 0:
                    E3._tally('db_pdown')
                    if down:
                        E3._tally('db_tp')
                if bm is not None and bm < 0:
                    E3._tally('dbs_pdown')
                    if down:
                        E3._tally('dbs_tp')
            if ht0 is not None and F is not None:                             # the bench against the TRUE goal (first level)
                dt = _h_true(F, T, g1) - ht0
                E3._tally('dbt_n')
                if dt < 0:
                    E3._tally('dbt_down')
                if pm is not None and pm < 0:
                    E3._tally('dbt_pdown')
                    if dt < 0:
                        E3._tally('dbt_tp')
                if bm is not None and bm < 0:
                    E3._tally('dbts_pdown')
                    if dt < 0:
                        E3._tally('dbts_tp')
        except Exception as _e:
            self._v1176_error = f'after: {type(_e).__name__}: {_e}'[:160]

    def value(self, sig, O, akey):
        if getattr(self, 'ectx', 0) < 2 or self.goal is None or self.goal_w <= 0:
            return _value0(self, sig, O, akey)
        gw = self.goal_w
        self.goal_w = 0.0
        try:
            v, p = _value0(self, sig, O, akey)
        finally:
            self.goal_w = gw
        try:
            g = self._cur[0] if getattr(self, '_cur', None) else None
            s, c = sig_ctx(self, akey, O, g)
            m, _src = ctx_pred(self, s, c)
            if m is not None:
                v += gw * max(-1.0, min(1.0, -m))
        except Exception:
            pass
        return v, p

    def raises_h(self, sig):
        if getattr(self, 'ectx', 0) < 2:
            return _raises0(self, sig)
        cur = getattr(self, '_ctx_cur', None)
        ctx = cur[1] if (cur and cur[0] == sig) else None
        m, _src = ctx_pred(self, sig, ctx)
        return bool(m is not None and m > 0.5)

    def stats(self):
        s = _stats0(self)
        s.update(effect_ctx=getattr(self, 'ectx', 0), ctx_entries=len(getattr(self, 'ctx_eff', {}) or {}),
                 controls=len(getattr(self, '_ctl76', ()) or ()), goal_recall=getattr(self, 'recall76', 0), goal_prior=getattr(self, 'prior76', 0),
                 v1176_error=getattr(self, '_v1176_error', None))
        return s

    Steer4.__init__ = __init__; Steer4.start_level = start_level; Steer4._after_action = _after_action
    Steer4.value = value; Steer4.raises_h = raises_h; Steer4.stats = stats; Steer4.sig_of = sig_of
    if Steer5 is not None and not getattr(Steer5, '_v1176_effect', False):
        Steer5._v1176_effect = True
        _dm0 = Steer5._dh_mean

        def _dh_mean(self, sig):
            if getattr(self, 'ectx', 0) < 1:
                return _dm0(self, sig)
            cur = getattr(self, '_ctx_cur', None)
            ctx = cur[1] if (cur and cur[0] == sig) else None
            m, _src = ctx_pred(self, sig, ctx)
            return m
        Steer5._dh_mean = _dh_mean
    return True
