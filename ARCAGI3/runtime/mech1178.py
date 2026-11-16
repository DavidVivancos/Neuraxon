"""mech1178.py -- NAS 1178 change 2: THE MECHANISMS AS GENES, DECIDED PER LINE AND PER GAME, NEVER FORCED.

WHAT 1177 MEASURED (nas_1177_report.md sections 7 and 10): with every milestone-2 mechanism forced on every brain, the same brains on
the same seeds played 34.6% forced and 34.6% as their 1176 selves -- +11.0 / +9.4 / +8.9 / +5.2 points on bp35 / su15 / lf52 / ar25 (the
games with an undo), -9.6 on vc33, -7.8 on ls20, -7.2 on sk48, -3.5 on the keys games, +12.5 on the lines selected with them. All ten were
on together, so nothing said which one did what; the fatal penalty fired 450 / 237 / 195 times per 1,000 steps on su15 / vc33 / tu93, the
trained policies chose the newly offered ACTION7 3.41 million times (vetoed), the pace rule did not shorten the keys games' routes.

WHAT 1178 DOES:
  * TEN FACTORS, each a gene (absent = off) a pin can override (MULTINXON_PIN_<GENE>):
        bus     bus78_gain     the sensory bus's 1177 synapses (bus1178)            ctx     effect_ctx 3 vs the signature
        anim    anim77         animations in the effect model                        death   death77    the BAR rule and the pace
        fatal   fatal78        the fatal-context penalty and the suspects (was inside death77)
        undo    undo77         undo as a probe; off = the brain's own 1176 action mask (no ACTION7 it never had)
        guard   guard77        no-op guards                                          macro   macro77 + transfer77
        plan    plan78         the in-play planner (plan1178)                        prior   prior78   the cross-game effect prior (prior1178)
  * THE BUILD'S DEFAULT for every brain: the configuration it was SELECTED under -- a brain bred or kept by a 1177 line plays as it was
    selected (the 1177 mechanisms on, its 1177 bus synapses on); every older brain plays as its 1176 self, except that undo is on for the
    games where the bench measured it (bp35, su15, lf52, ar25); plan / prior on for half of the lines (a hash of the cell), so the run
    compares lines as well as seeds.
  * THE FACTORIAL BENCH (nas1178): at generation 0 of every level line the top champion plays the 32 runs of a 2^(10-5) resolution-IV
    fractional factorial (every main effect clear of every two-factor interaction) on 5 seeds each (160 plays): one main effect per
    factor per line (80 plays a side). A line switches a factor only when its own effect is >= 2 standard errors; every line's effects
    go into the box's advance file, pooled per game, for the next build -- the per-game decision the 1177 bench could not make.
TALLY: st_m8_<factor>_on per brain built."""
from __future__ import annotations

import hashlib
import math
import os

FACTORS = ('bus', 'ctx', 'anim', 'death', 'fatal', 'undo', 'guard', 'macro', 'plan', 'prior')
GENE_OF = {'bus': ('bus78_gain',), 'ctx': ('effect_ctx',), 'anim': ('anim77',), 'death': ('death77',), 'fatal': ('fatal78',),
           'undo': ('undo77',), 'guard': ('guard77',), 'macro': ('macro77', 'transfer77'), 'plan': ('plan78',), 'prior': ('prior78',)}
PIN_OF = {'bus78_gain': 'MULTINXON_PIN_BUS78', 'effect_ctx': 'MULTINXON_PIN_EFFECT_CTX', 'anim77': 'MULTINXON_PIN_ANIM77',
          'death77': 'MULTINXON_PIN_DEATH77', 'fatal78': 'MULTINXON_PIN_FATAL78', 'undo77': 'MULTINXON_PIN_UNDO77',
          'guard77': 'MULTINXON_PIN_GUARD77', 'macro77': 'MULTINXON_PIN_MACRO77', 'transfer77': 'MULTINXON_PIN_TRANSFER77',
          'plan78': 'MULTINXON_PIN_PLAN78', 'prior78': 'MULTINXON_PIN_PRIOR78'}
UNDO_GAMES = ('bp35', 'su15', 'lf52', 'ar25')
MECH77 = ('anim77', 'death77', 'undo77', 'guard77', 'transfer77', 'macro77')
NEW_GENES = ('fatal78', 'plan78', 'prior78')
SEEDS_PER_RUN = 5
SWITCH_SE = 2.0


# ======================================================================================================== the design
def design():
    """the 32 runs of the 2^(10-5) resolution-IV design: base factors A..E (bus, ctx, anim, death, fatal) full factorial; generators
    undo = ABCD, guard = ABCE, macro = ABDE, plan = ACDE, prior = BCDE (every word >= 5 letters; pairs of words >= 4: resolution IV).
    Returns [{factor: +1 / -1}] in run order."""
    runs = []
    for i in range(32):
        A, B, C, D, E = [1 if (i >> j) & 1 else -1 for j in range(5)]
        lv = {'bus': A, 'ctx': B, 'anim': C, 'death': D, 'fatal': E, 'undo': A * B * C * D, 'guard': A * B * C * E, 'macro': A * B * D * E,
              'plan': A * C * D * E, 'prior': B * C * D * E}
        runs.append(lv)
    return runs


def check_design():
    """every factor balanced (16 / 16) and every pair of factors orthogonal (resolution >= III), and no main effect aliased with a
    two-factor interaction (resolution IV)"""
    R = design()
    for f in FACTORS:
        if sum(r[f] for r in R) != 0:
            return False, f'unbalanced {f}'
    for i, a in enumerate(FACTORS):
        for b in FACTORS[i + 1:]:
            if sum(r[a] * r[b] for r in R) != 0:
                return False, f'not orthogonal {a} {b}'
    for a in FACTORS:
        for i, b in enumerate(FACTORS):
            for c in FACTORS[i + 1:]:
                if a in (b, c):
                    continue
                if abs(sum(r[a] * r[b] * r[c] for r in R)) == len(R):
                    return False, f'{a} aliased with {b}x{c}'
    return True, f'{len(R)} runs, {len(FACTORS)} factors, resolution IV'


def env_of(run, ctx_off='0'):
    """the pins of one run of the design (every factor explicit), with the SAME values the line's configuration writes
    (genes_of_config): the bench measures what a line would run -- plan78 2 (not 1), prior78 1 WITH l1probe78 4"""
    e = {}
    for g, v in genes_of_config({f: (1 if lv > 0 else 0) for f, lv in run.items()}, b0_ctx=ctx_off).items():
        e[PIN_OF.get(g) or 'MULTINXON_PIN_' + g.upper()] = str(v)
    return e


def effects(rows):
    """rows: [(run levels dict, clears, plays)] -> {factor: {'d': rate(+) - rate(-), 'se', 'n_plus', 'n_minus', 'p_plus', 'p_minus'}}"""
    out = {}
    for f in FACTORS:
        cp = np_ = cm = nm = 0
        for lv, c, n in rows:
            if lv.get(f, 0) > 0:
                cp += int(c); np_ += int(n)
            else:
                cm += int(c); nm += int(n)
        if not np_ or not nm:
            continue
        pp = cp / float(np_); pm = cm / float(nm)
        pbar = (cp + cm) / float(np_ + nm)
        se = math.sqrt(max(1e-9, pbar * (1 - pbar) * (1.0 / np_ + 1.0 / nm)))
        out[f] = {'d': round(pp - pm, 4), 'se': round(se, 4), 'p_plus': round(pp, 4), 'p_minus': round(pm, 4), 'n_plus': np_, 'n_minus': nm,
                  'c_plus': cp, 'c_minus': cm}
    return out


def decide(cur, eff, se_k=SWITCH_SE):
    """the line's configuration after its bench: a factor switches only when its own effect is >= se_k standard errors"""
    new = dict(cur); switched = {}
    for f, e in (eff or {}).items():
        if e['se'] <= 0:
            continue
        z = e['d'] / e['se']
        want = 1 if z >= se_k else (0 if z <= -se_k else None)
        if want is not None and int(cur.get(f, 0)) != want:
            new[f] = want; switched[f] = round(z, 2)
    return new, switched


# ======================================================================================================== the configurations
def config_of_arch(arch):
    """{factor: 0/1} of a brain's arch (genes; absent = off; effect_ctx 3 = ctx on)"""
    nn = (arch or {}).get('neural') or {}

    def g(name):
        try:
            return int(float(nn.get(name) or 0))
        except Exception:
            return 0
    return {'bus': 1 if g('bus78_gain') > 0 else 0, 'ctx': 1 if g('effect_ctx') == 3 else 0, 'anim': g('anim77'), 'death': g('death77'),
            'fatal': g('fatal78'), 'undo': g('undo77'), 'guard': g('guard77'), 'macro': 1 if (g('macro77') or g('transfer77')) else 0,
            'plan': 1 if g('plan78') > 0 else 0, 'prior': 1 if g('prior78') > 0 else 0}


def half_on(cell, salt='plan78'):
    """plan / prior on for half the lines, by a hash of the cell name (stable across builds)"""
    return int(hashlib.md5(f'{salt}:{cell}'.encode()).hexdigest()[:8], 16) % 2


def default_config(selected77, game, cell, b0=None):
    """the build's default configuration: the one the brain was selected under, + undo on the undo games; the planner and the prior OFF
    (no brain was selected with them: the line's bench switches them on where they win at 2 SE -- measured before shipping: the planner
    +1 / 78 plays (+0.6 SE), the prior's values -1 / 88 at level-1 k0 (-1.0 SE))"""
    if selected77:
        cfg = {'bus': 1, 'ctx': 1, 'anim': 1, 'death': 1, 'fatal': 1, 'undo': 1, 'guard': 1, 'macro': 1}
    else:
        cfg = {'bus': 0, 'ctx': 0, 'anim': 0, 'death': 0, 'fatal': 0, 'undo': 1 if game in UNDO_GAMES else 0, 'guard': 0, 'macro': 0}
    cfg['plan'] = 0
    cfg['prior'] = 0
    return cfg


def set_config(arch, cfg, b0_ctx=None):
    """write a configuration into an arch (neural genes and _meta.genes when present)"""
    nn = arch.setdefault('neural', {})
    genes = genes_of_config(cfg, b0_ctx)
    nn.update(genes)
    mg = (arch.get('_meta') or {}).get('genes')
    if isinstance(mg, dict):
        for k, v in genes.items():
            mg[k] = v
    return arch


# ======================================================================================================== the steering side
def gene(params, name):
    """pin > gene > 0 (no forcing environment in 1178)"""
    e = os.environ.get(PIN_OF.get(name, 'MULTINXON_PIN_' + name.upper()), '')
    try:
        if str(e).strip() != '':
            return int(float(e))
        v = getattr(params, name, None)
        return int(float(v)) if v not in (None, '') else 0
    except Exception:
        return 0


def _cell77():
    """a 1177 cell (the 1177 forcing environment): 1178's rules stay out of it"""
    return str(os.environ.get('MULTINXON_EYES77', '0')) == '1'


def install(Steer4, ArcBrain):
    """the 1177 steering reads its mechanisms per brain (no cell forcing): e77_on = any mechanism on (or a live bus block reading the
    steering's state, bus1178); the context upscale of 1177 change 9 comes with the ctx factor (effect_ctx 3); fatal78 split from death77
    (absent = as 1177: inside death77); undo off = the brain's own 1176 action mask (no ACTION7 that 1177 added), undo on = ACTION7 in the
    mask; the bench's bus-off arm (MULTINXON_PIN_BUS78 = 0) reads zeros on the bus. Must be installed AFTER bus1178.install."""
    if getattr(Steer4, '_v1178_mech', False):
        return True
    Steer4._v1178_mech = True
    import steer1177 as S7
    _init0 = Steer4.__init__

    def __init__(self, params, seed=0):
        if _cell77():
            return _init0(self, params, seed=seed)
        saved = getattr(params, 'eyes77', None)
        try:
            try:
                params.eyes77 = 0                                           # e77_on from the genes, never from the eyes
            except Exception:
                pass
            _init0(self, params, seed=seed)
        finally:
            try:
                params.eyes77 = saved
            except Exception:
                pass
        try:
            g = dict(getattr(self, 'g77', {}) or {})
            fp = os.environ.get(PIN_OF['fatal78'], '')
            fv = getattr(params, 'fatal78', None)
            if str(fp).strip() != '' or fv not in (None, ''):
                g['fatal78'] = gene(params, 'fatal78')
            else:
                g['fatal78'] = int(g.get('death77', 0) or 0)                # absent: as 1177 (inside death77)
            self.g77 = g
            ctrl = str(os.environ.get('MULTINXON_V1177_CTRL', '0')) == '1'
            on = any(int(v or 0) for v in g.values()) and not ctrl
            self.e77_on = bool(on)
            if on:
                self.sense_w = 16
            ctx3 = int(getattr(self, 'ectx', 0) or 0) == 3
            for k_, v_ in S7.UPSCALE77.items():
                if on and ctx3:
                    setattr(self, k_, v_)
                elif k_ in self.__dict__:
                    delattr(self, k_)                                       # the 1176 defaults (getattr fallbacks)
            self._m78 = {'on': int(on), 'upscale': int(bool(on and ctx3))}
        except Exception as e:
            self._v1178_error = f'mech init: {type(e).__name__}: {e}'[:160]
    Steer4.__init__ = __init__
    _start0 = Steer4.start_level

    def start_level(self, g):
        _start0(self, g)
        c = getattr(self, '_m78_cfg', None)
        if c and not getattr(self, '_m78_tallied', False):
            self._m78_tallied = True
            for f, v in c.items():
                if v:
                    _tally(f'm8_{f}_on')
            _tally('m8_plays')
    Steer4.start_level = start_level
    _binit0 = ArcBrain.__init__

    def binit(self, params, *a, **k):
        _binit0(self, params, *a, **k)
        if _cell77():
            return
        try:
            u = gene(params, 'undo77')
            b0 = getattr(params, 'eyes77_b0', None) or {}
            m0 = b0.get('action_mask') if isinstance(b0, dict) else None
            am = getattr(self, '_action_mask', None)
            if u == 0 and m0 not in (None, '') and '7' not in str(m0) and am is not None and 7 in am:
                self._action_mask = set(int(c) for c in str(m0) if c.isdigit()) or None     # 1177 added ACTION7: take it out
                _tally('m8_undo_off')
            elif u == 1 and am is not None and 7 not in am:
                self._action_mask = set(am) | {7}                           # the undo offered (the engine's own list still decides)
                _tally('m8_undo_on')
            # (the bus-off arm -- MULTINXON_PIN_BUS78 = 0 or the gene bus78_gain 0 -- is set by bus1178's init: the bus reads zeros)
            st = getattr(self, '_steer', None)
            if st is not None and getattr(st, '_m78', None) is not None and (getattr(self, '_b78', False) or any(
                    getattr(params, g_, None) not in (None, '') for g_ in NEW_GENES)):
                st._m78_cfg = config_of_params(params, st)                  # tallied at the play's first level (construction tallies are reset)
        except Exception as e:
            self._v1178_error = f'mech brain: {type(e).__name__}: {e}'[:160]
    ArcBrain.__init__ = binit
    return True


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def config_of_params(params, st=None):
    """the configuration a brain plays with (genes and pins), for the tallies"""
    g = (getattr(st, 'g77', None) or {}) if st is not None else {}
    try:
        bus = 1 if (int(float(os.environ.get('MULTINXON_PIN_BUS78', '') or getattr(params, 'bus78_gain', 0) or 0)) > 0) else 0
    except Exception:
        bus = 0
    return {'bus': bus, 'ctx': 1 if int(getattr(st, 'ectx', 0) or 0) == 3 else 0, 'anim': int(g.get('anim77', 0) or 0),
            'death': int(g.get('death77', 0) or 0), 'fatal': int(g.get('fatal78', 0) or 0), 'undo': int(g.get('undo77', 0) or 0),
            'guard': int(g.get('guard77', 0) or 0), 'macro': 1 if (g.get('macro77') or g.get('transfer77')) else 0,
            'plan': 1 if gene(params, 'plan78') > 0 else 0, 'prior': 1 if gene(params, 'prior78') > 0 else 0}


# ======================================================================================================== the build's per-brain configuration
BEHAVIOUR77 = ('settled_view', 'change_border', 'hud_border', 'death_memory')


def self76(arch, undo=0):
    """the arch of a brain as its 1176 self: the behaviour genes, effect_ctx and the action mask back to its own pre-1177 values
    (neural.eyes77_b0), every 1177 mechanism off -- except undo when asked (ACTION7 in the mask)"""
    nn = arch.setdefault('neural', {})
    b0 = dict(nn.get('eyes77_b0') or {})
    for g in BEHAVIOUR77 + ('effect_ctx',):
        if g in b0:
            if b0[g] is None:
                nn.pop(g, None)
            else:
                nn[g] = b0[g]
    if b0.get('action_mask') not in (None, ''):
        nn['action_mask'] = b0['action_mask']
    if undo:
        import evo1177 as EV7
        if nn.get('action_mask') not in (None, ''):
            nn['action_mask'] = EV7.with_undo(nn['action_mask'])
    for g in MECH77:
        nn[g] = 0
    nn['undo77'] = 1 if undo else 0
    mg = (arch.get('_meta') or {}).get('genes')
    if isinstance(mg, dict):
        for g in BEHAVIOUR77 + ('effect_ctx', 'action_mask') + MECH77:
            if g in nn:
                mg[g] = nn[g]
            else:
                mg.pop(g, None)
    return arch


def build_config(arch, selected77, game, cell):
    """the build's configuration of a migrated brain (written into its arch): the 1177 one when it was selected under it (bus gain 1, every
    1177 mechanism and behaviour gene as 1177 set them, fatal78 = death77), else its 1176 self (+ undo on the undo games); plan78 / prior78
    on for half the lines (2 = 200 boards a plan; the prior on); l1probe78 4 where the prior is on"""
    cfg = default_config(selected77, game, cell)
    nn = arch.setdefault('neural', {})
    if not selected77:
        self76(arch, undo=cfg['undo'])
    else:
        for g in MECH77:
            nn[g] = 1
    nn['fatal78'] = 1 if cfg['fatal'] else 0
    nn['plan78'] = 2 if cfg['plan'] else 0
    nn['prior78'] = 1 if cfg['prior'] else 0
    nn['l1probe78'] = 4 if cfg['prior'] else 0
    mg = (arch.get('_meta') or {}).get('genes')
    if isinstance(mg, dict):
        for g in MECH77 + NEW_GENES + ('l1probe78',):
            if g in nn:
                mg[g] = nn[g]
    return cfg


def genes_of_config(cfg, b0_ctx=None):
    """the genes that realise a configuration in an arch (plan 2 = 200 boards; l1probe follows the prior)"""
    out = {}
    for f, lv in cfg.items():
        for gname in GENE_OF[f]:
            if gname == 'effect_ctx':
                out[gname] = 3 if lv else (int(b0_ctx) if b0_ctx not in (None, '') and int(b0_ctx) != 3 else 0)
            elif gname == 'plan78':
                out[gname] = 2 if lv else 0
            else:
                out[gname] = 1 if lv else 0
        if f == 'prior':
            out['l1probe78'] = 4 if lv else 0
    return out
