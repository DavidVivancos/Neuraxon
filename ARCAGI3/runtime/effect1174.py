"""effect1174.py -- NAS 1174: the brain side of nas_1173_report.md section 25 (changes 2, 5, 7) as revised by FrameLab 18.

Every brain still proposes every action; nothing here knows a game's solution. All of it is behind genes (bandit arms, twins) or
environment switches that the NAS sets for a cell; a brain without the new genes plays exactly as in 1173.

warm_prefix 3 (change 2: GAME MEMORY; FL18: +1.4 frames-only, +3.2 / +4.5 with seen signatures)
  warm_prefix 1 (the level's verified prefix moves feed the steering's memories: their effects, never the sequence) plus the
  EFFECTS of the earlier levels' verified moves (game_loop records them as warm_game while it replays the prefix): the exact
  (board, action) -> change memo, the signature replays and the signature's change counts -- only moves that CHANGED the board
  (a move that did nothing on an earlier level is never taught as a no-op here: FL15's silent moves). Not the graph: the earlier
  levels' boards are dropped from the steering's board graph after the feed (FL18's 'game' memory: kset, record, learn; not the
  graph). Tally st_warm_game. warm_prefix 0 / 1 / 2 are 1173's; MULTINXON_WARM_DEFAULT is the value of a brain WITHOUT the gene
  (1174: 1 on the end-first lines -- the default the 1173 twins earned, +779 / -573, p 2e-8).

goal_strict 0 / 1 / 2 (change 7 + FL18: "name fewer, surer predicates"; precision costs 2-3 x as much as recall)
  the 1163 goal map's cells (the steering's h, the value term, effect_mem 2's look) are the top n by the cell model's probability,
  n = the mean arrival count of the levels seen (0). 1 keeps the top 50% with p >= 0.5, 2 the top 25% with p >= 0.7. Tally
  st_goal_strict. FrameLab 19 (23 Oct) maps how much precision is worth how much recall: its answer moves the arm's prior.

THE GOAL AUDIT (change 7; measurement only -- the steering never decides with it)
  At the first goal fit of a play the predicted goal cells are compared with the level's TRUE arrivals from the board the play is
  on (boards1164.npz: the verified finished board of MULTINXON_SENSE_TRUTH's level MULTINXON_START_LEVEL): the precision and recall
  on the cells still open, where FL18 says the goal is used. Tallies st_goal_audit (plays audited), st_goal_pred (predicted cells),
  st_goal_truth (true open arrivals), st_goal_tp (predicted and true). precision = tp / pred, recall = tp / truth.

home_games (change 5: the home rule generalised)
  the sense applies another game's priors only where the brain has no measured strength: a brain keeps its own genes on every game
  of its home_games gene (a list, from its measured record: build_champions_1174.py) as well as on MULTINXON_HOME_GAME (now a list).

MULTINXON_SENSE_LOGO = 1 (change 5: the hidden-game case)
  the game sense classifies as if the game being played were not in its index (leave one game out): it must pick the NEAREST known
  game, as on a game it has never seen. Used by the universality rows of the generalist cells (univ_logo). Tally st_sense_logo."""
from __future__ import annotations

import os

import numpy as np

import effect1173 as E3

WARM_GAME_MAX = 400                                     # the earlier levels' moves taught (the most recent)
GOAL_STRICT = {0: (1.0, 0.0), 1: (0.5, 0.5), 2: (0.25, 0.7)}   # goal_strict -> (share of the n cells kept, probability floor)
_TRUTH = {'z': None, 'ok': None}


def _pin(env, params, gene, hi, default=0):
    e = os.environ.get(env, '')
    try:
        if e.strip() != '':
            v = int(float(e))
        else:
            g = getattr(params, gene, None) if params is not None else None
            v = int(float(g)) if g not in (None, '') else int(default)
    except Exception:
        v = int(default)
    return max(0, min(hi, v))


def warm_default():
    try:
        return max(0, min(3, int(float(os.environ.get('MULTINXON_WARM_DEFAULT', '0') or 0))))
    except Exception:
        return 0


def warm_of(params):
    """0 / 1 / 2 / 3: the pin, else the brain's gene, else the cell's default for a brain without the gene"""
    return _pin('MULTINXON_PIN_WARM', params, 'warm_prefix', 3, default=warm_default())


def strict_of(params):
    return _pin('MULTINXON_PIN_GOAL_STRICT', params, 'goal_strict', 2)


def home_games_of(params):
    out = set()
    for s in str(os.environ.get('MULTINXON_HOME_GAME', '') or '').split(','):
        if s.strip():
            out.add(s.strip())
    hg = getattr(params, 'home_games', None) if params is not None else None
    if isinstance(hg, (list, tuple, set)):
        out |= {str(x).strip() for x in hg if str(x).strip()}
    elif hg not in (None, ''):
        out |= {x.strip() for x in str(hg).split(',') if x.strip()}
    h1 = getattr(params, 'home_game', None) if params is not None else None
    if h1 not in (None, '') and not os.environ.get('MULTINXON_HOME_GAME', '').strip():
        out.add(str(h1).strip())
    return out


# ================================================================================================ game memory (warm_prefix 3)
def _feed_game(st, items):
    """the earlier levels' moves that changed the board: their effects join the steering's memories (not its graph)"""
    import goal1170 as G70
    from steer1162 import components
    n = 0
    g_keys = set(getattr(st, 'graph', {}) or {}); o_keys = set(getattr(st, 'objs', {}) or {})
    for it in items:
        try:
            a_str, g0, g1 = it[1], it[2], it[3]
        except Exception:
            continue
        ak = E3.akey_of(a_str)
        if ak is None or g0 is None or g1 is None:
            continue
        try:
            g0 = np.asarray(g0, dtype=np.int16); g1 = np.asarray(g1, dtype=np.int16)
            if g0.shape != g1.shape or not bool((g0 != g1).any()):
                continue                                                    # only moves that changed the board teach (FL15)
            O0 = components(g0, st.bg, st.ignore())
            sig = st.sig_of(ak, O0)
            if getattr(st, 'v3', 0):                                        # the signature model learns the move as Steer3._after_action
                O1 = components(g1, st.bg, st.ignore())                     # does: its count AND its effects (a count without the
                st.sig_n[sig] += 1; st.sig_chg[sig] += 1                   # effects would make the model predict "nothing happens")
                for e in st.effects(O0, O1, ak):
                    st.mem[sig][e] += 1
            if getattr(st, 'fl_exact', None) is not None:
                G70.record(st, ak, g0, g1)                                  # the exact memo and the signature's replay
            n += 1
        except Exception as e:
            st._v1174_error = f'game memory: {type(e).__name__}: {e}'[:160]
    try:                                                                    # not the graph (FL18): the earlier levels' boards leave it
        for k in set(getattr(st, 'graph', {}) or {}) - g_keys:
            st.graph.pop(k, None)
        for k in set(getattr(st, 'objs', {}) or {}) - o_keys:
            st.objs.pop(k, None)
    except Exception:
        pass
    return n


_feed_warm73 = E3.feed_warm


def feed_warm(brain, prior):
    """1173's feed (warm 1 / 2), and for warm_prefix 3: the earlier levels' effects first, then this level's prefix as warm 1"""
    st = getattr(brain, '_steer', None)
    mode = getattr(st, 'warm', None) if st is not None else None
    if mode is None:
        mode = warm_of(getattr(brain, 'params', None))
    if mode != 3:
        return _feed_warm73(brain, prior)
    n = 0
    if st is not None and not getattr(brain, '_v1174_game_done', False):
        brain._v1174_game_done = True
        items = [x for x in (prior or {}).get('warm_game') or [] if x is not None][-WARM_GAME_MAX:]
        if items:
            n = _feed_game(st, items)
            if n:
                E3._tally('warm_game', n)
    try:
        st.warm = 1                                                         # this level's prefix as warm_prefix 1 (no watch)
        n += _feed_warm73(brain, prior)
    finally:
        st.warm = 3
    return n


# ================================================================================================ the goal: goal_strict
def _install_goalmap():
    import goalmap1163 as GM
    if getattr(GM.GoalModel, '_v1174_strict', False):
        return
    GM.GoalModel._v1174_strict = True
    _pred0 = GM.GoalModel.predict

    def predict(self, board, ignore=None, n=None):
        s = int(getattr(self, 'strict', 0) or 0)
        if s <= 0:
            return _pred0(self, board, ignore, n)
        frac, pmin = GOAL_STRICT.get(s, GOAL_STRICT[2])
        if self.fit() is None:
            return None, None
        n0 = int(n if n is not None else self.n_goal())
        goal, p = _pred0(self, board, ignore, max(1, int(round(frac * max(1, n0)))))
        if goal is not None and p is not None and pmin > 0:
            goal = goal & (p >= pmin)
        return goal, p
    GM.GoalModel.predict = predict


# ================================================================================================ the goal audit
def _truth(game, L):
    """the level's verified finished board (boards1164.npz), or None"""
    if _TRUTH['z'] is None and _TRUTH['ok'] is None:
        try:
            _TRUTH['z'] = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boards1164.npz'))
            _TRUTH['ok'] = True
        except Exception:
            _TRUTH['ok'] = False
    z = _TRUTH['z']
    if z is None:
        return None
    key = f'{game}:{int(L)}:finished'
    return np.asarray(z[key], dtype=np.int16) if key in z.files else None


def audit(st):
    """once per play, at the first goal fit ON THE PLAY'S FIRST LEVEL: the goal's precision / recall on the cells still open (tallies
    only; a continuation level is another level than MULTINXON_START_LEVEL's truth)"""
    if getattr(st, '_v1174_audited', False) or getattr(st, 'goal', None) is None or int(getattr(st, '_v1174_nstart', 0) or 0) > 1:
        return
    st._v1174_audited = True
    game = os.environ.get('MULTINXON_SENSE_TRUTH', '').strip()
    try:
        L = int(os.environ.get('MULTINXON_START_LEVEL', '1') or 1)
    except Exception:
        L = 1
    if not game:
        return
    try:
        import goalmap1163 as GM
        F = _truth(game, L)
        cur = st._cur[0] if getattr(st, '_cur', None) else st.start_board
        if F is None or cur is None:
            return
        cur = np.asarray(cur, dtype=np.int16); goal = np.asarray(st.goal, bool)
        if F.shape != cur.shape or goal.shape != cur.shape:
            return
        ign = None
        try:
            ign = st.ignore()
        except Exception:
            ign = None
        tr = np.asarray(GM.arrivals(cur, F, ign if (ign is not None and np.asarray(ign).shape == cur.shape) else None), bool)
        E3._tally('goal_audit'); E3._tally('goal_pred', int(goal.sum())); E3._tally('goal_truth', int(tr.sum()))
        E3._tally('goal_tp', int((goal & tr).sum()))
    except Exception as e:
        st._v1174_error = f'audit: {type(e).__name__}: {e}'[:160]


# ================================================================================================ the sense: home_games, LOGO
def _sense_set(st, game, final=True):
    """1173's _sense_set with the generalised home rule (a set of home games)"""
    import sense1173 as S
    prev = st.sense_game
    st.sense_game = game; st.sense_final = bool(final)
    truth = os.environ.get('MULTINXON_SENSE_TRUTH', '').strip()
    if truth:
        E3._tally(('sense_hit' if truth == game else 'sense_miss') if final else ('sense_bhit' if truth == game else 'sense_bmiss'))
    homes = getattr(st, 'home_games', None) or ({st.home_game} if st.home_game else set())
    if game in homes:
        if final or prev is None:
            E3._tally('sense_home')
        st._sense_pending = None
        if st.sense_applied and st.sense_applied != game:
            st._sense_pending = dict(st._sense_own, game=game) if getattr(st, '_sense_own', None) else None
        return
    if st.sense_applied == game:
        return
    pri = S.priors_cached().get(game)
    st._sense_pending = dict(pri, game=game) if pri else None


def _install_logo():
    import sense1173 as S
    if getattr(S.Index, '_v1174_logo', False):
        return
    S.Index._v1174_logo = True
    _cls0 = S.Index.classify

    def classify(self, board, obs=None, exclude=None, w_fx=S.W_FX, w_efx=S.W_EFX):
        g, m, rank = _cls0(self, board, obs=obs, exclude=exclude, w_fx=w_fx, w_efx=w_efx)
        truth = os.environ.get('MULTINXON_SENSE_TRUTH', '').strip()
        if os.environ.get('MULTINXON_SENSE_LOGO', '') != '1' or not truth or not rank:
            return g, m, rank
        rank = [(x, s) for x, s in rank if x != truth]                 # the game being played is not in the index
        E3._tally('sense_logo')
        if not rank:
            return None, 0.0, []
        margin = rank[0][1] - rank[1][1] if len(rank) > 1 else float('inf')
        return rank[0][0], margin, rank
    S.Index.classify = classify


# ================================================================================================ install
def install(Steer4):
    """idempotent; after effect1173.install"""
    if getattr(Steer4, '_v1174_effect', False):
        return True
    Steer4._v1174_effect = True
    E3.warm_of = warm_of                                                   # 1173's __init__ reads it at call time: 0..3 + the default
    E3.feed_warm = feed_warm                                               # game_loop calls effect1173.feed_warm at call time
    E3._sense_set = _sense_set                                             # the home rule over a set of games
    _install_goalmap(); _install_logo()
    _init0 = Steer4.__init__; _start0 = Steer4.start_level; _goal_only0 = getattr(Steer4, 'start_level_goal_only', None)
    _stats0 = Steer4.stats

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        self.goal_strict = strict_of(params)
        try:
            self.gm.strict = self.goal_strict
        except Exception:
            pass
        self.home_games = home_games_of(params)
        self._v1174_audited = False; self._v1174_nstart = 0

    def start_level(self, g):
        self._v1174_nstart = int(getattr(self, '_v1174_nstart', 0) or 0) + 1
        if self._v1174_nstart == 1 and getattr(self, 'goal_strict', 0):
            E3._tally('goal_strict')                                        # once per play (the per-try tally starts after the replay)
        _start0(self, g)
        audit(self)

    def stats(self):
        s = _stats0(self)
        s.update(goal_strict=getattr(self, 'goal_strict', 0), home_games=sorted(getattr(self, 'home_games', set()) or []),
                 v1174_error=getattr(self, '_v1174_error', None))
        return s

    Steer4.__init__ = __init__; Steer4.start_level = start_level; Steer4.stats = stats
    if _goal_only0 is not None:
        def start_level_goal_only(self):
            _goal_only0(self)
            audit(self)
        Steer4.start_level_goal_only = start_level_goal_only
    return True
