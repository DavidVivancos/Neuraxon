"""effect1173.py -- NAS 1173: THREE GENES OF UNDERSTANDING IN THE STEERING (nas_1172_report.md §23 change 9 + the user's 21 Oct 2026
decisions: prefix-watching "as long as we are not cheating", and "detect a game just with the board or a couple of exploration
actions ... the pixels that made it and the feedback when actions are performed").

Every brain still proposes every action; nothing here knows a game's solution. All three genes are OFF in the seeds (an absent gene
plays exactly as 1172), all three are bandit arms (evo1173), and each has an environment pin for A/B tests.

effect_mem 0 / 1 / 2  (MULTINXON_PIN_EFFECT_MEM) -- KNOW WHAT A MOVE DOES BEFORE MAKING IT
  The steering's table of what each action did on each screen it has seen (the exact (board, action) -> change memo and the
  signature replay of goal1170, the 'known' model of FrameLab 11/17) is recorded whatever steer_goal_fl says, and:
  1  KNOWN LOOP: a proposal the play has already made on this very board and that led to a board already visited in this level
     attempt (a cycle -- not a no-op: the 1160 memory already vetoes those) is replaced by the best UNTRIED candidate of this board
     (the steering's own order); when every candidate has been tried, the proposal stands. Only while the game looks Markov.
     FL15 is respected: a silent move is never vetoed here (silent moves are on the verified paths: 28% of the moves 9-34 from the
     end, 42% in the keys games).
  2  1 + THE TWO-STEP LOOK (FL16: a two-step goal crosses the silent moves -- M5 48 -> 81%, M8 34 -> 57% with a perfect model;
     FL17: with a model of SEEN effects d2 beats d1 6 / 0): when the steering has a goal (the structured goal of goal1167/1170 or
     the 1163 goal map) and the proposal is not predicted to lower h within two actions, the candidate whose KNOWN predicted boards
     reach the lowest h within two actions (a drop below the current h) replaces it; ties: the lower one-step h, then the
     steering's order. Only seen effects predict (exact / no change / replay of a seen signature): an unknown proposal is a probe
     and stands, an unknown candidate is never chosen (FL11's known-probe rule). Budget per step: EM_MAX_H boards scored, the
     second step looked past for EM_BEAM first-step boards over EM_SECOND actions.

warm_prefix 0 / 1 / 2  (MULTINXON_PIN_WARM) -- TEACHING BY THE VERIFIED PREFIX, WITH UNDERSTANDING
  An end-first start places the brain k moves before the end of level L after the verified prefix is replayed without it (1158).
  A brain that had made those moves itself would remember what they did. FL17 measured exactly that memory (the prefix's effects
  kept): frames-only 14 / 2 (p 0.004), the known-signature model 26 / 3 (p 2e-5) -- the strongest frames-only gain FrameLab found.
  1  the steering's memories learn the prefix's moves of THIS level (the last WARM_MAX): the graph of boards and tried actions, the
     action-class effect statistics, M1's signature effects, the exact memo and the signature replays -- the effects, never the
     sequence: the brain still chooses every move of its own play.
  2  1 + the brain WATCHES the last WATCH_N of those moves: its network steps on each recorded frame (its plasticity runs) while
     the expert's action is the executed one in its books (the 1160 override rule), then it plays.
  Inert at 0. The prefix itself is unchanged (the same moves, the same start).

game_sense 0 / 1  (MULTINXON_PIN_GAME_SENSE) -- WHICH GAME IS THIS?
  sense1173.py: at a level's first frame the steering finds the nearest known game from the board's pixels (colours, objects,
  layout); after its first two actions in the level (the expert's, when warm-fed) it decides with their feedback (did the board
  change, which colours changed, where, how much). When the game it senses is not the brain's home game (MULTINXON_HOME_GAME or
  the arch's home_game; unknown = no home) it plays with that game's priors (game_priors_1173.json: the book holders' candidate
  tier, key sweep and click band); at home it keeps its own genes. MULTINXON_SENSE_TRUTH (set by the play loop to the game being
  played) is used ONLY to count sense_hit / sense_miss: the measure of the NAS's new goal.

Tallies (st_*): em_loop, em_look, em_look_none, em_unknown, em_budget, warm_fed, warm_watch, sense_board, sense_fx, sense_hit,
sense_miss, sense_home, sense_prior."""
from __future__ import annotations

import os

import numpy as np

EM_MAX_H = 96            # boards scored by h per step (both steps)
EM_BEAM = 6              # first-step boards looked past
EM_SECOND = 24           # second-step actions per looked-past board
WARM_MAX = 60            # prefix moves of this level the steering's memories learn (the most recent)
WATCH_N = 8              # warm_prefix 2: the last moves the brain's network watches
SENSE_PROBES = 2         # feedback observations before the sense's final decision (the board alone decides provisionally)


def _gene(params, name, default):
    v = getattr(params, name, None)
    return default if v in (None, '') else v


def _pin(env, params, gene, hi):
    e = os.environ.get(env, '')
    try:
        v = int(float(e)) if e.strip() != '' else int(float(_gene(params, gene, 0) or 0))
    except Exception:
        v = 0
    return max(0, min(hi, v))


def em_of(params):
    return _pin('MULTINXON_PIN_EFFECT_MEM', params, 'effect_mem', 2)


def warm_of(params):
    return _pin('MULTINXON_PIN_WARM', params, 'warm_prefix', 2)


def sense_of(params):
    return _pin('MULTINXON_PIN_GAME_SENSE', params, 'game_sense', 1)


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def akey_of(a_str):
    """'ACTIONn' -> ('K', n); 'ACTION6:x,y' -> ('C', y, x); RESET / unknown -> None"""
    s = str(a_str)
    if s.startswith('ACTION6:'):
        x, y = map(int, s.split(':')[1].split(','))
        return ('C', int(y), int(x))
    if s.startswith('ACTION'):
        try:
            v = int(s[6:])
            return ('K', v) if 1 <= v <= 7 and v != 6 else None
        except Exception:
            return None
    return None


def _ga(akey):
    from arc_env.structs import GameAction
    if akey[0] == 'K':
        return GameAction(int(akey[1])), {}
    return GameAction.ACTION6, {'x': int(akey[2]), 'y': int(akey[1])}


# ================================================================================================ the known model (no M1)
def known_predict(st, g, O, akey, k=None):
    """(board, src) from SEEN effects only: 'exact' (this board + action executed before), 'same' (a signature seen to change
    nothing), 'replay' (a seen signature's change replayed where it still fits) -- or (None, None)"""
    import goal1170 as G70
    if k is None:
        try:
            k = st.key(g)
        except Exception:
            k = None
    d = st.fl_exact.get((k, akey)) if (k is not None and getattr(st, 'fl_exact', None) is not None) else None
    if d is not None:
        if isinstance(d, str):
            return g, 'same'
        B = np.array(g, copy=True); idx, vals = d
        B.ravel()[idx] = vals
        return B, 'exact'
    sig = st.sig_of(akey, O)
    if st.sig_n.get(sig, 0) and st.sig_chg.get(sig, 0) == 0:
        return g, 'same'
    rp = (getattr(st, 'fl_sig', None) or {}).get(sig)
    if rp is not None:
        B = G70._replay(st, g, O, akey, rp)
        if B is not None:
            return B, 'replay'
    return None, None


def goal_h(st, B):
    """the steering's h on board B: the structured goal's unmet predicates, else the 1163 goal map's cells not yet arrived"""
    try:
        if getattr(st, 'sg_pred', None) and st.start_board is not None:
            import goal1167 as G7
            return G7.h_of(st.sg_pred, B, st.start_board, st.ignore())
        if getattr(st, 'goal', None) is not None and st.start_board is not None:
            import goalmap1163 as GM
            return GM.h_of(B, st.start_board, st.goal)
    except Exception:
        return None
    return None


# ================================================================================================ the warm prefix
def _save(st):
    keep = ('prev', 'pred', '_wake', '_wake_skip', 'last_changed', 'last_trans', 'last_move', 'last_key', 'selected', 'last_align',
            'last_new', 'last_miss', '_cur', '_from_brain', 'dh_last', 'pre')
    return {a: getattr(st, a) for a in keep if hasattr(st, a)}


def _restore(st, saved):
    for a, v in saved.items():
        try:
            setattr(st, a, v)
        except Exception:
            pass


def _feed_steer(brain, st, items):
    """the steering's memories learn each (obs before, action, board before, board after) of the prefix -- as in its own play"""
    n = 0
    saved = _save(st)
    try:
        st._from_brain = False; st._wake = False; st._wake_skip = False
        for ob, a_str, g0, g1 in items:
            ak = akey_of(a_str)
            if ak is None or g0 is None or g1 is None:
                continue
            g0 = np.asarray(g0, dtype=np.int16); g1 = np.asarray(g1, dtype=np.int16)
            if g0.shape != g1.shape:
                continue
            try:
                avail = [int(getattr(x, 'value', x)) for x in brain._masked_avail(ob)]
            except Exception:
                avail = [int(getattr(x, 'value', x)) for x in (getattr(ob, 'available_actions', None) or [])]
            k0, nd = st.node(g0, avail)
            if ak[0] == 'K':
                cls = ('K', ak[1])
            else:
                a, d = _ga(ak)
                _, cls = st.brain_key(g0, a, d, nd)
            ch = st.observe(k0, ak, cls, g0, g1, False, False, frames=None)
            st._from_brain = False; st._wake = False; st._wake_skip = False
            st._after_action(ak, g0, g1, ch, False)
            if getattr(st, 'em', 0) >= 1:
                try:
                    vis = st._em_vis(); vis.add(st.key(g0)); vis.add(st.key(g1))
                except Exception:
                    pass
            n += 1
    finally:
        _restore(st, saved)
    return n


def _follow_brain(brain, ob, a, d):
    """the executed (expert) action replaces the network's proposal in the brain's books (frontier1167._follow's brain part)"""
    try:
        v = int(getattr(a, 'value', a))
    except Exception:
        v = 0
    brain.last_action = a
    try:
        brain._eff_prev_idx = v - 1 if 1 <= v <= 6 else -1
        if v == 6 and d:
            brain._last_click_cand = (int(d.get('y', 0)), int(d.get('x', 0)))
    except Exception:
        pass
    if getattr(brain, '_cap_on', False):
        try:
            fr = getattr(ob, 'frame', None) or []
            g = np.asarray(fr[-1]) if fr else None
            lv = int(getattr(ob, 'levels_completed', 0) or 0)
            brain._cap_prev = (g.tolist() if g is not None else None, v, lv) if v > 0 else None
        except Exception:
            pass


def _watch(brain, items):
    """warm_prefix 2: the brain's own network steps on each recorded frame (the unwrapped ArcBrain step: no steering, no sweep, no
    frontier), the expert's action is the executed one in its books; the episode memory starts at the first watched frame"""
    import arc_brain as AB
    base = getattr(AB, '_v1160_play_step', None); sreset = getattr(AB, '_v1160_soft_reset', None)
    if base is None or not items:
        return 0
    n = 0
    try:
        if sreset is not None:
            sreset(brain, items[0][0])
    except Exception:
        pass
    for ob, a_str, g0, g1 in items:
        ak = akey_of(a_str)
        if ak is None:
            continue
        try:
            base(brain, ob)
        except Exception as e:
            brain._v1173_error = f'watch: {type(e).__name__}: {e}'[:160]
            break
        a, d = _ga(ak)
        _follow_brain(brain, ob, a, d)
        n += 1
    return n


def feed_warm(brain, prior):
    """called once per play after the steering's prior levels (game_loop._v1163_feed_prior); inert unless warm_prefix >= 1"""
    st = getattr(brain, '_steer', None)
    mode = getattr(st, 'warm', None) if st is not None else None
    if mode is None:
        mode = warm_of(getattr(brain, 'params', None))
    if mode <= 0 or getattr(brain, '_v1173_warm_done', False):
        return 0
    brain._v1173_warm_done = True
    items = [x for x in (prior or {}).get('warm') or [] if x is not None]
    if not items:
        return 0
    items = items[-int(getattr(st, '_warm_max', WARM_MAX) or WARM_MAX):]                  # v1.177 change 9: 120 / 16 on a 1177 brain
    watch = items[-int(getattr(st, '_watch_n', WATCH_N) or WATCH_N):] if mode >= 2 else []
    learn = items[:-len(watch)] if watch else items
    n = 0
    if st is not None and learn:
        try:
            n = _feed_steer(brain, st, learn)
        except Exception as e:
            brain._v1173_error = f'warm: {type(e).__name__}: {e}'[:160]
    if n:
        _tally('warm_fed', n)
    if watch:
        w = _watch(brain, watch)
        if st is not None and w:
            try:                                                   # the watched moves join the steering's memories too
                _feed_steer(brain, st, watch[:w])
            except Exception:
                pass
        if w:
            _tally('warm_watch', w)
        n += w
    return n


# ================================================================================================ the steering hooks
def install(Steer4):
    """idempotent; after goal1170: effect_mem / warm_prefix / game_sense (inert at 0)"""
    if getattr(Steer4, '_v1173_effect', False):
        return True
    Steer4._v1173_effect = True
    import sense1173 as S
    _init0 = Steer4.__init__; _after0 = Steer4._after_action; _start0 = Steer4.start_level; _reset0 = Steer4.on_reset
    _obsf0 = Steer4.observe_frame; _dec0 = Steer4.decide; _stats0 = Steer4.stats

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        self.em = em_of(params); self.warm = warm_of(params); self.sense_on = sense_of(params) == 1
        self.em_visited = set(); self._em_gid = id(self.graph); self._em_cache = {}; self._em_nh = 0
        self.n_em = {'loop': 0, 'look': 0}
        hg = getattr(params, 'home_game', None)
        self.home_game = (os.environ.get('MULTINXON_HOME_GAME', '') or (str(hg) if hg not in (None, '') else '')).strip() or None
        self.sense_game = None; self.sense_margin = None; self.sense_obs = []; self.sense_final = False; self._sense_pending = None
        self.sense_applied = None; self._sense_own = None

    # -------------------------------------------------------------------- visited boards (this level attempt)
    def _em_vis(self):
        if self._em_gid != id(self.graph):                     # a fresh graph (HUD / clock re-key): the keys changed
            self.em_visited = set(); self._em_gid = id(self.graph)
        return self.em_visited

    def start_level(self, g):
        _start0(self, g)
        self.em_visited = set(); self._em_gid = id(self.graph)
        if self.sense_on and g is not None:
            try:
                idx = S.index_cached()
                if idx.entries:
                    game, margin, _ = idx.classify(S.board_desc(g))
                    self.sense_obs = []; self.sense_final = False
                    self.sense_margin = margin
                    _tally('sense_board')
                    if game is not None:
                        _sense_set(self, game, final=False)        # provisional: the board alone (93.8% held out)
            except Exception as e:
                self._v1173_error = f'sense: {type(e).__name__}: {e}'[:160]

    def on_reset(self, frame):
        _reset0(self, frame)
        self.em_visited = set(); self._em_gid = id(self.graph)

    def observe_frame(self, frame):
        _obsf0(self, frame)
        if self.em >= 1 and self._cur:
            try:
                self._em_vis().add(self.key(self._cur[0]))
            except Exception:
                pass

    def _after_action(self, ak, g0, g1, ch, trans):
        _after0(self, ak, g0, g1, ch, trans)
        if self.em >= 1 and not getattr(self, 'fl_on', False):
            try:
                import goal1170 as G70
                G70.record(self, ak, g0, g1)                       # the effect table is kept whatever steer_goal_fl says
            except Exception as e:
                self._v1173_error = f'record: {type(e).__name__}: {e}'[:160]
        if self.sense_on and not self.sense_final and ak is not None and g0 is not None and g1 is not None \
                and len(self.sense_obs) < SENSE_PROBES:
            try:
                kind = 'C' if ak[0] == 'C' else f'K{int(ak[1])}'
                same = g0.shape == g1.shape
                changed = bool(ch or trans or not same)
                self.sense_obs.append((kind, changed, [float(x) for x in S.effect_desc(g0, g1)] if changed else None))
                if len(self.sense_obs) >= SENSE_PROBES:
                    idx = S.index_cached()
                    board = self.start_board if self.start_board is not None else g0
                    game, margin, _ = idx.classify(S.board_desc(board), obs=self.sense_obs)
                    self.sense_margin = margin
                    _tally('sense_fx')
                    if game is not None:
                        _sense_set(self, game, final=True)         # the board + the feedback (97.1% held out, generic actions)
            except Exception as e:
                self._v1173_error = f'sense fx: {type(e).__name__}: {e}'[:160]

    # -------------------------------------------------------------------- the choice
    def decide(self, brain, frame, a, d):
        out = _dec0(self, brain, frame, a, d)
        if self._sense_pending is not None:
            _sense_apply(self, brain)
        if self.em <= 0 or not self._cur or self.prev is None or not getattr(self, '_from_brain', False):
            return out
        try:
            k, bkey, bcls, g, lv = self.prev
            nd = self.graph.get(k)
            if nd is None or bkey is None:
                return out
            O = self.objs.get(k)
            if O is None:
                O = self.objects_of(k, g)
            self._em_cache = {}; self._em_nh = 0
            if self.em >= 2:
                ch = _look(self, g, O, nd, bkey, k)
                if ch is not None:
                    self.n_em['look'] += 1
                    return _take(self, ch, g, lv, k, 'em_look')
            if self.markov:
                t = nd['tried'].get(bkey)
                if t is not None and t[1] and not t[2] and t[0] is not None and t[0] in self._em_vis():
                    alt, _cv = self.best3(nd, bkey, O)
                    if alt is not None and alt[0] != bkey:
                        self.n_em['loop'] += 1
                        return _take(self, alt, g, lv, k, 'em_loop')
        except Exception as e:
            self._v1173_error = f'decide: {type(e).__name__}: {e}'[:160]
        return out

    def stats(self):
        s = _stats0(self)
        s.update(effect_mem=self.em, warm_prefix=self.warm, game_sense=int(self.sense_on), em_loop=self.n_em['loop'],
                 em_look=self.n_em['look'], sense_game=self.sense_game, sense_applied=self.sense_applied,
                 v1173_error=getattr(self, '_v1173_error', None))
        return s

    Steer4.__init__ = __init__; Steer4._after_action = _after_action; Steer4.start_level = start_level
    Steer4.on_reset = on_reset; Steer4.observe_frame = observe_frame; Steer4.decide = decide; Steer4.stats = stats
    Steer4._em_vis = _em_vis
    return True


def _take(st, choice, g, lv, k, why):
    akey, cls = choice
    st.prev = (k, akey, cls, g, lv); st.pred = None; st._from_brain = False
    _tally(why)
    return st.to_action(akey)


def _pred(st, g, O, akey, k):
    key = (k, akey)
    if key in st._em_cache:
        return st._em_cache[key]
    B, src = known_predict(st, g, O, akey, k)
    st._em_cache[key] = B
    return B


def _h(st, B):
    if st._em_nh >= int(getattr(st, '_em_max_h', EM_MAX_H) or EM_MAX_H):
        _tally('em_budget')
        return None
    st._em_nh += 1
    return goal_h(st, B)


def _second(st, B1, nd):
    """the lowest h one KNOWN action past B1 (the current node's actions; clicks at their points)"""
    from steer1162 import components
    try:
        O1 = components(B1, st.bg, st.ignore()); k1 = st.key(B1)
    except Exception:
        return None
    best = None; n = 0
    keys = [x for x in nd['cands'] if x[0][0] == 'K']
    clicks = [x for x in nd['cands'] if x[0][0] != 'K']
    for a2, cl in (keys + clicks):
        if n >= int(getattr(st, '_em_second', EM_SECOND) or EM_SECOND):
            break
        if st.dead(cl):
            continue
        B2 = _pred(st, B1, O1, a2, k1)
        if B2 is None:
            continue
        n += 1
        h2 = _h(st, B2)
        if h2 is not None and (best is None or h2 < best):
            best = h2
    return best


def _look(st, g, O, nd, bkey, k):
    """the two-step look (effect_mem 2): None when the proposal stands"""
    h0 = goal_h(st, g)
    if h0 is None or h0 <= 0:
        return None
    pb = _pred(st, g, O, bkey, k)
    if pb is None:
        _tally('em_unknown')
        return None                                                # an unknown proposal is a probe: it stands (FL11)
    hb = _h(st, pb)
    if hb is not None and hb < h0:
        return None                                                # predicted to lower h: it stands
    if hb is not None and pb is not g and not np.array_equal(pb, g):
        v = _second(st, pb, nd)
        if v is not None and v < h0:
            return None                                            # the proposal reaches a drop within two actions: it stands
    first = []
    for a2, cl in nd['cands']:
        if a2 == bkey or st.dead(cl):
            continue
        t = nd['tried'].get(a2)
        if t is not None and t[2]:
            continue                                               # it ended the game here before
        B1 = _pred(st, g, O, a2, k)
        if B1 is None:
            continue
        h1 = _h(st, B1)
        if h1 is None:
            continue
        first.append((h1, a2, cl, B1))
    best = None; bv = None
    for h1, a2, cl, B1 in first:
        if h1 < h0 and (bv is None or (h1, h1) < bv):
            best, bv = (a2, cl), (h1, h1)
    if best is not None:
        return best
    # nobody drops in one: look one action past the changed boards with the lowest h
    beam = sorted([x for x in first if x[3] is not g and not np.array_equal(x[3], g)], key=lambda x: x[0])[:int(getattr(st, '_em_beam', EM_BEAM) or EM_BEAM)]
    for h1, a2, cl, B1 in beam:
        v = _second(st, B1, nd)
        if v is not None and v < h0 and (bv is None or (v, h1) < bv):
            best, bv = (a2, cl), (v, h1)
    if best is None:
        _tally('em_look_none')
    return best


# ================================================================================================ the game sense
def _sense_set(st, game, final=True):
    """the sensed game (provisional from the board, final with the feedback): its priors are applied at the next decision unless
    it is the brain's home game or already applied"""
    import sense1173 as S
    prev = st.sense_game
    st.sense_game = game; st.sense_final = bool(final)
    truth = os.environ.get('MULTINXON_SENSE_TRUTH', '').strip()
    if truth:
        _tally(('sense_hit' if truth == game else 'sense_miss') if final else ('sense_bhit' if truth == game else 'sense_bmiss'))
    if st.home_game and game == st.home_game:
        if final or prev is None:
            _tally('sense_home')
        st._sense_pending = None
        if st.sense_applied and st.sense_applied != game:
            st._sense_pending = dict(st._sense_own, game=game) if getattr(st, '_sense_own', None) else None
        return
    if st.sense_applied == game:
        return
    pri = S.priors_cached().get(game)
    st._sense_pending = dict(pri, game=game) if pri else None


def _sense_apply(st, brain):
    """the sensed game's priors replace the brain's candidate tier, key sweep and click band for this level"""
    pri = st._sense_pending; st._sense_pending = None
    if not pri or brain is None:
        return
    if getattr(st, '_sense_own', None) is None:
        cs = getattr(brain, '_v1172_cs', None)
        st._sense_own = {'cand_tier': int(getattr(brain, '_v1172_tier', 0) or 0), 'commit_sweep': 1 if cs is not None else 0,
                         'commit_stall': int((cs or {}).get('stall_at', 0) or 0), 'click_lo': int(getattr(brain, '_cm_lo', 1) or 1),
                         'click_hi': int(getattr(brain, '_cm_hi', 8) or 8)}
    try:
        if 'cand_tier' in pri:
            t = max(0, min(2, int(pri['cand_tier'])))
            brain._v1172_tier = t; brain._v1172_cc = None; st._cand_tier = t
        if 'commit_sweep' in pri:
            if int(pri['commit_sweep']) == 1:
                if getattr(brain, '_v1172_cs', None) is None:
                    stall = int(pri.get('commit_stall', 0) or 0) or int(os.environ.get('MULTINXON_COMMIT_STALL', '8') or 8)
                    brain._v1172_cs = {'lv': None, 'pressed': set(), 'seen': set(), 'stall': 0, 'stall_at': max(1, stall), 'n': 0}
            else:
                brain._v1172_cs = None
        if 'click_lo' in pri and 'click_hi' in pri:
            brain._cm_lo = max(1, int(pri['click_lo'])); brain._cm_hi = max(brain._cm_lo, int(pri['click_hi'])); brain._cm_sig = None
        st.sense_applied = pri.get('game')
        _tally('sense_prior')
    except Exception as e:
        st._v1173_error = f'sense apply: {type(e).__name__}: {e}'[:160]
