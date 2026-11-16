"""plan1178.py -- NAS 1178 change 4: SEARCH IN PLAY OVER THE BRAIN'S OWN MODEL.

WHAT 1177 MEASURED (nas_1177_report.md sections 10 and 23d): the keys games die of the budget (77% of 1177's deaths; wa30 28 : 1, tu93
3 : 1) because their routes are long (tu93 level 1: 148 actions vs 19 for the human) -- and the pace rule did not shorten them. FrameLab
21: "the missing piece is search"; FrameLab 22: a best-first planner from each level's start reaches 4.15-6.48 per send against 1.87 for the
best one-step agent. The steering already keeps the level's state graph and looks two boards ahead (effect_mem 2).

WHAT 1178 DOES (gene plan78 0 / 1 / 2 / 3 = off / 30 / 60 / 150 boards per plan; pin MULTINXON_PIN_PLAN78):
  * A BOUNDED BEST-FIRST PLANNER (FL22's weighted A*: priority = depth + W x h, W = 2) over the brain's OWN learnt model: the boards the
    level's transitions showed (known_predict: 'exact' -- this board and action seen before, 'same', 'replay'), the M1 grammar for the
    rest (predict_board); the candidate actions are the current node's (keys and click points), an action that ended the game here is
    never planned, h is the steering's goal measure (the structured goal, else the goal map) -- so it plans on levels 2+ (a level's goal
    is predicted from the completed ones), where the steering knows what it wants.
  * WHEN: no plan running, the goal known, the level stalled (no drop of h for 3 steps), at most 3 plans a level, never inside a 1177
    macro, never over a proposal of the brain's that the model predicts to lower h; a plan longer than the budget the deaths revealed
    (death77's budget count) is not proposed, nor one the TRY'S OWN CAP cannot finish (game_loop's brain._v78_left: the plan's length + 2
    actions must remain -- an end start's try is often 10-16 actions).
  * THE PLANNER PROPOSES, THE STEP IS SWAPPED IN (the steering's _swap, as every steering rule): the plan's next action replaces the
    brain's unless the brain chose it or the model predicts the brain's own move to lower h (then the plan is dropped: the brain decides); after each step the board is checked against the plan's prediction -- a step whose board differs
    AND whose h is above the predicted h drops the plan (replanned later, with what the miss taught: the graph now holds the real move);
    a planned step that raised h is TAKEN BACK with ACTION7 where the game undoes (undo77's take-back; the planner never overrides it).
  * the bus's plan block (bus1178) reads the running plan: running, length, next is a click, predicted drop, confidence, near the goal.
Inert unless plan78 >= 1. TALLY: st_l8_plan (plans made), st_l8_none (searched, nothing below h), st_l8_step (steps swapped in),
st_l8_agree (the brain chose the planned step), st_l8_miss (dropped on a miss), st_l8_back (dropped for a take-back), st_l8_cap (a plan
the try's cap could not finish), st_l8_long (one the death budget could not), st_l8_done (plans run to the end), st_l8_exp (boards)."""
from __future__ import annotations

import heapq
import os

import numpy as np

BUDGET = {0: 0, 1: 30, 2: 60, 3: 150}
W_H = 2.0
MAX_DEPTH = 40
MAX_PLANS = 3
STALL = 3
MAX_ACTS = 24
CAP_RESERVE = 2                # a plan starts only if the try's cap leaves its length + 2 actions


def _tally(k, n=1):
    try:
        from arc_env import adapter as AD
        t = AD._V1160; t['st_' + k] = t.get('st_' + k, 0) + n
    except Exception:
        pass


def gene(params, name):
    e = os.environ.get('MULTINXON_PIN_' + name.upper(), '')
    try:
        if str(e).strip() != '':
            return int(float(e))
        v = getattr(params, name, None)
        return int(float(v)) if v not in (None, '') else 0
    except Exception:
        return 0


# ======================================================================================================== the planner
def _succ(st, B, O, akey, k):
    """(board, src) after akey on board B from the brain's own model, or (None, None)"""
    import effect1173 as E3
    import goal1169 as G9
    B2 = None; src = None
    try:
        if getattr(st, 'fl_exact', None) is not None or getattr(st, 'fl_sig', None) is not None:
            B2, src = E3.known_predict(st, B, O, akey, k)
        else:
            sig = st.sig_of(akey, O)
            if st.sig_n.get(sig, 0) and st.sig_chg.get(sig, 0) == 0:
                B2, src = B, 'same'
    except Exception:
        B2 = None
    if B2 is None:
        try:
            B2 = G9.predict_board(st, B, O, akey); src = 'm1' if B2 is not None else None
        except Exception:
            B2 = None
    return B2, src


def search(st, g, nd, budget):
    """the plan from board g: [(akey, cls, predicted key, predicted h)], h0, h_end, conf -- or None"""
    import effect1173 as E3
    from steer1162 import components
    h0 = E3.goal_h(st, g)
    if h0 is None or h0 <= 0:
        return None
    acts = []
    for a, cl in nd['cands']:
        if st.dead(cl) or st.deadly(cl):
            continue
        t = nd['tried'].get(a)
        if t is not None and t[2]:
            continue                                                       # it ended the game here before
        acts.append((a, cl))
    keys = [x for x in acts if x[0][0] == 'K']; clicks = [x for x in acts if x[0][0] != 'K']
    acts = (keys + clicks)[:MAX_ACTS]
    if not acts:
        return None
    k0 = st.key(g)
    boards = {k0: np.asarray(g)}; depth = {k0: 0}; parent = {k0: None}; hh = {k0: h0}; conf = {k0: 1.0}
    heap = [(W_H * h0, 0, 0, k0)]; tie = 1; nexp = 0
    best = (h0, 0, k0)
    ign = st.ignore()
    while heap and nexp < budget:
        f, d, _, k = heapq.heappop(heap)
        if d > depth.get(k, 1 << 30):
            continue
        B = boards[k]
        try:
            O = st.objs.get(k) if k == k0 else None
            if O is None:
                O = components(B, st.bg, ign)
        except Exception:
            continue
        nexp += 1
        done = False
        for a, cl in acts:
            B2, src = _succ(st, B, O, a, k)
            if B2 is None or B2 is B or src == 'same':
                continue
            if np.array_equal(B2, B):
                continue
            try:
                k2 = st.key(B2)
            except Exception:
                continue
            if k2 in depth and depth[k2] <= d + 1:
                continue
            h2 = E3.goal_h(st, B2)
            if h2 is None:
                continue
            depth[k2] = d + 1; parent[k2] = (k, a, cl); boards[k2] = B2; hh[k2] = h2
            conf[k2] = conf[k] * (1.0 if src in ('exact', 'same') else (0.8 if src == 'replay' else 0.6))
            if (h2, d + 1) < (best[0], best[1]):
                best = (h2, d + 1, k2)
            if h2 <= 0:
                done = True
                break
            if d + 1 < MAX_DEPTH:
                heapq.heappush(heap, (d + 1 + W_H * h2, d + 1, tie, k2)); tie += 1
        if done:
            break
    _tally('l8_exp', nexp)
    if best[0] >= h0:
        return None
    path = []; k = best[2]
    while parent.get(k) is not None:
        kp, a, cl = parent[k]
        path.append((a, cl, k, hh[k]))
        k = kp
    path.reverse()
    return {'plan': path, 'h0': h0, 'h_end': best[0], 'conf': conf.get(best[2], 0.0), 'exp': nexp}


def _brain_drops(st, g, akey, nd):
    """the brain's own proposal is predicted (by the same model) to lower h"""
    try:
        import effect1173 as E3
        h0 = E3.goal_h(st, g)
        if h0 is None:
            return False
        O = st.objs.get(st.key(g)) or []
        B, src = _succ(st, np.asarray(g), O, akey, st.key(g))
        if B is None or src == 'same':
            return False
        h1 = E3.goal_h(st, B)
        return h1 is not None and h1 < h0
    except Exception:
        return False


# ======================================================================================================== the steering
def install(Steer4, Steer5, ArcBrain):
    if getattr(Steer4, '_v1178_plan', False):
        return True
    Steer4._v1178_plan = True
    _init0 = Steer4.__init__
    _start0 = Steer4.start_level

    def __init__(self, params, seed=0):
        _init0(self, params, seed=seed)
        self.plan78 = max(0, min(3, gene(params, 'plan78')))
        self._pl78 = None; self._pl78_n = 0; self._pl78_in = False; self._pl78_t = 0; self._pl78_hmin = None; self._pl78_stall = 0

    def start_level(self, g):
        _start0(self, g)
        self._pl78 = None; self._pl78_n = 0; self._pl78_t = 0; self._pl78_hmin = None; self._pl78_stall = 0

    def _drop(self, why):
        self._pl78 = None
        _tally(why)

    def decide_outer(self, orig, brain, frame, a, d):
        if self._pl78_in or not getattr(self, 'plan78', 0):
            return orig(self, brain, frame, a, d)
        self._pl78_in = True
        try:
            out = orig(self, brain, frame, a, d)
        finally:
            self._pl78_in = False
        try:
            if not self._cur or self.prev is None or getattr(self, 'goal', None) is None and not getattr(self, 'sg_pred', None):
                return out
            k, akey, cls = self.prev[0], self.prev[1], self.prev[2]
            if akey == ('K', 7) and not getattr(self, '_from_brain', True):
                # the steering TAKES BACK the last step (undo77's take-back of a step that raised h, the prior's probe take-back): a
                # failed planned step is undone where the game undoes; the plan is void (replanned from the board the undo restores)
                if self._pl78 is not None:
                    _drop(self, 'l8_back')
                return out
            nd = self.graph.get(k)
            if nd is None:
                return out
            g = self._cur[0]
            self._pl78_t += 1
            import effect1173 as E3
            h_now = E3.goal_h(self, g)                                     # the planner's own stall clock (steering v5's since_drop
            if h_now is not None:                                          # exists only on a v5 steering)
                if self._pl78_hmin is None or h_now < self._pl78_hmin:
                    self._pl78_hmin = h_now; self._pl78_stall = 0
                else:
                    self._pl78_stall += 1
            P = self._pl78
            if P is not None:                                              # check the board the last planned step reached
                exp_k, exp_h = P.get('expect') or (None, None)
                if exp_k is not None and exp_k != k:
                    h = h_now
                    if h is None or exp_h is None or h > exp_h:
                        _drop(self, 'l8_miss'); P = None
                if P is not None and P['i'] >= len(P['plan']):
                    _drop(self, 'l8_done'); P = None
            if P is None:
                if self._pl78_n >= MAX_PLANS or getattr(self, '_e77_mac', None) is not None:
                    return out
                sd = max(int(getattr(self, 'since_drop', 0) or 0), int(self._pl78_stall))
                if sd < STALL:
                    return out                                             # the brain is getting somewhere: no plan
                left = getattr(brain, '_v78_left', None)
                if left is not None and int(left) < 2 + CAP_RESERVE:
                    return out                                             # no room left for any plan
                if getattr(self, '_from_brain', False) and _brain_drops(self, g, akey, nd):
                    return out                                             # the brain's own move is predicted to lower h: it stands
                self._pl78_n += 1
                res = search(self, g, nd, BUDGET.get(self.plan78, 0))
                if res is None:
                    _tally('l8_none')
                    return out
                bn = getattr(self, '_e77_budget_n', None)
                if bn and int(getattr(self, '_e77_steps', 0) or 0) + len(res['plan']) > int(bn):
                    _tally('l8_long')
                    return out
                left = getattr(brain, '_v78_left', None)                   # the try's own action cap (game_loop): a plan the try
                if left is not None and len(res['plan']) + CAP_RESERVE > int(left):   # cannot finish is never started
                    _tally('l8_cap')
                    return out
                P = {'plan': res['plan'], 'i': 0, 'h0': res['h0'], 'h_end': res['h_end'], 'conf': res['conf'], 'expect': None,
                     'dh': res['h_end'] - res['h0']}
                self._pl78 = P
                _tally('l8_plan')
            want, wcls, wk, wh = P['plan'][P['i']]
            P['i'] += 1
            P['expect'] = (wk, wh)
            if want == akey:
                _tally('l8_agree')
                return out
            if getattr(self, '_from_brain', False) and _brain_drops(self, g, akey, nd):
                _drop(self, 'l8_brain')                                    # the planner proposes; the brain's predicted drop decides
                return out
            if want not in [x[0] for x in nd['cands']]:
                _drop(self, 'l8_gone')
                return out
            _tally('l8_step')
            return self._swap(want, wcls, 'l8_swap')
        except Exception as e:
            self._v1178l_error = f'plan: {type(e).__name__}: {e}'[:160]
            self._pl78 = None
        return out

    Steer4.__init__ = __init__; Steer4.start_level = start_level
    _dec4 = Steer4.decide

    def decide4(self, brain, frame, a, d):
        return decide_outer(self, _dec4, brain, frame, a, d)
    Steer4.decide = decide4
    if Steer5 is not None and not getattr(Steer5, '_v1178_plan', False) and Steer5.decide is not decide4:
        Steer5._v1178_plan = True
        _dec5 = Steer5.decide

        def decide5(self, brain, frame, a, d):
            return decide_outer(self, _dec5, brain, frame, a, d)
        Steer5.decide = decide5
    return True
