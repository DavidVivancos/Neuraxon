"""steer1160.py -- NAS 1160: the STEERING LAYER on top of the Neuraxon brain (the 1158 learnings, embedded).

THE BRAIN PLAYS. (v1.160b: + hidden-state safety and transient effects, from the 1158 status 2.) Every step the brain proposes its action (its network, fovea, candidates, exploration, memories --
everything NAS evolves). The steering layer sees the same frame and the brain's proposal and may correct it:

  1. VETO WASTE (Blind Squirrel; 1159: lp85-lv6 98% no-change actions, su15 89% repeated states). If the brain's action
     was already tried in THIS exact state (hash of the board without the HUD) and changed nothing, or killed, it is
     replaced by the best untried action of this state. Re-doing a known no-op can never help.
  2. NAVIGATE when the state is exhausted: BFS over the observed state graph to the nearest state with untried actions.
  3. TRUST (gene steer_trust, 0..1): with that probability the steering's best untried action replaces the brain's
     when it scores higher. 0 = the brain leads, the steering only vetoes; 1 = the steering leads, the brain is a prior.
  4. What the steering scores (1158 / the transcripts, all online, no game-specific code):
       * effects by PROBING: per action class (a key; a click on an object of a (colour, size)), how often it changed the
         board, killed, led to a level-up (StochasticGoose, counted);
       * the GOAL from FINISHED boards: on a level-up the response's first frame is the finished board (the stale-frame
         discovery); the colour change start -> finished is a signature that later levels are steered along;
       * PATH memory: the action classes that cleared earlier levels are preferred (goals transfer within a game);
       * the brain's own intent: candidates near the brain's click (or the same key) get a bonus (steer_brain_w).
  The memory lives as long as the brain does: a newborn per seed (first try) starts empty, exactly as on Kaggle.

GENES (in the arch's neural dict, so NAS evolves them; absent = off / default):
  steer            0/1   the layer exists                      steer_trust     0..1  P(steering's best replaces the brain's)
  steer_no_memory  0/1   1 = no veto                           steer_no_nav    0/1   1 = no navigation
  steer_no_goal    0/1   1 = no goal signature                 steer_no_path   0/1   1 = no path memory
  steer_no_effect  0/1   1 = no effect model                   steer_max_clicks      click candidates per state (48)
  steer_hud              HUD change-rate threshold (0.35)      steer_brain_w         weight of the brain's intent (1.0)
The environment variable MULTINXON_STEER_FORCE=1 turns the layer on for every brain (newborn architecture search)."""
from __future__ import annotations

import os
import random

import numpy as np

from agent1160 import DEFAULT, Agent, first_frame, objects, settled, size_bucket


def _gene(params, name, default):
    v = getattr(params, name, None)
    return default if v in (None, '') else v


REGISTRY = []                        # the last steering layers built in this process (tests / diagnostics)


class Steer(Agent):
    def __init__(self, params, seed=0):
        cfg = dict(DEFAULT)
        cfg.update(memory=0 if int(_gene(params, 'steer_no_memory', 0) or 0) else 1,
                   effect=0 if int(_gene(params, 'steer_no_effect', 0) or 0) else 1,
                   nav=0 if int(_gene(params, 'steer_no_nav', 0) or 0) else 1,
                   goal=0 if int(_gene(params, 'steer_no_goal', 0) or 0) else 1,
                   max_clicks=int(_gene(params, 'steer_max_clicks', 48) or 48),
                   hud_rate=float(_gene(params, 'steer_hud', 0.35) or 0.35),
                   eps=0.0, reset=0)
        if int(_gene(params, 'steer_no_path', 0) or 0):
            cfg['w_path'] = 0.0
        super().__init__(cfg, random.Random((int(seed) * 7919 + 1160) & 0x7FFFFFFF))
        self.trust = float(_gene(params, 'steer_trust', 0.0) or 0.0)
        self.w_brain = float(_gene(params, 'steer_brain_w', 1.0) or 1.0)
        self.prev = None; self.level = None; self.died = False
        self.n_steps = 0; self.n_veto = 0; self.n_trust = 0; self.n_nav = 0
        self.markov = True; self.obs = {}; self.n_transient = 0; self.n_inconsistent = 0   # v1.160b (1158 status 2)
        REGISTRY.append(self); del REGISTRY[:-8]

    # -------------------------------------------------------------------------------------------- lifecycle
    def on_death(self):
        self.died = True

    def on_reset(self, frame):
        g = settled(frame, None)
        if self.prev is not None and self.died:
            k0, ak, cl, g0, lv0 = self.prev
            self.observe(k0, ak, cl, g0, g if g is not None else g0, True, False)
        self.prev = None; self.died = False
        lv = int(getattr(frame, 'levels_completed', 0) or 0)
        if g is not None and (self.level is None or lv != self.level):
            self.start_level(g); self.level = lv

    # -------------------------------------------------------------------------------------------- learning (1158 status 2)
    def observe(self, k0, akey, cls, g0, g1, died, up, frames=None):
        """Agent.observe + two corrections from the 1158 research:
        * HIDDEN STATE (g50t: ACTION5 replays the moves recorded since the last ACTION5, so one board can be several
          states): when the same board + action gives a different outcome, the board is not the whole state -> from then
          on the veto needs >= 2 no-change tries of that action in that state (deaths are always vetoed);
        * TRANSIENT EFFECTS (sp80's pour, tn36's run button: the response animates, then the board reverts): an action
          whose response frames differ from the board counts as an effect for its class (it is not a dead class); the
          veto still uses the settled board (re-pouring the same configuration cannot help)."""
        nd = self.graph.get(k0); gid = id(self.graph)
        old = nd['tried'].get(akey) if nd is not None else None
        ch = Agent.observe(self, k0, akey, cls, g0, g1, died, up)
        rec = self.obs.setdefault((k0, akey), [0, 0]); rec[0] += 1; rec[1] += int(bool(ch))
        if old is not None and not died and not up and id(self.graph) == gid and nd is not None:
            new = nd['tried'].get(akey) or (None, ch, died)
            if bool(old[1]) != bool(ch) or (old[0] is not None and new[0] is not None and old[0] != new[0]):
                self.markov = False; self.n_inconsistent += 1
        if not ch and not died and frames and g0 is not None:
            ign = self.ignore()
            for f in list(frames)[:-1]:
                try:
                    a = np.asarray(f, dtype=np.int16)
                except Exception:
                    break
                if a.shape == g0.shape and bool(((a != g0) & ~ign).any()):
                    st = self.eff.get(cls)
                    if st is not None:
                        st[1] += 1
                    self.n_transient += 1
                    break
        return ch

    # -------------------------------------------------------------------------------------------- mapping
    @staticmethod
    def to_action(akey):
        from arc_env.structs import GameAction
        if akey[0] == 'K':
            return GameAction(int(akey[1])), {}
        return GameAction.ACTION6, {'x': int(akey[2]), 'y': int(akey[1])}

    def brain_key(self, g, a, d, nd):
        v = int(getattr(a, 'value', a) or 0)
        if v == 6:
            y, x = int((d or {}).get('y', 0)), int((d or {}).get('x', 0))
            ak = ('C', y, x)
            for a2, cl in nd['cands']:
                if a2 == ak:
                    return ak, cl
            try:
                col = int(g[y, x])
                objs = [o for o in objects(g, self.ignore(), self.bg) if o[0] == col]
                s = min(objs, key=lambda o: abs(o[2] - y) + abs(o[3] - x))[1] if objs else 1
            except Exception:
                col, s = -1, 0
            return ak, ('C', col, size_bucket(s))
        return ('K', v), ('K', v)

    def near(self, akey, bkey):
        if akey[0] == 'C' and bkey[0] == 'C':
            return 1.0 / (1.0 + (abs(akey[1] - bkey[1]) + abs(akey[2] - bkey[2])) / 4.0)
        return 1.0 if akey == bkey else 0.0

    def best_untried(self, nd, bkey):
        pool = [x for x in nd['cands'] if x[0] not in nd['tried'] and not self.deadly(x[1])]
        if not pool:
            return None
        return max(pool, key=lambda x: self.score(x[1]) + self.w_brain * self.near(x[0], bkey))

    # -------------------------------------------------------------------------------------------- the step
    def decide(self, brain, frame, a, d):
        g = settled(frame, None)
        if g is None:
            return a, d
        lv = int(getattr(frame, 'levels_completed', 0) or 0)
        if self.prev is not None:                                   # learn what the last action did
            k0, ak, cl, g0, lv0 = self.prev
            up = lv > lv0
            self.observe(k0, ak, cl, g0, g, False, up, frames=getattr(frame, 'frame', None))
            if up:
                fin = first_frame(frame)
                self.level_done(fin if fin is not None else g0)
        if self.level is None or lv != self.level:
            self.start_level(g); self.level = lv
        v = int(getattr(a, 'value', a) or 0) if a is not None else 0
        if v == 0:                                                  # RESET: the loop handles it
            self.prev = None
            return a, d
        try:
            avail = [int(getattr(x, 'value', x)) for x in brain._masked_avail(frame)]
        except Exception:
            avail = [int(getattr(x, 'value', x)) for x in (getattr(frame, 'available_actions', None) or [])]
        k, nd = self.node(g, avail)
        bkey, bcls = self.brain_key(g, a, d, nd)
        choice = None
        t = nd['tried'].get(bkey)
        if self.markov:
            wasted = t is not None and ((not t[1]) or t[2])
        else:                                                       # hidden state seen: only a repeated no-op, or a death
            rec = self.obs.get((k, bkey), [0, 0])
            wasted = t is not None and (t[2] or (rec[0] >= 2 and rec[1] == 0))
        if self.cfg['memory'] and (wasted or (t is None and self.deadly(bcls))):
            choice = self.best_untried(nd, bkey)
            if choice is None and self.cfg['nav']:
                step = self.bfs(k)
                if step is not None:
                    choice = (step, next((cl for a2, cl in nd['cands'] if a2 == step), ('NAV',))); self.n_nav += 1
            if choice is not None:
                self.n_veto += 1
        elif self.trust > 0 and self.rng.random() < self.trust:
            alt = self.best_untried(nd, bkey)
            if alt is not None and alt[0] != bkey and self.score(alt[1]) > self.score(bcls) + 1e-9:
                choice = alt; self.n_trust += 1
        if choice is None:
            akey, cls, out = bkey, bcls, (a, d)
        else:
            akey, cls = choice
            out = self.to_action(akey)
        self.prev = (k, akey, cls, g, lv); self.n_steps += 1
        return out

    def stats(self):
        return {'steps': self.n_steps, 'veto': self.n_veto, 'trust': self.n_trust, 'nav': self.n_nav,
                'states': len(self.graph), 'classes': len(self.eff), 'signature': self.sig is not None,
                'markov': self.markov, 'inconsistent': self.n_inconsistent, 'transient': self.n_transient}
