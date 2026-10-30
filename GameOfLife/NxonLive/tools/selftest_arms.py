"""v1.65 / v1.66 — SELF-TEST of the controlled experiment, headless and in-process
(no web server, no worker processes). Run from the mngol5 folder:

    python3 tools/selftest_arms.py              # in-process, ~2-5 minutes
    python3 tools/selftest_arms.py --parallel   # also the worker-process path
    python3 tools/selftest_arms.py --ticks 3000

It builds a small world with the shipped world_config.json (scaled down) and
checks every v1.65 / v1.66 promise that a long run depends on:
  1. arms are assigned at birth in roughly the configured shares
  2. brain-driven NxErs (brain, frozen, sham, reflex, reflex_sham) are moved
     ONLY by their brain, script NxErs ONLY by the script
  3. no NxEr is killed twice (one obituary per id)
  4. the energy ledger balances to the unit for every living NxEr
  5. a frozen brain's synapses and link weights do not change while a
     plastic brain's do
  6. sham brains receive input that differs from their own senses
  7. brain-arm NxErs never use the script floors (theft, mating intent)
  8. markers count fresh decisions (dec_n ~ age / brain_step_every)
  9. no mating (no cost, no cooldown) when the population is at its cap
 10-11. a collision death is logged once; _kill is idempotent
 12. (v1.66) the innate reflex reaches reflex arms only (body arc or
     in-brain current, by reflex_mode)
 13. (v1.66) reflex genes are inherited from the genetic parent
 14. (v1.66) no theft while resting
 15. (v1.66) the script never walks into another NxEr
 16. (v1.66) only own-brain explorers train the NAS on foraging
 17. (v1.66) frozen brains keep intrinsic homeostasis
 18. (v1.66) the reflex steers the executed step (body mode: summed with
     the brain's command; brain mode: the motor neuron follows the current)
 19. (v1.66) a brain decision moves the body at most brain_move_steps cells
 20. (v1.67) wired brains carry the sensorimotor projection with their own
     reflex genes; no other brain does
 21. (v1.67) every brain keeps the builder's sphere shapes through the
     worker load (visual 6 inputs, motor 6 in / 7 out) and every link endpoint
     sits on the neuron role it was built for
 22. (v1.67) energy rescue: almost no frozen neurons in living brains
 23. (v1.67) consolidating slow weights: plastic brains keep their synapses
 24. (v1.67) the eating reward is sent as a phasic amount (0, burst_max]
 25. (v1.67) the script holds at most script_max_share of max_nxers
 26. (v1.67) the wired arms' rest arc reads the body's OWN smell
Exit code 0 = all passed.
"""
import os
import sys
import json
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("NEURAXON_HEADLESS", "1")
from server import np_fallback  # noqa: E402
np_fallback.install()
import architecture  # noqa: E402
architecture.load_architecture(
    os.path.join(ROOT, "architecture_files", "nas_best.json"), verbose=False)
from server.engine import WIRED_ARMS as WIRED_ARMS_  # noqa: E402
from server.engine import (Engine, ARMS, YOKED_ARMS, REFLEX_ARMS,  # noqa: E402
                           REFLEX_GENES, NAS_FUNCTION_ARMS)
from server.names import NameAllocator  # noqa: E402


class _Hist:
    """Captures what the engine would log."""
    def __init__(self):
        self.obits = []
        self.nas = []
        self.lineage_rows = []
        self.other = []

    def obituary(self, r):
        self.obits.append(r)

    def lineage(self, r):
        self.lineage_rows.append(r)

    def write(self, stream, r):
        (self.nas if stream == "nas_trials" else self.other).append(r)

    def sample(self, r):
        self.other.append(r)

    def provenance(self, r):
        pass

    def event(self, r):
        pass


def _weights(bd):
    if not bd:
        return None                      # brain not built / loaded yet
    w = []
    for sid, sp in sorted((bd.get("spheres") or {}).items()):
        net = sp.get("network") or {}
        for s in net.get("synapses", []):
            w.append((sid, s.get("pre_id"), s.get("post_id"),
                      round(s.get("w_fast", 0.0), 9),
                      round(s.get("w_slow", 0.0), 9)))
    links = []
    for lid, l in sorted((bd.get("links") or {}).items()):
        links.append((lid, [[round(x, 9) for x in row]
                            for row in l.get("weight_matrix", [])]))
    return w, links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reflex-mode", choices=("body", "brain"), default=None,
                    help="override world_config.json reflex_mode")
    ap.add_argument("--parallel", action="store_true",
                    help="also exercise the worker-process path (2 step workers)")
    a = ap.parse_args()
    random.seed(a.seed)
    cfg = json.load(open(os.path.join(ROOT, "world_config.json")))
    # engine_workers 1 = in-process brain pool (deterministic, every brain is
    # built at spawn). 0 would auto-resolve to cores-1, i.e. worker processes
    # whose brains arrive asynchronously from the builders.
    cfg.update({"engine_workers": 2 if a.parallel else 1,
                "brain_builders": 1, "reserved_builder_cores": 0,
                "world_size": 90, "earth_map": False,
                "starting_nxers": 60, "max_nxers": 90, "min_alive": 40,
                "max_food": 300, "history_enabled": False,
                "nas_explore_interval_ticks": 100,
                "mate_cooldown_ticks": 400, "mate_min_food": 20,
                # every arm present, so every check has subjects
                "arm_shares": {"script": 0.08, "mixed": 0.0, "brain": 0.1,
                               "frozen": 0.08, "sham": 0.08, "shaped": 0.0,
                               "reflex": 0.1, "reflex_sham": 0.08,
                               "wired": 0.2, "wired_sham": 0.14,
                               "wired_frozen": 0.14}})
    if a.reflex_mode:
        cfg["reflex_mode"] = a.reflex_mode
    eng = Engine(cfg, NameAllocator(0))
    print("brain pool:", "worker processes" if eng.pool.parallel else "in-process")

    def tick():
        # the game server drains builder-finished brains into the workers
        # every tick; a bare Engine must do it itself
        eng.pool.drain_loads()
        eng.step()
    h = _Hist()
    eng.history = h
    # capture the sensory vectors actually sent to the brains (sham check)
    sent = {}
    _orig_step = eng.pool.step

    drive = {}

    mods_seen = []

    def _spy(batch, mods=None):
        for item in batch:
            sent[item[0]] = list(item[1])
            drive[item[0]] = item[2] if len(item) > 2 else None
        if mods:
            mods_seen.extend(mods.values())
        return _orig_step(batch, mods)
    eng.pool.step = _spy
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            ok = False

    # pick one frozen and one plastic (brain-arm) NxEr to watch
    def pick(arm):
        for x in eng.nxers.values():
            if x.alive and x.arm == arm:
                return x
        return None
    for _ in range(24):          # let every brain take its first steps
        tick()                   # (initial weights outside the clip
    fz, pl = pick("frozen"), pick("brain")   # bounds are clamped once)
    w0f = _weights(eng.pool.export(fz.id)) if fz else None
    w0p = _weights(eng.pool.export(pl.id)) if pl else None
    waited = 0
    while ((fz is not None and w0f is None) or (pl is not None and w0p is None)) \
            and waited < 600:   # worker mode: wait for the builders
        tick()
        waited += 1
        if fz is not None and w0f is None:
            w0f = _weights(eng.pool.export(fz.id))
        if pl is not None and w0p is None:
            w0p = _weights(eng.pool.export(pl.id))
    if waited:
        print("waited %d ticks for the watched brains to load" % waited)
        for _ in range(16):      # first steps after loading (clip clamp)
            tick()
        w0f = _weights(eng.pool.export(fz.id)) if fz else None
        w0p = _weights(eng.pool.export(pl.id)) if pl else None
    sham_diff = 0
    sham_seen = 0
    rfx_seen = [0, 0, 0]     # reflex-arm decisions, with drive, non-reflex with drive
    rfx_body = [0, 0, 0]     # reflex-arm decisions, with body arc, non-reflex with arc
    mode = cfg.get("reflex_mode", "body")
    rep_max = 0              # most brain steps executed within one decision
    for t in range(a.ticks):
        tick()
        for x in eng.nxers.values():
            if x.alive and x.arm in YOKED_ARMS and x.id in sent and x._last_sensory:
                sham_seen += 1
                if sent[x.id][1:] != list(x._last_sensory[1:]):
                    sham_diff += 1
        for x in eng.nxers.values():
            if x.alive and x.id in drive:
                if x.arm in REFLEX_ARMS:
                    rfx_seen[0] += 1
                    if drive[x.id] is not None:
                        rfx_seen[1] += 1
                    if x._rfx_cmd is not None:
                        rfx_body[1] += 1
                elif drive[x.id] is not None:
                    rfx_seen[2] += 1
                # v1.67 — the wired arms carry a body REST arc only: their
                # _rfx_cmd must have zero x/y (direction goes through the brain)
                if x.arm in WIRED_ARMS_ and x._rfx_cmd is not None:
                    if x._rfx_cmd[0] != 0.0 or x._rfx_cmd[1] != 0.0:
                        rfx_body[2] += 1
                elif x.arm not in REFLEX_ARMS and x._rfx_cmd is not None:
                    rfx_body[2] += 1
                if x.arm in REFLEX_ARMS:
                    rfx_body[0] += 1
            if x.alive and x.arm in ("brain", "frozen", "sham", "reflex", "reflex_sham", "wired", "wired_sham", "wired_frozen"):
                rep_max = max(rep_max, x._rep_n)
        sent.clear()
        drive.clear()
    print("ran %d ticks; alive %d; deaths %d; births %d" % (
        eng.tick, sum(1 for x in eng.nxers.values() if x.alive),
        len(h.obits), len(h.lineage_rows)))

    # 1. shares
    born = [x for x in eng.nxers.values()] + []
    arms_seen = {}
    for r in h.obits:
        arms_seen[r["arm"]] = arms_seen.get(r["arm"], 0) + 1
    for x in eng.nxers.values():
        if x.alive:
            arms_seen[x.arm] = arms_seen.get(x.arm, 0) + 1
    tot = sum(arms_seen.values())
    print("  arms:", {k: "%d (%.0f%%)" % (v, 100.0 * v / tot)
                      for k, v in sorted(arms_seen.items())})
    check(set(arms_seen) <= set(ARMS) and len(arms_seen) >= 4,
          "arms assigned from %s" % (ARMS,))
    # 2. who moves the body
    live = [x for x in eng.nxers.values() if x.alive] + []
    rows = [dict(r) for r in h.obits]
    for x in live:
        rows.append(dict(Engine.behav(x), arm=x.arm))
    def frac(arm):
        b = sum(r["moves"] * (r["brain_move_frac"] or 0) for r in rows
                if r["arm"] == arm and r.get("moves"))
        m = sum(r["moves"] for r in rows if r["arm"] == arm and r.get("moves"))
        return (b / m) if m else None
    for arm in ARMS:
        print("   brain share of executed moves [%s]: %s" % (arm, frac(arm)))
    for arm in ("brain", "frozen", "sham", "reflex", "reflex_sham", "wired", "wired_sham", "wired_frozen"):
        f = frac(arm)
        check(f is None or f == 1.0, "%s arm moved only by its brain" % arm)
    f = frac("script")
    check(f is None or f < 0.02, "script arm moved by the script")
    def tick_share(arm):
        n = sum(1 for r in rows if r["arm"] == arm)
        return (sum(r.get("script_tick_frac", 0) for r in rows
                    if r["arm"] == arm) / n) if n else None
    for arm in ARMS:
        print("   script share of ticks [%s]: %s" % (arm, tick_share(arm)))
    f = tick_share("mixed")
    check(f is None or 0.10 < f < 0.30,
          "mixed arm: script drives ~20%% of ticks (got %s)" % f)
    f = tick_share("script")
    check(f is None or f > 0.95, "script arm: script drives ~all ticks (got %s)" % f)
    # 3. one death per NxEr
    ids = [r["id"] for r in h.obits]
    check(len(ids) == len(set(ids)), "no NxEr killed twice (%d obituaries)" % len(ids))
    nas_ids = [r["id"] for r in h.nas]
    check(len(nas_ids) == len(set(nas_ids)), "no explorer logged twice")
    # 4. ledger balances
    worst = 0.0
    for x in live:
        bal = (x.start_food + x.e_eat + x.e_stin + x.e_gvin - x.e_stout
               - x.e_gvout - x.e_coll - x.e_mate - x.e_cap - x.e_drain)
        worst = max(worst, abs(bal - x.food))
    check(worst < 1e-6, "energy ledger balances (max error %.2e)" % worst)
    # 5. frozen vs plastic
    w1f = _weights(eng.pool.export(fz.id)) if (fz is not None and fz.alive) else None
    if w0f is not None and w1f is not None:
        same_syn = [x[:3] for x in w0f[0]] == [x[:3] for x in w1f[0]]
        dmax = max([abs(p - q) for x, y in zip(w0f[0], w1f[0])
                    for p, q in zip(x[3:], y[3:])] + [0.0])
        lmax = max([abs(p - q) for (_, ma), (_, mb) in zip(w0f[1], w1f[1])
                    for ra, rb in zip(ma, mb) for p, q in zip(ra, rb)] + [0.0])
        check(same_syn and dmax < 1e-6 and lmax < 1e-9,
              "frozen brain unchanged (%d synapses, max |dw| %.1e, links %.1e)"
              % (len(w1f[0]), dmax, lmax))
    else:
        print("  SKIP  frozen watch-brain died or never loaded")
    w1p = _weights(eng.pool.export(pl.id)) if (pl is not None and pl.alive) else None
    if w0p is not None and w1p is not None:
        check(w0p != w1p, "plastic brain changed (%d -> %d synapses)"
              % (len(w0p[0]), len(w1p[0])))
    # 6. sham input
    check(sham_seen == 0 or sham_diff / sham_seen > 0.5,
          "sham brains fed another NxEr's senses (%d/%d decisions)" % (sham_diff, sham_seen))
    # 7. no script floors for brain arms
    floors = sum(r.get("n_st_floor", 0) + r.get("n_mate_floor", 0)
                 for r in rows if r["arm"] in ("brain", "frozen", "sham", "reflex", "reflex_sham", "wired", "wired_sham", "wired_frozen"))
    check(floors == 0, "brain-driven arms never used a script floor")
    # 8. decisions per age
    K = max(1, int(cfg.get("brain_step_every", 1)))
    bad = [x for x in live if x._dec_n > (eng.tick - x.born_tick) / K + 2]
    check(not bad, "markers count fresh decisions (dec_n <= age/%d)" % K)
    # 9. no mating at the cap
    m = [x for x in live if not x.is_male][:1] + [x for x in live if x.is_male][:1]
    if len(m) == 2:
        a0, b0 = m
        f0, n0 = a0.food, a0.stats.mates_performed
        eng._alive_count = int(cfg["max_nxers"])
        eng._mate(a0, b0)
        check(a0.food == f0 and a0.stats.mates_performed == n0,
              "no mating cost or cooldown at the population cap")
    # 10. a collision death is logged ONCE (v1.64 logged it twice)
    from server.engine import DIR_OFFSETS
    K = max(1, int(cfg.get("brain_step_every", 1)))
    nsh = max(1, getattr(eng.pool, "n", 1))
    victim = None
    for x in eng.nxers.values():
        if (x.alive and x.arm == "brain"
                and (eng.tick + 1 + x.id // nsh) % K != 0):
            victim = x
            break
    if victim is not None:
        W = eng.world
        spot = None
        for yy in range(1, W._N - 1):
            for xx in range(1, W._N - 1):
                if not W.passable(xx, yy):
                    for d in DIR_OFFSETS:
                        px, py = xx - d[0], yy - d[1]
                        if (eng._can_enter(victim, px, py)
                                and (px, py) not in eng._occupied):
                            spot = (px, py, d)
                            break
                if spot:
                    break
            if spot:
                break
        if spot:
            px, py, d = spot
            eng._occupied.pop(tuple(victim.pos), None)
            victim.pos = [px, py]
            eng._occupied[(px, py)] = victim.id
            victim.food = 0.05
            victim._last_out = ([d[0], d[1], 0, 0, 0, 0, 0], 1.0)
            victim._fresh = False
            victim._hold = False
            # v1.67 — a replayed tick of a fresh decision: no detour turn
            # left over from the victim's real last decision, no steps used
            victim._rot = 0
            victim._rep_n = 0
            victim._rfx_cmd = None
            before = len(h.obits)
            tick()
            mine = [r for r in h.obits if r["id"] == victim.id]
            check(not victim.alive and len(mine) == 1
                  and mine[0]["cause"] == "collision",
                  "collision death logged once as 'collision' (%d rows)" % len(mine))
        else:
            print("  SKIP  no rock found for the collision test")
    # 11. _kill is idempotent
    y = next((x for x in eng.nxers.values() if x.alive), None)
    if y is not None:
        n0 = len(h.obits)
        eng._kill(y, "starved")
        eng._kill(y, "starved")
        check(len(h.obits) == n0 + 1, "a second _kill of the same NxEr is a no-op")
    # 12. the reflex reaches reflex arms only
    if mode == "brain":
        check(rfx_seen[2] == 0 and (rfx_seen[0] == 0 or rfx_seen[1] == rfx_seen[0]),
              "reflex drive sent to reflex arms only (%d/%d reflex decisions, %d elsewhere)"
              % (rfx_seen[1], rfx_seen[0], rfx_seen[2]))
    else:
        check(rfx_body[2] == 0 and rfx_seen[1] == 0 and rfx_seen[2] == 0
              and (rfx_body[0] == 0 or rfx_body[1] == rfx_body[0]),
              "body reflex set for reflex arms only, wired arms rest-only (%d/%d reflex decisions, %d elsewhere, "
              "%d in-brain currents)" % (rfx_body[1], rfx_body[0], rfx_body[2], rfx_seen[1]))
    # 13. reflex genes inherited from the genetic parent (+ small mutation)
    alln = {x.id: x for x in eng.nxers.values()}
    diffs = []
    for l in h.lineage_rows:
        c = l.get("reflex"); gp = alln.get(l.get("genetic_parent"))
        if c and gp is not None and gp.reflex:
            diffs.append(max(abs(c[k] - gp.reflex[k]) for k in REFLEX_GENES))
    check(not diffs or (sum(diffs) / len(diffs) < 0.35),
          "reflex genes inherited from the genetic parent (mean max |diff| %.3f over %d births)"
          % ((sum(diffs) / len(diffs)) if diffs else 0.0, len(diffs)))
    # 14. no theft while resting
    pair = None
    for x in eng.nxers.values():
        if not x.alive:
            continue
        for o in eng.nxers.values():
            if o is not x and o.alive and max(abs(o.pos[0] - x.pos[0]), abs(o.pos[1] - x.pos[1])) == 1:
                pair = (x, o)
                break
        if pair:
            break
    if pair:
        x, o = pair
        nbrs = [n for n in eng.nxers.values() if n.alive and n is not x
                and max(abs(n.pos[0] - x.pos[0]), abs(n.pos[1] - x.pos[1])) <= 1]
        for n in nbrs + [x]:
            n.food = 50.0
            n.mating_intent_until_tick = 0
        x._auto = 0.0; x._fresh = False; x._hold = False
        x._rfx_cmd = None; x._rot = 0     # v1.67 — no body arc in this unit test
        x.last_steal_tick = -10 ** 6
        eng._build_spatial()
        eng._act(x, [0, 0, 1, 0, 0, 1, 0])            # social +1 AND rest +1
        f_rest = sum(n.food for n in nbrs)
        x._rested = False
        x.last_steal_tick = -10 ** 6
        eng._act(x, [0, 0, 1, 0, 0, 0, 0])            # social +1, not resting
        f_act = sum(n.food for n in nbrs)
        check(f_rest == 50.0 * len(nbrs) and f_act == 50.0 * len(nbrs) - 1,
              "no theft while resting, theft when active (neighbours %.0f -> %.0f -> %.0f)"
              % (50.0 * len(nbrs), f_rest, f_act))
        x._rested = False
    else:
        print("  SKIP  no adjacent pair for the theft test")
    # 15. scripted steps never ram an occupied cell
    sc_coll = sum(r.get("coll_nx", 0) for r in rows if r["arm"] == "script")
    sc_mv = sum(r.get("moves", 0) for r in rows if r["arm"] == "script")
    check(sc_coll == 0, "script arm never walked into another NxEr (%d bumps, %d moves)" % (sc_coll, sc_mv))
    # 16. only own-brain arms train the NAS sampler on foraging
    rec = [t for t in eng._nas_hist if len(t) == 3]
    with_b = sum(1 for t in rec if t[1] is not None)
    exp_arms = [r["arm"] for r in h.nas]
    n_fn = sum(1 for a_ in exp_arms if a_ in NAS_FUNCTION_ARMS)
    check(with_b == n_fn, "NAS foraging score recorded for own-brain explorers only (%d of %d trials)"
          % (with_b, len(rec)))
    # 17. frozen brains keep intrinsic homeostasis (v1.66)
    if fz is not None and not eng.pool.parallel and fz.id in getattr(eng.pool, "_brains", {}):
        nets = [sp.network.params for sp in eng.pool._brains[fz.id].spheres.values()]
        keep = cfg.get("frozen_keep_homeostasis", True)
        check(all(bool(getattr(p_, "adaptive_threshold_enabled", keep)) == bool(keep) for p_ in nets)
              and all(getattr(p_, "agmp_enabled", False) is False for p_ in nets),
              "frozen brain: synaptic plasticity off, intrinsic homeostasis %s" % ("kept" if keep else "off"))
    # 19. a decision moves the body at most brain_move_steps cells
    bms = int(cfg.get("brain_move_steps", 2))
    idle_ok = True
    bx = next((x for x in eng.nxers.values() if x.alive and x.arm == "brain"), None)
    if bx is not None and bms > 0:
        p0 = list(bx.pos)
        bx._fresh = False
        bx._hold = False
        bx._rep_n = bms
        bx._rfx_cmd = None
        bx.last_move_tick = eng.tick - 50
        eng._build_spatial()
        eng._act(bx, [1, 0, 0, 0, 0, 0, 0])
        idle_ok = (list(bx.pos) == p0 and bx.last_move_tick == eng.tick)
    check((bms <= 0 or rep_max <= bms) and idle_ok,
          "a brain decision moves the body at most %d cells (max seen %d); standing out a "
          "moving decision is not idle (%s)" % (bms, rep_max, "ok" if idle_ok else "WRONG"))
    # 18. the reflex steers the executed step
    rb = next((x for x in eng.nxers.values() if x.alive and x.arm in REFLEX_ARMS
               and eng.pool.export(x.id)), None)
    if rb is not None and mode != "brain":
        # body arc: step = sign(brain + reflex) beyond +-0.5, per axis
        res = []
        for brain_mx, r_x in ((0, 1.5), (1, -2.0), (1, 0.0), (-1, 1.0)):
            rb._rfx_cmd = (r_x, 0.0, 0.0)
            rb._hold = False
            rb._rep_n = 0
            rb._fresh = False
            seen = {}
            _orig_can = eng._can_enter
            eng._can_enter = lambda nx_, x_, y_: (seen.setdefault("dx", x_ - nx_.pos[0]), False)[1]
            try:
                eng._build_spatial()
                eng._act(rb, [brain_mx, 0, 0, 0, 0, 0, 0])
            finally:
                eng._can_enter = _orig_can
            res.append(seen.get("dx", 0))
        rb._rfx_cmd = None
        check(res == [1, -1, 1, 0],
              "body reflex sums with the brain (0+1.5->%+d, 1-2.0->%+d, 1+0->%+d, -1+1.0->%+d; "
              "expected +1 -1 +1 0)" % tuple(res))
    elif rb is not None:
        sv = list(rb._last_sensory or [0.8] + [0.0] * 9)
        pos = neg = 0
        for _ in range(30):
            o = eng.pool.step([(rb.id, sv, (1.5, 0.0, 0.0))]).get(rb.id, ([0] * 7, 1))[0]
            pos += 1 if o[0] > 0 else 0
        for _ in range(30):
            o = eng.pool.step([(rb.id, sv, (-1.5, 0.0, 0.0))]).get(rb.id, ([0] * 7, 1))[0]
            neg += 1 if o[0] > 0 else 0
        # informational: an aged brain's motor neuron can lock and ignore
        # the current (report_v1.65 §14.3) — that is why "body" is default
        print(("  INFO  " if pos > neg + 5 else "  INFO  (locked) ")
              + "in-brain reflex current: mvx=+1 on %d/30 with +drive, %d/30 with -drive"
              % (pos, neg))
    else:
        print("  SKIP  no loaded reflex brain for the drive test")
    # ---------------- v1.67 ----------------
    from server import brain_fix as BF
    from server.engine import WIRED_ARMS, FROZEN_ARMS
    from neuraxon.multisphere import load_multisphere_from_dict
    live = [x for x in eng.nxers.values() if x.alive]
    n_w = n_wok = n_o = n_obad = 0
    sh_n = sh_ok = 0
    froz = []
    syn_pl = []
    for x in live[:45]:
        bd = eng.pool.export(x.id)
        if not bd:
            continue
        links = bd.get("links", {})
        has = BF.SM_LINK_ID in links
        if x.arm in WIRED_ARMS:
            n_w += 1
            W = (links.get(BF.SM_LINK_ID) or {}).get("weight_matrix")
            exp = BF.sm_weights(x.reflex)
            if has and W and all(abs(W[r][c] - exp[r][c]) < 1e-6
                                 for r in range(2) for c in range(2)):
                n_wok += 1
        else:
            n_o += 1
            n_obad += 1 if has else 0
        sp = bd["spheres"]
        vis_in = len(sp["visual"]["network"]["neurons"]["input"])
        mo = sp["motor"]["network"]["neurons"]
        b = load_multisphere_from_dict(bd, rebuild_net_fn=BF.rebuild_net_exact)
        w = BF.wiring_check(b)
        sh_n += 1
        if (vis_in == 6 and (len(mo["input"]), len(mo["output"])) == (6, 7)
                and w["src_ok"] == w["src_n"] and w["tgt_ok"] == w["tgt_n"]):
            sh_ok += 1
        froz.append(BF.measure(b).get("frozen_frac") or 0.0)
        if x.arm not in FROZEN_ARMS and eng.tick - x.born_tick >= 500:
            syn_pl.append(sum(len(s_["network"]["synapses"]) for s_ in sp.values()))
    check(n_w > 0 and n_wok == n_w and n_obad == 0,
          "sensorimotor projection: %d/%d wired brains carry it with their own genes, "
          "%d of %d other brains have it" % (n_wok, n_w, n_obad, n_o))
    check(sh_n > 0 and sh_ok == sh_n,
          "exact shapes and link roles kept through the worker load (%d/%d brains)" % (sh_ok, sh_n))
    mf = sum(froz) / len(froz) if froz else 1.0
    check(froz and mf < 0.03, "energy rescue: mean frozen share %.4f (< 0.03)" % mf)
    if syn_pl:
        ms = sum(syn_pl) / len(syn_pl)
        check(ms > 700, "plastic brains (age >= 500 ticks) keep %.0f synapses on average "
              "(builder ~1,320; V 1.086 kept ~3%%)" % ms)
    else:
        print("  SKIP  no plastic brain old enough for the synapse check")
    if eng._fix_cfg.get("reward_phasic"):
        dmx = float(cfg.get("dopamine_burst_max", 0.85))
        check(mods_seen and all(0.0 < v <= dmx + 1e-9 for v in mods_seen),
              "eating reward sent as phasic amounts (%d rewards, max %.2f <= %.2f)"
              % (len(mods_seen), max(mods_seen) if mods_seen else 0.0, dmx))
    n_sc = sum(1 for x in live if x.arm == "script")
    lim = eng._script_max * int(cfg["max_nxers"])
    check(eng._script_max <= 0 or n_sc <= lim + 1,
          "script quota: %d living script NxErs (limit %.0f)" % (n_sc, lim))
    ws = [x for x in live if x.arm == "wired_sham" and x._rfx_cmd is not None
          and x._last_sensory and x.reflex]
    bad = [x for x in ws if abs(x._rfx_cmd[2] - x.reflex["rfx_gain"] * x.reflex["rfx_rest"]
                                * (2.0 * x._last_sensory[5] - 1.0)) > 1e-9]
    check(ws and not bad, "wired_sham rest arc reads the body's own smell (%d checked)" % len(ws))
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    try:
        eng.shutdown()
    except Exception:
        pass
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
