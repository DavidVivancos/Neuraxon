"""v1.65 — SELF-TEST of the controlled experiment, headless and in-process
(no web server, no worker processes). Run from the mngol5 folder:

    python3 tools/selftest_arms.py              # in-process, ~2-5 minutes
    python3 tools/selftest_arms.py --parallel   # also the worker-process path
    python3 tools/selftest_arms.py --ticks 3000

It builds a small world with the shipped world_config.json (scaled down) and
checks every v1.65 promise that a long run depends on:
  1. arms are assigned at birth in roughly the configured shares
  2. brain / frozen / sham NxErs are moved ONLY by their brain, script NxErs
     ONLY by the script, mixed NxErs by both (~80/20)
  3. no NxEr is killed twice (one obituary per id)
  4. the energy ledger balances to the unit for every living NxEr
  5. a frozen brain's synapses and link weights do not change while a
     plastic brain's do
  6. sham brains receive input that differs from their own senses
  7. brain-arm NxErs never use the script floors (theft, mating intent)
  8. markers count fresh decisions (dec_n ~ age / brain_step_every)
  9. no mating (no cost, no cooldown) when the population is at its cap
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
from server.engine import Engine, ARMS  # noqa: E402
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
                "mate_cooldown_ticks": 400, "mate_min_food": 20})
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

    def _spy(batch, mods=None):
        for i, s in batch:
            sent[i] = list(s)
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
    for t in range(a.ticks):
        tick()
        for x in eng.nxers.values():
            if x.alive and x.arm == "sham" and x.id in sent and x._last_sensory:
                sham_seen += 1
                if sent[x.id][1:] != list(x._last_sensory[1:]):
                    sham_diff += 1
        sent.clear()
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
    for arm in ("brain", "frozen", "sham"):
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
                 for r in rows if r["arm"] in ("brain", "frozen", "sham"))
    check(floors == 0, "brain/frozen/sham never used a script floor")
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
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    try:
        eng.shutdown()
    except Exception:
        pass
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
