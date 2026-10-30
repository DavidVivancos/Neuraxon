"""v1.67 — headless pilot of the live engine (no web server): a small world
with the shipped world_config.json scaled down, run for --ticks, then a per-arm
table from the live NxErs (bites per 1000 ticks, executed chemotaxis, alive)
and the brains' health readouts (brain_fix.measure through the science
sample). Run from the mngol5 folder:

    python3 tools/pilot_v167.py --ticks 3000 --parallel
"""
import os
import sys
import json
import time
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("NEURAXON_HEADLESS", "1")
from server import np_fallback  # noqa: E402
np_fallback.install()
from server.engine import Engine  # noqa: E402
from server.names import NameAllocator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--n", type=int, default=80, help="starting NxErs (max = 1.5x)")
    a = ap.parse_args()
    random.seed(a.seed)
    cfg = json.load(open(os.path.join(ROOT, "world_config.json")))
    cfg.update({"engine_workers": 2 if a.parallel else 1, "brain_builders": 1,
                "reserved_builder_cores": 0, "world_size": 120, "earth_map": False,
                "starting_nxers": a.n, "max_nxers": int(a.n * 1.5), "min_alive": a.n // 2,
                "max_food": int(2600 * (120 * 120) / (360 * 360)), "history_enabled": False,
                "arm_shares": {"script": 0.1, "brain": 0.15, "sham": 0.1, "reflex": 0.1,
                               "wired": 0.3, "wired_sham": 0.15, "wired_frozen": 0.1}})
    eng = Engine(cfg, NameAllocator(0))
    t0 = time.time()
    for k in range(a.ticks):
        eng.pool.drain_loads()
        eng.step()
    dt = time.time() - t0
    rows = {}
    for x in eng.nxers.values():
        if not x.alive:
            continue
        age = max(1, eng.tick - x.born_tick)
        if age < 1000:
            continue
        r = rows.setdefault(x.arm, {"n": 0, "bites": [], "cx": [], "cxn": 0})
        r["n"] += 1
        r["bites"].append(x.stats.food_found / age * 1000.0)
        if x._cx_n >= 10:
            r["cx"].append(x._cx_sum / x._cx_n)
    print("ticks %d in %.0f s (%.2f TPS) | alive %d" % (a.ticks, dt, a.ticks / dt, eng._alive_count))
    print("%-13s %4s %10s %10s" % ("arm (age>=1k)", "n", "bites/1kt", "chemo_exec"))
    for arm in ("wired", "wired_sham", "wired_frozen", "reflex", "brain", "sham", "script"):
        r = rows.get(arm)
        if not r:
            continue
        b = sum(r["bites"]) / len(r["bites"])
        c = (sum(r["cx"]) / len(r["cx"])) if r["cx"] else float("nan")
        print("%-13s %4d %10.2f %+10.3f" % (arm, r["n"], b, c))
    # the reflex genes start random (evolution has to find the chemotactic
    # wiring), so split the gene-expressing arms by genome:
    # chemotactic = xx > 0, yy > 0 and gain*(xx + yy) > 1
    print("by genome (age >= 1k): chemotactic vs other")
    for arm in ("wired", "wired_sham", "wired_frozen", "reflex"):
        for lab, want in (("chemotactic", True), ("other", False)):
            b_ = []; c_ = []
            for x in eng.nxers.values():
                if not x.alive or x.arm != arm:
                    continue
                age = max(1, eng.tick - x.born_tick)
                g = x.reflex or {}
                ok = (g.get("rfx_xx", 0) > 0 and g.get("rfx_yy", 0) > 0
                      and g.get("rfx_gain", 0) * (g.get("rfx_xx", 0) + g.get("rfx_yy", 0)) > 1.0)
                if age < 1000 or ok != want:
                    continue
                b_.append(x.stats.food_found / age * 1000.0)
                if x._cx_n >= 10:
                    c_.append(x._cx_sum / x._cx_n)
            if b_:
                print("  %-13s %-12s n=%3d bites/1kt %6.2f chemo_exec %s" % (
                    arm, lab, len(b_), sum(b_) / len(b_),
                    ("%+.3f (n=%d)" % (sum(c_) / len(c_), len(c_))) if c_ else "-"))
    try:
        sci, per_brain = eng.pool.sample_science()
    except Exception:
        per_brain = {}
    hs = {}
    for i, m in (per_brain or {}).items():
        x = eng.nxers.get(i)
        f = (m or {}).get("fix")
        if x is None or not f:
            continue
        hs.setdefault(x.arm, []).append(f)
    print("brain health (brain_fix.measure, live sample)")
    for arm, ff in sorted(hs.items()):
        def mean(k):
            v = [f[k] for f in ff if f.get(k) is not None]
            return (sum(v) / len(v)) if v else float("nan")
        print("  %-13s n=%3d frozen %.4f frozen_in %.2f DA phasic %.3f tonic %.3f agmp %.2e link_src_ok %.2f sm_link %.2f" % (
            arm, len(ff), mean("frozen_frac"), mean("frozen_in"), mean("DA_phasic"), mean("DA_tonic"),
            mean("agmp"), mean("link_src_ok"), mean("sm_link")))
    try:
        eng.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
