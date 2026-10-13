"""v1.65 — ARM REPORT: read a run's history and answer the experiment.

    python3 tools/arm_report.py state            # or path/to/state
    python3 tools/arm_report.py state --min-life 2000

Pure standard library (runs under PyPy on the server). Prints, per arm
(script / mixed / brain / frozen / sham):
  * deaths, causes, survival to fixed ages (right-censoring handled: only
    NxErs born early enough to reach an age count toward it; the living are
    read from the snapshot)
  * where energy came from and went (the v1.65 ledger)
  * brain markers: motor_frac, chemotaxis (+ t-test vs 0), lock_frac,
    social/give/mate/rest outputs, who moved the body
  * repeatability (ICC) of chemotaxis and motor_frac from brain_samples,
    on NON-OVERLAPPING intervals between consecutive readings
  * the key contrasts with bootstrap 95% CIs: brain vs sham (does the brain
    use its senses?), brain vs frozen (does plasticity help?), brain vs
    script (does the brain beat the scripted body?)
  * heritability inside the brain-driven arms: genetic parent -> child,
    with the other (non-genetic) parent as the shared-environment control
"""
import os
import sys
import json
import math
import random
import argparse
from collections import defaultdict

ARMS = ("script", "mixed", "brain", "frozen", "sham")
BRAINY = ("brain", "frozen", "sham")


def _jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


def _boot_diff(a, b, stat=_mean, n=1000, seed=1):
    """95% CI of stat(a) - stat(b) by bootstrap."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 10 or len(b) < 10:
        return None
    rnd = random.Random(seed)
    d = []
    for _ in range(n):
        ra = [a[rnd.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rnd.randrange(len(b))] for _ in range(len(b))]
        d.append(stat(ra) - stat(rb))
    d.sort()
    return stat(a) - stat(b), d[int(0.025 * n)], d[int(0.975 * n)]


def _icc(groups):
    groups = [g for g in groups if len(g) >= 3]
    k = len(groups)
    if k < 10:
        return None
    N = sum(len(g) for g in groups)
    grand = sum(sum(g) for g in groups) / N
    ssb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ssw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)
    msb = ssb / (k - 1)
    msw = ssw / max(1, N - k)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    s2b = max(0.0, (msb - msw) / n0)
    return (s2b / (s2b + msw) if s2b + msw > 0 else 0.0), k, N


def _pearson(x, y):
    n = len(x)
    if n < 10:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", nargs="?", default="state")
    ap.add_argument("--min-life", type=int, default=2000,
                    help="rates are computed for NxErs that lived at least this long")
    a = ap.parse_args()
    hist = os.path.join(a.state, "history")
    prov = _jsonl(os.path.join(hist, "provenance.jsonl"))
    if prov:
        p = prov[-1]
        sci = p.get("science", {})
        print("server:", p.get("server_version"), "| arms enabled:",
              sci.get("arms_enabled"), "| shares:", sci.get("arm_cumulative_shares"))
    obits = {}
    for o in _jsonl(os.path.join(hist, "obituaries.jsonl")):
        obits.setdefault(o["id"], o)            # first record wins (dedupe)
    lineage = _jsonl(os.path.join(hist, "lineage.jsonl"))
    samples = _jsonl(os.path.join(hist, "brain_samples.jsonl"))
    snap = None
    for fn in ("snapshot.json", "snapshot.prev.json"):
        try:
            snap = json.load(open(os.path.join(a.state, fn)))
            break
        except (OSError, ValueError):
            continue
    t_end = (snap or {}).get("tick") or max([o["death_tick"] for o in obits.values()] + [0])
    alive = {}
    for n in (snap or {}).get("nxers", []):
        if n.get("alive") and not n.get("is_managed"):
            alive[n["id"]] = n
    print("deaths (unique):", len(obits), "| alive at end:", len(alive), "| end tick:", t_end)
    if not obits:
        return
    have_arm = sum(1 for o in obits.values() if o.get("arm"))
    if not have_arm:
        print("no 'arm' field in obituaries: this is not a v1.65+ run")
        return
    by = defaultdict(list)
    for o in obits.values():
        if o.get("managed"):
            continue
        by[o.get("arm") or "mixed"].append(o)
    for o in obits.values():
        L = max(1, o["lifespan_ticks"])
        o["eat_rate"] = (o.get("e_eat") or 0) / L * 1000
        o["net_theft"] = ((o.get("e_stin") or 0) - (o.get("e_stout") or 0)) / L * 1000
        o["coll_per_dec"] = (o.get("collisions") or 0) / max(1, o.get("dec_n") or 0)
        o["off_rate"] = o.get("n_offspring", 0) / L * 100000

    print("\n== 1. DEATHS AND SURVIVAL (censoring-aware)")
    ages = (2000, 5000, 20000, 50000, 100000, 200000)
    print("%-7s %6s %8s %8s %8s %8s  " % ("arm", "deaths", "starved", "collis", "idle", "medL")
          + " ".join("S(%dk)" % (x // 1000) for x in ages))
    for arm in ARMS:
        v = by.get(arm, [])
        if not v:
            continue
        c = defaultdict(int)
        for o in v:
            c[o["cause"]] += 1
        surv = []
        al = [n for n in alive.values() if (n.get("arm") or "mixed") == arm]
        for A in ages:
            elig_d = [o for o in v if o["born_tick"] <= t_end - A]
            elig_a = [n for n in al if n.get("born_tick", 0) <= t_end - A]
            tot = len(elig_d) + len(elig_a)
            ok = sum(1 for o in elig_d if o["lifespan_ticks"] >= A) + len(elig_a)
            surv.append(_fmt(ok / tot if tot else None, 3))
        n = len(v)
        print("%-7s %6d %7.1f%% %7.1f%% %7.1f%% %8s  " % (
            arm, n, 100.0 * c["starved"] / n, 100.0 * c["collision"] / n,
            100.0 * c["idle"] / n, _med([o["lifespan_ticks"] for o in v]))
            + " ".join("%6s" % s for s in surv))

    print("\n== 2. ENERGY LEDGER per 1000 ticks (dead, lived >= %d), medians" % a.min_life)
    keys = ("e_eat", "e_bonus", "e_stin", "e_stout", "e_gvin", "e_gvout",
            "e_drain", "e_relief", "e_coll", "e_mate", "e_cap")
    print("%-7s " % "arm" + " ".join("%8s" % k[2:] for k in keys))
    for arm in ARMS:
        v = [o for o in by.get(arm, []) if o["lifespan_ticks"] >= a.min_life]
        if not v:
            continue
        row = []
        for k in keys:
            row.append(_fmt(_med([(o.get(k) or 0) / o["lifespan_ticks"] * 1000 for o in v]), 2))
        print("%-7s " % arm + " ".join("%8s" % x for x in row))

    print("\n== 3. BRAIN MARKERS (dead, lived >= %d)" % a.min_life)
    print("%-7s %6s %7s %8s %8s %6s %6s %6s %6s %6s %7s %7s %7s" % (
        "arm", "n", "motor", "chemo", "t(chemo)", "lock", "soc", "give", "mate", "rest",
        "brainmv", "scripT", "coll/d"))
    for arm in ARMS:
        v = [o for o in by.get(arm, []) if o["lifespan_ticks"] >= a.min_life]
        if not v:
            continue
        ch = [o["chemotaxis"] for o in v if o.get("chemotaxis") is not None
              and (o.get("chemo_n") or 0) >= 50]
        t = None
        if len(ch) > 10:
            m = sum(ch) / len(ch)
            sd = math.sqrt(sum((x - m) ** 2 for x in ch) / (len(ch) - 1))
            t = m / (sd / math.sqrt(len(ch))) if sd > 0 else None
        print("%-7s %6d %7s %8s %8s %6s %6s %6s %6s %6s %7s %7s %7s" % (
            arm, len(v), _fmt(_mean([o.get("motor_frac") for o in v])),
            _fmt(_mean(ch), 4), _fmt(t, 2),
            _fmt(_mean([o.get("lock_frac") for o in v])),
            _fmt(_mean([o.get("out_soc") for o in v])),
            _fmt(_mean([o.get("out_give") for o in v])),
            _fmt(_mean([o.get("out_mate") for o in v])),
            _fmt(_mean([o.get("out_rest") for o in v])),
            _fmt(_mean([o.get("brain_move_frac") for o in v])),
            _fmt(_mean([o.get("script_tick_frac") for o in v])),
            _fmt(_mean([o["coll_per_dec"] for o in v]))))

    print("\n== 4. REPEATABILITY (ICC) on non-overlapping intervals, >=3 per brain")
    per = defaultdict(list)
    for s in samples:
        per[s["id"]].append(s)
    for arm in ARMS:
        gch, gmf = [], []
        for nid, rows in per.items():
            rows.sort(key=lambda r: r["tick"])
            if (rows[0].get("arm") or "mixed") != arm:
                continue
            ch, mf = [], []
            for r0, r1 in zip(rows, rows[1:]):
                d0, d1 = r0.get("dec_n") or 0, r1.get("dec_n") or 0
                if d1 - d0 >= 50:
                    mf.append((r1["motor_frac"] * d1 - r0["motor_frac"] * d0) / (d1 - d0))
                c0, c1 = r0.get("chemotaxis"), r1.get("chemotaxis")
                n0, n1 = r0.get("chemo_n") or 0, r1.get("chemo_n") or 0
                if c0 is not None and c1 is not None and n1 - n0 >= 50:
                    ch.append((c1 * n1 - c0 * n0) / (n1 - n0))
            gch.append(ch)
            gmf.append(mf)
        ic, im = _icc(gch), _icc(gmf)
        if ic or im:
            print("%-7s chemotaxis ICC=%s (brains %s)   motor_frac ICC=%s (brains %s)" % (
                arm, _fmt(ic and ic[0]), ic and ic[1], _fmt(im and im[0]), im and im[1]))

    print("\n== 5. CONTRASTS (bootstrap 95% CI of the difference in means)")
    def col(arm, f):
        return [f(o) for o in by.get(arm, []) if o["lifespan_ticks"] >= a.min_life]
    for x, y, why in (("brain", "sham", "does the brain USE its senses?"),
                      ("brain", "frozen", "does within-life plasticity help?"),
                      ("brain", "script", "does the brain beat the scripted body?"),
                      ("mixed", "script", "does 80% brain control cost or help?")):
        for lab, f in (("eat/1000t", lambda o: o["eat_rate"]),
                       ("log life", lambda o: math.log(max(1, o["lifespan_ticks"]))),
                       ("offspring/100kt", lambda o: o["off_rate"]),
                       ("chemotaxis", lambda o: o.get("chemotaxis") if (o.get("chemo_n") or 0) >= 50 else None)):
            r = _boot_diff(col(x, f), col(y, f))
            if r:
                sig = "*" if (r[1] > 0 or r[2] < 0) else " "
                print("  %-6s - %-6s %-16s %+.4f  [%+.4f, %+.4f] %s  (%s)" % (
                    x, y, lab, r[0], r[1], r[2], sig, why))

    print("\n== 6. HERITABILITY in brain-driven arms (genetic parent -> child)")
    for trait, f in (("eat_rate", lambda o: o["eat_rate"]),
                     ("log life", lambda o: math.log(max(1, o["lifespan_ticks"]))),
                     ("chemotaxis", lambda o: o.get("chemotaxis") if (o.get("chemo_n") or 0) >= 50 else None),
                     ("motor_frac", lambda o: o.get("motor_frac")),
                     ("m_fit", lambda o: o.get("m_fit"))):
        xs, ys, xo, yo = [], [], [], []
        for l in lineage:
            c = obits.get(l["id"])
            gp = l.get("genetic_parent")
            if c is None or gp is None or c.get("arm") not in BRAINY:
                continue
            g = obits.get(gp)
            other = [p for p in l.get("parents", []) if p != gp]
            ot = obits.get(other[0]) if other else None
            if g is not None and g.get("arm") in BRAINY:
                vg, vc = f(g), f(c)
                if vg is not None and vc is not None:
                    xs.append(vg); ys.append(vc)
            if ot is not None and ot.get("arm") in BRAINY:
                vo, vc = f(ot), f(c)
                if vo is not None and vc is not None:
                    xo.append(vo); yo.append(vc)
        print("  %-11s genetic r=%s (n=%d)   other-parent r=%s (n=%d)" % (
            trait, _fmt(_pearson(xs, ys)), len(xs), _fmt(_pearson(xo, yo)), len(xo)))


if __name__ == "__main__":
    main()
