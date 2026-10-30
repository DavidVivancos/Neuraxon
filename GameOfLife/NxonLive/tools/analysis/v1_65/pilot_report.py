# Markdown tables for report §16.1 from pilot66.py outputs (pooled over seeds)
import json, sys, math
runs = [json.load(open(p)) for p in sys.argv[1:]]
ARMS = ("script", "brain", "frozen", "sham", "reflex", "reflex_sham")
m = lambda v: (sum(v) / len(v)) if v else float("nan")
T = min(r["snaps"][-1]["tick"] for r in runs)
print("Pilot end tick (min over runs):", T, " runs:", len(runs))
def snap_at(r, t):
    return max((s for s in r["snaps"] if s["tick"] <= t), key=lambda s: s["tick"])
print("\n| arm | alive | bites/1kt | chemo_exec (n≥20) | approach | drain/1kt | collisions/1kt | stolen in/1kt | rest |")
print("|---|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    v = [x for r in runs for x in snap_at(r, T)["live"] if x["arm"] == arm and x["age"] >= 1000]
    cx = [x["cx"] for x in v if x["cx"] is not None and (x["cxn"] or 0) >= 20]
    ap = [x["appr"] for x in v if x["appr"] is not None and (x["cxn"] or 0) >= 20]
    print("| %s | %d | %.2f | %+.2f (%d) | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
        arm, len(v), m([x["food_found"] / x["age"] * 1000 for x in v]), m(cx), len(cx), m(ap),
        m([(x["e_drain"] or 0) / x["age"] * 1000 for x in v]), m([(x["e_coll"] or 0) / x["age"] * 1000 for x in v]),
        m([(x["e_stin"] or 0) / x["age"] * 1000 for x in v]), m([x["rest"] or 0 for x in v])))
print("\nReflex genome class (score = gain·(xx+yy)), living NxErs aged ≥ 1000 at tick", T)
print("\n| arm | class | n | bites/1kt | chemo_exec |")
print("|---|---|---|---|---|")
for arm in ("reflex", "reflex_sham"):
    v = [x for r in runs for x in snap_at(r, T)["live"] if x["arm"] == arm and x["reflex"] and x["age"] >= 1000]
    for lab, f in (("chemotactic (score > 1)", lambda s: s > 1), ("neutral (−1..1)", lambda s: -1 <= s <= 1), ("anti (score < −1)", lambda s: s < -1)):
        g = [x for x in v if f(x["reflex"]["rfx_gain"] * (x["reflex"]["rfx_xx"] + x["reflex"]["rfx_yy"]))]
        cx = [x["cx"] for x in g if x["cx"] is not None and (x["cxn"] or 0) >= 20]
        print("| %s | %s | %d | %.2f | %+.2f |" % (arm, lab, len(g), m([x["food_found"] / x["age"] * 1000 for x in g]), m(cx)))
print("\nTime course (pooled): tick, alive, reflex vs reflex_sham chemo_exec and bites, chemotactic share of all genomes")
for t in sorted(set(s["tick"] for r in runs for s in r["snaps"] if s["tick"] <= T)):
    rows = [x for r in runs for x in snap_at(r, t)["live"]]
    def a(arm, k):
        v = [x for x in rows if x["arm"] == arm and x["age"] >= 500]
        if k == "cx":
            return m([x["cx"] for x in v if x["cx"] is not None and (x["cxn"] or 0) >= 20])
        return m([x["food_found"] / x["age"] * 1000 for x in v])
    g = [x["reflex"] for x in rows if x["reflex"]]
    share = sum(1 for q in g if q["rfx_xx"] > 0 and q["rfx_yy"] > 0) / max(1, len(g))
    print("  t=%5d alive=%3d | reflex cx %+.2f bites %.2f | reflex_sham cx %+.2f bites %.2f | brain bites %.2f sham %.2f script %.2f | share %.2f gain %.2f" % (
        t, len(rows), a("reflex", "cx"), a("reflex", "b"), a("reflex_sham", "cx"), a("reflex_sham", "b"),
        a("brain", "b"), a("sham", "b"), a("script", "b"), share, m([q["rfx_gain"] for q in g])))
ob = [o for r in runs for o in r["obits"] if o["death_tick"] <= T]
from collections import Counter
print("\ndeaths:", len(ob), dict(Counter((o.get("arm"), o.get("cause")) for o in ob)))
lin = [l for r in runs for l in r["lineage"]]
print("births by mating:", len(lin))
