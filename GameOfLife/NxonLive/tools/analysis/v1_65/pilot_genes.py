import json, sys, math
def pear(a, b):
    n = len(a); ma = sum(a)/n; mb = sum(b)/n
    sa = math.sqrt(sum((x-ma)**2 for x in a)); sb = math.sqrt(sum((y-mb)**2 for y in b))
    return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / (sa*sb) if sa*sb > 0 else float('nan')
rows = []
for p in sys.argv[1:]:
    d = json.load(open(p)); s = d["snaps"][-1]
    for r in s["live"]:
        r["_t"] = s["tick"]; rows.append(r)
print("tick", rows[0]["_t"] if rows else None, "rows", len(rows))
for arm in ("reflex", "reflex_sham"):
    v = [r for r in rows if r["arm"] == arm and r["reflex"] and r["age"] >= 1000]
    sc = [r["reflex"]["rfx_gain"] * (r["reflex"]["rfx_xx"] + r["reflex"]["rfx_yy"]) for r in v]
    bt = [r["food_found"] / r["age"] * 1000 for r in v]
    good = [b for s_, b in zip(sc, bt) if s_ > 1.0]; bad = [b for s_, b in zip(sc, bt) if s_ < -1.0]; mid = [b for s_, b in zip(sc, bt) if -1.0 <= s_ <= 1.0]
    m = lambda x: sum(x)/len(x) if x else float('nan')
    print("%-11s n=%d  genome score vs bites r=%+.2f | bites: chemotactic genome (score>1) %.2f (n=%d), neutral %.2f (n=%d), anti (score<-1) %.2f (n=%d) | rest %.2f" % (
        arm, len(v), pear(sc, bt), m(good), len(good), m(mid), len(mid), m(bad), len(bad), m([r["rest"] or 0 for r in v])))
    # executed chemotaxis by genome class
    cx = [(s_, r["cx"]) for s_, r in zip(sc, v) if r["cx"] is not None and (r["cxn"] or 0) >= 20]
    print("    chemo_exec: chemotactic %.2f neutral %.2f anti %.2f" % (
        m([c for s_, c in cx if s_ > 1]), m([c for s_, c in cx if -1 <= s_ <= 1]), m([c for s_, c in cx if s_ < -1])))
for arm in ("script", "brain", "frozen", "sham"):
    v = [r for r in rows if r["arm"] == arm and r["age"] >= 1000]
    print("%-11s n=%d bites %.2f rest %.2f e_stin/e_eat %.2f/%.2f" % (arm, len(v), sum(r["food_found"]/r["age"]*1000 for r in v)/max(1,len(v)),
          sum((r["rest"] or 0) for r in v)/max(1,len(v)), sum((r["e_stin"] or 0) for r in v)/max(1,len(v)), sum((r["e_eat"] or 0) for r in v)/max(1,len(v))))
