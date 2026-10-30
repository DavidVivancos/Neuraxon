import json, sys, math
def mean(v): v=[x for x in v if x is not None]; return (sum(v)/len(v)) if v else float('nan')
for path in sys.argv[1:]:
    d = json.load(open(path))
    print("==", path.split('/')[-1], "checkpoints", len(d["snaps"]), "deaths", len(d["obits"]))
    for s in d["snaps"]:
        line = []
        for arm in ("script","brain","frozen","sham","reflex","reflex_sham"):
            v = [r for r in s["live"] if r["arm"] == arm]
            if not v: continue
            cx = mean([r["cx"] for r in v if (r["cxn"] or 0) >= 20])
            bites = mean([r["food_found"]/max(r["age"],500)*1000 for r in v])
            line.append("%s n=%d cx=%+.2f bites=%.1f" % (arm[:6], len(v), cx, bites))
        print(" t=%5d alive=%3d | " % (s["tick"], s["alive"]) + " | ".join(line))
    s = d["snaps"][-1]
    rf = [r for r in s["live"] if r["arm"] == "reflex" and r["reflex"]]
    if rf:
        sc = [(r["reflex"]["rfx_gain"]*(r["reflex"]["rfx_xx"]+r["reflex"]["rfx_yy"]), r["cx"]) for r in rf if r["cx"] is not None and (r["cxn"] or 0)>=20]
        if len(sc) > 5:
            xs=[a for a,b in sc]; ys=[b for a,b in sc]; mx=mean(xs); my=mean(ys)
            r = sum((a-mx)*(b-my) for a,b in sc)/math.sqrt(sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))
            print("  reflex arm: genome chemo score vs executed chemotaxis r=%.2f (n=%d)" % (r, len(sc)))
    allr = [r for r in s["live"] if r["reflex"]]
    if allr:
        print("  population genes: xx %+.3f yy %+.3f gain %.3f share(xx>0&yy>0) %.2f" % (
            mean([r["reflex"]["rfx_xx"] for r in allr]), mean([r["reflex"]["rfx_yy"] for r in allr]),
            mean([r["reflex"]["rfx_gain"] for r in allr]),
            sum(1 for r in allr if r["reflex"]["rfx_xx"]>0 and r["reflex"]["rfx_yy"]>0)/len(allr)))
    ob = d["obits"]
    if ob:
        for arm in ("script","brain","frozen","sham","reflex","reflex_sham"):
            v=[o for o in ob if o.get("arm")==arm]
            if v: print("  deaths %-11s n=%3d medL=%6d starved=%.2f" % (arm, len(v), sorted(o["lifespan_ticks"] for o in v)[len(v)//2],
                         sum(1 for o in v if o.get("cause")=="starvation")/len(v)))
