# Metabolic cost of standing vs resting vs moving in V1.085 (idle-atrophy ramp)
import json, sys
ob = [json.loads(l) for l in open(sys.argv[1])]
BASE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
print("gross metabolism (drain + M-relief) per 1000 ticks, in multiples of the base drain (%.3f/tick), dead NxErs lived >= 2000" % BASE)
print("arm      rest<0.1 (n)        0.1-0.5 (n)        0.5-0.9 (n)        >=0.9 (n)")
for arm in ("script", "mixed", "brain", "frozen", "sham"):
    row = []
    for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)):
        v = [o for o in ob if o.get("arm") == arm and o["lifespan_ticks"] >= 2000 and lo <= (o.get("out_rest") or 0) < hi]
        if not v:
            row.append("      -        "); continue
        T = sum(o["lifespan_ticks"] for o in v)
        g = sum((o.get("e_drain") or 0) + (o.get("e_relief") or 0) for o in v) / T / BASE
        row.append("%6.2fx (%5d)" % (g, len(v)))
    print("%-8s %s" % (arm, "   ".join(row)))
# brain arm by motor_frac (moving a lot = less idle)
print("\nbrain+sham arms, rest < 0.1, by motor_frac:")
for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
    v = [o for o in ob if o.get("arm") in ("brain", "sham") and o["lifespan_ticks"] >= 2000 and (o.get("out_rest") or 0) < 0.1 and lo <= (o.get("motor_frac") or 0) < hi]
    if v:
        T = sum(o["lifespan_ticks"] for o in v)
        print("  motor %.1f-%.1f: %5.2fx base (n=%d)" % (lo, hi, sum((o.get("e_drain") or 0) + (o.get("e_relief") or 0) for o in v) / T / BASE, len(v)))
