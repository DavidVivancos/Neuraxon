import json, sys
snap = json.load(open(sys.argv[1])); T = snap["tick"]
alive = [x for x in snap["nxers"] if x.get("alive", True)]
last = {}
for l in open(sys.argv[2]):
    r = json.loads(l)
    last[r.get("id")] = r
f = [x for x in alive if x.get("born_tick", 0) < 1000]
print("alive", len(alive), "founders alive (born<1000)", len(f), "arms", {a: sum(1 for x in f if x.get("arm") == a) for a in set(x.get("arm") for x in f)})
import statistics as S
def g(x, k):
    r = last.get(x["id"]); return r.get(k) if r else None
for k in ("out_rest", "out_soc", "lock_frac", "motor_frac", "W_n_syn_total", "M1_E", "M5_branch", "e_stin", "e_eat", "e_drain"):
    v = [g(x, k) for x in f if g(x, k) is not None]
    if v: print("  founders %-14s median %.3f (n=%d)" % (k, S.median(v), len(v)))
tk = sum(x["stats"]["food_taken"] for x in f); fd = sum(x["stats"]["food_found"] for x in f)
print("  founders total found %d taken %d (taken/found %.0fx)" % (fd, tk, tk / max(1, fd)))
young = [x for x in alive if x.get("born_tick", 0) >= 1000]
print("  non-founders alive", len(young), "median age", S.median([T - x["born_tick"] for x in young]) if young else None,
      "found", sum(x["stats"]["food_found"] for x in young), "taken", sum(x["stats"]["food_taken"] for x in young))
by = {}
for x in alive:
    by.setdefault(x.get("arm"), []).append(T - x["born_tick"])
print("  alive by arm (n, median age):", {a: (len(v), int(S.median(v))) for a, v in by.items()})
