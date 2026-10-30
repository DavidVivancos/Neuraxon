import json, sys
snap = json.load(open(sys.argv[1])); T = snap["tick"]
ob = [json.loads(l) for l in open(sys.argv[2])]
rows = []
for x in snap["nxers"]:
    if not x.get("alive", True): continue
    st = x.get("stats", {}); b = x.get("behav") or {}
    rows.append(dict(name=x.get("name"), arm=x.get("arm"), alive=True, L=T - x.get("born_tick", 0),
                     found=st.get("food_found"), taken=st.get("food_taken"),
                     rest=b.get("out_rest"), soc=b.get("out_soc"), lock=b.get("lock_frac"),
                     explorer=x.get("explorer"), e_stin=b.get("e_stin"), e_eat=b.get("e_eat")))
for o in ob:
    rows.append(dict(name=o.get("name"), arm=o.get("arm"), alive=False, L=o["lifespan_ticks"], found=o.get("food_found"),
                     taken=o.get("food_taken"), rest=o.get("out_rest"), soc=o.get("out_soc"), lock=o.get("lock_frac"),
                     explorer=o.get("explorer"), e_stin=o.get("e_stin"), e_eat=o.get("e_eat")))
rows.sort(key=lambda r: -r["L"])
print("snapshot tick", T, "alive", sum(1 for r in rows if r["alive"]))
print("keys sample nxer:", sorted(snap["nxers"][0].keys())[:60])
print("top 15 lifespans (alive and dead):")
for r in rows[:15]:
    print("  %-5s %-7s %-5s L=%7d found=%5s taken=%5s rest=%s soc=%s lock=%s e_eat=%s e_stin=%s" % (
        r["name"], r["arm"], "alive" if r["alive"] else "dead", r["L"], r["found"], r["taken"], r["rest"], r["soc"], r["lock"], r["e_eat"], r["e_stin"]))
# share of the 50 longest that are rest>=0.8 and taken>found
top = rows[:50]
print("top50: rest>=0.8 %d, taken>found %d, arms %s" % (sum(1 for r in top if (r["rest"] or 0) >= 0.8),
      sum(1 for r in top if (r["taken"] or 0) > (r["found"] or 0)),
      {a: sum(1 for r in top if r["arm"] == a) for a in set(r["arm"] for r in top)}))
