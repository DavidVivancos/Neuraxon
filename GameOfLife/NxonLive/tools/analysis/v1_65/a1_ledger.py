import json, sys, math
from collections import defaultdict
ob=[json.loads(l) for l in open(sys.argv[1])]
ARMS=("script","mixed","brain","frozen","sham")
K=["e_eat","e_bonus","e_stin","e_stout","e_gvin","e_gvout","e_drain","e_relief","e_coll","e_mate","e_cap"]
def cls(L):
    return "a<5k" if L<5000 else ("b5-50k" if L<50000 else ("c50-200k" if L<200000 else "d>200k"))
print("== energy ledger SUMS per lifespan class (food units per 1000 ticks lived, pooled), dead NxErs")
print("arm     class      n   eat  (bonus) stin gvin | drain  coll  stout gvout mate cap | net  start/L  | bites/1kt moves/1kt coll/dec rest  soc  motor lock")
for arm in ARMS:
    for c in ("a<5k","b5-50k","c50-200k","d>200k"):
        v=[o for o in ob if o.get('arm')==arm and cls(o['lifespan_ticks'])==c]
        if not v: continue
        T=sum(max(1,o['lifespan_ticks']) for o in v)/1000
        s={k:sum(o.get(k) or 0 for o in v)/T for k in K}
        start=len(v)*120/T
        net=s['e_eat']+s['e_stin']+s['e_gvin']-s['e_drain']-s['e_coll']-s['e_stout']-s['e_gvout']-s['e_mate']-s['e_cap']
        bites=sum(o['food_found'] for o in v)/T; mv=sum(o.get('moves') or 0 for o in v)/T
        cd=sum(o.get('collisions') or 0 for o in v)/max(1,sum(o.get('dec_n') or 0 for o in v))
        m=lambda k: sum(o.get(k) or 0 for o in v)/len(v)
        print(f"{arm:7s} {c:8s} {len(v):5d} {s['e_eat']:5.2f} ({s['e_bonus']:4.2f}) {s['e_stin']:4.2f} {s['e_gvin']:4.2f} | {s['e_drain']:5.2f} {s['e_coll']:5.2f} {s['e_stout']:5.2f} {s['e_gvout']:5.2f} {s['e_mate']:4.2f} {s['e_cap']:4.2f} | {net:+6.2f} {start:6.2f} | {bites:6.2f} {mv:7.1f} {cd:6.3f} {m('out_rest'):.2f} {m('out_soc'):.2f} {m('motor_frac'):.2f} {m('lock_frac'):.2f}")
print()
print("== collisions by kind (pooled per 1000 ticks)")
for arm in ARMS:
    v=[o for o in ob if o.get('arm')==arm]
    T=sum(max(1,o['lifespan_ticks']) for o in v)/1000
    print(f"{arm:7s} wall {sum(o.get('coll_wall') or 0 for o in v)/T:7.2f}  nx {sum(o.get('coll_nx') or 0 for o in v)/T:7.2f}  thefts_brain {sum(o.get('n_st_brain') or 0 for o in v)/T:5.2f} floor {sum(o.get('n_st_floor') or 0 for o in v)/T:5.2f} gives {sum(o.get('n_give') or 0 for o in v)/T:5.2f} matefloor {sum(o.get('n_mate_floor') or 0 for o in v)/T:5.3f}  offspring/100kt {sum(o['n_offspring'] for o in v)/T*100:6.2f}")
