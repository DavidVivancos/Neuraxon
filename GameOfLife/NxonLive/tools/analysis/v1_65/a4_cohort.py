import json, sys, math
from collections import defaultdict
ob=[json.loads(l) for l in open(sys.argv[1])]
T=751393; W=8
print("cohort (birth window) trends, dead NxErs lived>=2000: bites/1kt (mean), medL, motor, rest, soc, lock, chemo(chemo_n>=50), share explorers")
for arm in ("script","mixed","brain","frozen","sham"):
    print(" ",arm)
    for w in range(W):
        lo,hi=T*w/W,T*(w+1)/W
        g=[o for o in ob if o.get('arm')==arm and lo<=o['born_tick']<hi and o['lifespan_ticks']>=2000]
        if len(g)<20: continue
        b=sum(o['food_found']/o['lifespan_ticks']*1000 for o in g)/len(g)
        L=sorted(o['lifespan_ticks'] for o in g)[len(g)//2]
        m=lambda k: sum(o.get(k) or 0 for o in g)/len(g)
        ch=[o['chemotaxis'] for o in g if o.get('chemotaxis') is not None and (o.get('chemo_n') or 0)>=50]
        ex=sum(1 for o in g if o.get('explorer'))/len(g)
        print(f"    born {lo/1000:5.0f}-{hi/1000:5.0f}k n={len(g):4d} bites={b:5.2f} medL={L:7d} motor={m('motor_frac'):.2f} rest={m('out_rest'):.2f} soc={m('out_soc'):.2f} lock={m('lock_frac'):.2f} chemo={sum(ch)/max(1,len(ch)):+.3f} explorers={ex:.2f}")
# explorers vs evolved within brain arm
print("\nexplorer vs born-by-mating, brain-driven arms (lived>=2000)")
for arm in ("brain","frozen","sham","mixed","script"):
    for ex in (True, False):
        g=[o for o in ob if o.get('arm')==arm and o.get('explorer')==ex and o['lifespan_ticks']>=2000]
        if not g: continue
        b=sum(o['food_found']/o['lifespan_ticks']*1000 for o in g)/len(g)
        L=sorted(o['lifespan_ticks'] for o in g)[len(g)//2]
        print(f"  {arm:7s} {'explorer' if ex else 'mated   '} n={len(g):5d} bites={b:5.2f} medL={L:7d} mfit={sum(o.get('m_fit') or 0 for o in g)/len(g):.3f} off={sum(o['n_offspring'] for o in g)/len(g):.2f}")
