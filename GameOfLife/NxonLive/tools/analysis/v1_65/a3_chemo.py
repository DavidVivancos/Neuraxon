import json, sys, math
ob=[json.loads(l) for l in open(sys.argv[1])]
def st(xs):
    xs=[x for x in xs if x is not None]
    if len(xs)<5: return "n=%d"%len(xs)
    m=sum(xs)/len(xs); sd=math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))
    return f"n={len(xs):5d} mean={m:+.4f} se={sd/math.sqrt(len(xs)):.4f} t={m/(sd/math.sqrt(len(xs))):+.1f}"
for o in ob:
    o['cpd']=(o.get('collisions') or 0)/max(1,o.get('dec_n') or 1)
    o['wpd']=(o.get('coll_wall') or 0)/max(1,o.get('dec_n') or 1)
    L=max(1,o['lifespan_ticks']); o['bites']=o['food_found']/L*1000
print("chemotaxis by arm and collision-per-decision band (chemo_n>=50)")
for arm in ("script","mixed","brain","frozen","sham"):
    for lo,hi in ((0,0.02),(0.02,0.1),(0.1,0.5),(0.5,99)):
        v=[o['chemotaxis'] for o in ob if o.get('arm')==arm and o.get('chemotaxis') is not None and (o.get('chemo_n') or 0)>=50 and lo<=o['cpd']<hi]
        print(f"  {arm:7s} coll/dec [{lo},{hi}) {st(v)}")
print("\nbites per 1000 ticks (food_found, excludes M-bonus) vs m_fit tercile, per arm (lived>=2000)")
for arm in ("script","mixed","brain","frozen","sham"):
    v=sorted([o for o in ob if o.get('arm')==arm and o['lifespan_ticks']>=2000 and o.get('m_fit') is not None], key=lambda o:o['m_fit'])
    n=len(v); out=[]
    for t in range(3):
        g=v[t*n//3:(t+1)*n//3]
        med=lambda k: sorted(o[k] for o in g)[len(g)//2]
        out.append(f"T{t+1} mfit {sum(o['m_fit'] for o in g)/len(g):.3f} bites {sum(o['bites'] for o in g)/len(g):.2f} medL {med('lifespan_ticks')} cpd {sum(o['cpd'] for o in g)/len(g):.3f} rest {sum(o.get('out_rest') or 0 for o in g)/len(g):.2f} motor {sum(o.get('motor_frac') or 0 for o in g)/len(g):.2f}")
    print(f"  {arm:7s} | "+" | ".join(out))
# fraction of brain-arm NxErs that ever ate
print("\nshare that ate at least once / >=10 bites (lived>=5000)")
for arm in ("script","mixed","brain","frozen","sham"):
    v=[o for o in ob if o.get('arm')==arm and o['lifespan_ticks']>=5000]
    print(f"  {arm:7s} n={len(v)} ate>=1 {sum(1 for o in v if o['food_found']>=1)/len(v):.3f}  >=10 {sum(1 for o in v if o['food_found']>=10)/len(v):.3f}  >=100 {sum(1 for o in v if o['food_found']>=100)/len(v):.3f}")
