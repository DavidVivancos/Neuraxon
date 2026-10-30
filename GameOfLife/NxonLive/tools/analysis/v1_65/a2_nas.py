import json, sys, math
nas=[json.loads(l) for l in open(sys.argv[1])]
def rank(v):
    idx=sorted(range(len(v)),key=lambda i:v[i]); r=[0.0]*len(v); i=0
    while i<len(v):
        j=i
        while j+1<len(v) and v[idx[j+1]]==v[idx[i]]: j+=1
        for k in range(i,j+1): r[idx[k]]=(i+j)/2.0
        i=j+1
    return r
def pear(x,y):
    n=len(x)
    if n<20: return float('nan')
    mx=sum(x)/n; my=sum(y)/n
    sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y); sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    return sxy/math.sqrt(sxx*syy) if sxx>0 and syy>0 else float('nan')
sp=lambda x,y: pear(rank(x),rank(y))
for n in nas:
    L=max(1,n['lifespan']); n['loglife']=math.log(L)
    n['eat']=(n.get('e_eat') or 0)/L*1000; n['stin']=(n.get('e_stin') or 0)/L*1000
    n['cpd']=(n.get('collisions') or 0)/max(1,n.get('dec_n') or 1)
    n['mv']=(n.get('moves') or 0)/L*1000
knobs=sorted(set(k for n in nas for k in n['arch']))
outs=['eat','loglife','stin','motor_frac','out_rest','out_soc','chemotaxis','cpd','mv','m_fit']
for arms,lab in ((('brain','frozen'),'BRAIN+FROZEN (own senses)'),(('sham',),'SHAM (yoked senses)'),(('script',),'SCRIPT'),(('mixed',),'MIXED')):
    sel=[n for n in nas if n.get('arm') in arms and n['lifespan']>=2000 and n.get('m_fit') is not None]
    print(f"\n== {lab}: n={len(sel)}  noise |rho|~{1.96/math.sqrt(max(1,len(sel))):.3f}")
    print(f"{'knob':30s} "+" ".join(f"{o[:8]:>8s}" for o in outs))
    for k in knobs:
        row=[]
        for o in outs:
            pr=[(n['arch'][k],n[o]) for n in sel if n['arch'].get(k) is not None and n.get(o) is not None]
            row.append(sp([a for a,b in pr],[b for a,b in pr]))
        if max(abs(r) for r in row if r==r) >= 0.08:
            print(f"{k:30s} "+" ".join(f"{r:8.3f}" for r in row))
    # behaviour-outcome correlations
    print("  behaviour vs outcomes (rho): ", end="")
    for a,b in (('motor_frac','eat'),('out_rest','loglife'),('stin','loglife'),('eat','loglife'),('chemotaxis','eat'),('cpd','loglife'),('m_fit','loglife'),('m_fit','eat')):
        pr=[(n[a],n[b]) for n in sel if n.get(a) is not None and n.get(b) is not None]
        print(f"{a}-{b} {sp([x for x,y in pr],[y for x,y in pr]):+.3f}  ", end="")
    print()
