import json, sys
from collections import defaultdict
rows=[]
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except: continue
    rows.append(d)
print("rows", len(rows), "keys", sorted(rows[-1].keys())[:60])
bins=[0,1000,2000,4000,8000,32000,128000,10**9]
B=defaultdict(list)
for d in rows:
    a=d['age']
    for i in range(len(bins)-1):
        if bins[i]<=a<bins[i+1]:
            B[(d.get('arm'),i)].append(d); break
print("\nsynapses p50 / motor / lock / rest / E / N / M5 / M6 / m_fit by arm x age")
for arm in ("script","mixed","brain","frozen","sham"):
    print(" ",arm)
    for i in range(len(bins)-1):
        v=B.get((arm,i),[])
        if len(v)<20: continue
        sy=sorted(x.get('W_n_syn_total') or x.get('W_n') or 0 for x in v)
        m=lambda k: sum(x.get(k) or 0 for x in v if x.get(k) is not None)/max(1,len([x for x in v if x.get(k) is not None]))
        print(f"    age {bins[i]:>6}-{bins[i+1]:<10} n={len(v):5d} syn={sy[len(sy)//2]:5d} motor={m('motor_frac'):.2f} lock={m('lock_frac'):.2f} rest={m('out_rest'):.2f} soc={m('out_soc'):.2f} E={m('M1_E'):.3f} N={m('M1_N'):.3f} M5={m('M5_branch'):.3f} M6={m('M6_acw'):.1f} mfit={m('m_fit'):.3f} food={m('food'):.0f}")
# band pass per arm (all ages >= 4000)
print("\nper-brain band pass (age>=4000) by arm")
for arm in ("script","mixed","brain","frozen","sham"):
    v=[d for d in rows if d.get('arm')==arm and d['age']>=4000]
    f=lambda k,lo,hi: sum(1 for d in v if d.get(k) is not None and lo<=d[k]<=hi)/max(1,len(v))
    print(f"  {arm:7s} n={len(v):6d} E {f('M1_E',0.18,0.28):.2f} N {f('M1_N',0.60,0.78):.2f} M5 {f('M5_branch',0.92,1.10):.2f} M6 {f('M6_acw',3,60):.2f} gate {f('M2_gate',0.40,0.85):.2f} mfit {sum(d.get('m_fit') or 0 for d in v)/max(1,len(v)):.3f}")
