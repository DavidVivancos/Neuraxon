import json, sys
H=sys.argv[1]
ts=[json.loads(l) for l in open(H+'/timeseries.jsonl')]
ob=[json.loads(l) for l in open(H+'/obituaries.jsonl')]
lin=[json.loads(l) for l in open(H+'/lineage.jsonl')]
nas=[json.loads(l) for l in open(H+'/nas_trials.jsonl')]
T=ts[-1]['tick']; W=12
print("ticks", T, "days", ts[-1]['uptime_s']/86400)
print("| w | ticks k | hours | TPS | alive | script | mixed | brain | frozen | sham | births | explorers born | deaths | bites/min | M1_E | M1_I | M1_N | M5 | M6 | lesion | Mc | in band |")
for w in range(W):
    lo,hi=T*w/W,T*(w+1)/W
    s=[r for r in ts if lo<r['tick']<=hi]
    hrs=(s[-1]['uptime_s']-s[0]['uptime_s'])/3600; tps=(s[-1]['tick']-s[0]['tick'])/max(1,(s[-1]['uptime_s']-s[0]['uptime_s']))
    al=sum(r['alive'] for r in s)/len(s)
    arm=lambda k: sum(((r.get('arms') or {}).get(k) or {}).get('alive',0) for r in s)/len(s)
    m=lambda k: (lambda v: sum(v)/len(v) if v else float('nan'))([r['m12'][k] for r in s if r.get('m12') and r['m12'].get(k) is not None])
    eat=sum((r.get('rates') or {}).get('food_eaten_per_min',0) for r in s)/len(s)
    b=sum(1 for l in lin if lo<l['born_tick']<=hi); e=sum(1 for n in nas if lo<n['born_tick']<=hi); d=sum(1 for o in ob if lo<o['death_tick']<=hi)
    print(f"| {w+1} | {lo/1000:.0f}–{hi/1000:.0f} | {hrs:.1f} | {tps:.2f} | {al:.0f} | {arm('script'):.0f} | {arm('mixed'):.0f} | {arm('brain'):.0f} | {arm('frozen'):.0f} | {arm('sham'):.0f} | {b} | {e} | {d} | {eat:.0f} | {m('M1_E'):.3f} | {m('M1_I'):.3f} | {m('M1_N'):.3f} | {m('M5_branching'):.3f} | {m('M6_acw_heterogeneity'):.1f} | {m('M10_lesion_retention'):.2f} | {m('Mc_mean'):.3f} | {m('_in_band_count'):.1f} |")
print("perf last:", ts[-1]['perf'])
print("min alive rows:", min(r['alive'] for r in ts[100:]), "max", max(r['alive'] for r in ts))
