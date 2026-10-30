import os, sys, random, json, math
sys.path.insert(0, os.getcwd())
os.environ.setdefault("NEURAXON_HEADLESS","1")
from server import np_fallback; np_fallback.install()
import architecture; architecture.load_architecture("architecture_files/nas_best.json", verbose=False)
from server.engine import make_params
from neuraxon.multisphere import build_brain, load_multisphere_from_dict
import neuraxon.network as NN
DIRS=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
def sgn(v): return 1 if v>0 else (-1 if v<0 else 0)
def trial(d, gain, steps=40, seed=1, age_steps=0):
    random.seed(seed)
    b=load_multisphere_from_dict(json.loads(json.dumps(d)), NN._rebuild_net_from_dict)
    ids=[n.id for n in b.sensory_sphere.network.input_neurons]
    outs=[n.id for n in b.motor_sphere.network.output_neurons]
    # optionally age the brain first (synaptic collapse) with random input
    for t in range(age_steps):
        s=[random.uniform(-1,1) for _ in range(10)]
        b.simulate_step({"sensory": {ids[k]: s[k] for k in range(10)}})
    cs=[]; mv=0; tot=0
    for ci,(fx,fy) in enumerate(DIRS):
        sens=[0.8,fx,fy,0,1,0.25,0,0,0,0]
        for t in range(steps):
            ext={"sensory": {ids[k]: sens[k] for k in range(10)}}
            if gain>0:
                ext["motor"]={outs[0]: gain*fx, outs[1]: gain*fy}
            b.simulate_step(ext)
            o=b.motor_sphere.network.get_output_states()
            mx,my=sgn(o[0]),sgn(o[1]); tot+=1
            if mx or my:
                mv+=1; cs.append((mx*fx+my*fy)/math.sqrt((mx*mx+my*my)*(fx*fx+fy*fy)))
    return sum(cs)/max(1,len(cs)), mv/tot
for seed in (1,2,3):
    random.seed(10+seed); d=build_brain(make_params()).to_dict()
    for age in (0, 400):
        row=[]
        for g in (0, 0.5, 1, 2, 4):
            c,m=trial(d,g,seed=seed,age_steps=age)
            row.append(f"g={g}: chemo {c:+.2f} move {m:.2f}")
        print(f"brain {seed} aged {age:3d} steps | "+" | ".join(row), flush=True)
