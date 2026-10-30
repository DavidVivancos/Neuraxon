import os, sys, random, json
sys.path.insert(0, os.getcwd())
os.environ.setdefault("NEURAXON_HEADLESS","1")
from server import np_fallback; np_fallback.install()
import architecture; architecture.load_architecture("architecture_files/nas_best.json", verbose=False)
from server.engine import make_params
from neuraxon.multisphere import build_brain, load_multisphere_from_dict
import neuraxon.network as NN
random.seed(5)
p=make_params()
d=build_brain(p).to_dict()
def run(sens, seed, steps=60):
    random.seed(seed)
    b=load_multisphere_from_dict(json.loads(json.dumps(d)), NN._rebuild_net_from_dict)
    ids=[n.id for n in b.sensory_sphere.network.input_neurons]
    hist={sid: [] for sid in b.spheres}
    for t in range(steps):
        b.simulate_step({"sensory": {ids[k]: sens[k] for k in range(len(ids))}})
        for sid, sp in b.spheres.items():
            hist[sid].append(tuple(n.trinary_state for n in sp.network.all_neurons))
    return hist, b
A=[0.8, 1.0, 1.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0]
B=[0.8, -1.0, -1.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0]
Z=[0.0]*10
hA,b=run(A, 7); hB,_=run(B, 7); hZ,_=run(Z, 7)
print("spheres:", list(b.spheres))
for sid in b.spheres:
    nA=len(hA[sid][0])
    diffAB=sum(1 for t in range(60) for x,y in zip(hA[sid][t],hB[sid][t]) if x!=y)
    diffAZ=sum(1 for t in range(60) for x,y in zip(hA[sid][t],hZ[sid][t]) if x!=y)
    first=next((t for t in range(60) if hA[sid][t]!=hB[sid][t]), None)
    print(f"{sid:12s} neurons={nA:4d}  differing neuron-steps A vs B: {diffAB:6d}  A vs zero: {diffAZ:6d}  first divergence step: {first}")
# motor outputs
mo=lambda h: [ [s for s in st] for st in h['motor']]
sn=b.sensory_sphere.network
print("sensory input neurons:", len(sn.input_neurons), "afferent synapses:", sum(len(v) for v in sn._afferent_by_pre.values()) if hasattr(sn,'_afferent_by_pre') else '?')
print("links:", [(l.source_sphere_id, l.target_sphere_id, len(l.source_output_ids), len(l.target_input_ids)) for l in b.links.values()])
mn=b.motor_sphere.network
print("motor outputs ids:", [n.id for n in mn.output_neurons])
