import os, sys, random, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'tools'))
sys.argv=['x']
import probe_brain as pb
from server.engine import make_params
from neuraxon.multisphere import build_brain
res={}
for gain in (0.3, 1.0, 2.0, 4.0, 8.0):
    for cpl in (0.5, 3.0):
        vals=[]
        for seed in range(3):
            random.seed(100+seed)
            p=make_params({"cross_sphere_coupling": cpl} if False else None)
            p.sensory_input_gain=gain
            p.cross_sphere_coupling=cpl
            r=pb.probe(None, 12, 40, seed+1, build=lambda: build_brain(p))
            vals.append(r)
        m=lambda k: sum(v[k] for v in vals if v[k] is not None)/len(vals)
        print(f"gain={gain:4.1f} coupling={cpl:3.1f}: MI={m('mi_bits'):.3f} chemo={m('chemo_probe'):+.3f} move={m('move_rate'):.2f} eta2={m('eta2_x'):.3f}/{m('eta2_y'):.3f} amp_sil={m('amp_silent'):.2f} amp_drv={m('amp_driven'):.2f}", flush=True)
