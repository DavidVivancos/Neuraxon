import os, sys, json, random, functools
print=functools.partial(__builtins__.print, flush=True)
os.environ["NEURAXON_HEADLESS"]="1"; sys.path.insert(0, os.getcwd())
from server import np_fallback; np_fallback.install()
import architecture; architecture.load_architecture("architecture_files/nas_best.json", verbose=False)
from server.engine import make_params, _MUTABLE, _ARCH_ALIAS, _params_to_dict
from neuraxon.multisphere import build_brain, load_multisphere_from_dict
import neuraxon.network as NN
random.seed(5)
arch={k: round(random.uniform(lo,hi),5) for k,(lo,hi) in _MUTABLE.items()}
p=make_params(arch)
print("=== GENE ROUND-TRIP: does every searched knob survive save -> load? ===")
b=build_brain(p)
d=b.to_dict()
b2=load_multisphere_from_dict(d, NN._rebuild_net_from_dict)
sph=list(b2.spheres.values())[0]
got=sph.network.params
bad=[]
for k in _MUTABLE:
    a=_ARCH_ALIAS.get(k,k)
    want=getattr(p,a,None); have=getattr(got,a,None)
    if want is None: bad.append((k,'unset','-')); continue
    if have is None or abs(float(have)-float(want))>1e-6:
        bad.append((k,round(float(want),5), have if have is None else round(float(have),5)))
print("  knobs checked: %d | SURVIVED: %d | LOST: %d"%(len(_MUTABLE),len(_MUTABLE)-len(bad),len(bad)))
for k,w,h in bad: print("    LOST %-32s wanted=%s got=%s"%(k,w,h))
print("  => %s"%("ALL GENES SURVIVE" if not bad else "*** %d GENES DO NOT SURVIVE SAVE->LOAD ***"%len(bad)))
