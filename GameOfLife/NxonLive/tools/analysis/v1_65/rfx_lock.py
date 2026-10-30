# Does a strong reflex current LOCK the motor output, and is plasticity the cause?
import os, sys, random, json, math
sys.path.insert(0, os.getcwd()); os.environ.setdefault("NEURAXON_HEADLESS", "1")
from server import np_fallback; np_fallback.install()
import architecture; architecture.load_architecture("architecture_files/nas_best.json", verbose=False)
from server.engine import make_params
from server.brain_pool import _freeze_brain
from neuraxon.multisphere import build_brain, load_multisphere_from_dict
import neuraxon.network as NN
DIRS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
sg = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)
def load(d): return load_multisphere_from_dict(json.loads(json.dumps(d)), NN._rebuild_net_from_dict)
def age(d, steps, seed):
    random.seed(seed); b = load(d); ids = [n.id for n in b.sensory_sphere.network.input_neurons]
    for t in range(steps):
        s = [random.uniform(-1, 1) for _ in range(10)]
        b.simulate_step({"sensory": {ids[k]: s[k] for k in range(10)}})
    return b.to_dict()
def trial(d, g, frozen, seed, steps=320, switch=4):
    random.seed(seed); b = load(d)
    if frozen: _freeze_brain(b, keep_homeostasis=True)
    ids = [n.id for n in b.sensory_sphere.network.input_neurons]
    outs = [n.id for n in b.motor_sphere.network.output_neurons]
    rnd = random.Random(seed + 5); cs = []; mv = 0
    for t in range(steps):
        if t % switch == 0: fx, fy = rnd.choice(DIRS)
        sens = [0.8, fx, fy, 0, 1, 0.25, 0, 0, 0, 0]
        ext = {"sensory": {ids[k]: sens[k] for k in range(10)}}
        if g > 0: ext["motor"] = {outs[0]: g * fx, outs[1]: g * fy}
        b.simulate_step(ext)
        o = b.motor_sphere.network.get_output_states(); mx, my = sg(o[0]), sg(o[1])
        if mx or my:
            mv += 1; cs.append((mx*fx + my*fy) / math.sqrt((mx*mx + my*my) * (fx*fx + fy*fy)))
    q = len(cs) // 4
    return (sum(cs)/len(cs) if cs else 0.0), mv/steps, (sum(cs[-q:])/q if q else 0.0)
for seed in (1, 2, 3):
    random.seed(10 + seed); d0 = build_brain(make_params()).to_dict()
    for ag in (0, 400):
        d = age(d0, ag, seed) if ag else d0
        for frozen in (False, True):
            row = []
            for g in (0.0, 0.5, 1.0, 2.0, 4.0):
                c, m, late = trial(d, g, frozen, seed)
                row.append("g=%.1f chemo %+.2f (last quarter %+.2f) move %.2f" % (g, c, late, m))
            print("brain %d aged %3d %-7s | %s" % (seed, ag, "frozen" if frozen else "plastic", " | ".join(row)), flush=True)
