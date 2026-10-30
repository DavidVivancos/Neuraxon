"""v1.66 — CLOSED-LOOP LEARNING TEST: can a Neuraxon brain LEARN to move
toward food when reward depends on its own steps?

A single brain lives on an empty, obstacle-free plane with one food item at
a random offset (2-6 cells). Each brain step it receives the engine's exact
10-channel sensory vector, its (mvx, mvy) output moves it one cell, and on
reaching the food it eats and a new food item appears. Dopamine is applied
on the NEXT brain step exactly as the server does (level = baseline + banked
reward, capped). Conditions:

  none     no reward
  eat      +reward_eat on eating only (what the server did up to v1.65)
  shaped   eat reward + an APPROACH reward each step the brain moves toward
           the food (cosine > 0), the v1.66 'shaped' arm
  sham     shaped reward computed on the true food, but the brain's food
           direction input is scrambled each step (it cannot learn the map)
  reflex   (v1.66 test) no reward; an innate chemotactic reflex current
           (gain --gain) is injected into the motor OUTPUT neurons of the
           brain (mvx, mvy) from the food direction, as the V1.086 arc
           first did inside the brain
  body     (v1.66) no reward; the same reflex acts at BODY level: the step is
           sign(brain step + gain * food direction) per axis, i.e. a
           spinal-style reflex arc summed with the brain's command

Reported per block of steps: chemotaxis (mean cosine of the brain's step with
the food direction, moves only), move rate, and food eaten. Learning shows up
as chemotaxis rising across blocks in 'shaped' and not in 'sham'.

Usage (from the mngol5 folder):
  python3 tools/learn_test.py --brains 3 --steps 3000
  python3 tools/learn_test.py --conditions shaped sham --approach 0.3
  python3 tools/learn_test.py --phasic          # reward reaches AGMP (v1.66)
The vendored substrate is used read-only.
"""
import os
import sys
import json
import math
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("NEURAXON_HEADLESS", "1")
from server import np_fallback  # noqa: E402
np_fallback.install()
import architecture  # noqa: E402
try:
    architecture.load_architecture(
        os.path.join(ROOT, "architecture_files", "nas_best.json"), verbose=False)
except Exception:
    architecture._ARCH = {}
from neuraxon.multisphere import build_brain, load_multisphere_from_dict  # noqa: E402
from server import brain_fix as BF  # noqa: E402
_REBUILD = BF.rebuild_net_exact   # v1.67 — exact-shape loader (see brain_fix)
from server.engine import make_params, _MUTABLE  # noqa: E402

DIRS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]


def _sign(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _set_da(b, level, phasic):
    """Apply the banked reward. set_global_modulator only writes the dopamine
    LEVEL; the network re-attributes it to the TONIC pool (total - phasic),
    while AGMP consolidates eligibility traces with the PHASIC pool. With
    phasic=True the reward is also written into the phasic pool of every
    sphere's neuromodulator system, so AGMP actually sees it."""
    try:
        b.set_global_modulator("dopamine", level)
    except Exception:
        pass
    if phasic is not None:
        for sp in b.spheres.values():
            ns = getattr(sp.network, "neuromod_system", None)
            if ns is not None and "DA" in getattr(ns, "levels", {}):
                ns.levels["DA"]["phasic"] = phasic


def run(bdict, cond, steps, block, seed, approach, eat_da, da_base, da_max, phasic=False,
        gain=1.0):
    rnd = random.Random(seed)
    random.seed(seed)
    b = load_multisphere_from_dict(json.loads(json.dumps(bdict)),
                                   rebuild_net_fn=_REBUILD)
    ids = [n.id for n in b.sensory_sphere.network.input_neurons]
    oids = [n.id for n in b.motor_sphere.network.output_neurons]
    x = y = 0
    heading = 0

    def new_food():
        while True:
            d = rnd.randint(2, 6)
            a = rnd.random() * 2 * math.pi
            fx, fy = x + int(round(d * math.cos(a))), y + int(round(d * math.sin(a)))
            if (fx, fy) != (x, y):
                return fx, fy
    fx, fy = new_food()
    bank = 0.0
    out = []
    acc = {"cos": 0.0, "n": 0, "mv": 0, "eat": 0, "steps": 0}
    for t in range(steps):
        dx, dy = fx - x, fy - y
        d2 = dx * dx + dy * dy
        sfx, sfy = _sign(dx), _sign(dy)
        if cond == "sham":
            sfx, sfy = rnd.choice(DIRS)          # scrambled food direction
        sens = [0.8, float(sfx), float(sfy), 0.0,
                1.0 if d2 < 25 else 0.0, 1.0 / (1.0 + math.sqrt(d2)),
                0.0, heading / 7.0, 0.0, 0.0]
        if bank > 0.0:
            _set_da(b, da_base + min(bank, da_max), min(bank, da_max) if phasic else None)
            bank = 0.0
        ext = {"sensory": {ids[k]: sens[k] for k in range(len(ids))}}
        if cond == "reflex" and len(oids) >= 2:
            ext["motor"] = {oids[0]: gain * sens[1], oids[1]: gain * sens[2]}
        b.simulate_step(ext)
        o = list(b.motor_sphere.network.get_output_states()) + [0] * 7
        mx, my = _sign(o[0]), _sign(o[1])
        if cond == "body":
            vx, vy = o[0] + gain * sens[1], o[1] + gain * sens[2]
            mx = 1 if vx > 0.5 else (-1 if vx < -0.5 else 0)
            my = 1 if vy > 0.5 else (-1 if vy < -0.5 else 0)
        acc["steps"] += 1
        if mx or my:
            acc["mv"] += 1
            tx, ty = _sign(dx), _sign(dy)          # TRUE direction
            if tx or ty:
                c = (mx * tx + my * ty) / math.sqrt((mx * mx + my * my) * (tx * tx + ty * ty))
                acc["cos"] += c
                acc["n"] += 1
                if cond in ("shaped", "sham") and c > 0:
                    bank += approach * c
            x += mx
            y += my
            if (mx, my) in DIRS:
                heading = DIRS.index((mx, my))
        if (x, y) == (fx, fy):
            acc["eat"] += 1
            if cond in ("eat", "shaped", "sham"):
                bank += eat_da
            fx, fy = new_food()
        if acc["steps"] == block:
            out.append({"chemo": round(acc["cos"] / acc["n"], 4) if acc["n"] else None,
                        "move_rate": round(acc["mv"] / block, 3),
                        "eaten": acc["eat"]})
            acc = {"cos": 0.0, "n": 0, "mv": 0, "eat": 0, "steps": 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--approach", type=float, default=0.3,
                    help="dopamine banked per approach step (times cosine)")
    ap.add_argument("--eat", type=float, default=0.5)
    ap.add_argument("--conditions", nargs="+",
                    default=["none", "eat", "shaped", "sham"])
    ap.add_argument("--random-arch", action="store_true",
                    help="sample each brain's architecture uniformly from the NAS space")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="reflex gain for the reflex / body conditions")
    ap.add_argument("--phasic", action="store_true",
                    help="also write the reward into the PHASIC dopamine pool (what AGMP reads)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    results = []
    for bi in range(a.brains):
        random.seed(a.seed * 1000 + bi)
        arch = ({k: random.uniform(lo, hi) for k, (lo, hi) in _MUTABLE.items()}
                if a.random_arch else None)
        bdict = build_brain(make_params(arch)).to_dict()
        for cond in a.conditions:
            blocks = run(bdict, cond, a.steps, a.block, a.seed * 7919 + bi,
                         a.approach, a.eat, 0.15, 0.85, a.phasic, a.gain)
            r = {"brain": bi, "cond": cond, "blocks": blocks}
            results.append(r)
            print(json.dumps(r), flush=True)
    print("\n== mean chemotaxis per block (across brains)")
    for cond in a.conditions:
        rs = [r for r in results if r["cond"] == cond]
        nb = len(rs[0]["blocks"]) if rs else 0
        line = []
        for k in range(nb):
            v = [r["blocks"][k]["chemo"] for r in rs if r["blocks"][k]["chemo"] is not None]
            line.append("%+.3f" % (sum(v) / len(v)) if v else "  -  ")
        eat = sum(sum(bl["eaten"] for bl in r["blocks"]) for r in rs) / max(1, len(rs))
        mv = sum(sum(bl["move_rate"] for bl in r["blocks"]) / max(1, len(r["blocks"])) for r in rs) / max(1, len(rs))
        print("  %-7s %s   eaten/brain %.1f   move rate %.2f" % (cond, " ".join(line), eat, mv))
    if a.json:
        json.dump(results, open(a.json, "w"))


if __name__ == "__main__":
    main()
