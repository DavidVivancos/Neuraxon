"""v1.65 — OPEN-LOOP BRAIN PROBE: is a brain's motor output coupled to its
input, or is it noise?

In the live world a brain's output is entangled with everything else
(scripted moves, crowding, luck). This tool takes an archived brain out of
the world and asks it controlled questions: food is placed in each of the 8
directions (and nowhere), the matching sensory vector is clamped, and the
motor outputs are recorded. If the brain is sensorimotor-coupled, its motor
step depends on where the food is; if it is not, the output is the same
whatever the input and the measured "chemotaxis" is zero.

Reported per brain:
  chemo_probe   mean cosine between the brain's step and the food direction
                over all steps where it moved while food was present
                (+1 = always toward food, 0 = unrelated, -1 = away)
  mi_bits       mutual information I(food direction ; motor direction) in
                bits, minus the mean of 20 label-shuffled controls
                (bias-corrected; 0 = output carries no information about
                where the food is). Max possible log2(9) = 3.17.
  move_rate     fraction of steps with a non-zero step
  eta2_x/eta2_y share of the variance of mvx / mvy explained by the
                food-direction condition (0 = input-blind)
  silent_vs_drv motor amplitude with NO input vs with food present
  newborn_*     the same probe on a NEWBORN brain built from the same genes
                (--newborn) — tells whether life experience changed coupling

Usage (from the mngol5 folder):
  python3 tools/probe_brain.py state/best/brains/*.json state/best/elite/*.json
  python3 tools/probe_brain.py --newborn --steps 64 file1.json ...
  python3 tools/probe_brain.py --csv out.csv files...

The vendored substrate is used read-only; nothing in the world changes.
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
from neuraxon.multisphere import load_multisphere_from_dict, build_brain  # noqa: E402
try:
    from neuraxon.network import _rebuild_net_from_dict as _REBUILD  # noqa: E402
except Exception:
    _REBUILD = None

DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def sensory(fd, hunger=0.8):
    """The engine's 10-channel sensory vector (engine._sense) for food at
    distance ~3 in direction fd, or no food when fd is None."""
    if fd is None:
        return [hunger, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [hunger, float(fd[0]), float(fd[1]), 0.0, 1.0, 0.25,
            0.0, 0.0, 0.0, 0.0]


def _load(bdict):
    return load_multisphere_from_dict(bdict, rebuild_net_fn=_REBUILD)


def _run(bdict, sens, warm, steps, seed, build=None):
    random.seed(seed)
    b = build() if build is not None else _load(bdict)
    ids = [n.id for n in b.sensory_sphere.network.input_neurons]
    ext = {"sensory": {ids[k]: (sens[k] if k < len(sens) else 0.0)
                       for k in range(len(ids))}}
    out = []
    for t in range(warm + steps):
        b.simulate_step(ext)
        if t >= warm:
            o = list(b.motor_sphere.network.get_output_states())
            o = (o + [0] * 7)[:7]
            out.append(o)
    return out


def _mi(pairs, n_cls_a, n_cls_b):
    n = len(pairs)
    if n == 0:
        return 0.0
    ca = [0] * n_cls_a
    cb = [0] * n_cls_b
    cab = {}
    for a, b in pairs:
        ca[a] += 1
        cb[b] += 1
        cab[(a, b)] = cab.get((a, b), 0) + 1
    mi = 0.0
    for (a, b), c in cab.items():
        p = c / n
        mi += p * math.log2(p / ((ca[a] / n) * (cb[b] / n)))
    return mi


def _dir_class(mx, my):
    mx = int(max(-1, min(1, mx)))
    my = int(max(-1, min(1, my)))
    if mx == 0 and my == 0:
        return 8
    return DIRS.index((mx, my))


def probe(bdict, warm=16, steps=48, seed=1, build=None):
    """Return the probe metrics for one brain."""
    per = {}
    for ci, fd in enumerate(DIRS):
        per[ci] = _run(bdict, sensory(fd), warm, steps, seed + ci, build)
    silent = _run(bdict, sensory(None), warm, steps, seed + 99, build)
    cos_sum = 0.0
    cos_n = 0
    moves = 0
    tot = 0
    pairs = []
    xs_by = {}
    ys_by = {}
    amp_drv = 0.0
    for ci, outs in per.items():
        fx, fy = DIRS[ci]
        for o in outs:
            mx, my = int(max(-1, min(1, o[0]))), int(max(-1, min(1, o[1])))
            tot += 1
            amp_drv += sum(abs(v) for v in o) / 7.0
            if mx or my:
                moves += 1
                cos_sum += (mx * fx + my * fy) / math.sqrt(
                    (mx * mx + my * my) * (fx * fx + fy * fy))
                cos_n += 1
            pairs.append((ci, _dir_class(mx, my)))
            xs_by.setdefault(ci, []).append(mx)
            ys_by.setdefault(ci, []).append(my)
    mi = _mi(pairs, 8, 9)
    rnd = random.Random(seed + 7)
    shuf = []
    labels = [a for a, _ in pairs]
    for _ in range(20):
        rnd.shuffle(labels)
        shuf.append(_mi(list(zip(labels, [b for _, b in pairs])), 8, 9))
    mi_c = mi - sum(shuf) / len(shuf)

    def eta2(by):
        allv = [v for vs in by.values() for v in vs]
        if not allv:
            return 0.0
        g = sum(allv) / len(allv)
        sst = sum((v - g) ** 2 for v in allv)
        ssb = sum(len(vs) * (sum(vs) / len(vs) - g) ** 2 for vs in by.values())
        return ssb / sst if sst > 0 else 0.0
    amp_sil = sum(sum(abs(v) for v in o) / 7.0 for o in silent) / max(1, len(silent))
    # per-channel non-zero rates while driven
    ch = [0] * 7
    for outs in per.values():
        for o in outs:
            for k in range(7):
                if o[k]:
                    ch[k] += 1
    return {
        "chemo_probe": round(cos_sum / cos_n, 4) if cos_n else None,
        "chemo_n": cos_n,
        "mi_bits": round(max(0.0, mi_c), 4),
        "mi_raw": round(mi, 4),
        "move_rate": round(moves / max(1, tot), 4),
        "eta2_x": round(eta2(xs_by), 4),
        "eta2_y": round(eta2(ys_by), 4),
        "amp_silent": round(amp_sil, 4),
        "amp_driven": round(amp_drv / max(1, tot), 4),
        "ch_rate": [round(c / max(1, tot), 3) for c in ch],
    }


def _brain_dict(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict) and "brain" in d and isinstance(d["brain"], dict):
        return d["brain"], d.get("params"), d.get("name") or os.path.basename(path)
    return d, None, os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--warm", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--newborn", action="store_true",
                    help="also probe a newborn brain built from the same genes")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = []
    for p in a.files:
        try:
            bd, params, name = _brain_dict(p)
        except Exception as e:
            print("skip", p, e)
            continue
        if not bd or "spheres" not in bd:
            print("skip (no brain)", p)
            continue
        r = {"file": os.path.basename(p), "name": name}
        try:
            r.update(probe(bd, a.warm, a.steps, a.seed))
        except Exception as e:
            print("probe failed", p, e)
            continue
        if a.newborn and params:
            try:
                from server.engine import make_params
                pp = make_params(params)
                nb = probe(None, a.warm, a.steps, a.seed,
                           build=lambda: build_brain(pp))
                for k, v in nb.items():
                    r["newborn_" + k] = v
            except Exception as e:
                r["newborn_error"] = str(e)[:80]
        rows.append(r)
        print(json.dumps(r))
    if rows:
        def m(k):
            v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)
        print("\n== SUMMARY over %d brains" % len(rows))
        for k in ("chemo_probe", "mi_bits", "move_rate", "eta2_x", "eta2_y",
                  "amp_silent", "amp_driven", "newborn_chemo_probe",
                  "newborn_mi_bits", "newborn_move_rate"):
            mu, n = m(k)
            if n:
                print("  %-22s mean %.4f  (n=%d)" % (k, mu, n))
    if a.csv and rows:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(json.dumps(r.get(k, "")).replace(",", ";")
                                 for k in keys) + "\n")


if __name__ == "__main__":
    main()
