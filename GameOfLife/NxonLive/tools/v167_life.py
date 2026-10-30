"""v1.67 — one offline life of one brain, closed loop, server-faithful, to check
the six questions before (or without) a live run.

The brain is built exactly as the v1.67 builder builds it (make_params ->
build_brain -> brain_fix.apply_living -> add_sensorimotor for the wired arms)
and stepped the way the v1.67 worker steps it (energy rescue every step, the
eating reward written into phasic dopamine, the worker's timescale
relaxation). The body forages on an open plane: food density as in the
earlier offline tests, one decision per brain step, at most 2 cells per
decision (brain_move_steps), eating on landing, rest = stand still.

Arms (--arm):
  wired       living brain + innate sensorimotor projection (reflex genes)
  wired_sham  the same, but its senses come from a yoked ghost elsewhere
  plain       living brain, no projection
  sham        plain brain, yoked senses
  frozen      wired brain with plasticity frozen (homeostasis kept)
  legacy      the V 1.086 brain: no fixes, no projection, reward into tonic

Logged every --log steps: frozen share, frozen inputs/motor outputs,
synapses, mean |w|, tonic/phasic dopamine, bites, chemotaxis of the steps
taken, how often the fdx/fdy input neurons follow the world.
At the end the brain is probed open loop (8 food directions x 3 seeds, 16
warm-up + 32 recorded steps): MI(direction; motor step) in bits (shuffle-
corrected) and chemotaxis, with and without the projection.

Usage: v167_life.py <mngol5 root> <brain json or '-'> <out json> [--arm wired]
       [--steps 900] [--seed 7] [--log 50] [--genes chemo|own]
"""
import os, sys, json, math, random, argparse
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--arm", default="wired")
ap.add_argument("--steps", type=int, default=900)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--log", type=int, default=50)
ap.add_argument("--genes", default="own", help="own (from the brain json) | chemo (xx=yy=0.8, gain 1.5)")
ap.add_argument("--probe-seeds", type=int, default=3)
ap.add_argument("--food", type=int, default=75, help="food items in the 61x61 window around the walker (75 = 2%%, the live world's 2,600 items on 360x360)")
A = ap.parse_args()
os.environ["NEURAXON_HEADLESS"] = "1"; os.environ["MNGOL5_AGMP"] = "1"
sys.path.insert(0, A.root)
from server import np_fallback; np_fallback.install()
import architecture; architecture.load_architecture(os.path.join(A.root, "architecture_files", "nas_best.json"), verbose=False)
from server.engine import make_params, REFLEX_GENES
from server import brain_fix as BF
from server.brain_pool import _freeze_brain
from neuraxon.multisphere import build_brain, load_multisphere_from_dict
import neuraxon.network as NN

src = json.load(open(A.src)) if A.src != "-" else {}
legacy = A.arm == "legacy"
wired = A.arm in ("wired", "wired_sham", "frozen")
yoked = A.arm in ("wired_sham", "sham")
genes = dict(src.get("reflex") or {})
if A.genes == "chemo" or not genes:
    genes = {"rfx_xx": 0.8, "rfx_xy": 0.0, "rfx_yx": 0.0, "rfx_yy": 0.8, "rfx_rest": 0.6, "rfx_gain": 1.5}
CFG = dict(BF.DEFAULTS)
random.seed(A.seed)
p = make_params(src.get("params"))
b = build_brain(p)
if legacy:
    # V 1.086: the builder's brain reaches the worker through the substrate
    # loader, which rebuilds every sphere as >=10 inputs / >=7 outputs
    b = load_multisphere_from_dict(json.loads(json.dumps(b.to_dict())), NN._rebuild_net_from_dict)
else:
    BF.apply_living(b, CFG)
    if wired:
        BF.add_sensorimotor(b, genes, CFG)
if A.arm == "frozen":
    _freeze_brain(b, True)
neurons = BF.neuron_list(b)
relays = BF.relay_list(b, CFG) if not legacy else []
rnd = random.Random(A.seed + 1)
DIRS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
sg = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)


class Field:
    """an open plane with food around a walker"""
    def __init__(self, r):
        self.r = r; self.food = {}; self.x = self.y = 0
    def refill(self):
        # an endless plane at constant density: A.food items in the 61x61 window
        # around the walker; items left behind are dropped and regrow at random
        # places in the window (the earlier version kept the first 200 items, so
        # a walker that strayed >22 cells from them never sensed food again)
        for k in [k for k in self.food if abs(k[0] - self.x) > 30 or abs(k[1] - self.y) > 30]:
            del self.food[k]
        while len(self.food) < A.food:
            fx, fy = self.x + self.r.randint(-30, 30), self.y + self.r.randint(-30, 30)
            if (fx, fy) != (self.x, self.y) and (fx, fy) not in self.food:
                self.food[(fx, fy)] = 25
    def senses(self):
        best = None; bd = 1e9
        for (fx, fy) in self.food:
            d = (fx - self.x) ** 2 + (fy - self.y) ** 2
            if d < bd and d <= 22 * 22:
                bd = d; best = (fx, fy)
        if best is None:
            return 0, 0, 0.0, 0.0
        return sg(best[0] - self.x), sg(best[1] - self.y), (1.0 if bd < 25 else 0.0), 1.0 / (1.0 + math.sqrt(bd))


me = Field(rnd); me.refill()
ghost = Field(random.Random(A.seed + 99)); ghost.refill()      # yoked senses for the sham arms
energy = 120.0; bank = 0.0; last_eat = -99; bites = 0; heading = 0
ids = [n.id for n in b.sensory_sphere.network.input_neurons]
log = []; cs = []; cs_lag = []; prev_dir = (0, 0); follow = [0, 0]; moves = 0; rescued = 0; rewards = 0


def snap(step):
    m = BF.measure(b)
    syn = [s for sp in b.spheres.values() for s in sp.network.synapses if s.integrity > 0]
    w = [abs(s.w_fast + s.w_slow) for s in syn]
    r = {"step": step, "ticks": step * 8, "bites": bites, "syn": len(syn),
         "w_abs": round(sum(w) / len(w), 4) if w else 0.0,
         "chemo_last": round(sum(cs[-200:]) / max(1, len(cs[-200:])), 3),
         "chemo_lag": round(sum(cs_lag[-200:]) / max(1, len(cs_lag[-200:])), 3),
         "follow": round(follow[0] / max(1, follow[1]), 3), "moves": moves, "rescued": rescued,
         "M1_E": round(sum(1 for n in neurons if n.trinary_state > 0) / len(neurons), 3),
         "agmp": BF.agmp_rate(b), "rewards": rewards}
    r.update(m)
    return r


for step in range(A.steps + 1):
    if step % A.log == 0:
        log.append(snap(step)); follow = [0, 0]
        if step == A.steps:
            break
    fdx, fdy, sight, smell = me.senses()
    if yoked:
        ghost.x += rnd.choice((-1, 0, 1)); ghost.y += rnd.choice((-1, 0, 1)); ghost.refill()
        gfx, gfy, gsi, gsm = ghost.senses()
    else:
        gfx, gfy, gsi, gsm = fdx, fdy, sight, smell
    hunger = max(-1.0, min(1.0, 1.0 - energy / 60.0))
    sens = [hunger, float(gfx), float(gfy), 0.0, gsi, gsm, math.sin(step * 8 * 0.01), heading / 7.0, 0.0, 0.0]
    if bank > 0:
        if legacy or not BF.cfg_get(CFG, "reward_phasic"):
            b.set_global_modulator("dopamine", 0.15 + min(bank, 0.85))
        else:
            BF.reward_phasic(b, min(bank, 0.85), CFG["reward_phasic_cap"])
        bank = 0.0; rewards += 1
    if not legacy:
        rescued += BF.rescue_energy(neurons)
        BF.hold_relays(relays)
    for sp in b.spheres.values():                       # worker: timescale relaxation (v1.63)
        for n in sp.network.all_neurons:
            bs = getattr(n, "membrane_time_constant", None)
            if bs: n.intrinsic_timescale += 0.05 * (bs - n.intrinsic_timescale)
    BF.step_brain(b, {"sensory": {ids[k]: sens[k] for k in range(len(ids))}}, BF.cfg_get(CFG, "sm_same_step") and not legacy)
    inp = b.sensory_sphere.network.input_neurons
    for k, f in ((1, gfx), (2, gfy)):
        if f:
            follow[1] += 1; follow[0] += 1 if inp[k].trinary_state == f else 0
    o = list(b.motor_sphere.network.get_output_states()) + [0] * 7
    mx, my, rest = sg(o[0]), sg(o[1]), o[5]
    # body-level rest arc of the wired arms (outside the brain, so it reads the
    # body's OWN smell in wired_sham too: the arms differ only in the brain's input)
    if wired and genes.get("rfx_gain"):
        v = rest + genes["rfx_gain"] * genes.get("rfx_rest", 0.0) * (2.0 * smell - 1.0)
        rest = 1 if v > 0.5 else (-1 if v < -0.5 else 0)
    if rest == 1:
        mx = my = 0
    if (mx or my) and (fdx or fdy):
        cs.append((mx * fdx + my * fdy) / math.sqrt((mx * mx + my * my) * (fdx * fdx + fdy * fdy)))
    # the same against the direction one brain step earlier: links between
    # spheres carry the previous step's states, so the motor output answers
    # the direction the brain sensed one step before
    px, py = prev_dir
    if (mx or my) and (px or py):
        cs_lag.append((mx * px + my * py) / math.sqrt((mx * mx + my * my) * (px * px + py * py)))
    prev_dir = (fdx, fdy)
    for k in range(2):
        if not (mx or my):
            break
        me.x += mx; me.y += my; moves += 1; heading = DIRS.index((mx, my))
        if (me.x, me.y) in me.food:
            break                                        # landed on food: stop here
    # eat on the current cell, as the server does: one bite per 20 ticks while on it
    if (me.x, me.y) in me.food and (step - last_eat) * 8 >= 20:
        me.food[(me.x, me.y)] -= 1; bites += 1; bank += 0.5; energy += 1; last_eat = step
        if me.food[(me.x, me.y)] <= 0:
            del me.food[(me.x, me.y)]
    energy -= 0.003 * 8 * (0.7 if rest == 1 else 1.0)
    me.refill()

aged = b.to_dict()


def probe(bd, with_link=True, seeds=(100, 200, 300)):
    """open loop: food clamped in 8 directions; MI(direction; motor step)"""
    pairs = []; cos = []
    for di, (fx, fy) in enumerate(DIRS):
        for seed in seeds:
            random.seed(seed)
            bb = load_multisphere_from_dict(json.loads(json.dumps(bd)),
                                            NN._rebuild_net_from_dict if legacy else BF.rebuild_net_exact)
            if not legacy:
                BF.apply_living(bb, CFG)
            if not with_link:
                BF.remove_sensorimotor(bb)
            nl = BF.neuron_list(bb)
            rl = BF.relay_list(bb, CFG) if not legacy else []
            ii = [n.id for n in bb.sensory_sphere.network.input_neurons]
            s = [0.6, float(fx), float(fy), 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0]
            for t in range(48):
                if not legacy:
                    BF.rescue_energy(nl)
                    BF.hold_relays(rl)
                BF.step_brain(bb, {"sensory": {ii[k]: s[k] for k in range(len(ii))}}, BF.cfg_get(CFG, "sm_same_step") and not legacy)
                if t >= 16:
                    o = bb.motor_sphere.network.get_output_states()
                    mx, my = sg(o[0]), sg(o[1])
                    pairs.append((di, 8 if (mx, my) == (0, 0) else DIRS.index((mx, my))))
                    if mx or my:
                        cos.append((mx * fx + my * fy) / math.sqrt((mx * mx + my * my) * (fx * fx + fy * fy)))
    def mi(pp):
        n = len(pp); ca = {}; cb = {}; cab = {}
        for a, c in pp:
            ca[a] = ca.get(a, 0) + 1; cb[c] = cb.get(c, 0) + 1; cab[(a, c)] = cab.get((a, c), 0) + 1
        return sum(v / n * math.log2((v / n) / ((ca[a] / n) * (cb[c] / n))) for (a, c), v in cab.items())
    m = mi(pairs); r = random.Random(5); labs = [a for a, _ in pairs]; sh = []
    for _ in range(20):
        r.shuffle(labs); sh.append(mi(list(zip(labs, [c for _, c in pairs]))))
    return {"mi_bits": round(max(0.0, m - sum(sh) / len(sh)), 4), "chemo": round(sum(cos) / max(1, len(cos)), 3),
            "move": round(sum(1 for _, c in pairs if c != 8) / len(pairs), 3)}


seeds = tuple(100 * (k + 1) for k in range(A.probe_seeds))
res = {"arm": A.arm, "seed": A.seed, "genes": genes, "src": os.path.basename(A.src), "log": log,
       "probe": probe(aged, True, seeds), "aged_brain": aged, "params": src.get("params"), "reflex": genes}
if wired:
    res["probe_no_link"] = probe(aged, False, seeds)
json.dump(res, open(A.out, "w"))
for r in log[:: max(1, len(log) // 8)] + [log[-1]]:
    print("step %5d ticks %6d | frozen %.3f in %s mo %s | syn %4d |w| %.3f | DA tonic %.3f phasic %.3f | bites %4d chemo %+.2f (lag %+.2f) follow %.2f rescued %d M1_E %.2f" % (
        r["step"], r["ticks"], r.get("frozen_frac") or 0, r.get("frozen_in"), r.get("frozen_mo"), r["syn"], r["w_abs"],
        r.get("DA_tonic") or 0, r.get("DA_phasic") or 0, r["bites"], r["chemo_last"], r["chemo_lag"], r["follow"], r["rescued"], r["M1_E"]), flush=True)
print("probe:", res["probe"], "| without the projection:", res.get("probe_no_link"))
