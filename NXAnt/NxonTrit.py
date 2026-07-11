# Neuraxon Ant Colony 1.03 internal version 10
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonTrit.py — INTEGER trit-CA simulation + anti-attractor mining walk (v1.03).

This is the "ant": the ~1-second experiment every node re-runs to re-verify a
submission. It is pure integer.

RUN SIMULATION (per the recorded spec):
    state[*] = NEUTRAL
    for tick in 0..ticks-1:
        for k,n in input:  state[n] = T[tick][k]          # inject drive (held)
        for n in hidden+output:                            # update via LUT
            idx = state[src[n][0]] + 3*state[src[n][1]] + 9*state[src[n][2]]
            next[n] = lut[n][idx]
        for n in hidden+output:  state[n] = next[n]        # commit non-input
        accumulate metrics this tick, BY PHASE

The per-neuron update is identical every tick. What changes by phase is only
what is injected into the input neurons (SILENCE injects nothing) and which
metrics accumulate. There is NO training target and NO output comparison — we
measure the DYNAMICS (neuron-state statistics per phase). Scoring is a band-score
over those metrics (higher = healthier), in NxonScore.

MINING (anti-attractor walk; nonce = base_L_K, L lines per step, K escape steps):
    cur = best = score(runSim())
    for s in 0..walk_steps-1:
        mutate L lut lines (record undo)
        r = score(runSim())
        accept = (s < K) ? allow-worse : better-only       # escape K, then climb
        cur = accept ? r : undo()
        best = max(best, cur)                                # keep best LUT seen
    return best_lut, best_score

The walk RNG is seeded from K12(pubkey||nonce), so the whole walk is
deterministic and any node reproduces the identical best_lut and best_score.
"""

import time

import NxonGenome as G
import NxonScore

SCALE = NxonScore.METRIC_SCALE


def run_sim(lut, epoch):
    """Run the integer trit CA and return FIXED-POINT integer metrics."""
    N = epoch["N"]
    ticks = epoch["ticks"]
    src0 = epoch["src0"]; src1 = epoch["src1"]; src2 = epoch["src2"]
    phase_of = epoch["phase_of"]
    T = epoch["T"]
    input_ids = epoch["input_ids"]
    noninput = epoch["noninput_ids"]
    pt = epoch["phase_ticks"]
    nnon = len(noninput)

    state = [G.NEUTRAL] * N
    nextstate = [G.NEUTRAL] * N

    # Accumulators (integers).
    d_active = d_pos = d_neg = 0            # DRIVEN
    s_active = 0                            # SILENCE
    tr_active = 0                           # TRANSFER
    br_sum = 0                              # sum of fixed-point active ratios
    br_cnt = 0
    persist = persist_cnt = 0              # DRIVEN state persistence
    prev_active = None
    prev_commit = None                     # previous committed state (for persistence)
    prev_phase = None

    for tick in range(ticks):
        ph = phase_of[tick]
        drive = T[tick]
        # 1) inject drive into input neurons (held).
        for k in range(len(input_ids)):
            state[input_ids[k]] = drive[k]
        # 2) compute next state for non-input via the LUT (base-3 index).
        for n in noninput:
            idx = state[src0[n]] + 3 * state[src1[n]] + 9 * state[src2[n]]
            nextstate[n] = lut[n][idx]
        # 3) commit non-input.
        for n in noninput:
            state[n] = nextstate[n]

        # 4) accumulate metrics by phase (over non-input neurons).
        act = pos = neg = 0
        for n in noninput:
            v = state[n]
            if v != G.NEUTRAL:
                act += 1
                if v == G.POS:
                    pos += 1
                else:
                    neg += 1

        if ph == G.DRIVEN:
            d_active += act
            d_pos += pos
            d_neg += neg
            if prev_phase == G.DRIVEN and prev_active is not None and prev_active > 0:
                br_sum += (act * SCALE) // prev_active
                br_cnt += 1
            if prev_phase == G.DRIVEN and prev_commit is not None:
                for n in noninput:
                    persist_cnt += 1
                    if state[n] == prev_commit[n]:
                        persist += 1
            prev_active = act
            prev_commit = state[:]        # snapshot for next-tick persistence
        elif ph == G.SILENCE:
            s_active += act
        elif ph == G.TRANSFER:
            tr_active += act

        prev_phase = ph

    driven_ticks = pt[G.DRIVEN]
    silence_ticks = pt[G.SILENCE]
    transfer_ticks = pt[G.TRANSFER]

    def ratio(num, den):
        return (num * SCALE) // den if den > 0 else 0

    rate_driven = ratio(d_active, nnon * driven_ticks)
    rate_transfer = ratio(tr_active, nnon * transfer_ticks)

    metrics = {
        "excitatory_fraction": ratio(d_pos, d_pos + d_neg),
        "active_fraction": rate_driven,
        "branching": (br_sum // br_cnt) if br_cnt > 0 else 0,
        "spontaneous_fraction": ratio(s_active, nnon * silence_ticks),
        "transfer_ratio": (rate_transfer * SCALE) // rate_driven if rate_driven > 0 else 0,
        "saturation": ratio(persist, persist_cnt),
    }
    return metrics


def score_lut(lut, epoch):
    return NxonScore.consensus_score_int(run_sim(lut, epoch))


def mining_walk(parent_lut, pubkey, nonce, epoch, walk_steps, deadline=None):
    """Anti-attractor walk from parent_lut. Returns (best_lut, best_score,
    best_metrics, sims_done). Deterministic from K12(pubkey||nonce)."""
    base, L, K = G.parse_nonce(nonce)
    rng = G.HashRng(G.k12(b"walk", str(pubkey), str(nonce)))
    lut = G.copy_lut(parent_lut)

    cur_metrics = run_sim(lut, epoch)
    cur = NxonScore.consensus_score_int(cur_metrics)
    best = cur
    best_lut = G.copy_lut(lut)
    best_metrics = cur_metrics
    sims = 1

    for s in range(walk_steps):
        if deadline is not None and time.time() >= deadline:
            break
        undo = G.mutate_lines(lut, rng, L, epoch["noninput_ids"])
        m = run_sim(lut, epoch)
        r = NxonScore.consensus_score_int(m)
        sims += 1
        if s < K:
            accept = True                 # allow worse — escape attractors
        else:
            accept = (r > cur)            # better only — climb
        if accept:
            cur = r
        else:
            G.apply_undo(lut, undo)
        if cur > best:
            best = cur
            best_lut = G.copy_lut(lut)
            best_metrics = m if accept else run_sim(best_lut, epoch)
    return best_lut, best, best_metrics, sims


if __name__ == "__main__":
    # Quick self-test + benchmark.
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    walk = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    ep = G.build_epoch("demo_digest", N=N, ticks=ticks)
    root = G.root_lut(ep)
    t0 = time.time()
    rm = run_sim(root, ep)
    t1 = time.time()
    print("single sim: {:.4f}s".format(t1 - t0))
    print("ROOT metrics (fixed-point):")
    for k, v in rm.items():
        print("  {:<22} {:>10}  ({:.3f})".format(k, v, NxonScore.fp_to_float(v)))
    print("ROOT score:", NxonScore.consensus_score_int(rm),
          "| bands satisfied {:.0%}".format(NxonScore.band_satisfaction(rm)))

    nonce = G.make_nonce(1234, 3, 8)
    t0 = time.time()
    bl, bs, bm, sims = mining_walk(root, "pk_demo", nonce, ep, walk)
    t1 = time.time()
    print("\nmining walk: {} steps, {} sims, {:.3f}s (budget 1.0s)".format(
        walk, sims, t1 - t0))
    print("walk best score:", bs, "(root was", NxonScore.consensus_score_int(rm), ")",
          "| bands {:.0%}".format(NxonScore.band_satisfaction(bm)))
