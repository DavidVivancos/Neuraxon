# Neuraxon Ant Colony 1.03 internal version 10
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonGenome.py — INTEGER trit-LUT genome + epoch structure (v1.03).

v1.03 replaces the floating-point chc6 brain with a PURE-INTEGER trinary
cellular automaton, per the recorded AntColony spec:

  STRUCTURES (epoch-fixed, derived from the salted spectrum digest -> identical
  on every node):
    sourceNeuron[N][3]  wiring: 3 neighbours feeding each non-input neuron
    T[ticks][nInput]    drive value per tick, by phase
    placement           input / output / hidden neuron ids
    phase map           tick -> WARMUP / DRIVEN / SILENCE / TRANSFER

  GENOME (mined per nonce):
    lut[N][27]          activation lookup table. For each non-input neuron, 27
                        entries (3^3, one per combination of its 3 neighbour
                        trits). Each entry is a trit 0/1/2. This IS the searched
                        genome. lut[n][line] = random_trit(K12(pubkey||nonce)).

Everything here is integer and PUBLIC. No floats anywhere in the state or the
update. Because trit updates are exact integer table lookups, the simulation is
bit-identical on every machine — the cross-node FP determinism caveat from
earlier versions is GONE. (Resolution is preserved not in the 3-valued state but
in the metrics, which accumulate as large integer counts over all ticks and
neurons; see NxonTrit.py / NxonScore.py.)

Trit encoding: NEUTRAL=0 (resting), POS=1 (+ polarity), NEG=2 (- polarity),
i.e. the paper's trinary -1/0/+1 remapped to 0/1/2.
"""

import hashlib

# Trit alphabet.
NEUTRAL, POS, NEG = 0, 1, 2

# Game I/O contract (matches the real Neuraxon game: 10 senses, 7 actions).
N_INPUT = 10
N_OUTPUT = 7
N_LINES = 27                      # 3^3 lookup lines per neuron

# Phase labels.
WARMUP, DRIVEN, SILENCE, TRANSFER = "WARMUP", "DRIVEN", "SILENCE", "TRANSFER"


# =============================================================================
# HASH RNG — deterministic, platform-independent integer stream
# =============================================================================
# Expands a seed into an unbounded stream of integers by hashing (seed || ctr).
# Uses SHA3-256 (Keccak family, a KangarooTwelve stand-in) + pure integer ops,
# so it produces the identical sequence on every machine. This is `random2` from
# the spec: random2(hash, pool).

class HashRng:
    __slots__ = ("seed", "ctr", "buf")

    def __init__(self, seed_bytes):
        if not isinstance(seed_bytes, bytes):
            seed_bytes = str(seed_bytes).encode("utf-8")
        self.seed = seed_bytes
        self.ctr = 0
        self.buf = b""

    def _refill(self):
        self.buf += hashlib.sha3_256(self.seed + self.ctr.to_bytes(8, "big")).digest()
        self.ctr += 1

    def u32(self):
        while len(self.buf) < 4:
            self._refill()
        v = int.from_bytes(self.buf[:4], "big")
        self.buf = self.buf[4:]
        return v

    def below(self, m):
        return self.u32() % m

    def trit(self):
        return self.u32() % 3


def k12(*parts) -> bytes:
    """KangarooTwelve-style public hash (SHA3-256 stand-in). Swap in real K12 in
    deployment; call sites don't change."""
    h = hashlib.sha3_256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.digest()


def k12_int(*parts) -> int:
    return int.from_bytes(k12(*parts)[:8], "big")


# =============================================================================
# EPOCH STRUCTURE — fixed for the epoch, derived from the salted digest
# =============================================================================

def build_epoch(salted_digest, N=48, ticks=128,
                warmup=32, driven=48, silence=24, transfer=24):
    """Everything a node needs to run the sim, deterministically derived from the
    public salted spectrum digest so every node builds the identical epoch."""
    if warmup + driven + silence + transfer != ticks:
        raise ValueError("phase lengths must sum to ticks")
    if N < N_INPUT + N_OUTPUT + 1:
        raise ValueError("N too small")

    rng = HashRng("epoch|" + str(salted_digest))

    # Placement: a digest-seeded permutation assigns input/output/hidden ids.
    perm = list(range(N))
    for i in range(N - 1, 0, -1):
        j = rng.below(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    input_ids = perm[:N_INPUT]
    output_ids = perm[N_INPUT:N_INPUT + N_OUTPUT]
    hidden_ids = perm[N_INPUT + N_OUTPUT:]
    noninput_ids = hidden_ids + output_ids     # these get sources + LUTs

    # Wiring: 3 source neighbours per non-input neuron (sources may be any
    # neuron, including inputs, so drive flows in).
    src0 = [0] * N
    src1 = [0] * N
    src2 = [0] * N
    for n in noninput_ids:
        src0[n] = rng.below(N)
        src1[n] = rng.below(N)
        src2[n] = rng.below(N)

    # Phase map.
    phase_of = []
    for t in range(ticks):
        if t < warmup:
            phase_of.append(WARMUP)
        elif t < warmup + driven:
            phase_of.append(DRIVEN)
        elif t < warmup + driven + silence:
            phase_of.append(SILENCE)
        else:
            phase_of.append(TRANSFER)

    # Drive table T[tick][k] (trits). SILENCE injects nothing (all NEUTRAL).
    T = []
    for t in range(ticks):
        ph = phase_of[t]
        row = [NEUTRAL] * N_INPUT
        if ph == WARMUP:
            for k in range(N_INPUT):
                row[k] = rng.trit()
        elif ph == DRIVEN:
            base = (t // 4) % 2
            for k in range(N_INPUT):
                row[k] = POS if ((k + base) % 2 == 0) else NEG
        elif ph == SILENCE:
            pass  # no input
        else:  # TRANSFER — a novel pattern, different from DRIVEN
            for k in range(N_INPUT):
                row[k] = POS if (((k * (t % 3)) % 3) == 0) else NEUTRAL
        T.append(row)

    return {
        "N": N, "ticks": ticks,
        "input_ids": input_ids, "output_ids": output_ids,
        "hidden_ids": hidden_ids, "noninput_ids": noninput_ids,
        "src0": src0, "src1": src1, "src2": src2,
        "phase_of": phase_of, "T": T,
        "phase_ticks": {WARMUP: warmup, DRIVEN: driven,
                        SILENCE: silence, TRANSFER: transfer},
        "salted_digest": str(salted_digest),
    }


# =============================================================================
# LUT GENOME — seed, root, mutate (with undo), hash, (de)serialize
# =============================================================================

def empty_lut(N):
    return [[NEUTRAL] * N_LINES for _ in range(N)]


def seed_lut(pubkey, nonce, epoch):
    """Mined genome: lut[n][line] = trit from K12(pubkey || nonce)."""
    rng = HashRng(k12(b"lut", str(pubkey), str(nonce)))
    lut = empty_lut(epoch["N"])
    for n in epoch["noninput_ids"]:
        row = lut[n]
        for line in range(N_LINES):
            row[line] = rng.below(3)
    return lut


def root_lut(epoch):
    """The ROOT genome — a BLANK all-NEUTRAL brain (every LUT line = NEUTRAL).

    Deterministic and public (no randomness needed). It is the true floor: a
    brain that does nothing (all neurons frozen at rest), so the colony must
    evolve it toward healthy dynamics — there is maximum headroom to climb. ROOT
    score = 0 by definition (never scored); this LUT is the shared start every
    node reproduces bit-for-bit."""
    return empty_lut(epoch["N"])


def copy_lut(lut):
    return [row[:] for row in lut]


def mutate_lines(lut, rng, L, noninput_ids):
    """Flip L LUT lines in place; return an undo list so the walk can revert."""
    undo = []
    m = len(noninput_ids)
    for _ in range(L):
        n = noninput_ids[rng.below(m)]
        line = rng.below(N_LINES)
        old = lut[n][line]
        new = rng.below(3)
        lut[n][line] = new
        undo.append((n, line, old))
    return undo


def apply_undo(lut, undo):
    for (n, line, old) in reversed(undo):
        lut[n][line] = old


def hash_lut(lut, epoch):
    parts = ["".join(str(t) for t in lut[n]) for n in sorted(epoch["noninput_ids"])]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def lut_to_str(lut, epoch):
    return "".join("".join(str(t) for t in lut[n])
                   for n in sorted(epoch["noninput_ids"]))


def lut_from_str(s, epoch):
    lut = empty_lut(epoch["N"])
    i = 0
    for n in sorted(epoch["noninput_ids"]):
        for line in range(N_LINES):
            lut[n][line] = int(s[i])
            i += 1
    return lut


# =============================================================================
# NONCE = (base, L, K)   — L lines mutated per step, K anti-attractor steps
# =============================================================================

def make_nonce(base, L, K):
    return "{}_{}_{}".format(base, L, K)


def parse_nonce(nonce):
    try:
        base, L, K = str(nonce).split("_")
        return int(base), max(1, int(L)), max(0, int(K))
    except Exception:
        return 0, 1, 0


def derive_eval_note(pubkey, nonce, parent_ref):
    """Kept for interface parity; the trit sim seed is not needed separately —
    the walk RNG is seeded from K12(pubkey||nonce) and the sim is deterministic
    from the LUT + epoch."""
    return k12_int(b"eval", str(pubkey), str(nonce), str(parent_ref))
