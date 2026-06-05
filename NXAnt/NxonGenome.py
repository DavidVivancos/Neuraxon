# Neuraxon Ant Colony 1.02 internal version 09
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
"""
NxonGenome.py — PUBLIC genome law for the consensus edition.

Everything here is PUBLIC and identical on every node, because in the Qubic
node-consensus model every node re-applies the same mutation to the same parent
and re-runs the same simulation. Nothing in this file is secret.

Contents:
  - default_genome()         the architecture template (the searchable sections)
  - SEARCH_SPACE             the legal mutation bounds (chc6 levers included)
  - k12 / k12_int            KangarooTwelve-style public hash (stand-in here)
  - mutate_genome_k12        DETERMINISTIC mutation derived from pubkey||nonce
  - derive_eval_seed         DETERMINISTIC sim seed from pubkey||nonce||parentRef
  - root_genome              the ROOT ANN, seeded from the salted spectrum digest

Determinism note: the mutation and the eval seed are derived from public inputs
(pubkey, nonce, parentRef) via a hash, and the mutation uses a seeded Mersenne
Twister whose sequence is platform-independent for a given seed. So every node
reproduces the same child genome and the same eval seed bit-for-bit. (The SIM
itself must also be made deterministic across hardware for full cross-node
agreement — see NxonScore.py.)
"""

import copy
import hashlib
import random


# =============================================================================
# K12-STYLE PUBLIC HASH (deterministic from public inputs)
# =============================================================================
# Real Qubic derives mutations as K12(pubkey || nonce). KangarooTwelve isn't in
# the stdlib, so we use a Keccak-family hash (SHA3-256) as a faithful stand-in:
# same property (deterministic, public, collision-resistant). Swap in real K12
# in deployment; the call sites don't change.

def k12(*parts) -> bytes:
    h = hashlib.sha3_256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.digest()


def k12_int(*parts) -> int:
    return int.from_bytes(k12(*parts)[:8], "big")


def derive_eval_seed(pubkey, nonce, parent_ref) -> int:
    """The simulation seed for a submission. Public + reproducible: every node
    derives the identical seed from (pubkey, nonce, parentRef)."""
    return k12_int(b"eval", str(pubkey), str(nonce), str(parent_ref)) & 0xFFFFFFFF


# =============================================================================
# GENOME TEMPLATE + SEARCH SPACE  (chc6 levers included)
# =============================================================================

def default_genome():
    return {
        "_meta": {"name": "nxonant_genesis", "source": "NxonAnt1.02-consensus"},
        "neural": {
            "num_input_neurons": 10,
            "num_output_neurons": 7,
            "num_hidden_neurons_default": 24,
            "connection_probability": 0.30,
            "afferent_synapse_strength": 0.70,
            "sensory_input_gain": 0.90,
            "firing_threshold_excitatory": 0.50,
            "firing_threshold_inhibitory": -0.90,
            "spontaneous_firing_rate": 0.010,
            "intrinsic_timescale_default": 20.0,
            "resting_potential_decay": 0.20,
            "sensorimotor_coupling": 1.0,
            "symmetric_stdp": False,
            "refractory_period_ticks": 3,
            "post_spike_mp_reset": 0.30,
            "sphere_topology": "chc6",
            "cross_sphere_coupling": 1.0,    # kappa
            "cryst_capacity": 1.0,           # lambda_c
            "free_energy_beta": 1.0,         # beta_f
        },
        "operating_ranges": {
            "learning_rate": 0.020,
            "plasticity_threshold": 0.50,
            "adaptation_tau_ticks": 30.0,
            "adaptation_target_excitatory_multiplier": 1.5,
            "adaptation_target_inhibitory_multiplier": 1.2,
            "autoreceptor_coefficient": 0.15,
            "autoreceptor_tau_ticks": 150.0,
            "autoreceptor_rate_coeff": 0.35,
            "sensory_boost_scale": 1.0,
            "plasticity_brake_threshold": 0.5,
            "plasticity_brake_slope": 1.8,
            "plasticity_brake_floor": 0.1,
        },
        "genetic_lottery": {
            "intrinsic_timescale_jitter": 3.0,
            "firing_threshold_jitter": 0.04,
            "mutation_strength": 0.05,
        },
        "biology": {"circadian_cycle_ticks": 300},
    }


SEARCH_SPACE = [
    ("neural", "num_hidden_neurons_default", 12, 40, True),
    ("neural", "connection_probability", 0.10, 0.70, False),
    ("neural", "afferent_synapse_strength", 0.30, 1.20, False),
    ("neural", "sensory_input_gain", 0.40, 1.50, False),
    ("neural", "firing_threshold_excitatory", 0.15, 0.80, False),
    ("neural", "firing_threshold_inhibitory", -1.20, -0.40, False),
    ("neural", "spontaneous_firing_rate", 0.001, 0.030, False),
    ("neural", "intrinsic_timescale_default", 8.0, 40.0, False),
    ("neural", "resting_potential_decay", 0.05, 0.40, False),
    ("neural", "sensorimotor_coupling", 0.0, 3.0, False),
    ("neural", "refractory_period_ticks", 0, 8, True),
    ("neural", "post_spike_mp_reset", 0.0, 1.0, False),
    ("neural", "cross_sphere_coupling", 0.0, 3.0, False),    # kappa
    ("neural", "cryst_capacity", 0.3, 2.5, False),           # lambda_c
    ("neural", "free_energy_beta", 0.0, 2.5, False),         # beta_f
    ("operating_ranges", "learning_rate", 0.001, 0.060, False),
    ("operating_ranges", "plasticity_threshold", 0.20, 0.80, False),
    ("operating_ranges", "adaptation_tau_ticks", 10.0, 60.0, False),
    ("operating_ranges", "autoreceptor_coefficient", 0.02, 0.40, False),
    ("genetic_lottery", "intrinsic_timescale_jitter", 0.0, 8.0, False),
    ("genetic_lottery", "firing_threshold_jitter", 0.0, 0.20, False),
]
BOOLEAN_LEVERS = [("neural", "symmetric_stdp")]


def mutate_genome_k12(parent_genome, pubkey, nonce, strength):
    """Deterministic child = parent + K12(pubkey||nonce)-seeded mutation.

    Every node calls this with the same (parent_genome, pubkey, nonce, strength)
    and gets the identical child. This is the on-node, reproducible mutation —
    the analogue of the Qubic miner's K12(pubkey||nonce) child mutations.
    """
    rng = random.Random(k12_int(b"mut", str(pubkey), str(nonce)))
    g = copy.deepcopy(parent_genome)
    for (section, key, lo, hi, is_int) in SEARCH_SPACE:
        if section not in g or key not in g[section]:
            continue
        span = hi - lo
        val = g[section][key] + rng.gauss(0.0, strength * span)
        if val < lo:
            val = lo
        elif val > hi:
            val = hi
        if is_int:
            val = int(round(val))
        g[section][key] = val
    for (section, key) in BOOLEAN_LEVERS:
        if section in g and key in g[section] and rng.random() < strength * 2.0:
            g[section][key] = not g[section][key]
    g["_meta"] = {"name": "child", "source": "NxonAnt1.02-consensus",
                  "pubkey": str(pubkey), "nonce": str(nonce)}
    return g


def root_genome(salted_spectrum_digest):
    """The ROOT ANN — a deterministic function of consensus (the salted spectrum
    digest every machine already agrees on). The genome itself is the public
    default; the digest is recorded so the root is reproducible from public
    information, and ROOT score = 0 by definition (it is never scored)."""
    g = default_genome()
    g["_meta"] = {"name": "ROOT", "source": "NxonAnt1.02-consensus",
                  "salted_spectrum_digest": str(salted_spectrum_digest)}
    return g
