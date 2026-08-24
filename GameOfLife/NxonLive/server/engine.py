# Multi Neuraxon Game of Life 5 — headless world engine  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# A clean, pygame-free, log-free reimplementation of the Game-of-Life
# world dynamics, reusing the proven v184 neural substrate
# (neuraxon.multisphere.build_brain + neuraxon.gfactor) verbatim.
#
# Design goals (from the spec):
#   * runs forever; respawns when the world would die out
#   * all-time ranking over BOTH live and dead NxErs
#   * deterministic-ish, snapshot/restore for crash recovery
#   * no visuals, no diagnostic logging — just the fundamentals
#   * NxErs may be user-owned (password) or fully autonomous
#
# The engine is single-threaded and stepped by game_server.py. The web
# layer never touches engine internals directly — it reads snapshots.
# ===================================================================
import os
import math
import time
import random
import hashlib
from collections import deque

os.environ.setdefault("NEURAXON_HEADLESS", "1")

# numpy shim must be registered before ANY neuraxon import (PyPy fix)
from server import np_fallback  # noqa: E402
np_fallback.install()

import architecture  # noqa: E402

# Load the proven NAS-best architecture (fitness 6.88) so every NxEr's
# brain uses tuned neural + operating-range parameters instead of raw
# defaults. This is THE fix for "they die too fast / models not best".
_ARCH_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "architecture_files", "nas_best.json")
try:
    architecture.load_architecture(_ARCH_FILE, verbose=False)
except Exception:
    architecture._ARCH = {}
    architecture._ARCH_PATH = None

from config import NetworkParameters            # noqa: E402
from architecture import get_param as arch_get  # noqa: E402

# v1.52 — learning loop: when True, make_params enables the substrate's
# three-factor AGMP plasticity so dopamine bursts can consolidate learning.
# Set from world_config by Engine.__init__ before any NxEr is built.
_LEARNING_AGMP = False
# brains are built inside BrainPool workers (see brain_pool.py)
from neuraxon.gfactor import compute_population_g, MIN_AGENTS_FOR_G  # noqa: E402
import neuraxon.gfactor as _gfactor  # noqa: E402
# gfactor auto-detects numpy and uses a heavy linear-algebra path
# (corrcoef / eigh / triu_indices ...) that the pure-Python numpy
# shim does NOT implement. When the shim is active, force gfactor's
# own exact pure-Python g computation instead — otherwise the numpy
# path throws and g silently stays 0 forever (the bug seen on PyPy).
import sys as _sys  # noqa: E402
if getattr(_sys.modules.get("numpy"), "IS_FALLBACK", False):
    _gfactor._HAVE_NUMPY = False

# 8-neighbourhood (NW..W), matching the Lite client convention.
DIR_OFFSETS = [(-1, -1), (0, -1), (1, -1), (1, 0),
               (1, 1), (0, 1), (-1, 1), (-1, 0)]

RANK_METRICS = ("food_found", "food_taken", "explored",
                "time_lived", "mates_performed", "fitness", "g")
# Metrics that start at 0 and only grow — when a NxEr's value is 0
# here it has literally not scored anything, so "#N" is meaningless;
# the rank is reported as None and the client shows "—". g is excluded
# (it can legitimately be 0 or negative without meaning "no data").
COUNTER_METRICS = frozenset(("food_found", "food_taken", "explored",
                             "time_lived", "mates_performed",
                             "fitness"))

# v1.44 — target bands for the offline-GoL M-metrics we can faithfully
# record online (subset of research_probes.TARGET_BANDS). Used only to
# tag each logged value in-band/out-of-band; not enforced anywhere.
# M3 (PAC), M4 (multi-timescale weights) and M9 (compositional) are NOT
# recorded online — M3 needs a 200-tick per-brain oscillator window, M4
# needs weight-increment history and is moot with AGMP off, and M9 needs
# signed/multi-level sight+song channels (online sight/song are binary).
M_BANDS = {
    "M1_E":            (0.18, 0.28),
    "M1_I":            (0.08, 0.15),
    "M1_N":            (0.60, 0.78),
    "M2_mean_gate":    (0.40, 0.85),
    "M2_gate_xlink_std": (0.05, 0.45),
    "M5_branching":    (0.92, 1.10),
    "M6_acw_heterogeneity": (3.0, 60.0),
    "M7_zero_input_ratio":  (0.40, 1.20),
    "M8_sensory_vs_assoc":  (0.0, 1.0),   # sensory should sit >= association
    "M10_heritability_r":   (0.20, 1.00),
    "M10_lesion_retention": (0.70, 1.10),
}


def _m_in_band(key, val):
    b = M_BANDS.get(key)
    if b is None or val is None:
        return None
    return 1 if (b[0] <= val <= b[1]) else 0


# --------------------------------------------------------------------
# Stats container — attribute names match what neuraxon.gfactor reads.
# --------------------------------------------------------------------
class Stats:
    __slots__ = ("food_found", "food_taken", "explored", "time_lived_s",
                 "mates_performed", "energy_efficiency",
                 "branching", "fitness", "g_factor")

    def __init__(self):
        self.food_found = 0
        self.food_taken = 0
        self.explored = 0
        self.time_lived_s = 0.0
        self.mates_performed = 0
        self.energy_efficiency = 0.0
        self.branching = 1.0
        self.fitness = 0.0
        self.g_factor = 0.0

    def as_dict(self):
        return {
            "food_found": self.food_found,
            "food_taken": self.food_taken,
            "explored": self.explored,
            "time_lived": round(self.time_lived_s, 1),
            "mates_performed": self.mates_performed,
            "energy_efficiency": round(self.energy_efficiency, 3),
            "branching": round(self.branching, 3),
            "fitness": round(self.fitness, 4),
            "g": round(self.g_factor, 4),
        }


# --------------------------------------------------------------------
# A single NxEr (agent). Exposes the attributes neuraxon.gfactor needs:
#   .stats (food_taken/explored/mates_performed/time_lived_s)
#   .visited (set)  .known_food_ids (set)  .last_sing_level
#   .net.branching_ratio  .born_ts  ._g_score
# --------------------------------------------------------------------
class _NetShim:
    """gfactor reads nxer.net.branching_ratio. The real brain lives in
    a worker process; this tiny shim carries the value back each tick."""
    __slots__ = ("branching_ratio",)

    def __init__(self):
        self.branching_ratio = 1.0


class NxEr:
    def __init__(self, nid, name, pos, params, owner_token=None,
                 password_hash=None, parents=(None, None)):
        self.id = nid
        self.name = name
        self.pos = list(pos)
        self.heading = random.randint(0, 7)
        self.alive = True
        self.food = 60.0
        self.energy = 100.0
        self.params = params
        # The brain lives in a BrainPool worker (added by the Engine).
        # net is a shim so gfactor + stats keep working unchanged.
        self.net = _NetShim()
        self.stats = Stats()
        self.visited = {self._cell(pos)}
        # v1.36 — explored is now a MONOTONIC counter (stats.explored)
        # incremented on entering a genuinely-new cell, while `visited`
        # is an LRU-bounded recent-cell set used only as the novelty
        # filter. This uncaps the explored metric (was frozen at 5000)
        # and means camping a food cluster no longer inflates it.
        self._visit_q = deque([self._cell(pos)])
        self._last_move_dir = (0, 0)   # momentum for exploration
        self.known_food_ids = set()
        self.last_sing_level = 0.0
        # Brain output cache. Initialized here so the Phase C hot loop
        # can use direct attribute access (`nx._last_out`) instead of
        # `getattr(nx, "_last_out", default)`. The default `(motors,
        # branching)` matches the historic getattr fallback.
        self._last_out = ([0] * 7, 1.0)
        self._last_sensory = None
        # stable per-NxEr colour, exactly Lite's randomColor():
        # rgb with each channel in [30,235]. Sent in public_view so
        # every client renders the same colours (and they match Lite).
        self.color = "rgb(%d,%d,%d)" % (
            random.randint(30, 235), random.randint(30, 235),
            random.randint(30, 235))
        self.born_ts = time.time()
        self.born_tick = 0
        self._g_score = 0.0
        # ownership / auth
        self.owner_token = owner_token          # session token of owner
        self.password_hash = password_hash      # only valid while alive
        self.is_managed = password_hash is not None
        # lineage
        self.parents = list(parents)
        self.offspring_ids = []
        self.nas_trial = None        # v1.51 — set on system-spawned NAS explorers
        # behaviour bookkeeping
        self.last_move_tick = 0
        self.last_steal_tick = -10000  # v1.46 — theft cooldown gate
        self.last_eat_tick = -10000   # so first eat is allowed immediately
        self._da_accum = 0.0          # v1.52 — reward (dopamine) banked since last brain step
        # v1.53 — within-life foraging curve. food_found is sampled at fixed
        # AGES, so at death we can ask whether this NxEr got better at
        # foraging as it aged (real within-life learning) instead of only
        # correlating lifetime totals with lifespan, which is confounded by
        # survivorship (good foragers live longer in a no-learning world too).
        self.food_by_age = {}
        self._ck_i = 0
        # v1.54 — per-NxEr M-compliance (measurement only; see
        # _score_m_compliance). None until this brain's first science sample.
        self.m_score = None        # instantaneous fraction of bands in-band
        self.m_score_ema = None    # smoothed — the one to correlate on
        self.m_fit = None          # v1.55 continuous graded compliance (no ties)
        self.m_fit_ema = None      # v1.55 smoothed graded compliance
        self.m_fit_w = None        # v1.58 hard-band-weighted compliance
        self.m_in_band = 0
        self.m_n_checked = 0
        self.m_deviation = None    # mean band-widths outside, when out
        self.m_samples = 0
        self.m_last = None         # this brain's own raw M values
        # After each bite the hunger-toward-food floor is suppressed
        # until this tick — so the NxEr wanders away from a food
        # source for a while before being dragged back. Mirrors the
        # "satiation" / post-meal exploration biological pattern and
        # stops them from camping the same food cell continuously.
        self.wander_until_tick = 0
        self.mate_cooldown_until = 0
        self.mating_with = None
        # Mating overhaul (matches Neuraxon v189 offline reference):
        # NxErs have a sex; only opposite-sex pairs can mate; BOTH
        # parties must signal MateIntent within the same window for a
        # mating to actually trigger. mngol5 v0.8 only checked the
        # current NxEr's mate output and skipped sex entirely, which is
        # why mating almost never happened.
        self.is_male = random.random() < 0.5
        self.mating_intent_until_tick = 0
        self._last_sensory = None     # most recent sensor vector
        self._last_out = ([0] * 7, 1.0)   # reused between LOD steps
        # Terrain specialisation (matches offline v189). At birth:
        #   spawned on LAND → can_land=True, can_sea=False
        #   spawned on SEA  → can_land=False, can_sea=True
        # Offspring at the shore (one parent on land, the other on
        # sea) inherit BOTH — they're the only way to get an
        # amphibious NxEr. The engine sets these at spawn.
        self.can_land = True
        self.can_sea = False

    @staticmethod
    def _cell(p):
        return (int(p[0]), int(p[1]))

    def public_view(self):
        """Minimal info every viewer may see (no internals)."""
        return {
            "id": self.id, "name": self.name,
            "x": self.pos[0], "y": self.pos[1],
            "alive": self.alive,
            "managed": self.is_managed,
            "c": self.color,
            "s": 1 if self.last_sing_level > 0 else 0,
        }

    def owner_view(self):
        """Full detail — only for the authenticated owner / god."""
        d = self.public_view()
        # `energy` was a leftover field set once to 100.0 and never
        # updated. Compute it as food-percentage so it actually tracks
        # the NxEr's vitality (100 = just spawned / full belly, 0 =
        # about to die).
        full = getattr(self, "start_food", 0) or 120.0
        energy_pct = max(0.0, min(100.0, self.food / full * 100.0))
        d.update({
            "stats": self.stats.as_dict(),
            "food": round(self.food, 1),
            "energy": round(energy_pct, 1),
            "heading": self.heading,
            "born_tick": self.born_tick,
            "parents": self.parents,
            "offspring_ids": list(self.offspring_ids),
            "brain": {
                "topology": getattr(self.params, "sphere_topology",
                                    "chc6"),
                "spheres": ["visual", "auditory", "intero",
                            "assoc_fluid", "assoc_cryst", "motor"],
                "branching_ratio": round(
                    getattr(self.net, "branching_ratio", 0.0), 4),
            },
        })
        return d

    def export_model(self, brain_dict):
        """Serialisable params + brain state for save/restore.
        brain_dict is fetched from the BrainPool by the Engine."""
        return {
            "name": self.name,
            "params": _params_to_dict(self.params),
            "brain": brain_dict,
            "stats": self.stats.as_dict(),
        }


# --------------------------------------------------------------------
# Parameter (de)serialisation — only the knobs a user is allowed to set
# plus everything needed to faithfully rebuild a brain.
# --------------------------------------------------------------------
# User-tunable brain knobs exposed as SLIDERS in the client. Every value
# is range-clamped server-side (min,max). The brain topology is ALWAYS
# the CHC g-capable 6-sphere architecture — it is NOT user-selectable.
USER_TUNABLE = {
    # v1.39 — cap user-created brains at 24 hidden neurons (was 48).
    # A 48-neuron brain is ~2x the per-step compute of the trial-53
    # arch's 24, so an unmanaged user NxEr could be twice as expensive
    # as an autonomous one; capping keeps every NxEr's brain cost on
    # the same budget, which matters on the path to 1000s of NxErs.
    "num_hidden_neurons":          (4,    24,   int),
    "connection_probability":      (0.05, 0.6,  float),
    "learning_rate":               (0.001, 0.05, float),
    "spontaneous_firing_rate":     (0.0,  0.12, float),
    "intrinsic_timescale_default": (4.0,  60.0, float),
    "firing_threshold_excitatory": (0.3,  0.8,  float),
    "plasticity_threshold":        (0.2,  0.9,  float),
    "afferent_synapse_strength":   (0.5,  2.5,  float),
    "sensory_input_gain":          (0.3,  2.0,  float),
    "adaptation_tau_ticks":        (5.0,  60.0, float),
    "resting_potential_decay":     (0.05, 0.4,  float),
    "refractory_period_ticks":     (0,    8,    int),
    "post_spike_mp_reset":         (0.0,  1.0,  float),
    "cross_sphere_coupling":       (0.1,  3.0,  float),
    "cryst_capacity":              (0.3,  3.0,  float),
    "free_energy_beta":            (0.5,  2.5,  float),
    "symmetric_stdp":              (0,    1,    bool),
}


def _params_to_dict(p):
    out = {}
    for k in dir(p):
        if k.startswith("_"):
            continue
        v = getattr(p, k, None)
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


# v1.34 — optional per-birth genetic operator. Perturbs the heritable
# neural traits by ±strength (relative), clamped to physiological ranges.
# Crucially includes firing_threshold_inhibitory (the v195 E/I lever),
# which verbatim inheritance left frozen. Enabled only when
# offspring_mutation_strength > 0; default keeps the historic behaviour.
_MUTABLE = {
    # key: (lo, hi) clamp
    #
    # v1.58 — PRUNED. Across the V1.076 and V1.077 runs (15,119 trials,
    # 23 knobs) only a handful moved any outcome; the rest sat at
    # |r| < 0.05 against m_fit, m_score, M1_E and fitness_hi alike.
    # Sampling them added architectural variance for no information — the
    # population's mean compliance fell from ~0.52 to ~0.44 when the 13
    # extra knobs went in — so they are retired to _MUTABLE_RETIRED below,
    # where they stay documented and one line from being restored.
    #
    # refractory_period_ticks is the dominant parameter found so far:
    # r = -0.766 with m_fit (58.7% of variance) and r = -0.752 with M1_E.
    # M1_E (band 0.18-0.28) reads 0.219 at refr 2.5 and 0.181 at 3.5, but
    # only 0.151 at 4.6 — so the band is satisfiable ONLY below ~4. The
    # old floor of 2.0 still showed M1_E descending, i.e. the optimum may
    # lie under the box, so the range now opens at 1.0.
    "refractory_period_ticks": (1.0, 5.0),
    # second-strongest, and the only other knob above |r| = 0.05
    "firing_threshold_excitatory": (0.12, 0.80),
    # weak but repeatedly significant (t = +3.8 / -3.7 in v1.077)
    "sensorimotor_coupling": (0.1, 3.0),
    "post_spike_mp_reset": (0.0, 1.0),
    # direct M1_E levers: small effects (t = +3.5 / +3.2) but they act on
    # exactly the claim that fails most, so they stay in the search
    "target_excitatory_fraction": (0.10, 0.35),
    "max_excitatory_fraction": (0.18, 0.45),
    # structural; near-null so far but cheap and mechanistically central
    "connection_probability": (0.10, 0.60),
    "learning_rate": (0.0005, 0.08),
}

# v1.58 — retired knobs. Each was sampled across its full range for at
# least one multi-day run and showed |r| < 0.05 with every logged outcome.
# Kept here so the negative result is not lost and so any can be moved
# back into _MUTABLE in one edit if a hypothesis warrants it.
_MUTABLE_RETIRED = {
    "firing_threshold_inhibitory": (-1.30, -0.30),   # r = -0.007
    "intrinsic_timescale_default": (5.0, 40.0),      # r = -0.011
    "agmp_eta": (0.0005, 0.02),                      # r = -0.020
    "homeostatic_rate": (0.00005, 0.002),            # r = +0.001
    "target_firing_rate": (0.02, 0.25),              # r = +0.019
    "weight_homeostasis_target": (0.15, 0.90),       # r = -0.004
    "weight_homeostasis_rate": (0.002, 0.10),        # r = +0.021
    "weight_saturation_threshold": (0.35, 0.95),     # r = +0.001
    "hebbian_ltp_rate": (0.02, 0.30),                # r = -0.007
    "ltd_neutral_scale": (0.02, 0.30),               # r = +0.013
    "synapse_formation_prob": (0.002, 0.08),         # r = -0.005
    "adaptive_threshold_adjustment": (0.02, 0.30),   # r = +0.003
    "oscillator_strength": (0.0, 0.35),              # r = -0.000
    "max_intrinsic_timescale": (30.0, 400.0),        # r = -0.009
    "min_excitatory_fraction": (0.05, 0.25),         # r = +0.007
}


# v1.55 — some NAS knob names are not the attribute the substrate reads.
# Sampling them was a silent no-op: the value was logged in nas_trials but
# never reached a neuron, which is why they all showed r ~ 0 with every
# outcome. Map each to the attribute that is actually read.
_ARCH_ALIAS = {
    # neurons read membrane_time_constant (neuron.py: intrinsic_timescale
    # = membrane_time_constant); intrinsic_timescale_default is inert.
    "intrinsic_timescale_default": "membrane_time_constant",
    # there is no 'sensorimotor_coupling' attribute at all; the real
    # inter-sphere drive knob is cross_sphere_coupling.
    "sensorimotor_coupling": "cross_sphere_coupling",
}


def _mutate_params(pd, strength):
    out = dict(pd)
    for k, (lo, hi) in _MUTABLE.items():
        v = out.get(k)
        if not isinstance(v, (int, float)):
            continue
        scale = abs(v) if v != 0 else (hi - lo) * 0.1
        nv = v + random.uniform(-strength, strength) * scale
        if nv < lo:
            nv = lo
        elif nv > hi:
            nv = hi
        out[k] = nv
    # refractory is an int tick count — mutate discretely
    r = out.get("refractory_period_ticks")
    if isinstance(r, (int, float)) and random.random() < strength * 3:
        out["refractory_period_ticks"] = max(0, min(8, int(r)
                                                    + random.choice((-1, 1))))
    return out


def _apply_heritable(params_obj, pd):
    """Force the heritable neural traits from a parent/mutated dict onto
    a NetworkParameters object, BYPASSING the USER_TUNABLE whitelist that
    make_params() applies. This is the inheritance path: without it,
    make_params() resets non-tunable traits (notably
    firing_threshold_inhibitory — the v195 E/I lever) to the arch value
    every birth, which is exactly why it was frozen across generations.
    Only used when offspring_mutation_strength > 0, so default behaviour
    is unchanged."""
    for k in _MUTABLE:
        v = pd.get(k)
        if not isinstance(v, (int, float)) or not hasattr(params_obj, k):
            continue
        try:
            cur = getattr(params_obj, k)
            setattr(params_obj, k, int(round(v))
                    if isinstance(cur, int) and not isinstance(cur, bool)
                    else float(v))
        except (TypeError, ValueError):
            pass
    # refractory + the timescale alias make_params keeps in sync
    r = pd.get("refractory_period_ticks")
    if isinstance(r, (int, float)) and hasattr(params_obj,
                                               "refractory_period_ticks"):
        try:
            params_obj.refractory_period_ticks = int(round(r))
        except (TypeError, ValueError):
            pass
    ts = pd.get("intrinsic_timescale_default")
    if isinstance(ts, (int, float)):
        if hasattr(params_obj, "membrane_time_constant"):
            params_obj.membrane_time_constant = max(5.0, min(60.0, float(ts)))
    return params_obj


def make_params(overrides=None):
    p = NetworkParameters()
    # ---- 1. apply the NAS-best architecture (tuned, fitness 6.88) ----
    # neural section
    _na = lambda k, d: arch_get("neural", k, d)
    try:
        p.num_hidden_neurons = max(
            3, int(round(float(_na("num_hidden_neurons_default",
                                   p.num_hidden_neurons)))))
        p.connection_probability = float(_na(
            "connection_probability", p.connection_probability))
        p.afferent_synapse_strength = float(_na(
            "afferent_synapse_strength",
            getattr(p, "afferent_synapse_strength", 1.1)))
        p.sensory_input_gain = float(_na(
            "sensory_input_gain",
            getattr(p, "sensory_input_gain", 0.9)))
        p.firing_threshold_excitatory = float(_na(
            "firing_threshold_excitatory",
            p.firing_threshold_excitatory))
        p.firing_threshold_inhibitory = float(_na(
            "firing_threshold_inhibitory",
            p.firing_threshold_inhibitory))
        p.spontaneous_firing_rate = float(_na(
            "spontaneous_firing_rate", p.spontaneous_firing_rate))
        p.resting_potential_decay = float(_na(
            "resting_potential_decay",
            getattr(p, "resting_potential_decay", 0.2)))
        _itd = _na("intrinsic_timescale_default", None)
        if _itd is not None:
            p.membrane_time_constant = max(5.0, min(60.0, float(_itd)))
            if hasattr(p, "intrinsic_timescale_default"):
                p.intrinsic_timescale_default = float(_itd)
        for k in ("symmetric_stdp", "refractory_period_ticks",
                  "post_spike_mp_reset", "cross_sphere_coupling",
                  "cryst_capacity", "free_energy_beta"):
            v = _na(k, None)
            if v is not None and hasattr(p, k):
                cur = getattr(p, k)
                if isinstance(cur, bool):
                    setattr(p, k, bool(v))
                elif isinstance(cur, int):
                    setattr(p, k, int(round(float(v))))
                else:
                    setattr(p, k, float(v))
        # operating_ranges section
        _or = lambda k, d: arch_get("operating_ranges", k, d)
        p.learning_rate = float(_or("learning_rate", p.learning_rate))
        if hasattr(p, "plasticity_threshold"):
            p.plasticity_threshold = float(_or(
                "plasticity_threshold", p.plasticity_threshold))
        if hasattr(p, "adaptation_tau_ticks"):
            p.adaptation_tau_ticks = float(_or(
                "adaptation_tau_ticks",
                getattr(p, "adaptation_tau_ticks", 20.0)))
    except Exception:
        pass
    # ---- 2. the CHC g-capable 6-sphere brain is mandatory ----
    p.sphere_topology = "chc6"
    # ---- 3. user slider overrides (range-clamped) ----
    if overrides:
        for k, v in overrides.items():
            spec = USER_TUNABLE.get(k)
            if spec is None or not hasattr(p, k):
                continue
            lo, hi, typ = spec
            try:
                if typ is bool:
                    setattr(p, k, bool(v))
                elif typ is int:
                    setattr(p, k, int(max(lo, min(hi, int(float(v))))))
                else:
                    setattr(p, k, float(max(lo, min(hi, float(v)))))
            except (TypeError, ValueError):
                pass
    # mngol5 v1.41 — optional global AGMP override, read from the env so it
    # reaches the worker processes too (they build the brains). AGMP runs a
    # full per-synapse plasticity pass every step but is gated by phasic
    # dopamine, which sits at baseline because the reward/learning loop is
    # not wired — so it burns CPU on unfocused weight drift, not learning.
    # Disabling it makes the brain step materially cheaper; the trinary
    # firing distribution is held by SEPARATE homeostatic scaling, so the
    # core science readout is preserved. Reversible when the loop ships.
    _agmp = os.environ.get("MNGOL5_AGMP")
    if _agmp is not None:
        p.agmp_enabled = (_agmp == "1")
    elif _LEARNING_AGMP:
        p.agmp_enabled = True        # v1.52 — learning loop on by config
    # ---- 4. v1.55 — architecture / NAS knobs -------------------------
    # These are NOT user-facing sliders and must not be filtered through
    # USER_TUNABLE. Doing so silently dropped firing_threshold_inhibitory
    # and sensorimotor_coupling (absent from that table) and clamped
    # firing_threshold_excitatory to >= 0.30 and learning_rate to <= 0.05,
    # so 3 of 7 sampled knobs never reached a brain and 2 more were
    # truncated. Applied last, with _MUTABLE's own ranges, via the alias
    # map so each lands on the attribute the substrate actually reads.
    if overrides:
        for k, (lo, hi) in _MUTABLE.items():
            if k not in overrides:
                continue
            attr = _ARCH_ALIAS.get(k, k)
            if not hasattr(p, attr):
                continue
            try:
                setattr(p, attr, float(max(lo, min(hi, float(overrides[k])))))
            except (TypeError, ValueError):
                pass
    p.sphere_topology = "chc6"   # never overridable
    return p


# --------------------------------------------------------------------
# World — terrain grid + food.
# --------------------------------------------------------------------
class World:
    def __init__(self, size, sea_pct, rock_pct, seed=0,
                 earth_map=False, pole_frac=None):
        self.size = int(size)
        self.sea_pct = float(sea_pct)
        self.rock_pct = float(rock_pct)
        self.earth_map = bool(earth_map)
        # Sphere projection collapses the top/bottom rows toward the
        # poles. y=0 is literally a single 3D point, y=1 a tiny ring,
        # etc. To stop NxErs and food from piling on top of each other
        # at that infinitesimal apex we BAN a thin band of rows. 7%
        # was MUCH too generous on big worlds (N=1000 → 70 rows ≈ a
        # 13° cap at each pole, the huge empty disc the user noticed).
        # Default now 1% — at N=1000 → 10 rows ≈ a 1.8° cap, basically
        # invisible. Tunable via world_config "pole_frac".
        if pole_frac is None:
            pole_frac = 0.01
        self._pole = max(1, int(self.size * float(pole_frac)))
        self.terrain = {}
        if self.earth_map:
            # Earth-map mode (same as Game of Life Lite "useEarth"):
            # nearest-neighbour scale the embedded ASCII Earth map onto
            # the world grid.  '.' → sea, ':' → land, '^' → rock.
            from .earth_map import EARTH_MAP, MAP_H, MAP_W
            for x in range(self.size):
                for y in range(self.size):
                    my = min(MAP_H - 1, int(y / self.size * MAP_H))
                    mx = min(MAP_W - 1, int(x / self.size * MAP_W))
                    row = EARTH_MAP[my]
                    ch = row[mx] if mx < len(row) else "."
                    if ch == "^":
                        self.terrain[(x, y)] = 2     # rock / mountains
                    elif ch == ":":
                        self.terrain[(x, y)] = 0     # land
                    else:
                        self.terrain[(x, y)] = 1     # sea
        else:
            # Procedural value-noise terrain.
            rnd = random.Random(seed)
            for x in range(self.size):
                for y in range(self.size):
                    r = rnd.random()
                    if r < sea_pct:
                        self.terrain[(x, y)] = 1
                    elif r < sea_pct + rock_pct:
                        self.terrain[(x, y)] = 2
                    else:
                        self.terrain[(x, y)] = 0
        self._finalize_flat()

    def in_pole(self, y):
        """True for the unreachable polar bands (top & bottom rows)."""
        return y < self._pole or y >= self.size - self._pole

    def _finalize_flat(self):
        """Build a flat bytearray of terrain codes indexed by y*N + x.
        At 2500 alive NxErs the per-tick code calls is_sea() / passable()
        thousands of times — dict.get((x,y), 0) is one of the hottest
        operations on the main core. The flat array is read via direct
        integer indexing which is roughly 5× faster than the dict and
        also frees the int(x), int(y) defensive conversions (positions
        are integers everywhere). At this point we also DROP the dict:
        every hot caller uses `_t` and the legacy method shims read it
        too, so the dict was duplicate memory (an N=2400 earth-map dict
        is ~250 MB on PyPy; the bytearray is 5.7 MB)."""
        N = self.size
        ba = bytearray(N * N)
        for (x, y), v in self.terrain.items():
            ba[y * N + x] = v
        self._t = ba
        self._N = N
        # release the dict; subsequent reads go through self._t
        self.terrain = None

    def passable(self, x, y):
        N = self._N
        if not (0 <= x < N and 0 <= y < N):
            return False
        if y < self._pole or y >= N - self._pole:
            return False
        return self._t[y * N + x] != 2

    def is_sea(self, x, y):
        N = self._N
        if not (0 <= x < N and 0 <= y < N):
            return False
        return self._t[y * N + x] == 1

    def terrain_rows(self):
        """Compact terrain for the client: one char per cell per row.
        '.' sea · ',' land · '#' rock. Sent once over REST (static).
        Reads from the flat bytearray (the legacy `terrain` dict was
        dropped to save memory)."""
        N = self._N
        t = self._t
        CHARS = b",.#"      # 0=land, 1=sea, 2=rock
        out = []
        for y in range(N):
            base = y * N
            # one bytes slice → translate to chars in a single C call
            out.append(bytes(CHARS[t[base + x]] for x in range(N))
                       .decode("ascii"))
        return out


# --------------------------------------------------------------------
# The engine.
# --------------------------------------------------------------------
class Engine:
    def _init_runtime(self, cfg):
        """All derived/runtime state. Called by BOTH __init__ and the
        crash-restore path so the two can never drift out of sync
        (this is what caused the missing _cell_size on restart)."""
        # v1.51 — embedded NAS. Every NxEr the SYSTEM spawns (founders,
        # anti-extinction respawns, and the steady explorer trickle) gets a
        # freshly sampled architecture so we can measure which brains do
        # best; only mating offspring inherit genetics. Each explorer's
        # architecture + lifetime outcomes are logged once, at death.
        #
        # v1.52 — learning loop. AGMP plasticity runs in the substrate every
        # brain step; the missing piece was a reward that bursts phasic
        # dopamine so the eligibility traces consolidate. Reward is banked
        # per NxEr between brain steps (LOD-safe) and applied on the next.
        #
        # v1.53 — these MOVED here from __init__. _restore() builds the
        # Engine via Engine.__new__ and only calls _init_runtime, so with
        # them in __init__ a snapshot reboot produced an Engine with no
        # _learning / _nas_enabled and step() died on the first tick with
        # AttributeError. Any restart from snapshot was fatal.
        global _LEARNING_AGMP
        self._nas_enabled = bool(cfg.get("nas_explore_enabled", True))
        self._nas_interval = int(cfg.get("nas_explore_interval_ticks", 500))
        # preserve the trial counter across a restore so NAS trial ids
        # stay unique in the log (fresh boot starts at 0)
        if not isinstance(getattr(self, "_nas_trial_seq", None), int):
            self._nas_trial_seq = 0
        self._learning = bool(cfg.get("learning_enabled", True))
        _LEARNING_AGMP = self._learning
        self._da_base = float(cfg.get("dopamine_baseline", 0.15))
        self._da_max = float(cfg.get("dopamine_burst_max", 0.85))
        self._reward_eat_da = float(cfg.get("reward_eat_dopamine", 0.5))
        self._reward_mate_da = float(cfg.get("reward_mate_dopamine", 0.7))
        # v1.57 — M-CLAIM SELECTION. Until now nothing in the world cared
        # whether a brain satisfied the M claims: m_fit was measured and
        # logged, but survival depended only on foraging luck, and the
        # 6.87-day V1.076 run showed the most M-compliant quartile actually
        # STARVING MORE (43% of deaths vs 23% for the least compliant).
        # Compliance was a liability.
        #
        # This makes the M targets a real selection pressure, on the two
        # channels that actually kill NxErs (63% idle, 37% starved):
        #   * metabolic: a compliant brain spends less energy per tick
        #     (efficient neural coding is cheaper — the biological
        #     rationale, and it directly offsets the starvation penalty);
        #   * idle tolerance: a compliant brain gets longer before the
        #     stuck-cull fires.
        # Both scale with m_fit_ema (0..1, continuous and strongly
        # heritable at r=+0.43), are bounded, and are off when
        # m_selection_enabled is false — so a run can be split cleanly into
        # with/without segments.
        self._m_sel = bool(cfg.get("m_selection_enabled", True))
        self._m_sel_drain = float(cfg.get("m_selection_drain_relief", 0.60))
        self._m_sel_idle = float(cfg.get("m_selection_idle_bonus", 0.50))
        self._m_sel_mate = float(cfg.get("m_selection_mate_relief", 0.25))
        # v1.58 — the V1.077 run showed 35% drain relief was nowhere near
        # enough: starvation still climbed from 25% (least compliant decile)
        # to 57% (most compliant), so compliance stayed a net liability and
        # no champion passed half the run. Three changes:
        #   * relief raised to 0.60;
        #   * relief concentrated where it decides life or death — scaled up
        #     as an NxEr approaches starvation rather than paid flat to
        #     already-full agents;
        #   * a foraging bonus, because the deficit is on the INCOME side:
        #     whatever makes a brain compliant also makes it worse at
        #     finding food, so relief alone only slows the bleed.
        self._m_sel_hunger = float(cfg.get("m_selection_hunger_focus", 1.5))
        self._m_sel_forage = float(cfg.get("m_selection_forage_bonus", 0.35))
        # Rescale m_fit against a floor before using it as an advantage.
        # In the V1.076 run m_fit spanned 0.742..0.991 across deciles, so
        # feeding it in raw would hand ~the same relief to everyone and
        # exert almost no selection. Mapping [floor,1] -> [0,1] turns that
        # narrow band into a real gradient (0.74 -> 0.14, 0.99 -> 0.97).
        self._m_sel_floor = float(cfg.get("m_selection_floor", 0.70))
        # v1.58 — adaptive NAS state (see _nas_sample)
        self._nas_adaptive = bool(cfg.get("nas_adaptive", True))
        self._nas_elite_prob = float(cfg.get("nas_elite_prob", 0.70))
        self._nas_elite_frac = float(cfg.get("nas_elite_frac", 0.25))
        self._nas_sd_inflate = float(cfg.get("nas_sd_inflate", 1.5))
        self._nas_min_hist = int(cfg.get("nas_min_history", 60))
        if not isinstance(getattr(self, "_last_per_brain", None), dict):
            self._last_per_brain = {}      # v1.58 — W audit source
        if not isinstance(getattr(self, "_nas_hist", None), deque):
            self._nas_hist = deque(
                maxlen=int(cfg.get("nas_history", 600)))
        # v1.34 — science history logger (set by GameServer after init;
        # None means logging disabled). The engine only ever ENQUEUES
        # records (cheap); a background thread does the file I/O.
        if not hasattr(self, "history"):
            self.history = None
        # baseline for per-interval rate deltas (births/deaths/matings)
        self._hist_prev = None
        self._hist_last_tick = 0
        self._obit_rate = float(cfg.get("obituary_sample_rate", 1.0))
        self.dt = 1.0 / float(cfg.get("global_time_steps", 30))
        # max_atrophy: the new NAS arch sets this to ~11.8, which makes
        # a NxEr that is idle for even half a second burn food ~8x
        # faster and die in ~20s — far too harsh for a multiplayer
        # world where NxErs routinely pause (blocked by a neighbour,
        # resting, random 0,0 brain output). A world-config
        # "max_atrophy" overrides the arch value; default 5.0 restores
        # the gentle pre-arch behaviour.
        cfg_atrophy = cfg.get("max_atrophy", 0)
        if cfg_atrophy and float(cfg_atrophy) > 0:
            self.bio_max_atrophy = float(cfg_atrophy)
        else:
            self.bio_max_atrophy = min(5.0, float(
                arch_get("biology", "max_atrophy", 4.46)))
        self.bio_metab_ramp = float(
            arch_get("biology", "metabolic_ramp_per_sec", 13.0))
        # config can also scale the idle ramp steepness (1.0 = arch
        # value). Lower = gentler idle penalty.
        self.bio_metab_ramp *= float(cfg.get("metabolic_ramp_scale", 1.0))
        self.bio_start_food = float(
            cfg.get("start_food",
                    arch_get("biology", "start_food_default", 120.0))
            or 120.0)
        if self.bio_start_food < 60.0:
            self.bio_start_food = 120.0
        self.bio_idle_explore_s = float(
            arch_get("biology", "idle_explore_seconds", 2.0))
        self.bio_explore_prob = float(
            arch_get("biology", "explore_probability", 0.23))
        self.bio_base_drain = float(cfg.get("base_food_drain", 0.018))
        # v1.49 — reproduction now requires a NxEr to be well-fed, i.e. a
        # competent forager. Default 5 keeps the historic behaviour; the
        # shipped world_config raises it so foraging skill (a brain task)
        # gates reproduction, giving selection something brain-related to
        # act on instead of pure spatial luck.
        self.bio_mate_min_food = float(cfg.get("mate_min_food", 5))
        # Exploration tuning. hunger_threshold_pct < 1.0 keeps NxErs
        # in wandering mode while they still have appreciable food,
        # rather than the previous 0.85 which kept them perpetually
        # seeking. food_wander_ticks is how long the hunger floor
        # stays silent after each bite so the brain (random walk)
        # drives a real excursion before the floor re-engages.
        self.bio_hunger_threshold = float(
            cfg.get("hunger_threshold_pct", 0.55))
        self.bio_food_wander_ticks = int(
            cfg.get("food_wander_ticks", 40))
        # v1.46 — three balance levers (all configurable):
        #  * max_nxer_energy: hard cap on a NxEr's stored food. Without
        #    it a few NxErs hoard thousands (seen: 7700+), which (a) makes
        #    them giant theft targets and (b) defeats the idle-atrophy
        #    drain (a 7700-food NxEr survives ~85k ticks even stuck), so
        #    stuck hoarders never die. Capping restores both. "Full" in
        #    the hunger model is ~60, so 150 still allows a 2.5x reserve.
        #  * idle_death_ticks: a NxEr that hasn't MOVED in this many ticks
        #    is culled outright (0 = disabled). Foraging NxErs move every
        #    ~food_wander_ticks via the post-meal excursion, so only the
        #    genuinely stuck (blocked at a mountain edge / packed cluster)
        #    are caught. 1800 ticks ~= 90 s at 20 tps.
        #  * steal_cooldown_ticks: min ticks between thefts by one NxEr.
        #    Eating already has eat_cooldown_ticks (20); stealing had
        #    NONE, so theft was strictly better than foraging and ran away
        #    (seen: 21 827 stolen vs ~760 foraged). Matching the eat
        #    cooldown puts theft on equal footing with foraging.
        self.bio_energy_cap = float(cfg.get("max_nxer_energy", 150.0))
        self.bio_idle_death_ticks = int(cfg.get("idle_death_ticks", 1800))
        self._steal_cd = int(cfg.get("steal_cooldown_ticks", 20))
        self._cell_size = max(4, int(cfg["world_size"]) // 24)
        self._food_grid = {}
        self._food_at = {}
        self._nxer_grid = {}
        # one NxEr per cell — `(x, y) -> nxer_id`. Maintained
        # incrementally by _spawn_nxer / _kill / movement so two
        # NxErs can never occupy the same grid cell (offline v189
        # semantics; before this fix NxErs slid through each other
        # because only food and rock were collision-blockers).
        self._occupied = {}
        self._events = []          # [{t,x,y}] visual FX, cleared each tick
        # full-population rank pool (id -> best value, per metric)
        self._rank_pool = {m: {} for m in RANK_METRICS}
        # v1.43 — id+value of the current all-time record-holder per
        # metric, captured the moment a LIVE NxEr breaks the record so
        # the game server can export its full brain before it dies.
        self._record_breakers = {}
        self._rank_index = {m: {} for m in RANK_METRICS}
        self._rank_top = {m: [] for m in RANK_METRICS}
        self._nxer_names = {}        # id -> name (survives pruning)
        # v1.44 — main-process M-metric accumulators (behavioural metrics
        # computed from the sensory/motor vectors the act loop already
        # has; reset each history sample). M7 self-sustained activity:
        # motor amplitude in zero-external-drive vs driven windows.
        self._m12_on = bool(self.cfg.get("m12_metrics_enabled", True))
        self._m7_zero_sum = 0.0
        self._m7_zero_n = 0
        self._m7_drv_sum = 0.0
        self._m7_drv_n = 0
        # M10 heritability: parent-avg fitness recorded at a child's birth
        # (persists across samples; paired with the child's current
        # fitness at report time). Capped to bound memory.
        if not hasattr(self, "_herit_pending"):
            self._herit_pending = {}   # child_id -> parent_avg_fitness
        self._food_dirty = True
        if not hasattr(self, "_g_cache"):
            self._g_cache = {"pc1": 0.0, "pos_manifold": 0.0,
                             "mean_r": 0.0, "lambda_ratio": 1.0,
                             "n": 0}
        from .brain_pool import BrainPool
        ew = cfg.get("engine_workers", 0)
        if not ew or int(ew) <= 0:
            # Auto-detect. os.cpu_count() reports SYSTEM cpus which can
            # be wrong inside containers/cgroups; sched_getaffinity is
            # the CPUs THIS process may actually use. Prefer it, fall
            # back to cpu_count, floor at 1.
            ncpu = None
            try:
                ncpu = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                ncpu = os.cpu_count()
            ncpu = ncpu or 1
            ew = max(1, ncpu - 1)
            if ew <= 1 and (os.cpu_count() or 1) > 2:
                # affinity said 1 but the box clearly has more cores —
                # something pinned us; use cpu_count instead so we don't
                # silently degrade to single-core in-process mode.
                ew = max(1, (os.cpu_count() or 1) - 1)
            print("[Engine] engine_workers auto-resolved to %d "
                  "(sched_affinity=%s cpu_count=%s)" % (
                      ew,
                      (len(os.sched_getaffinity(0))
                       if hasattr(os, "sched_getaffinity") else "n/a"),
                      os.cpu_count()))
        nb = cfg.get("brain_builders", 0)
        nb = int(nb) if nb and int(nb) > 0 else None
        # mngol5 v1.41 — publish the AGMP setting to the environment BEFORE
        # the workers spawn so they inherit it (brains are built inside the
        # workers, in a separate process from this one). v1.52: the learning
        # loop REQUIRES AGMP (it's how dopamine consolidates eligibility
        # traces), so learning_enabled forces it on even if the older
        # agmp_enabled flag is false. With learning off, agmp_enabled (default
        # True) still controls the reward-less plasticity pass.
        _agmp_on = (cfg.get("agmp_enabled", True)
                    or cfg.get("learning_enabled", True))
        os.environ["MNGOL5_AGMP"] = "1" if _agmp_on else "0"
        if not _agmp_on:
            print("[Engine] AGMP plasticity DISABLED via config "
                  "(cheaper brain step; reversible)")
        self.pool = BrainPool(
            int(ew), num_builders=nb,
            step_timeout=float(cfg.get("worker_step_timeout", 5.0)),
            reserved_builder_cores=int(
                cfg.get("reserved_builder_cores", 2)),
            pin_main_process=bool(cfg.get("pin_main_process", False)))
        # v1.39 — per-phase timing (EWMA, in ms) so we can SEE where the
        # tick budget goes on the path to 1000s of NxErs: Phase A (sense),
        # Phase B (brain pool round-trip), Phase C (act+metabolism), and
        # the whole tick. Surfaced in the admin console + history log.
        self._perf = {"a_ms": 0.0, "b_ms": 0.0, "c_ms": 0.0,
                      "tick_ms": 0.0, "due": 0, "alpha": 0.1}
        # v1.39 opt-in slim broadcast — foods/colors are near-static, so
        # when slim_broadcast is on we send the 1800-item food list and
        # the per-NxEr colour only periodically (the client caches them),
        # leaving per-frame frames to the fundamentals (id/pos/flags).
        self._last_food_bcast = 0.0
        self._last_food_count = -1
        self._last_color_bcast = 0.0

    def __init__(self, cfg, name_allocator):
        self.cfg = cfg
        self.names = name_allocator
        self.world = World(cfg["world_size"], cfg["sea_pct"],
                           cfg["rock_pct"], cfg.get("world_seed", 12345),
                           earth_map=cfg.get("earth_map", False),
                           pole_frac=cfg.get("pole_frac", 0.01))
        self.nxers = {}                 # id -> NxEr (alive + recently dead)
        self.foods = {}                 # food_id -> {pos, amount}
        self.tick = 0
        self.next_nxer_id = 0
        self.next_food_id = 0
        # all-time best record (survives deaths + restarts)
        self.all_time = {m: [] for m in RANK_METRICS}   # list of dicts
        # ----- Lifetime counters (persisted, monotonic) -------------
        # Cumulative stats for the whole server's history. Each event
        # below increments the counter once, even when state survives
        # restarts (loaded from snapshot.json).
        self.lifetime = {
            "started_at_unix": time.time(),
            "uptime_seconds":  0.0,    # accumulated across restarts
            "total_ticks":     0,      # accumulated across restarts
            "total_spawns":    0,      # every NxEr ever created
            "total_managed_registrations": 0,  # registered by a user
            "total_births_mating": 0,  # NxErs born from mating
            "total_deaths":    0,
            "total_food_eaten":0,
            "total_food_spawned":0,
            "total_matings":   0,
            "peak_alive":      0,
            "peak_managed":    0,
        }
        self._uptime_t0 = time.time()
        self._g_cache = {"pc1": 0.0, "pos_manifold": 0.0,
                         "mean_r": 0.0, "lambda_ratio": 1.0, "n": 0}
        # NOTE: the NAS + learning runtime flags are deliberately set in
        # _init_runtime(), NOT here — see the comment there. _restore()
        # builds the Engine with __new__ and only calls _init_runtime.
        self._init_runtime(cfg)
        self._spawn_food(self._effective_food_cap())
        for _ in range(int(cfg["starting_nxers"])):
            if self._nas_enabled:
                self._spawn_explorer()
            else:
                self._spawn_nxer()

    # ---- spawning ---------------------------------------------------
    def _free_cell(self, want_land=True, want_sea=True):
        """Random passable cell, optionally restricted by terrain so
        an amphibious offspring spawns near its parents and a
        sea-specialist isn't dropped on land. Also avoids cells
        already occupied by another living NxEr (one NxEr per cell).
        Falls back to ANY passable if no matching cell is found."""
        occupied = self._occupied
        for _ in range(200):
            x = random.randint(0, self.world.size - 1)
            y = random.randint(0, self.world.size - 1)
            if not self.world.passable(x, y):
                continue
            if (x, y) in occupied:
                continue
            is_sea = self.world.is_sea(x, y)
            if (is_sea and want_sea) or (not is_sea and want_land):
                return [x, y]
        # fallback — accept any passable, unoccupied cell
        for _ in range(200):
            x = random.randint(0, self.world.size - 1)
            y = random.randint(0, self.world.size - 1)
            if self.world.passable(x, y) and (x, y) not in occupied:
                return [x, y]
        return [self.world.size // 2, self.world.size // 2]

    def _spawn_food(self, n):
        # Each food source has `remaining` 25 units (matches offline
        # v189). NxErs harvest 1 per tick while on the cell; food
        # only disappears when `remaining <= 0` — so a food source
        # persists for ~25 ticks of harvesting, not vanishing on
        # first touch.
        while len(self.foods) < n:
            fid = self.next_food_id
            self.next_food_id += 1
            self.foods[fid] = {"pos": self._free_cell(),
                               "remaining": 25}
            self.lifetime["total_food_spawned"] += 1
        self._food_dirty = True

    def _spawn_nxer(self, params=None, owner_token=None,
                    password_hash=None, parents=(None, None),
                    name=None, terrain_caps=None):
        if len([a for a in self.nxers.values() if a.alive]) \
                >= int(self.cfg["max_nxers"]):
            return None
        nid = self.next_nxer_id
        self.next_nxer_id += 1
        nm = name or self.names.next_name()
        # Pick spawn position respecting requested terrain caps; for
        # founders (no caps) the cell type decides their spec.
        if terrain_caps is None:
            pos = self._free_cell()
            is_sea = self.world.is_sea(pos[0], pos[1])
            can_land, can_sea = (not is_sea), is_sea
            if can_land == can_sea == False:    # paranoia
                can_land = True
        else:
            can_land, can_sea = terrain_caps
            pos = self._free_cell(want_land=can_land, want_sea=can_sea)
        nx = NxEr(nid, nm, pos,
                  params or make_params(), owner_token,
                  password_hash, parents)
        nx.can_land = can_land
        nx.can_sea = can_sea
        nx.food = self.bio_start_food
        nx.start_food = self.bio_start_food   # for energy% display
        nx.born_tick = self.tick
        self.pool.add(nid, _params_to_dict(nx.params))
        self.nxers[nid] = nx
        self._nxer_names[nid] = nx.name
        self._occupied[(nx.pos[0], nx.pos[1])] = nx.id
        # lifetime counters
        self.lifetime["total_spawns"] += 1
        if nx.is_managed:
            self.lifetime["total_managed_registrations"] += 1
        return nx

    def _nas_sample(self):
        """Draw one architecture from the search space.

        v1.58 — ADAPTIVE. Uniform sampling meant the explorer stream was
        permanently pulling the population back toward the middle of every
        range, fighting selection: in the V1.077 run selection drove the
        inherited refractory period from 6.67 down to 6.15 (r = -0.55)
        while immigration kept injecting a uniform mean of 7.0, and the
        compliance optimum is nearer 3. Coverage is what found that effect,
        so it is not abandoned — instead each explorer is drawn either from
        a distribution fitted to the best trials seen so far (exploit) or
        uniformly (explore), with the mix set by nas_elite_prob.

        This is a plain estimation-of-distribution / cross-entropy step:
        keep the top nas_elite_frac of a rolling history by m_fit, take the
        per-knob mean and sd, widen by nas_sd_inflate so the distribution
        cannot collapse, and clip to the declared range.
        """
        if (not self._nas_adaptive
                or len(self._nas_hist) < self._nas_min_hist
                or random.random() >= self._nas_elite_prob):
            return {k: round(random.uniform(lo, hi), 5)
                    for k, (lo, hi) in _MUTABLE.items()}
        # elite set: best nas_elite_frac of the rolling history
        hist = sorted(self._nas_hist, key=lambda t: -t[0])
        k_elite = max(8, int(len(hist) * self._nas_elite_frac))
        elite = hist[:k_elite]
        out = {}
        for k, (lo, hi) in _MUTABLE.items():
            vals = [a.get(k) for _, a in elite if a.get(k) is not None]
            if len(vals) < 4:
                out[k] = round(random.uniform(lo, hi), 5)
                continue
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            sd = (var ** 0.5) * self._nas_sd_inflate
            # never let a dimension collapse: keep >=10% of its range
            sd = max(sd, 0.10 * (hi - lo))
            v = random.gauss(m, sd)
            out[k] = round(lo if v < lo else (hi if v > hi else v), 5)
        return out

    def _nas_record(self, arch, score):
        """v1.58 — feed a finished trial back into the sampler's history."""
        if arch and score is not None:
            try:
                self._nas_hist.append((float(score), arch))
            except (TypeError, ValueError):
                pass

    def _spawn_explorer(self):
        """System-spawned NxEr carrying a sampled architecture, tagged so
        its lifetime outcome is logged at death for the NAS dataset."""
        arch = self._nas_sample()
        nx = self._spawn_nxer(params=make_params(arch))
        if nx is not None:
            self._nas_trial_seq += 1
            nx.nas_trial = {"id": self._nas_trial_seq, "arch": arch}
        return nx

    def register_user_nxer(self, param_overrides, password_hash,
                           owner_token, name=None):
        """Create a user-owned NxEr (name assigned server-side, or a
        pre-allocated one passed in by the queued-register path)."""
        params = make_params(param_overrides)
        nx = self._spawn_nxer(params=params, owner_token=owner_token,
                              password_hash=password_hash, name=name)
        return nx

    def _can_enter(self, nx, x, y):
        """world.passable() still blocks rock + poles uniformly; on top
        of that, a land-only NxEr cannot enter a sea cell and a
        sea-only NxEr cannot enter a land cell. Amphibious NxErs
        (born of a shore mating, can_land and can_sea both true) can
        enter either."""
        if not self.world.passable(x, y):
            return False
        is_sea = self.world.is_sea(x, y)
        return (nx.can_sea if is_sea else nx.can_land)

    # ---- sensory / motor codec -------------------------------------
    def _build_spatial(self):
        """Rebuild per-tick spatial hash buckets so _sense is O(1)
        per NxEr instead of O(food)+O(nxers). Essential at 1000s.
        The food grid is rebuilt ONLY when food changed (eat/spawn);
        the NxEr grid every tick since they move every tick."""
        cs = self._cell_size
        if self._food_dirty or not self._food_grid:
            fg = {}
            fat = {}
            for fid, f in self.foods.items():
                fx, fy = f["pos"]
                fg.setdefault((fx // cs, fy // cs), []).append((fx, fy))
                fat[(fx, fy)] = fid
            self._food_grid = fg
            self._food_at = fat
            self._food_dirty = False
        ng = {}
        for o in self.nxers.values():
            if o.alive:
                ox, oy = o.pos
                ng.setdefault((ox // cs, oy // cs), []).append(o.id)
        self._nxer_grid = ng

    def _sense(self, nx):
        """10 sensory channels (mirrors the Lite convention).
        Uses the spatial hash → scales to thousands of NxErs.

        Hot-loop micro-optimisations (these mattered at 2500+ alive):
        - Read terrain via the flat bytearray `world._t` directly
          (one indexed memory access vs `world.is_sea()` method call +
          dict.get + tuple boxing).
        - Bind `food_grid.get`, `can_sea`, `can_land`, `N` etc as
          locals so the hot inner loops don't re-resolve attributes."""
        x, y = nx.pos
        hunger = 1.0 - nx.food / 60.0
        if hunger < -1.0: hunger = -1.0
        elif hunger > 1.0: hunger = 1.0
        cs = self._cell_size
        cx = x // cs
        cy = y // cs
        # locals (re-resolved-once vs per-iteration attribute lookups)
        world = self.world
        t = world._t
        N = world._N
        fg_get = self._food_grid.get
        ng_get = self._nxer_grid.get
        can_sea = nx.can_sea
        can_land = nx.can_land
        my_id = nx.id
        # nearest food: scan only the 3x3 neighbouring buckets, and
        # skip food on terrain this NxEr cannot enter.
        best_d, fdx, fdy = 1e9, 0.0, 0.0
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                bucket = fg_get((gx, gy))
                if not bucket:
                    continue
                for fx, fy in bucket:
                    # inline: is sea? = `t[fy*N+fx] == 1`
                    f_sea = (0 <= fx < N and 0 <= fy < N
                             and t[fy * N + fx] == 1)
                    if (f_sea and not can_sea) or \
                            (not f_sea and not can_land):
                        continue
                    dx = fx - x; dy = fy - y
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best_d = d
                        fdx = (1 if dx > 0 else -1 if dx < 0 else 0)
                        fdy = (1 if dy > 0 else -1 if dy < 0 else 0)
        sight = 1.0 if best_d < 25 else 0.0
        smell = 1.0 / (1.0 + math.sqrt(best_d)) if best_d < 1e8 else 0.0
        # terrain channel — inline the lookup
        terrain = 1.0 if (0 <= x < N and 0 <= y < N
                          and t[y * N + x] == 1) else 0.0
        # nearest neighbour: only the local + adjacent buckets
        nb = 0.0
        all_nxers = self.nxers
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                bucket = ng_get((gx, gy))
                if not bucket:
                    continue
                for oid in bucket:
                    if oid == my_id:
                        continue
                    o = all_nxers.get(oid)
                    if o is not None:
                        op = o.pos
                        if abs(op[0] - x) <= 2 and abs(op[1] - y) <= 2:
                            nb = 1.0
                            break
                if nb:
                    break
            if nb:
                break
        daynight = math.sin(self.tick * 0.01)
        proprio = float(nx.heading) / 7.0
        song = nx.last_sing_level
        return [hunger, fdx, fdy, terrain, sight, smell,
                daynight, proprio, nb, song]

    def _nearest_mate_dir(self, nx):
        """v1.36 — unit step toward the nearest ELIGIBLE mate within the
        3×3 spatial-grid neighbourhood (opposite sex, off cooldown,
        food>=5, not a parent/child), or None. Only called for well-fed
        off-cooldown NxErs, so the neighbourhood scan is rare + cheap.
        This is what lets a sated NxEr actively court instead of camping
        food — 'seek a mate if the time allows'."""
        cs = self._cell_size
        cx, cy = nx.pos[0] // cs, nx.pos[1] // cs
        best = None
        bestd = 1 << 30
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for oid in self._nxer_grid.get((gx, gy), ()):
                    if oid == nx.id:
                        continue
                    o = self.nxers.get(oid)
                    if (o is not None and o.alive
                            and o.is_male != nx.is_male
                            and self.tick >= o.mate_cooldown_until
                            and o.food >= self._mate_gate(o)
                            and o.id not in nx.parents
                            and nx.id not in o.parents):
                        ddx = o.pos[0] - nx.pos[0]
                        ddy = o.pos[1] - nx.pos[1]
                        d = ddx * ddx + ddy * ddy
                        if d < bestd:
                            bestd = d
                            best = o
        if best is None:
            return None
        ddx = best.pos[0] - nx.pos[0]
        ddy = best.pos[1] - nx.pos[1]
        return ((0 if ddx == 0 else (1 if ddx > 0 else -1)),
                (0 if ddy == 0 else (1 if ddy > 0 else -1)))

    def _explore_dir(self, nx):
        """v1.36 — momentum + novelty exploration. Prefer continuing in
        the current heading (so excursions cover ground instead of
        oscillating), but steer toward a not-recently-visited enterable
        cell; fall back to any enterable cell. Returns (dx, dy) or None.
        This is the 'exploration is rewarded' drive for well-fed NxErs
        and the search behaviour for hungry NxErs with no food in sight."""
        px, py = nx.pos[0], nx.pos[1]
        ld = nx._last_move_dir
        # 65% of the time, keep heading if it leads somewhere novel
        if (ld != (0, 0) and random.random() < 0.65):
            tx, ty = px + ld[0], py + ld[1]
            if self._can_enter(nx, tx, ty) and (tx, ty) not in nx.visited:
                return ld
        # otherwise prefer a novel (unvisited) enterable neighbour
        dirs = list(DIR_OFFSETS)
        random.shuffle(dirs)
        for d in dirs:
            tx, ty = px + d[0], py + d[1]
            if self._can_enter(nx, tx, ty) and (tx, ty) not in nx.visited:
                return d
        # all neighbours visited/blocked — keep heading, else any open cell
        if ld != (0, 0) and self._can_enter(nx, px + ld[0], py + ld[1]):
            return ld
        for d in dirs:
            if self._can_enter(nx, px + d[0], py + d[1]):
                return d
        return None

    def _act(self, nx, outs):
        """7 motor channels → world actions. Trinary {-1,0,1}."""
        if len(outs) < 7:
            outs = list(outs) + [0] * (7 - len(outs))
        mvx, mvy, social, mate, give, rest, sing = outs[:7]
        # v1.44 — M7 self-sustained activity (Nengo-distinguishing signal):
        # motor amplitude when the NxEr has NO external drive vs when it
        # is externally driven. External-drive channels in _last_sensory
        # are fdx,fdy,sight,smell,nb,song (idx 1,2,4,5,8,9); hunger and
        # day/night are internal and excluded so a quiet-patch NxEr counts
        # as zero-input. Free — reuses the sensory + motor vectors already
        # in hand. (Logged at history cadence; see history_sample.)
        if self._m12_on:
            s = getattr(nx, "_last_sensory", None)
            if s is not None and len(s) >= 10:
                drive = 0
                for i in (1, 2, 4, 5, 8, 9):
                    if s[i] != 0:
                        drive += 1
                mamp = (abs(mvx) + abs(mvy) + abs(social) + abs(mate)
                        + abs(give) + abs(rest) + abs(sing)) / 7.0
                if drive == 0:
                    self._m7_zero_sum += mamp
                    self._m7_zero_n += 1
                elif drive >= 3:
                    self._m7_drv_sum += mamp
                    self._m7_drv_n += 1
        # Singing indicator. The brain's `sing` motor output rarely
        # fires with the current NAS arch, so singing is driven by
        # EVENTS instead (eating, mating, rare spontaneous) — matching
        # the v5.0 Lite "sing on food" behaviour. last_sing_level is a
        # 0..1 level that EVENTS set to 1.0 and which decays ~0.9/tick
        # (so a song lasts ~1s); the snapshot reports s=1 while it's
        # above a small threshold. A positive brain sing output still
        # boosts it if it ever fires.
        nx.last_sing_level = max(float(sing) if sing > 0 else 0.0,
                                 nx.last_sing_level * 0.85)
        if nx.last_sing_level < 0.05:
            nx.last_sing_level = 0.0
        # rare spontaneous song (Lite has a ~0.2% spontaneous path)
        if nx.last_sing_level == 0.0 and random.random() < 0.0015:
            nx.last_sing_level = 1.0
        # ---- v1.36 BEHAVIOURAL PRIORITY -----------------------------
        # Replaces the old idle-explore safety net + hunger-only floor.
        # The brain's motor output is still the default, but a drive
        # overrides it: hungry NxErs forage (or search if no food is in
        # sight); well-fed NxErs court an eligible mate if one is near,
        # otherwise explore. Because a well-fed NxEr now ALWAYS has an
        # explore/court drive, the "camp the food cluster" and
        # "well-fed-but-stuck starves in place" behaviours both go away,
        # and exploration + mate-seeking are actively rewarded.
        #   hunger_threshold_pct (0.55): food < threshold ⇒ hungry.
        #   food_wander_ticks (40): after a bite, foraging-toward-food
        #     is suppressed this long so the NxEr leaves the source and
        #     explores instead of orbiting it.
        li = nx._last_sensory
        forced = False
        hungry = nx.food < (self.bio_start_food * self.bio_hunger_threshold)
        food_sensed = li is not None and (li[1] or li[2])
        if hungry:
            if food_sensed and self.tick >= nx.wander_until_tick:
                mvx, mvy = int(li[1]), int(li[2])      # forage to food
                rest = 0
                forced = True
            else:
                ed = self._explore_dir(nx)             # search for food
                if ed is not None:
                    mvx, mvy = ed
                    rest = 0
                    forced = True
        else:
            md = None
            if self.tick >= nx.mate_cooldown_until and nx.food >= self._mate_gate(nx):
                md = self._nearest_mate_dir(nx)
            if md is not None:
                mvx, mvy = md                          # court: move to mate
                nx.mating_intent_until_tick = self.tick + 60
                rest = 0
                forced = True
            else:
                ed = self._explore_dir(nx)             # explore
                if ed is not None:
                    mvx, mvy = ed
                    rest = 0
                    forced = True
        if rest == 1 and not forced:
            nx.food -= 0.005          # resting costs less
            return
        dx = int(max(-1, min(1, mvx)))
        dy = int(max(-1, min(1, mvy)))
        if dx or dy:
            nxp = nx.pos[0] + dx
            nyp = nx.pos[1] + dy
            target = (nxp, nyp)
            occupant = self._occupied.get(target)
            # One NxEr per cell: a step is blocked when another
            # LIVING NxEr already stands on the target. Adjacency is
            # still fine — mating, stealing, and social signals all
            # work from neighbouring cells (the _neighbours()
            # generator includes the 8 surrounding cells).
            blocked = (occupant is not None
                       and occupant != nx.id
                       and occupant in self.nxers
                       and self.nxers[occupant].alive)
            if (not blocked) and self._can_enter(nx, nxp, nyp):
                # vacate the old cell and claim the new one
                old = (nx.pos[0], nx.pos[1])
                if self._occupied.get(old) == nx.id:
                    del self._occupied[old]
                nx.pos = [nxp, nyp]
                self._occupied[target] = nx.id
                nx.heading = _dir_index(dx, dy)
                nx._last_move_dir = (dx, dy)
                nx.last_move_tick = self.tick
                if target not in nx.visited:
                    # v1.36 — explored is a MONOTONIC counter; `visited`
                    # is an LRU-bounded (~5000) recent-cell set used only
                    # as the novelty filter. Entering genuinely-new
                    # ground increments explored without bound, so the
                    # metric keeps rewarding roaming (was frozen at
                    # 5000), while camping — where the NxEr stays within
                    # its recent set — no longer increments it.
                    nx.stats.explored += 1
                    nx.visited.add(target)
                    q = nx._visit_q
                    q.append(target)
                    if len(q) > 5000:
                        nx.visited.discard(q.popleft())
        # eat any food on the new cell (O(1) via position index). Two
        # rate-limits matching the user's expectation:
        #   * the food source has `remaining = 25` and only depletes
        #     when an NxEr actually takes a bite (not just by being
        #     present), so crowd-camping does not deplete faster than
        #     a single eater
        #   * each NxEr has its own eat cooldown
        #     (cfg.eat_cooldown_ticks, default 20), so staying on the
        #     same cell does NOT give food every tick — you get +1
        #     every cooldown ticks. With default 20 ticks at 20 tps a
        #     food source lasts ~25 seconds for one NxEr camping on
        #     it (25 bites × 1s between bites). Set lower for faster
        #     worlds, higher for very slow harvesting.
        fid = self._food_at.get((nx.pos[0], nx.pos[1]))
        if fid is not None and fid in self.foods:
            f = self.foods[fid]
            rem = f.get("remaining", 0)
            cd = int(self.cfg.get("eat_cooldown_ticks", 20))
            if (rem > 0
                    and (self.tick - nx.last_eat_tick) >= cd):
                f["remaining"] = rem - 1
                # v1.58 — compliant brains extract more from the same food
                # item. This is the income-side counterpart to drain relief:
                # M-compliance correlated with WORSE foraging (top decile
                # starved 57% vs 25%), so an efficiency bonus attacks the
                # actual deficit instead of only slowing the loss.
                nx.food += (1.0 + self._m_sel_forage * self.m_advantage(nx)
                            if self._m_sel else 1.0)
                nx.stats.food_found += 1
                self.lifetime["total_food_eaten"] += 1
                if self._learning:        # v1.52 — dopamine reward on food
                    nx._da_accum += self._reward_eat_da
                # sing on DISCOVERY of a new food source (like Lite's
                # "first-time food" trigger), not on every bite — keeps
                # singing discrete rather than a constant choir.
                if fid not in nx.known_food_ids:
                    nx.last_sing_level = 1.0
                nx.known_food_ids.add(fid)
                nx.last_eat_tick = self.tick
                # post-meal exploration: floor disabled for
                # food_wander_ticks ticks so the NxEr leaves the food
                # cell and wanders before coming back for another
                # bite. Without this they camp the same source
                # continuously.
                nx.wander_until_tick = (self.tick
                                        + self.bio_food_wander_ticks)
                if len(self._events) < 200:
                    self._events.append({"k": "eat",
                                         "x": nx.pos[0],
                                         "y": nx.pos[1]})
                if f["remaining"] <= 0:
                    del self.foods[fid]
                    self._food_at.pop((nx.pos[0], nx.pos[1]), None)
                    self._food_dirty = True

        def _neighbours():
            cs = self._cell_size
            cx, cy = nx.pos[0] // cs, nx.pos[1] // cs
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for oid in self._nxer_grid.get((gx, gy), ()):
                        if oid == nx.id:
                            continue
                        o = self.nxers.get(oid)
                        if (o and o.alive
                                and abs(o.pos[0] - nx.pos[0]) <= 1
                                and abs(o.pos[1] - nx.pos[1]) <= 1):
                            yield o

        # social: steal from / give to an adjacent NxEr. Two paths:
        #   (1) brain-driven: social==1 → 1-food swap
        #   (2) STOCHASTIC HUNGER FLOOR (matches offline v189): when
        #       hungry (food<5) and an adjacent NxEr has surplus
        #       (food>5), there's a 30% chance per tick of an
        #       opportunistic steal — even with no brain output. This
        #       is what kept "Stolen" at 0: brand-new brains never
        #       output social==1 reliably, so without this floor the
        #       behaviour never appears in the leaderboard.
        # v1.46 — theft is now rate-limited like eating (steal_cooldown_
        # ticks). Both paths below respect it so one NxEr can't drain a
        # neighbour every single tick.
        can_steal = (self.tick - nx.last_steal_tick) >= self._steal_cd
        if social == 1 or give == 1:
            for o in _neighbours():
                if give == 1 and nx.food > 10:
                    nx.food -= 1
                    o.food += 1
                elif social == 1 and can_steal and o.food > 1:
                    o.food -= 1
                    nx.food += 1
                    nx.stats.food_taken += 1
                    nx.last_steal_tick = self.tick
                break
        elif nx.food < 5 and can_steal:
            for o in _neighbours():
                if o.food > 5 and random.random() < 0.30:
                    o.food -= 1
                    nx.food += 1
                    nx.stats.food_taken += 1
                    nx.last_steal_tick = self.tick
                    break
        # mating — overhauled to match the v189 offline reference so
        # mating actually happens. Now:
        #   * brain mate==1 OR a 0.3% stochastic floor (was 3% in
        #     offline and 1% in v1.022; both still mated too often
        #     because untrained brains × wide window × every
        #     opposite-sex encounter saturated the cap). Opens an
        #     intent WINDOW of 60 ticks (was 180; offline's `6 × 30`
        #     was sized for slower 30 GTS systems where mating is
        #     rare; on our 20 tps loop 60 ticks = 3 s is plenty).
        #   * mating only triggers when BOTH parties are within their
        #     intent windows in the same tick, opposite sex, food>=5,
        #     not on cooldown, not parent/child. Each pays 1 food.
        # Stochastic mating-intent floor. Original code computed
        # `opposite_neighbour = any(o.is_male != nx.is_male for o in
        # _neighbours())` UNCONDITIONALLY for every NxEr, then almost
        # always discarded the result (the 0.003 dice make 99.7% of the
        # scans wasted work). At 2500+ alive that single line was a
        # measurable fraction of main-thread time. Now we only scan
        # neighbours on the rare tick when the dice actually triggers.
        if mate == 1:
            nx.mating_intent_until_tick = self.tick + 60
        elif random.random() < 0.003:
            for o in _neighbours():
                if o.is_male != nx.is_male:
                    nx.mating_intent_until_tick = self.tick + 60
                    break
        if (nx.mating_intent_until_tick > self.tick
                and self.tick >= nx.mate_cooldown_until
                and nx.food >= self._mate_gate(nx)):
            for o in _neighbours():
                if (o.is_male != nx.is_male
                        and self.tick >= o.mate_cooldown_until
                        and o.food >= self._mate_gate(o)
                        and o.mating_intent_until_tick > self.tick
                        and o.id not in nx.parents
                        and nx.id not in o.parents):
                    self._mate(nx, o)
                    break

    def _mate(self, a, b):
        a.stats.mates_performed += 1
        a.last_sing_level = 1.0          # courtship/celebration song
        b.last_sing_level = 1.0
        b.stats.mates_performed += 1
        if self._learning:               # v1.52 — dopamine reward on mating
            a._da_accum += self._reward_mate_da
            b._da_accum += self._reward_mate_da
        a.food -= 1                       # offline cost
        b.food -= 1
        cd = int(self.cfg.get("mate_cooldown_ticks", 90))
        a.mate_cooldown_until = self.tick + cd
        b.mate_cooldown_until = self.tick + cd
        a.mating_intent_until_tick = 0
        b.mating_intent_until_tick = 0
        # Offspring terrain capability (matches offline v189 rules).
        # Parents are at a.pos and b.pos; mating at the shore (one
        # parent on land, the other on sea, AND each is a specialist
        # for that terrain) creates the only amphibious offspring.
        A_land = a.can_land and not a.can_sea
        A_sea  = a.can_sea  and not a.can_land
        B_land = b.can_land and not b.can_sea
        B_sea  = b.can_sea  and not b.can_land
        a_on_sea = self.world.is_sea(a.pos[0], a.pos[1])
        b_on_sea = self.world.is_sea(b.pos[0], b.pos[1])
        shore_mating = ((A_land and B_sea and not a_on_sea and b_on_sea)
                        or (A_sea and B_land and a_on_sea and not b_on_sea))
        if shore_mating:
            c_land, c_sea = True, True
        elif A_land and B_land:
            c_land, c_sea = True, False
        elif A_sea and B_sea:
            c_land, c_sea = False, True
        else:
            c_land = a.can_land or b.can_land
            c_sea  = a.can_sea  or b.can_sea
        if not (c_land or c_sea):
            c_land = True
        # child inherits a parent's params. By default this is a verbatim
        # copy (the substrate's own plasticity drives within-life
        # adaptation). v1.34: an OPTIONAL genetic operator — set
        # "offspring_mutation_strength" > 0 in world_config.json to let
        # heritable neural traits (INCLUDING firing_threshold_inhibitory,
        # the v195 E/I lever, which was previously frozen at the founder
        # value) mutate per birth. With the new history logger this lets
        # you watch whether free evolution rediscovers or abandons the
        # NAS-found trinary corner. Default 0.0 = unchanged behaviour.
        base_parent = a.params if random.random() < 0.5 else b.params
        child_pd = _params_to_dict(base_parent)
        mut = float(self.cfg.get("offspring_mutation_strength", 0.0))
        if mut > 0.0:
            child_pd = _mutate_params(child_pd, mut)
            child_params = make_params(child_pd)
            # carry the mutated heritable traits past the USER_TUNABLE
            # filter so firing_threshold_inhibitory et al. actually
            # inherit + evolve (otherwise make_params resets them).
            _apply_heritable(child_params, child_pd)
        else:
            child_params = make_params(child_pd)
        child = self._spawn_nxer(
            params=child_params,
            parents=(a.id, b.id),
            terrain_caps=(c_land, c_sea))
        if child:
            a.offspring_ids.append(child.id)
            b.offspring_ids.append(child.id)
            self.lifetime["total_matings"] += 1
            self.lifetime["total_births_mating"] += 1
            # v1.44 — M10 heritability: record the parents' mean fitness at
            # the moment of birth, keyed by child id. Paired with the
            # child's own fitness at report time (history_sample). Capped
            # so it can't grow without bound on a 24/7 server.
            if self._m12_on:
                try:
                    pavg = 0.5 * (float(a.stats.fitness)
                                  + float(b.stats.fitness))
                    self._herit_pending[child.id] = pavg
                    if len(self._herit_pending) > 6000:
                        for k in list(self._herit_pending)[:1000]:
                            del self._herit_pending[k]
                except Exception:
                    pass
            if len(self._events) < 200:
                self._events.append({"k": "mate",
                                     "x": a.pos[0], "y": a.pos[1]})
                self._events.append({"k": "birth", "x": child.pos[0],
                                     "y": child.pos[1]})
            if self.history is not None:
                p = child.params
                self.history.lineage({
                    "tick": self.tick, "id": child.id, "name": child.name,
                    "parents": [a.id, b.id], "born_tick": self.tick,
                    "is_male": child.is_male, "managed": child.is_managed,
                    # snapshot the heritable E/I levers at birth so the
                    # lineage file alone can reconstruct trait evolution
                    "thr_inh": round(getattr(p, "firing_threshold_inhibitory", 0.0), 4),
                    "thr_exc": round(getattr(p, "firing_threshold_excitatory", 0.0), 4),
                    "refr": getattr(p, "refractory_period_ticks", 0),
                })

    # ---- per-tick ---------------------------------------------------
    def _effective_food_cap(self):
        """max_food is the explicit cap. But on a big world the shipped
        default of 160 is far below survival density (a 800² world has
        640k cells; 160 food = 0.025% density, NxErs starve before
        finding any). When max_food is left at default-ish values and
        the world is large, scale up to keep ~0.3% food density (one
        food per ~330 passable cells), matching the calibrated 50²
        balance. Set `max_food` explicitly to override."""
        explicit = int(self.cfg["max_food"])
        N = int(self.cfg["world_size"])
        # ~0.3% density baseline (160/(50*50) = 6.4% on a 50² world)
        scaled = max(160, (N * N) // 330)
        # if the user explicitly set a high max_food (>= scaled),
        # respect it; otherwise use the scaled target so big worlds
        # don't starve
        return max(explicit, scaled) if N > 100 else explicit

    def step(self):
        self.tick += 1
        self.lifetime["total_ticks"] += 1
        # SINGLE PASS to build the alive list and update peak
        # watermarks — replaces three separate iterations over
        # self.nxers.values() (alive count, managed-alive count,
        # alive-list comprehension). At 2500 alive that's ~5000
        # fewer Python-level iterations per tick.
        order = []
        n_alive = 0
        n_managed = 0
        for a in self.nxers.values():
            if a.alive:
                order.append(a)
                n_alive += 1
                if a.is_managed:
                    n_managed += 1
        if n_alive > self.lifetime["peak_alive"]:
            self.lifetime["peak_alive"] = n_alive
        if n_managed > self.lifetime["peak_managed"]:
            self.lifetime["peak_managed"] = n_managed
        self._events = []
        _t0 = time.perf_counter()          # v1.39 perf: tick + Phase A start

        # --- Phase A: sense (read-only) — cheap pure-Python ----------
        self._build_spatial()
        # Neural LOD: with brain_step_every = K, each NxEr's brain is
        # stepped once every K ticks (round-robin staggered by id so the
        # load is even), reusing its last motor output in between. K=1 =
        # full fidelity; K=2 ≈ 2x fewer brain computations for a modest
        # behavioural change. The single biggest tunable speed lever
        # besides core count, and it never blocks the web server.
        # brain_step_every (K): cuts how often each NxEr's brain is
        # actually stepped. K=2 → each brain runs every other tick (the
        # last motor output is reused), halving both _sense calls AND
        # the brain-pool pipe traffic. This used to be a flat config;
        # now it AUTO-SCALES with the alive population (the configured
        # value is the FLOOR, auto-scaling only raises it). At 2500+
        # alive the user reported main-core saturation and TPS in the
        # ~1Hz range; bumping K from 1 → 2 brought TPS back to ~5Hz on
        # the same world without visible behavioural change.
        K_user = max(1, int(self.cfg.get("brain_step_every", 1)))
        K_thresh = int(self.cfg.get("brain_step_auto_above_alive", 1500))
        if K_thresh > 0 and n_alive > K_thresh:
            auto_K = 2
            if n_alive > K_thresh * 2.5:
                auto_K = 3
            if n_alive > K_thresh * 4:
                auto_K = 4
            K = max(K_user, auto_K)
        else:
            K = K_user
        # v1.40 — stagger the LOD by each brain's index WITHIN its worker
        # shard (id // n_shards), not by raw id. Brains live in worker
        # `id % n_shards`, so staggering by raw id made the per-tick "due"
        # set (ids in one residue class mod K) collide with the sharding
        # whenever gcd(K, n_shards) > 1: e.g. K=5 with 10 workers put
        # every due brain into just 2 of 10 workers, so 8 workers idled
        # while 2 ran the whole batch serially — Phase B ballooned and
        # raising K made it WORSE. Keying the stagger to id//n_shards
        # steps an equal slice of EACH worker's own brains every tick, so
        # all workers stay busy for any K. (Each brain still steps once
        # per K ticks — identical behaviour, just load-balanced.)
        n_shards = max(1, getattr(self.pool, "n", 1))
        batch = []
        due = []
        mods = {}                      # v1.52 — id -> dopamine level (rewarded only)
        learning = self._learning
        da_base, da_max = self._da_base, self._da_max
        for nx in order:
            nx.stats.time_lived_s += self.dt
            if K == 1 or (self.tick + (nx.id // n_shards)) % K == 0:
                sens = self._sense(nx)
                nx._last_sensory = sens  # used by _act's hunger floor
                batch.append((nx.id, sens))
                due.append(nx)
                # v1.52 — if this NxEr banked reward since its last brain
                # step, raise phasic dopamine for THIS step so AGMP
                # consolidates the eligibility traces, then clear the bank.
                if learning and nx._da_accum > 0.0:
                    mods[nx.id] = da_base + (nx._da_accum if nx._da_accum
                                             < da_max else da_max)
                    nx._da_accum = 0.0

        # --- Phase B: brains in PARALLEL across worker processes -----
        # While this blocks on pipe.recv() the GIL is free, so the
        # aiohttp web server stays responsive and clients keep loading.
        _tA = time.perf_counter()          # perf: Phase A done
        results = self.pool.step(batch, mods if mods else None)
        _tB = time.perf_counter()          # perf: Phase B done
        for nx in due:
            o = results.get(nx.id)
            if o is not None:
                nx._last_out = o          # cache for the skipped ticks

        # --- Phase C: apply actions + metabolism (serial, cheap) ----
        # Hoist hot attributes/methods as locals so the per-NxEr loop
        # doesn't re-resolve `self.bio_*` / `self.dt` / `self._kill` /
        # `self.tick` thousands of times.
        bio_max = self.bio_max_atrophy
        bio_ramp = self.bio_metab_ramp
        bio_drain = self.bio_base_drain
        # v1.57 — hoisted so the per-NxEr loop stays cheap
        m_sel = self._m_sel
        m_drain = self._m_sel_drain
        m_idle = self._m_sel_idle
        m_hunger = self._m_sel_hunger
        energy_cap = self.bio_energy_cap          # v1.46
        idle_death = self.bio_idle_death_ticks    # v1.46 (0 = off)
        dt = self.dt
        tick = self.tick
        n_ckpt = len(_AGE_CKPTS)
        for nx in order:
            outs, br = nx._last_out         # initialized in NxEr.__init__
            self._act(nx, outs)
            # v1.53 — within-life foraging curve: one int compare per NxEr
            # per tick, and a dict write only when an age boundary is
            # crossed (at most 8 times in a whole lifetime).
            if nx._ck_i < n_ckpt:
                _ck = _AGE_CKPTS[nx._ck_i]
                if (tick - nx.born_tick) >= _ck:
                    nx.food_by_age[_ck] = nx.stats.food_found
                    nx._ck_i += 1
            if nx.food > energy_cap:        # v1.46 — cap hoarding
                nx.food = energy_cap
            idle_ticks = tick - nx.last_move_tick
            # v1.57 — M-claim selection advantage. adv is 0 until this NxEr
            # has its first science sample, so newborns get no free pass.
            adv = self.m_advantage(nx) if m_sel else 0.0
            if idle_death and idle_ticks > (idle_death * (1.0 + m_idle * adv)):
                self._kill(nx, "idle")      # v1.46 — cull the stuck
                continue
            idle = idle_ticks * dt
            if idle < 0.0:
                idle = 0.0
            atrophy = 1.0 + bio_ramp * idle
            if atrophy > bio_max:
                atrophy = bio_max
            # v1.58 — weight the relief toward hungry NxErs: an agent at
            # 10% energy gets the full discount, one at the cap gets a
            # fraction of it. Same average cost, aimed where it saves lives.
            if adv > 0.0:
                _need = 1.0 - (nx.food / energy_cap if energy_cap else 0.0)
                if _need < 0.0:
                    _need = 0.0
                _rel = m_drain * adv * (1.0 + m_hunger * _need) / (1.0 + m_hunger)
                if _rel > 0.95:
                    _rel = 0.95
                nx.food -= bio_drain * atrophy * (1.0 - _rel)
            else:
                nx.food -= bio_drain * atrophy
            nx.net.branching_ratio = br
            nx.stats.branching = br
            if nx.food <= 0:
                self._kill(nx)
        # v1.39 perf: fold this tick's phase timings into the EWMA (ms).
        _tC = time.perf_counter()
        _p = self._perf
        _al = _p["alpha"]
        _p["a_ms"] += _al * ((_tA - _t0) * 1000.0 - _p["a_ms"])
        _p["b_ms"] += _al * ((_tB - _tA) * 1000.0 - _p["b_ms"])
        _p["c_ms"] += _al * ((_tC - _tB) * 1000.0 - _p["c_ms"])
        _p["tick_ms"] += _al * ((_tC - _t0) * 1000.0 - _p["tick_ms"])
        _p["due"] = len(due)
        # Heavy periodic ops auto-throttle at high N. `_update_g` does
        # PCA-class numerical work over the full population (heavy on
        # the main core at scale); `_update_all_time` iterates every
        # alive NxEr × every metric and sorts. Stretch their cadence
        # linearly with alive count so the main loop isn't blocked
        # for hundreds of ms every K ticks at 2500+ alive.
        rank_int = max(1, int(self.cfg.get("rank_interval_ticks", 10)))
        g_int = max(1, int(self.cfg.get("g_interval_ticks", 30)))
        heavy_thresh = int(self.cfg.get("heavy_ops_throttle_above_alive", 1500))
        if heavy_thresh > 0 and n_alive > heavy_thresh:
            mult = 1 + n_alive // heavy_thresh   # 2× at 1.5k, 3× at 3k…
            rank_int *= mult
            g_int *= mult
        if self.tick % rank_int == 0:
            for nx in order:
                if nx.alive:
                    nx.stats.energy_efficiency = max(
                        0.0, nx.food / max(1.0, nx.stats.time_lived_s))
                    nx.stats.fitness = _fitness(nx.stats)
        # food respawn — keep the world well-fed (a starving map kills
        # everyone regardless of brain quality). Refill briskly toward
        # the effective cap (auto-scales with world area on >100²
        # worlds so the calibrated 50² balance still holds at 800²).
        cap = self._effective_food_cap()
        if len(self.foods) < cap:
            deficit = cap - len(self.foods)
            self._spawn_food(min(
                cap, len(self.foods) + max(2, deficit // 3)))
        # population g (throttled — uses the population-aware g_int
        # computed above so heavy PCA work stretches with alive count)
        if self.tick % g_int == 0:
            self._update_g()
        # anti-extinction respawn
        alive = [a for a in self.nxers.values() if a.alive]
        if len(alive) <= int(self.cfg.get("min_alive", 1)):
            for _ in range(int(self.cfg.get("respawn_batch", 8))):
                if self._nas_enabled:
                    self._spawn_explorer()
                else:
                    self._spawn_nxer()
        # v1.51 — steady NAS explorer trickle: one fresh-architecture NxEr
        # every nas_explore_interval_ticks, so the search keeps sampling
        # without the explorers ever outnumbering the evolved population.
        elif (self._nas_enabled and self._nas_interval > 0
                and self.tick % self._nas_interval == 0
                and len(alive) < int(self.cfg["max_nxers"])):
            self._spawn_explorer()
        # all-time ranking is exact-enough at coarse cadence; rebuilding
        # + sorting it every tick is wasted serial work on the main core
        # (uses the population-aware rank_int from above)
        if self.tick % rank_int == 0:
            self._update_all_time()
        self._prune_dead()

    def _kill(self, nx, cause="starved"):
        nx.alive = False
        self.lifetime["total_deaths"] += 1
        # free the cell so other NxErs (and respawns) can use it
        cell = (nx.pos[0], nx.pos[1])
        if self._occupied.get(cell) == nx.id:
            del self._occupied[cell]
        if len(self._events) < 200:
            self._events.append({"k": "die",
                                 "x": nx.pos[0], "y": nx.pos[1]})
        nx.password_hash = None       # password only valid while alive
        nx.death_tick = self.tick
        # v1.34 — compact obituary (off the hot path: enqueue only).
        # Sampled by obituary_sample_rate so a mass die-off can't flood
        # the queue. The all-time board keeps the TOP NxErs; this keeps
        # the DISTRIBUTION (lifespans, causes, reproductive success).
        if (self.history is not None
                and random.random() < self._obit_rate):
            st = nx.stats
            p = nx.params
            born = getattr(nx, "born_tick", 0)
            self.history.obituary({
                "tick": self.tick, "id": nx.id, "name": nx.name,
                "born_tick": born, "death_tick": self.tick,
                "lifespan_ticks": self.tick - born,
                "cause": cause,        # v1.47 — "starved" | "idle"
                "managed": nx.is_managed, "is_male": nx.is_male,
                "n_offspring": len(nx.offspring_ids),
                "food_found": round(st.food_found, 1),
                "food_taken": round(st.food_taken, 1),
                "mates": st.mates_performed,
                "explored": st.explored,
                "fitness": round(st.fitness, 3),
                # v1.53 — unsaturated companion metric + within-life
                # foraging curve (see _fitness_hi / _AGE_CKPTS).
                "fitness_hi": round(_fitness_hi(st), 4),
                "food_by_age": nx.food_by_age,
                # v1.54 — per-NxEr M-claim compliance
                "m_score": nx.m_score_ema,
                "m_fit": nx.m_fit_ema,          # v1.55 continuous
                "m_fit_w": nx.m_fit_w,          # v1.58 hard-band weighted
                "m_sel_adv": (round(self.m_advantage(nx), 4)   # v1.58 fix
                              if (self._m_sel and nx.m_fit_ema is not None)
                              else 0.0),
                "m_in_band": nx.m_in_band,
                "m_n_checked": nx.m_n_checked,
                "m_deviation": nx.m_deviation,
                "m_samples": nx.m_samples,
                "g": round(getattr(st, "g_factor", 0.0), 3),
                "thr_inh": round(getattr(p, "firing_threshold_inhibitory", 0.0), 4),
                "thr_exc": round(getattr(p, "firing_threshold_excitatory", 0.0), 4),
            })
        # v1.51 — NAS dataset: every explorer logs its architecture and the
        # outcomes it achieved, ONCE, here at death. Explorers are a small
        # minority so this is not sampled (one compact line each), giving a
        # clean architecture -> performance table without flooding the log.
        nt = getattr(nx, "nas_trial", None)
        # v1.58 — close the search loop: every finished explorer trains the
        # sampler, whether or not history logging is enabled. m_fit is the
        # target (continuous, tie-free, heritable r=+0.58); fall back to
        # m_score then fitness_hi if the brain was never scored.
        if nt is not None:
            _sc = nx.m_fit_ema
            if _sc is None:
                _sc = nx.m_score_ema
            if _sc is None:
                _sc = _fitness_hi(nx.stats)
            self._nas_record(nt.get("arch"), _sc)
        if nt is not None and self.history is not None:
            st = nx.stats
            born = getattr(nx, "born_tick", 0)
            self.history.write("nas_trials", {
                "trial": nt["id"], "arch": nt["arch"],
                "born_tick": born, "death_tick": self.tick,
                "lifespan": self.tick - born, "cause": cause,
                "n_offspring": len(nx.offspring_ids),
                "food_found": round(st.food_found, 1),
                "food_taken": round(st.food_taken, 1),
                "mates": st.mates_performed,
                "explored": st.explored,
                "fitness": round(st.fitness, 3),
                "fitness_hi": round(_fitness_hi(st), 4),   # v1.53
                "food_by_age": nx.food_by_age,             # v1.53
                # v1.54 — does this architecture BUILD an M-compliant brain?
                "m_score": nx.m_score_ema,
                "m_fit": nx.m_fit_ema,          # v1.55 continuous
                "m_fit_w": nx.m_fit_w,          # v1.58 hard-band weighted
                "m_sel_adv": round(self.m_advantage(nx), 4),   # v1.57
                "m_score_last": nx.m_score,
                "m_in_band": nx.m_in_band,
                "m_n_checked": nx.m_n_checked,
                "m_deviation": nx.m_deviation,
                "m_samples": nx.m_samples,
                "m_last": nx.m_last,
                "g": round(getattr(st, "g_factor", 0.0), 3),
            })
        self.pool.remove(nx.id)       # free the brain in its worker

    def export_model_for(self, nx):
        """Build the downloadable model dict (brain comes from pool)."""
        return nx.export_model(self.pool.export(nx.id))

    def shutdown(self):
        try:
            self.pool.close()
        except Exception:
            pass

    def _prune_dead(self):
        # keep dead NxErs only long enough for the all-time scan; their
        # scores are already captured in self.all_time.
        dead = [a for a in self.nxers.values() if not a.alive]
        if len(dead) > 200:
            dead.sort(key=lambda a: getattr(a, "death_tick", 0))
            for a in dead[:len(dead) - 200]:
                self.nxers.pop(a.id, None)

    def _update_g(self):
        alive = [a for a in self.nxers.values() if a.alive]
        try:
            res = compute_population_g(alive, write_back=True)
            for a in alive:
                a.stats.g_factor = getattr(
                    a, "_g_score", getattr(a.stats, "g_factor", 0.0))
            self._g_cache = {
                "pc1": float(res.get("g_pc1_fraction", 0.0)),
                "pos_manifold": float(res.get("g_positive_manifold", 0.0)),
                "mean_r": float(res.get("g_mean_offdiag_r", 0.0)),
                "lambda_ratio": float(
                    res.get("g_lambda1_over_lambda2", 1.0)),
                "n": len(alive),
            }
        except Exception as _e:
            if not getattr(self, "_g_err_logged", False):
                self._g_err_logged = True
                print("[engine] population-g failed (g stays 0):",
                      repr(_e))

    # ---- v1.34: science history sampling (called at low cadence by the
    # server loop, NOT every tick; reads live objects fast, enqueues one
    # record, never touches disk). pool.sample_firing() is one cheap
    # round-trip per worker at this cadence. ------------------------------
    def history_provenance(self, server_version, cfg):
        if self.history is None:
            return
        rec = {"event": "boot", "tick": self.tick,
               "server_version": server_version,
               "started_at_unix": self.lifetime.get("started_at_unix"),
               "arch": {}, "cfg": {}}
        try:
            import architecture as _arch
            meta = {}
            # _ARCH is rebuilt from KNOWN_KEYS and may drop _meta, so read
            # it straight from the loaded arch file for a reliable stamp.
            ap = getattr(_arch, "_ARCH_PATH", None)
            if ap and os.path.exists(ap):
                import json as _json
                meta = (_json.load(open(ap)) or {}).get("_meta", {})
            if not meta:
                meta = (getattr(_arch, "_ARCH", {}) or {}).get("_meta", {})
            rec["arch"] = {k: meta.get(k) for k in
                           ("name", "trial_id", "fitness", "version")}
        except Exception:
            pass
        # v1.53 — log the WHOLE config, not a 12-key whitelist. The 44-day
        # V1.072 run could not be fully interpreted afterwards because
        # learning_enabled / agmp_enabled / the reward knobs / the NAS
        # settings were simply not recorded, so there was no way to confirm
        # from the logs alone what the run had actually been doing. This is
        # one record per boot, so logging everything costs nothing and makes
        # every run self-documenting. Non-JSON values are stringified so a
        # stray object can never break the provenance write.
        for k, v in cfg.items():
            rec["cfg"][k] = (v if isinstance(v, (int, float, str, bool))
                             or v is None else str(v))
        # explicit science stamp: the flags that define what the run means
        rec["science"] = {
            "learning_enabled": bool(getattr(self, "_learning", False)),
            "agmp_env": os.environ.get("MNGOL5_AGMP"),
            "nas_explore_enabled": bool(getattr(self, "_nas_enabled", False)),
            "nas_explore_interval_ticks": int(
                getattr(self, "_nas_interval", 0)),
            "reward_eat_dopamine": float(getattr(self, "_reward_eat_da", 0.0)),
            "reward_mate_dopamine": float(
                getattr(self, "_reward_mate_da", 0.0)),
            "dopamine_baseline": float(getattr(self, "_da_base", 0.0)),
            "dopamine_burst_max": float(getattr(self, "_da_max", 0.0)),
            "fitness_hi": "logged at death (unsaturated companion metric)",
            "age_checkpoints": list(_AGE_CKPTS),
        }
        self.history.provenance(rec)

    def m_advantage(self, nx):
        """v1.57 — selection advantage in [0,1] from this NxEr's smoothed
        M-claim compliance. 0 until its brain has been scored, so newborns
        earn nothing for free."""
        if not self._m_sel:
            return 0.0
        mf = getattr(nx, "m_fit_ema", None)
        if mf is None:
            return 0.0
        lo = self._m_sel_floor
        adv = (mf - lo) / max(1e-6, 1.0 - lo)
        return 0.0 if adv < 0.0 else (1.0 if adv > 1.0 else adv)

    def _mate_gate(self, nx):
        """v1.57 — food needed before this NxEr will mate. M-compliant
        brains get a discount, so satisfying the M claims raises
        reproductive rate as well as survival. Falls back to the flat
        threshold when selection is off or the NxEr has no score yet."""
        base = self.bio_mate_min_food
        if not self._m_sel or self._m_sel_mate <= 0.0:
            return base
        return base * (1.0 - self._m_sel_mate * self.m_advantage(nx))

    def _score_m_compliance(self, per_brain):
        """v1.54 — give every NxEr its own M-compliance score.

        For each brain we check its own M values against M_BANDS and record
        the fraction that land in band. Eight of the bands are per-brain
        measurable (M1 E/I/N, M2 gate mean + cross-link spread, M5, M6, M8,
        M10 dead/lesion); M10_heritability_r and the Mg g-structure are
        irreducibly population-level and are excluded here.

        This is MEASUREMENT ONLY in v1.54 — nothing selects on it yet. It
        exists so we can finally ask which architectures build brains that
        satisfy the M claims, which the 46,957-trial NAS could never answer
        while fitness (foraging) was the only per-NxEr outcome recorded.

        A smoothed score is kept alongside the instantaneous one because a
        single ~1/min sample of a spiking network is noisy; the EMA is the
        one worth correlating against architecture."""
        alpha = float(self.cfg.get("m_score_ema_alpha", 0.25))
        for nid, m in per_brain.items():
            nx = self.nxers.get(nid)
            if nx is None or not nx.alive:
                continue
            hits = 0
            n = 0
            dev = 0.0
            graded = 0.0
            # v1.58 — weighted variant. Equal weighting lets a brain score
            # well while failing the claims that never pass: across runs
            # M1_E passed 14% and M6 0%, while M2_gate_xlink_std passed
            # 100%. m_fit_w up-weights the hard bands so selection pushes
            # where the science actually needs movement. Logged alongside
            # the equal-weight m_fit, which stays the comparable series.
            gw = 0.0
            wsum = 0.0
            for k, v in m.items():
                band = M_BANDS.get(k)
                if band is None or v is None:
                    continue
                n += 1
                lo, hi = band
                w = max(1e-9, hi - lo)
                bw = _M_HARD.get(k, 1.0)
                wsum += bw
                if lo <= v <= hi:
                    hits += 1
                    graded += 1.0
                    gw += bw
                else:
                    # how far outside, normalised by the band's width
                    d = ((lo - v) if v < lo else (v - hi)) / w
                    dev += d
                    # v1.55 — graded credit: a brain just outside a band
                    # scores far better than one wildly outside. The binary
                    # hit-fraction saturates (M6 is structurally
                    # unachievable, so 7/8 = 0.875 is a hard ceiling and 45
                    # NxErs tied there exactly, unrankable). This is
                    # continuous and never ties.
                    graded += 1.0 / (1.0 + d)
                    gw += bw / (1.0 + d)
            if n == 0:
                continue
            score = hits / n
            nx.m_last = m
            nx.m_score = round(score, 4)
            nx.m_in_band = hits
            nx.m_n_checked = n
            nx.m_deviation = round(dev / n, 4)
            nx.m_fit = round(graded / n, 4)
            nx.m_fit_w = round(gw / wsum, 4) if wsum > 0 else None
            prev = getattr(nx, "m_score_ema", None)
            nx.m_score_ema = round(
                score if prev is None else (1 - alpha) * prev + alpha * score,
                4)
            pf = getattr(nx, "m_fit_ema", None)
            nx.m_fit_ema = round(
                nx.m_fit if pf is None
                else (1 - alpha) * pf + alpha * nx.m_fit, 4)
            nx.m_samples = int(getattr(nx, "m_samples", 0)) + 1

    def compute_m12(self, alive):
        """v1.44 — assemble the offline-GoL M-metrics we can faithfully
        measure online, from (a) the worker science sample (M1/M2/M5/M6/
        M8/M10-lesion — one ~1/min pass over the brains) and (b) the
        main-process behavioural accumulators (M7) plus heritability
        (M10-r). Returns a compact dict (each value plus an in_band flag
        where a band is defined). M3 (PAC), M4 (weights) and M9
        (compositional) are intentionally absent — see M_BANDS comment.
        Cheap and low-volume: one extra worker round-trip + ~25 numbers
        logged per history sample. Resets the M7 accumulators."""
        import math
        out = {}

        # ---- worker science sample (brain-internal metrics) ----
        sci = None
        try:
            sci, per_brain = self.pool.sample_science()
        except Exception:
            sci, per_brain = None, {}
        # v1.54 — score each NxEr against the M bands from its OWN brain
        self._last_per_brain = per_brain or {}     # v1.58 — for W audit
        if per_brain:
            try:
                self._score_m_compliance(per_brain)
            except Exception:
                pass
        if sci:
            nneu = sci.get("nneu", 0) or 0
            if nneu > 0:
                e = sci["e"] / nneu
                i = sci["h"] / nneu
                nt = sci["z"] / nneu
                out["M1_E"] = round(e, 4)
                out["M1_I"] = round(i, 4)
                out["M1_N"] = round(nt, 4)
                out["M1_deviation"] = round(abs(e - 0.22) + abs(i - 0.10)
                                            + abs(nt - 0.68), 4)
            # M5 branching
            if sci.get("br_n", 0) > 0:
                bn = sci["br_n"]
                mean_br = sci["br_sum"] / bn
                var = max(0.0, sci["br_sq"] / bn - mean_br * mean_br)
                out["M5_branching"] = round(mean_br, 4)
                out["M5_branching_std"] = round(math.sqrt(var), 4)
                out["M5_subcritical_frac"] = round(sci["br_sub"] / bn, 3)
                out["M5_supercritical_frac"] = round(sci["br_sup"] / bn, 3)
            # M6 ACW heterogeneity (std of intrinsic timescale)
            if sci.get("ts_n", 0) > 1:
                tn = sci["ts_n"]
                mean_ts = sci["ts_sum"] / tn
                var = max(0.0, sci["ts_sq"] / tn - mean_ts * mean_ts)
                out["M6_acw_mean"] = round(mean_ts, 3)
                out["M6_acw_heterogeneity"] = round(math.sqrt(var), 3)
            # M2 CTC gate (instantaneous mean + cross-link spread)
            if sci.get("gate_n", 0) > 0:
                gn = sci["gate_n"]
                mean_g = sci["gate_sum"] / gn
                var = max(0.0, sci["gate_sq"] / gn - mean_g * mean_g)
                out["M2_mean_gate"] = round(mean_g, 4)
                out["M2_gate_xlink_std"] = round(math.sqrt(var), 4)
                out["M2_n_links"] = gn
            # v1.53 — W: synaptic weight drift. Lets the excitatory-collapse
            # question ("are the brains going quiet because plasticity is
            # net-depressing the weights?") be answered from the log.
            if sci.get("w_n", 0) > 0:
                wn = sci["w_n"]
                mean_w = sci["w_sum"] / wn
                wvar = max(0.0, sci["w_sq"] / wn - mean_w * mean_w)
                out["W_mean"] = round(mean_w, 5)
                mean_w_abs = sci["w_abs"] / wn
                out["W_mean_abs"] = round(mean_w_abs, 5)
                out["W_std"] = round(math.sqrt(wvar), 5)
                out["W_pos_frac"] = round(sci["w_pos"] / wn, 4)
                out["W_n_sampled"] = wn
                out["W_n_syn_total"] = sci.get("w_total", wn)   # v1.57
                # v1.58 — SELF-AUDIT. The population figure is by
                # construction the synapse-weighted mean of the per-brain
                # figures, so recomputing it FROM per_brain in the same
                # pass must reproduce it exactly. It has not: per-brain read
                # 0.0107 against a population 0.3365 (31x), and synapse
                # count was ruled out as the cause. If W_audit_ratio is ~1
                # the aggregation is sound and the gap is a property of
                # which brains get logged; if it is far from 1 there is a
                # real fault in one of the two paths. Either answer closes
                # the question that has blocked every W_* conclusion.
                try:
                    _num = _den = 0.0
                    for _m in (getattr(self, "_last_per_brain", None) or {}).values():
                        _wn = _m.get("W_n")
                        _wa = _m.get("W_mean_abs")
                        if _wn and _wa is not None:
                            _num += _wa * _wn
                            _den += _wn
                    if _den > 0:
                        _recomp = _num / _den
                        out["W_audit_recomputed"] = round(_recomp, 6)
                        out["W_audit_ratio"] = round(
                            mean_w_abs / _recomp, 4) if _recomp else None
                        out["W_audit_n_brains"] = len(
                            getattr(self, "_last_per_brain", None) or {})
                        out["W_audit_n_syn"] = int(_den)
                except Exception:
                    pass
            # M8 sphere specialisation — sensory firing fraction minus
            # association firing fraction (sensory should sit higher)
            sa = (sci["sensory_act"] / sci["sensory_n"]
                  if sci.get("sensory_n", 0) else None)
            aa = (sci["assoc_act"] / sci["assoc_n"]
                  if sci.get("assoc_n", 0) else None)
            ma = (sci["motor_act"] / sci["motor_n"]
                  if sci.get("motor_n", 0) else None)
            if sa is not None:
                out["M8_sensory_act"] = round(sa, 4)
            if aa is not None:
                out["M8_assoc_act"] = round(aa, 4)
            if ma is not None:
                out["M8_motor_act"] = round(ma, 4)
            if sa is not None and aa is not None:
                out["M8_sensory_vs_assoc"] = round(sa - aa, 4)
            # M10 lesion robustness
            if sci.get("dead_n", 0) > 0:
                out["M10_dead_neuron_frac"] = round(
                    sci["dead_sum"] / sci["dead_n"], 4)
            ok = (sci["mamp_ok_sum"] / sci["mamp_ok_n"]
                  if sci.get("mamp_ok_n", 0) else 0.0)
            les = (sci["mamp_les_sum"] / sci["mamp_les_n"]
                   if sci.get("mamp_les_n", 0) else None)
            if les is not None and ok > 0.05:
                out["M10_lesion_retention"] = round(les / ok, 3)
            out["_n_brains_sampled"] = sci.get("n_brains", 0)

        # ---- v1.54: population distribution of per-NxEr M-compliance ----
        # The population averages above can sit in-band while almost no
        # individual brain is compliant (and vice versa). This says how many
        # actual brains satisfy the claims, which is what we want to select
        # for later.
        try:
            sc = [a.m_score_ema for a in alive
                  if getattr(a, "m_score_ema", None) is not None]
            if sc:
                sc_sorted = sorted(sc)
                n = len(sc)
                mean_sc = sum(sc) / n
                out["Mc_mean"] = round(mean_sc, 4)
                out["Mc_median"] = round(sc_sorted[n // 2], 4)
                out["Mc_best"] = round(sc_sorted[-1], 4)
                out["Mc_frac_ge_0_75"] = round(
                    sum(1 for x in sc if x >= 0.75) / n, 4)
                out["Mc_frac_ge_0_90"] = round(
                    sum(1 for x in sc if x >= 0.90) / n, 4)
                out["Mc_n"] = n
            # v1.55 — continuous graded compliance (no ceiling ties)
            fc = [a.m_fit_ema for a in alive
                  if getattr(a, "m_fit_ema", None) is not None]
            if fc:
                fc_s = sorted(fc)
                out["Mc_fit_mean"] = round(sum(fc) / len(fc), 4)
                out["Mc_fit_median"] = round(fc_s[len(fc) // 2], 4)
                out["Mc_fit_best"] = round(fc_s[-1], 4)
        except Exception:
            pass

        # ---- M7 self-sustained activity (behavioural accumulator) ----
        if self._m7_zero_n > 0 and self._m7_drv_n > 0:
            zero = self._m7_zero_sum / self._m7_zero_n
            drv = self._m7_drv_sum / self._m7_drv_n
            if drv > 1e-6:
                out["M7_zero_input_ratio"] = round(zero / drv, 4)
            out["M7_zero_motor_amp"] = round(zero, 4)
            out["M7_driven_motor_amp"] = round(drv, 4)
            out["M7_n_zero"] = self._m7_zero_n
            out["M7_n_driven"] = self._m7_drv_n
        # reset the M7 window for the next sample
        self._m7_zero_sum = self._m7_drv_sum = 0.0
        self._m7_zero_n = self._m7_drv_n = 0

        # ---- M10 heritability (parent-avg fitness at birth vs child) ----
        ps, cs = [], []
        living = {a.id: a for a in alive}
        for cid, pavg in self._herit_pending.items():
            a = living.get(cid)
            if a is not None:
                try:
                    ps.append(pavg)
                    cs.append(float(a.stats.fitness))
                except Exception:
                    pass
        if len(ps) >= 8:
            n = len(ps)
            mp = sum(ps) / n
            mc = sum(cs) / n
            cov = sum((p - mp) * (c - mc) for p, c in zip(ps, cs))
            sp = math.sqrt(sum((p - mp) ** 2 for p in ps))
            sc = math.sqrt(sum((c - mc) ** 2 for c in cs))
            if sp * sc > 0:
                out["M10_heritability_r"] = round(cov / (sp * sc), 4)
            out["M10_heritability_pairs"] = n

        # ---- in-band flags ----
        bands = {}
        for k, v in list(out.items()):
            f = _m_in_band(k, v)
            if f is not None:
                bands[k] = f
        if bands:
            out["_in_band"] = bands
            out["_in_band_count"] = sum(bands.values())
            out["_band_total"] = len(bands)
        return out

    def history_sample(self):
        if self.history is None:
            return
        import statistics as _st
        alive = [a for a in self.nxers.values() if a.alive]
        n = len(alive)
        if n == 0:
            return
        n_man = sum(1 for a in alive if a.is_managed)

        def _col(getter):
            out = []
            for a in alive:
                try:
                    v = getter(a)
                    if isinstance(v, (int, float)):
                        out.append(v)
                except Exception:
                    pass
            return out

        def _dist(vals):
            if not vals:
                return None
            vals = sorted(vals)
            m = len(vals)
            def pct(p):
                return vals[min(m - 1, int(p * m))]
            return {"mean": round(_st.mean(vals), 4),
                    "sd": round(_st.pstdev(vals), 4) if m > 1 else 0.0,
                    "min": round(vals[0], 4), "max": round(vals[-1], 4),
                    "p10": round(pct(0.10), 4), "p50": round(pct(0.50), 4),
                    "p90": round(pct(0.90), 4)}

        # heritable traits — the live mirror of the NAS search
        traits = {}
        for key, gp in (
                ("thr_inh", lambda a: getattr(a.params, "firing_threshold_inhibitory", None)),
                ("thr_exc", lambda a: getattr(a.params, "firing_threshold_excitatory", None)),
                ("refr", lambda a: getattr(a.params, "refractory_period_ticks", None)),
                ("ahp", lambda a: getattr(a.params, "post_spike_mp_reset", None)),
                ("timescale", lambda a: getattr(a.params, "intrinsic_timescale_default", None)),
                ("learning_rate", lambda a: getattr(a.params, "learning_rate", None))):
            d = _dist(_col(gp))
            if d:
                traits[key] = {"mean": d["mean"], "sd": d["sd"]}

        # trinary firing distribution (the v195 corner) — best effort
        trinary = None
        try:
            fr = self.pool.sample_firing()
            if fr is not None:
                exc, neu, inh, nn = fr
                trinary = {"M1_exc": round(exc, 4),
                           "M1_neutral": round(neu, 4),
                           "M1_inh": round(inh, 4),
                           "neurons_sampled": nn,
                           # signed distance from the paper corner
                           "L1_to_paper": round(abs(exc - 0.22)
                                                + abs(neu - 0.68)
                                                + abs(inh - 0.10), 4),
                           "excitation_dominant": bool(exc > inh)}
        except Exception:
            pass

        # per-interval rates from lifetime-counter deltas
        lt = self.lifetime
        cur = {k: lt.get(k, 0) for k in
               ("total_spawns", "total_births_mating", "total_deaths",
                "total_matings", "total_food_eaten")}
        rates = {}
        prev = self._hist_prev
        dticks = self.tick - self._hist_last_tick
        if prev is not None and dticks > 0:
            secs = dticks * self.dt
            for k in cur:
                per_min = (cur[k] - prev.get(k, cur[k])) / max(1e-6, secs) * 60.0
                rates[k.replace("total_", "") + "_per_min"] = round(per_min, 2)
        self._hist_prev = cur
        self._hist_last_tick = self.tick

        rec = {
            "tick": self.tick,
            "uptime_s": round(lt.get("uptime_seconds", 0.0), 1),
            "alive": n, "managed": n_man, "tracked": len(self.nxers),
            "food": len(self.foods),
            "trinary": trinary,
            "g": _dist(_col(lambda a: getattr(a.stats, "g_factor", 0.0))),
            "fitness": _dist(_col(lambda a: a.stats.fitness)),
            "lifespan_ticks": _dist(_col(
                lambda a: self.tick - getattr(a, "born_tick", self.tick))),
            "offspring": _dist(_col(lambda a: len(a.offspring_ids))),
            "traits": traits,
            "rates": rates,
            "g_structure": getattr(self, "_g_cache", {}),
            "pool_mode": self.pool.mode_info().get("mode"),
            "perf": self.get_perf(),     # v1.39 per-phase tick timing (ms)
        }
        # v1.44 — offline-GoL M-metrics (M1/M2/M5/M6/M7/M8/M10 + heritability)
        if self._m12_on:
            try:
                rec["m12"] = self.compute_m12(alive)
                self._last_m12 = rec["m12"]
            except Exception as _e:
                rec["m12"] = {"_error": str(_e)}
        self.history.sample(rec)

    def _update_all_time(self):
        """Update the full-population rank pool (id -> best value ever
        for each metric) and rebuild ONE deterministic sorted list per
        metric. Both the all-time board and rank_of() read from the
        same list, so the position a NxEr sees in "My ranks" matches
        exactly the row it occupies in the panel — even on ties."""
        pool = self._rank_pool
        # 1. refresh pool with current LIVE NxEr values (counter-like
        #    metrics monotonically increase; max() keeps best-ever)
        for a in self.nxers.values():
            if not a.alive:
                continue
            for m in RANK_METRICS:
                v = _metric(a, m)
                pm = pool[m]
                if a.id not in pm or v > pm[a.id]:
                    pm[a.id] = v
                    self._nxer_names[a.id] = a.name
                    # v1.43 — note that this live NxEr just set (or
                    # raised) the all-time record for metric m. The
                    # game server flushes these to state/best/ hourly by
                    # exporting the FULL brain WHILE THE NXER IS STILL
                    # ALIVE, so the true all-time champion's model is
                    # archived even though it will usually be dead by the
                    # time the hourly sweep runs. id is enough; the
                    # server re-checks it's still alive and still #1.
                    rec = self._record_breakers.get(m)
                    if rec is None or v > rec[1]:
                        self._record_breakers[m] = (a.id, v)
        # 2. bound memory — keep the CAP best entries per metric
        CAP = 8000
        for m in RANK_METRICS:
            pm = pool[m]
            if len(pm) > CAP:
                top = sorted(pm.items(), key=lambda kv: -kv[1])[:CAP]
                pool[m] = dict(top)
        # 3. ONE sorted list per metric. Ties broken by id ascending so
        #    every NxEr gets a UNIQUE deterministic position, the same
        #    one displayed in the panel. (Previous code returned the
        #    same rank for tied values; the panel listed them in three
        #    different rows — that was the visible mismatch.)
        for m in RANK_METRICS:
            sl = sorted(pool[m].items(),
                        key=lambda kv: (-kv[1], kv[0]))
            self._rank_index[m] = {nid: i + 1
                                   for i, (nid, _) in enumerate(sl)}
            top = []
            for nid, val in sl[:10]:
                a = self.nxers.get(nid)
                nm = a.name if a else self._nxer_names.get(nid, "?")
                top.append({"id": nid, "name": nm, "value": val,
                            "alive": bool(a and a.alive)})
            self._rank_top[m] = top
            self.all_time[m] = top      # legacy alias for callers
        self._ranking_cache = None       # rebuilt by ranking()

    def rank_of(self, nx):
        """Integer rank position (1 = best) for nx in each metric,
        using the SAME sorted list the panel renders, so positions
        match exactly. Returns None for a metric when:
          (a) the metric has no data at all (max value <= 0), or
          (b) nx hasn't scored in a counter metric (its value <= 0)
        — the client renders None as "—" instead of a meaningless "#1"."""
        out = {}
        for m in RANK_METRICS:
            idx = self._rank_index.get(m, {})
            top = self._rank_top.get(m, [])
            if not top or top[0]["value"] <= 0:
                out[m] = None          # no data anywhere
                continue
            my_v = self._rank_pool[m].get(nx.id, _metric(nx, m))
            if m in COUNTER_METRICS and my_v <= 0:
                out[m] = None          # haven't scored
                continue
            pos = idx.get(nx.id)
            out[m] = pos if pos is not None else None
        return out

    # ---- snapshots --------------------------------------------------
    def _brain_building(self, a):
        """True while a freshly created NxEr's brain is still being
        built by the dedicated builder process (it has not acted yet).
        Zero protocol/lookup cost — pure heuristic: young AND has not
        explored a single cell. Clears the instant it starts moving."""
        return ((self.tick - a.born_tick) < 90
                and a.stats.explored == 0
                and a.is_managed)

    def get_perf(self):
        """v1.39 — per-phase tick timing (EWMA, ms) for the admin console
        and history log. tps is the instantaneous tick_ms→Hz estimate."""
        p = self._perf
        tick_ms = p["tick_ms"]
        return {
            "tick_ms": round(tick_ms, 2),
            "phase_a_sense_ms": round(p["a_ms"], 2),
            "phase_b_brains_ms": round(p["b_ms"], 2),
            "phase_c_act_ms": round(p["c_ms"], 2),
            "brains_due_per_tick": p["due"],
            "tps_est": round(1000.0 / tick_ms, 1) if tick_ms > 0.01 else 0.0,
        }

    def world_snapshot(self):
        """Public, viewer-safe broadcast payload."""
        slim = bool(self.cfg.get("slim_broadcast", False))
        alive_nx = [a for a in self.nxers.values() if a.alive]
        nx_pv = []
        for a in alive_nx:
            pv = a.public_view()
            if self._brain_building(a):
                pv["b"] = 1            # brain still building (cue)
            if slim:
                pv.pop("c", None)      # colour sent via periodic roster
            nx_pv.append(pv)
        snap = {
            "tick": self.tick,
            "world": {"size": self.world.size},
            "nxers": nx_pv,
            "alive": len(alive_nx),
            "g": self._g_cache,
            "events": self._events,
            "ranking": self.ranking(),
        }
        if not slim:
            snap["foods"] = [{"x": f["pos"][0], "y": f["pos"][1]}
                             for f in self.foods.values()]
            return snap
        # --- slim: foods + colour roster only periodically -----------
        now = time.time()
        nf = len(self.foods)
        if (now - self._last_food_bcast
                >= float(self.cfg.get("food_refresh_secs", 1.0))
                or nf != self._last_food_count):
            snap["foods"] = [{"x": f["pos"][0], "y": f["pos"][1]}
                             for f in self.foods.values()]
            self._last_food_bcast = now
            self._last_food_count = nf
        # else: omit "foods" → client keeps its cached set
        if (now - self._last_color_bcast
                >= float(self.cfg.get("color_refresh_secs", 2.0))):
            snap["colors"] = {a.id: a.color for a in alive_nx}
            self._last_color_bcast = now
        return snap

    def world_snapshot_raw(self):
        """COMPACT snapshot for the snapshot-worker subprocess.

        Builds tuples instead of dicts: orders of magnitude cheaper
        than `world_snapshot()` because tuples skip key-hashing and
        dict-resize cost in Python. The worker rebuilds the full
        dict-shaped payload on its own core. Tuple field order MUST
        stay in sync with `server/snapshot_worker.py`.
        """
        raw = []
        ap = raw.append
        building = self._brain_building
        for a in self.nxers.values():
            if a.alive:
                ap((a.id, a.name, a.pos[0], a.pos[1],
                    True, a.is_managed, a.color,
                    1 if a.last_sing_level > 0 else 0,
                    1 if building(a) else 0))
        raw_foods = [(f["pos"][0], f["pos"][1])
                     for f in self.foods.values()]
        return (self.tick, self.world.size,
                raw, raw_foods, len(raw),
                self._g_cache, self._events, self.ranking())

    def ranking(self):
        # cached — boards only change every rank_interval_ticks, but
        # world_snapshot() (hence this) runs every broadcast (~10 Hz).
        # Rebuilding the formatted dict each time was wasted serial work.
        rk = getattr(self, "_ranking_cache", None)
        if rk is not None:
            return rk
        rk = {}
        for m in RANK_METRICS:
            entries = []
            for e in self._rank_top.get(m, [])[:5]:
                # for counter metrics a value of 0 means "hasn't scored
                # at all" — skip rather than show "#1 Foo: 0", which
                # was the misleading display
                if m in COUNTER_METRICS and e["value"] <= 0:
                    continue
                entries.append({"name": e["name"],
                                "value": round(e["value"], 4),
                                "alive": e["alive"]})
            rk[m] = entries
        self._ranking_cache = rk
        return rk

    def state_dict(self):
        """Crash-recovery snapshot (full).

        ``snapshot_brains`` (config, default true) controls whether each
        NxEr's full multi-sphere brain is serialised. With brains:
        exact resume (learning preserved) but ~250-300 KB / NxEr. Without:
        the world, names, rankings and lineage still resume, but brains
        are rebuilt fresh on restart — far smaller snapshots for a busy
        24/7 server with hundreds of NxErs.
        """
        keep_brains = bool(self.cfg.get("snapshot_brains", True))
        nxers = []
        for a in self.nxers.values():
            rec = {
                "id": a.id, "name": a.name, "pos": a.pos,
                "alive": a.alive, "food": a.food,
                "is_managed": a.is_managed,
                "is_male": a.is_male,
                "can_land": a.can_land,
                "can_sea": a.can_sea,
                "stats": a.stats.as_dict(),
                "params": _params_to_dict(a.params),
                "parents": a.parents,
                "offspring_ids": a.offspring_ids,
                # v1.53 — born_tick was never saved, so every NxEr restored
                # from a snapshot came back with born_tick = 0 and reported
                # an age of "the whole world's tick count". That corrupted
                # lifespan in obituaries/nas_trials after any reboot, and
                # would have fired every within-life age checkpoint at once.
                "born_tick": getattr(a, "born_tick", 0),
                "food_by_age": getattr(a, "food_by_age", {}),
            }
            if keep_brains:
                bd = self.pool.export(a.id)
                if bd is not None:
                    rec["brain"] = bd
            nxers.append(rec)
        # update accumulated uptime on every save so a hard kill
        # doesn't lose more than `snap_secs` of session time
        now = time.time()
        self.lifetime["uptime_seconds"] += max(0.0, now - self._uptime_t0)
        self._uptime_t0 = now
        return {
            "tick": self.tick,
            "next_nxer_id": self.next_nxer_id,
            "next_food_id": self.next_food_id,
            # v1.53 — keep NAS trial ids unique across a reboot
            "nas_trial_seq": int(getattr(self, "_nas_trial_seq", 0)),
            "all_time": self.all_time,
            "lifetime":  self.lifetime,
            "names_state": self.names.state(),
            # v1.43 — persist the id->name map so all-time record-holders
            # keep their names across a reboot (the board's "?" bug). Kept
            # compact: only the ids that still appear in some rank pool.
            "nxer_names": {
                str(nid): self._nxer_names.get(nid)
                for m in RANK_METRICS
                for nid in self._rank_pool.get(m, {})
                if self._nxer_names.get(nid)
            },
            "foods": [{"id": k, "pos": v["pos"],
                       "remaining": v.get("remaining", 25)}
                      for k, v in self.foods.items()],
            "nxers": nxers,
        }


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
def _dir_index(dx, dy):
    try:
        return DIR_OFFSETS.index((dx, dy))
    except ValueError:
        return 0


def _fitness(s):
    # Matches the offline v189 reference exactly (game_loop.py ~1849).
    # The previous formula here was broken: it had no mates_performed
    # term AND a "branching" term that always evaluated near 0.15
    # regardless of behaviour, so every NxEr's fitness rounded to 0.3.
    norm_food = min(s.food_found / 100.0, 1.0)
    norm_expl = min(s.explored / 1000.0, 1.0)
    norm_time = min(s.time_lived_s / 1000.0, 1.0)
    norm_ener = (min(s.energy_efficiency / 10.0, 1.0)
                 if s.energy_efficiency else 0.0)
    # offline uses temporal_sync_score / 2; we don't track that, so
    # derive a sync proxy from how close branching is to 1.0 (critical
    # state). 0 when far from critical, 1 when exactly at branching=1.
    sync_proxy = max(0.0, 1.0 - abs(s.branching - 1.0))
    norm_sync = min(sync_proxy, 1.0)
    norm_mates = min(s.mates_performed / 5.0, 1.0)
    return (norm_food  * 0.25
            + norm_expl  * 0.15
            + norm_time  * 0.20
            + norm_ener  * 0.10
            + norm_sync  * 0.10
            + norm_mates * 0.20)


# v1.53 — ages (in ticks) at which each NxEr's cumulative food_found is
# sampled, so a within-life foraging curve can be reconstructed at death.
# Roughly doubling, covering ~1 min to ~2.2 h of lived time at 20 tps.
# v1.58 — per-band weights for m_fit_w. Bands that persistently fail get
# more say, so compliance pressure lands where the claims are unmet.
# Observed pass rates: M2_gate_xlink_std 100%, M8 85%, M5 64%, M1_N 62%,
# M2_mean_gate 61%, M1_I 43%, M1_E 14%, M6 0%.
_M_HARD = {
    "M1_E": 3.0,
    "M6_acw_heterogeneity": 3.0,
    "M1_I": 2.0,
    "M5_branching": 1.5,
    "M1_N": 1.5,
}

_AGE_CKPTS = (1250, 2500, 5000, 10000, 20000, 40000, 80000, 160000)


def _fitness_hi(s):
    """v1.53 — high-resolution companion to _fitness (MEASUREMENT ONLY;
    selection, ranking and best/ still use _fitness so 44 days of history
    stay comparable).

    The legacy formula saturates badly. Over the 44-day V1.072 run, 69% of
    NxErs capped norm_food, 90% capped norm_expl, 54% capped norm_time, and
    53% capped all three at once. Inside that group food_found spanned
    100..10,610 — a 100x range — while fitness sat at 0.714 +/- 0.040. A
    ruler that blind cannot show whether architecture or learning matters,
    which is very likely why every knob correlation came back ~0.

    This version keeps the same 6 terms and weights but scales the
    unbounded ones logarithmically against the observed ceilings, so it
    keeps resolving differences at the top of the distribution instead of
    flattening them."""
    def lg(x, cap):
        x = float(x) if x else 0.0
        if x <= 0.0:
            return 0.0
        return min(1.0, math.log1p(x) / math.log1p(cap))
    norm_food = lg(s.food_found, 10000.0)      # observed max ~10,610
    norm_expl = lg(s.explored, 50000.0)        # median alone was ~10,871
    norm_time = lg(s.time_lived_s, 100000.0)   # observed max ~95,700 s
    norm_ener = (min(s.energy_efficiency / 10.0, 1.0)
                 if s.energy_efficiency else 0.0)
    sync_proxy = max(0.0, 1.0 - abs(s.branching - 1.0))
    norm_mates = lg(s.mates_performed, 20.0)
    return (norm_food * 0.25
            + norm_expl * 0.15
            + norm_time * 0.20
            + norm_ener * 0.10
            + min(sync_proxy, 1.0) * 0.10
            + norm_mates * 0.20)


def _metric(a, m):
    if m == "food_found":
        return float(a.stats.food_found)
    if m == "food_taken":
        return float(a.stats.food_taken)
    if m == "explored":
        return float(a.stats.explored)
    if m == "time_lived":
        return float(a.stats.time_lived_s)
    if m == "mates_performed":
        return float(a.stats.mates_performed)
    if m == "fitness":
        return float(a.stats.fitness)
    if m == "g":
        return float(a.stats.g_factor)
    return 0.0


def hash_password(pw, salt):
    return hashlib.sha256((salt + ":" + pw).encode("utf-8")).hexdigest()
