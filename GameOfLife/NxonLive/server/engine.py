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
        self.known_food_ids = set()
        self.last_sing_level = 0.0
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
        # behaviour bookkeeping
        self.last_move_tick = 0
        self.last_eat_tick = -10000   # so first eat is allowed immediately
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
        d.update({
            "stats": self.stats.as_dict(),
            "food": round(self.food, 1),
            "energy": round(self.energy, 1),
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
    p.sphere_topology = "chc6"   # never overridable
    return p


# --------------------------------------------------------------------
# World — terrain grid + food.
# --------------------------------------------------------------------
class World:
    def __init__(self, size, sea_pct, rock_pct, seed=0,
                 earth_map=False):
        self.size = int(size)
        self.sea_pct = float(sea_pct)
        self.rock_pct = float(rock_pct)
        self.earth_map = bool(earth_map)
        # The flat world is projected onto a sphere, so the first/last
        # rows in y all converge at the two poles -> heavy crowding.
        # Make a thin band at each pole UNREACHABLE (treated like rock)
        # so NxErs/food never pile up there. ~7% of the height each.
        self._pole = max(2, int(self.size * 0.07))
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

    def in_pole(self, y):
        """True for the unreachable polar bands (top & bottom rows)."""
        return y < self._pole or y >= self.size - self._pole

    def passable(self, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if self.in_pole(y):
            return False
        return self.terrain.get((int(x), int(y)), 0) != 2

    def is_sea(self, x, y):
        return self.terrain.get((int(x), int(y)), 0) == 1

    def terrain_rows(self):
        """Compact terrain for the client: one char per cell per row.
        '.' sea · ',' land · '#' rock. Sent once over REST (static)."""
        out = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                t = self.terrain.get((x, y), 0)
                row.append("#" if t == 2 else ("." if t == 1 else ","))
            out.append("".join(row))
        return out


# --------------------------------------------------------------------
# The engine.
# --------------------------------------------------------------------
class Engine:
    def _init_runtime(self, cfg):
        """All derived/runtime state. Called by BOTH __init__ and the
        crash-restore path so the two can never drift out of sync
        (this is what caused the missing _cell_size on restart)."""
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
        self._rank_index = {m: {} for m in RANK_METRICS}
        self._rank_top = {m: [] for m in RANK_METRICS}
        self._nxer_names = {}        # id -> name (survives pruning)
        self._food_dirty = True
        if not hasattr(self, "_g_cache"):
            self._g_cache = {"pc1": 0.0, "pos_manifold": 0.0,
                             "mean_r": 0.0, "lambda_ratio": 1.0,
                             "n": 0}
        from .brain_pool import BrainPool
        ew = cfg.get("engine_workers", 0)
        if not ew or int(ew) <= 0:
            ew = max(1, (os.cpu_count() or 1) - 1)
        nb = cfg.get("brain_builders", 0)
        nb = int(nb) if nb and int(nb) > 0 else None
        self.pool = BrainPool(
            int(ew), num_builders=nb,
            step_timeout=float(cfg.get("worker_step_timeout", 5.0)))

    def __init__(self, cfg, name_allocator):
        self.cfg = cfg
        self.names = name_allocator
        self.world = World(cfg["world_size"], cfg["sea_pct"],
                           cfg["rock_pct"], cfg.get("world_seed", 12345),
                           earth_map=cfg.get("earth_map", False))
        self.nxers = {}                 # id -> NxEr (alive + recently dead)
        self.foods = {}                 # food_id -> {pos, amount}
        self.tick = 0
        self.next_nxer_id = 0
        self.next_food_id = 0
        # all-time best record (survives deaths + restarts)
        self.all_time = {m: [] for m in RANK_METRICS}   # list of dicts
        self._g_cache = {"pc1": 0.0, "pos_manifold": 0.0,
                         "mean_r": 0.0, "lambda_ratio": 1.0, "n": 0}
        self._init_runtime(cfg)
        self._spawn_food(self._effective_food_cap())
        for _ in range(int(cfg["starting_nxers"])):
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
        nx.born_tick = self.tick
        self.pool.add(nid, _params_to_dict(nx.params))
        self.nxers[nid] = nx
        self._nxer_names[nid] = nx.name
        self._occupied[(nx.pos[0], nx.pos[1])] = nx.id
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
        Uses the spatial hash → scales to thousands of NxErs."""
        x, y = nx.pos
        hunger = max(-1.0, min(1.0, 1.0 - nx.food / 60.0))
        cs = self._cell_size
        cx, cy = x // cs, y // cs
        # nearest food: scan only the 3x3 neighbouring buckets, and
        # skip food on terrain this NxEr cannot enter (so a land-only
        # NxEr never gets a "go sea" direction that would just wedge
        # it against the shore).
        best_d, fdx, fdy = 1e9, 0.0, 0.0
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for fx, fy in self._food_grid.get((gx, gy), ()):
                    f_sea = self.world.is_sea(fx, fy)
                    if (f_sea and not nx.can_sea) or \
                            (not f_sea and not nx.can_land):
                        continue
                    d = (fx - x) ** 2 + (fy - y) ** 2
                    if d < best_d:
                        best_d = d
                        fdx = (1 if fx > x else -1 if fx < x else 0)
                        fdy = (1 if fy > y else -1 if fy < y else 0)
        sight = 1.0 if best_d < 25 else 0.0
        smell = 1.0 / (1.0 + math.sqrt(best_d)) if best_d < 1e8 else 0.0
        terrain = 1.0 if self.world.is_sea(x, y) else 0.0
        # nearest neighbour: only the local + adjacent buckets
        nb = 0.0
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for oid in self._nxer_grid.get((gx, gy), ()):
                    if oid == nx.id:
                        continue
                    o = self.nxers.get(oid)
                    if o and abs(o.pos[0] - x) <= 2 \
                            and abs(o.pos[1] - y) <= 2:
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

    def _act(self, nx, outs):
        """7 motor channels → world actions. Trinary {-1,0,1}."""
        if len(outs) < 7:
            outs = list(outs) + [0] * (7 - len(outs))
        mvx, mvy, social, mate, give, rest, sing = outs[:7]
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
        # ---- idle-exploration safety net (NAS biology) -------------
        # If the brain has produced no movement for idle_explore_s and
        # the dice say so, inject a random heading. v152+ mechanism
        # stops well-fed-but-stuck NxErs from starving in place.
        idle_s = (self.tick - nx.last_move_tick) * self.dt
        forced = False
        if (idle_s >= self.bio_idle_explore_s
                and random.random() < self.bio_explore_prob):
            d = random.choice(DIR_OFFSETS)
            mvx, mvy = d[0], d[1]
            rest = 0
            forced = True
        # ---- STANDALONE hunger reflex (matches offline v189
        # dopamine-driven foraging). When hungry AND the sensors
        # report a food direction, override the brain's (usually
        # random) motor output and step TOWARD the food.
        #
        # Two knobs in world_config.json control how exploratory
        # NxErs are:
        #   * hunger_threshold_pct (default 0.55): the NxEr only
        #     seeks food when food < threshold * start_food. At 0.55
        #     with start_food=120 the threshold is 66 — so a fresh
        #     NxEr wanders freely under brain control until it
        #     metabolises down to ~half full, instead of being
        #     anchored to food cells from the moment it's slightly
        #     peckish.
        #   * food_wander_ticks (default 40): after every bite the
        #     floor is suppressed for this many ticks so the NxEr
        #     walks away under brain control before being dragged
        #     back. 40 ticks ≈ 2 s @ 20 tps gives a visible random
        #     walk excursion of ~6-8 cells before they orbit back
        #     for the next bite.
        li = nx._last_sensory
        if (li is not None
                and nx.food < (self.bio_start_food
                               * self.bio_hunger_threshold)
                and (li[1] or li[2])
                and self.tick >= nx.wander_until_tick):
            mvx, mvy = int(li[1]), int(li[2])
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
                nx.last_move_tick = self.tick
                if target not in nx.visited:
                    nx.visited.add(target)
                    nx.stats.explored = len(nx.visited)
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
                nx.food += 1.0
                nx.stats.food_found += 1
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
        if social == 1 or give == 1:
            for o in _neighbours():
                if give == 1 and nx.food > 10:
                    nx.food -= 1
                    o.food += 1
                elif social == 1 and o.food > 1:
                    o.food -= 1
                    nx.food += 1
                    nx.stats.food_taken += 1
                break
        elif nx.food < 5:
            for o in _neighbours():
                if o.food > 5 and random.random() < 0.30:
                    o.food -= 1
                    nx.food += 1
                    nx.stats.food_taken += 1
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
        opposite_neighbour = any(
            o.is_male != nx.is_male for o in _neighbours())
        if mate == 1 or (opposite_neighbour and random.random() < 0.003):
            nx.mating_intent_until_tick = self.tick + 60
        if (nx.mating_intent_until_tick > self.tick
                and self.tick >= nx.mate_cooldown_until
                and nx.food >= 5):
            for o in _neighbours():
                if (o.is_male != nx.is_male
                        and self.tick >= o.mate_cooldown_until
                        and o.food >= 5
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
        # child inherits a parent's params (no genetic op needed here —
        # the substrate's own plasticity drives adaptation)
        child = self._spawn_nxer(
            params=make_params(_params_to_dict(
                a.params if random.random() < 0.5 else b.params)),
            parents=(a.id, b.id),
            terrain_caps=(c_land, c_sea))
        if child:
            a.offspring_ids.append(child.id)
            b.offspring_ids.append(child.id)
            if len(self._events) < 200:
                self._events.append({"k": "mate",
                                     "x": a.pos[0], "y": a.pos[1]})
                self._events.append({"k": "birth", "x": child.pos[0],
                                     "y": child.pos[1]})

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
        self._events = []
        order = [nx for nx in self.nxers.values() if nx.alive]

        # --- Phase A: sense (read-only) — cheap pure-Python ----------
        self._build_spatial()
        # Neural LOD: with brain_step_every = K, each NxEr's brain is
        # stepped once every K ticks (round-robin staggered by id so the
        # load is even), reusing its last motor output in between. K=1 =
        # full fidelity; K=2 ≈ 2x fewer brain computations for a modest
        # behavioural change. The single biggest tunable speed lever
        # besides core count, and it never blocks the web server.
        K = max(1, int(self.cfg.get("brain_step_every", 1)))
        batch = []
        due = []
        for nx in order:
            nx.stats.time_lived_s += self.dt
            if K == 1 or (self.tick + nx.id) % K == 0:
                sens = self._sense(nx)
                nx._last_sensory = sens  # used by _act's hunger floor
                batch.append((nx.id, sens))
                due.append(nx)

        # --- Phase B: brains in PARALLEL across worker processes -----
        # While this blocks on pipe.recv() the GIL is free, so the
        # aiohttp web server stays responsive and clients keep loading.
        results = self.pool.step(batch)
        for nx in due:
            o = results.get(nx.id)
            if o is not None:
                nx._last_out = o          # cache for the skipped ticks

        # --- Phase C: apply actions + metabolism (serial, cheap) ----
        for nx in order:
            outs, br = getattr(nx, "_last_out", ([0] * 7, 1.0))
            self._act(nx, outs)
            idle = max(0.0, (self.tick - nx.last_move_tick) * self.dt)
            # soft metabolic ramp (NAS biology): drain rises with idle
            # time but is capped by max_atrophy, and the base drain is
            # gentle so an active NxEr comfortably outlives its food.
            atrophy = min(self.bio_max_atrophy,
                          1.0 + self.bio_metab_ramp * idle)
            nx.food -= self.bio_base_drain * atrophy
            nx.net.branching_ratio = br
            nx.stats.branching = br
            if nx.food <= 0:
                self._kill(nx)
        # fitness & energy_efficiency are display/ranking-only (they do
        # NOT affect simulation dynamics), so compute them at the rank
        # cadence instead of every tick — a real per-tick saving over
        # the whole population at scale.
        if self.tick % int(self.cfg.get("rank_interval_ticks", 10)) == 0:
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
        # population g (throttled)
        if self.tick % int(self.cfg.get("g_interval_ticks", 30)) == 0:
            self._update_g()
        # anti-extinction respawn
        alive = [a for a in self.nxers.values() if a.alive]
        if len(alive) <= int(self.cfg.get("min_alive", 1)):
            for _ in range(int(self.cfg.get("respawn_batch", 8))):
                self._spawn_nxer()
        # update all-time ranking + prune very old corpses
        # all-time ranking is exact-enough at coarse cadence; rebuilding
        # + sorting it every tick is wasted serial work on the main core
        if self.tick % int(self.cfg.get("rank_interval_ticks", 10)) == 0:
            self._update_all_time()
        self._prune_dead()

    def _kill(self, nx):
        nx.alive = False
        # free the cell so other NxErs (and respawns) can use it
        cell = (nx.pos[0], nx.pos[1])
        if self._occupied.get(cell) == nx.id:
            del self._occupied[cell]
        if len(self._events) < 200:
            self._events.append({"k": "die",
                                 "x": nx.pos[0], "y": nx.pos[1]})
        nx.password_hash = None       # password only valid while alive
        nx.death_tick = self.tick
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

    def world_snapshot(self):
        """Public, viewer-safe broadcast payload."""
        alive_nx = [a for a in self.nxers.values() if a.alive]
        nx_pv = []
        for a in alive_nx:
            pv = a.public_view()
            if self._brain_building(a):
                pv["b"] = 1            # brain still building (cue)
            nx_pv.append(pv)
        return {
            "tick": self.tick,
            "world": {"size": self.world.size},
            "nxers": nx_pv,
            "foods": [{"x": f["pos"][0], "y": f["pos"][1]}
                      for f in self.foods.values()],
            "alive": len(alive_nx),
            "g": self._g_cache,
            "events": self._events,
            "ranking": self.ranking(),
        }

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
            }
            if keep_brains:
                bd = self.pool.export(a.id)
                if bd is not None:
                    rec["brain"] = bd
            nxers.append(rec)
        return {
            "tick": self.tick,
            "next_nxer_id": self.next_nxer_id,
            "next_food_id": self.next_food_id,
            "all_time": self.all_time,
            "names_state": self.names.state(),
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
