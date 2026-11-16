"""
MultiNxonArcAGI3 — the cognitive bridge.

Wraps a Neuraxon Multi-Sphere 2.0 brain (the CHC six-sphere architecture by
default, or the legacy 3-sphere hierarchy) and connects it to an ARC-AGI-3
environment:

    perceive(frame)  : ARC colour grid  ->  trinary sensory channels
    think(reward)    : inject dopamine reward, run the brain N micro-steps
    act(frame)       : motor-sphere outputs  ->  GameAction (+ click x,y)

Design notes
------------
* Attention cursor.  ARC's complex action (ACTION6) lives in a 64x64 click
  space.  Rather than wire 4096 motor neurons, the brain steers a small
  *attention cursor* with two motor outputs and "clicks here" with another.
  The cursor also IS the directional readout (ACTION1..4), so sensory input
  and motor output are spatially coupled — which is exactly what makes the
  paper's sensory->motor correlation metric meaningful (and what the NAS
  fitness rewards).

* Saliency prior.  The encoder gives the brain a movable foveal receptive
  field plus a coarse global view, encoded as signed *gradients toward
  non-background structure*.  "Interesting things are not the background"
  is the one generic, game-agnostic prior ARC-AGI-3 allows.

* Reward as dopamine.  Score increases (and, weakly, useful frame changes)
  are injected as a phasic dopamine pulse via the brain's global
  neuromodulator bus, which gates the DA-dependent STDP already in the
  Neuraxon synapse model.  This is the learning signal — no backprop.

* Exploration safety net.  Mirrors the Game-of-Life idle-explore mechanism:
  if the agent stalls (frame unchanged for a while) or the motor sphere is
  silent, it explores with an architecture-controlled probability.  A frozen
  agent scores zero on ARC, so never-freeze is a hard requirement.
"""

from __future__ import annotations

import os   # v1.115: the v1.112 P0-C encoder check uses os.environ and this was
            # MISSING -- it only fires on a width mismatch, so it lay dormant through
            # v1112-v1114 and crashed the first brain whose encoder did not match n_in.
import random
from typing import Any, Dict, List, Optional, Tuple

from arc_env.structs import FrameData, GameState, GameAction, MAX_GRID

# Brain (vendored Neuraxon Multi-Sphere 2.0)
from config import NetworkParameters
from neuraxon.multisphere import build_brain


def _frame_signature(grid: List[List[int]]) -> int:
    """Cheap order-sensitive hash of a grid, for count-based novelty.

    Matches game_loop._frame_signature so brain-side novelty and the loop's
    distinct-frame metric count the same states.
    """
    h = 1469598103934665603  # FNV-1a 64-bit offset
    for row in grid:
        for v in row:
            h ^= (int(v) + 1) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        h ^= 0xFF  # row separator
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
#  Genome -> NetworkParameters
# --------------------------------------------------------------------------- #
# Keys we copy straight from the architecture JSON onto NetworkParameters.
# (Same wiring philosophy as the Game-of-Life game_loop: the architecture
#  defines the value, the simulation reads it.)
_NEURAL_KEYS = [
    "planning_scale", "plan_margin_thresh", "plan_depth",   # v1.078/1.079 real planning
    "num_input_neurons", "num_output_neurons", "num_hidden_neurons_default",
    "connection_probability", "firing_threshold_excitatory",
    "firing_threshold_inhibitory", "spontaneous_firing_rate",
    "intrinsic_timescale_default", "resting_potential_decay",
    "sensorimotor_coupling", "symmetric_stdp", "refractory_period_ticks",
    "post_spike_mp_reset", "sphere_topology", "cross_sphere_coupling",
    "cryst_capacity", "free_energy_beta",
    # v1.062 — SEARCHABLE architecture genes (evolved topology)
    "evo_num_spheres", "evo_ff_density", "evo_fb_density",
    "evo_lateral_density", "evo_recurrent", "evo_middle_width",
    # v1.063 searchable sphere-type + long-range wiring genes
    "evo_sphere_types", "evo_skip_density", "evo_recurrent_frac",
    # v1.005 — input-drive / afferent gains (how strongly perception drives
    # the spheres; meaningful levers for ARC responsiveness)
    "afferent_synapse_strength", "sensory_input_gain",
    "proprioceptive_afferent_gain",
    # v1.006 "deep" genes — inter-sphere coupling, dopamine/reward learning,
    # STDP time constants, and E/I balance / membrane dynamics.
    "ms_ff_gain", "ms_fb_gain", "ms_coherence_strength",
    "ms_projection_plasticity", "dopamine_reward_magnitude",
    "da_phasic_release_multiplier", "dopamine_baseline",
    "tau_ltp", "tau_ltd", "hebbian_ltp_rate",
    "target_excitatory_fraction", "membrane_time_constant",
    "spontaneous_current_magnitude",
    # v1.007 — meta-plasticity, sphere volume transmission, neuromodulator
    # crosstalk, dendritic computation, sensory gating, homeostasis.
    "stdp_window", "tau_meta", "meta_influence_gain",
    "ms_volume_diffusion_rate", "da_ach_crosstalk_strength",
    "proprioceptive_stdp_boost", "sensory_gating_threshold",
    "target_firing_rate", "num_dendritic_branches", "ws_beta",
    # v1.008 — dendritic thresholding, inter-sphere delays, oscillatory/phase
    # binding, neuromod decay, adaptation, structural plasticity, weight
    # ceiling, adaptive-threshold gain, meta-DA boost.
    "branch_threshold", "ms_ff_delay", "ms_fb_delay", "oscillator_strength",
    "phase_coupling_strength", "neuromod_decay_rate", "adaptation_rate",
    "synapse_formation_prob", "synapse_death_prob", "max_weight_magnitude",
    "threshold_mod_k", "meta_da_boost",
    # v1.010 — neuromodulator baselines/reuptake, tonic/phasic timescales,
    # weight homeostasis, proprioceptive lead, oscillator coupling.
    "serotonin_baseline", "acetylcholine_baseline", "norepinephrine_baseline",
    "reuptake_vmax_da", "reuptake_km", "tau_tonic", "tau_phasic",
    "weight_homeostasis_rate", "proprioception_lead_ticks",
    "oscillator_coupling",
    # v1.011 — Multi-Neuraxon MECHANISM genes (previously frozen at config
    # defaults): adaptive threshold, AGMP astrocyte-gated plasticity,
    # ChronoPlastic time-warping, CTSN complemented state, homeostasis,
    # associative neighbour plasticity, tonic/phasic DA receptor split.
    # All verified to be consumed by neuraxon/ (neuron.py / network.py /
    # components.py), so they are live levers, not inert JSON.
    "adaptive_threshold_adjustment", "adaptive_threshold_check_interval",
    "agmp_eta", "agmp_lambda_e", "agmp_lambda_a",
    "chrono_alpha_f", "chrono_alpha_s",
    "ctsn_rho", "ctsn_phi_gain",
    "homeostatic_plasticity_rate", "firing_rate_alpha",
    "associativity_strength",
    "dopamine_low_affinity_threshold", "da_ach_tonic_suppression",
    # v1.012 — deep-tier mechanism genes (active with --space deep): receptor
    # sensitivity, MSTH multi-timescale gains, sensory gating, energy budget,
    # weight-homeostasis tuning, oscillator frequency bands, small-world
    # wiring, phase coupling, LTD scaling, fast/slow tau, dual-trace deltas,
    # neuromod kinetics, proprioceptron reflexes, meta-plasticity. All verified
    # consumed by neuraxon/.
    "receptor_slope_phasic", "receptor_slope_tonic",
    "receptor_concentration_cap",
    "msth_fast_gain", "msth_medium_gain", "msth_ultrafast_ceiling",
    "sensory_gating_suppression",
    "energy_threshold_coupling", "energy_recovery_boost",
    "energy_threshold_floor",
    "weight_homeostasis_target", "weight_saturation_threshold",
    "weight_drift_correction",
    "oscillator_low_freq", "oscillator_mid_freq", "oscillator_high_freq",
    "small_world_k", "small_world_rewire_prob",
    "phase_coupling_local_strength", "phase_clustering_init",
    "phase_coupling_momentum",
    "ltd_inhibitory_scale", "ltd_neutral_scale",
    "tau_fast", "tau_slow",
    "w_fast_delta_share", "w_slow_delta_share",
    "neuromod_release_rate", "neuromod_excess_decay_rate",
    "proprioceptron_clear_path_threshold",
    "proprioceptron_force_turn_threshold", "proprioceptron_rock_memory",
    "meta_accumulation_rate", "meta_target_gain",
    # v1.013 — deepx-tier mechanism genes (active with --space deepx): ITU
    # timescale map, MSTH time constants, oscillator frequency spread,
    # structural plasticity / health, full metabolic economy, circadian rest,
    # non-DA reuptake kinetics, weight-init / drive thresholds. All verified
    # consumed by neuraxon/.
    "itu_niche_strength", "itu_target_cohesion", "itu_freq_mutation_scale",
    "itu_timescale_mutation_scale", "itu_circle_radius",
    "msth_fast_tau", "msth_medium_tau", "msth_slow_tau",
    "natural_freq_range_min", "natural_freq_range_max",
    "diffusion_rate", "neuron_health_decay", "neuron_death_threshold",
    "afferent_synapse_reliability",
    "metabolic_rate", "firing_energy_cost", "plasticity_energy_cost",
    "critical_energy_level", "energy_baseline",
    "circadian_rest_tendency",
    "reuptake_vmax_5ht", "reuptake_vmax_ach", "reuptake_vmax_ne",
    "w_fast_init_max", "w_meta_init_max", "weight_homeostasis_interval",
    "driven_input_threshold", "learning_rate_mod", "meta_clamp_max",
    # v1.014 — dense intrinsic-reward shaping (sparse-reward fix). Consumed in
    # ArcBrain.play_step / _intrinsic_reward.
    "intrinsic_reward_scale", "novelty_reward_scale", "novelty_half_life",
    "structured_change_scale", "intrinsic_reward_decay",
    # v1.015 — potential-based progress shaping (goal-directed reward).
    "progress_reward_scale", "progress_penalty_scale",
    # v1.017 — planning / goal-persistence (long-horizon mechanism).
    "goal_persistence_scale", "goal_persistence_decay",
    "action_commit_scale", "action_commit_steps", "progress_event_threshold",
    # v1.020 — learning DYNAMICS (in-episode acquisition reliability).
    "plasticity_anneal_start", "plasticity_anneal_end", "plasticity_anneal_steps",
    "reward_consolidation_boost", "reward_consolidation_window",
    "reward_eligibility_decay",
    # v1.021 — closed-loop neuromodulatory self-regulation + cross-episode memory.
    "autoreg_strength", "autoreg_reward_tau", "autoreg_explore_gain",
    "autoreg_explore_floor", "autoreg_explore_ceil", "autoreg_plasticity_gain",
    "autoreg_patience",
    # v1.022 — richer perception (scene understanding).
    "retina_size", "retina_saliency_levels", "colour_channels",
    # v1.104 — high-acuity fovea patch (see _fovea_patch)
    "fovea_patch_size", "fovea_patch_colour",
    "efference_copy", "delta_retina_size", "action_mask",   # v1.111
    # v1.118: the click mask, so it reaches params from the arch like every other gene
    "click_mask", "click_mask_min", "click_mask_max",
    "hud_border", "relevel_mask",   # v1.122, v1.123
    "pair_mask",                     # v1.125
    "select_phase", "think_ticks",   # v1.126
    "stuck_switch",                  # v1.135
    "click_colour", "noop_memory",   # v1.140
    "productive_memory", "after_select",   # v1.141
    "dir_memory",                          # v1.142
    "diff_mask",                           # v1.146
    "cand_cursor",                         # v1.147
    "reset_carry",                         # v1.150: soft resets clear the carried novelty/click state
    "agency_sense",                        # v1.152: 8 inputs -- what moved after my action, where the thing I control is
    "cand_ring",                           # v1.154: click candidates include the centres of hollow (8-connected) shapes
    "cand_edge",                           # v1.154: candidates near an edge also offered clamped m cells inward
    "click_free_r",                        # v1.159: a click within R of the snapped candidate keeps the brain's own point
    "steer", "steer_trust", "steer_no_memory", "steer_no_nav", "steer_no_goal", "steer_no_path", "steer_no_effect", "steer_max_clicks", "steer_hud", "steer_brain_w", "steer_v2", "steer_classveto", "steer_no_classveto", "steer_dest", "steer_no_dest", "steer_no_lattice", "steer_no_commit", "steer_teach", "steer_sense", "steer_classveto_on", "steer_v3", "steer_no_model", "steer_no_novelty", "steer_v4", "steer_goal_w", "steer_goal_override", "steer_no_goalmap", "steer_no_clock", "steer_no_wake", "steer_no_reprobe", "steer_no_nodeclick", "steer_v5", "steer_goal_auth", "steer_stall", "steer_goal_oracle", "net_explore", "cap_univ", "cap_open", "cap_explore", "cap_pen", "cap_wire", "cap_wire_w", "cap_decay", "steer_frontier", "frontier_stall", "frontier_probe", "steer_goal_struct", "steer_goal_struct_oracle", "steer_goal_struct_mode", "steer_goal_struct_override", "steer_veto_th", "steer_plan_w", "steer_goal_choose", "steer_goal_how", "steer_goal_when", "steer_goal_stay", "steer_goal_fl", "steer_goal_support",   # v1.160: the steering layer (steer1160.py); v1.164: v5 + brain-driven exploration
    "change_border", "budget_sense", "object_sense", "transient_sense", "type_prior", "death_memory", "worked_prior",   # v1.155
    "cand_tier", "commit_sweep", "commit_stall",   # v1.172: candidate coverage tiers and the per-level key sweep (wall1172.py)
    "effect_mem", "warm_prefix", "game_sense", "home_game",   # v1.173: effect memory, the warm prefix, the game sense (effect1173.py)
    "goal_strict", "home_games",           # v1.174: fewer, surer goal cells; the measured home games of the sense (effect1174.py)
    "goal_what",                           # v1.175: the goal map's cells carry their target colour (effect1175.py)
    "effect_ctx", "ctx_min_n", "ctx_agree", "goal_recall", "goal_prior",   # v1.176: effects in context, recall, the colour prior (effect1176.py)
    "eyes77", "eyes77_nin0", "eyes77_b0", "anim77", "death77", "undo77", "guard77", "transfer77", "macro77",   # v1.177: the milestone-2 capabilities (eyes1177.py, steer1177.py)
    "bus78", "bus78_nin0", "bus78_gain", "fatal78", "plan78", "prior78", "l1probe78",   # v1.178: the sensory bus, the fatal penalty apart, the planner, the effect prior (bus1178 / mech1178 / plan1178 / prior1178)
    "eval_explore", "effect_fovea",        # v1.149: the network decides / the effect fovea
    "settled_view",                        # v1.149: read frame[-1] (the settled board)
    "change_track",   # v1.119
    "lattice_px",                                          # v1.112 — the cursor's step size
    "perception_cursor_local",
    # v1.023 — level-conditioned malleable brain (per-level reconfiguration).
    "level_context_scale", "level_gate_scale", "level_plasticity_bias",
    "max_level_context",
    # v1.027 — experience replay & consolidation of solved-level solutions.
    "replay_enabled", "replay_capacity", "replay_batch", "replay_interval",
    "replay_plasticity", "replay_trace_len", "dream_cycles",
    # v1.090 — the LEARNED PRE-CLEAR value function. RETIRED in 1.091 (proven inert: they
    # climbed the engine's score/progress signal, which 1.090 showed is 0 on every trial for
    # the unreached games). Kept in the key list ONLY for backward-compatible genome loading;
    # 1.091 removes them from the SEARCH SPACE so they are no longer explored. 0 == off.
    "preclear_value_scale", "score_delta_value", "controllability_value",
    "reachable_novelty_value", "preclear_anneal_steps", "partial_progress_seed_weight",
    # v1.091 — the LEARNED WORLD-MODEL. 1.090 proved the ENGINE gives no pre-clear gradient
    # (level_progress was 0 for all 6576 trials), so the brain must judge progress against
    # its OWN model. A small online forward model (predict next grid-features from an action)
    # yields two gradients available on step 1 of an unseen game: CURIOSITY (learning-progress
    # = prediction error that decreases when revisited -> drives toward the not-yet-understood
    # rule) and EMPOWERMENT (diversity of reliably-reachable next states). Annealed out once a
    # real success prototype seeds. 0 on all == 1.090 behaviour (world model off).
    "world_model_scale", "curiosity_value", "empowerment_value",
    "wm_learning_rate", "wm_horizon", "wm_anneal_steps",
    "goal_gradient_scale", "goal_gradient_trace", "goal_gradient_memory",
    "goal_gradient_decay",
    "structured_explore_scale", "structured_explore_buckets",
    "goal_conditioned_explore",
    "action_memory_scale", "action_memory_decay", "state_coarsen",
    "action_commit_len",
    "frontier_focus", "frontier_mastery", "frontier_bonus_frac",
    "warm_restart", "warm_stall_steps",
    # v1.016 — sustained-activity / memory-horizon genes.
    "chrono_trace_clip", "w_slow_post_trace_fraction", "membrane_negative_bias",
]
_OPRANGE_KEYS = [
    "learning_rate", "plasticity_threshold", "adaptation_tau_ticks",
    "adaptation_target_excitatory_multiplier",
    "adaptation_target_inhibitory_multiplier", "autoreceptor_coefficient",
    "sensory_boost_scale",
    # v1.005 — autoreceptor (presynaptic feedback) dynamics
    "autoreceptor_tau_ticks", "autoreceptor_rate_coeff",
]
# special-cased: num_hidden_neurons_default -> num_hidden_neurons


def build_params_from_arch(arch_path: Optional[str] = None,
                           verbose: bool = False) -> NetworkParameters:
    """Build a NetworkParameters from an architecture JSON (or code defaults)."""
    import architecture as arch_mod
    arch_mod.load_architecture(arch_path, verbose=verbose)
    get = arch_mod.get_param

    p = NetworkParameters(network_name="MultiNxonArcAGI3 Brain")

    n_in = get("neural", "num_input_neurons", getattr(p, "num_input_neurons", 12))
    n_out = get("neural", "num_output_neurons", getattr(p, "num_output_neurons", 7))
    n_hid = get("neural", "num_hidden_neurons_default",
                getattr(p, "num_hidden_neurons", 16))
    p.num_input_neurons = int(n_in)
    p.num_output_neurons = int(n_out)
    p.num_hidden_neurons = int(n_hid)

    for k in _NEURAL_KEYS:
        if k in ("num_input_neurons", "num_output_neurons",
                 "num_hidden_neurons_default"):
            continue
        v = get("neural", k, None)
        if v is not None and hasattr(p, k):
            setattr(p, k, v)
        elif v is not None and (k.startswith("evo_")
                                or k in ("planning_scale", "plan_margin_thresh", "plan_depth",
                                         # v1.105: the fovea genes were not pre-declared on the
                                         # params object, so hasattr(p, k) was False and setattr
                                         # was SKIPPED. v1104 raised num_input_neurons to fit a
                                         # K x K patch (82 -> 163 -> 251) while _fovea_patch()
                                         # returned [] -- every "fovea" cell ran the BASELINE
                                         # encoder with K^2 channels of PADDING. v1104 therefore
                                         # measured dead input dimensions, not acuity.
                                         "fovea_patch_size", "fovea_patch_colour",
                                         # v1.111 — same trap as the fovea: these are not
                                         # pre-declared on the params object, so hasattr() is
                                         # False and setattr would be SILENTLY SKIPPED. That
                                         # cost v1104 an entire run.
                                         "efference_copy", "delta_retina_size",
                                         "action_mask",
                                         # v1.112 — the lattice, on the motor side
                                         "lattice_px",
                                         # v1.149 -- same trap; these must never be skipped
                                         "eval_explore", "effect_fovea", "settled_view", "reset_carry", "agency_sense", "cand_ring", "click_free_r", "cand_edge", "steer", "steer_trust", "steer_no_memory", "steer_no_nav", "steer_no_goal", "steer_no_path", "steer_no_effect", "steer_max_clicks", "steer_hud", "steer_brain_w", "steer_v2", "steer_classveto", "steer_no_classveto", "steer_dest", "steer_no_dest", "steer_no_lattice", "steer_no_commit", "steer_teach", "steer_sense", "steer_classveto_on", "steer_v3", "steer_no_model", "steer_no_novelty", "steer_v4", "steer_goal_w", "steer_goal_override", "steer_no_goalmap", "steer_no_clock", "steer_no_wake", "steer_no_reprobe", "steer_no_nodeclick", "steer_v5", "steer_goal_auth", "steer_stall", "steer_goal_oracle", "net_explore", "cap_univ", "cap_open", "cap_explore", "cap_pen", "cap_wire", "cap_wire_w", "cap_decay", "steer_frontier", "frontier_stall", "frontier_probe", "steer_goal_struct", "steer_goal_struct_oracle", "steer_goal_struct_mode", "steer_goal_struct_override", "steer_veto_th", "steer_plan_w", "steer_goal_choose", "steer_goal_how", "steer_goal_when", "steer_goal_stay", "steer_goal_fl", "steer_goal_support",
                                         "change_border", "budget_sense", "object_sense", "transient_sense", "type_prior", "death_memory", "worked_prior",
                                         "cand_tier", "commit_sweep", "commit_stall",
                                         "effect_mem", "warm_prefix", "game_sense", "home_game",
                                         "goal_strict", "home_games", "goal_what",
                                         "effect_ctx", "ctx_min_n", "ctx_agree", "goal_recall", "goal_prior",
                                         "eyes77", "eyes77_nin0", "eyes77_b0", "anim77", "death77", "undo77", "guard77", "transfer77", "macro77",
                                         "bus78", "bus78_nin0", "bus78_gain", "fatal78", "plan78", "prior78", "l1probe78")):
            # v1.062: evolved-architecture genes aren't pre-declared on the params object.
            # v1.078: planning_scale/plan_margin_thresh likewise are new genes read via
            # getattr with defaults — set them unconditionally so planning actually engages.
            setattr(p, k, v)
    # ============ v1.112 P0-B: the _meta.genes fallback ============
    # A gene that cannot survive save -> load is not a gene.  Anything the NAS stamped into
    # _meta.genes is applied here unconditionally, so a reloaded brain runs the encoder it
    # TRAINED with even when the gene is absent from _NEURAL_KEYS or from the params schema.
    # This is deliberately a blanket setattr: the failure it prevents (a silently different
    # encoder) is far worse than an unused attribute.
    _mg = get("_meta", "genes", None) or get("neural", "_v1112_genes", None)
    if isinstance(_mg, dict):
        for k, v in _mg.items():
            if v is not None:
                setattr(p, k, v)
    for k in _OPRANGE_KEYS:
        v = get("operating_ranges", k, None)
        if v is not None and hasattr(p, k):
            setattr(p, k, v)

    # ensure multi-sphere is on so build_brain produces the sphere graph
    if hasattr(p, "multisphere_enabled"):
        p.multisphere_enabled = True
    return p


def _v55_interior_equal(a, b, m):
    """v1.155: are two grids equal INSIDE a border of m cells (the action-budget bar lives in the border)"""
    if a is None or b is None:
        return a is b
    if len(a) != len(b) or (a and b and len(a[0]) != len(b[0])):
        return False
    H = len(a)
    if H == 0:
        return True
    W = len(a[0])
    for y in range(m, H - m):
        if list(a[y][m:W - m]) != list(b[y][m:W - m]):
            return False
    return True


def _v55_interior_count(a, b, m):
    if not a or not b or len(a) != len(b) or len(a[0]) != len(b[0]):
        return 0
    H = len(a); W = len(a[0])
    return sum(1 for y in range(m, H - m) for x in range(m, W - m) if a[y][x] != b[y][x])


def _v55_interior_sig(grid, m):
    if not grid:
        return 0
    if m <= 0:
        return _frame_signature(grid)
    H = len(grid); W = len(grid[0])
    return _frame_signature([list(r[m:W - m]) for r in grid[m:H - m]])


def _settled_frame(frame):
    """v1.149: the same frame with only its LAST grid (the settled board). A copy -- the caller's
    frame is untouched; a single-grid or non-dataclass frame is returned as is."""
    try:
        fr = getattr(frame, "frame", None)
        if isinstance(fr, list) and len(fr) > 1:
            import dataclasses as _dc
            if _dc.is_dataclass(frame):
                return _dc.replace(frame, frame=[fr[-1]])
    except Exception:
        pass
    return frame


def _trinarize(value: float, thresh: float = 0.0) -> int:
    if value > thresh:
        return 1
    if value < -thresh:
        return -1
    return 0


def _sign(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# --------------------------------------------------------------------------- #
#  The ARC brain
# --------------------------------------------------------------------------- #
class ArcBrain:
    """A Neuraxon Multi-Sphere brain that perceives ARC grids and acts."""

    def __init__(self, params: NetworkParameters, seed: int = 0,
                 micro_steps: Optional[int] = None,
                 explore_probability: float = 0.5,
                 idle_explore_steps: int = 6,
                 prebuilt_brain=None):
        self.params = params
        self.rng = random.Random(seed)
        # v1.054: if a pre-LEARNED brain is injected (from a snapshot), use it
        # instead of growing a newborn — so BestTest --learned can play warmed-up
        # with the synaptic state the brain learned during the NAS. Plasticity
        # stays on (the architecture always learns), so it still adapts.
        self.brain = prebuilt_brain if prebuilt_brain is not None \
            else build_brain(params)

        self.sensory = self.brain.spheres.get("sensory")
        self.motor = self.brain.spheres.get("motor")
        if self.sensory is None or self.motor is None:
            raise RuntimeError("Brain is missing a 'sensory' or 'motor' sphere")
        self.sensory_ids = list(self.sensory.interface.sensory_input_ids)
        self.motor_sensory_ids = list(self.motor.interface.sensory_input_ids)
        self.n_in = len(self.sensory_ids)

        # v1.022 richer perception config. The grid is encoded as: the legacy
        # fovea/proprioception channels (still useful) + an R×R spatial retina
        # over the whole grid + per-colour presence + colour-at-cursor, so the
        # brain can parse multi-colour scene structure (the level-2 bottleneck).
        # ---- v1.104: HIGH-ACUITY FOVEA PATCH ----
        # The global retina quantises a K-cell block to {-1,0,+1} on its non-background
        # FRACTION, so a 2-pixel token in a 144-cell block (R=5, the population median)
        # scores 0.014 and is encoded IDENTICALLY TO EMPTY SPACE. m0r0 level 1 is "move two
        # cyan 1x1 squares upward" -- 30 human actions -- and the fleet median is 588,
        # because the brain cannot see the squares. ls20 requires matching a ~2-4 pixel
        # status glyph to a piece, and the retina also discards colour position entirely.
        # Raising R fixes the resolution but costs 17x per step at R=24 (measured), which
        # is unaffordable. A K x K patch at FULL resolution around the cursor costs K^2
        # channels instead of R^2 and gives exact local vision -- coarse periphery plus an
        # acute fovea. 0 disables it, so a brain without the gene behaves exactly as before.
        self._fovea_k = max(0, int(getattr(params, "fovea_patch_size", 0)))
        self._fovea_patch_colour = bool(int(getattr(params, "fovea_patch_colour", 1)))
        self._retina_size = max(2, int(getattr(params, "retina_size", 5)))
        self._retina_levels = int(getattr(params, "retina_saliency_levels", 3))
        self._colour_channels = max(0, int(getattr(params, "colour_channels", 8)))

        # v1.023 level-conditioned malleable brain. The agent perceives which
        # level it is on and that context gates different sub-circuits per level
        # (the "spheres per level" idea), so one brain can specialise per level
        # instead of forcing a single weight set to master all of them.
        self._level_context_scale = float(getattr(params, "level_context_scale", 0.0))
        self._level_gate_scale = float(getattr(params, "level_gate_scale", 0.0))
        self._level_plasticity_bias = float(getattr(params, "level_plasticity_bias", 0.0))
        self._max_level_ctx = max(1, int(getattr(params, "max_level_context", 4)))
        self._cur_level: int = 0      # current level index (from levels_completed)
        self.n_out = int(params.num_output_neurons)

        # micro-steps per macro game step (kept small for speed)
        if micro_steps is None:
            micro_steps = max(2, min(5, int(getattr(params, "simulation_steps", 30)) // 8))
        self.micro_steps = int(micro_steps)
        # v1.126: think longer per action, but only where the fleet is behind human.
        # There is no wall-clock limit per action, so this costs trials-per-day and nothing
        # in the score. r11l is at 0.55x human and gets think_ticks=0; sp80 (3.74x),
        # cd82 (6.95x) and lp85 (14.2x) are the ones that can earn it.
        _tt = int(getattr(params, 'think_ticks', 0) or 0)
        if _tt > 1:
            self.micro_steps = int(self.micro_steps * _tt)

        self.explore_probability = float(explore_probability)
        self.idle_explore_steps = int(idle_explore_steps)

        # perception / action state
        self.cursor = [0, 0]          # [row, col] in grid-cell space
        self.prev_grid: List[List[int]] = []
        self.prev_score: int = 0
        self.idle_steps: int = 0
        self.last_action: Optional[GameAction] = None
        # ---- v1.111 efference copy + delta retina (both default OFF) ----
        self._eff_copy = int(getattr(params, "efference_copy", 0) or 0) > 0
        self._delta_cells = max(0, int(getattr(params, "delta_retina_size", 0) or 0))
        # ---- v1.111 ACTION MASK ----
        # The transcripts of a 100.00-RHAE campaign show most games use only a SUBSET of the
        # seven buttons, and the subset is CONSTANT for the whole game:
        #   r11l "Only ACTION6 (click x,y)"        s5i5 "Only ACTION6"
        #   su15 "Only ACTION6 and ACTION7 (undo)" tn36 "Only ACTION6 (click)"
        #   bp35 "Action set: constant [3,4,6,7] for all 365 events"
        #   cd82 "ACTION1/ACTION2 changed nothing on the board; only ticked the counter"
        # An action outside the live set is a SILENT NO-OP THAT STILL COSTS AN ACTION (13 of
        # 320 verified claims). A brain that spends a third of its budget on dead buttons can
        # never reach human efficiency, and it has no way to learn the set from a one-bit
        # frame_changed signal. The mask is a per-game prior, exactly like the human baseline.
        # "1234567" = all live (default, identical to v1.110 behaviour).
        _am = str(getattr(params, "action_mask", "") or "").strip()
        self._action_mask = set(int(c) for c in _am if c.isdigit()) if _am else None
        # ============ v1.112 — THE LATTICE, ON THE SIDE THAT MOVES ============
        # 28 of 320 verified mechanics give the board's block size (2,3,4,5,6 px) and 31 say
        # the arrows move EXACTLY ONE LATTICE STEP.  v1111 encoded that as retina_size = 64/px,
        # which makes n_in scale as R^2 (R=8 -> 142 inputs, R=32 -> 1102) and therefore turned
        # the lattice hypothesis into an input-width sweep -- running AGAINST v1109's one clean
        # result, which is that width is monotone and points small (48->89, 96->101, 192->117,
        # 384->130 median actions).  Every R>=21 machine in v1111 cleared nothing.
        #
        # Meanwhile the attention cursor moved ONE PIXEL per step on a board whose logical cell
        # is 4px, and _cursor_click mapped that pixel position straight to a 64x64 click.  So
        # the brain needed 4 steps to traverse one cell that the GAME moves in one press, and
        # clicked at pixel precision on a board that only distinguishes 16x16 cells.
        #
        # lattice_px puts the lattice where the mined claims actually put it -- in the STEP --
        # at zero input-width cost.  1 = v1.111 behaviour, bit-identical.
        self._lattice_px = max(1, int(getattr(params, "lattice_px", 1) or 1))
        # ============ v1.118: the click mask, from the GENE ============
        # The env var still wins when set, so an experiment can override a brain; but a brain
        # that trained with the mask now carries it, and every offline tool gets it for free.
        # ============ v1.120: the change-tracker gene ============
        # v1119 declared this gene, serialised it into 279 of 279 brains, and NEVER READ IT --
        # _change_features was defined and never called. Four channels when on; the encoder
        # width is unchanged when off, so change_track=0 is bit-identical to v1119.
        self._change_track = int(getattr(params, 'change_track', 0) or 0)
        # ============ v1.122: THE HUD BAND ============
        # Width of the frame border excluded from change detection. Four games draw an action
        # counter there -- r11l's transcript: "The click was a pure no-op (only the counter
        # ticked)." Measured over 120 random actions, "any cell changed" vs "any cell >= 2 from
        # the edge changed": vc33 100.0% -> 0.0%, cd82 79.0% -> 33.0%, r11l 100.0% -> 65.0%.
        # 0 is v1.121 behaviour, bit-identical.
        self._hud_border = max(0, int(getattr(params, 'hud_border', 0) or 0))
        # ============ v1.123: re-derive the mask bound at each level transition ============
        # The transcripts read level by level show the board changes SHAPE, not just difficulty:
        # lp85's tiles go from 4x4 at level 1 to 2x2 at level 6 and its arrow controls move to
        # a different row at level 3. A bound tuned on level 1 is wrong later by construction,
        # which is why every tight bound died past level 1 and why r11l -- whose nodes are 1-2
        # cells at EVERY level -- is the one game that holds.
        self._relevel_mask = int(getattr(params, 'relevel_mask', 0) or 0)
        # v1.125: max components a colour may occupy and still count as a core/socket colour.
        # See config.py for the per-game counts this was derived from.
        self._pair_mask = int(getattr(params, 'pair_mask', 0) or 0)
        # v1.126: the SELECT -> MOVE cycle. sp80 clicks to take control then D-pads;
        # cd82 spacebars to pick up then D-pads then spacebars to stamp. Nothing in the
        # encoder has ever told the brain it is HOLDING something.
        self._select_phase = int(getattr(params, 'select_phase', 0) or 0)
        # v1.135: switch candidate after N no-change actions (0 = off)
        self._stuck_switch = int(getattr(params, 'stuck_switch', 0) or 0)
        self._click_colour = int(getattr(params, 'click_colour', -1) if getattr(params, 'click_colour', None) is not None else -1)
        self._noop_memory = int(getattr(params, 'noop_memory', 0) or 0)
        self._noop_until = {}; self._click_t = 0; self._click_log = []; self._noop_floor = None
        self._productive_memory = int(getattr(params, 'productive_memory', 0) or 0)
        self._after_select = int(getattr(params, 'after_select', 0) or 0)
        self._prod_score = {}; self._prod_until = {}; self._after_select_n = 0; self._action_log = []; self._last_action_id = None
        self._dir_memory = int(getattr(params, 'dir_memory', 0) or 0); self._dir_noop_until = {}; self._step_t = 0
        self._diff_mask = int(getattr(params, 'diff_mask', 0) or 0)     # v1.146: the target-difference channel
        self._cand_cursor = int(getattr(params, 'cand_cursor', 0) or 0); self._cand_idx = 0   # v1.147: the network chooses the target
        self._stuck_n = 0; self._last_click_cand = None; self._stuck_switched = 0
        self._sp_since = 0          # actions since the last select-type action
        self._sp_worked = 0.0       # did that select change the board?
        self._sp_last_sel = 0.0     # was the last action itself a select?
        self._rl_level = 0          # last level seen, so a transition can be detected
        self._rl_hi = None          # the bound re-derived for the current level
        _MODES = {0: 'off', 1: 'centroid', 2: 'snap', 3: 'auto'}
        _g = int(getattr(params, "click_mask", 0) or 0)
        self._click_mask = os.environ.get("MULTINXON_CLICK_MASK") or _MODES.get(_g, 'off')
        self._cm_lo = int(os.environ.get("MULTINXON_CLICK_MASK_MIN",
                                         getattr(params, "click_mask_min", 1)) or 1)
        self._cm_hi = int(os.environ.get("MULTINXON_CLICK_MASK_MAX",
                                         getattr(params, "click_mask_max", 8)) or 8)
        # the band every WINNING v1117 game landed in; the two disasters sat at 1 and 180
        self._cm_band = (int(os.environ.get("MULTINXON_CLICK_MASK_LO", "2") or 2),
                         int(os.environ.get("MULTINXON_CLICK_MASK_HI", "30") or 30))
        self._eff_prev_idx = -1        # index of the action taken at the previous step
        self._eff_prev_click = (-1.0, -1.0)
        # v1.076 ACTION-GENES: per-brain action histogram + ordered sequence.
        self._action_hist: Dict[str, int] = {}
        self._action_seq: List[str] = []
        self._action_seq_cap: int = 4000   # cap the stored trace length
        self._avail_actions: set = set()   # v1.077: available actions seen this episode
        self._recent_sigs: List[int] = []     # novelty memory (frame signatures)
        self._sig_cap = 24

        # v1.014 dense intrinsic-reward shaping (sparse-reward fix). Pull the
        # searchable gains off the params object so the NAS can tune them; when
        # intrinsic_reward_scale==0 this is a no-op and behaviour is identical
        # to v1.013. Novelty is count-based: each grid state's bonus halves
        # every `novelty_half_life` revisits, so the brain is rewarded for
        # reaching genuinely NEW configurations rather than oscillating.
        self._ir_scale = float(getattr(params, "intrinsic_reward_scale", 0.0))
        # v1.078 REAL PLANNING params (searchable genes; 0 disables — the dead genes were
        # removed, these are new and actually consumed by _plan_action/act).
        self._planning_scale = float(getattr(params, "planning_scale", 0.0))
        self._plan_margin_thresh = float(getattr(params, "plan_margin_thresh", 0.15))
        self._plan_depth = int(getattr(params, "plan_depth", 1))   # v1.079: 2-step for L2->L3
        self._plan_fired = 0
        self._plan_steps = 0
        self._plan_total = 0
        self._ir_novelty = float(getattr(params, "novelty_reward_scale", 0.5))
        self._ir_halflife = max(0.5, float(getattr(params, "novelty_half_life", 6.0)))
        self._ir_structured = float(getattr(params, "structured_change_scale", 0.3))
        self._ir_decay = float(getattr(params, "intrinsic_reward_decay", 1.0))
        self._visit_counts: Dict[int, int] = {}   # grid-signature -> times seen
        self._play_steps: int = 0                  # for the intrinsic anneal
        # v1.015 potential-based progress shaping. When the environment exposes
        # within-level progress (offline now; future real engines), reward the
        # *increase* in progress Δφ — the policy-invariant shaping form (Ng et
        # al. 1999). This makes dopamine fire as the brain closes in on
        # completing a level, the dense goal-directed signal that pure novelty
        # (v1.014) could not provide. Gene-controlled; 0 = off.
        self._progress_reward_scale = float(
            getattr(params, "progress_reward_scale", 0.0))
        self._progress_penalty_scale = float(
            getattr(params, "progress_penalty_scale", 0.0))
        self._prev_progress: float = 0.0

        # v1.017 planning / goal-persistence (long-horizon mechanism). A working-
        # memory-for-goals: when an action produces "progress" (a grid change
        # beyond threshold, or a score gain) the brain (a) records the motor
        # direction that worked into a slowly-decaying goal latent that biases
        # the cursor next steps, and (b) can COMMIT to repeating that action for
        # a few steps (an option) instead of re-deciding every tick. All gene-
        # controlled; 0 = off (legacy reactive behaviour).
        self._goal_persist_scale = float(getattr(params, "goal_persistence_scale", 0.0))
        self._goal_persist_decay = min(0.995, max(0.0, float(
            getattr(params, "goal_persistence_decay", 0.9))))
        self._commit_scale = float(getattr(params, "action_commit_scale", 0.0))
        self._commit_steps = int(getattr(params, "action_commit_steps", 0))
        self._progress_event_thr = float(getattr(params, "progress_event_threshold", 0.0))
        # runtime planning state
        self._goal_dy: float = 0.0      # goal latent: working vertical direction
        self._goal_dx: float = 0.0      # goal latent: working horizontal direction
        self._commit_action = None      # currently committed option (or None)
        self._commit_remaining: int = 0
        self._last_dir = (0, 0)         # last cursor delta (for crediting)

        # v1.020 learning-dynamics state (in-episode acquisition reliability).
        self._anneal_start = float(getattr(params, "plasticity_anneal_start", 1.0))
        self._anneal_end = float(getattr(params, "plasticity_anneal_end", 1.0))
        self._anneal_steps = max(1, int(getattr(params, "plasticity_anneal_steps", 5000)))
        self._consol_boost = float(getattr(params, "reward_consolidation_boost", 1.0))
        self._consol_window = int(getattr(params, "reward_consolidation_window", 0))
        self._elig_decay = min(0.99, max(0.0, float(
            getattr(params, "reward_eligibility_decay", 0.0))))
        self._da_trace: float = 0.0        # eligibility trace of dopamine
        self._consol_remaining: int = 0    # steps left in a consolidation window

        # v1.027 experience replay & consolidation of solved-level solutions.
        # When a level is completed, the recent (sensory-input → action) sequence
        # is stored in an episodic buffer that persists across the brain's whole
        # life; the brain periodically replays those winning sequences with
        # elevated plasticity so a one-off crossing consolidates into a
        # repeatable skill. This directly targets the wall where brains touch
        # level 2 but cannot make it stick.
        self._replay_enabled = float(getattr(params, "replay_enabled", 0.0)) > 0.0
        self._replay_capacity = max(1, int(getattr(params, "replay_capacity", 64)))
        self._replay_batch = max(1, int(getattr(params, "replay_batch", 8)))
        self._replay_interval = int(getattr(params, "replay_interval", 0))
        self._replay_plasticity = float(getattr(params, "replay_plasticity", 1.5))
        self._replay_trace_len = max(1, int(getattr(params, "replay_trace_len", 12)))
        self._dream_cycles = float(getattr(params, "dream_cycles", 3.0))

        # v1.032 learned success-proximity (self-supervised goal gradient). The
        # brain stores compact feature vectors of grids it occupied just before
        # completing a level, averages them into a life-persistent "success
        # prototype", and rewards itself for moving toward that prototype — a
        # learned within-level gradient where the engine gives none.
        self._gg_scale = float(getattr(params, "goal_gradient_scale", 0.0))
        self._gg_trace_len = max(1, int(getattr(params, "goal_gradient_trace", 8)))
        self._gg_memory = max(1, int(getattr(params, "goal_gradient_memory", 32)))
        self._gg_decay = min(1.0, max(0.0, float(getattr(params, "goal_gradient_decay", 0.2))))
        # v1.033: PER-LEVEL success prototypes. The single shared prototype of
        # v1.032 mastered the level-1 crossing (floor rose to 0.68, 96% ever-L1)
        # but could not also encode the level-2 target, so L2 stayed a lottery.
        # Now each level keeps its OWN prototype: when on level N the brain climbs
        # toward "what solving level N looks like", learned distinctly. Keyed by
        # the levels_completed value the brain is currently trying to advance past.
        self._gg_prototypes = {}           # level -> success-feature prototype
        self._gg_proto_n = {}              # level -> count folded in
        self._gg_recent_feats = []         # rolling recent grid features
        self._gg_prev_dist = None          # previous distance-to-current-proto
        self._gg_levels = 0                # last seen levels_completed

        # v1.034 structured (count-based) exploration. When exploring, prefer the
        # action least-tried in the current state-bucket, so the brain
        # systematically covers the action space and can re-find a solution
        # instead of re-sampling randomly. Visit-counts persist across the life.
        self._se_scale = float(getattr(params, "structured_explore_scale", 0.0))
        self._se_buckets = max(2, int(getattr(params, "structured_explore_buckets", 8)))
        self._se_counts = {}               # (state_bucket, action_idx) -> count
        # v1.037 goal-conditioned exploration: bias exploration toward actions
        # that move the grid toward the learned per-level success prototype.
        self._gce = float(getattr(params, "goal_conditioned_explore", 0.0))

        # ============ v1.090 — LEARNED PRE-CLEAR VALUE FUNCTION ============
        # 1089 proved planning depth transfers (d1/d2/d3 -> 62/76/95% L1 transfer) but
        # that transfer NEVER reached an unsolved game: the planner's value function
        # content() is a hand-coded "non-background = interesting" prior, and the one
        # learned signal (the success prototype) only seeds after a completion. So on
        # the 10 unreached games the learned goal gradient is 0.0 forever.
        #
        # This adds a value gradient the planner can use BEFORE any clear, learned from
        # signals present every step: recent score movement, whether an action changes
        # the world (controllability), and count-based novelty of reachable states. It
        # ANNEALS OUT once a real success prototype exists, so it is a bridge across the
        # chicken-and-egg, not a replacement for the mechanism that already works on
        # solvable games. All-zero genes reproduce 1.089 exactly.
        self._pcv_scale = float(getattr(params, "preclear_value_scale", 0.0))
        self._pcv_score = float(getattr(params, "score_delta_value", 0.0))
        self._pcv_ctrl = float(getattr(params, "controllability_value", 0.0))
        self._pcv_novel = float(getattr(params, "reachable_novelty_value", 0.0))
        self._pcv_anneal = max(1, int(getattr(params, "preclear_anneal_steps", 1000)))
        self._partial_seed_w = float(getattr(params, "partial_progress_seed_weight", 0.0))
        # running pre-clear state (per life)
        self._pcv_ever_cleared = False     # has ANY level been cleared this life?
        self._pcv_steps_since_clear = 0    # for the anneal
        self._pcv_score_trace = 0.0        # decaying trace of recent score movement
        self._pcv_ctrl_by_action = {}      # action_idx -> P(this action changes world)
        self._pcv_ctrl_tries = {}          # action_idx -> attempts
        self._pcv_reach_counts = {}        # coarse next-state signature -> visits
        self._pcv_last_action_idx = None

        # ============ v1.091 — THE LEARNED WORLD-MODEL ============
        # 1.090 proved the ENGINE exposes no pre-clear gradient (level_progress == 0 for every
        # trial, even on solved games). So the pre-clear value above reduced to novelty+
        # controllability = better exploration, which 1.089 already showed is not the missing
        # piece. 1.091's answer: the brain builds its OWN model of the game and judges progress
        # against it. A small ONLINE forward model predicts next grid-features from (features,
        # action); two gradients fall out of it, both alive on step 1 of an unseen game:
        #   * CURIOSITY (learning-progress): value actions whose outcome the model predicts
        #     POORLY but LEARNABLY (error that shrinks on revisit) -> drives toward the part of
        #     the game not yet understood, i.e. where the undiscovered rule lives.
        #   * EMPOWERMENT: value states from which many DISTINCT next-states are reachable per
        #     the model's own predictions -> a reward-free driver of goal-discovery.
        # Annealed out once a real success prototype seeds (same bridge logic as 1.090, but the
        # gradient is now self-generated, not engine-provided). 0 genes == 1.090 behaviour.
        self._wm_scale = float(getattr(params, "world_model_scale", 0.0))
        self._wm_curio = float(getattr(params, "curiosity_value", 0.0))
        self._wm_emp = float(getattr(params, "empowerment_value", 0.0))
        self._wm_lr = min(0.9, max(0.001, float(getattr(params, "wm_learning_rate", 0.1))))
        self._wm_horizon = max(1, int(getattr(params, "wm_horizon", 2)))
        self._wm_anneal = max(1, int(getattr(params, "wm_anneal_steps", 1000)))
        # the forward model itself: a low-rank/linear predictor over a small fixed feature
        # vector (kept tiny on purpose -- this is a spiking substrate, the goal is a USABLE
        # self-generated gradient, not a perfect model). Lazily sized on first observation.
        self._wm_W = None                  # weight matrix action -> feature-delta prediction
        self._wm_feat_dim = 0
        self._wm_err_by_action = {}        # action -> recent mean prediction error
        self._wm_err_prev = {}             # action -> previous error (for learning-progress)
        self._wm_progress_by_action = {}   # action -> learning-progress (error decrease)
        self._wm_reach_by_state = {}       # coarse state -> set of distinct predicted next-sigs
        self._wm_prev_feat = None          # feature vector at previous step
        self._wm_prev_action_idx = None
        self._wm_ever_cleared = False
        self._wm_steps_since_clear = 0
        self._wm_best_progress = 0.0       # peak learning-progress seen (telemetry / seeding)

        # v1.041 ACTION-SEQUENCE MEMORY — attack non-repeatability at the
        # state→action map. An explicit, life-persistent dict binding a coarse
        # state signature to the action that last produced PROGRESS from it, so a
        # crossing found once is recorded and replayed rather than rediscovered.
        self._am_scale = float(getattr(params, "action_memory_scale", 0.0))
        self._am_decay = min(1.0, max(0.0, float(getattr(params, "action_memory_decay", 0.05))))
        self._state_coarsen = max(1, int(getattr(params, "state_coarsen", 4)))
        self._commit_len = max(0, int(getattr(params, "action_commit_len", 0)))
        self._action_mem = {}          # coarse_state -> {'action':int,'data':dict,'w':float}
        self._am_prev_state = None     # coarse state at previous step
        self._am_prev_action = None    # action taken at previous step
        self._am_prev_data = None      # its click data
        self._am_prev_levels = 0       # levels_completed at previous step
        self._am_commit_action = None  # action currently being committed to
        self._am_commit_left = 0       # remaining committed micro-steps

        # v1.042 frontier flag: when the lifelong scheduler has granted bonus
        # episodes because L1 is mastered, it sets this so the brain concentrates
        # exploration on the unmastered L1→L2 boundary (less random re-clearing of
        # L1, stronger goal-gradient/action-memory pull toward the frontier level).
        self._at_frontier = False
        self._last_mech = 'brain_act'   # v1.044 region telemetry: pathway tag
        self._success_memory: List[Dict[int, float]] = []   # stored winning inputs
        self._recent_inputs: List[Dict[int, float]] = []     # rolling recent inputs
        self._steps_since_replay: int = 0
        self._prev_levels: int = 0

        # v1.021 closed-loop neuromodulatory self-regulation. The brain senses
        # its own recent reward rate and re-regulates exploration + plasticity in
        # real time (noradrenergic/cholinergic-style), instead of using fixed
        # set-points. autoreg_strength==0 disables the loop.
        self._autoreg = float(getattr(params, "autoreg_strength", 0.0))
        self._autoreg_tau = max(10.0, float(getattr(params, "autoreg_reward_tau", 200.0)))
        self._autoreg_explore_gain = float(getattr(params, "autoreg_explore_gain", 0.0))
        self._autoreg_explore_floor = float(getattr(params, "autoreg_explore_floor", 0.05))
        self._autoreg_explore_ceil = float(getattr(params, "autoreg_explore_ceil", 0.9))
        self._autoreg_plasticity_gain = float(getattr(params, "autoreg_plasticity_gain", 0.0))
        self._autoreg_patience = max(1, int(getattr(params, "autoreg_patience", 150)))
        self._base_explore_prob = float(getattr(params, "explore_probability", 0.5))
        # ============ v1.149: THE NETWORK DECIDES ============
        # eval_explore >= 0 pins explore_probability (autoregulation included) AND replaces every
        # random fallback of the decode: when the motor outputs select no available action, the
        # network's own SELECTION is clicked (cand_cursor) or its last action repeated -- no random
        # number is drawn at eval_explore = 0. Hand overrides of the network's choice (planning,
        # action memory, dir_memory rotation, after_select, the noop/productive/stuck click pools)
        # are off in this mode. -1 (the default) = the 1148 behaviour, bit for bit.
        try:
            _ee = getattr(params, "eval_explore", None)
            self._eval_explore = float(_ee) if _ee is not None else -1.0
        except Exception:
            self._eval_explore = -1.0
        self._net_decides = self._eval_explore >= 0.0
        if self._net_decides:
            self.explore_probability = self._eval_explore
            self._base_explore_prob = self._eval_explore
        self._net_last_action = None
        self._net_default_n = 0
        # v1.149: the EFFECT FOVEA (12 inputs about the SELECTED candidate; see _effect_fovea_channels)
        self._effect_fovea = int(getattr(params, "effect_fovea", 0) or 0)
        # v1.149: SETTLED VIEW. An animated step returns several grids and FrameData.grid is frame[0]:
        # on r11l that is the board BEFORE the move plus one counter cell, and after a level clear it
        # is still the previous level's board (27/27 transitions measured). 1 = the brain reads
        # frame[-1], the board the action produced (what the Kaggle adapter already feeds it).
        self._settled_view = int(getattr(params, "settled_view", 0) or 0)
        self._reset_carry = int(getattr(params, "reset_carry", 0) or 0)      # v1.150
        self._agency = int(getattr(params, "agency_sense", 0) or 0)          # v1.152
        self._cand_ring = int(getattr(params, "cand_ring", 0) or 0)          # v1.154: ring centres as click candidates
        self._cand_edge = int(getattr(params, "cand_edge", 0) or 0)          # v1.154: edge-clamped candidates
        _fr = os.environ.get('MULTINXON_CLICK_FREE_R')                             # v1.159: the free pointer near objects
        self._click_free_r = int(_fr) if _fr not in (None, '') else int(getattr(params, "click_free_r", 0) or 0)
        self._steer = None                                                      # v1.160: the steering layer (steer1160)
        if float(getattr(params, "steer", 0) or 0) > 0 or str(os.environ.get('MULTINXON_STEER_FORCE', '0') or '0') == '1':
            try:
                if float(getattr(params, "steer_v5", 0) or 0) > 0 or str(os.environ.get('MULTINXON_STEER_V5', '0') or '0') == '1':
                    import steer1164 as _s164                                   # v1.164: steering v5 (v4 + a goal allowed to act)
                    self._steer = _s164.Steer5(params, seed=seed)
                elif float(getattr(params, "steer_v4", 0) or 0) > 0 or str(os.environ.get('MULTINXON_STEER_V4', '0') or '0') == '1':
                    import steer1163 as _s163                                   # v1.163: steering v4 (v3 + goal map + hygiene)
                    self._steer = _s163.Steer4(params, seed=seed)
                elif float(getattr(params, "steer_v3", 0) or 0) > 0 or str(os.environ.get('MULTINXON_STEER_V3', '0') or '0') == '1':
                    import steer1162 as _s162                                   # v1.162: steering v3 (M1 + M3 online)
                    self._steer = _s162.Steer3(params, seed=seed)
                elif float(getattr(params, "steer_v2", 0) or 0) > 0 or str(os.environ.get('MULTINXON_STEER_V2', '0') or '0') == '1':
                    import steer1161 as _s161                                   # v1.161: steering v2
                    self._steer = _s161.Steer2(params, seed=seed)
                else:
                    import steer1160 as _s160
                    self._steer = _s160.Steer(params, seed=seed)
            except Exception as _se160:
                self._steer = None; self._steer_error = f'{type(_se160).__name__}: {_se160}'[:200]
        self._steer_sense = int(getattr(params, "steer_sense", 0) or 0)            # v1.161: 8 steering-sense inputs
        # ============ v1.155 ============
        self._v55_border = int(getattr(params, "change_border", 0) or 0)
        self._v55_budget = int(getattr(params, "budget_sense", 0) or 0)
        self._v55_obj = int(getattr(params, "object_sense", 0) or 0)
        self._v55_trans = int(getattr(params, "transient_sense", 0) or 0)
        self._v55_tprior = float(getattr(params, "type_prior", 0) or 0)
        self._v55_death = int(getattr(params, "death_memory", 0) or 0)
        self._v55_wprior = float(getattr(params, "worked_prior", 0) or 0)
        self._v55_init_state(full=True)
        self._ag_prev_comps = None; self._ag_conf = {}
        self._fov_reset(full=True)
        # runtime self-regulation state
        self._reward_rate: float = 0.0     # EMA of per-step reward
        self._steps_since_reward: int = 0
        self._autoreg_plasticity_mult: float = 1.0   # current plasticity scaling
        # cross-episode memory (developmental accumulation across seeds)
        self._crossep_frac = float(getattr(params, "crossep_memory_fraction", 0.0))

        # rolling telemetry for fitness shaping (sensorimotor coupling, etc.)
        self.telemetry = _Telemetry()

    # ---- v1.059: 2a episodic working-state persistence -------------------
    def export_episodic_state(self) -> Dict[str, Any]:
        """Export the ACCUMULATED working state the brain learned about this game
        instance (the "2a" tier) — distinct from the slow synaptic weights and
        from the transient live-frame scaffolding. This is the game-memory that,
        if restored on reload, lets a warmed brain skip much of its re-entrainment:
        visit/novelty counts, play-step counter (drives plasticity annealing),
        goal latent, and whether any reward has been seen. Small and JSON-safe."""
        try:
            # visit counts can be large; keep the most-visited signatures (cap)
            vc = self._visit_counts or {}
            if len(vc) > 4000:
                items = sorted(vc.items(), key=lambda kv: -kv[1])[:4000]
                vc = dict(items)
            return {
                'ever_scored': bool(getattr(self, '_ever_scored', False)),
                'play_steps': int(getattr(self, '_play_steps', 0)),
                'goal_dy': float(getattr(self, '_goal_dy', 0.0)),
                'goal_dx': float(getattr(self, '_goal_dx', 0.0)),
                'visit_counts': {str(k): int(v) for k, v in vc.items()},
            }
        except Exception:
            return {}

    def import_episodic_state(self, state: Dict[str, Any],
                              restore_anneal_clock: bool = False,
                              restore_visit_counts: bool = False) -> None:
        """Restore the 2a working state captured by export_episodic_state, so a
        reloaded brain starts with its game-memory. Does NOT touch synaptic weights
        (those come from the brain snapshot) and does not require the live frame.
        Call AFTER reset()/construction.

        v1.060b CRITICAL FIX — the annealing clock must NOT be restored on a fresh
        game. The captured ``play_steps`` is tens of thousands (e.g. 59,842 / 105,911),
        but the brain's exploration/plasticity anneal COMPLETES at ~3,684 steps.
        Restoring that huge play_steps saturates the anneal to "fully settled" at step
        0 of a fresh game, dropping exploration gain from ~0.91 (early) to ~0.34
        (settled) before the brain has re-found anything — it gets locked into
        exploitation mode with nothing to exploit, collapsing performance (this caused
        the Kaggle 0.13 -> 0.01 regression). So by default we force play_steps = 0
        (anneal starts cold, brain explores normally) and we DON'T restore the
        saturated visit_counts map (its hashes are from a different episode and the
        high counts suppress needed re-exploration). The learned WEIGHTS — the real
        warm-start — are untouched in the snapshot. ``ever_scored`` and the goal vector
        are harmless and still restored. The two restore_* flags exist only for
        controlled A/B experiments; leave them False for any real run/deployment."""
        if not state:
            return
        try:
            self._ever_scored = bool(state.get('ever_scored', False))
            # FIX: do not let the captured play-step count poison the anneal clock.
            if restore_anneal_clock:
                self._play_steps = int(state.get('play_steps', 0) or 0)
            else:
                self._play_steps = 0
            self._goal_dy = float(state.get('goal_dy', 0.0) or 0.0)
            self._goal_dx = float(state.get('goal_dx', 0.0) or 0.0)
            if restore_visit_counts:
                vc = state.get('visit_counts', {}) or {}
                restored = {}
                for k, v in vc.items():
                    try:
                        restored[int(k)] = int(v)
                    except (ValueError, TypeError):
                        pass
                if restored:
                    self._visit_counts = restored
        except Exception:
            pass

    # ---- lifecycle --------------------------------------------------------
    def reset(self, frame: FrameData) -> None:
        self._ag_prev_comps = None; self._ag_conf = {}; self._ag_track = None   # v1.152: the agency sense starts fresh
        if hasattr(self, '_v55_border'):
            self._v55_init_state(full=True)      # v1.155: a new try forgets deaths, what worked, the type statistics
            try:
                self._v55_lv = int(getattr(frame, 'levels_completed', 0) or 0)   # a level-L start is not a level-up
            except Exception:
                pass
        if getattr(self, "_settled_view", 0):
            frame = _settled_frame(frame)
        h, w = frame.grid_size()
        if hasattr(self, "_fov_reset"):
            self._fov_reset(full=True)     # v1.149
        # v1.076/1.077: fresh action-genes capture each episode
        self._action_hist = {}
        self._action_seq = []
        self._avail_actions = set()
        self._plan_steps = 0; self._plan_total = 0   # v1.078 planning telemetry
        self.cursor = [h // 2 if h else 0, w // 2 if w else 0]
        self.prev_grid = [row[:] for row in frame.grid]
        self.prev_score = int(frame.score)
        self.idle_steps = 0
        self.last_action = None
        self._recent_sigs = []
        self._ever_scored = False     # persistent: has any reward been seen?
        # v1.014: novelty counts persist ACROSS in-trial resets (they are only
        # created fresh in __init__), so a brain that re-reaches an earlier
        # configuration on a new attempt gets a smaller bonus — it is rewarded
        # for net progress over the whole trial, not for replaying the start.
        if not hasattr(self, "_visit_counts"):
            self._visit_counts = {}
        self._play_steps = getattr(self, "_play_steps", 0)
        self._prev_progress = 0.0     # v1.015: reset progress potential
        # v1.017: reset planning/goal-persistence state
        self._goal_dy = 0.0
        self._goal_dx = 0.0
        self._commit_action = None
        self._commit_remaining = 0
        self._last_dir = (0, 0)
        # v1.020: reset learning-dynamics traces (annealing uses _play_steps)
        # v1.033: full reset clears the per-level success prototypes (new brain)
        self._gg_prototypes = {}
        self._gg_proto_n = {}
        # v1.090: pre-clear value state is per-life; clear it on a full reset
        self._pcv_ever_cleared = False
        self._pcv_steps_since_clear = 0
        self._pcv_score_trace = 0.0
        self._pcv_ctrl_by_action = {}
        self._pcv_ctrl_tries = {}
        self._pcv_reach_counts = {}
        self._pcv_last_action_idx = None
        # v1.091: reset the world-model state on a full reset (per-life)
        self._wm_W = None
        self._wm_feat_dim = 0
        self._wm_err_by_action = {}
        self._wm_err_prev = {}
        self._wm_progress_by_action = {}
        self._wm_reach_by_state = {}
        self._wm_prev_feat = None
        self._wm_prev_action_idx = None
        self._wm_ever_cleared = False
        self._wm_steps_since_clear = 0
        self._wm_best_progress = 0.0
        self._gg_recent_feats = []
        self._gg_prev_dist = None
        self._gg_levels = 0
        self._se_counts = {}               # v1.034: clear coverage (new brain)
        # v1.041: full reset clears the action-sequence memory (new brain)
        self._action_mem = {}
        self._am_prev_state = None
        self._am_prev_action = None
        self._am_prev_data = None
        self._am_prev_levels = 0
        self._am_commit_action = None
        self._am_commit_left = 0
        self._da_trace = 0.0
        self._consol_remaining = 0
        # v1.027: full reset clears ALL replay memory (new brain/new life)
        self._success_memory = []
        self._recent_inputs = []
        self._steps_since_replay = 0
        self._prev_levels = 0
        # v1.021: reset closed-loop self-regulation state
        self._reward_rate = 0.0
        self._steps_since_reward = 0
        self._autoreg_plasticity_mult = 1.0
        self._cur_level = 0           # v1.023: reset current-level tracker
        self.telemetry = _Telemetry()

    def set_frontier(self, on: bool = True) -> None:
        """v1.042 — the lifelong scheduler calls this once the brain has mastered
        level 1 and is being granted bonus episodes to practise the L1→L2
        boundary. When at the frontier the brain leans harder on its goal-directed
        machinery (goal-gradient + action memory) and less on undirected re-
        exploration, so the extra episodes push the boundary instead of re-
        clearing the level it already owns."""
        self._at_frontier = bool(on)

    def warm_perturb(self, frame: FrameData) -> None:
        """v1.043 — a LIGHT reset used when the brain has already cleared level 1
        this episode and is practising the L2 crossing. Unlike soft_reset it KEEPS
        the frontier flag, the action memory, the goal-gradient prototypes AND the
        exploration counts (so the brain stays 'warmed up' at the boundary), but
        perturbs the per-step bookkeeping and nudges exploration so it tries a
        DIFFERENT approach to the crossing rather than repeating the failed one.
        This is what turns one cold L2 attempt per episode into many warm ones."""
        h, w = frame.grid_size()
        self.cursor = [self.rng.randint(0, h - 1) if h else 0,
                       self.rng.randint(0, w - 1) if w else 0]
        self.prev_grid = [row[:] for row in frame.grid]
        self.prev_score = int(frame.score)
        self.idle_steps = 0
        self._recent_sigs = []
        # keep: action memory, goal-gradient prototypes, visit counts, weights,
        #       _at_frontier, _play_steps  (stay warmed at the frontier)
        self._am_prev_state = None
        self._am_prev_action = None
        self._am_commit_left = 0
        self._at_frontier = True
        # nudge exploration: bump the count-based explorer so the next attempt
        # diverges from the path that just failed
        self._gg_prev_dist = None

    def soft_reset(self, frame: FrameData) -> None:
        self._ag_prev_comps = None       # v1.152: the board jumped; what I control (the confidences) is kept
        if hasattr(self, '_v55_border'):
            self._v55_prev_comps = None; self._v55_ctx = 0   # v1.155: deaths / type stats / what worked are KEPT (same level)
            try:
                self._v55_lv = int(getattr(frame, 'levels_completed', 0) or 0)
            except Exception:
                pass
        """v1.024 continuous learning: start a new EPISODE while KEEPING what the
        brain has learned. The underlying Neuraxon network (self.brain) and its
        synaptic weights are never rebuilt here, so learning persists; we only
        reset the per-episode bookkeeping (cursor, idle counter, recent-frame
        signatures, the planning/eligibility traces) and — crucially — we do NOT
        reset _play_steps (so plasticity annealing spans the whole lifetime) nor
        the novelty visit-counts (so the brain isn't re-rewarded for replaying
        the start each episode). This is the "no training/inference, always
        learning" path: one continuous life across many episodes."""
        if getattr(self, "_settled_view", 0):
            frame = _settled_frame(frame)
        h, w = frame.grid_size()
        # ============ v1.150: RESET THE CARRIED STATE (gene reset_carry) ============
        # NAS 1149, 8-episode lives: a FROZEN twin (weights restored every step) cleared 0.89 in episode 1 and
        # 0.22 after it; the plastic brain 1.00 -> 0.11. Only state carried by soft_reset can do that: the
        # novelty visit counts and the fovea's click memory steer a brain away from targets it already used.
        # reset_carry=1 clears exactly what reset() clears and soft_reset kept. The weights are untouched.
        _rc = bool(getattr(self, "_reset_carry", 0))
        if _rc:
            self._visit_counts = {}
        if hasattr(self, "_fov_reset"):
            self._fov_reset(full=_rc)      # v1.149: the board jumps; click memory kept (v1.150: unless reset_carry)
        self.cursor = [h // 2 if h else 0, w // 2 if w else 0]
        self.prev_grid = [row[:] for row in frame.grid]
        self.prev_score = int(frame.score)
        self.idle_steps = 0
        self.last_action = None
        self._recent_sigs = []
        # keep: _ever_scored, _visit_counts, _play_steps, learned weights
        self._prev_progress = 0.0
        self._goal_dy = 0.0; self._goal_dx = 0.0
        self._commit_action = None; self._commit_remaining = 0
        # v1.041: KEEP the action-sequence memory across episodes (the whole point
        # — replay a crossing found in a previous life), reset per-step trackers.
        self._am_prev_state = None
        self._am_prev_action = None
        self._am_prev_data = None
        self._am_prev_levels = 0
        self._am_commit_action = None
        self._am_commit_left = 0
        self._last_dir = (0, 0)
        # keep DA eligibility/consolidation soft (halve, don't zero) so a reward
        # learned at the end of one episode still biases the start of the next
        # v1.033: KEEP the per-level success prototypes across episodes (a level
        # solved once keeps guiding its own crossing later), reset per-episode
        # rolling features / shaping baseline / current-level counter.
        self._gg_recent_feats = []
        self._gg_prev_dist = None
        self._gg_levels = 0
        self._da_trace *= 0.5
        self._consol_remaining = 0
        # v1.027: KEEP _success_memory across episodes — that is the whole point
        # (a solution found in one episode is replayed/reinforced in later ones).
        # Only the rolling recent-input buffer resets for the new episode.
        self._recent_inputs = []
        self._prev_levels = 0
        # keep the closed-loop reward-rate estimate (it is part of "how am I
        # doing" and should carry across episodes), but reset the stuck counter
        self._steps_since_reward = 0
        # keep _cur_level handling to the perception path
        self.telemetry = _Telemetry()

        # v1.028 BETWEEN-EPISODE "DREAM" CONSOLIDATION. At the episode boundary
        # (the analogue of sleep), replay the accumulated success-memory several
        # times under strong plasticity — extra learning that costs NO game
        # steps, so each PC learns MORE per life without slowing evaluation. This
        # is hippocampal replay during rest: the brain rehearses what worked
        # between attempts, so the next episode starts with the solution more
        # firmly consolidated. Scales with replay_batch; only runs if replay is
        # enabled and there is something worth rehearsing.
        if getattr(self, '_replay_enabled', False) and self._success_memory:
            n_dream = max(1, int(round(self._dream_cycles)))
            for _ in range(n_dream):
                self._replay_consolidate()

    # ---- perception: grid -> trinary sensory vector -----------------------
    # ==================== v1.111 EFFERENCE COPY ====================
    def _efference_channels(self) -> List[int]:
        """One-hot over the 6 game actions + click position + a did-it-do-anything bit.

        8 channels, trinary like every other channel in this encoder:
            0..5  one-hot of the action taken at the PREVIOUS step (+1 taken, -1 not)
            6     click row, signed relative to grid centre (0 when the action was not a click)
            7     click col, likewise

        This is the ONLY route by which the network learns what its own outputs do. Without it
        the brain sees a screen and must guess; with it, an action and its consequence arrive on
        the same tick and STDP can bind them.
        """
        out = [-1] * 6
        i = self._eff_prev_idx
        if 0 <= i < 6:
            out[i] = 1
        cr, cc = self._eff_prev_click
        out.append(_sign(cr) if cr >= 0 else 0)
        out.append(_sign(cc) if cc >= 0 else 0)
        return out

    def _delta_retina(self, grid, h: int, w: int) -> List[int]:
        """Per-cell CHANGE since the previous frame, at K x K blocks — the consequence half.

        Channel 0 (`frame_changed`) collapses this to one bit. A brain cannot learn "this action
        moved the object north" from a bit. Each block is +1 if anything in it changed, -1 if
        nothing did, and the block grid is coarse (K x K over the whole grid) so the cost is
        K^2 channels rather than h*w.
        """
        k = self._delta_cells
        n = k * k
        if not grid or not self.prev_grid:
            return [-1] * n
        ph = len(self.prev_grid)
        pw = len(self.prev_grid[0]) if ph else 0
        out: List[int] = []
        bh = max(1, h // k)
        bw = max(1, w // k)
        for by in range(k):
            for bx in range(k):
                y0, x0 = by * bh, bx * bw
                y1 = h if by == k - 1 else min(h, y0 + bh)
                x1 = w if bx == k - 1 else min(w, x0 + bw)
                ch = -1
                for yy in range(y0, y1):
                    if ch == 1:
                        break
                    row = grid[yy]
                    prow = self.prev_grid[yy] if yy < ph else None
                    for xx in range(x0, x1):
                        pv = prow[xx] if (prow is not None and xx < pw) else None
                        if pv is None or row[xx] != pv:
                            ch = 1
                            break
                out.append(ch)
        return out

    def _relevel_bound(self, grid):
        """Bound bracketing the modal object size on THIS board.

        The same rule that predicted sp80 correctly in v1121 — its interactive bars are 80 cells,
        every bound before then topped out at 1-64 and excluded them, and widening took it from
        0 level-2 events to 36 (Fisher p = 6.8e-12). Applied here per LEVEL rather than per
        game, because lp85's objects shrink from 16 cells at level 1 to 4 at level 6 and one
        bound cannot cover both.
        """
        if not grid or not grid[0]:
            return None
        bg = self._bg_colour(grid)
        comps = self._components_tagged(grid, bg)
        if not comps:
            return None
        sizes = sorted(n for (n, c, y, x) in comps)
        # the 75th-percentile object, rounded up to a power of two: covers the bulk of the
        # board's objects without reaching for the one huge background blob
        p75 = sizes[min(len(sizes) - 1, int(0.75 * len(sizes)))]
        hi = 2
        while hi < p75 and hi < 256:
            hi *= 2
        return max(2, min(256, hi))

    def perceive(self, frame: FrameData) -> Dict[str, Dict[int, float]]:
        # v1.117: the click mask needs the current grid, and perceive() is the only place that
        # reliably sees every frame.
        try:
            self._last_grid = frame.grid
        except Exception:
            pass
        # ============ v1.123: RE-DERIVE THE BOUND ON A LEVEL TRANSITION ============
        # Once per LEVEL, not once per frame. `auto` did the latter in v1118 and v1120, was
        # noisy, and produced zero level-2 events in two runs. This is stable within a level
        # and correct across them.
        if getattr(self, '_relevel_mask', 0):
            try:
                _lv = int(float(getattr(frame, 'levels_completed', 0) or 0))
            except Exception:
                _lv = 0
            if _lv != getattr(self, '_rl_level', 0):
                self._rl_level = _lv
                self._cm_sig = None
                try:
                    _hi = self._relevel_bound(frame.grid)
                    if _hi:
                        self._rl_hi = _hi
                        self._cm_hi = _hi
                except Exception:
                    pass
        grid = frame.grid
        # v1.147: THE BLIND TEST, on the verified path. MULTINXON_BLIND_INPUT=1 blanks what the NETWORK
        # sees (every cell the background) while self._last_grid -- which the click mask reads -- keeps
        # the real board. If actions-to-clear do not change, the network is not using its input.
        if os.environ.get('MULTINXON_BLIND_INPUT', '') == '1':
            try:
                _bgv = grid[0][0]
                grid = [[_bgv for _ in r] for r in grid]
            except Exception:
                pass
        h, w = frame.grid_size()
        if h == 0 or w == 0:
            channels = [0] * self.n_in
            return self._route(channels, frame)

        cy, cx = self.cursor
        cy = max(0, min(h - 1, cy)); cx = max(0, min(w - 1, cx))
        self.cursor = [cy, cx]

        def sal(yy: int, xx: int) -> int:
            """saliency: +1 non-background, -1 wall-ish (colour 8), 0 background."""
            if not (0 <= yy < h and 0 <= xx < w):
                return -1            # off-grid behaves like a wall
            c = grid[yy][xx]
            if c == 0:
                return 0
            if c == 8:
                return -1
            return 1

        _b55 = int(getattr(self, '_v55_border', 0) or 0)
        changed = 0 if (_v55_interior_equal(grid, self.prev_grid, _b55) if _b55 > 0 else _grids_equal(grid, self.prev_grid)) else 1
        score_delta = int(frame.score) - int(self.prev_score)
        if score_delta > 0 or int(frame.score) > 0:
            self._ever_scored = True

        cur_color = grid[cy][cx] if (0 <= cy < h and 0 <= cx < w) else 0
        prev_cur = (self.prev_grid[cy][cx]
                    if (self.prev_grid and 0 <= cy < len(self.prev_grid)
                        and 0 <= cx < len(self.prev_grid[0])) else cur_color)

        # ordered channel list — most informative first (so trimming is safe)
        ch: List[int] = []
        ch.append(1 if changed else -1)                                   # 0 frame_changed
        ch.append(_sign(score_delta))                                     # 1 score delta
        ch.append(_trinarize(sal(cy, cx)))                                # 2 fovea centre saliency
        ch.append(_sign(sal(cy - 1, cx) - sal(cy, cx)))                   # 3 north pull
        ch.append(_sign(sal(cy + 1, cx) - sal(cy, cx)))                   # 4 south pull
        ch.append(_sign(sal(cy, cx - 1) - sal(cy, cx)))                   # 5 west pull
        ch.append(_sign(sal(cy, cx + 1) - sal(cy, cx)))                   # 6 east pull
        # global quadrant occupancy (coarse layout)
        ch.append(self._quad_occupancy(grid, h, w, 0, 0))                 # 7 TL
        ch.append(self._quad_occupancy(grid, h, w, 0, 1))                 # 8 TR
        ch.append(self._quad_occupancy(grid, h, w, 1, 0))                 # 9 BL
        ch.append(self._quad_occupancy(grid, h, w, 1, 1))                 # 10 BR
        ch.append(self._novelty(grid, h, w))                              # 11 novelty
        # --- v1.009 richer perception (cursor-relative + colour/object) ---
        # where is the cursor in the grid (coarse self-localisation)
        ch.append(_sign(cy - (h - 1) / 2.0))                              # 12 cursor row pos
        ch.append(_sign(cx - (w - 1) / 2.0))                              # 13 cursor col pos
        # diagonal saliency gradients (8-connected attention, not just 4)
        ch.append(_sign(sal(cy - 1, cx + 1) - sal(cy, cx)))               # 14 NE pull
        ch.append(_sign(sal(cy - 1, cx - 1) - sal(cy, cx)))               # 15 NW pull
        ch.append(_sign(sal(cy + 1, cx + 1) - sal(cy, cx)))               # 16 SE pull
        ch.append(_sign(sal(cy + 1, cx - 1) - sal(cy, cx)))               # 17 SW pull
        # colour dynamics & object membership at the cursor
        ch.append(1 if cur_color != prev_cur else -1)                     # 18 cursor colour changed
        ch.append(self._same_colour_neighbours(grid, h, w, cy, cx,        # 19 object membership
                                                cur_color))
        ch.append(self._local_colour_diversity(grid, h, w, cy, cx))       # 20 local 3x3 diversity
        ch.append(1 if self._ever_scored else -1)                         # 21 ever-scored (persistent)
        # ============ v1.120: THE CHANGE TRACKER, FINALLY CALLED ============
        # v1119 built _change_features and never invoked it, so the four channels that run was
        # designed around were computed zero times. Channel 0 above already carries one bit of
        # "did the frame change"; these add the COMPONENT-level facts that bit cannot express.
        #
        # `moved` is the one that matters and the one the delta retina could never see: on r11l
        # a click TRANSLATES THE WHOLE RIGID STRUCTURE, a component event that is nearly
        # invisible pixel by pixel. Gated on the gene, so width is unchanged when off.
        #
        # Worth knowing how sparse the signal is: a random agent's actions change the frame on
        # only 3.2% of steps on lp85 and 8.4% on ft09 (5 seeds, 100 actions each), against
        # 61% on r11l and 77% on cd82. On the sparse games nearly the whole budget is spent on
        # no-ops, and nothing in the terminal level reward can tell the brain that.
        if getattr(self, '_change_track', 0):
            _cf = self._change_features(grid)
            ch.append(2.0 * _cf[0] - 1.0)                                  # 22 did_change
            ch.append(2.0 * _cf[1] - 1.0)                                  # 23 n_changed
            ch.append(2.0 * _cf[2] - 1.0)                                  # 24 moved
            ch.append(2.0 * _cf[3] - 1.0)                                  # 25 novel
        # ============ v1.126: THE SELECT -> MOVE CYCLE ============
        # sp80 level 1 is "Click. Down, down. Right, right, right. Spacebar." -- the click takes
        # CONTROL of the bar and the D-pad then moves it. cd82 level 1 is "Up, up. Spacebar.
        # Right, down, down, left. Spacebar, spacebar." -- nine actions, zero clicks, and the
        # spacebar picks the block UP so the D-pad carries it.
        #
        # Both need the brain to know it is currently HOLDING something. r11l does not -- one
        # click, one move, no state between actions -- which is exactly why r11l is at 0.55x
        # human and these two are at 3.74x and 6.95x against 7- and 9-action solutions.
        #
        # Three channels, derived from the action stream with no per-game assumption:
        #   26  was the LAST action a select-type action (click or spacebar)
        #   27  how long ago the last select was, saturating -- "am I mid-carry?"
        #   28  did that select actually change the board -- did I really pick something up
        if getattr(self, '_select_phase', 0):
            ch.append(2.0 * float(getattr(self, '_sp_last_sel', 0.0)) - 1.0)   # 26 just selected
            _s = float(getattr(self, '_sp_since', 0))
            ch.append(2.0 * min(1.0, _s / 8.0) - 1.0)                          # 27 since select
            ch.append(2.0 * float(getattr(self, '_sp_worked', 0.0)) - 1.0)     # 28 select worked

        # ==================== v1.111: EFFERENCE COPY + FRAME DELTA ====================
        # THE OMISSION THIS FIXES. Across 105 NAS runs and 416,728 trials the brain has been
        # given the current screen and asked for an action, with NO input telling it WHICH
        # ACTION PRODUCED THE SCREEN IT IS LOOKING AT. Channel 0 compresses the entire
        # consequence of an action into one bit (`frame_changed`). Every synapse in every
        # harvested brain was therefore trained on state-only input, and the network is
        # structurally incapable of learning "ACTION3 moves the cursor north".
        #
        # There IS an action->effect model in this file (_wm_W, action -> feature-delta) but it
        # lives in a Python attribute, is absent from the saved snapshot (verified: '_wm_W' does
        # not appear anywhere in a harvested .brain.json.gz), and is therefore discarded on every
        # save, reload and cold seed. Its genes correlate with nothing: replay_enabled +0.026 on
        # cold pass, goal_gradient_scale -0.081, all |rho| < 0.12 over 283 evidenced brains.
        #
        # This matters because of where the variance actually lives. Two-way decomposition over
        # 115 sp80 brains x 20 SHARED cold seeds (1,956 cleared episodes, log actions):
        #     which brain  ....  8.8%      which seed  ....  31.5%
        #     brain x seed interaction  ....  59.7%
        # The interaction term is "this brain suits THIS layout" and it is the majority of the
        # outcome. Adapting to it requires knowing what you did and what changed. One of those
        # two facts has never been available to the network.
        #
        # Both blocks are gated OFF by default so the baseline is bit-identical to v1.110.
        # ============ v1.149: the EFFECT FOVEA (+12) goes FIRST after the 22 scalars ============
        # The pinned n_in trims the END of this list, and evolution varies genes that change the width
        # (colour_channels, fovea_patch_size, level_context_scale): a genome with a 5x5 patch emits 157
        # channels for n_in 130 (measured in the 1149 smoke). Here no trim can ever reach the fovea;
        # everything after it pads or trims exactly as in 1148.
        if getattr(self, "_effect_fovea", 0):
            ch.extend(self._effect_fovea_channels(frame, grid))
        if getattr(self, "_agency", 0):
            ch.extend(self._agency_channels(grid))      # v1.152: the agency sense (+8)
        if getattr(self, "_v55_budget", 0):
            ch.extend(self._v55_budget_channels(grid))   # v1.155 (+2)
        if getattr(self, "_v55_obj", 0):
            ch.extend(self._v55_object_channels(grid))   # v1.155 (+6)
        if getattr(self, "_v55_trans", 0):
            ch.extend(self._v55_transient_channels())    # v1.155 (+2)
        if self._eff_copy:
            ch.extend(self._efference_channels())
        if self._delta_cells > 0:
            ch.extend(self._delta_retina(grid, h, w))

        # --- v1.104 HIGH-ACUITY FOVEA, placed BEFORE the global retina --------
        # Channel order is "most informative first" because the list is TRIMMED to n_in.
        # The patch goes ahead of the R x R retina so a small n_in never truncates the one
        # part of the encoder that can resolve a single-pixel object.
        ch.extend(self._fovea_patch(grid, h, w, cy, cx))

        # --- v1.023 richer scene perception -------------------------------
        # R×R spatial retina over the WHOLE grid (global object layout)
        ch.extend(self._retina(grid, h, w))
        if getattr(self, '_steer_sense', 0):
            ch.extend(self._v1161_sense())                 # v1.161: the steering sense (+8), LAST
        # per-colour presence (distinguish objects by colour)
        if self._colour_channels > 0:
            ch.extend(self._colour_presence(grid, h, w))
            ch.extend(self._colour_at_cursor(grid, h, w, cy, cx))
        # v1.023 LEVEL CONTEXT: a per-level identity signal so the brain knows
        # WHICH level it is on and can reconfigure for it (malleable brain).
        if self._level_context_scale > 0.0:
            lvl = int(getattr(frame, "levels_completed", 0))
            self._cur_level = lvl
            slot = min(lvl, self._max_level_ctx - 1)
            for s in range(self._max_level_ctx):
                ch.append(self._level_context_scale if s == slot else 0.0)

        self._v1112_last_raw = len(ch)      # v1112: preflight check 4 reads the PRE-TRIM width
        # ============ v1.112 P0-C: PADDING AND TRUNCATION ARE NOW LOUD ============
        # Channels are ordered "most informative first" and the R x R retina sits near the
        # END, so an undersized n_in truncates the retina MID-ROW -- the brain sees the top of
        # the board and not the bottom -- and an oversized one feeds the network constant
        # zeros.  Neither was ever logged.  v1104 spent an entire run measuring K^2 channels
        # of PADDING and reporting it as fovea acuity.
        #
        # n_in is computable from the genome (22 scalars + eff + delta + K^2 + R^2 + 2C +
        # level slots), so a mismatch is always a CONFIGURATION BUG, never a legitimate state.
        # v1.177 (eyes1177): a rewired brain has n_in = its own base width + the eyes block; the base channels are padded /
        # trimmed to the base width exactly as before, then the eyes block is appended.
        _nb = int(getattr(self, '_n_in_base', None) or self.n_in)
        if getattr(self, '_e77_w', 0) and _nb < self.n_in:
            ch = (ch + [0] * (_nb - len(ch)))[:_nb] if len(ch) != _nb else ch
            try:
                ch = ch + self._eyes77_channels(frame, grid)
            except Exception as _e77x:
                self._e77_error = f'perceive: {type(_e77x).__name__}: {_e77x}'[:160]
        _n = len(ch)
        if _n != self.n_in and not getattr(self, "_v1112_warned", False):
            self._v1112_warned = True
            _kind = "TRUNCATED" if _n > self.n_in else "PADDED"
            _msg = (f"[v1112] ENCODER {_kind}: produced {_n} channels, n_in={self.n_in} "
                    f"(delta {_n - self.n_in:+d}).  retina={getattr(self, '_retina_size', '?')} "
                    f"colour={getattr(self, '_colour_channels', '?')} "
                    f"fovea={getattr(self, '_fovea_k', 0)} "
                    f"eff={int(bool(getattr(self, '_eff_copy', False)))} "
                    f"delta={getattr(self, '_delta_cells', 0)}")
            if os.environ.get("MULTINXON_STRICT_CHANNELS") == "1":
                raise ValueError(_msg + "  [MULTINXON_STRICT_CHANNELS=1]")
            print(_msg, flush=True)
        # pad / trim to exactly n_in
        if len(ch) < self.n_in:
            ch = ch + [0] * (self.n_in - len(ch))
        else:
            ch = ch[: self.n_in]

        return self._route(ch, frame)

    def _route(self, channels: List[int], frame: FrameData) -> Dict[str, Dict[int, float]]:
        ext: Dict[str, Dict[int, float]] = {"sensory": {}}
        for i, nid in enumerate(self.sensory_ids):
            if i < len(channels):
                ext["sensory"][nid] = float(channels[i])
        # direct-to-motor injections (mirrors GoL: proprioception + urgency).
        # motor input 0 <- frame_changed (proprioceptive "did I move things"),
        # motor input 1 <- score-delta urgency.
        motor_ext: Dict[int, float] = {}
        if self.motor_sensory_ids:
            if len(self.motor_sensory_ids) > 0 and len(channels) > 0:
                motor_ext[self.motor_sensory_ids[0]] = float(channels[0])
            if len(self.motor_sensory_ids) > 1 and len(channels) > 1:
                motor_ext[self.motor_sensory_ids[1]] = float(channels[1])
        if motor_ext:
            ext["motor"] = motor_ext
        self._last_channels = channels
        return ext

    def _quad_occupancy(self, grid, h, w, qy, qx) -> int:
        y0 = qy * (h // 2); y1 = h if qy else (h // 2)
        x0 = qx * (w // 2); x1 = w if qx else (w // 2)
        nonbg = 0; total = 0
        for y in range(y0, max(y0 + 1, y1)):
            row = grid[y]
            for x in range(x0, max(x0 + 1, x1)):
                total += 1
                if row[x] != 0:
                    nonbg += 1
        if total == 0:
            return 0
        frac = nonbg / total
        return 1 if frac > 0.10 else (-1 if frac < 0.02 else 0)

    # ---- v1.022 richer perception ----------------------------------------
    def _retina(self, grid, h, w) -> List[float]:
        """Downsample the WHOLE grid into an R×R saliency map so the brain sees
        global object layout, not just a 2×2 blur. Each cell is the mean
        saliency (non-background fraction) of its block, quantised to trinary/
        multi-level. This is the single biggest perceptual upgrade: it turns the
        rest of the scene from 4 numbers into R² spatially-arranged signals."""
        R = self._retina_size
        out: List[float] = []
        for ry in range(R):
            y0 = (ry * h) // R; y1 = max(y0 + 1, ((ry + 1) * h) // R)
            for rx in range(R):
                x0 = (rx * w) // R; x1 = max(x0 + 1, ((rx + 1) * w) // R)
                nonbg = 0; total = 0
                for y in range(y0, min(y1, h)):
                    row = grid[y]
                    for x in range(x0, min(x1, w)):
                        total += 1
                        if row[x] != 0 and row[x] != 8:
                            nonbg += 1
                frac = (nonbg / total) if total else 0.0
                # quantise to {-1,0,1} (levels=3) or finer if more levels asked
                if self._retina_levels <= 3:
                    out.append(1.0 if frac > 0.15 else (-1.0 if frac < 0.03 else 0.0))
                else:
                    out.append(round(2.0 * frac - 1.0, 2))   # signed density
        return out

    def _fovea_patch(self, grid, h, w, cy, cx) -> List[float]:
        """v1.104 — a K x K window of the grid around the cursor, at FULL resolution.

        Each cell is the raw colour, signed-normalised to [-1, 1] (out-of-bounds -> -1), so a
        single 1x1 token is a distinct value in a distinct channel instead of being averaged
        into a 144-cell block and thresholded away. With fovea_patch_colour off the patch is
        occupancy only (cheaper, colour-blind).

        Cost is K^2 channels: a 7x7 patch is 49, against 576 for the R=24 global retina that
        would be needed to resolve the same detail. Measured step cost at R=24 was 951 ms
        against 55 ms at R=6; the patch buys the acuity without the quadratic global blow-up.
        """
        K = self._fovea_k
        if K <= 0:
            return []
        half = K // 2
        out: List[float] = []
        for dy in range(-half, half + 1):
            y = cy + dy
            for dx in range(-half, half + 1):
                x = cx + dx
                if 0 <= y < h and 0 <= x < w:
                    v = grid[y][x]
                    if self._fovea_patch_colour:
                        out.append(round((v / 4.5) - 1.0, 3))      # colour 0..9 -> -1..1
                    else:
                        out.append(1.0 if (v != 0 and v != 8) else -1.0)
                else:
                    out.append(-1.0)
        return out

    def _colour_presence(self, grid, h, w) -> List[float]:
        """Per-colour presence: +1 if colour c occupies a meaningful share of
        the grid, else -1. Lets the brain DISTINGUISH objects by colour (a red
        goal vs a blue key) instead of collapsing everything to 'saliency'."""
        C = self._colour_channels
        counts = [0] * 10
        total = 0
        for y in range(h):
            row = grid[y]
            for x in range(w):
                c = row[x]
                if 0 <= c < 10:
                    counts[c] += 1
                total += 1
        out: List[float] = []
        # colours 1..C (skip 0 background); +1 present, 0 trace, -1 absent
        for c in range(1, min(10, C + 1)):
            frac = (counts[c] / total) if total else 0.0
            out.append(1.0 if frac > 0.02 else (0.0 if frac > 0.0 else -1.0))
        return out

    def _colour_at_cursor(self, grid, h, w, cy, cx) -> List[float]:
        """One-hot-ish encoding of the colour currently under the cursor, so the
        brain knows WHAT it is touching (not just that something is there)."""
        C = self._colour_channels
        cur = grid[cy][cx] if (0 <= cy < h and 0 <= cx < w) else 0
        out: List[float] = []
        for c in range(1, min(10, C + 1)):
            out.append(1.0 if cur == c else -1.0)
        return out

    def _same_colour_neighbours(self, grid, h, w, cy, cx, c) -> int:
        """Object membership: how many 4-neighbours share the cursor's colour.
        +1 = inside a same-colour blob, -1 = isolated, 0 = edge."""
        if c == 0:
            return -1            # background: not part of an object
        same = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < h and 0 <= xx < w and grid[yy][xx] == c:
                same += 1
        return 1 if same >= 3 else (-1 if same == 0 else 0)

    def _local_colour_diversity(self, grid, h, w, cy, cx) -> int:
        """Distinct non-background colours in the 3x3 around the cursor.
        +1 = busy/edge region (3+), -1 = uniform (≤1), 0 = some structure."""
        seen = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < h and 0 <= xx < w:
                    v = grid[yy][xx]
                    if v != 0:
                        seen.add(v)
        n = len(seen)
        return 1 if n >= 3 else (-1 if n <= 1 else 0)

    def _novelty(self, grid, h, w) -> int:
        # coarse signature: 4x4 occupancy bitmap hashed
        bits = 0
        for by in range(4):
            for bx in range(4):
                y0 = by * h // 4; y1 = max(y0 + 1, (by + 1) * h // 4)
                x0 = bx * w // 4; x1 = max(x0 + 1, (bx + 1) * w // 4)
                nonbg = 0; total = 0
                for y in range(y0, min(y1, h)):
                    for x in range(x0, min(x1, w)):
                        total += 1
                        if grid[y][x] != 0:
                            nonbg += 1
                bit = 1 if (total and nonbg / total > 0.10) else 0
                bits = (bits << 1) | bit
        novel = bits not in self._recent_sigs
        self._recent_sigs.append(bits)
        if len(self._recent_sigs) > self._sig_cap:
            self._recent_sigs.pop(0)
        return 1 if novel else -1

    # ---- cognition: reward + brain step -----------------------------------
    def think(self, ext_inputs: Dict[str, Dict[int, float]],
              reward: float = 0.0, score_reward: float = 0.0) -> List[int]:
        # v1.020 learning dynamics. The dopamine that drives STDP is shaped by
        # three mechanisms that make in-episode acquisition RELIABLE (reducing
        # the seed-to-seed variance that keeps level 2 a lottery):
        #
        #  (1) plasticity annealing — a within-episode learning-gain schedule
        #      that starts high (fast acquisition) and decays toward the end
        #      (consolidation), so the brain settles on what it found instead of
        #      drifting. gain(t) lerps anneal_start → anneal_end over
        #      anneal_steps.
        #  (2) reward-gated consolidation — a real SCORE reward gets an extra DA
        #      boost and opens a short window of boosted plasticity, locking a
        #      level crossing in ("tag-and-capture") rather than letting later
        #      exploration overwrite it.
        #  (3) reward eligibility trace — DA decays gradually after a reward
        #      instead of snapping to baseline, crediting the action sequence
        #      that led to the (sparse, delayed) reward.
        t = float(getattr(self, "_play_steps", 0))
        frac = min(1.0, t / float(self._anneal_steps))
        gain = self._anneal_start + (self._anneal_end - self._anneal_start) * frac

        # base phasic DA from this step's reward
        da_phasic = max(0.0, min(0.6, 0.5 * reward)) if reward != 0.0 else 0.0
        # reward-gated consolidation: real score reward → stronger pulse + window
        if score_reward > 0.0 and self._consol_boost > 1.0:
            da_phasic = min(0.9, da_phasic * self._consol_boost)
            if self._consol_window > 0:
                self._consol_remaining = self._consol_window
        # eligibility trace: persist a decaying DA signal across steps
        if self._elig_decay > 0.0:
            self._da_trace = max(self._da_trace * self._elig_decay, da_phasic)
            da_phasic = max(da_phasic, self._da_trace)

        # consolidation window boosts the effective learning gain briefly
        eff_gain = gain
        if self._consol_remaining > 0:
            eff_gain = gain * 1.5
            self._consol_remaining -= 1
        # v1.021: closed-loop adaptive plasticity multiplier (stuck → learn more)
        if self._autoreg > 0.0:
            eff_gain = eff_gain * self._autoreg_plasticity_mult

        # v1.023 level-conditioned malleability: drive a per-level gating
        # modulator (acetylcholine-like) so DIFFERENT LEVELS route through
        # different effective sub-circuits — the malleable "spheres per level".
        # Also bias the learning gain per level so plasticity that solves L1
        # doesn't overwrite L2 (reduces the measured task interference).
        if self._level_gate_scale > 0.0 or self._level_plasticity_bias != 0.0:
            lvl = int(self._cur_level)
            if self._level_gate_scale > 0.0:
                # distinct ACh level per level index → different gating regime
                slot = min(lvl, self._max_level_ctx - 1)
                ach = 0.3 + self._level_gate_scale * (slot / max(1, self._max_level_ctx - 1))
                self.brain.set_global_modulator("acetylcholine", max(0.0, min(1.0, ach)))
            if self._level_plasticity_bias != 0.0:
                eff_gain = eff_gain * (1.0 + self._level_plasticity_bias * lvl)
                eff_gain = max(0.2, min(4.0, eff_gain))

        # apply: dopamine baseline + (gain-scaled) phasic component
        if da_phasic > 0.0 or eff_gain != 1.0:
            da = 0.15 + min(0.75, eff_gain * da_phasic)
            self.brain.set_global_modulator("dopamine", da)

        for _ in range(self.micro_steps):
            self.brain.simulate_step(ext_inputs)
        outs = self.motor.network.get_output_states()
        outs = (list(outs) + [0] * self.n_out)[: self.n_out]
        return outs

    def _replay_consolidate(self) -> None:
        """v1.027 hippocampal-style replay. Re-present a batch of stored winning
        perceptions to the brain under a strong, sustained dopamine pulse and
        elevated plasticity, so the synapses that produced a successful level
        crossing are consolidated into a repeatable skill. Replays are drawn from
        the persistent success-memory (which spans the brain's whole life), so a
        solution found in one episode is reinforced across all later ones. Replay
        does not emit actions — it is internal 'offline' consolidation."""
        if not self._success_memory:
            return
        batch = self._success_memory[-self._replay_batch:]
        # strong consolidation signal during replay
        da = 0.15 + min(0.8, 0.6 * self._replay_plasticity)
        self.brain.set_global_modulator("dopamine", da)
        for inp in batch:
            # one micro-step per replayed perception is enough to reinforce the
            # association without runaway drift; keep it cheap.
            self.brain.simulate_step(inp)

    # ---- v1.091 world-model helpers --------------------------------------
    def _wm_features(self, grid):
        """A small fixed-length feature vector summarising the grid, for the forward model.
        Cheap and coarse on purpose: colour histogram (10 bins) + fill fraction + a 2x2
        coarse occupancy (4) = 15 dims. Enough for the model to learn 'this action changes
        the colour balance / the occupied region' without a heavy representation."""
        try:
            h = len(grid); w = len(grid[0]) if h else 0
            if h == 0 or w == 0:
                return [0.0] * 15
            hist = [0.0] * 10
            occ = [0.0, 0.0, 0.0, 0.0]
            flat = 0
            for y in range(h):
                row = grid[y]
                qy = 0 if y < h / 2 else 1
                for x in range(w):
                    c = int(row[x])
                    if 0 <= c < 10:
                        hist[c] += 1.0
                    qx = 0 if x < w / 2 else 1
                    occ[qy * 2 + qx] += 1.0 if c != 0 else 0.0
                    flat += 1
            n = float(max(1, flat))
            hist = [v / n for v in hist]
            occ = [v / n for v in occ]
            fill = sum(occ)
            return hist + [fill] + occ
        except Exception:
            return [0.0] * 15

    def _wm_state_sig(self, grid):
        """Coarse signature of a state for the empowerment reach table."""
        try:
            f = self._wm_features(grid)
            return tuple(int(round(v * 8)) for v in f)   # coarse-quantised feature vector
        except Exception:
            return (0,)

    def _wm_update(self, frame, changed):
        """Train the online forward model on the last transition and refresh the per-action
        learning-progress + empowerment tables. Model: for each action a, a linear map W[a]
        from previous features to the OBSERVED feature-delta. Error = |predicted - actual|.
        Learning-progress = how much that error DROPPED vs last time this action was tried."""
        feat = self._wm_features(frame.grid)
        d = len(feat)
        if self._wm_feat_dim == 0:
            self._wm_feat_dim = d
        ai = self._wm_prev_action_idx
        if self._wm_prev_feat is not None and ai is not None and len(self._wm_prev_feat) == d:
            # actual feature-delta from the previous step to now
            actual = [feat[i] - self._wm_prev_feat[i] for i in range(d)]
            if self._wm_W is None:
                self._wm_W = {}
            w = self._wm_W.get(ai)
            if w is None:
                w = [0.0] * d          # predicts a per-dim delta (bias-only linear model)
                self._wm_W[ai] = w
            # prediction is the model's current per-dim delta estimate for this action
            pred = w
            err = sum(abs(pred[i] - actual[i]) for i in range(d)) / d
            # online update toward the observed delta (least-mean-squares / Hebbian-ish)
            lr = self._wm_lr
            for i in range(d):
                w[i] += lr * (actual[i] - w[i])
            # learning-progress = previous error - current error (positive = the model is
            # LEARNING this action's effect; that is where useful structure is)
            prev_err = self._wm_err_by_action.get(ai)
            if prev_err is not None:
                lp = max(0.0, prev_err - err)
                # smooth it
                old = self._wm_progress_by_action.get(ai, 0.0)
                self._wm_progress_by_action[ai] = 0.7 * old + 0.3 * min(1.0, lp * 4.0)
                self._wm_best_progress = max(self._wm_best_progress,
                                             self._wm_progress_by_action[ai])
            self._wm_err_by_action[ai] = 0.7 * self._wm_err_by_action.get(ai, err) + 0.3 * err
            # empowerment bookkeeping: record that (prev-state, action) reached THIS state-sig
            if self._wm_emp > 0.0 and self._wm_prev_feat is not None:
                psig = tuple(int(round(v * 8)) for v in self._wm_prev_feat)
                reach = self._wm_reach_by_state.get(psig)
                if reach is None:
                    reach = set()
                    self._wm_reach_by_state[psig] = reach
                if len(reach) < 64:
                    reach.add(self._wm_state_sig(frame.grid))
        # remember this step for the next update
        self._wm_prev_feat = feat
        try:
            self._wm_prev_action_idx = int(getattr(self.last_action, 'value', 0)) \
                if self.last_action is not None else None
        except Exception:
            self._wm_prev_action_idx = None

    # ---- action: motor outputs -> GameAction ------------------------------
    def _plan_action(self, avail, frame, h, w):
        """v1.078 REAL PLANNING — genuine 1-step lookahead that ACTIVATES (unlike the dead
        goal_persist gene). It scores each AVAILABLE action by a cheap predicted effect and
        returns the best action + its margin over the runner-up. The heuristic is grounded
        and prototype-free so it engages from step 1: directional actions score by whether
        the target cursor cell has interactive CONTENT (non-background), biased toward the
        learned success prototype when one exists; click/grab score the current cell. This
        turns the brain from purely reactive to goal-directed (move the cursor toward the
        cells that matter, then act there)."""
        if not frame.grid or h == 0 or w == 0:
            return None, 0.0
        grid = frame.grid
        cy, cx = self.cursor
        # background = most common colour (cheap mode over the grid)
        from collections import Counter as _C
        flat = [c for row in grid for c in row]
        bg = _C(flat).most_common(1)[0][0] if flat else 0
        # optional prototype tilt (list of success features) when the brain has learned one
        proto = self._gg_prototypes.get(getattr(self, '_cur_level', 0))
        proto_vec = proto if isinstance(proto, list) else None

        # v1.090: pre-clear value weight. Full strength before any clear; anneals to 0
        # over preclear_anneal_steps once the success prototype has been seeded, so the
        # learned prototype (which is more specific) takes over rather than being fought.
        _pcv_w = 0.0
        if self._pcv_scale > 0.0:
            if not self._pcv_ever_cleared:
                _pcv_w = self._pcv_scale
            else:
                _frac = 1.0 - min(1.0, self._pcv_steps_since_clear / self._pcv_anneal)
                _pcv_w = self._pcv_scale * _frac
        # colour-level controllability: has acting on THIS colour tended to change the world?
        _score_trace = self._pcv_score_trace

        # v1.091: world-model gradient weight. Full strength before any clear, annealing to 0
        # once a success prototype has seeded (the prototype is more specific). This gates the
        # per-action curiosity + empowerment bonuses added to the candidate scores below.
        _wm_w = 0.0
        if self._wm_scale > 0.0:
            if not self._wm_ever_cleared:
                _wm_w = self._wm_scale
            else:
                _wf = 1.0 - min(1.0, self._wm_steps_since_clear / self._wm_anneal)
                _wm_w = self._wm_scale * _wf

        def content(yy, xx):
            if yy < 0 or xx < 0 or yy >= h or xx >= w:
                return -1.0
            s = 1.0 if grid[yy][xx] != bg else 0.0   # is there something to act on here?
            if proto_vec:
                # tilt toward cells whose colour appears in the success prototype
                try:
                    s += 0.3 if float(grid[yy][xx]) in proto_vec else 0.0
                except Exception:
                    pass
            # v1.090 PRE-CLEAR VALUE: before any clear (and fading after), value a cell by
            # signals available every step, not just "is it non-background":
            #   * score_delta_value   — recent score movement makes ALL structure more
            #                            worth engaging (the score is moving, keep going)
            #   * controllability     — colours the brain has been able to CHANGE are worth
            #                            targeting (they are the manipulable parts of the game)
            #   * reachable_novelty    — folded in at the action level below, not per-cell
            if _pcv_w > 0.0:
                try:
                    col = int(grid[yy][xx])
                    if col != bg:
                        s += _pcv_w * self._pcv_score * _clip(_score_trace, 0.0, 1.0)
                        ctrl = self._pcv_ctrl_by_action.get(col, 0.0)
                        s += _pcv_w * self._pcv_ctrl * ctrl
                except Exception:
                    pass
            return s

        cand = {}
        if GameAction.ACTION1 in avail: cand[GameAction.ACTION1] = content(cy - 1, cx)
        if GameAction.ACTION2 in avail: cand[GameAction.ACTION2] = content(cy + 1, cx)
        if GameAction.ACTION3 in avail: cand[GameAction.ACTION3] = content(cy, cx - 1)
        if GameAction.ACTION4 in avail: cand[GameAction.ACTION4] = content(cy, cx + 1)
        here = content(cy, cx)
        if GameAction.ACTION6 in avail: cand[GameAction.ACTION6] = here
        if GameAction.ACTION5 in avail: cand[GameAction.ACTION5] = here
        if GameAction.ACTION7 in avail: cand[GameAction.ACTION7] = here * 0.9
        if not cand:
            return None, 0.0
        # v1.091 WORLD-MODEL per-action bonus. Curiosity (learning-progress) and empowerment
        # are properties of an ACTION, not a cell, so they are added here to each candidate.
        # This is the pre-clear gradient that is SELF-GENERATED (from the brain's forward
        # model) rather than engine-provided -- the whole point of 1.091.
        if _wm_w > 0.0 and (self._wm_curio > 0.0 or self._wm_emp > 0.0):
            try:
                cur_sig = self._wm_state_sig(grid)
                for _a in list(cand.keys()):
                    ai = int(getattr(_a, 'value', 0))
                    bonus = 0.0
                    if self._wm_curio > 0.0:
                        # learning-progress: reward actions whose prediction error is high
                        # BUT decreasing (learnable), not just high (noise). Falls back to raw
                        # error early, before a progress estimate exists.
                        lp = self._wm_progress_by_action.get(ai)
                        if lp is None:
                            lp = self._wm_err_by_action.get(ai, 0.5)  # untried -> curious
                        bonus += self._wm_curio * _clip(lp, 0.0, 1.0)
                    if self._wm_emp > 0.0:
                        # empowerment: how many DISTINCT next-states this action's neighbourhood
                        # can reach (per the model). More reach = more control = more valuable.
                        reach = self._wm_reach_by_state.get(cur_sig)
                        emp = (len(reach) / (self._wm_horizon * 4.0)) if reach else 0.0
                        bonus += self._wm_emp * _clip(emp, 0.0, 1.0)
                    cand[_a] = cand[_a] + _wm_w * bonus
            except Exception:
                pass
        # v1.079 D: optional 2-step lookahead — for each directional candidate, add the
        # best CONTENT reachable one more move out (cheap 2-ply). Helps the L2->L3 crossing
        # where the payoff is two moves away (steer toward, then act).
        _depth = getattr(self, '_plan_depth', 1)
        if _depth >= 2:
            moves = {GameAction.ACTION1: (-1, 0), GameAction.ACTION2: (1, 0),
                     GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)}
            for a, (dy, dx) in moves.items():
                if a in cand:
                    ny, nx = cy + dy, cx + dx
                    nxt = max(content(ny + ddy, nx + ddx)
                              for ddy, ddx in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)))
                    cand[a] = cand[a] + 0.5 * nxt
                    # v1.080: 3-step — look one more ply out (discounted) for the L3 payoff
                    if _depth >= 3:
                        n2 = max(content(ny + 2*ddy, nx + 2*ddx)
                                 for ddy, ddx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                        cand[a] = cand[a] + 0.25 * n2
        best = max(cand, key=cand.get)
        vals = sorted(cand.values(), reverse=True)
        margin = vals[0] - (vals[1] if len(vals) > 1 else 0.0)
        return best, float(margin)

    def _masked_avail(self, frame):
        """v1.159: the available actions intersected with the action mask, never empty (enum members or plain ints)"""
        avail = list(getattr(frame, 'available_actions', None) or [])
        m = getattr(self, '_action_mask', None)
        if m:
            keep = [a for a in avail if a is GameAction.RESET or int(getattr(a, 'value', a) if not isinstance(a, int) else a) in m]
            if any(a is not GameAction.RESET for a in keep):
                return keep
        return avail

    def act(self, outs: List[int], frame: FrameData) -> Tuple[GameAction, Dict[str, Any]]:
        avail = set(frame.available_actions or [])
        # v1.111: intersect with the per-game live set. Never empty the set — if the mask and
        # the engine disagree the engine wins, because a brain with no legal action is worse
        # than a brain wasting one.
        if self._action_mask:
            _keep = {a for a in avail if int(getattr(a, 'value', a) if not isinstance(a, int) else a) in self._action_mask}
            if _keep:
                avail = _keep
        h, w = frame.grid_size()

        o = (list(outs) + [0] * 7)[:7]
        o0, o1, o2, o3, o4, o5, o6 = o  # vert, horiz, confirm, click, secondary, rest, spare

        # v1.017 planning: bias the motor outputs toward the goal latent (the
        # direction that has been producing progress) and, if an action-commit
        # option is active, steer toward the committed direction. This is added
        # to the network's own readout, so the brain still leads — planning only
        # tilts the decision toward what has been working (working-memory bias).
        if self._goal_persist_scale > 0.0:
            o0 = o0 + self._goal_persist_scale * _clip(self._goal_dy, -1.0, 1.0)
            o1 = o1 + self._goal_persist_scale * _clip(self._goal_dx, -1.0, 1.0)
        if self._commit_scale > 0.0 and self._commit_remaining > 0 \
                and self._commit_action is not None:
            cdy, cdx = self._commit_action
            o0 = o0 + self._commit_scale * cdy
            o1 = o1 + self._commit_scale * cdx
        # re-trinarise the (possibly biased) directional outputs
        o0 = _sign(o0) if abs(o0) > 0.5 else 0
        o1 = _sign(o1) if abs(o1) > 0.5 else 0

        # 1) steer the attention cursor (also the directional readout)
        cy, cx = self.cursor
        # v1.112: one press = one LATTICE step, matching "arrows move exactly one lattice
        # step" (31 of 320 verified claims).  _lattice_px == 1 reproduces v1.111 exactly.
        _lp = getattr(self, "_lattice_px", 1)
        ny = max(0, min(max(0, h - 1), cy + _sign(o0) * _lp))
        nx = max(0, min(max(0, w - 1), cx + _sign(o1) * _lp))
        self._last_dir = (ny - cy, nx - cx)   # v1.017: remember the move made
        cy, cx = ny, nx
        self.cursor = [cy, cx]
        # ============ v1.147: CAND_CURSOR -- the network CHOOSES the target ============
        # Until now o0/o1 moved a cursor ONE PIXEL per action and the click landed on the mask
        # candidate nearest to it: which object got clicked was decided by where a random walk
        # happened to be, not by the network (hence the r1146 seed spread of 1.0 on 40/40 lp85
        # brains, and a newborn with the mask clearing as well as the evolved brain). With
        # cand_cursor=1, o1 = +-1 steps to the next/previous candidate in reading order and
        # o0 = +-1 jumps a row of candidates; the cursor is placed ON the selected candidate, so
        # the click lands exactly on the network's choice, the fovea (when enabled) sees the
        # target, and the dopamine that follows a productive click reinforces the choice.
        if int(getattr(self, '_cand_cursor', 0) or 0):
            try:
                _cc = sorted(self._click_candidates(getattr(self, '_last_grid', None)) or [])
                if _cc:
                    _n = len(_cc); _step = max(1, int(round(_n ** 0.5)))
                    self._cand_idx = (int(getattr(self, '_cand_idx', 0)) + int(o1 or 0) + int(o0 or 0) * _step) % _n
                    _ty, _tx = _cc[self._cand_idx]
                    cy = int(round(_ty * 64.0 / max(1, h))); cx = int(round(_tx * 64.0 / max(1, w)))
                    self.cursor = [cy, cx]
            except Exception:
                pass

        # 2) choose an action by priority, restricted to available_actions
        action: Optional[GameAction] = None
        data: Dict[str, Any] = {}

        if o3 == 1 and GameAction.ACTION6 in avail:
            action = GameAction.ACTION6
            data = self._cursor_click(h, w)
        elif o0 == -1 and GameAction.ACTION1 in avail:
            action = GameAction.ACTION1
        elif o0 == 1 and GameAction.ACTION2 in avail:
            action = GameAction.ACTION2
        elif o1 == -1 and GameAction.ACTION3 in avail:
            action = GameAction.ACTION3
        elif o1 == 1 and GameAction.ACTION4 in avail:
            action = GameAction.ACTION4
        elif o2 == 1 and GameAction.ACTION5 in avail:
            action = GameAction.ACTION5
        elif o4 == 1 and GameAction.ACTION7 in avail:
            action = GameAction.ACTION7
        # v1.155 death_memory: a key action that ended an attempt from THIS state is not repeated
        if action is not None and action != GameAction.ACTION6 and getattr(self, '_v55_death', 0) and self._v55_dead:
            try:
                _sg = self._v55_sig(getattr(self, '_last_grid', None))
                if (_sg, getattr(action, 'name', '')) in self._v55_dead:
                    for _alt in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
                                 GameAction.ACTION5, GameAction.ACTION7):
                        if _alt in avail and _alt != action and (_sg, _alt.name) not in self._v55_dead:
                            action = _alt; data = {}; self._v55_beh['death_avoided'] += 1
                            break
            except Exception:
                pass

        motor_active = any(v != 0 for v in (o0, o1, o2, o3, o4))

        # v1.078 REAL PLANNING override: when planning is active and the lookahead has a
        # CONFIDENT goal-directed choice (margin over runner-up above threshold), take the
        # planned action instead of the reactive one. This is counted in telemetry
        # (mech_frac_planning), so — unlike the dead goal_persist gene — the mechanism is
        # observably active. Blended by _planning_scale: 0 disables it entirely.
        self._plan_fired = 0
        if getattr(self, '_planning_scale', 0.0) > 0.0 and avail and not getattr(self, '_net_decides', False):
            planned, margin = self._plan_action(avail, frame, h, w)
            if planned is not None and margin >= self._plan_margin_thresh:
                # probability of following the plan scales with planning_scale
                if self.rng.random() < self._planning_scale:
                    if planned in (GameAction.ACTION6,):
                        action, data = planned, self._cursor_click(h, w)
                    else:
                        action, data = planned, {}
                    self._plan_fired = 1
                    self._last_mech = 'planning'
                    motor_active = True

        # 3) exploration safety net (never freeze)
        explore = False
        if getattr(self, "_net_decides", False):
            # v1.149: a random action only with probability eval_explore; otherwise a step on which the
            # network selected nothing available acts on the network's own selection.
            _ee = self._eval_explore
            if action is not None:
                self._net_last_action = action
            if action is None:
                if _ee > 0.0 and self.rng.random() < _ee:
                    explore = True
                else:
                    action, data = self._network_default(avail, h, w)
                    self._net_default_n += 1
            elif self.idle_steps >= self.idle_explore_steps and _ee > 0.0 and self.rng.random() < _ee:
                explore = True
        elif action is None:
            explore = True
        elif self.idle_steps >= self.idle_explore_steps:
            # stalled despite a command — diversify
            if self.rng.random() < self.explore_probability:
                explore = True

        if explore:
            action, data = self._explore(frame, h, w)
            self._last_mech = 'structured_explore'  # v1.044 region telemetry

        if action is None:  # absolute fallback so the loop always advances
            playable = [a for a in self._masked_avail(frame) if a is not GameAction.RESET]
            if playable:
                action = self.rng.choice(playable)
                if action.is_complex():
                    data = self._cursor_click(h, w)
            else:
                action = GameAction.RESET

        # telemetry for fitness shaping
        self.telemetry.note(channels=getattr(self, "_last_channels", []),
                            motor=o, motor_active=motor_active, explored=explore)
        # v1.076 ACTION-GENES capture: record which action was emitted this step.
        # Histogram = per-action-type counts (answers "is the brain pressing X?");
        # sequence = the ordered action trace (mined into motifs for clears).
        try:
            _an = getattr(action, 'name', str(action))
            self._action_hist[_an] = self._action_hist.get(_an, 0) + 1
            if len(self._action_seq) < self._action_seq_cap:
                self._action_seq.append(_an)
        except Exception:
            pass
        self.last_action = action
        # ---- v1.111: record the efference copy for the NEXT perceive() ----
        # GameAction values are 1-based (ACTION1..ACTION6); the one-hot is 0-based.
        try:
            _v = int(getattr(action, 'value', 0))
            self._eff_prev_idx = _v - 1 if 1 <= _v <= 6 else -1
        except Exception:
            self._eff_prev_idx = -1
        try:
            _d = data if isinstance(data, dict) else {}
            if 'y' in _d and 'x' in _d:
                self._eff_prev_click = (float(_d['y']), float(_d['x']))
            else:
                self._eff_prev_click = (-1.0, -1.0)
        except Exception:
            self._eff_prev_click = (-1.0, -1.0)
        # v1.090: remember the action index so next step can credit its controllability
        try:
            self._pcv_last_action_idx = int(getattr(action, 'value', 0))
        except Exception:
            self._pcv_last_action_idx = None
        # v1.078: tally planning activation (mech_frac_planning telemetry)
        self._plan_steps += 1
        self._plan_total += int(getattr(self, '_plan_fired', 0))
        # v1.142: dir_memory -- noop_memory for DIRECTIONS. The cd82 action log (15,520 actions)
        # showed 50-68% of D-pad presses changing nothing (walls, corners). A direction whose last
        # press changed nothing is flagged for N steps; if the network picks a flagged direction
        # the next unflagged one in rotation is emitted instead. Direction choice stays the
        # network's whenever its choice is not a known dead end.
        try:
            _dm = int(getattr(self, '_dir_memory', 0) or 0)
            if _dm and not getattr(self, '_net_decides', False) and action in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
                _t = int(getattr(self, '_step_t', 0))
                _flag = getattr(self, '_dir_noop_until', {})
                if _flag.get(action.value, -1) >= _t:
                    _rot = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
                    _i = _rot.index(action)
                    for _k in range(1, 4):
                        _c = _rot[(_i + _k) % 4]
                        if _c in avail and _flag.get(_c.value, -1) < _t:
                            action = _c; data = {}; break
        except Exception:
            pass
        # v1.141: the select->move handoff. sp80's grammar is click-a-bar, D-pad it, spacebar;
        # the brain learns the select and not what follows (cold pass bimodal: half at 1x, half
        # at 3x). With the countdown running, a click is replaced by the D-pad direction the
        # network chose (or space if it chose none). Direction stays the network's.
        try:
            if int(getattr(self, '_after_select_n', 0) or 0) > 0 and not getattr(self, '_net_decides', False):
                self._after_select_n -= 1
                if action == GameAction.ACTION6:
                    _alt = None
                    if o0 == -1 and GameAction.ACTION1 in avail: _alt = GameAction.ACTION1
                    elif o0 == 1 and GameAction.ACTION2 in avail: _alt = GameAction.ACTION2
                    elif o1 == -1 and GameAction.ACTION3 in avail: _alt = GameAction.ACTION3
                    elif o1 == 1 and GameAction.ACTION4 in avail: _alt = GameAction.ACTION4
                    elif GameAction.ACTION5 in avail: _alt = GameAction.ACTION5
                    if _alt is not None:
                        action = _alt; data = {}
        except Exception:
            pass
        self._last_action_was_click = (action == GameAction.ACTION6)   # v1.140
        try:                                                              # v1.155: what was done, from which state
            self._v55_asig = self._v55_sig(getattr(self, '_last_grid', None))
            _lcc = getattr(self, '_last_click_cand', None)
            self._v55_akey = ('C', tuple(_lcc)) if (action == GameAction.ACTION6 and _lcc is not None) else getattr(action, 'name', None)
            if action == GameAction.ACTION7:
                self._v55_beh['undo'] += 1
        except Exception:
            pass
        try:
            self._last_action_id = int(getattr(action, 'value', 0) or 0)   # v1.141
        except Exception:
            self._last_action_id = None
        return action, data

    # ============ v1.149 helpers ============
    def _network_default(self, avail, h, w):
        """The network selected no available action: act on ITS state, draw no random number.
        Click the selected candidate if clicks exist (cand_cursor put the cursor on it); else repeat
        the last action the network chose; else the first available action."""
        if GameAction.ACTION6 in avail:
            return GameAction.ACTION6, self._cursor_click(h, w)
        la = getattr(self, "_net_last_action", None)
        if la is not None and la in avail:
            return la, ({} if la != GameAction.ACTION6 else self._cursor_click(h, w))
        for a in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
                  GameAction.ACTION5, GameAction.ACTION7):
            if a in avail:
                return a, {}
        return None, {}

    def _fov_reset(self, full=True):
        self._fov_prev_settled = None
        self._fov_last_click = None
        self._fov_last_lv = 0 if full else getattr(self, "_fov_last_lv", 0)
        if full:
            self._fov_clicks = {}
            self._fov_effect = {}
            self._fov_floor = None

    @staticmethod
    def _component_size(grid, y0, x0, cap=4096):
        try:
            col = grid[y0][x0]; H = len(grid); W = len(grid[0])
        except Exception:
            return 0
        seen = {(y0, x0)}; st = [(y0, x0)]
        while st and len(seen) < cap:
            y, x = st.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u, v = y + dy, x + dx
                if 0 <= u < H and 0 <= v < W and (u, v) not in seen and grid[u][v] == col:
                    seen.add((u, v)); st.append((u, v))
        return len(seen)

    # ============ v1.152: THE AGENCY SENSE ============
    # The solver's first probe on every game answers "what do I control?" (ar25: the yellow L; ls20: the maroon
    # block; tu93: the maze node; su15: the avatar). A brain has no such notion. These 8 inputs give it one:
    #   0    a component MOVED since the last frame (1/0)
    #   1-2  its displacement dx, dy (/8, clipped to [-1, 1])
    #   3    confidence that one colour is what my actions move (an EMA: +0.1 when that colour moves, x0.9 per step)
    #   4-5  the controlled object's centroid (x, y) in [-1, 1]
    #   6-7  the controlled object relative to the cursor (x, y) (/32, clipped)
    # Movers: components matched by (colour, size) between consecutive boards, displaced by >= 0.5 cell.
    def _ag_components(self, grid):
        H = len(grid); W = len(grid[0]) if H else 0
        if not H or not W:
            return []
        cnt = {}
        for row in grid:
            for v in row:
                cnt[v] = cnt.get(v, 0) + 1
        bg = max(cnt, key=cnt.get); seen = [[False] * W for _ in range(H)]; out = []
        for y0 in range(H):
            for x0 in range(W):
                if seen[y0][x0] or grid[y0][x0] == bg:
                    continue
                col = grid[y0][x0]; st = [(y0, x0)]; seen[y0][x0] = True; n = 0; sy = 0; sx = 0
                while st:
                    y, x = st.pop(); n += 1; sy += y; sx += x
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = y + dy, x + dx
                        if 0 <= u < H and 0 <= v < W and not seen[u][v] and grid[u][v] == col:
                            seen[u][v] = True; st.append((u, v))
                if n <= (H * W) // 4:
                    out.append((col, n, sx / n, sy / n))
        return out

    def _agency_channels(self, grid):
        out = [0.0] * 8
        try:
            H = len(grid); W = len(grid[0]) if H else 0
            cur = self._ag_components(grid); prev = self._ag_prev_comps; self._ag_prev_comps = cur
            for k in list(self._ag_conf):
                self._ag_conf[k] *= 0.9
            track = getattr(self, '_ag_track', None)
            if prev is not None:
                pk = {}
                for c in prev:
                    pk.setdefault((c[0], c[1]), []).append(c)
                movers = []
                for c in cur:
                    cands = pk.get((c[0], c[1]))
                    if not cands:
                        continue
                    # nearest earlier component of the same colour and size: duplicates (m0r0's mirrored
                    # avatars) match their own previous position; a still component matches itself at distance 0
                    q = min(cands, key=lambda p_: (p_[2] - c[2]) ** 2 + (p_[3] - c[3]) ** 2)
                    dx = c[2] - q[2]; dy = c[3] - q[3]
                    if abs(dx) + abs(dy) >= 0.5:
                        movers.append((c, dx, dy))
                if movers:
                    c, dx, dy = max(movers, key=lambda t: t[0][1])
                    self._ag_conf[c[0]] = self._ag_conf.get(c[0], 0.0) + 0.1
                    self._ag_track = track = {'col': c[0], 'size': c[1], 'x': c[2], 'y': c[3]}
                    out[0] = 1.0; out[1] = max(-1.0, min(1.0, dx / 8.0)); out[2] = max(-1.0, min(1.0, dy / 8.0))
            if self._ag_conf and track:
                col, conf = max(self._ag_conf.items(), key=lambda kv: kv[1])
                if conf > 0.05:
                    out[3] = min(1.0, conf)
                    # the controlled object = the TRACKED mover (not the largest component of its colour)
                    same = [c for c in cur if c[0] == track['col'] and c[1] == track['size']] or [c for c in cur if c[0] == track['col']]
                    if same:
                        c = min(same, key=lambda c: (c[2] - track['x']) ** 2 + (c[3] - track['y']) ** 2)
                        track['x'], track['y'] = c[2], c[3]
                        out[4] = (c[2] / max(1, W - 1)) * 2 - 1; out[5] = (c[3] / max(1, H - 1)) * 2 - 1
                        cy, cx = (self.cursor if getattr(self, 'cursor', None) else (None, None))
                        if cx is not None:
                            out[6] = max(-1.0, min(1.0, (c[2] - cx) / 32.0)); out[7] = max(-1.0, min(1.0, (c[3] - cy) / 32.0))
        except Exception:
            return [0.0] * 8
        return out

    def _effect_fovea_channels(self, frame, grid):
        """v1.149 EFFECT FOVEA -- 12 trinary inputs about the candidate cand_cursor has SELECTED (the one
        a click would hit), with effects measured on the SETTLED board (frame.frame[-1]):
          0-3  its colour: one-hot over the board's 4 most frequent non-background colours
          4    its size: -1 tiny (<= 2 cells), 0 small (<= 32), +1 large
          5-7  the last effect of clicking it: none / few (<= 16 cells) / many  (all 0 = never clicked)
          8    the level counter moved on the last step
          9    it has been clicked before
          10   the last action changed the settled board (beyond the episode's quietest change + 1)
          11   a candidate is selected"""
        out = [0] * 12
        try:
            fr = getattr(frame, "frame", None)
            settled = fr[-1] if (isinstance(fr, list) and fr and isinstance(fr[-1], list) and fr[-1]
                                 and isinstance(fr[-1][0], list)) else grid
            prev = self._fov_prev_settled
            n = None
            if prev is not None and len(prev) == len(settled) and prev and len(prev[0]) == len(settled[0]):
                n = sum(1 for y in range(len(settled)) for x in range(len(settled[0])) if prev[y][x] != settled[y][x])
            self._fov_prev_settled = [list(r) for r in settled]
            if n is not None:
                self._fov_floor = n if self._fov_floor is None else min(self._fov_floor, n)
            floor = self._fov_floor or 0
            changed = 1 if (n is not None and n > floor + 1) else 0
            lv = int(getattr(frame, "levels_completed", 0) or 0)
            moved = 1 if lv > self._fov_last_lv else 0
            self._fov_last_lv = lv
            lc = self._fov_last_click
            if lc is not None and n is not None:
                self._fov_effect[lc] = n
            self._fov_last_click = None
            cands = sorted(self._click_candidates(getattr(self, "_last_grid", None)) or [])
            if cands:
                cy, cx = cands[int(getattr(self, "_cand_idx", 0)) % len(cands)]
                g = getattr(self, "_last_grid", None) or settled
                col = g[cy][cx]
                cnt = {}
                for row in g:
                    for v in row:
                        cnt[v] = cnt.get(v, 0) + 1
                bg = max(cnt, key=cnt.get)
                top = [c for c, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if c != bg][:4]
                if col in top:
                    out[top.index(col)] = 1
                size = self._component_size(g, cy, cx)
                out[4] = -1 if size <= 2 else (0 if size <= 32 else 1)
                eff = self._fov_effect.get((cy, cx))
                if eff is not None:
                    if eff <= floor + 1: out[5] = 1
                    elif eff <= 16: out[6] = 1
                    else: out[7] = 1
                out[9] = 1 if self._fov_clicks.get((cy, cx), 0) > 0 else 0
                out[11] = 1
            out[8] = moved
            out[10] = changed
        except Exception:
            pass
        return out

    def _change_features(self, grid):
        """What the LAST action did to the frame. Four numbers in [0, 1].

        Every one of the 25 solve transcripts is driven by this: the agent predicts the next
        frame, compares, and grades itself -- "46/46 predictions", "29/29, zero misses". It is
        also what StochasticGoose used to win the ARC-AGI-3 preview at 12.58% with no LLM:
        *predict which actions change the frame*.

        The click mask answers "which cells respond to a click", and half the games never click.
        This answers "did what I just did do anything", which every game needs:

            did_change   the frame differs from the previous one at all
            n_changed    how many cells differ, log-scaled against the board size
            moved        a component TRANSLATED (same size and colour, new centroid)
            novel        a component appeared or vanished

        `moved` is the one the delta retina was always supposed to see and never did, because
        it was never paired with a component detector: on r11l a click translates the WHOLE
        rigid structure, which is a component-level event and nearly invisible pixel by pixel.
        """
        prev = getattr(self, '_ct_prev', None)
        self._ct_prev = [row[:] for row in grid] if grid else None
        if not prev or not grid or len(prev) != len(grid):
            return (0.0, 0.0, 0.0, 0.0)
        H = len(grid); W = len(grid[0])
        # v1.122: ignore the HUD frame. Four games draw an action counter into the border and it
        # ticks on every action -- r11l's transcript calls one such action "a pure no-op (only
        # the counter ticked)". Without this, did_change is 1.0 on a frame where nothing on the
        # board moved: measured vc33 100% of frames "changed" while 0% of BOARDS did.
        _b = getattr(self, '_hud_border', 0)
        _ys, _ye = (_b, H - _b) if 2 * _b < H else (0, H)
        _xs, _xe = (_b, W - _b) if 2 * _b < W else (0, W)
        n = 0
        for y in range(_ys, _ye):
            gy = grid[y]; py = prev[y]
            for x in range(_xs, _xe):
                if gy[x] != py[x]:
                    n += 1
        if n == 0:
            self._ct_last = (0.0, 0.0, 0.0, 0.0)
            return self._ct_last
        import math as _m
        frac = min(1.0, _m.log1p(n) / _m.log1p(H * W))
        bg = self._bg_colour(grid)
        nowc = self._components_tagged(grid, bg)
        wasc = self._components_tagged(prev, bg)
        now = {(s, c) for (s, c, _y, _x) in nowc}
        was = {(s, c) for (s, c, _y, _x) in wasc}
        wmap = {}
        for (s, c, y, x) in wasc:
            wmap.setdefault((s, c), []).append((y, x))
        moved = 0.0
        for (s, c, y, x) in nowc:
            old = wmap.get((s, c))
            if old and (y, x) not in old:
                moved = 1.0
                break
        novel = 1.0 if ((now - was) or (was - now)) else 0.0
        self._ct_last = (1.0, frac, moved, novel)
        return self._ct_last

    def _components_tagged(self, grid, bg: int):
        """(size, colour, cy, cx) for every 4-connected non-background component."""
        H = len(grid); W = len(grid[0])
        seen = [[False] * W for _ in range(H)]
        out = []
        for y0 in range(H):
            for x0 in range(W):
                if seen[y0][x0] or grid[y0][x0] == bg:
                    continue
                col = grid[y0][x0]; stack = [(y0, x0)]; seen[y0][x0] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] \
                                and grid[ny][nx] == col:
                            seen[ny][nx] = True; stack.append((ny, nx))
                out.append((len(cells), col,
                            sum(c[0] for c in cells) // len(cells),
                            sum(c[1] for c in cells) // len(cells)))
        return out

    def _bg_colour(self, grid) -> int:
        """Background colour. Prefers the BORDER mode over the global mode.

        "Background = modal colour" holds when one colour owns 79-93% of the frame, which is
        true of every game the v1117 mask helped. It fails where it matters: on dc22 the modal
        colour is 30.7% of the frame, on bp35 46.2%, on lp85 48.3% -- and those are three of
        the games the mask broke. The colour around the frame EDGE is background in all 25
        transcripts, and is far more stable than the global mode on a busy board.
        """
        H = len(grid); W = len(grid[0])
        edge: Dict[int, int] = {}
        for x in range(W):
            for y in (0, H - 1):
                edge[grid[y][x]] = edge.get(grid[y][x], 0) + 1
        for y in range(H):
            for x in (0, W - 1):
                edge[grid[y][x]] = edge.get(grid[y][x], 0) + 1
        if edge:
            b, n = max(edge.items(), key=lambda kv: kv[1])
            # only trust it if it actually dominates the border
            if n >= 0.6 * (2 * H + 2 * W):
                return b
        cnt: Dict[int, int] = {}
        for row in grid:
            for v in row:
                cnt[v] = cnt.get(v, 0) + 1
        return max(cnt.items(), key=lambda kv: kv[1])[0] if cnt else 0

    def _components(self, grid, bg: int):
        """4-connected same-colour components of non-background cells: (size, cy, cx)."""
        H = len(grid); W = len(grid[0])
        seen = [[False] * W for _ in range(H)]
        out = []
        for y0 in range(H):
            for x0 in range(W):
                if seen[y0][x0] or grid[y0][x0] == bg:
                    continue
                col = grid[y0][x0]; stack = [(y0, x0)]; seen[y0][x0] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] \
                                and grid[ny][nx] == col:
                            seen[ny][nx] = True; stack.append((ny, nx))
                out.append((len(cells),
                            sum(c[0] for c in cells) // len(cells),
                            sum(c[1] for c in cells) // len(cells)))
        return out

    def _diff_candidates(self, grid):
        """v1.146: TWO REGIONS OF EQUAL SIZE, ONE THE TARGET, ONE THE WORK AREA (tn36 level 2+,
        cd82, sc25, ft09, sb26 all describe exactly this). Regions are the bounding boxes of the
        large connected components (frames, panels, filled blocks); boxes of the same (h, w) are
        paired; the pair whose contents differ most is (target, work); the work box is the one
        with FEWER distinct colours (a work area starts blank). Candidates = work cells whose
        colour differs from the aligned target cell, nearest the box's top-left first. Returns []
        when no equal-size pair exists, so single-region boards fall through untouched. The HUD
        bands are excluded by requiring boxes to be at least 4x4 and at most 60%% of the board."""
        H = len(grid); W = len(grid[0]) if H else 0
        if H < 12 or W < 12:
            return []
        bg = self._bg_colour(grid)
        seen = [[False] * W for _ in range(H)]; boxes = []
        for y0 in range(H):
            for x0 in range(W):
                if seen[y0][x0] or grid[y0][x0] == bg:
                    continue
                col = grid[y0][x0]; stack = [(y0, x0)]; seen[y0][x0] = True; n = 0
                ymin = ymax = y0; xmin = xmax = x0
                while stack:
                    y, x = stack.pop(); n += 1
                    ymin = min(ymin, y); ymax = max(ymax, y); xmin = min(xmin, x); xmax = max(xmax, x)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] and grid[ny][nx] == col:
                            seen[ny][nx] = True; stack.append((ny, nx))
                h = ymax - ymin + 1; w = xmax - xmin + 1
                if n >= 12 and h >= 4 and w >= 4 and h <= 0.6 * H and w <= 0.6 * W:
                    boxes.append((h, w, ymin, xmin, col))
        by = {}
        for h, w, y, x, c in boxes:
            by.setdefault((h, w), []).append((y, x, c))
        best = None
        for (h, w), lst in by.items():
            if len(lst) < 2:
                continue
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    ya, xa, _ = lst[i]; yb, xb, _ = lst[j]
                    if abs(ya - yb) < h and abs(xa - xb) < w:
                        continue                                   # overlapping boxes are one structure
                    diff = 0
                    for dy in range(h):
                        for dx in range(w):
                            if grid[ya + dy][xa + dx] != grid[yb + dy][xb + dx]:
                                diff += 1
                    frac = diff / float(h * w)
                    if 0.05 <= frac <= 0.95 and (best is None or diff > best[0]):
                        best = (diff, h, w, (ya, xa), (yb, xb))
        if best is None:
            return []
        _, h, w, A, B = best
        def ncol(y0, x0):
            return len({grid[y0 + dy][x0 + dx] for dy in range(h) for dx in range(w)})
        work, targ = (A, B) if ncol(*A) <= ncol(*B) else (B, A)
        out = []
        for dy in range(h):
            for dx in range(w):
                if grid[work[0] + dy][work[1] + dx] != grid[targ[0] + dy][targ[1] + dx]:
                    out.append((work[0] + dy, work[1] + dx))
        return out[:64]

    # ======================================================================================================
    #  v1.155 -- truthful change, budget / object / transient senses, type statistics, deaths, what worked
    # ======================================================================================================
    def _v55_init_state(self, full=True):
        self._v55_tstats = {}; self._v55_level_types = []; self._v55_border0 = None; self._v55_bunion = set()
        self._v55_prev_comps = None; self._v55_ctx = 0; self._v55_last_type = None; self._v55_bprev = None
        if full:
            self._v55_dead = set(); self._v55_worked = {}; self._v55_lv = 0; self._v55_raw = None
            self._v55_akey = None; self._v55_asig = None; self._v55_sigs = set()
            self._v55_beh = {'steps': 0, 'effective': 0, 'border_only': 0, 'nochange': 0, 'deaths': 0, 'levelups': 0,
                             'undo': 0, 'repeats': 0, 'distinct': 0, 'death_avoided': 0, 'dead_type_skips': 0}

    def _v55_m(self):
        return int(getattr(self, '_v55_border', 0) or 0) or 4

    def _v55_sig(self, grid):
        return _v55_interior_sig(grid, int(getattr(self, '_v55_border', 0) or 0)) if grid else 0

    def _v55_filters_on(self):
        return bool(getattr(self, '_v55_tprior', 0) or getattr(self, '_v55_death', 0) or getattr(self, '_v55_wprior', 0))

    def _v55_type(self, grid, y, x):
        """an object TYPE: (colour, size band) of the component under (y, x)"""
        try:
            col = int(grid[y][x]); n = self._component_size(grid, y, x, cap=512)
            band = 0 if n <= 2 else (1 if n <= 8 else (2 if n <= 32 else (3 if n <= 128 else 4)))
            return (col, band)
        except Exception:
            return (-1, -1)

    def _v55_filter(self, pool, grid):
        """soft filters on the click pool, each skipped if it would empty the pool"""
        import random as _r
        p = list(pool)
        if not p or not grid:
            return p
        if getattr(self, '_v55_death', 0) and self._v55_dead:
            sg = self._v55_sig(grid)
            q = [c for c in p if (sg, ('C', tuple(c))) not in self._v55_dead]
            if q:
                p = q
        ctx = self._v55_ctx
        if getattr(self, '_v55_tprior', 0) > 0:
            T = {c: self._v55_type(grid, c[0], c[1]) for c in p}
            st = self._v55_tstats
            q = [c for c in p if not (st.get((T[c], ctx), (0, 0))[0] >= 2 and st.get((T[c], ctx), (0, 0))[1] == 0)]
            if q and len(q) < len(p):
                self._v55_beh['dead_type_skips'] += 1
            if q:
                p = q
            if _r.random() < min(1.0, self._v55_tprior):
                sc = {c: (st.get((T[c], ctx), (0, 0))[1] + 1.0) / (st.get((T[c], ctx), (0, 0))[0] + 2.0) for c in p}
                best = max(sc.values())
                q = [c for c in p if sc[c] >= best - 1e-9]
                if q:
                    p = q
        if getattr(self, '_v55_wprior', 0) > 0 and self._v55_worked and _r.random() < min(1.0, self._v55_wprior):
            q = [c for c in p if self._v55_type(grid, c[0], c[1]) in self._v55_worked]
            if q:
                p = q
        return p

    def _v55_step(self, frame, changed, whole):
        """per-step bookkeeping: behaviour counters, type statistics, deaths, level-ups"""
        b = self._v55_beh; b['steps'] += 1
        g = frame.grid
        if changed:
            b['effective'] += 1
        elif whole:
            b['border_only'] += 1
        else:
            b['nochange'] += 1
        sg = self._v55_sig(g)
        if sg in self._v55_sigs:
            b['repeats'] += 1
        else:
            self._v55_sigs.add(sg); b['distinct'] = len(self._v55_sigs)
        # the click statistics: (type, context) -> [clicks, changes]
        # the memories judge change on the INTERIOR (the budget bar excluded) even when the brain's own change_border
        # is 0 -- so a bred brain keeps its learning signal while its memories see the truth (v1.155 Kaggle runtime)
        ci = 1 if changed else 0                       # NAS 1155 behaviour (the memories follow change_border)
        if getattr(self, '_v55_mem_interior', 0):     # the Kaggle agent sets this: memories always see the interior
            try:
                ci = 0 if _v55_interior_equal(g, self.prev_grid, max(4, int(getattr(self, '_v55_border', 0) or 0))) else 1
            except Exception:
                pass
        lt = self._v55_last_type
        if lt is not None and getattr(self, '_last_action_was_click', False):
            v = self._v55_tstats.get(lt, [0, 0]); v[0] += 1; v[1] += 1 if ci else 0; self._v55_tstats[lt] = v
            if ci:
                self._v55_level_types.append(lt[0])
        self._v55_last_type = None
        self._v55_ctx = 1 if ci else 0
        lv = int(getattr(frame, 'levels_completed', 0) or 0)
        if lv > self._v55_lv:
            b['levelups'] += lv - self._v55_lv
            for t in self._v55_level_types:
                self._v55_worked[t] = self._v55_worked.get(t, 0) + 1
            self._v55_tstats = {}; self._v55_level_types = []; self._v55_border0 = None; self._v55_bunion = set()
            self._v55_lv = lv

    def v55_export(self):
        """the memories a FULL reset would wipe (Kaggle's agent full-resets after every game over)"""
        return (set(self._v55_dead), dict(self._v55_worked), {k: list(v) for k, v in self._v55_tstats.items()},
                list(self._v55_level_types))

    def v55_import(self, mem):
        try:
            d, w, t, lt = mem
            self._v55_dead = set(d); self._v55_worked = dict(w); self._v55_tstats = {k: list(v) for k, v in t.items()}
            self._v55_level_types = list(lt)
        except Exception:
            pass

    def v55_note_death(self):
        """v1.155: called by the player when the last action ended the attempt (the brain never sees that frame)"""
        try:
            self._v55_beh['deaths'] += 1
            if getattr(self, '_v55_death', 0) and self._v55_asig is not None and self._v55_akey is not None:
                self._v55_dead.add((self._v55_asig, self._v55_akey))
        except Exception:
            pass

    def _v55_budget_channels(self, grid):
        """+2: how much of the border band that ever moved this level now differs from the level start; moved now"""
        m = self._v55_m()
        try:
            H = len(grid); W = len(grid[0])
            cells = [(y, x) for y in range(H) for x in range(W) if y < m or y >= H - m or x < m or x >= W - m]
            cur = {c: grid[c[0]][c[1]] for c in cells}
            if self._v55_border0 is None:
                self._v55_border0 = cur; self._v55_bprev = cur
            moved_now = any(cur[c] != self._v55_bprev.get(c) for c in cells)
            for c in cells:
                if cur[c] != self._v55_border0.get(c):
                    self._v55_bunion.add(c)
            U = self._v55_bunion
            used = (sum(1 for c in U if cur[c] != self._v55_border0.get(c)) / len(U)) if U else 0.0
            self._v55_bprev = cur
            return [2.0 * used - 1.0, 1 if moved_now else -1]
        except Exception:
            return [0, 0]

    def _v55_comps(self, grid):
        """same-colour 4-connected components inside the border, excluding the most common colour"""
        m = self._v55_m(); H = len(grid); W = len(grid[0])
        cnt = {}
        for y in range(m, H - m):
            for v in grid[y][m:W - m]:
                cnt[v] = cnt.get(v, 0) + 1
        bg = max(cnt, key=cnt.get) if cnt else 0
        seen = set(); out = []
        for y0 in range(m, H - m):
            for x0 in range(m, W - m):
                if (y0, x0) in seen or grid[y0][x0] == bg:
                    continue
                col = grid[y0][x0]; stack = [(y0, x0)]; seen.add((y0, x0)); cs = []
                while stack:
                    y, x = stack.pop(); cs.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if m <= ny < H - m and m <= nx < W - m and (ny, nx) not in seen and grid[ny][nx] == col:
                            seen.add((ny, nx)); stack.append((ny, nx))
                n = len(cs)
                out.append((int(col), n, sum(c[0] for c in cs) / n, sum(c[1] for c in cs) / n))
        return out

    def _v55_object_channels(self, grid):
        """+6: object count band, moved, appeared, disappeared, recoloured, the clicked object changed in place"""
        try:
            cur = self._v55_comps(grid); prev = self._v55_prev_comps; self._v55_prev_comps = cur
            nb = 1 if len(cur) > 12 else (-1 if len(cur) <= 3 else 0)
            if prev is None:
                return [nb, -1, -1, -1, -1, -1]
            pu = list(prev); cu = []; moved = 0
            for c in cur:
                hit = next((p for p in pu if p[0] == c[0] and p[1] == c[1] and abs(p[2] - c[2]) < 0.5 and abs(p[3] - c[3]) < 0.5), None)
                if hit is not None:
                    pu.remove(hit)
                else:
                    cu.append(c)
            rest = []
            for c in cu:
                mv = next((p for p in pu if p[0] == c[0] and p[1] == c[1]), None)
                if mv is not None:
                    pu.remove(mv); moved += 1
                else:
                    rest.append(c)
            recol = 0; app = []
            for c in rest:
                rc = next((p for p in pu if p[1] == c[1] and p[0] != c[0] and abs(p[2] - c[2]) < 1.0 and abs(p[3] - c[3]) < 1.0), None)
                if rc is not None:
                    pu.remove(rc); recol += 1
                else:
                    app.append(c)
            clicked = -1
            lc = getattr(self, '_last_click_cand', None)
            if lc is not None and getattr(self, '_last_action_was_click', False) and (recol or moved or app):
                y, x = lc
                for c in rest + cu:
                    if abs(c[2] - y) <= 3 and abs(c[3] - x) <= 3:
                        clicked = 1; break
            return [nb, 1 if moved else -1, 1 if app else -1, 1 if pu else -1, 1 if recol else -1, clicked]
        except Exception:
            return [0, 0, 0, 0, 0, 0]

    def _v55_transient_channels(self):
        """+2: the last action animated (more than one grid); some pixel changed more than once inside it"""
        fr = getattr(self, '_v55_raw', None)
        if not fr or len(fr) < 2:
            return [-1, -1]
        try:
            idx = sorted(set(int(round(i * (len(fr) - 1) / 11.0)) for i in range(12))) if len(fr) > 12 else list(range(len(fr)))
            seq = [fr[i] for i in idx]
            H = len(seq[0]); W = len(seq[0][0]); m = self._v55_m()
            changes = {}
            for a, b in zip(seq, seq[1:]):
                for y in range(m, H - m):
                    ra, rb = a[y], b[y]
                    if ra == rb:
                        continue
                    for x in range(m, W - m):
                        if ra[x] != rb[x]:
                            changes[(y, x)] = changes.get((y, x), 0) + 1
            return [1, 1 if any(v >= 2 for v in changes.values()) else -1]
        except Exception:
            return [1, 0]

    def _click_candidates(self, grid) -> List[Tuple[int, int]]:
        """Cells worth clicking, derived from the frame -- no per-game knowledge.

        The transcripts show every click-bearing game has a handful of live targets: r11l has
        THREE structure nodes and a human cleared level 1 in six clicks. The brain samples
        4,096 cells uniformly, so 0.07% of its clicks can do anything.

        Background is the modal colour; non-background cells are grouped into 4-connected
        components; components within [min,max] cells contribute their centroid. Cached on the
        frame signature so the label pass runs once per distinct frame, not once per action.
        """
        if not grid or not grid[0]:
            return []
        # v1.172 (nas_1171_report.md section 20, change 1; BrainFinder 1.2): the cache was keyed on id(grid) -- no _frame_sig method exists --
        # so a later frame whose grid list landed at a freed address reused an old frame's click targets (the play depended on the
        # memory allocator). Keyed on the frame's CONTENT, as the docstring always said (BrainFinder's --click-cache content).
        sig = _frame_signature(grid)
        if getattr(self, '_cm_sig', None) == sig:
            return self._cm_cache
        # ============ v1.146: THE TARGET-DIFFERENCE CHANNEL (diff_mask) ============
        # The transcripts of tn36 (level 2+), cd82, sc25 and ft09 all say the same thing: one
        # region of the board is a TARGET and another is the WORK area, and the level is done
        # when they match. A 6x6 retina cannot see that; a reactive policy has no comparison.
        # diff_mask=1 splits the board at its dominant seam (the column or row whose two sides
        # differ most in colour histogram), maps every work cell onto its target twin, and makes
        # the candidates the work cells whose colour differs from the twin's. "Click the cell
        # that is wrong" is then a tap policy, which is the thing this brain can do. When the
        # split finds no seam (single-region boards), it returns [] and the ordinary mask runs.
        _dmode = int(getattr(self, '_diff_mask', 0) or 0)
        # v1.147: diff_mask=1 must never REPLACE a working level-1 mask (r1146: tn36-diff 0/235 because
        # tn36's level-1 board has an equal-size pair and the diff cells overrode the colour mask).
        # It now acts only at level >= 2, or when the ordinary mask finds no candidates.
        if _dmode == 1:
            _lvl = int(getattr(self, '_cur_level', 0) or 0)
            if _lvl < 1:
                # level 1: the ordinary mask ALWAYS -- even when it empties mid-level (tn36's circles
                # change colour as they are toggled; an empty mask there is normal play, and handing
                # those clicks to the diff cells killed every tn36 diff cell in r1146 and in 1147's
                # own check 21 run)
                self._diff_mask = 0
                try:
                    _base = self._click_candidates(grid) or []
                finally:
                    self._diff_mask = 1
                self._cm_sig = sig; self._cm_cache = _base
                return _base
        if _dmode == 1:                                   # diff only (falls through when no pair)
            try:
                dc = self._diff_candidates(grid)
                if dc:
                    self._cm_sig = sig; self._cm_cache = dc
                    return dc
            except Exception:
                pass
        elif _dmode == 2:                                 # union: diff cells + the ordinary mask
            try:
                dc = self._diff_candidates(grid)
            except Exception:
                dc = []
            if dc:
                self._diff_mask = 0
                try:
                    base = self._click_candidates(grid) or []
                finally:
                    self._diff_mask = 2
                merged = list(dict.fromkeys(list(dc) + list(base)))
                self._cm_sig = sig; self._cm_cache = merged
                return merged
        # ============ v1.118 AUTO SIZING ============
        # v1117 used one fixed bound per game. Every WINNING game landed 2-29 candidates; the
        # two disasters sat at 1 (vc33: every click to the same cell, 10.3% -> 0) and 180
        # (bp35: no reduction at all). Auto searches the bound for a count inside the band and
        # returns [] when nothing works, which makes the caller fall through to a normal click
        # -- so a game the mask cannot help is left as it was rather than broken.
        if getattr(self, '_click_mask', 'off') == 'auto':
            bg = self._bg_colour(grid)
            comps = self._components(grid, bg)
            lo, hi = getattr(self, '_cm_band', (2, 30))
            # ============ v1.119: BEST FIT, AND A CEILING THAT SCALES ============
            # v1118 took the FIRST bound inside the band and used a fixed ceiling of 30. Both
            # were wrong, and together they cost the run:
            #   * first-fit gave s5i5 10 candidates where the fixed bound used 18 (92.6% -> 6.1%)
            #   * a ceiling of 30 sits ONE ABOVE r11l's level-1 count of 29, so a level holding
            #     two structures exceeds it, auto returns nothing, and the click reverts to
            #     4,096 blind targets. r11l reached level 2 ZERO times with 7,889 actions spare.
            # The ceiling now scales with how many objects the frame holds, and the bound chosen
            # is the one landing NEAREST THE MIDDLE of the band rather than the first that fits.
            hi = max(hi, min(4 * max(len(comps), 1), 256))
            cands = []
            for _hi in (2, 4, 8, 16, 32, 64, 128, 4096):
                c = [(y, x) for (n, y, x) in comps if 1 <= n <= _hi]
                if lo <= len(c) <= hi:
                    cands.append(c)
            mid = (lo + hi) / 2.0
            best = min(cands, key=lambda c: abs(len(c) - mid)) if cands else []
            self._cm_sig = sig
            self._cm_cache = best
            self._cm_n = len(best)
            return best
        H = len(grid); W = len(grid[0])
        cnt: Dict[int, int] = {}
        for row in grid:
            for v in row:
                cnt[v] = cnt.get(v, 0) + 1
        bg = max(cnt.items(), key=lambda kv: kv[1])[0] if cnt else 0
        # ============ v1.120: THE GENE IS THE ONLY SOURCE OF TRUTH ============
        # This read `os.environ['MULTINXON_CLICK_MASK_MIN'/'_MAX']`. v1118 made the mask a gene
        # and renamed those to MULTINXON_PIN_*, updating the `auto` branch and NOT this one --
        # so from v1118 onward every fixed-mask machine silently fell back to the 1/64 defaults.
        #
        # It cost two whole runs. v1119's `r11l-x-base` was built to reproduce v1117's
        # 100%-clear-at-median-8 cell exactly and scored 15.1% at median 28, because it ran
        # 1-64 (34 candidates) instead of 1-4 (29). `r11l-x-auto` beat every fixed arm purely
        # because `auto` read the gene and the others did not.
        #
        # self._cm_lo / self._cm_hi are set in __init__ from the gene, with the env var as an
        # explicit override for experiments. There is now ONE path.
        lo = int(getattr(self, '_cm_lo', 1) or 1)
        hi = int(getattr(self, '_cm_hi', 64) or 64)
        seen = [[False] * W for _ in range(H)]
        out: List[Tuple[int, int]] = []
        for y0 in range(H):
            for x0 in range(W):
                if seen[y0][x0] or grid[y0][x0] == bg:
                    continue
                col = grid[y0][x0]
                stack = [(y0, x0)]; seen[y0][x0] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] \
                                and grid[ny][nx] == col:
                            seen[ny][nx] = True; stack.append((ny, nx))
                # v1.140: click_colour -- keep only components of ONE colour. The s5i5 trace
                # showed the 18-18 size mask losing the yellow button after the first tap (it
                # grows); 49 of 100 clicks fell to the (32,32) fallback. The transcript defines
                # the control by COLOUR ("the yellow button", "the blue circle"), and colour is
                # the invariant. -1 = off.
                _cc = int(getattr(self, '_click_colour', -1))
                if _cc >= 0 and col != _cc:
                    continue
                if lo <= len(cells) <= hi:
                    my = sum(c[0] for c in cells) // len(cells)
                    mx = sum(c[1] for c in cells) // len(cells)
                    out.append((my, mx))
        # ============ v1.125: PREFER CORE/SOCKET COLOURS ============
        # The r11l transcript: "Hollow COLOURED RINGS are sockets ... docking a MATCHING core
        # into its ring collapses that molecule." The objects that matter come in colour PAIRS;
        # the gray stems and satellite nodes are scaffolding and are numerous. So a colour
        # occupying few components is a core/socket colour, and one occupying fifteen is not.
        #
        # On r11l level 1 this takes 29 size-matched candidates down to 2 (K=2) or 6 (K=4) --
        # the purple core and its ring rather than the fifteen gray nodes.
        #
        # THE GUARD MATTERS AS MUCH AS THE MASK. v1118 gave vc33 a one-candidate mask and took
        # it from 10.3% to ZERO, because every click then lands on the same cell -- strictly
        # worse than random. r11l and sp80 and m0r0 all return exactly 2 here, which is the
        # danger zone. So the pair set is used ONLY if it holds at least 3 candidates;
        # otherwise the size-matched set stands. A game the mechanism cannot help is left
        # exactly as it was, which is also what happens on ft09, where no colour occupies
        # 8 or fewer components and the pair set is empty by construction.
        _pk = getattr(self, '_pair_mask', 0)
        if _pk and len(out) >= 3:
            _bycol: Dict[int, int] = {}
            for (n_, c_, _y, _x) in self._components_tagged(grid, bg):
                _bycol[c_] = _bycol.get(c_, 0) + 1
            _pair = []
            for (n_, c_, cy_, cx_) in self._components_tagged(grid, bg):
                if lo <= n_ <= hi and 2 <= _bycol.get(c_, 0) <= _pk:
                    _pair.append((cy_, cx_))
            if len(_pair) >= 3:
                out = _pair
        # ============ v1.154: RING CENTRES (8-connected hollow shapes) ============
        # r11l's goals are hollow diamonds whose pixels touch only DIAGONALLY: 4-connected labelling splits each into
        # twelve 1-px components, so the centre -- the expert's destination click, "docking a core into its ring" --
        # was never a candidate. NAS 1153: levels 3-6 had 0 clears in 12,917 cold tries; expressible.py showed every
        # r11l level failing at exactly those clicks. Gene cand_ring (env MULTINXON_CAND_RING) adds the centroid of
        # every same-colour cluster (pixels within 2 cells of each other) of >= 4 px whose centroid cell is NOT its colour
        # (it is hollow). 8-connectivity was not enough: the diamonds have 1-px gaps at their corners.
        _ring = os.environ.get('MULTINXON_CAND_RING')
        _ring = int(_ring) if _ring not in (None, '') else int(getattr(self, '_cand_ring', 0) or 0)
        if _ring:
            seen8 = [[False] * W for _ in range(H)]
            extra = []
            for y0 in range(H):
                for x0 in range(W):
                    if seen8[y0][x0] or grid[y0][x0] == bg:
                        continue
                    col = grid[y0][x0]; stack = [(y0, x0)]; seen8[y0][x0] = True; cells = []
                    while stack:
                        y, x = stack.pop(); cells.append((y, x))
                        for dy in (-2, -1, 0, 1, 2):          # same colour within 2 cells: r11l's diamonds are
                            for dx in (-2, -1, 0, 1, 2):      # four 3-px diagonal segments with 1-px gaps at the corners
                                ny, nx = y + dy, x + dx
                                if (dy or dx) and 0 <= ny < H and 0 <= nx < W and not seen8[ny][nx] and grid[ny][nx] == col:
                                    seen8[ny][nx] = True; stack.append((ny, nx))
                    if 4 <= len(cells) <= 400:
                        cy = sum(c[0] for c in cells) // len(cells); cx = sum(c[1] for c in cells) // len(cells)
                        if grid[cy][cx] != col:
                            extra.append((cy, cx))
            out = list(dict.fromkeys(list(out) + extra))
        # ============ v1.154: EDGE-CLAMPED CANDIDATES ============
        # r11l moves the selected piece CENTRED on the click and rejects a move that leaves the board or collides. A
        # collectible near an edge cannot be reached by clicking its own centroid; the expert clicks a little INWARD
        # (L5 (25,4), L6 (9,24), (5,34): every move expressible.py --logical could not reproduce was such an
        # absorption). Gene cand_edge m (env MULTINXON_CAND_EDGE): every candidate within m cells of an edge also gets
        # a copy clamped m cells inward.
        _edge = os.environ.get('MULTINXON_CAND_EDGE')
        _edge = int(_edge) if _edge not in (None, '') else int(getattr(self, '_cand_edge', 0) or 0)
        if _edge > 0:
            clamped = []
            for (cy, cx) in out:
                c2 = (min(max(cy, _edge), H - 1 - _edge), min(max(cx, _edge), W - 1 - _edge))
                if c2 != (cy, cx):
                    clamped.append(c2)
            out = list(dict.fromkeys(list(out) + clamped))
        self._cm_sig = sig; self._cm_cache = out
        return out

    def _cursor_click(self, h: int, w: int) -> Dict[str, Any]:
        cy, cx = self.cursor
        if h == 0 or w == 0:
            return {"x": 32, "y": 32}
        x = int((cx + 0.5) * 64 / w); y = int((cy + 0.5) * 64 / h)
        # v1.112: snap to the CENTRE of the lattice cell.  ACTION6 = click(x,y) is a no-op
        # anywhere but on a live cell (23 of 320 claims), and a click one pixel off the cell a
        # human would hit is a silent no-op THAT STILL COSTS AN ACTION (13 of 320).  Snapping
        # turns a 64x64 addressing problem into a (64/px)^2 one -- for sp80's 4px lattice that
        # is 4096 -> 256 distinct targets.
        _lp = getattr(self, "_lattice_px", 1)
        if _lp > 1:
            x = min(63, (x // _lp) * _lp + _lp // 2)
            y = min(63, (y // _lp) * _lp + _lp // 2)
        # v1.117 CLICK MASK. See _click_candidates. The lattice snap quantised POSITION; this
        # restricts to cells the frame says are LIVE, which is what the transcripts describe.
        # ============ v1.120: THE MODE COMES FROM THE GENE TOO ============
        # This read `os.environ['MULTINXON_CLICK_MASK']` directly, so a brain whose GENE said
        # `centroid` never fired the mask unless an env var happened to be set. That is the
        # same defect as the size bounds and it survived v1118 and v1119 undetected -- the
        # bounds bug was found by reading the code, this one by preflight check 7, which
        # compares the CLICK COORDINATE a brain emits with the gene off and on.
        #
        # self._click_mask is resolved once in __init__: env var if present (an experiment
        # override), otherwise the gene.
        _mode = getattr(self, '_click_mask', 'off')
        if _mode != 'off':
            cands = self._click_candidates(getattr(self, '_last_grid', None))
            if cands:
                self._cm_n = len(cands)
                # ============ v1.120: `auto` APPLIES ITS CANDIDATES ============
                # This read `if _mode in ('centroid', 'snap')`, so `auto` computed a candidate
                # list on every frame and THREW IT AWAY. It never moved a single click in v1118
                # or v1119 -- which means v1118's "auto broke r11l" and v1119's "auto beat every
                # fixed arm" were both comparisons against a mask that was not running.
                #
                # Three separate reads of the same mechanism were wrong at once: the size
                # bounds (env, renamed), the mode (env, not the gene), and this. Preflight
                # check 7 catches all three because it compares the EMITTED CLICK.
                if _mode in ('centroid', 'snap', 'auto'):
                    gy = y * (h or 64) / 64.0
                    gx = x * (w or 64) / 64.0
                    # ============ v1.135: SWITCH WHEN NOTHING CHANGES ============
                    # The click snaps to the candidate NEAREST the cursor, so a brain whose
                    # cursor sits by the wrong control clicks it again and again. That is the
                    # lp85 seed-3847 failure exactly: ~50% of near-perfect brains commit to the
                    # wrong plus on one first-click draw and never switch inside the cap. It
                    # survived a reward cliff, a gradient, a cold-pass objective, a reseed and a
                    # fresh subprocess, because it is not the harness -- it is this line.
                    # stuck_switch=N: after N consecutive actions with NO board change, the
                    # last-clicked candidate is excluded and the nearest OTHER one is taken.
                    # Game-agnostic: every select-then-move game has the same wrong-first-select
                    # failure, and every one of them has >= 2 candidates under its mask.
                    _pool = cands
                    # v1.140: noop_memory -- a SOFT bias, not a rule. Candidates whose last
                    # click changed nothing are skipped for the next N picks, unless every
                    # candidate is flagged. lp85's tail is 151 presses on a dead plus; this is
                    # the memory that stops the 20th press. It is soft on purpose: every hard
                    # deterministic rule tested on r11l scored 0/20 against the random cursor.
                    _nm = int(getattr(self, '_noop_memory', 0) or 0)
                    if _nm:
                        _bad = getattr(self, '_noop_until', {})
                        _t = int(getattr(self, '_click_t', 0))
                        _ok = [c for c in cands if _bad.get(c, -1) < _t]
                        if _ok:
                            _pool = _ok
                    # v1.141: productive_memory -- the INVERSE of noop_memory. s5i5's solution is
                    # five taps on ONE button and vc33's is repeated taps on one blue cell;
                    # noop_memory measurably hurt s5i5 (2.10x) because it suppresses exactly that.
                    # If any candidate's last click changed the board by more than the
                    # background, restrict the pick to the best such candidate for N picks.
                    _pm = int(getattr(self, '_productive_memory', 0) or 0)
                    if _pm:
                        _ps = getattr(self, '_prod_score', {}); _pu = getattr(self, '_prod_until', {})
                        _t = int(getattr(self, '_click_t', 0)); _fl = getattr(self, '_noop_floor', None) or 0
                        _good = [(c, _ps[c]) for c in _pool if c in _ps and _ps[c] > _fl + 1 and _pu.get(c, -1) >= _t]
                        if _good:
                            _best = max(v for _, v in _good)
                            _pool = [c for c, v in _good if v >= _best * 0.5]
                    _ss = int(getattr(self, '_stuck_switch', 0) or 0)
                    if _ss and len(cands) >= 2 and int(getattr(self, '_stuck_n', 0)) >= _ss:
                        _lc = getattr(self, '_last_click_cand', None)
                        _others = [c for c in cands if c != _lc]
                        if _others:
                            _pool = _others
                            self._stuck_switched = int(getattr(self, '_stuck_switched', 0)) + 1
                    if getattr(self, "_net_decides", False):
                        _pool = cands                      # v1.149: no hand pool overrides the network
                    if self._v55_filters_on():
                        _pool = self._v55_filter(_pool, getattr(self, '_last_grid', None))   # v1.155
                    my, mx = min(_pool, key=lambda c: (c[0] - gy) ** 2 + (c[1] - gx) ** 2)
                    self._last_click_cand = (my, mx)
                    _fr = int(getattr(self, '_click_free_r', 0) or 0)
                    if _fr > 0 and max(abs(my - gy), abs(mx - gx)) <= _fr:      # v1.159: near an object -> the brain's own point
                        my, mx = int(round(gy)), int(round(gx)); self._free_clicks = int(getattr(self, '_free_clicks', 0)) + 1
                    if self._v55_filters_on():
                        self._v55_last_type = (self._v55_type(getattr(self, '_last_grid', None), my, mx), self._v55_ctx)
                    x = int((mx + 0.5) * 64 / (w or 64))
                    y = int((my + 0.5) * 64 / (h or 64))
        return {"x": max(0, min(63, x)), "y": max(0, min(63, y))}

    def _coarse_state(self, grid) -> int:
        """v1.041 — a COARSE signature of the grid for the action memory. Finer
        than a single bucket but coarser than the exact frame, so near-identical
        level-2 states map to the same key and a solvable state is re-recognised.
        Built from a downsampled colour-region summary; granularity set by the
        state_coarsen gene (higher = coarser)."""
        if not grid or not grid[0]:
            return 0
        H = len(grid); W = len(grid[0])
        step = self._state_coarsen
        vals = []
        # sample a coarse grid of cells (every `step` rows/cols) -> tuple hash
        for r in range(0, H, step):
            row = grid[r]
            for c in range(0, W, step):
                vals.append(row[c] & 0x7)
        # fold into a stable integer
        h = 1469598103934665603
        for v in vals:
            h = (h ^ (v + 1)) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
        return h

    def _action_memory_recall(self, frame, playable):
        """v1.041 — if the current coarse state matches a remembered progress-
        producing action, return it (with its click data) to bias toward replaying
        a found solution. Returns (action, data) or None."""
        if self._am_scale <= 0.0 or not self._action_mem:
            return None
        key = self._coarse_state(frame.grid)
        rec = self._action_mem.get(key)
        if rec is None:
            return None
        # only fire probabilistically, scaled by strength and binding weight
        # v1.042: when at the frontier (L1 mastered, practising L1→L2), lean
        # harder on the remembered progress-action so bonus episodes push the
        # boundary instead of re-exploring.
        eff_scale = self._am_scale * (1.5 if self._at_frontier else 1.0)
        if self.rng.random() > min(0.95, eff_scale * rec.get('w', 1.0)):
            return None
        act_val = rec.get('action')
        for a in playable:
            if int(a.value) == act_val:
                return a, dict(rec.get('data') or {})
        return None

    def _action_memory_update(self, frame, action, data):
        """v1.041 — after each step, if PROGRESS occurred (a level was completed
        or the score rose) since the previous step, bind the PREVIOUS coarse
        state → the action that produced the progress, so it can be replayed.
        Also lightly decays unreinforced bindings."""
        if self._am_scale <= 0.0:
            return
        cur_levels = int(getattr(frame, "levels_completed", 0))
        progressed = cur_levels > self._am_prev_levels
        if progressed and self._am_prev_state is not None \
                and self._am_prev_action is not None:
            key = self._am_prev_state
            rec = self._action_mem.get(key)
            if rec is None:
                self._action_mem[key] = {'action': self._am_prev_action,
                                         'data': self._am_prev_data, 'w': 1.0}
            else:
                # reinforce (and switch to this action if it just worked)
                rec['action'] = self._am_prev_action
                rec['data'] = self._am_prev_data
                rec['w'] = min(4.0, rec.get('w', 1.0) + 1.0)
            # occasional decay of the rest so stale bindings fade
            if self._am_decay > 0.0 and self.rng.random() < 0.1:
                for k, r in list(self._action_mem.items()):
                    if k != key:
                        r['w'] *= (1.0 - self._am_decay)
                        if r['w'] < 0.05:
                            del self._action_mem[k]
        self._am_prev_levels = cur_levels
        self._am_prev_state = self._coarse_state(frame.grid)
        self._am_prev_action = int(action.value) if action is not None else None
        self._am_prev_data = dict(data) if data else None

    def _state_bucket(self, frame: FrameData) -> int:
        """v1.034 — a coarse hash of the current grid into one of B buckets, so
        visit-counts generalise across near-identical states. Uses the cheap
        grid signature folded into B buckets."""
        try:
            sig = _v55_interior_sig(frame.grid, int(getattr(self, '_v55_border', 0) or 0))
        except Exception:
            sig = 0
        return sig % self._se_buckets

    def _explore(self, frame: FrameData, h: int, w: int) -> Tuple[GameAction, Dict[str, Any]]:
        playable = [a for a in self._masked_avail(frame) if a is not GameAction.RESET]
        if not playable:
            return GameAction.RESET, {}

        # v1.034 structured exploration: prefer the action least-tried in the
        # current state-bucket, so the brain systematically covers the action
        # space (and can re-find a solution) instead of re-sampling randomly.
        if self._se_scale > 0.0:
            bucket = self._state_bucket(frame)
            # pick the playable action with the lowest visit count in this bucket
            # (ties broken randomly); weight randomness down as scale rises.
            counts = [(self._se_counts.get((bucket, int(a.value)), 0), a)
                      for a in playable]
            min_c = min(c for c, _ in counts)
            least = [a for c, a in counts if c == min_c]
            if self.rng.random() < self._se_scale:
                a = self.rng.choice(least)         # directed: under-tried action
            else:
                a = self.rng.choice(playable)      # keep some pure randomness
            self._se_counts[(bucket, int(a.value))] = \
                self._se_counts.get((bucket, int(a.value)), 0) + 1
            data: Dict[str, Any] = {}
            if a.is_complex() and h and w:
                # for clicks, choose a cell from a least-visited region too
                self.cursor = self._least_visited_cell(
                    bucket, h, w, grid=frame.grid)
                data = self._cursor_click(h, w)
            return a, data

        a = self.rng.choice(playable)
        data = {}
        if a.is_complex():
            if h and w:
                self.cursor = [self.rng.randint(0, h - 1), self.rng.randint(0, w - 1)]
            data = self._cursor_click(h, w)
        return a, data

    def _least_visited_cell(self, bucket: int, h: int, w: int,
                            grid=None) -> List[int]:
        """v1.034/v1.037 — pick a grid cell to click. Base behaviour: the cell
        the click-explorer has visited least (systematic coverage). v1.037 goal-
        conditioning: when a per-level success prototype exists, also prefer
        cells whose LOCAL features are furthest from the prototype — i.e. the
        parts of the board most 'unsolved' relative to what solving looked like —
        so exploration is directed toward the goal, not just spread evenly."""
        candidates = []
        for _ in range(6):
            ry = self.rng.randint(0, h - 1)
            rx = self.rng.randint(0, w - 1)
            visit = self._se_counts.get((bucket, 1000 + ry * w + rx), 0)
            candidates.append((ry, rx, visit))
        # goal term: distance of the CURRENT grid features to this level's
        # prototype (a scalar that rises when the board is far from solved). It
        # biases WHEN we push harder on coverage; the cell-level signal stays
        # coverage since we cannot evaluate per-cell effects without the engine.
        goal_push = 0.0
        if self._gce > 0.0 and grid is not None and self._gg_prototypes:
            proto = self._gg_prototypes.get(self._gg_levels)
            if proto is not None:
                feats = self._grid_features(grid)
                goal_push = sum((f - p) ** 2 for f, p in zip(feats, proto)) ** 0.5
        # rank candidates: least-visited first, but when far from goal
        # (goal_push large) prefer the most-novel cells more strongly.
        novelty_w = 1.0 + self._gce * min(2.0, goal_push)
        candidates.sort(key=lambda t: t[2] / novelty_w)
        ry, rx, visit = candidates[0]
        self._se_counts[(bucket, 1000 + ry * w + rx)] = visit + 1
        return [ry, rx]

    # ---- one full macro step ----------------------------------------------
    def play_step(self, frame: FrameData) -> Tuple[GameAction, Dict[str, Any]]:
        """perceive -> compute reward -> think -> act, updating internal state."""
        if getattr(self, "_v55_trans", 0):            # v1.155: the action's intermediate frames, before settling
            try:
                _fr = getattr(frame, 'frame', None)
                self._v55_raw = _fr if (isinstance(_fr, list) and _fr and isinstance(_fr[0], list)) else None
            except Exception:
                self._v55_raw = None
        if getattr(self, "_settled_view", 0):
            frame = _settled_frame(frame)
        # v1.077: record the AVAILABLE action set (union over the episode) so the
        # action-coverage bonus knows which actions the brain SHOULD be trying — this
        # is how we detect wa30's problem (ACTION5 available but never issued).
        try:
            for _a in (getattr(frame, 'available_actions', None) or []):
                self._avail_actions.add(getattr(_a, 'name', str(_a)))
        except Exception:
            pass
        ext = self.perceive(frame)
        score_delta = int(frame.score) - int(self.prev_score)
        _b55 = int(getattr(self, '_v55_border', 0) or 0)
        _whole = 0 if _grids_equal(frame.grid, self.prev_grid) else 1
        changed = (0 if _v55_interior_equal(frame.grid, self.prev_grid, _b55) else 1) if (_b55 > 0 and _whole) else _whole
        # v1.140: how many cells the last action changed (the click log and noop_memory read it)
        try:
            _pg = self.prev_grid; _g = frame.grid
            if changed and _pg and _g and _pg[0] and _g[0]:
                self._last_change_cells = sum(1 for y in range(min(len(_pg), len(_g)))
                                              for x in range(min(len(_pg[0]), len(_g[0]))) if _pg[y][x] != _g[y][x])
            else:
                self._last_change_cells = 0
        except Exception:
            self._last_change_cells = 0
        if _b55 > 0:                                   # v1.155: count only the interior (not the budget bar)
            try:
                self._last_change_cells = _v55_interior_count(self.prev_grid, frame.grid, _b55) if changed else 0
            except Exception:
                pass
        try:
            self._v55_step(frame, changed, _whole)
        except Exception:
            pass
        # ============ v1.140: THE CLICK-EFFECT LOG ============
        # Eleven hand-written click rules for r11l scored 0/20 against the random cursor,
        # because the mechanic was guessed from component statistics. This records what each
        # click DID: target, cells changed, and the centroid of the changed region. Over a run
        # it is a click-effect table the decode can be read from instead of guessed.
        try:
            _lc = getattr(self, '_last_click_cand', None)
            if _lc is not None and getattr(self, '_last_action_was_click', False):
                self._click_t = int(getattr(self, '_click_t', 0)) + 1
                _n = int(getattr(self, '_last_change_cells', 0))
                if len(self._click_log) < 60:
                    self._click_log.append((int(_lc[0]), int(_lc[1]), _n))
                # a no-op is a change no bigger than the episode's BACKGROUND change (the HUD
                # tick on r11l is 1 cell; lp85's meter is ~293 every step) -- adaptive, not 1
                _bgc = getattr(self, '_noop_floor', None)
                self._noop_floor = _n if _bgc is None else min(_bgc, _n)
                if getattr(self, '_noop_memory', 0) and _n <= (self._noop_floor + 1):
                    self._noop_until[_lc] = self._click_t + int(self._noop_memory)
                # v1.141: productive_memory bookkeeping
                if getattr(self, '_productive_memory', 0):
                    self._prod_score[_lc] = _n
                    if _n > (self._noop_floor + 1):
                        self._prod_until[_lc] = self._click_t + int(self._productive_memory)
                # v1.141: a click that CHANGED the board is a selection -> start the handoff countdown
                if getattr(self, '_after_select', 0) and _n > (self._noop_floor + 1):
                    self._after_select_n = int(self._after_select)
        except Exception:
            pass
        # v1.142: dir_memory bookkeeping -- flag the last D-pad direction if it changed nothing
        try:
            self._step_t = int(getattr(self, '_step_t', 0)) + 1
            _la2 = getattr(self, '_last_action_id', None)
            if getattr(self, '_dir_memory', 0) and _la2 in (1, 2, 3, 4):
                _fl = getattr(self, '_noop_floor', None) or 0
                if int(getattr(self, '_last_change_cells', 0)) <= _fl + 1:
                    self._dir_noop_until[_la2] = self._step_t + int(self._dir_memory)
        except Exception:
            pass
        # ============ v1.141: THE ACTION LOG -- every action, not only clicks ============
        # Eleven of the 25 games are D-pad games and the click log records clicks only. This
        # records (action, y, x, cells changed) for EVERY action; y,x = -1 for non-clicks.
        try:
            _la = getattr(self, '_last_action_id', None)
            if _la is not None:
                _lc2 = getattr(self, '_last_click_cand', None) if getattr(self, '_last_action_was_click', False) else None
                if len(self._action_log) < 80:
                    self._action_log.append((int(_la), int(_lc2[0]) if _lc2 else -1, int(_lc2[1]) if _lc2 else -1,
                                             int(getattr(self, '_last_change_cells', 0))))
        except Exception:
            pass
        # ============ v1.090 UPDATE PRE-CLEAR VALUE STATE ============
        # These traces are what content() reads to value cells before any clear. Cheap,
        # per-step, and no-op when the pre-clear genes are 0.
        if self._pcv_scale > 0.0:
            # decaying trace of recent score movement (bounded 0..~1)
            self._pcv_score_trace = 0.9 * self._pcv_score_trace + (1.0 if score_delta > 0 else 0.0)
            # controllability: did the LAST action change the world? credit the colour that
            # was under the cursor when we took it (approx: current cursor colour).
            if self._pcv_last_action_idx is not None:
                try:
                    cy, cx = self.cursor
                    col = int(frame.grid[cy][cx]) if (0 <= cy < len(frame.grid)
                          and 0 <= cx < len(frame.grid[0])) else None
                    if col is not None:
                        t = self._pcv_ctrl_tries.get(col, 0) + 1
                        prev = self._pcv_ctrl_by_action.get(col, 0.0)
                        # running mean of "changed" for actions taken on this colour
                        self._pcv_ctrl_by_action[col] = prev + (changed - prev) / t
                        self._pcv_ctrl_tries[col] = min(t, 200)
                except Exception:
                    pass
            # reachable-novelty: count coarse next-state signatures visited this life
            if self._pcv_novel > 0.0:
                try:
                    sig = self._coarse_state_sig(frame.grid) if hasattr(self, '_coarse_state_sig') \
                        else hash(tuple(frame.grid[0]))
                    self._pcv_reach_counts[sig] = self._pcv_reach_counts.get(sig, 0) + 1
                except Exception:
                    pass
            if self._pcv_ever_cleared:
                self._pcv_steps_since_clear += 1

        # ============ v1.091 UPDATE THE LEARNED WORLD-MODEL ============
        # Train the online forward model on the transition we just observed, and update the
        # per-action learning-progress + empowerment tables that _plan_action reads. Cheap,
        # per-step, and a no-op when world_model_scale is 0.
        if self._wm_scale > 0.0:
            try:
                self._wm_update(frame, changed)
            except Exception:
                pass
            if self._wm_ever_cleared:
                self._wm_steps_since_clear += 1
        # extrinsic reward (the true objective) — dominant when it fires.
        reward = 0.0 if getattr(self, '_teacher_only', False) else (float(score_delta) + 0.05 * changed)   # learntest teacher: native reward off
        # v1.014 dense intrinsic reward — a per-step learning signal derived
        # from grid dynamics, so the dopamine→STDP loop has something to learn
        # from between the (very rare) score events. No-op when _ir_scale==0.
        if self._ir_scale > 0.0:
            reward += self._intrinsic_reward(frame.grid, changed)
        # v1.032 learned success-proximity — the self-supervised within-level
        # gradient. Rewards moving the grid toward the learned success prototype,
        # giving the brain a climbable signal where the engine exposes none. No-op
        # when goal_gradient_scale==0.
        if self._gg_scale > 0.0:
            reward += self._goal_gradient_reward(frame.grid, frame)
        # v1.015 potential-based progress shaping (the goal-directed signal).
        # Reward Δφ when the env exposes within-level progress; this teaches the
        # brain to *move toward completing a level*, which novelty alone cannot.
        lp = float(getattr(frame, "level_progress", -1.0))
        if lp >= 0.0 and (self._progress_reward_scale > 0.0
                          or self._progress_penalty_scale > 0.0):
            dphi = lp - self._prev_progress
            if dphi > 0.0:
                reward += self._progress_reward_scale * dphi
            elif dphi < 0.0:
                # optional, usually-small penalty for moving AWAY from the goal
                reward += self._progress_penalty_scale * dphi
            self._prev_progress = lp
        self._play_steps += 1

        # v1.021 closed-loop self-regulation. Update the reward-rate trace and
        # re-regulate exploration + plasticity from the brain's OWN performance,
        # in real time — the noradrenergic/cholinergic feedback loop biology uses
        # instead of fixed set-points. This is what lets a single brain reliably
        # FIND then HOLD the solution rather than depending on a lucky fixed
        # trajectory (the source of the seed-to-seed level-2 variance).
        if self._autoreg > 0.0:
            inst = 1.0 if score_delta > 0 else 0.0
            alpha = 1.0 / self._autoreg_tau
            self._reward_rate = (1.0 - alpha) * self._reward_rate + alpha * inst
            if score_delta > 0:
                self._steps_since_reward = 0
            else:
                self._steps_since_reward += 1
            # "stuckness" in [0,1]: how far past patience we are with no reward
            stuck = min(1.0, self._steps_since_reward / float(self._autoreg_patience))
            # adaptive exploration: stuck → explore more (NE network reset);
            # reward flowing → exploit. Centred on the genome's base rate.
            drive = self._autoreg * (self._autoreg_explore_gain * stuck
                                     - 2.0 * self._reward_rate)
            ep = self._base_explore_prob + drive
            self.explore_probability = max(self._autoreg_explore_floor,
                                           min(self._autoreg_explore_ceil, ep))
            if getattr(self, "_net_decides", False):
                self.explore_probability = self._eval_explore      # v1.149: pinned
            # adaptive plasticity: stuck/surprised → learn more, stable → settle
            self._autoreg_plasticity_mult = 1.0 + self._autoreg * \
                self._autoreg_plasticity_gain * (stuck - self._reward_rate)
            self._autoreg_plasticity_mult = max(0.3, min(3.0,
                                                self._autoreg_plasticity_mult))

        outs = self.think(ext, reward=reward, score_reward=float(score_delta))

        # v1.027 experience replay & consolidation. Keep a rolling buffer of the
        # recent perceptions; when a LEVEL is completed (the rare, real success),
        # commit the trace that led to it into a persistent success-memory, then
        # replay stored winning perceptions with elevated plasticity so the
        # solution consolidates and becomes repeatable across later episodes.
        if self._replay_enabled:
            self._recent_inputs.append(ext)
            if len(self._recent_inputs) > self._replay_trace_len:
                self._recent_inputs.pop(0)
            cur_levels = int(getattr(frame, "levels_completed", 0))
            self._steps_since_replay += 1
            fresh_success = cur_levels > self._prev_levels
            if fresh_success:
                # store the sequence that just solved a level (what worked)
                for inp in self._recent_inputs:
                    self._success_memory.append(inp)
                    if len(self._success_memory) > self._replay_capacity:
                        self._success_memory.pop(0)
                self._prev_levels = cur_levels
            # consolidation event: on a fresh success, or every replay_interval
            do_replay = (self._success_memory and
                         (fresh_success or
                          (self._replay_interval > 0 and
                           self._steps_since_replay >= self._replay_interval)))
            if do_replay:
                self._steps_since_replay = 0
                self._replay_consolidate()

        # v1.017 planning: detect a PROGRESS EVENT (score gain, or a grid change
        # above threshold) and use it to (a) reinforce the goal latent toward
        # the motor direction that just worked, and (b) arm an action-commit
        # option. Otherwise the latent decays. This is computed before act() so
        # the bias applies to the current decision.
        planning_on = (self._goal_persist_scale > 0.0 or self._commit_scale > 0.0)
        if planning_on:
            # fraction of cells changed (cheap structuredness of the change)
            frac_changed = 0.0
            if changed and self.prev_grid:
                diff = total = 0
                for ra, rb in zip(frame.grid, self.prev_grid):
                    for ca, cb in zip(ra, rb):
                        total += 1
                        diff += 1 if ca != cb else 0
                frac_changed = diff / total if total else 0.0
            progress_event = (score_delta > 0) or (frac_changed > self._progress_event_thr
                                                   and frac_changed > 0.0)
            dy, dx = self._last_dir
            if progress_event:
                # credit the direction that worked into the goal latent
                self._goal_dy = self._goal_persist_decay * self._goal_dy + dy
                self._goal_dx = self._goal_persist_decay * self._goal_dx + dx
                # arm/refresh the commit option toward this direction
                if self._commit_scale > 0.0 and self._commit_steps > 0:
                    self._commit_action = (dy, dx)
                    self._commit_remaining = self._commit_steps
            else:
                # no progress: decay the latent and wind down any commitment
                self._goal_dy *= self._goal_persist_decay
                self._goal_dx *= self._goal_persist_decay
                if self._commit_remaining > 0:
                    self._commit_remaining -= 1
                    if self._commit_remaining <= 0:
                        self._commit_action = None

        # v1.041 ACTION-SEQUENCE MEMORY. Before committing to the brain's chosen
        # action, check whether the current (coarse) state matches a remembered
        # progress-producing action and, if so, replay it — so a level-2 crossing
        # found once is reproduced instead of rediscovered. Then record the
        # progress→action binding after the step. Honour an in-progress commit.
        action, data = self.act(outs, frame)
        self._last_mech = 'brain_act'        # v1.044: default — brain chose it
        if getattr(self, "_effect_fovea", 0) and action == GameAction.ACTION6:
            _c = getattr(self, "_last_click_cand", None)
            if _c is not None:
                self._fov_clicks[_c] = self._fov_clicks.get(_c, 0) + 1
                self._fov_last_click = _c
        if self._am_scale > 0.0 and not getattr(self, "_net_decides", False):
            if self._am_commit_left > 0 and self._am_commit_action is not None:
                for a in self._masked_avail(frame):
                    if int(a.value) == self._am_commit_action:
                        action = a
                        break
                self._am_commit_left -= 1
                self._last_mech = 'action_memory'
            else:
                playable = [a for a in self._masked_avail(frame)
                            if a is not GameAction.RESET]
                recalled = self._action_memory_recall(frame, playable)
                if recalled is not None:
                    action, data = recalled
                    self._last_mech = 'action_memory'
                    if self._commit_len > 0:
                        self._am_commit_action = int(action.value)
                        self._am_commit_left = self._commit_len
            self._action_memory_update(frame, action, data)

        # v1.044 mid-game brain-region telemetry: record per-sphere activation,
        # which control pathway chose the action, and the current level, so the
        # analyzer can see which parts of the Multi-Neuraxon brain do what.
        try:
            sphere_acts = {}
            for sname, sph in self.brain.spheres.items():
                try:
                    states = sph.network.get_output_states()
                    if states:
                        sphere_acts[sname] = sum(1 for s in states if s) / max(1, len(states))
                except Exception:
                    pass
            mech = getattr(self, '_last_mech', 'brain_act')
            self.telemetry.note_region(
                sphere_acts, mech, int(getattr(frame, 'levels_completed', 0)),
                motor_active=(data is not None and bool(data)),
                explored=(self.idle_steps == 0))
        except Exception:
            pass

        # bookkeeping
        if changed:
            self.idle_steps = 0
        else:
            self.idle_steps += 1
        self.prev_grid = [row[:] for row in frame.grid]
        self.prev_score = int(frame.score)
        # ============ v1.126: track the SELECT -> MOVE cycle ============
        # A select-type action is the one that takes control of a piece: ACTION6 (click) on
        # r11l/sp80/lp85, ACTION5 (spacebar) on cd82/sp80. Everything else is a move.
        # `changed` is already computed above against prev_grid, so "did the select work" costs
        # nothing extra -- and it is the signal that separates picking a block up from clicking
        # empty space, which is the whole difficulty of these games.
        # v1.135: the stuck counter -- consecutive actions with no board change
        if getattr(self, '_stuck_switch', 0):
            self._stuck_n = 0 if changed else int(getattr(self, '_stuck_n', 0)) + 1
        if getattr(self, '_select_phase', 0):
            try:
                _an = getattr(action, 'name', str(action))
            except Exception:
                _an = ''
            _is_sel = _an in ('ACTION5', 'ACTION6')
            self._sp_last_sel = 1.0 if _is_sel else 0.0
            if _is_sel:
                self._sp_since = 0
                self._sp_worked = 1.0 if changed else 0.0
            else:
                self._sp_since = min(99, int(getattr(self, '_sp_since', 0)) + 1)
        return action, data

    def _grid_features(self, grid: List[List[int]]) -> List[float]:
        """v1.032 — a compact, scale-invariant feature vector of a grid state,
        used to learn a self-supervised goal gradient. No privileged info: just
        observable structure that tends to differ between a mid-puzzle mess and a
        near-solved configuration. Features (all normalised ~[0,1]):
          - colour histogram (fraction of cells of each colour, 0..9)
          - fill fraction (non-zero cells)
          - horizontal + vertical symmetry scores
          - edge density (fraction of adjacent cells that differ)
        Cheap and bounded so it can run every step.
        """
        if not grid or not grid[0]:
            return [0.0] * 14
        H = len(grid)
        W = len(grid[0])
        total = H * W
        hist = [0] * 10
        diff_adj = 0
        adj_total = 0
        nonzero = 0
        for r in range(H):
            row = grid[r]
            for c in range(W):
                v = row[c]
                if 0 <= v <= 9:
                    hist[v] += 1
                if v != 0:
                    nonzero += 1
                if c + 1 < W:
                    adj_total += 1
                    if row[c] != row[c + 1]:
                        diff_adj += 1
                if r + 1 < H:
                    adj_total += 1
                    if grid[r][c] != grid[r + 1][c]:
                        diff_adj += 1
        feats = [h / total for h in hist]           # 10 colour fractions
        feats.append(nonzero / total)               # fill fraction
        # horizontal symmetry (left-right mirror match fraction)
        hsym = 0
        for r in range(H):
            row = grid[r]
            for c in range(W // 2):
                if row[c] == row[W - 1 - c]:
                    hsym += 1
        feats.append(hsym / max(1, H * (W // 2)))
        # vertical symmetry (top-bottom mirror match fraction)
        vsym = 0
        for r in range(H // 2):
            for c in range(W):
                if grid[r][c] == grid[H - 1 - r][c]:
                    vsym += 1
        feats.append(vsym / max(1, (H // 2) * W))
        feats.append(diff_adj / max(1, adj_total))  # edge density
        return feats

    def _goal_gradient_reward(self, grid, frame) -> float:
        """v1.033 — PER-LEVEL learned goal gradient. Each level keeps its own
        success prototype, so the brain climbs toward "what solving the CURRENT
        level looks like" — distinctly for level 1, level 2, etc. When it
        completes level N (levels_completed N→N+1) the recent approach is folded
        into level N's prototype; while on level N it is rewarded for moving its
        grid features toward level N's prototype (potential-based shaping). All
        prototypes are life-persistent, so a level solved once keeps guiding its
        own crossing in every later episode — and crucially the level-2 gradient
        is learned separately from level-1, instead of being swamped by the much
        more common level-1 completions."""
        if self._gg_scale <= 0.0:
            return 0.0
        feats = self._grid_features(grid)
        self._gg_recent_feats.append(feats)
        if len(self._gg_recent_feats) > self._gg_trace_len:
            self._gg_recent_feats.pop(0)
        cur_levels = int(getattr(frame, "levels_completed", 0))
        # v1.090 PARTIAL-PROGRESS SEEDING. The 1.089 prototype seeded ONLY on a completion
        # (cur_levels > _gg_levels), so on a game we never clear the prototype was never
        # created and the goal gradient was 0.0 forever. Also fold the recent approach into
        # the CURRENT level's prototype on a smaller progress event — a score increase or a
        # large frame change — weighted by partial_progress_seed_weight (0 == 1.089). This
        # gives a never-cleared game SOMETHING to climb toward.
        if self._partial_seed_w > 0.0 and cur_levels == self._gg_levels:
            _sc_up = int(getattr(frame, "score", 0)) > int(getattr(self, "prev_score", 0))
            _big_change = False
            try:
                if self.prev_grid and len(self.prev_grid) == len(grid):
                    _d = sum(1 for r in range(len(grid)) for c in range(len(grid[0]))
                             if grid[r][c] != self.prev_grid[r][c])
                    _big_change = _d > (0.02 * len(grid) * len(grid[0]))
            except Exception:
                pass
            if (_sc_up or _big_change) and self._gg_recent_feats:
                m = len(self._gg_recent_feats)
                approx = [sum(f[i] for f in self._gg_recent_feats) / m
                          for i in range(len(self._gg_recent_feats[0]))]
                lvl = cur_levels
                w = self._partial_seed_w
                if lvl not in self._gg_prototypes:
                    self._gg_prototypes[lvl] = list(approx)
                    self._gg_proto_n[lvl] = 1
                else:
                    self._gg_prototypes[lvl] = [(1 - w) * p + w * a
                                                for p, a in zip(self._gg_prototypes[lvl], approx)]
        # a completion: fold the approach into the prototype for the level we
        # were ON (i.e. the level we just advanced past = self._gg_levels)
        if cur_levels > self._gg_levels:
            lvl = self._gg_levels            # the level that was just solved
            m = len(self._gg_recent_feats)
            succ = [sum(f[i] for f in self._gg_recent_feats) / m
                    for i in range(len(feats))]
            if lvl not in self._gg_prototypes:
                self._gg_prototypes[lvl] = list(succ)
                self._gg_proto_n[lvl] = 1
            else:
                a = self._gg_decay
                self._gg_prototypes[lvl] = [(1 - a) * p + a * s
                                            for p, s in zip(self._gg_prototypes[lvl], succ)]
                self._gg_proto_n[lvl] = min(self._gg_memory,
                                            self._gg_proto_n[lvl] + 1)
            self._gg_levels = cur_levels
            self._gg_prev_dist = None         # reset baseline for the new level
            self._pcv_ever_cleared = True     # v1.090: start annealing the pre-clear value
            self._pcv_steps_since_clear = 0
            self._wm_ever_cleared = True      # v1.091: start annealing the world-model gradient
            self._wm_steps_since_clear = 0
            return 0.0
        # climb toward the CURRENT level's prototype (if learned yet)
        proto = self._gg_prototypes.get(cur_levels)
        if proto is None:
            return 0.0
        dist = sum((f - p) ** 2 for f, p in zip(feats, proto)) ** 0.5
        shaped = 0.0
        if self._gg_prev_dist is not None:
            shaped = self._gg_prev_dist - dist
        self._gg_prev_dist = dist
        return self._gg_scale * shaped

    def _intrinsic_reward(self, grid: List[List[int]], changed: int) -> float:
        """v1.014 dense reward: count-based novelty + structured-change size.

        novelty: a grid state seen for the n-th time is worth
        0.5**(n / half_life) — first visit ≈ full, heavily-revisited ≈ 0. This
        rewards reaching genuinely new configurations (purposeful exploration)
        and starves the oscillate-between-two-states pathology that plain
        "any change" reward (and the old +0.05) happily fed.

        structured change: the *fraction of cells* that changed vs the previous
        grid, so a substantive edit is worth more than 1-cell jitter.
        """
        sig = _v55_interior_sig(grid, int(getattr(self, '_v55_border', 0) or 0))
        n = self._visit_counts.get(sig, 0)
        self._visit_counts[sig] = n + 1
        novelty = 0.5 ** (n / self._ir_halflife)

        structured = 0.0
        if changed and self._ir_structured > 0.0 and self.prev_grid:
            diff = 0
            total = 0
            for ra, rb in zip(grid, self.prev_grid):
                for ca, cb in zip(ra, rb):
                    total += 1
                    if ca != cb:
                        diff += 1
            if total:
                structured = diff / total

        intrinsic = (self._ir_novelty * novelty
                     + self._ir_structured * structured)
        # optional anneal so intrinsic drive fades and extrinsic reward leads
        # late in a long episode (1.0 = no anneal).
        if self._ir_decay < 1.0:
            intrinsic *= self._ir_decay ** (self._play_steps / 1000.0)
        return self._ir_scale * intrinsic


def _grids_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True


# --------------------------------------------------------------------------- #
#  Telemetry (drives NAS fitness shaping)
# --------------------------------------------------------------------------- #
class _Telemetry:
    """Tracks behavioural signals used by the NAS fitness function, plus (v1.044)
    mid-game brain-region behaviour so we can see WHICH parts of the Multi-Neuraxon
    brain are doing what, and how that differs by level."""

    def __init__(self):
        self.steps = 0
        self.explored_steps = 0
        self.motor_active_steps = 0
        # sensorimotor coupling: correlate "sensory changed" with "motor changed"
        self._prev_ch: Optional[List[int]] = None
        self._prev_mo: Optional[List[int]] = None
        self._sm_pairs = 0          # both changed
        self._s_changes = 0
        self._m_changes = 0
        self._co_changes = 0
        # v1.044 mid-game brain-region behaviour counters
        self._sphere_act = {}        # sphere name -> summed mean activation
        self._sphere_n = 0           # samples folded into sphere_act
        self._mech_counts = {        # how often each control pathway DECIDED the action
            'action_memory': 0, 'goal_gradient': 0,
            'structured_explore': 0, 'brain_act': 0}
        # per-level behaviour: level -> {'steps','explore','mech':{...},'motor'}
        self._by_level = {}

    def note(self, channels: List[int], motor: List[int],
             motor_active: bool, explored: bool):
        self.steps += 1
        if explored:
            self.explored_steps += 1
        if motor_active:
            self.motor_active_steps += 1
        if self._prev_ch is not None and self._prev_mo is not None:
            s_changed = channels != self._prev_ch
            m_changed = motor != self._prev_mo
            if s_changed:
                self._s_changes += 1
            if m_changed:
                self._m_changes += 1
            if s_changed and m_changed:
                self._co_changes += 1
        self._prev_ch = list(channels)
        self._prev_mo = list(motor)

    def note_region(self, sphere_activations: Dict[str, float], mech: str,
                    level: int, motor_active: bool, explored: bool) -> None:
        """v1.044 — record which brain spheres were active, which control pathway
        chose the action ('action_memory'/'goal_gradient'/'structured_explore'/
        'brain_act'), and the current level, so the analyzer can attribute
        behaviour to brain regions and to the level the brain is stuck at."""
        if sphere_activations:
            for name, act in sphere_activations.items():
                self._sphere_act[name] = self._sphere_act.get(name, 0.0) + float(act)
            self._sphere_n += 1
        if mech in self._mech_counts:
            self._mech_counts[mech] += 1
        lv = self._by_level.setdefault(int(level), {
            'steps': 0, 'explore': 0, 'motor': 0,
            'mech': {'action_memory': 0, 'goal_gradient': 0,
                     'structured_explore': 0, 'brain_act': 0}})
        lv['steps'] += 1
        if explored:
            lv['explore'] += 1
        if motor_active:
            lv['motor'] += 1
        if mech in lv['mech']:
            lv['mech'][mech] += 1

    def region_summary(self) -> Dict[str, Any]:
        """v1.044 — compact summary of brain-region behaviour for the metrics."""
        out: Dict[str, Any] = {}
        if self._sphere_n > 0:
            for name, tot in self._sphere_act.items():
                out['sphere_act_' + name] = tot / self._sphere_n
        tot_mech = sum(self._mech_counts.values()) or 1
        for k, v in self._mech_counts.items():
            out['mech_frac_' + k] = v / tot_mech
        # level where the brain spent the most steps (where it is 'stuck')
        if self._by_level:
            busiest = max(self._by_level.items(), key=lambda kv: kv[1]['steps'])
            out['busiest_level'] = float(busiest[0])
            # at the busiest level, the exploration vs goal-directed split
            lv = busiest[1]
            s = lv['steps'] or 1
            out['busiest_explore_frac'] = lv['explore'] / s
            out['busiest_goalgrad_frac'] = lv['mech']['goal_gradient'] / s
            out['busiest_actionmem_frac'] = lv['mech']['action_memory'] / s
        return out

    def sensorimotor_corr(self) -> float:
        """Jaccard-like coupling of sensory and motor change events (0..1)."""
        union = self._s_changes + self._m_changes - self._co_changes
        if union <= 0:
            return 0.0
        return self._co_changes / union

    def explore_rate(self) -> float:
        return self.explored_steps / max(1, self.steps)

    def motor_active_rate(self) -> float:
        return self.motor_active_steps / max(1, self.steps)


# ============ v1.160: THE STEERING LAYER wraps the brain's step and lifecycle (inert when self._steer is None) ============
_v1160_play_step = ArcBrain.play_step
_v1160_reset = ArcBrain.reset
_v1160_soft_reset = ArcBrain.soft_reset
_v1160_note_death = getattr(ArcBrain, 'v55_note_death', None)


def _v1160_ps(self, frame):
    a, d = _v1160_play_step(self, frame)
    st = getattr(self, '_steer', None)
    if st is None:
        return a, d
    try:
        a2, d2 = st.decide(self, frame, a, d)
    except Exception as _e:
        self._steer_error = f'{type(_e).__name__}: {_e}'[:200]
        return a, d
    if a2 is not a or (d2 or {}) != (d or {}):
        self.last_action = a2                                     # the brain's bookkeeping follows the executed action
        try:
            _v = int(getattr(a2, 'value', 0))
            self._eff_prev_idx = _v - 1 if 1 <= _v <= 6 else -1
            if _v == 6 and d2:
                self._last_click_cand = (int(d2.get('y', 0)), int(d2.get('x', 0)))
        except Exception:
            pass
    return a2, d2


def _v1160_rs(self, frame):
    out = _v1160_reset(self, frame)
    st = getattr(self, '_steer', None)
    if st is not None:
        try:
            st.on_reset(frame)
        except Exception:
            pass
    return out


def _v1160_srs(self, frame):
    out = _v1160_soft_reset(self, frame)
    st = getattr(self, '_steer', None)
    if st is not None:
        try:
            st.on_reset(frame)
        except Exception:
            pass
    return out


def _v1160_death(self, *a, **k):
    st = getattr(self, '_steer', None)
    if st is not None:
        st.on_death()
    return _v1160_note_death(self, *a, **k) if _v1160_note_death else None


ArcBrain.play_step = _v1160_ps
ArcBrain.reset = _v1160_rs
ArcBrain.soft_reset = _v1160_srs
if _v1160_note_death is not None:
    ArcBrain.v55_note_death = _v1160_death


# ============ v1.161: STEERING v2 hooks -- the sense, observe-before-encode, teaching ============
def _v1161_sense(self):
    st = getattr(self, '_steer', None)
    try:
        if st is not None and hasattr(st, 'sense'):
            v = [float(x) for x in st.sense()]
            _ss = int(getattr(self, '_steer_sense', 0) or 0)
            _w = 16 if _ss >= 3 else (12 if _ss >= 2 else 8)                  # v1.162: 12 wide; v1.163: 16 (+ the goal)
            return (v + [0.0] * _w)[:_w]
    except Exception:
        pass
    _ss = int(getattr(self, '_steer_sense', 0) or 0)
    return [0.0] * (16 if _ss >= 3 else (12 if _ss >= 2 else 8))


ArcBrain._v1161_sense = _v1161_sense
_v1161_play_step = ArcBrain.play_step                   # the 1160 wrapper (brain proposes -> steering decides)


def _v1161_ps(self, frame):
    st = getattr(self, '_steer', None)
    if st is not None and hasattr(st, 'observe_frame'):
        try:
            st.observe_frame(frame)                     # learn from the last action BEFORE the brain encodes this frame
        except Exception as _e:
            self._steer_error = f'{type(_e).__name__}: {_e}'[:200]
    return _v1161_play_step(self, frame)


ArcBrain.play_step = _v1161_ps
_v1161_think = ArcBrain.think


def _v1161_th(self, ext_inputs, reward=0.0, score_reward=0.0):
    t = float(getattr(self, '_steer_teach_pending', 0.0) or 0.0)
    if t:
        self._steer_teach_pending = 0.0
    return _v1161_think(self, ext_inputs, reward=float(reward) + t, score_reward=score_reward)


ArcBrain.think = _v1161_th


# ============ v1.164: BRAIN-DRIVEN EXPLORATION (brain1164.py, gene net_explore): when the network proposes nothing, the
# exploratory action comes from its own motor potentials, not from the seed. Inert at net_explore = 0. ============
try:
    import brain1164 as _b164
    _b164.install(ArcBrain)
except Exception as _e164:                                                      # pragma: no cover
    print(f'  ! v1.164 brain1164 not installed: {type(_e164).__name__}: {_e164}', flush=True)
    raise


# ============ v1.165: UNIVERSAL CAPABILITY (cap1165.py): every brain can press every key and click anywhere the game offers;
# the bred action mask becomes a preference, the live set is learned from the board, the motor outputs are wired. Inert
# unless cap_univ = 1 / MULTINXON_CAP_UNIV = 1 (the loader's wiring: MULTINXON_CAP_WIRE > 0). ============
try:
    import cap1165 as _c165
    _c165.install(ArcBrain)
    _c165.install_loader()
except Exception as _e165:                                                      # pragma: no cover
    print(f'  ! v1.165 cap1165 not installed: {type(_e165).__name__}: {_e165}', flush=True)
    raise


# ============ v1.166: the LIVE SET DECAYS (cap1166.py): an opened action that stops changing the board leaves the play's live
# set after cap_decay uses (gene; the cell knob cap_decay is the default). Inert unless the capability is on and cap_decay > 0. ============
try:
    import cap1166 as _c166
    _c166.install(ArcBrain)
except Exception as _e166:                                                      # pragma: no cover
    print(f'  ! v1.166 cap1166 not installed: {type(_e166).__name__}: {_e166}', flush=True)
    raise


# ============ v1.167: the STRUCTURED GOAL (goal1167.py, research R3): structural predicates learned from the game's finished
# levels join Steer4's goal path under gene steer_goal_struct (inert at 0); FRONTIER STEERING (frontier1167.py, change 6): a
# graph of the play's boards, a stall, a route to the least-visited frontier (RESET first when only the start reaches one),
# then the brain proposes again (gene steer_frontier, inert at 0). The frontier wraps play_step OUTERMOST. ============
try:
    import steer1163 as _s163_67
    import goal1167 as _g167
    _g167.install(_s163_67.Steer4)
    import frontier1167 as _f167
    _f167.install(ArcBrain)
except Exception as _e167:                                                      # pragma: no cover
    print(f'  ! v1.167 goal1167 / frontier1167 not installed: {type(_e167).__name__}: {_e167}', flush=True)
    raise


# ============ v1.168: the steering's VETO and PLAN as genes (steer1168.py, change 7: steer_veto_th, steer_plan_w; the 1167 rule
# at their defaults); the structured goal CHOOSES among alternatives when the brain's proposal does nothing for h (goal1168.py,
# change 5, gene steer_goal_choose: LAB ONLY until it wins); the EARLY FRONTIER of FrameLab 6 (frontier1168.py: frontier_stall
# 1-16, the stall of a forced frontier from MULTINXON_FRONTIER_STALL_DEFAULT, a coarser board key when the graph finds no
# frontier). Inert at the defaults. ============
try:
    import steer1168 as _s168
    _s168.install()
    import goal1168 as _g168
    _g168.install(_s163_67.Steer4)
    import frontier1168 as _f168
    _f168.install()
except Exception as _e168:                                                      # pragma: no cover
    print(f'  ! v1.168 steer1168 / goal1168 / frontier1168 not installed: {type(_e168).__name__}: {_e168}', flush=True)
    raise


# ============ v1.169: the goal CHOOSES BY ITS MODEL'S PREDICTION (goal1169.py, change 7: genes steer_goal_how / steer_goal_when /
# steer_goal_stay -- M1's predicted board scores every candidate's change of h; the WHEN is goal1168's or the model's; h counts the
# colours that must stay). LAB ONLY until a learned arm wins; inert at the defaults. ============
try:
    import goal1169 as _g169
    _g169.install(_s163_67.Steer4)
except Exception as _e169:                                                      # pragma: no cover
    print(f'  ! v1.169 goal1169 not installed: {type(_e169).__name__}: {_e169}', flush=True)
    raise


# ============ v1.170: THE FRAMELAB GOAL IN THE FLEET (goal1170.py, nas_1169_report.md §19.7 + FrameLab 7-11): gene steer_goal_fl
# turns on the learned structured goal with stays, FrameLab's rule (WHEN the model does not predict the proposal lowers h, HOW the
# candidate with the best PREDICTED drop of h), no observed-mean term, no veto, and the SEEN-SIGNATURE model (exact for a seen
# (board, action), a replay of a seen signature's cell changes, M1 as the fallback); steer_goal_support is the precision floor.
# A bandit arm in the fleet (evo1170); inert at 0. ============
try:
    import goal1170 as _g170
    _g170.install(_s163_67.Steer4)
except Exception as _e170:                                                      # pragma: no cover
    print(f'  ! v1.170 goal1170 not installed: {type(_e170).__name__}: {_e170}', flush=True)
    raise


# ============ v1.172: CANDIDATE COVERAGE TIERS AND THE PER-LEVEL KEY SWEEP (wall1172.py; nas_1171_report.md section 20, change 4;
# FrameLab 15's movecheck): gene cand_tier 0 / 1 / 2 (today's candidates / + every object / + a 4-px grid, in the steering and in the
# brain's click mask) and gene commit_sweep 0 / 1 (after commit_stall steps without a new screen, each key not yet pressed in this
# level is pressed once). Inert at 0; the sweep wraps play_step OUTERMOST. ============
try:
    import os as _os172
    if _os172.environ.get('MULTINXON_NO_WALL1172', '') != '1':      # preflight1172 check 150 only: the 1171 brain, for the inertness test
        import wall1172 as _w172
        _w172.install(ArcBrain)
except Exception as _e172:                                                      # pragma: no cover
    print(f'  ! v1.172 wall1172 not installed: {type(_e172).__name__}: {_e172}', flush=True)
    raise


# ============ v1.173: THREE GENES OF UNDERSTANDING (effect1173.py; nas_1172_report.md section 23 change 9 + the user's decisions of
# 21 Oct 2026): effect_mem 0 / 1 / 2 (the effect table always kept; 1 known loops yield to an untried candidate, 2 + the two-step
# look through SEEN effects when there is a goal), warm_prefix 0 / 1 / 2 (the steering's memories learn the verified prefix's moves
# of the level; 2 + the network watches the last of them), game_sense 0 / 1 (the nearest known game from the board's pixels and the
# feedback of the first two actions; another game's priors when it is not the home game). Inert at 0. ============
try:
    import os as _os173
    if _os173.environ.get('MULTINXON_NO_EFFECT1173', '') != '1':     # preflight1173: the 1172 brain, for the inertness test
        import effect1173 as _e173
        _e173.install(_s163_67.Steer4)
except Exception as _e173x:                                                      # pragma: no cover
    print(f'  ! v1.173 effect1173 not installed: {type(_e173x).__name__}: {_e173x}', flush=True)
    raise


# ============ v1.174: GAME MEMORY, A STRICTER GOAL, THE GOAL AUDIT, THE HOME GAMES, THE HIDDEN-GAME SENSE (effect1174.py;
# nas_1173_report.md section 25 changes 2, 5, 7 as revised by FrameLab 18): warm_prefix 3 (the earlier levels' verified moves teach their
# effects, not the graph), goal_strict 0 / 1 / 2 (the goal map keeps fewer, surer cells), the goal audit (open-cell precision and recall
# against the verified finished board: tallies only), home_games (the sense keeps the brain's own genes on its measured games),
# MULTINXON_SENSE_LOGO (the index without the game being played). Inert unless a gene or a switch is set. ============
try:
    import os as _os174
    if _os174.environ.get('MULTINXON_NO_EFFECT1174', '') != '1':     # preflight1174: the 1173 brain, for the inertness test
        import effect1174 as _e174
        _e174.install(_s163_67.Steer4)
except Exception as _e174x:                                                      # pragma: no cover
    print(f'  ! v1.174 effect1174 not installed: {type(_e174x).__name__}: {_e174x}', flush=True)
    raise


# ============ v1.175: A GOAL WHOSE WRONG CELLS CANNOT STEER (effect1175.py; nas_1174_report.md section 25 change 3, FrameLab 19):
# goal_what 0 / 1 (each named goal cell may carry the target colour learned from the levels seen; a cell with a target arrives only at
# its colour) and goal_strict 3 (the strictness by the moves left: 2 at 1-2 moves from the end, 1 at 3-8, 0 beyond); the audit's colour
# column (tallies only). Inert unless a gene or a switch is set. ============
try:
    import os as _os175
    if _os175.environ.get('MULTINXON_NO_EFFECT1175', '') != '1':     # preflight1175: the 1174 brain, for the inertness test
        import effect1175 as _e175
        _e175.install(_s163_67.Steer4)
except Exception as _e175x:                                                      # pragma: no cover
    print(f'  ! v1.175 effect1175 not installed: {type(_e175x).__name__}: {_e175x}', flush=True)
    raise


# ============ v1.176: EFFECTS IN CONTEXT, MEASURED ON A DROP BENCH; RECALL WHERE COLOUR HOLDS PRECISION; A COLOUR PRIOR (effect1176.py;
# nas_1175_report.md section 25 changes 3 and 4, FrameLab 20 / 21): effect_ctx 0 / 1 / 2 (observed effects keyed by the action's
# signature and its local context, predicting only with >= ctx_min_n observations at >= ctx_agree agreement, abstaining otherwise; used
# where steering v5 uses its signature mean; 2 also in the one-step value), the in-play drop bench (tallies only), goal_what coupled to
# effect_ctx, goal_recall 0 / 1, goal_prior 0 / 1. Inert unless a gene or a switch is set. ============
try:
    import os as _os176
    if _os176.environ.get('MULTINXON_NO_EFFECT1176', '') != '1':     # preflight1176: the 1175 brain, for the inertness test
        import effect1176 as _e176
        import steer1164 as _s164_76
        _e176.install(_s163_67.Steer4, _s164_76.Steer5)
except Exception as _e176x:                                                      # pragma: no cover
    print(f'  ! v1.176 effect1176 not installed: {type(_e176x).__name__}: {_e176x}', flush=True)
    raise


# ============ v1.177: THE MILESTONE-2 CAPABILITIES, FORCED (nas_1176_report.md section 26 revised for milestone 2; dfranzen's write-up):
# eyes1177 -- the eyes block appended to a REWIRED brain's encoder (animation, frame_diff, the death and undo state, the steering's
# sense, a 9 x 9 full-resolution fovea, a 5 x 5 patch on the mover, the difference map; settled view, interior change) and the loader
# that rewires a brain on the fly in a 1177 cell; steer1177 -- animations in the model, effect_ctx 3 (back-off, never abstaining), the
# death frame and its diagnosis (budget vs fatal), undo as a probe, the no-op guards, level transfer and macros. Inert unless the
# brain is rewired / a 1177 gene is on; MULTINXON_NO_V1177=1 gives the 1176 brain. ============
try:
    import os as _os177
    if _os177.environ.get('MULTINXON_NO_V1177', '') != '1':
        import steer1160 as _s160_77
        import steer1164 as _s164_77
        import steer1177 as _s177
        _s177.install(_s160_77.Steer, _s163_67.Steer4, _s164_77.Steer5)
        _s177.install_brain(ArcBrain)
        import eyes1177 as _e177
        _e177.install(ArcBrain)
        _e177.install_loader()
except Exception as _e177x:                                                      # pragma: no cover
    print(f'  ! v1.177 eyes1177 / steer1177 not installed: {type(_e177x).__name__}: {_e177x}', flush=True)
    raise


# ============ v1.178: THE SENSORY BUS AND THE MECHANISMS PER BRAIN (nas_1177_report.md section 26, changes 1-4): bus1178 -- every
# 1178 sense in a fixed slot of one versioned appended layout (the 1177 eyes re-placed block by block, the new blocks silent; an input
# with no synapse is inactive and its block is not computed), migrated at load in a 1178 cell; mech1178 -- the 1177 mechanisms read
# per brain from its genes (no cell forcing; the fatal penalty its own gene; the context upscale with effect_ctx 3; undo off = the
# brain's own mask); prior1178 -- the cross-game effect prior of object types for the actions a level has not tried yet (level 1
# included); plan1178 -- a bounded best-first planner over the brain's own learnt model, proposing. Inert unless a 1178 gene is on or
# the brain carries the bus; MULTINXON_NO_V1178=1 gives the 1177 brain. ============
try:
    import os as _os178
    if _os178.environ.get('MULTINXON_NO_V1178', '') != '1':
        import steer1164 as _s164_78
        import bus1178 as _b178
        _b178.install(ArcBrain)
        _b178.install_loader()
        import mech1178 as _m178
        _m178.install(_s163_67.Steer4, ArcBrain)
        import prior1178 as _p178
        _p178.install(_s163_67.Steer4, ArcBrain, _s164_78.Steer5)
        import plan1178 as _pl178
        _pl178.install(_s163_67.Steer4, _s164_78.Steer5, ArcBrain)
except Exception as _e178x:                                                      # pragma: no cover
    print(f'  ! v1.178 bus1178 / mech1178 / prior1178 / plan1178 not installed: {type(_e178x).__name__}: {_e178x}', flush=True)
    raise
