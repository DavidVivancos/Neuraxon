# Architecture configuration files

This folder holds JSON files that parameterise the Neuraxon Game of Life.
**v155 (v4.63)** introduced this separation so you can test different
architectures **without modifying any Python code**.

## How to use

1. Copy `default.json` to a new file in this folder, e.g. `myarch.json`.
2. Edit the values you want to test. Anything you leave alone falls
   through to the existing code defaults.
3. Run the game with either:
   - Environment variable: `NEURAXON_ARCH=architectures/myarch.json python main.py`
   - Command-line flag: `python main.py --architecture architectures/myarch.json`

If no architecture is specified, `architectures/default.json` is used
(which contains the current v155 baseline values).

## File structure

The JSON has four sections:

### `biology`
**Game-world / bio-inspired dynamics.** Independent of neural
architecture — change these to study survival pressure without touching
the brain.

Examples:
- `metabolic_ramp_per_sec`: how fast atrophy compounds when an NxEr
  stops moving
- `max_atrophy`: hard cap on atrophy multiplier (v151 fix)
- `idle_explore_seconds` / `explore_probability`: idle-exploration safety
  net trigger (v152)
- `mate_cooldown_seconds`, `circadian_cycle_ticks`, etc.

### `neural`
**Network topology and per-neuron parameters.** Change these to test
different brain configurations.

Examples:
- `num_input_neurons`, `num_output_neurons`: sensory/motor counts
- `firing_threshold_excitatory` / `firing_threshold_inhibitory`: trinary
  decision boundaries
- `intrinsic_timescale_default`: membrane time constant τ
- `connection_probability`: how dense the random connectivity is

### `operating_ranges`
**Plasticity / adaptation / brake tunables.** Change these to test
different learning dynamics.

Examples:
- `learning_rate`, `plasticity_threshold`: synaptic STDP rate
- `adaptation_tau_ticks`, `adaptation_target_excitatory_multiplier`: the
  v149 adaptation brake parameters
- `sensory_boost_function`, `sensory_boost_scale`: the v152 saturating cap
- `plasticity_brake_threshold`, `plasticity_brake_slope`: the v152
  plasticity brake (currently reads input_saturation_fraction)

### `healthy_bands`
**Target ranges shown on the dashboard.** NxErs operating in these
bands are coloured green in the L-overlay.

Examples:
- M1-M10 metric target ranges
- `pop_mean_idle_seconds`: how idle is "too idle" before the dashboard
  warns
- `exploration_trigger_rate`: what trigger rate is healthy vs over-firing

## What this is NOT

This is a **first-cut** of architecture parameterisation. The v155 release
wires up a few of the most impactful parameters; many constants still
live in code (especially in `neuraxon/neuron.py` and `neuraxon/components.py`).
Future releases will migrate more.

Specifically NOT yet swappable via this file in v155:
- Multi-sphere topology (still uses hardcoded sensory/association/motor)
- Genetic mutation parameters
- Dopamine reward propagation rules
- Voice/song frequency parameters

## Example: testing a smaller network

```bash
cp default.json experiment_small.json
# Edit experiment_small.json:
#   "num_hidden_neurons_default": 5,   # was 12
#   "connection_probability": 0.5,      # was 0.3
# Save, then run:
NEURAXON_ARCH=architectures/experiment_small.json python main.py
```

## Example: testing tighter saturation cap

```bash
cp default.json experiment_tight_cap.json
# Edit:
#   "sensory_boost_scale": 0.5,   # was 1.0
# Now the saturating cap asymptotes at 0.5 instead of 1.0.
```

## Example: testing slower plasticity

```bash
cp default.json experiment_slow_learn.json
# Edit:
#   "learning_rate": 0.003,   # was 0.01
# STDP runs ~3× slower → input synapses potentiate slower → lock-in
# pattern (if any) takes 3× longer to form.
```

## Inspecting the loaded values

When the game starts with a custom architecture, it prints to stdout:
```
[ARCHITECTURE] Loaded architectures/myarch.json
[ARCHITECTURE]   biology: 4 overrides
[ARCHITECTURE]   neural: 2 overrides
[ARCHITECTURE]   operating_ranges: 1 override
[ARCHITECTURE]   healthy_bands: 0 overrides (using code defaults)
```

So you can confirm which values were actually changed.

## v155 contract

The values listed in `default.json` are the keys recognised by this
release. Add unknown keys (the loader prints a warning and ignores them
— so you can leave forward-compatible parameters in your files without
breaking older versions).
