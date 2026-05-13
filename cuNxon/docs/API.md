# cuNxon API Reference

This document covers every public function exposed in `<cuNxon.h>` and every
public type in `<cuNxon_types.h>`.  All functions are `extern "C"`, return
`cunxonStatus_t`, and are safe to call from C, C++, or via `ctypes`/CFFI.

A status of `CUNXON_OK` (= 0) indicates success.  Any other value indicates
an error; call `cunxonGetLastError()` for a human-readable description and
`cunxonGetStatusString(status)` for the symbolic enum name.

The library is **not** internally thread-safe across two threads calling
into the same `cunxonNetwork_t`.  Concurrent use of *different* networks (or
different contexts) is fine and is in fact how the per-sphere CUDA streams
enable intra-network parallelism.

---

## 1. Lifecycle

### `cunxonGetVersion()` → `const char*`
Returns a static string like `"cuNxon 0.1.0"`.  Useful for logging.

### `cunxonGetStatusString(status)` → `const char*`
Symbolic name of a `cunxonStatus_t` (`"CUNXON_OK"`, `"CUNXON_ERR_CUDA"`, …).
Never returns `NULL`.

### `cunxonGetLastError()` → `const char*`
Returns the most recent diagnostic message from any cuNxon call that failed
on **this thread**.  Cleared on each successful entry-point call.  Returns
the empty string when there is no error.

---

## 2. Context

A `cunxonContext_t` owns a CUDA device binding, an RNG seed, and a default
stream.  Most applications need only one context.

### `cunxonCreateContext(ctx*, device_id, seed, flags)`
Creates a context on the specified GPU.  `seed` is mixed with each sphere's
`random_seed_offset` to produce per-neuron / per-synapse RNG streams.
`flags` is reserved (pass 0).

**Errors**: `CUNXON_ERR_INVALID_ARGUMENT`, `CUNXON_ERR_CUDA`.

### `cunxonDestroyContext(ctx)`
Tears down the context.  Any networks created against it must already have
been destroyed.  After this call the handle is invalid.

### `cunxonContextSync(ctx)`
Blocks until all in-flight kernels on the context's streams complete.
Useful before timing measurements.

### `cunxonContextGetProperty(ctx, prop, out, out_size)`
Queries a device property.

| prop                            | out type | meaning                                |
|---------------------------------|----------|----------------------------------------|
| `CUNXON_PROP_DEVICE_ID`         | int      | CUDA device index                       |
| `CUNXON_PROP_DEVICE_NAME`       | char[N]  | NUL-terminated GPU name                 |
| `CUNXON_PROP_COMPUTE_CAPABILITY`| int      | `major*10 + minor` (e.g. 86)            |
| `CUNXON_PROP_TOTAL_GLOBAL_MEM`  | size_t   | total device memory in bytes            |
| `CUNXON_PROP_FREE_GLOBAL_MEM`   | size_t   | free device memory in bytes             |
| `CUNXON_PROP_MAX_THREADS_PER_BLK`| int     | block-size limit                        |
| `CUNXON_PROP_SM_COUNT`          | int      | SM count                                |

`out_size` is the byte size of the receiving buffer.  Returns
`CUNXON_ERR_INVALID_ARGUMENT` if `out_size` is too small.

---

## 3. Network construction

A `cunxonNetwork_t` is a directed multigraph of spheres connected by
inter-sphere links.  Construction is monotonic: spheres and links can only
be added before `cunxonNetworkFinalize`, after which the network is read-only
in topology (weights, traces, and dynamic state continue to evolve).

### `cunxonNetworkCreate(ctx, net*, name)`
Returns an empty network bound to `ctx`.  `name` is a label retained for
diagnostics and serialised by `cunxonNetworkSave` only at the network level.

### `cunxonNetworkDestroy(net)`
Frees all device memory associated with the network.

### `cunxonGetDefaultParameters(params*)`
Fills `params` with the defaults that match `MultiNeuraxon2.py`.  Override
any field before passing to `cunxonNetworkAddSphere`.  See the type
reference at the end for every field's meaning.

### `cunxonNetworkAddSphere(net, name, kind, params*, sphere_id*)`
Allocates a new sphere with the given parameters.  Returns the new
sphere's index in `*sphere_id` (= number of spheres added so far).

The sphere comes up with an auto-built Watts–Strogatz small-world topology
on the hidden block, plus dense `input→hidden` and `hidden→output` edges,
all sorted by `post_id` for CSR-friendly fan-in.

Synapse types are assigned with the priors:
- 50% ionotropic-fast
- 25% ionotropic-slow
- 15% metabotropic (modulatory, drives the slow `modulatory_pot` accumulator
  in the membrane kernel rather than direct branch current)
- 10% silent (placeholder synapses subject to structural plasticity)

**Errors**: `CUNXON_ERR_ALREADY_FINALIZED`, `CUNXON_ERR_OUT_OF_MEMORY`,
`CUNXON_ERR_CUDA`.

### `cunxonNetworkSetSphereInterface(net, sphere_id, sensory_ids, n_sensory, relay_ids, n_relay, relay_out_ids, n_relay_out, readout_ids, n_readout)`
Defines which neurons act as which kind of port:

- **sensory inputs** receive values from `cunxonNetworkStep*`'s `ext_inputs`
- **relay inputs** receive values projected from other spheres' relay-output ports
- **relay outputs** project to other spheres (via `cunxonNetworkAddLink`)
- **readout outputs** are surfaced to host via `cunxonSphereGetReadout`

Indices refer to **neurons within the sphere**, valid range `[0, n_total-1]`.

If you pass `NULL` and `0` for a category, that category is empty.  Sets
must be disjoint within {sensory inputs, relay inputs} and within
{relay outputs, readout outputs}, but the same input neuron may not act as
both a sensory and a relay input.

### `cunxonNetworkAddLink(net, src, dst, link_params*, link_id*)`
Creates a directed link from `src.relay_outputs` to
`(dst.relay_inputs ∪ dst.sensory_inputs)`.  The projection matrix is sized
`(n_dst_ports × n_src_ports)`.

`link_params->kind` selects defaults:

| kind                  | default band  |
|-----------------------|---------------|
| `FEEDFORWARD`         | gamma         |
| `FEEDBACK`            | beta          |
| `LATERAL`             | theta         |
| `THALAMIC`            | theta         |
| `CONTEXT`             | alpha         |

`coherence_strength` (`c` in the CTC gate) interpolates between always-on
(`c=0` → `g=1`) and pure frequency-gated transmission
(`c=1` → `g = ½(1+cos Δφ)`).

### `cunxonNetworkFinalize(net)`
Locks the topology.  Must be called once before any step.  Idempotent
calls return `CUNXON_ERR_ALREADY_FINALIZED`.

### `cunxonNetworkReset(net)`
Zeroes all *dynamic* state (membrane `U`, complement `h`, MSTH 4-loops,
synapse traces, oscillator phases, DSN ring buffers, inter-sphere delay
rings) while **keeping** all *learned* state (weights, DSN kernels, CTSN
gains/biases, ChronoPlastic ω).

### `cunxonNetworkNumSpheres(net)` → `int`
Convenience accessor.  Returns `-1` if `net` is `NULL`.

---

## 4. Execution

External inputs are passed as an array of `float*` pointers, one per sphere
**indexed by sphere_id**.  Pass `NULL` for spheres with no sensory inputs
this step.  Each non-null buffer must have exactly `n_sensory_input_neurons`
floats in `[-1, +1]`.

### `cunxonNetworkStepInfer(net, ext_inputs, dt_ms)`
Advances the network one time step in **inference** mode (no STDP, no
AGMP, no structural pruning, no projection plasticity, no DSN/CTSN online
learning).  Learned parameters remain frozen.

### `cunxonNetworkStepTrain(net, ext_inputs, dt_ms)`
As above, but **all plasticity rules are active**: STDP (DA-D1/D2 gated +
trinary coincidence), AGMP three-factor, structural pruning at the
configured period, online DSN kernel learning, online CTSN gain learning,
ChronoPlastic ω updates, and inter-sphere Hebbian projection plasticity.

### `cunxonNetworkRun(net, n_steps, ext_inputs_per_step, reward_per_step, dt_ms, training)`
Convenience loop.  `ext_inputs_per_step` is an array of length `n_steps`,
each element pointing at a per-sphere `float*` array as in `StepInfer`.
`reward_per_step` (length `n_steps`, may be `NULL`) is injected as a phasic
DA pulse before each step, driving the AGMP eligibility trace.

### `cunxonNetworkInjectNeuromodulator(net, nm_index, amount)`
Injects a phasic neuromodulator pulse into the global field.  Indices:

| index | modulator       |
|-------|-----------------|
| 0     | dopamine (DA)   |
| 1     | serotonin (5-HT)|
| 2     | acetylcholine (ACh) |
| 3     | norepinephrine (NA) |

The pulse decays with the configured `tau_phasic` time constant.

---

## 5. Readout

### `cunxonSphereGetReadout(net, sphere_id, states_out, n_states*)`
Copies the current trinary states of a sphere's **readout outputs** to host.
If `states_out == NULL`, only `*n_states` is filled (size query).

### `cunxonSphereSnapshot(net, sphere_id, U_out, h_out, stilde_out, s_out, firing_rate_out, astrocyte_out, n_neurons*)`
Copies the per-neuron fields of an entire sphere (including hidden neurons).
Any of the output pointers may be `NULL` to skip that channel.

### `cunxonNetworkGetEnergy(net, energy_out*)`
Returns the running accumulator `Σ |Isyn| · |w_fast+w_slow+w_meta|`
since the last `Reset`.  Useful as a proxy for compute / metabolic cost.

---

## 6. Persistence

### `cunxonNetworkSave(net, filepath)`
Writes the network (topology + weights + learned kernels + interface
configuration + link projections) to a binary file with magic `CUNXONV1`.

### `cunxonNetworkLoad(net, filepath)`
Loads a previously saved network into an **empty** (freshly created,
unfinalised) `cunxonNetwork_t`.  Internally calls `cunxonNetworkAddSphere`
and `cunxonNetworkAddLink` to reconstruct topology, then uploads saved
weights and learned kernels, then calls `cunxonNetworkFinalize`.

Errors:
- `CUNXON_ERR_INCOMPATIBLE_SAVE` if the magic doesn't match or the file
  carries `(n_total, dsn_K)` that don't agree with the saved params.
- `CUNXON_ERR_FILE_IO` for short or unreadable files.
- `CUNXON_ERR_ALREADY_FINALIZED` if `net` already has spheres or links.

The file format is documented in detail in the header comment of
`cunxonNetworkSave` (in `src/cuNxon.cu`).

---

## 7. Aigarth evolutionary plasticity

### `cunxonNetworkAigarthConfig(net, population, mut_fast, mut_slow, mut_meta)`
Configures the genetic step:

- `population` (≥ 1) — number of mutated candidates to evaluate per call.
- `mut_fast`, `mut_slow`, `mut_meta` (≥ 0) — Gaussian σ for mutations on
  the corresponding synaptic-weight populations (`w_fast`, `w_slow`,
  `w_meta`).  Inter-sphere link weights use `mut_meta`.

### `cunxonNetworkAigarthStep(net, fitness_fn, user_data)`
Runs one generation:

1. Snapshots the current weights as the baseline candidate.
2. Evaluates baseline via `fitness_fn(net, user_data)`.
3. For each of `population` candidates:
   - Restores baseline weights, applies Gaussian mutation, resets dynamic
     state.
   - Calls `fitness_fn(net, user_data)`.
   - If the score is higher than the current best, snapshots the candidate.
4. Restores the best candidate's weights.

`fitness_fn` is a host-callable function pointer (`cunxonFitnessFn_t`)
returning a `float` (higher is better).  Inside it, you may use any cuNxon
call — typically `cunxonNetworkStepInfer` over a short eval window followed
by `cunxonSphereGetReadout`.

Combine Aigarth with normal training to mix evolutionary structural search
with synaptic plasticity, as in the Multi-Neuraxon paper's Aigarth hybrid.

---

## 8. Sphere layers (paper §P4)

Metadata-only grouping of spheres into processing tiers (sensory →
association → motor).  Layers do not affect kernel execution; they exist
for introspection, organisation, and downstream analysis tools.

### `cunxonNetworkAddLayer(net, name, depth, layer_id*)`
Creates a new layer.  `depth` is a free-form integer the paper uses to
order tiers (0 = sensory, 1 = association, 2 = motor by convention).

### `cunxonNetworkAddSphereToLayer(net, layer_id, sphere_id)`
Attaches an existing sphere to a layer.  Idempotent.  A sphere may live in
multiple layers.

### `cunxonNetworkNumLayers(net)` → `int`
Convenience accessor.

### `cunxonNetworkGetLayer(net, layer_id, name_out, name_buf_len, depth_out, sphere_ids, n_spheres*)`
Reads back a layer's name, depth, and member sphere ids.  Pass
`sphere_ids = NULL` for a size query; on the second call allocate
`n_spheres` ints and pass `&n_spheres` again.

---

## 9. Pattern application layer (paper Algorithm 8)

Helpers for **store-then-recall associative memory**: store an input
pattern under a name, train the network on it for some steps, then later
recall it from a partial cue.  Patterns live in host memory keyed by name;
recall drives the target sphere's sensory ports and reads back its
trinary readout after a short settle period.

### `cunxonNetworkStorePattern(net, sphere_id, name, pattern, len, n_present_steps, dt_ms)`
Stores `pattern` (length = `n_sensory_input_neurons` on `sphere_id`) under
`name` and consolidates it by calling `StepTrain` `n_present_steps` times.

### `cunxonNetworkRecallPattern(net, sphere_id, name, pattern_len, mask_fraction, n_settle_steps, dt_ms, readout_out, n_readout*)`
Masks `mask_fraction` (0–1) of the stored cue's entries to zero and
presents the result for `n_settle_steps` of inference, then copies the
readout to `readout_out`.  Pass `readout_out = NULL` for a size query.

### `cunxonNetworkTrainSequence(net, sphere_id, patterns, n_patterns, pattern_len, n_repetitions, n_steps_per_pattern, dt_ms)`
Loops `n_repetitions × n_patterns × n_steps_per_pattern` training steps
through a sequence of patterns.  Useful for sequence-learning benchmarks.

### `cunxonNetworkListPatterns(net, names_out, names_buf_len, n_patterns*)`
Returns the count, and optionally a NUL-terminated newline-separated
string of stored pattern names.

### `cunxonNetworkClearPatterns(net)`
Removes all stored patterns.  Network weights are not touched.

---

## 10. Type reference

### `cunxonStatus_t`

| Code                              | Meaning                                  |
|-----------------------------------|------------------------------------------|
| `CUNXON_OK`                       | success                                  |
| `CUNXON_ERR_INVALID_ARGUMENT`     | NULL pointer or out-of-range value       |
| `CUNXON_ERR_OUT_OF_MEMORY`        | cudaMalloc or std::vector failure        |
| `CUNXON_ERR_NOT_FINALIZED`        | called step before `Finalize`            |
| `CUNXON_ERR_ALREADY_FINALIZED`    | tried to mutate topology after Finalize  |
| `CUNXON_ERR_INVALID_SPHERE`       | sphere id out of range                   |
| `CUNXON_ERR_INVALID_LINK`         | link id out of range                     |
| `CUNXON_ERR_CUDA`                 | CUDA runtime returned an error           |
| `CUNXON_ERR_FILE_IO`              | open/read/write failure                  |
| `CUNXON_ERR_INCOMPATIBLE_SAVE`    | save file magic/version mismatch         |
| `CUNXON_ERR_NOT_IMPLEMENTED`      | feature stub (kept for ABI stability)    |

### `cunxonNetworkParameters_t`

87 fields mirroring `MultiNeuraxon2.py`'s `NetworkParameters` dataclass.
Categories (see `cuNxon_types.h` for full list):

- Architecture: `num_input_neurons`, `num_hidden_neurons`, `num_output_neurons`,
  `num_dendritic_branches`, `dendritic_spike_threshold`,
  `dendritic_supralinear_gamma`, Watts–Strogatz `ws_k`/`ws_beta`.
- Neuron membrane: `membrane_time_constant`, firing thresholds, adaptation,
  autoreceptor, spontaneous firing, neuron health decay.
- Homeostatic plasticity (paper Eq.1).
- MSTH four loops (taus and gains for ultrafast / fast / medium / slow).
- DSN dynamic decay (kernel size, online-learning flags, kernel clip).
- CTSN complement (ρ, gains, biases, learning).
- Synapse time constants and weight-init bounds (fast/slow/meta).
- ChronoPlasticity (Eqs. 5–7): α, λ, trace clip, ω bounds, EMA smoothing.
- AGMP (Eqs. 8–10): λe, λa, η.
- Plasticity: STDP window, associative α, structural thresholds.
- Neuromodulator baselines + tonic/phasic taus.
- Oscillator bank frequencies (6 bands) + amplitude + PAC strength.
- `random_seed_offset` (combined with the context seed).

### `cunxonLinkParameters_t`

14 fields controlling one directed inter-sphere link:

- `kind` (`cunxonLinkKind_t`), `coherence_band` (`cunxonBand_t`)
- `gain`, `delay_steps`, `transmission_threshold`, `coherence_strength`
- `topology` (DENSE / SPARSE / TOPOGRAPHIC / ONE_TO_ONE), `sparse_prob`
- `allow_negative_weights`
- `plasticity_rate`, `weight_decay`, `weight_clip`, `normalize_rows`
- `bias`

---

## 11. Memory & threading

- All allocations live on the **active CUDA device** at context creation.
- Each sphere has its own non-blocking CUDA stream so multi-sphere step
  pipelines overlap.  Inter-sphere kernels run on the context's default
  stream and synchronise the sphere streams as needed.
- Read-side calls (`GetReadout`, `Snapshot`, `GetEnergy`) issue
  synchronous device-to-host copies and call `cudaStreamSynchronize` on
  the relevant sphere stream first.
- All `__global__` kernels are launched with bounded block/grid dimensions
  derived from `n_total` or `n_syn` and a fixed block size of 128 or 256.
