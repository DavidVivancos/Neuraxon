# cuNxon Architecture

## 1. Multi-sphere brain

A cuNxon network is a directed multigraph of **spheres** connected by
**inter-sphere links**:

```
                 (sensory inputs)                  (sensory inputs)
                       |                                  |
                       v                                  v
                  +---------+    FF (gamma)         +---------+
                  | sphere0 |---------------------->| sphere2 |
                  | (VIS)   |                       |  (ASC)  |---FF (gamma)----+
                  +---------+                       +---------+                 |
                       ^                              |    ^                    v
                       |                  Thalamic    |    | FB (beta)     +---------+
                       +-Theta-+                      |    +---------------|sphere3  |
                               +-Lateral-+            |                    | (MTR)   |
                  +---------+    Theta   |            |                    +---------+
                  | sphere1 |------------+            v                         |
                  |  (AUD)  |<-----------------------(thalamic)                 |
                  +---------+                                                   v
                                                                      (final readout)
```

Each sphere is an instance of the full Neuraxon v2.0 model: input / hidden /
output neurons, a Watts–Strogatz small-world recurrent connectivity inside
the hidden block, multi-scale plasticity, a bank of oscillator phases, and
its own neuromodulator field.  Spheres communicate only through their
**relay-output → relay-input** ports, with messages gated by the CTC
coherence rule.

---

## 2. Algorithm 1 pipeline (one timestep)

For each step, the orchestrator (`step_impl` in `src/cuNxon.cu`) runs the
following sequence.  Per-sphere kernels execute on each sphere's own
non-blocking CUDA stream so multi-sphere brains parallelise; inter-sphere
kernels run on the context's default stream with synchronisation points.

```
            ┌──────────────────────────────────────────────────────────────┐
            │ (A) clear scratch buffers: branch_sum, modulatory_pot,       │
            │     ext_in;  upload external sensory inputs to ext_in        │
            ├──────────────────────────────────────────────────────────────┤
            │ (B) k_oscillator_advance                                     │
            │     - advance 6 band phases per sphere; apply theta→gamma PAC │
            ├──────────────────────────────────────────────────────────────┤
            │ (C) k_chrono_warp_and_isyn   (per-synapse, scatter)           │
            │     - update fast/slow ChronoPlastic traces (clipped)         │
            │     - update ω with EMA + bounds; α_s^ω via expf(ω·logf(α_s)) │
            │     - compute I_syn(t)                                       │
            │     - if synapse is metabotropic: atomicAdd into              │
            │       modulatory_pot[post]                                   │
            │     - else: atomicAdd into branch_sum[post * B + branch_id]   │
            ├──────────────────────────────────────────────────────────────┤
            │ (D) k_dendritic_gather  (per-neuron, supralinear)             │
            │     - for each branch b: gate via dendritic_spike_threshold,  │
            │       apply sign(σ)·|σ|^γ if super-threshold                 │
            │     - sum branches into branch_pot[i*B + 0] (= D_raw)         │
            ├──────────────────────────────────────────────────────────────┤
            │ ─ sync per-sphere streams ─                                   │
            ├──────────────────────────────────────────────────────────────┤
            │ (E) inter-sphere transmission                                 │
            │       k_ctc_gate          : g = (1-c) + ½c(1+cos(Δφ_band))    │
            │       k_intersphere_project: contrib = g·gain·(W · s_relay)+b │
            │       k_intersphere_inject: ext_in[dst_port] += contrib       │
            ├──────────────────────────────────────────────────────────────┤
            │ (F) k_sphere_activity_stats                                   │
            │       reduce mean|s|, excitatory fraction, change rate        │
            ├──────────────────────────────────────────────────────────────┤
            │ (G) k_neuromod_update                                         │
            │       - tonic relaxation + phasic decay for DA/5HT/ACh/NA     │
            │       - logistic-saturation activations for 9 receptor types  │
            ├──────────────────────────────────────────────────────────────┤
            │ (H) k_membrane_dsn_ctsn_emit  (FUSED, per-neuron)              │
            │       Steps 3-7:                                              │
            │         3. MSTH 4-loop update                                 │
            │         4. DSN: α_t = sigmoid(-conv(buffer, kernel))           │
            │            online L1-normalised kernel learning (optional)    │
            │         5. Membrane:                                          │
            │              if dsn_enabled:                                  │
            │                U ← α_t·U + (1-α_t)·(D_raw·MSTH_m·g_NA + ext + │
            │                                     osc - adapt + ξ)         │
            │              else:                                            │
            │                U ← (1-dt/τ_mem)·U + (dt/τ_mem)·drive          │
            │         6. CTSN: h(t) = ρ·tanh(φ_gain·D_raw + φ_bias);         │
            │            s_tilde = U + h                                    │
            │            online learning of BOTH φ_gain and φ_bias (opt)    │
            │         7. Trinary readout:                                   │
            │              s ← step(s_tilde, th_pos, th_neg)                │
            │            thresholds modulated by:                           │
            │              - receptor activations (M1/M2 + α2)               │
            │              - modulatory_pot[i] (metabotropic drive)         │
            │              - homeostasis + MSTH_f + autoreceptor             │
            │         Health update + neuron death:                         │
            │              health ← clip(health - (decay·over +              │
            │                            0.1·msth_slow_gain·msth_slow)·dt)  │
            │              if health < death_threshold: is_active=0, s=0    │
            ├──────────────────────────────────────────────────────────────┤
            │ (training branch only)                                        │
            │ (I) k_plasticity_stdp                                         │
            │       - pre/post trace updates (capped by stdp_window/tau)    │
            │       - DA-gated D1 / D2 multiplicative modulation             │
            │       - trinary coincidence boosts ({+,+} {+,-} {0,0})         │
            │       - integration into w_fast/w_slow/w_meta with decay leak │
            │       - stash dw_total into eligibility for assoc + AGMP       │
            │ (I.5) k_plasticity_associative                                │
            │       - diffuse dw across synapses sharing a post (paper Eq.)  │
            │       - Δw_i += α · Σ_{j∈N(i)} (Δw_j - Δw_i) / d_ij           │
            │       - neighbourhood = same CSR slice (post_offset);          │
            │         d_ij = |branch_i - branch_j| + 1                       │
            │ (J) k_plasticity_agmp                                         │
            │       - e ← λ_e·e + Hebbian sign(s_pre)·sign(s_post)           │
            │       - Δw = η · DA_phasic · astrocyte · e                     │
            │ (K) k_structural_prune                                        │
            │       - integrity prune (deterministic: |w|^2 < ε^2 → silent)  │
            │       - stochastic death (silenced w/ p = death_prob·dt·weak)  │
            │       - stochastic resurrection (silent → active w/ random w)  │
            │       - cuRAND Philox seeded by (idx, step_seed)               │
            │ (L) k_proj_plasticity                                         │
            │       - Hebbian ΔW = η·s_src·s_dst on each link's W            │
            └──────────────────────────────────────────────────────────────┘
```

The boundary between "inference" and "training" is exactly steps I–L —
inference runs only A–H.

---

## 3. Device-side data layout

Every per-neuron and per-synapse field is its own coalesced array.  All
arrays use Structure-of-Arrays (SoA) layout.

### Per neuron (length `n_total`)

| field name           | dtype     | purpose                                              |
|----------------------|-----------|------------------------------------------------------|
| `type`               | int8      | input / hidden / output                              |
| `is_active`          | int8      | structural-plasticity alive flag                     |
| `U`                  | float     | membrane potential                                   |
| `adaptation`         | float     | spike-frequency adaptation accumulator               |
| `autoreceptor`       | float     | presynaptic autoreceptor accumulator                 |
| `s`                  | int8      | current trinary state ∈ {-1, 0, +1}                  |
| `s_prev`             | int8      | previous trinary state                               |
| `complement_h`       | float     | CTSN complement value `h(t)`                         |
| `s_tilde`            | float     | combined `U + h` (pre-readout)                       |
| `ctsn_phi_gain`      | float     | learned CTSN gain                                    |
| `ctsn_phi_bias`      | float     | learned CTSN bias                                    |
| `dsn_kernel`         | float[K]  | learned DSN causal-conv kernel                       |
| `dsn_buffer`         | float[K]  | ring buffer of past raw inputs                       |
| `dsn_head`           | int       | ring index                                           |
| `dsn_alpha`          | float     | last computed decay coefficient                      |
| `branch_pot`         | float[B]  | per-branch potential (reused as scratch for D_raw)   |
| `branch_sum`         | float[B]  | per-step branch scatter target                       |
| `modulatory_pot`     | float     | per-step metabotropic scatter target                 |
| `firing_rate_avg`    | float     | running average for homeostasis                      |
| `msth_ultrafast`     | float     | MSTH loop 1 (~5 ms)                                 |
| `msth_fast`          | float     | MSTH loop 2 (~2 s)                                  |
| `msth_medium`        | float     | MSTH loop 3 (~5 min) — clipped to [0.5, 2.0]         |
| `msth_slow`          | float     | MSTH loop 4 (~1–24 h)                                |
| `astrocyte`          | float     | AGMP astrocyte state                                 |
| `health`             | float     | structural-plasticity score                          |
| `rng`                | curandState | per-neuron XORWOW state                            |

### Per synapse (length `n_syn`, sorted by `post_id`)

| field name          | dtype  | purpose                                  |
|---------------------|--------|------------------------------------------|
| `pre_id`            | int    | source neuron index                       |
| `post_id`           | int    | destination neuron index (sort key)       |
| `branch_id`         | int    | dendritic branch ∈ [0, B-1]               |
| `w_fast`            | float  | ionotropic-fast component weight          |
| `w_slow`            | float  | ionotropic-slow component weight          |
| `w_meta`            | float  | metabotropic weight                       |
| `is_silent`         | int8   | structural-plasticity silenced flag       |
| `is_modulatory`     | int8   | bypass dendritic branch (drives mod_pot)  |
| `synapse_type`      | int8   | type tag (fast/slow/meta/silent)          |
| `integrity`         | float  | structural-plasticity score               |
| `pre_trace`         | float  | STDP pre-spike trace                      |
| `post_trace`        | float  | STDP post-spike trace                     |
| `chrono_fast_trace` | float  | ChronoPlastic fast trace `f`              |
| `chrono_slow_trace` | float  | ChronoPlastic slow trace `z`              |
| `chrono_omega`      | float  | per-synapse warp factor ω                 |
| `eligibility`       | float  | AGMP eligibility trace                    |
| `post_offset`       | int    | CSR offset table (length `n_total + 1`)   |

### Per link (one per inter-sphere edge)

| field name    | dtype     | purpose                                       |
|---------------|-----------|-----------------------------------------------|
| `W`           | float[Sd*Ss] | projection matrix `[n_dst_ports x n_src_ports]` |
| `delay_ring`  | int8[D*Ss]| ring of source-port trinary states, depth = `delay_steps + 1` |
| `g_ctc`       | float     | last-computed CTC gate value                  |
| `contrib`     | float[Sd] | per-step contribution buffer                  |

### Per sphere (one struct)

| field name                                                       | purpose                       |
|------------------------------------------------------------------|-------------------------------|
| `n_in, n_hid, n_out, n_total, n_syn, n_branches, dsn_K`         | sizes                          |
| `kind`                                                          | sensory / association / motor |
| `N`                                                             | NeuronArraysDev                |
| `S`                                                             | SynapseArraysDev               |
| `O`                                                             | OscillatorBankDev (6 bands)    |
| `M`                                                             | NeuromodFieldDev (4 mods)      |
| `p_dev`                                                         | device pointer to params        |
| `port_in_sensory / port_in_relay / port_out_relay / port_out_readout` | device port-id arrays  |
| `ext_in`                                                        | per-step external input buffer  |
| `d_energy`                                                      | device energy accumulator        |
| `stream`                                                        | non-blocking CUDA stream         |

---

## 4. Threading and stream model

- Each sphere has a non-blocking `cudaStream_t`. All per-sphere kernels in
  phases A–D and F–L run on the sphere's own stream, so multi-sphere brains
  overlap automatically subject to GPU resources.
- Phase E (inter-sphere transmission) runs on the context's default stream.
  Sphere streams synchronise to it before phase E and after.
- Host-to-device input upload (phase A) uses `cudaMemcpyAsync` on the
  sphere's stream.  Readout/snapshot calls synchronise the relevant sphere
  stream before issuing a device-to-host copy.

---

## 5. Performance notes

- **Synapse-major loops** (`k_chrono_warp_and_isyn`, `k_plasticity_*`)
  dominate compute for large `n_syn`.  They use one thread per synapse and
  scatter into `branch_sum` / `modulatory_pot` via `atomicAdd`.
- **Neuron-major loops** (`k_dendritic_gather`,
  `k_membrane_dsn_ctsn_emit`) use one thread per neuron and avoid atomics.
- The membrane kernel is **fused** (MSTH + DSN + membrane + CTSN +
  readout + adaptation in one launch) to minimise launch overhead and
  global-memory traffic — each per-neuron field is loaded once and written
  back at most once.
- **CSR post_offset** lets the dendritic gather walk only the fan-in of
  each post-neuron (no full synapse scan).
- For very large networks (`n_syn > 10^7`), the activity-stat reductions
  may become host-roundtrip-bound; consider keeping the device-resident
  scalars on-device until the end of the step.
