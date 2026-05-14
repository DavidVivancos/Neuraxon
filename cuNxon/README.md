# cuNxon

**CUDA library and C API for building, training, and executing Multi-Neuraxon 2.0 networks on NVIDIA GPUs.**

cuNxon is a from-scratch GPU implementation of the neural blueprint introduced in:

1. D. Vivancos & J. Sanchez (2026). *Neuraxon V2.0: A New Neural Growth Computation Blueprint*. ICMLT'26 in print (Qubic Open Science.)
2. D. Vivancos & J. Sanchez (2026). *Multi-Neuraxon: a Hierarchical Brain-Inspired Architecture for AGI*. (companion paper tbr).

Both papers extend the original Neuraxon 1.0 model with:

* MSTH (Multi-Scale Temporal Homeostasis) — four nested adaptation loops (≈5 ms / ≈2 s / ≈5 min / ≈1–24 h).
* DSN (Dendritic Spectrotemporal Network) — per-neuron causal-convolution kernel with online (paper-style) kernel learning.
* CTSN (Complement Trinary Spectrotemporal Network) — an antagonist `h(t)` that produces a trinary `{-1, 0, +1}` readout.
* ChronoPlastic synapses — temporal-warp factor ω updated by an Eq.5–7 system with EMA smoothing and bounded clipping.
* Four neuromodulators (DA, 5-HT, ACh, NA) and nine receptor subtypes (D1, D2, 5-HT1A, 5-HT2A, 5-HT4, M1, M2, β1, α2).
* Six oscillator bands (infraslow, slow, theta, alpha, beta, gamma) with theta→gamma PAC.
* AGMP three-factor plasticity (eligibility × DA × astrocyte), STDP with trinary-coincidence boosts, structural pruning, and Aigarth-style evolutionary plasticity.
* Multi-sphere "Multi-Neuraxon" composition with CTC (Communication Through Coherence) frequency-gated inter-sphere links.

Reference implementation: <https://github.com/DavidVivancos/Neuraxon> ·  Demo: <https://huggingface.co/spaces/DavidVivancos/Neuraxon>.

---

## Why cuNxon?

The reference `MultiNeuraxon2.py` is a clean NumPy/PyTorch-style implementation, but a single sphere of a few hundred neurons runs at a few Hz on CPU because each step touches **per-neuron** MSTH state, **per-neuron** DSN convolutions, **per-synapse** ChronoPlastic warps, and a global neuromodulator field. cuNxon ports the full Algorithm 1 pipeline to CUDA kernels with:

* **Structure-of-Arrays (SoA)** device layout — every per-neuron / per-synapse field is its own coalesced array.
* **CSR-sorted synapses by post-id** — fan-in summation is one cache-friendly stride.
* **Per-sphere CUDA streams** — multi-sphere brains overlap automatically.
* **Fused membrane kernel** — MSTH update + DSN convolution + CTSN gain → trinary readout, all in one launch per sphere.
* **cuRAND XORWOW** per neuron for spontaneous firing, **cuRAND Philox** per synapse for structural plasticity, and for Aigarth mutations.

---

## Feature coverage

Every numbered feature from `MultiNeuraxon2.py`'s docstring and both papers is present in cuNxon, with the corresponding GPU kernel and C-API surface:

| # | Paper / Python feature                                | cuNxon kernel                                | Public API entry point                              |
|---|--------------------------------------------------------|-----------------------------------------------|------------------------------------------------------|
| i | Receptor subtypes with nonlinear tonic/phasic curves   | `k_neuromod_update`                           | `cunxonNetworkInjectNeuromodulator`                  |
| ii | Multi-band oscillator bank with PAC (θ→γ)             | `k_oscillator_advance`                        | params `osc_freq`, `osc_amplitude`, `osc_pac_strength` |
| iii | Nonlinear dendritic branch integration               | `k_dendritic_gather`                          | params `num_dendritic_branches`, `dendritic_supralinear_gamma` |
| iv | Temporal STDP traces with differential DA gating      | `k_plasticity_stdp`                           | `cunxonNetworkStepTrain`                             |
| v | Associative-neighbour plasticity                       | `k_plasticity_associative`                    | param `associative_alpha`                            |
| vi | Watts-Strogatz small-world topology                   | host `build_initial_topology`                 | params `ws_k`, `ws_beta`                             |
| vii | Full neuromodulator system + receptor crosstalk      | `k_neuromod_update`                           | `cunxonNetworkInjectNeuromodulator`                  |
| viii | Energy tracking                                     | accumulated in plasticity kernels             | `cunxonNetworkGetEnergy`                             |
| ix | Aigarth evolutionary mutation & selection             | `k_aigarth_mutate_weights`                    | `cunxonNetworkAigarthConfig` + `cunxonNetworkAigarthStep` |
| x | ChronoPlastic synaptic time warping (learned ω_t)      | `k_chrono_warp_and_isyn`                      | params `chrono_*`                                    |
| xi | DSN dynamic decay α_t via causal conv (+ online learning) | `k_membrane_dsn_ctsn_emit`              | params `dsn_*`, `dsn_learn_*`                        |
| xii | CTSN complement trinary state + online φ-gain/bias learning | `k_membrane_dsn_ctsn_emit`              | params `ctsn_*`, `ctsn_learn_*`                      |
| xiii | AGMP astrocyte-gated plasticity                      | `k_plasticity_agmp`                           | params `agmp_*`                                      |
| xiv | Homeostatic plasticity (intrinsic + synaptic scaling) | inside `k_membrane_dsn_ctsn_emit`            | params `homeostatic_rate`, `target_firing_rate`      |
| xv | MSTH 4 regulatory loops (5 ms / 2 s / 5 min / 1–24 h)  | inside `k_membrane_dsn_ctsn_emit`             | params `msth_*`                                      |
| xvi | Pattern storage & recall application layer (Algorithm 8) | host orchestration over Step kernels       | `cunxonNetwork{Store,Recall,TrainSequence,ListPatterns,ClearPatterns}` |
| P2 | Arbitrary directed-graph sphere connectivity          | `k_intersphere_project` + `k_intersphere_inject` | `cunxonNetworkAddLink`                            |
| P3 | CTC frequency-gated transmission g = (1−c)+½c(1+cosΔφ) | `k_ctc_gate`                                 | param `coherence_strength`, `coherence_band`         |
| P4 | Hierarchical sphere tiers (sensory → assoc → motor)   | host metadata                                 | `cunxonNetwork{AddLayer,AddSphereToLayer,GetLayer}`  |
| P5 | Volume-transmission neuromodulation                   | per-sphere `NeuromodFieldDev`                 | `cunxonNetworkInjectNeuromodulator`                  |
| —  | Neuron health + structural death                      | inside `k_membrane_dsn_ctsn_emit`             | params `neuron_health_decay`, `neuron_death_threshold` |
| —  | Synaptic structural growth (silent → active resurrection) | `k_structural_prune` (cuRAND Philox)      | params `synapse_formation_prob`, `synapse_death_prob` |
| —  | Metabotropic (modulatory) synaptic routing            | dedicated `N.modulatory_pot[]` accumulator    | `is_modulatory` synapse flag                         |

Every field declared in `cunxonNetworkParameters_t` is consumed somewhere — either by a device kernel (verified by the audit script in `tests/`) or by host-side topology/initialisation code.

---

## File layout

```
cuNxon/
├── include/
│   ├── cuNxon.h           – public C API  (extern "C", usable from C/C++/ctypes)
│   └── cuNxon_types.h     – enums, NetworkParameters, LinkParameters
├── src/
│   ├── cuNxon.cu          – host orchestration (context, network, step, save/load, Aigarth)
│   ├── cuNxon_internal.cuh– internal device structs + kernel decls
│   ├── cuNxon_kernels.cu  – Algorithm 1 forward kernels (oscillators, neuromod,
│   │                        ChronoPlastic, dendritic gather, MSTH+DSN+CTSN)
│   ├── cuNxon_plasticity.cu  – STDP, AGMP, structural prune
│   └── cuNxon_intersphere.cu – CTC gate, projection, plasticity
├── examples/
│   ├── example_4sphere.cu – 4-sphere VIS / AUD / ASC / MTR brain demo
│   ├── example_aigarth.cu – evolutionary plasticity (Aigarth) demo
│   └── python_binding.py  – ctypes wrapper exposing the full C API
├── tests/
│   └── test_cunxon.cu     – smoke / correctness suite (registered with CTest)
├── docs/
│   ├── API.md             – per-function reference (preconditions, semantics, threading)
│   └── ARCHITECTURE.md    – algorithm 1 pipeline + device data layout + stream model
├── CMakeLists.txt
└── README.md
```

---

## Building

### Requirements

* **CUDA Toolkit ≥ 11.0** (nvcc, cuRAND).
* **CMake ≥ 3.18**.
* C++14 compiler (GCC ≥ 9, Clang ≥ 10, or MSVC 19.30+).
* NVIDIA GPU with compute capability ≥ 7.0 (Volta, Turing, Ampere, Ada, Hopper, Blackwell).

### Build

```bash
git clone <your-fork>/cuNxon.git
cd cuNxon
cmake -S . -B build -DCUNXON_CUDA_ARCH="80;86;89"   # pick your arch(s)
cmake --build build -j
```

Outputs:

* `build/libcunxon.so`  (or `cunxon.dll` on Windows)
* `build/libcunxon_static.a`
* `build/example_4sphere`     — VIS+AUD→ASC→MTR four-sphere brain training demo
* `build/example_aigarth`     — evolutionary plasticity (Aigarth) demo on a single sphere
* `build/test_cunxon`         — smoke / correctness test suite

Build options:

| CMake variable             | Default       | Meaning                                          |
|----------------------------|---------------|--------------------------------------------------|
| `CUNXON_BUILD_SHARED`      | `ON`          | Build `libcunxon.so` / `cunxon.dll`              |
| `CUNXON_BUILD_STATIC`      | `ON`          | Build `libcunxon_static.a`                       |
| `CUNXON_BUILD_EXAMPLES`    | `ON`          | Build `example_4sphere` + `example_aigarth`      |
| `CUNXON_BUILD_TESTS`       | `ON`          | Build `test_cunxon` + register CTest target      |
| `CUNXON_CUDA_ARCH`         | `70`          | Semicolon-separated list of CC numbers           |
| `CMAKE_BUILD_TYPE`         | `Release`     | Release / Debug / RelWithDebInfo                 |

### Manual one-shot (no CMake)

```bash
nvcc -O3 -std=c++14 -arch=sm_80 \
     -Iinclude  src/*.cu  examples/example_4sphere.cu \
     -lcurand   -o example_4sphere
./example_4sphere 500 200            # 500 train steps, 200 eval steps
```

### Tests

```bash
cd build && ctest --output-on-failure
# or run directly:
./test_cunxon
```

The smoke suite covers: context lifecycle, default parameter sanity, single-sphere step, multi-sphere link, reset semantics (clears dynamics, preserves weights), Save/Load round-trip, modulatory-synapse path, and Aigarth fitness improvement.

---

## Quick-start (C++)

```cpp
#include <cuNxon.h>

cunxonContext_t ctx;
cunxonCreateContext(&ctx, /*device_id=*/0, /*seed=*/42, /*flags=*/0);

cunxonNetwork_t net;
cunxonNetworkCreate(ctx, &net, "brain");

cunxonNetworkParameters_t p;
cunxonGetDefaultParameters(&p);            // defaults match MultiNeuraxon2.py
p.num_input_neurons  = 8;
p.num_hidden_neurons = 64;
p.num_output_neurons = 4;

int s0; cunxonNetworkAddSphere(net, "S0", CUNXON_SPHERE_SENSORY, &p, &s0);

// (optional) set port mapping with cunxonNetworkSetSphereInterface(...)
// (optional) add inter-sphere links   with cunxonNetworkAddLink(...)

cunxonNetworkFinalize(net);

std::vector<float> x(8, 0.5f);
const float* inputs[1] = { x.data() };
for (int t = 0; t < 1000; ++t) {
    cunxonNetworkStepTrain(net, inputs, /*dt_ms=*/1.0f);
    cunxonNetworkInjectNeuromodulator(net, /*DA=*/0, +0.3f);  // reward
}

int n; std::vector<int8_t> y;
cunxonSphereGetReadout(net, s0, nullptr, &n);
y.resize(n);
cunxonSphereGetReadout(net, s0, y.data(), &n);   // y[i] in {-1,0,+1}

cunxonNetworkDestroy(net);
cunxonDestroyContext(ctx);
```

The full 4-sphere demo (`examples/example_4sphere.cu`) builds a VIS+AUD → ASC → MTR brain with feedforward, feedback, lateral and thalamic-like CTC links, drives it with toy stimuli, and reports detection hit-rate.

---

## Python access (ctypes)

The C API is `extern "C"` and can be called from Python via `ctypes`:

```python
import ctypes, numpy as np

lib = ctypes.CDLL("./build/libcunxon.so")

class Ctx(ctypes.c_void_p): pass
class Net(ctypes.c_void_p): pass

lib.cunxonCreateContext.argtypes  = [ctypes.POINTER(Ctx), ctypes.c_int,
                                     ctypes.c_uint64, ctypes.c_uint32]
lib.cunxonNetworkStepTrain.argtypes = [Net, ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_float]

ctx = Ctx()
lib.cunxonCreateContext(ctypes.byref(ctx), 0, 42, 0)
# ... etc.
```

A complete `ctypes` binding is sketched in `examples/python_binding.py` (if present in your tree).

---

## Algorithm 1 → kernel map

| Paper step (Algorithm 1)                                       | cuNxon kernel                                                          |
|-----------------------------------------------------------------|-------------------------------------------------------------------------|
| Oscillator phase advance + theta→gamma PAC                     | `k_oscillator_advance`                                                 |
| Neuromodulator dynamics + 9 receptor activations               | `k_neuromod_update`                                                    |
| Activity statistics (mean &#124;s&#124;, exc_frac, change_rate) | `k_sphere_activity_stats`                                              |
| ChronoPlastic ω-warp + I<sub>syn</sub> scatter (Eq.5–7)         | `k_chrono_warp_and_isyn`                                               |
| Dendritic supralinear gather (Algorithm 4)                     | `k_dendritic_gather`                                                   |
| MSTH 4-loop + DSN α<sub>t</sub> + CTSN h(t) + trinary readout  | `k_membrane_dsn_ctsn_emit`  *(fused, incl. CTSN φ-gain/bias online learning, membrane τ_m fallback, neuron health/death)* |
| STDP + DA-D1/D2 + trinary coincidence + window cap             | `k_plasticity_stdp`                                                    |
| Associative-neighbour diffusion Δw_i += α·Σ(Δw_j − Δw_i)/d_ij  | `k_plasticity_associative`                                             |
| AGMP three-factor (e × DA × astrocyte)                         | `k_plasticity_agmp`                                                    |
| Structural plasticity: integrity prune + stochastic death + resurrection | `k_structural_prune` *(cuRAND Philox)*                       |
| CTC frequency-gated transmission g = (1−c)+½c(1+cosΔφ)         | `k_ctc_gate`, `k_intersphere_project`, `k_intersphere_inject`          |
| Projection plasticity (Hebbian)                                | `k_proj_plasticity`                                                    |
| **Application layer (Algorithm 8)**: store / recall / sequence | host-side, drives `k_membrane_*` via the orchestrator                  |
| **Aigarth ITU (paper §VIII)**: mutate / evaluate / select       | `k_aigarth_mutate_weights` (per-weight cuRAND Philox Gaussian)         |

---

## Reproducing the brain

The paper's NAS-optimised "healthy brain" has four spheres VIS / AUD / ASC / MTR with **(input + hidden + output)** sizes and seven inter-sphere edges. `examples/example_4sphere.cu` builds a topologically faithful (scaled-down) version: two sensory spheres VIS and AUD feedforward into a larger association sphere ASC; ASC feedforwards into a narrow motor sphere MTR; MTR sends a feedback (efference-copy) projection to ASC; ASC sends thalamic-like top-down projections back to VIS and AUD; and VIS↔AUD have a lateral cross-sensory binding link in the theta band.

To replicate the paper's exact NAS configuration, plug your own NAS-found `(n_in, n_hid, n_out)` triples into `fill_sphere_params(...)` and the link parameters into `fill_link_params(...)`.

---

## Persistence

`cunxonNetworkSave(net, "brain.cunxon")` writes a binary file with the magic header `CUNXONV1`, then per-sphere parameter blocks, sphere name, port-interface arrays (sensory / relay-in / relay-out / readout), full synapse arrays (including silent/modulatory flags, integrity, and ChronoPlastic ω), and the per-neuron learned DSN kernels and CTSN φ-gain/bias arrays, followed by inter-sphere projection matrices. `cunxonNetworkLoad(net, path)` is the exact inverse: it expects an empty `cunxonNetwork_t`, calls `AddSphere` / `AddLink` to rebuild topology, uploads every saved field, and finalises the network. Round-tripped networks are bit-identical for weights and learned kernels. See `docs/API.md` §6 for the full layout.

---

## Limitations / known caveats

* The intra-step activity-stat reductions involve a small host roundtrip per sphere; very-large brains may benefit from keeping the device-resident scalars on-device until the end of the step.
* Aigarth's mutation rates are interpreted atm as Gaussian σ rather than as ternary +1/-1/0 sampling probabilities — see `docs/API.md` §7 for the precise semantics.

---

## Licence

Core Neuraxon: Licensed under MIT License (permissive, no restrictions)

Aigarth Hybrid Features: If you implement the Aigarth hybrid features described in our paper, you MUST comply with the Aigarth License, which includes:

❌ NO military use of any kind
❌ NO use by military-affiliated entities
❌ NO dual-use applications with military potential
The standalone Neuraxon implementation (without Aigarth integration) has no such restrictions.

---

## Citation

If you use cuNxon in research, please cite the underlying papers:

```bibtex
@article{Vivancos-Sanchez-2026neuraxon2,
    title={Neuraxon v2.0: A New Neural Growth \& Computation Blueprint},
    author={David Vivancos and Jose Sanchez},
    year={2026},
    journal={ResearchGate Preprint},
    institution={Artificiology Research, UNIR University, Qubic Science},
    url={https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint}
}

```

and acknowledge `cuNxon` as the CUDA implementation.
