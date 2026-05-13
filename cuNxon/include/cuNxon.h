/* =============================================================================
 *  cuNxon.h  -  CUDA library API for MultiNeuraxon 2.0
 *
 *  GPU-accelerated build / train / inference for MultiNeuraxon 2.0 networks
 *  as defined in:
 *     - Vivancos & Sanchez (2026)  "Neuraxon v2.0 - A New Neural Growth
 *       Computation Blueprint" (ICMLT 2026)
 *     - Vivancos & Sanchez (2026)  "Multi-Neuraxon: a modular multi-area
 *       architecture for AGI"
 *     - Reference Python implementation: MultiNeuraxon2.py
 *
 *  Implements the full Algorithm 1 pipeline per simulation step:
 *      (1) ChronoPlastic synaptic time warping  (omega_t, f-trace, z-trace)
 *      (2) Supralinear dendritic branch integration
 *      (3) MSTH 4-loop temporal homeostasis  (ultrafast / fast / medium / slow)
 *      (4) DSN dynamic decay  (alpha_t = sigmoid(CausalConv1D(X)))
 *      (5) CTSN complemented trinary state  (s_tilde = U + h)
 *      (6) Trinary readout                    (-1 / 0 / +1)
 *      (7) STDP + DA-gated + associative plasticity
 *      (8) AGMP astrocyte-gated multi-timescale plasticity
 *      (9) Inter-sphere CTC-gated projection  (g = (1-c) + c * 0.5*(1+cos dphi))
 *
 *  Design:
 *      - Multi-sphere "brain": each sphere is a complete Neuraxon v2.0 network
 *      - Structure-of-Arrays layout on device for coalesced access
 *      - Sparse synapse storage (post-indexed CSR-like) for plasticity locality
 *      - Embarrassingly parallel across spheres (per-sphere CUDA stream)
 *      - cuRAND XORWOW for spontaneous firing & structural plasticity
 *
 *  License: MIT-style.  Algorithms reflect the cited papers.
 *  Author bindings:  cuNxon is an independent CUDA port of MultiNeuraxon 2.0;
 *  the underlying architecture is by D. Vivancos & J. Sanchez (Qubic Science).
 *
 * =============================================================================
 *  PUBLIC C API  (also usable from C++ via extern "C")
 * ============================================================================= */
#ifndef CUNXON_H
#define CUNXON_H

#include <stddef.h>
#include <stdint.h>

#include "cuNxon_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 *  Library version & error handling
 * ----------------------------------------------------------------------------- */
#define CUNXON_VERSION_MAJOR 1
#define CUNXON_VERSION_MINOR 0
#define CUNXON_VERSION_PATCH 0

/** Return a human-readable string for a cuNxon status code. */
CUNXON_API const char* cunxonGetStatusString(cunxonStatus_t status);

/** Return library version as packed int (major*10000 + minor*100 + patch). */
CUNXON_API const char* cunxonGetVersion(void);

/** Get last CUDA error message captured by the library (empty when none). */
CUNXON_API const char* cunxonGetLastError(void);


/* =============================================================================
 *  1. CONTEXT
 *      A cunxonContext owns the CUDA device, cuRAND state, default stream,
 *      and a pool of streams (one per sphere) used for embarrassingly parallel
 *      intra-sphere execution.
 * ============================================================================= */

/** Create a cuNxon context on a specific CUDA device. Pass -1 to use device 0.
 *  `flags` is reserved for future expansion (pass 0). */
CUNXON_API cunxonStatus_t cunxonCreateContext(cunxonContext_t* ctx, int device_id,
                                   uint64_t random_seed, uint32_t flags);

/** Destroy a context and free its associated resources (streams, RNG state). */
CUNXON_API cunxonStatus_t cunxonDestroyContext(cunxonContext_t ctx);

/** Synchronize all streams owned by the context. */
CUNXON_API cunxonStatus_t cunxonContextSync(cunxonContext_t ctx);

/** Query a property (int / char[N] / size_t depending on `prop`).
 *  `out_size` is the byte capacity of the receiving buffer; the call returns
 *  CUNXON_ERR_INVALID_ARGUMENT if the buffer is too small. */
CUNXON_API cunxonStatus_t cunxonContextGetProperty(cunxonContext_t ctx,
                                        cunxonProperty_t prop,
                                        void* out_value,
                                        size_t out_size);


/* =============================================================================
 *  2. NETWORK / "BRAIN"
 *      A cunxonNetwork is a multi-sphere brain.  Spheres are added one at a
 *      time, each with its own NetworkParameters.  Inter-sphere links are
 *      added afterwards.  Finalisation allocates all device buffers.
 * ============================================================================= */

/** Create an empty multi-sphere network. */
CUNXON_API cunxonStatus_t cunxonNetworkCreate(cunxonContext_t ctx,
                                   cunxonNetwork_t* net,
                                   const char* name);

/** Destroy a network and free all device memory. */
CUNXON_API cunxonStatus_t cunxonNetworkDestroy(cunxonNetwork_t net);

/** Get default NetworkParameters values (matches MultiNeuraxon2.py defaults). */
CUNXON_API cunxonStatus_t cunxonGetDefaultParameters(cunxonNetworkParameters_t* params);

/** Add a sphere (a full Neuraxon v2.0 module) to the network.
 *  Returns the assigned sphere index in *sphere_id (>=0).
 *
 *  The function:
 *   - allocates device buffers for {input, hidden, output} neurons (SoA)
 *   - builds a Watts-Strogatz small-world synapse topology among hidden neurons
 *     and input->hidden / hidden->output feedforward synapses
 *   - assigns dendritic branches to each incoming synapse
 *   - initialises ChronoPlastic / DSN / CTSN per-neuron state
 *   - initialises MSTH 4-loop state per neuron
 *   - initialises oscillator bank phases and one neuromodulator field
 */
CUNXON_API cunxonStatus_t cunxonNetworkAddSphere(cunxonNetwork_t net,
                                      const char* sphere_name,
                                      cunxonSphereKind_t kind,
                                      const cunxonNetworkParameters_t* params,
                                      int* sphere_id);

/** Define the input/output port mapping for a sphere (which neurons are
 *  externally visible for inter-sphere projections - see paper P1). */
CUNXON_API cunxonStatus_t cunxonNetworkSetSphereInterface(cunxonNetwork_t net,
                                               int sphere_id,
                                               const int* sensory_input_ids,
                                               int n_sensory_inputs,
                                               const int* relay_input_ids,
                                               int n_relay_inputs,
                                               const int* relay_output_ids,
                                               int n_relay_outputs,
                                               const int* readout_output_ids,
                                               int n_readout_outputs);

/** Add a directed inter-sphere link with CTC frequency-gated transmission.
 *  link_params->kind selects {feedforward, feedback, lateral, thalamic_like}.
 *  link_params->coherence_band must match one oscillator band on both ends.
 *  Returns assigned link id (>=0). */
CUNXON_API cunxonStatus_t cunxonNetworkAddLink(cunxonNetwork_t net,
                                    int src_sphere_id,
                                    int dst_sphere_id,
                                    const cunxonLinkParameters_t* link_params,
                                    int* link_id);

/** Finalise the topology and upload everything to the device.
 *  Must be called after all spheres and links are added, before stepping. */
CUNXON_API cunxonStatus_t cunxonNetworkFinalize(cunxonNetwork_t net);

/** Reset all dynamic state (membrane, traces, MSTH, oscillator phases) but
 *  keep the trained weights and learned per-neuron kernels (DSN, CTSN). */
CUNXON_API cunxonStatus_t cunxonNetworkReset(cunxonNetwork_t net);

/** Number of spheres in the network. */
CUNXON_API int cunxonNetworkNumSpheres(cunxonNetwork_t net);


/* =============================================================================
 *  3. EXECUTION
 *      Forward step runs the full Algorithm 1 pipeline.
 *      Train step additionally enables all plasticity rules (STDP + AGMP +
 *      ChronoPlastic kernel learning + DSN kernel learning + CTSN gain learn
 *      + projection plasticity + structural plasticity).
 *      Inference step skips plasticity / structural plasticity entirely.
 * ============================================================================= */

/** One simulation step with NO plasticity ("frozen weights").
 *
 *  NOTE on Neuraxon semantics:  The Multi-Neuraxon paper explicitly states
 *  that "continuous processing enables real-time adjustments WITHOUT discrete
 *  training phases" — i.e., plasticity is always on.  cunxonNetworkStepInfer
 *  is provided for ML-deployment compatibility (frozen-weights inference),
 *  but for applications matching the paper's continuous-learning regime,
 *  prefer cunxonNetworkStepTrain everywhere and use reward injection to
 *  shape behaviour.
 *
 *  ext_inputs is an array of pointers, one per "input-receiving" sphere;
 *  each buffer must hold n_sensory_input_neurons[sphere] floats in [-1,+1].
 *  Pass NULL for spheres that have no external input this step. */
CUNXON_API cunxonStatus_t cunxonNetworkStepInfer(cunxonNetwork_t net,
                                      const float* const* ext_inputs,
                                      float dt_ms);

/** One simulation step with full plasticity enabled — this is the
 *  paper-canonical operating mode.  See cunxonNetworkStepInfer note. */
CUNXON_API cunxonStatus_t cunxonNetworkStepTrain(cunxonNetwork_t net,
                                      const float* const* ext_inputs,
                                      float dt_ms);

/** Run many steps; reward_per_step (optional) phasically biases dopamine and
 *  drives the AGMP eligibility-trace gating. Pass NULL to disable. */
CUNXON_API cunxonStatus_t cunxonNetworkRun(cunxonNetwork_t net,
                                int n_steps,
                                const float* const* ext_inputs_per_step,
                                const float* reward_per_step,
                                float dt_ms,
                                int training);

/** Inject a phasic neuromodulator pulse into the global field (volume
 *  transmission). nm is one of {DA=0, 5HT=1, ACh=2, NA=3}. */
CUNXON_API cunxonStatus_t cunxonNetworkInjectNeuromodulator(cunxonNetwork_t net,
                                                 int nm_index,
                                                 float amount);


/* =============================================================================
 *  4. READOUT  (device -> host)
 * ============================================================================= */

/** Copy the current trinary states of a sphere's readout (output-port)
 *  neurons into a host buffer.  Pass nullptr for out to query size only.
 *  states_out is filled with values in {-1, 0, +1} stored as int8_t. */
CUNXON_API cunxonStatus_t cunxonSphereGetReadout(cunxonNetwork_t net, int sphere_id,
                                      int8_t* states_out, int* n_states);

/** Copy a full sphere snapshot (membrane U, complement h, s_tilde, trinary s,
 *  firing-rate avg, astrocyte state) for inspection / logging.  Each pointer
 *  may be NULL to skip that channel. */
CUNXON_API cunxonStatus_t cunxonSphereSnapshot(cunxonNetwork_t net, int sphere_id,
                                    float*  U_out,
                                    float*  h_out,
                                    float*  stilde_out,
                                    int8_t* s_out,
                                    float*  firing_rate_out,
                                    float*  astrocyte_out,
                                    int*    n_neurons);

/** Energy accumulator (sum of |Isyn| * w_total per step). */
CUNXON_API cunxonStatus_t cunxonNetworkGetEnergy(cunxonNetwork_t net, double* energy_out);


/* =============================================================================
 *  5. PERSISTENCE
 * ============================================================================= */

/** Save complete network state (topology, weights, learned kernels) to file. */
CUNXON_API cunxonStatus_t cunxonNetworkSave(cunxonNetwork_t net, const char* filepath);

/** Load a previously saved network into a freshly created (empty) network.
 *  The context's device must have enough memory for the loaded topology. */
CUNXON_API cunxonStatus_t cunxonNetworkLoad(cunxonNetwork_t net, const char* filepath);


/* =============================================================================
 *  6. AIGARTH HYBRID  (evolutionary structural plasticity)
 *      Optional Aigarth Intelligent Tissue mutation/selection over a
 *      population of identical-topology spheres.  See Neuraxon v2.0 paper §VIII.
 * ============================================================================= */

/** Configure Aigarth evolution for the network (call once before
 *  cunxonNetworkAigarthStep). */
CUNXON_API cunxonStatus_t cunxonNetworkAigarthConfig(cunxonNetwork_t net,
                                          int population_size,
                                          float mutation_prob_fast,
                                          float mutation_prob_slow,
                                          float mutation_prob_meta);

/** Run one Aigarth generation: mutate -> evaluate -> select.
 *  fitness_fn is called with each clone's pointer.  Best clone replaces
 *  the live network. */
CUNXON_API cunxonStatus_t cunxonNetworkAigarthStep(cunxonNetwork_t net,
                                        cunxonFitnessFn_t fitness_fn,
                                        void* user_data);


/* =============================================================================
 *  7. SPHERE LAYERS  (Multi-Neuraxon paper §P4: hierarchical tiers)
 *
 *  Optional grouping of spheres into processing tiers (sensory → association
 *  → motor).  Layers are pure metadata: they affect bookkeeping and
 *  introspection only, not the step kernels.
 * ============================================================================= */

/** Create a named layer; returns the layer index via *layer_id. */
CUNXON_API cunxonStatus_t cunxonNetworkAddLayer(cunxonNetwork_t net,
                                     const char* layer_name,
                                     int depth,
                                     int* layer_id);

/** Add a sphere to a layer.  A sphere may belong to multiple layers. */
CUNXON_API cunxonStatus_t cunxonNetworkAddSphereToLayer(cunxonNetwork_t net,
                                             int layer_id,
                                             int sphere_id);

/** Number of layers defined so far. */
CUNXON_API int cunxonNetworkNumLayers(cunxonNetwork_t net);

/** Query a layer's contents.  Pass `sphere_ids = NULL` for a size query. */
CUNXON_API cunxonStatus_t cunxonNetworkGetLayer(cunxonNetwork_t net,
                                     int layer_id,
                                     char* name_out, int name_buf_len,
                                     int* depth_out,
                                     int* sphere_ids, int* n_spheres);


/* =============================================================================
 *  8. PATTERN APPLICATION LAYER  (Multi-Neuraxon paper Algorithm 8)
 *
 *  High-level helpers for storing input patterns and recalling them via
 *  partial-cue completion.  Sits on top of the core Step/Run API; patterns
 *  live on the host and drive the sensory input ports of a chosen sphere.
 *
 *  Typical use:
 *      cunxonNetworkStorePattern (net, sphere, "cat",   pat_cat,  16, 20);
 *      cunxonNetworkStorePattern (net, sphere, "dog",   pat_dog,  16, 20);
 *      ...
 *      cunxonNetworkRecallPattern(net, sphere, "cat",   16, 0.5f, 20, readout);
 * ============================================================================= */

/** Store a pattern by name and consolidate it via `n_present_steps` of
 *  training on the given sphere's sensory inputs. */
CUNXON_API cunxonStatus_t cunxonNetworkStorePattern(cunxonNetwork_t net,
                                         int sphere_id,
                                         const char* pattern_name,
                                         const float* pattern,
                                         int pattern_len,
                                         int n_present_steps,
                                         float dt_ms);

/** Recall a previously-stored pattern from a partial cue: each input value
 *  is masked to 0 with probability `mask_fraction` before being presented.
 *  After `n_settle_steps` of inference, the sphere's readout is copied into
 *  `readout_out` (caller-allocated, size = sphere readout neurons).        */
CUNXON_API cunxonStatus_t cunxonNetworkRecallPattern(cunxonNetwork_t net,
                                          int sphere_id,
                                          const char* pattern_name,
                                          int pattern_len,
                                          float mask_fraction,
                                          int n_settle_steps,
                                          float dt_ms,
                                          int8_t* readout_out,
                                          int* n_readout);

/** Present a sequence of patterns repeatedly (helper for sequence training). */
CUNXON_API cunxonStatus_t cunxonNetworkTrainSequence(cunxonNetwork_t net,
                                          int sphere_id,
                                          const float* const* patterns,
                                          int n_patterns,
                                          int pattern_len,
                                          int n_repetitions,
                                          int n_steps_per_pattern,
                                          float dt_ms);

/** List the names of stored patterns.  Pass NULL for a size query. */
CUNXON_API cunxonStatus_t cunxonNetworkListPatterns(cunxonNetwork_t net,
                                         char* names_out, int names_buf_len,
                                         int* n_patterns);

/** Clear the entire pattern memory. */
CUNXON_API cunxonStatus_t cunxonNetworkClearPatterns(cunxonNetwork_t net);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUNXON_H */
