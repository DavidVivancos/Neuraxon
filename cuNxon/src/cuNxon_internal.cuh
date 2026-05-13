/* =============================================================================
 *  cuNxon_internal.cuh  -  Internal device-side structures (SoA layout)
 *                          and kernel declarations.  Not part of public API.
 * ============================================================================= */
#ifndef CUNXON_INTERNAL_CUH
#define CUNXON_INTERNAL_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>

#include "../include/cuNxon.h"


/* ============================================================================
 *  DEVICE-SIDE STRUCTURES  (Structure-of-Arrays for coalesced access)
 *
 *  A sphere has:
 *      n_total = n_in + n_hid + n_out   neurons, indexed 0..n_total-1
 *      n_syn                            synapses (CSR-like, grouped by post-id)
 *
 *  Neuron arrays are length n_total.
 *  Synapse arrays are length n_syn, sorted by post_id so that all synapses
 *  for a given post-neuron form a contiguous slice.  The CSR offset array
 *  post_synapse_offset[i .. i+1] gives that slice for neuron i.
 * ============================================================================ */

struct NeuronArraysDev {
    /* identity */
    int8_t*  type;                   /* cunxonNeuronType_t */
    int8_t*  is_active;              /* 1 = alive          */
    /* membrane / state                                                    */
    float*   U;                      /* membrane potential                  */
    float*   adaptation;
    float*   autoreceptor;
    int8_t*  s;                      /* current trinary state {-1,0,+1}     */
    int8_t*  s_prev;
    /* CTSN                                                                 */
    float*   complement_h;
    float*   s_tilde;
    float*   ctsn_phi_gain;
    float*   ctsn_phi_bias;
    /* DSN per-neuron causal-conv kernel + ring buffer                     */
    float*   dsn_kernel;             /* [n_total * K]                       */
    float*   dsn_buffer;             /* [n_total * K]  ring of past inputs  */
    int*     dsn_head;               /* ring index per neuron               */
    float*   dsn_alpha;              /* last computed alpha_t               */
    /* dendritic branches                                                  */
    float*   branch_pot;             /* [n_total * B]                       */
    float*   branch_sum;             /* [n_total * B] scratch per step      */
    /* metabotropic (modulatory) synaptic drive (per-neuron, per-step)     */
    float*   modulatory_pot;         /* [n_total]   scratch, cleared/step  */
    /* homeostasis                                                          */
    float*   firing_rate_avg;
    /* MSTH 4-loop state                                                   */
    float*   msth_ultrafast;
    float*   msth_fast;
    float*   msth_medium;            /* gain (clipped to [0.5, 2.0])        */
    float*   msth_slow;
    /* AGMP astrocyte                                                       */
    float*   astrocyte;
    /* health                                                               */
    float*   health;
    /* RNG state per neuron (XORWOW)                                       */
    curandState_t* rng;
};

struct SynapseArraysDev {
    int*    pre_id;
    int*    post_id;
    int*    branch_id;               /* in [0, B-1]                         */
    /* weights                                                              */
    float*  w_fast;
    float*  w_slow;
    float*  w_meta;
    /* flags                                                                */
    int8_t* is_silent;
    int8_t* is_modulatory;
    int8_t* synapse_type;
    float*  integrity;
    /* STDP traces                                                          */
    float*  pre_trace;
    float*  post_trace;
    /* ChronoPlastic                                                        */
    float*  chrono_fast_trace;
    float*  chrono_slow_trace;
    float*  chrono_omega;            /* current learned warp factor         */
    /* AGMP eligibility                                                     */
    float*  eligibility;
    /* CSR: synapses are sorted by post_id; offset[i..i+1] = synapses of i */
    int*    post_offset;             /* size n_total + 1                    */
};

struct OscillatorBankDev {
    /* per-band global phase and amplitude (one set per sphere) */
    float* phase;                    /* [BAND_COUNT]                        */
    float* freq_hz;                  /* [BAND_COUNT]                        */
    float* amp;                      /* [BAND_COUNT]                        */
    float  pac_strength;
};

struct NeuromodFieldDev {
    /* tonic+phasic per modulator: index 0..3 = DA, 5HT, ACh, NA           */
    float* tonic;                    /* [4] */
    float* phasic;                   /* [4] */
    /* receptor activations (post-nonlinearity).  9 subtypes:
     *   0:D1 1:D2 2:5HT1A 3:5HT2A 4:5HT4 5:M1 6:M2 7:beta1 8:alpha2       */
    float* recept;                   /* [9] */
};

struct SphereDev {
    int                 n_in, n_hid, n_out, n_total;
    int                 n_syn;
    int                 n_branches;
    int                 dsn_K;
    cunxonSphereKind_t  kind;
    NeuronArraysDev     N;
    SynapseArraysDev    S;
    OscillatorBankDev   O;
    NeuromodFieldDev    M;
    /* per-sphere copy of parameters (so device kernels need not chase host) */
    cunxonNetworkParameters_t* p_dev;
    /* port indices (device side) */
    int *port_in_sensory,   n_port_in_sensory;
    int *port_in_relay,     n_port_in_relay;
    int *port_out_relay,    n_port_out_relay;
    int *port_out_readout,  n_port_out_readout;
    /* external input buffer (filled host-side each step) */
    float* ext_in;
    /* energy accumulator (float, reduced into host) */
    float* d_energy;
    /* per-sphere CUDA stream for parallel sphere execution */
    cudaStream_t stream;
};

struct LinkDev {
    int                     src, dst;
    cunxonLinkParameters_t  p;
    int                     n_src_ports, n_dst_ports;   /* relay_out_src x in_dst */
    /* projection weight matrix [n_dst_ports * n_src_ports], row-major     */
    float* W;
    /* delay ring of source-port trinary states; depth = delay_steps+1     */
    int8_t* delay_ring;
    int     delay_head;
    /* cached CTC gate value (scalar) and last phase difference            */
    float*  g_ctc;
    /* contribution buffer for this step (length = n_dst_ports)            */
    float*  contrib;
    /* combined destination port ids = [port_in_relay] + [port_in_sensory] *
     * Length = n_dst_ports; used by k_intersphere_inject so contributions  *
     * can also drive sensory-only sphere inputs (e.g., top-down attention).*/
    int*    dst_port_ids;
};


/* ============================================================================
 *  Host-side network impl
 * ============================================================================ */
struct SphereLayerHost {
    std::string         name;
    int                 depth;
    std::vector<int>    sphere_ids;
};

struct cunxonNetworkImpl_ {
    cunxonContext_t                      ctx;
    std::string                          name;
    std::vector<SphereDev>               spheres;
    std::vector<LinkDev>                 links;
    std::vector<cunxonNetworkParameters_t> sphere_params_host;  /* mirrors */
    std::vector<std::string>             sphere_names_host;     /* mirrors */
    int                                  finalized;
    /* energy accumulator on host */
    double                               energy_total;
    /* Aigarth */
    int                                  aigarth_population;
    float                                aigarth_mut_fast, aigarth_mut_slow, aigarth_mut_meta;
    /* Step counter (incremented on every Step* call; used to seed RNG in
     * stochastic structural plasticity).                                     */
    uint64_t                             step_index;
    /* Hierarchical sphere layers (metadata only) */
    std::vector<SphereLayerHost>         layers;
    /* Pattern memory:  name -> per-sphere input vector + sphere of origin   */
    struct PatternEntry { int sphere_id; std::vector<float> values; };
    std::map<std::string, PatternEntry>  patterns;
};


/* ============================================================================
 *  KERNEL DECLARATIONS  (defined across cuNxon_*.cu files)
 * ============================================================================ */

/* --- Oscillator bank: advance global phases for one sphere ----------------- */
__global__ void k_oscillator_advance(OscillatorBankDev O, float dt_ms);

/* --- Neuromodulator field: tonic relaxation + receptor nonlinearity -------- */
__global__ void k_neuromod_update(NeuromodFieldDev M,
                                  const cunxonNetworkParameters_t* p,
                                  float dt_ms,
                                  float mean_activity,
                                  float exc_fraction,
                                  float state_change_rate);

/* --- Per-sphere reductions for neuromod stats: returns (mean|s|, exc%, dr) - */
__global__ void k_sphere_activity_stats(const NeuronArraysDev N, int n_total,
                                        float* out_mean_abs,
                                        float* out_exc_frac,
                                        float* out_change_rate);

/* --- Algorithm 1 step 1: ChronoPlastic synaptic time-warp ----------------- */
__global__ void k_chrono_warp_and_isyn(NeuronArraysDev N, SynapseArraysDev S,
                                       int n_syn,
                                       const cunxonNetworkParameters_t* p,
                                       float dt_ms,
                                       float* d_energy);

/* --- Algorithm 1 step 2: gather dendritic branch sums per neuron ---------- */
__global__ void k_dendritic_gather(NeuronArraysDev N, const SynapseArraysDev S,
                                   int n_total, int n_branches,
                                   const cunxonNetworkParameters_t* p);

/* --- Algorithm 1 steps 3-7: MSTH + DSN + membrane + CTSN + trinary readout */
__global__ void k_membrane_dsn_ctsn_emit(NeuronArraysDev N,
                                         const NeuromodFieldDev M,
                                         OscillatorBankDev O,
                                         int n_total,
                                         const cunxonNetworkParameters_t* p,
                                         const float* ext_in,
                                         int  n_in,
                                         float dt_ms);

/* --- Algorithm 1 step 8: STDP + DA-gated + associative ---------------------- */
__global__ void k_plasticity_stdp(SynapseArraysDev S, const NeuronArraysDev N,
                                  int n_syn,
                                  const NeuromodFieldDev M,
                                  const cunxonNetworkParameters_t* p,
                                  float dt_ms);

/* --- Algorithm 1 step 9: AGMP eligibility-trace plasticity ------------------ */
__global__ void k_plasticity_agmp(SynapseArraysDev S, const NeuronArraysDev N,
                                  int n_syn,
                                  const NeuromodFieldDev M,
                                  const cunxonNetworkParameters_t* p,
                                  float dt_ms);

/* --- Algorithm 1 step 9b: associative-neighbour diffusion ------------------- */
__global__ void k_plasticity_associative(SynapseArraysDev S,
                                         int n_total, int n_syn,
                                         const cunxonNetworkParameters_t* p,
                                         float dt_ms);

/* --- Structural plasticity: prune integrity-low synapses,
       stochastically kill weak ones, stochastically reform silent ones --- */
__global__ void k_structural_prune(SynapseArraysDev S, int n_syn,
                                   const cunxonNetworkParameters_t* p,
                                   float dt_ms,
                                   uint64_t step_seed);

/* --- Inter-sphere: compute CTC gate from oscillator phase difference ------ */
__global__ void k_ctc_gate(const OscillatorBankDev Osrc,
                           const OscillatorBankDev Odst,
                           int band,
                           float coherence,
                           float* g_out);

/* --- Inter-sphere: produce projection contributions into dst input slot --- */
__global__ void k_intersphere_project(const NeuronArraysDev Nsrc,
                                      const int* src_port_ids,
                                      int n_src_ports,
                                      const float* W,
                                      int n_dst_ports,
                                      const float* g_ctc,
                                      float gain,
                                      float bias,
                                      float transmission_threshold,
                                      float* contrib_out);

/* --- Inter-sphere: inject contributions into dst sphere's port inputs ----- */
__global__ void k_intersphere_inject(float* dst_ext_in,
                                     const int* dst_port_ids,
                                     int n_dst_ports,
                                     const float* contrib);

/* --- Inter-sphere projection plasticity (Hebbian dwp = eta * s_src * s_tgt) - */
__global__ void k_proj_plasticity(float* W,
                                  const NeuronArraysDev Nsrc,
                                  const int* src_port_ids,
                                  int n_src_ports,
                                  const NeuronArraysDev Ndst,
                                  const int* dst_port_ids,
                                  int n_dst_ports,
                                  float eta,
                                  float decay,
                                  float clip);

/* --- Reset helpers --------------------------------------------------------- */
__global__ void k_reset_neuron_dynamic(NeuronArraysDev N, int n_total);


/* ============================================================================
 *  HOST-SIDE HELPERS  (defined in cuNxon.cu / build helpers)
 * ============================================================================ */

namespace cunxon_internal {

/* Build initial Watts-Strogatz small-world synapse list (host side) for a
 * hidden recurrent layer, plus dense input->hidden and hidden->output edges.
 * Returns a flat list of (pre, post, branch) triples, sorted by post for CSR. */
void build_initial_topology(const cunxonNetworkParameters_t& p,
                            int sphere_seed,
                            std::vector<int>& pre,
                            std::vector<int>& post,
                            std::vector<int>& branch,
                            std::vector<int>& post_offset);

/* Allocate all device buffers for a sphere from the host-side topology lists.
 * Initialises weights, traces, MSTH state, oscillator phases, RNG. */
cunxonStatus_t allocate_sphere(SphereDev& sd,
                               const cunxonNetworkParameters_t& p,
                               const std::vector<int>& pre,
                               const std::vector<int>& post,
                               const std::vector<int>& branch,
                               const std::vector<int>& post_offset,
                               uint64_t base_seed,
                               cudaStream_t stream);

void free_sphere(SphereDev& sd);

cunxonStatus_t allocate_link(LinkDev& ld,
                             int n_src_ports, int n_dst_ports,
                             const cunxonLinkParameters_t& p,
                             uint64_t seed);

void free_link(LinkDev& ld);

/* Convenience macro for kernel launch error checks. */
#define CUNXON_CUDA_CHECK(call)                                                \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            ::cunxon_internal::set_cuda_error(cudaGetErrorString(_e),          \
                                              __FILE__, __LINE__);             \
            return CUNXON_ERR_CUDA;                                            \
        }                                                                      \
    } while (0)

void set_cuda_error(const char* msg, const char* file, int line);

}  /* namespace cunxon_internal */

#endif  /* CUNXON_INTERNAL_CUH */
