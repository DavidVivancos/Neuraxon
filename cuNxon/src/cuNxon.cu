/* =============================================================================
 *  cuNxon.cu  -  Host-side library implementation
 *
 *  Implements the public C API declared in cuNxon.h on top of the CUDA
 *  kernels defined in cuNxon_kernels.cu, cuNxon_plasticity.cu, and
 *  cuNxon_intersphere.cu.
 *
 *  Compile with nvcc (separable compilation off; everything in one TU is fine
 *  because each kernel is __global__).
 * ============================================================================= */

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <new>
#include <fstream>
#include <cmath>
#include <map>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuNxon_internal.cuh"

/* =============================================================================
 *  Context implementation
 * ============================================================================= */
struct cunxonContextImpl_ {
    int        device_id;
    uint64_t   seed;
    cudaStream_t default_stream;
    /* per-sphere streams are owned by spheres themselves */
};


/* ----------- last-error storage (thread-unsafe but simple) ------------------ */
namespace cunxon_internal {
    static thread_local std::string g_last_error;
    void set_cuda_error(const char* msg, const char* file, int line) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "CUDA error at %s:%d: %s",
                      file, line, msg ? msg : "(unknown)");
        g_last_error = buf;
    }
}

extern "C" CUNXON_API const char* cunxonGetLastError(void) {
    return cunxon_internal::g_last_error.c_str();
}

extern "C" CUNXON_API const char* cunxonGetStatusString(cunxonStatus_t s)
{
    switch (s) {
    case CUNXON_OK:                    return "OK";
    case CUNXON_ERR_INVALID_ARGUMENT:  return "INVALID_ARGUMENT";
    case CUNXON_ERR_OUT_OF_MEMORY:     return "OUT_OF_MEMORY";
    case CUNXON_ERR_NOT_FINALIZED:     return "NOT_FINALIZED";
    case CUNXON_ERR_ALREADY_FINALIZED: return "ALREADY_FINALIZED";
    case CUNXON_ERR_INVALID_SPHERE:    return "INVALID_SPHERE";
    case CUNXON_ERR_INVALID_LINK:      return "INVALID_LINK";
    case CUNXON_ERR_CUDA:              return "CUDA_ERROR";
    case CUNXON_ERR_FILE_IO:           return "FILE_IO";
    case CUNXON_ERR_INCOMPATIBLE_SAVE: return "INCOMPATIBLE_SAVE";
    case CUNXON_ERR_NOT_IMPLEMENTED:   return "NOT_IMPLEMENTED";
    default:                           return "UNKNOWN";
    }
}

extern "C" CUNXON_API const char* cunxonGetVersion(void) {
    return "cuNxon 0.1.0";
}


/* =============================================================================
 *  RNG-init kernel for curand states (one per neuron)
 * ============================================================================= */
__global__ void k_init_curand(curandState_t* states, int n, uint64_t seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, (uint64_t)i, 0ULL, &states[i]);
}


/* =============================================================================
 *  Public: context create/destroy
 * ============================================================================= */
extern "C" CUNXON_API
cunxonStatus_t cunxonCreateContext(cunxonContext_t* out_ctx, int device_id,
                                   uint64_t random_seed, uint32_t /*flags*/)
{
    if (!out_ctx) return CUNXON_ERR_INVALID_ARGUMENT;
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        cunxon_internal::set_cuda_error("no CUDA device", __FILE__, __LINE__);
        return CUNXON_ERR_CUDA;
    }
    int chosen = (device_id < 0) ? 0 : device_id;
    if (chosen >= n_dev) return CUNXON_ERR_INVALID_ARGUMENT;
    CUNXON_CUDA_CHECK(cudaSetDevice(chosen));

    auto* impl = new (std::nothrow) cunxonContextImpl_;
    if (!impl) return CUNXON_ERR_OUT_OF_MEMORY;
    impl->device_id = chosen;
    impl->seed      = random_seed ? random_seed : 0xC0FFEE12345678ULL;
    CUNXON_CUDA_CHECK(cudaStreamCreateWithFlags(&impl->default_stream,
                                                cudaStreamNonBlocking));
    *out_ctx = impl;
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonDestroyContext(cunxonContext_t ctx)
{
    if (!ctx) return CUNXON_ERR_INVALID_ARGUMENT;
    cudaStreamDestroy(ctx->default_stream);
    delete ctx;
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonContextSync(cunxonContext_t ctx)
{
    if (!ctx) return CUNXON_ERR_INVALID_ARGUMENT;
    CUNXON_CUDA_CHECK(cudaDeviceSynchronize());
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonContextGetProperty(cunxonContext_t ctx,
                                        cunxonProperty_t prop,
                                        void* out_value,
                                        size_t out_size)
{
    if (!ctx || !out_value || out_size == 0) return CUNXON_ERR_INVALID_ARGUMENT;
    cudaDeviceProp dp{};
    CUNXON_CUDA_CHECK(cudaGetDeviceProperties(&dp, ctx->device_id));
    switch (prop) {
    case CUNXON_PROP_DEVICE_ID:
        if (out_size < sizeof(int)) return CUNXON_ERR_INVALID_ARGUMENT;
        *(int*)out_value = ctx->device_id; break;
    case CUNXON_PROP_DEVICE_NAME:
        /* dp.name is up to 256 chars NUL-terminated; honour out_size as cap. */
        std::strncpy((char*)out_value, dp.name, out_size);
        ((char*)out_value)[out_size - 1] = '\0';
        break;
    case CUNXON_PROP_COMPUTE_CAPABILITY:
        if (out_size < sizeof(int)) return CUNXON_ERR_INVALID_ARGUMENT;
        *(int*)out_value = dp.major*10 + dp.minor; break;
    case CUNXON_PROP_TOTAL_GLOBAL_MEM: {
        if (out_size < sizeof(size_t)) return CUNXON_ERR_INVALID_ARGUMENT;
        size_t free_b=0, tot_b=0;
        cudaMemGetInfo(&free_b, &tot_b);
        *(size_t*)out_value = tot_b; break;
    }
    case CUNXON_PROP_FREE_GLOBAL_MEM: {
        if (out_size < sizeof(size_t)) return CUNXON_ERR_INVALID_ARGUMENT;
        size_t free_b=0, tot_b=0;
        cudaMemGetInfo(&free_b, &tot_b);
        *(size_t*)out_value = free_b; break;
    }
    case CUNXON_PROP_MAX_THREADS_PER_BLK:
        if (out_size < sizeof(int)) return CUNXON_ERR_INVALID_ARGUMENT;
        *(int*)out_value = dp.maxThreadsPerBlock; break;
    case CUNXON_PROP_SM_COUNT:
        if (out_size < sizeof(int)) return CUNXON_ERR_INVALID_ARGUMENT;
        *(int*)out_value = dp.multiProcessorCount; break;
    default:
        return CUNXON_ERR_INVALID_ARGUMENT;
    }
    return CUNXON_OK;
}


/* =============================================================================
 *  Default parameters (match MultiNeuraxon2.py NetworkParameters defaults)
 * ============================================================================= */
extern "C" CUNXON_API
cunxonStatus_t cunxonGetDefaultParameters(cunxonNetworkParameters_t* p)
{
    if (!p) return CUNXON_ERR_INVALID_ARGUMENT;
    std::memset(p, 0, sizeof(*p));
    p->num_input_neurons       = 5;
    p->num_hidden_neurons      = 20;
    p->num_output_neurons      = 5;
    p->num_dendritic_branches  = 3;
    p->dendritic_spike_threshold     = 0.4f;
    p->dendritic_supralinear_gamma   = 1.3f;

    p->ws_k    = 6;
    p->ws_beta = 0.3f;

    p->membrane_time_constant        = 20.0f;
    p->firing_threshold_excitatory   = 0.4f;
    p->firing_threshold_inhibitory   = -0.4f;
    p->adaptation_tau                = 100.0f;
    p->autoreceptor_tau              = 200.0f;
    p->spontaneous_firing_rate       = 0.02f;
    p->neuron_health_decay           = 0.001f;

    p->target_firing_rate            = 0.2f;
    p->homeostatic_rate              = 0.0005f;
    p->firing_rate_alpha             = 0.01f;
    p->threshold_mod_k               = 0.3f;

    p->msth_ultrafast_tau            = 5.0f;
    p->msth_ultrafast_ceiling        = 2.0f;
    p->msth_fast_tau                 = 2000.0f;
    p->msth_fast_gain                = 0.1f;
    p->msth_medium_tau               = 300000.0f;
    p->msth_medium_gain              = 0.001f;
    p->msth_slow_tau                 = 3600000.0f;
    p->msth_slow_gain                = 0.0001f;

    p->dsn_kernel_size               = 4;
    p->dsn_enabled                   = 1;
    p->dsn_bias                      = 0.0f;
    p->dsn_learn_enabled             = 0;
    p->dsn_learn_lr                  = 0.01f;
    p->dsn_target_sensitivity        = 4.0f;
    p->dsn_target_bias               = 2.0f;
    p->dsn_kernel_clip               = 5.0f;

    p->ctsn_rho                      = 0.9f;
    p->ctsn_enabled                  = 1;
    p->ctsn_phi_gain                 = 0.5f;
    p->ctsn_phi_bias                 = 0.0f;
    p->ctsn_learn_enabled            = 0;
    p->ctsn_learn_lr                 = 0.005f;
    p->ctsn_phi_gain_clip            = 5.0f;
    p->ctsn_phi_bias_clip            = 5.0f;

    p->tau_fast = 5.0f;   p->tau_slow = 50.0f;
    p->tau_meta = 1000.0f; p->tau_stdp = 20.0f;
    p->w_fast_init_min = -0.8f; p->w_fast_init_max = 0.8f;
    p->w_slow_init_min = -0.4f; p->w_slow_init_max = 0.4f;
    p->w_meta_init_min = -0.3f; p->w_meta_init_max = 0.3f;

    p->chrono_alpha_f      = 0.95f;
    p->chrono_alpha_s      = 0.99f;
    p->chrono_lambda_f     = 0.15f;
    p->chrono_lambda_s     = 0.08f;
    p->chrono_enabled      = 1;
    p->chrono_trace_clip   = 10.0f;
    p->chrono_gate_norm    = 10.0f;
    p->chrono_raw_clip     = 8.0f;
    p->chrono_omega_min    = 0.05f;
    p->chrono_omega_max    = 0.95f;
    p->chrono_omega_smoothing = 0.2f;

    p->agmp_lambda_e = 0.95f;
    p->agmp_lambda_a = 0.999f;
    p->agmp_eta      = 0.005f;
    p->agmp_enabled  = 1;

    p->learning_rate                 = 0.01f;
    p->stdp_window                   = 20.0f;
    p->associative_alpha             = 0.005f;
    p->synapse_integrity_threshold   = 0.1f;
    p->synapse_formation_prob        = 0.05f;
    p->synapse_death_prob            = 0.01f;
    p->neuron_death_threshold        = 0.1f;

    p->dopamine_baseline       = 0.15f;
    p->serotonin_baseline      = 0.15f;
    p->acetylcholine_baseline  = 0.15f;
    p->norepinephrine_baseline = 0.15f;
    p->tau_tonic               = 5000.0f;
    p->tau_phasic              = 200.0f;
    p->neuromod_release_rate   = 0.02f;
    p->receptor_concentration_cap = 1.0f;

    /* Hz: infraslow, slow, theta, alpha, beta, gamma */
    p->osc_freq[CUNXON_BAND_INFRASLOW] = 0.05f;
    p->osc_freq[CUNXON_BAND_SLOW]      = 0.5f;
    p->osc_freq[CUNXON_BAND_THETA]     = 6.0f;
    p->osc_freq[CUNXON_BAND_ALPHA]     = 10.0f;
    p->osc_freq[CUNXON_BAND_BETA]      = 20.0f;
    p->osc_freq[CUNXON_BAND_GAMMA]     = 40.0f;
    p->osc_amplitude     = 1.0f;
    p->osc_pac_strength  = 0.1f;   /* keep oscillator drive subthreshold */
    p->random_seed_offset = 0;
    return CUNXON_OK;
}


/* =============================================================================
 *  Network create/destroy
 * ============================================================================= */
extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkCreate(cunxonContext_t ctx,
                                   cunxonNetwork_t* out,
                                   const char* name)
{
    if (!ctx || !out) return CUNXON_ERR_INVALID_ARGUMENT;
    auto* n = new (std::nothrow) cunxonNetworkImpl_;
    if (!n) return CUNXON_ERR_OUT_OF_MEMORY;
    n->ctx = ctx;
    n->name = name ? name : "MultiNeuraxon";
    n->finalized = 0;
    n->energy_total = 0.0;
    n->aigarth_population = 0;
    n->aigarth_mut_fast = n->aigarth_mut_slow = n->aigarth_mut_meta = 0.f;
    n->step_index = 0;
    *out = n;
    return CUNXON_OK;
}

namespace cunxon_internal {
void free_sphere(SphereDev& sd);
void free_link  (LinkDev& ld);
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkDestroy(cunxonNetwork_t net)
{
    if (!net) return CUNXON_ERR_INVALID_ARGUMENT;
    for (auto& s : net->spheres) cunxon_internal::free_sphere(s);
    for (auto& l : net->links)   cunxon_internal::free_link(l);
    delete net;
    return CUNXON_OK;
}

extern "C" CUNXON_API
int cunxonNetworkNumSpheres(cunxonNetwork_t net) {
    return net ? (int)net->spheres.size() : 0;
}


/* =============================================================================
 *  Topology construction (host-side)
 *  Watts-Strogatz small-world over hidden neurons, plus dense
 *  input->hidden and hidden->output feedforward edges.
 * ============================================================================= */
namespace cunxon_internal {

void build_initial_topology(const cunxonNetworkParameters_t& p,
                            int sphere_seed,
                            std::vector<int>& pre,
                            std::vector<int>& post,
                            std::vector<int>& branch,
                            std::vector<int>& post_offset)
{
    pre.clear(); post.clear(); branch.clear();
    const int n_in = p.num_input_neurons;
    const int n_hid = p.num_hidden_neurons;
    const int n_out = p.num_output_neurons;
    const int n_total = n_in + n_hid + n_out;
    const int B = std::max(1, p.num_dendritic_branches);

    std::mt19937 rng((uint32_t)(sphere_seed ^ 0xA5A5A5A5u));
    std::uniform_int_distribution<int> ud_branch(0, B - 1);
    std::uniform_real_distribution<float> uf(0.f, 1.f);

    /* Input neurons indices: [0, n_in)
       Hidden neurons indices: [n_in, n_in + n_hid)
       Output neurons indices: [n_in + n_hid, n_total)                       */

    /* 1) Dense input -> hidden */
    for (int i = 0; i < n_in; ++i) {
        for (int j = n_in; j < n_in + n_hid; ++j) {
            pre.push_back(i);
            post.push_back(j);
            branch.push_back(ud_branch(rng));
        }
    }

    /* 2) Watts-Strogatz on hidden block:
          ring lattice with k neighbours, then rewire with prob beta.        */
    int K = std::max(2, std::min(n_hid - 1, p.ws_k));
    if (K & 1) K += 1;   /* ensure even k */
    int half = K / 2;
    for (int i = 0; i < n_hid; ++i) {
        for (int d = 1; d <= half; ++d) {
            int j_ring = (i + d) % n_hid;
            int target = j_ring;
            if (uf(rng) < p.ws_beta) {
                /* rewire to a random non-self target */
                int r = (int)(uf(rng) * (n_hid - 1));
                if (r >= i) r += 1;
                target = r;
            }
            int pre_id  = n_in + i;
            int post_id = n_in + target;
            if (pre_id == post_id) continue;
            pre.push_back(pre_id);
            post.push_back(post_id);
            branch.push_back(ud_branch(rng));
            /* symmetric edge (back) */
            pre.push_back(post_id);
            post.push_back(pre_id);
            branch.push_back(ud_branch(rng));
        }
    }

    /* 3) Dense hidden -> output */
    for (int i = n_in; i < n_in + n_hid; ++i) {
        for (int j = n_in + n_hid; j < n_total; ++j) {
            pre.push_back(i);
            post.push_back(j);
            branch.push_back(ud_branch(rng));
        }
    }

    /* 4) Sort by post_id for CSR layout; build the offset table.           */
    int n_syn = (int)pre.size();
    std::vector<int> order(n_syn);
    for (int k = 0; k < n_syn; ++k) order[k] = k;
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return post[a] < post[b]; });

    std::vector<int> pre_s(n_syn), post_s(n_syn), br_s(n_syn);
    for (int k = 0; k < n_syn; ++k) {
        pre_s [k] = pre [order[k]];
        post_s[k] = post[order[k]];
        br_s  [k] = branch[order[k]];
    }
    pre.swap(pre_s); post.swap(post_s); branch.swap(br_s);

    post_offset.assign(n_total + 1, 0);
    for (int k = 0; k < n_syn; ++k) post_offset[post[k] + 1] += 1;
    for (int i = 1; i <= n_total; ++i) post_offset[i] += post_offset[i-1];
}


/* =============================================================================
 *  Sphere allocation
 * ============================================================================= */
template <typename T>
static T* cu_malloc_init(size_t count, T fill, cudaStream_t stream)
{
    T* d = nullptr;
    if (cudaMalloc(&d, count * sizeof(T)) != cudaSuccess) return nullptr;
    /* memset only valid for 0; do per-T fill via thrust would pull deps.
     * Instead, use a host-side temp buffer.                                */
    if (fill == T(0)) {
        cudaMemsetAsync(d, 0, count * sizeof(T), stream);
    } else {
        std::vector<T> h(count, fill);
        cudaMemcpyAsync(d, h.data(), count*sizeof(T), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    return d;
}

#define CHKM(ptr) do { if(!(ptr)) return CUNXON_ERR_OUT_OF_MEMORY; } while(0)

cunxonStatus_t allocate_sphere(SphereDev& sd,
                               const cunxonNetworkParameters_t& p,
                               const std::vector<int>& pre,
                               const std::vector<int>& post,
                               const std::vector<int>& branch,
                               const std::vector<int>& post_offset,
                               uint64_t base_seed,
                               cudaStream_t stream)
{
    sd.n_in        = p.num_input_neurons;
    sd.n_hid       = p.num_hidden_neurons;
    sd.n_out       = p.num_output_neurons;
    sd.n_total     = sd.n_in + sd.n_hid + sd.n_out;
    sd.n_syn       = (int)pre.size();
    sd.n_branches  = std::max(1, p.num_dendritic_branches);
    sd.dsn_K       = std::max(1, p.dsn_kernel_size);

    NeuronArraysDev& N = sd.N;
    int n = sd.n_total;
    N.type        = cu_malloc_init<int8_t>(n, 0, stream);          CHKM(N.type);
    N.is_active   = cu_malloc_init<int8_t>(n, 1, stream);          CHKM(N.is_active);
    N.U           = cu_malloc_init<float>(n, 0.f, stream);         CHKM(N.U);
    N.adaptation  = cu_malloc_init<float>(n, 0.f, stream);         CHKM(N.adaptation);
    N.autoreceptor= cu_malloc_init<float>(n, 0.f, stream);         CHKM(N.autoreceptor);
    N.s           = cu_malloc_init<int8_t>(n, 0, stream);          CHKM(N.s);
    N.s_prev      = cu_malloc_init<int8_t>(n, 0, stream);          CHKM(N.s_prev);
    N.complement_h= cu_malloc_init<float>(n, 0.f, stream);         CHKM(N.complement_h);
    N.s_tilde     = cu_malloc_init<float>(n, 0.f, stream);         CHKM(N.s_tilde);
    N.ctsn_phi_gain = cu_malloc_init<float>(n, p.ctsn_phi_gain, stream);  CHKM(N.ctsn_phi_gain);
    N.ctsn_phi_bias = cu_malloc_init<float>(n, p.ctsn_phi_bias, stream);  CHKM(N.ctsn_phi_bias);

    /* DSN kernel: triangular init normalised to L1=1 */
    int K = sd.dsn_K;
    std::vector<float> h_kernel(n * K, 0.f);
    for (int i = 0; i < n; ++i) {
        float s_ = 0.f;
        for (int k = 0; k < K; ++k) { h_kernel[i*K + k] = (float)(k + 1); s_ += h_kernel[i*K + k]; }
        if (s_ > 0.f) for (int k = 0; k < K; ++k) h_kernel[i*K + k] /= s_;
    }
    if (cudaMalloc(&N.dsn_kernel, n*K*sizeof(float)) != cudaSuccess) return CUNXON_ERR_OUT_OF_MEMORY;
    cudaMemcpyAsync(N.dsn_kernel, h_kernel.data(), n*K*sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    N.dsn_buffer  = cu_malloc_init<float>(n * K, 0.f, stream);     CHKM(N.dsn_buffer);
    N.dsn_head    = cu_malloc_init<int>(n, 0, stream);             CHKM(N.dsn_head);
    N.dsn_alpha   = cu_malloc_init<float>(n, 0.5f, stream);        CHKM(N.dsn_alpha);

    N.branch_pot  = cu_malloc_init<float>(n * sd.n_branches, 0.f, stream); CHKM(N.branch_pot);
    N.branch_sum  = cu_malloc_init<float>(n * sd.n_branches, 0.f, stream); CHKM(N.branch_sum);
    N.modulatory_pot = cu_malloc_init<float>(n, 0.f, stream);              CHKM(N.modulatory_pot);

    N.firing_rate_avg = cu_malloc_init<float>(n, p.target_firing_rate, stream); CHKM(N.firing_rate_avg);
    N.msth_ultrafast  = cu_malloc_init<float>(n, 0.f, stream); CHKM(N.msth_ultrafast);
    N.msth_fast       = cu_malloc_init<float>(n, 0.f, stream); CHKM(N.msth_fast);
    N.msth_medium     = cu_malloc_init<float>(n, 1.f, stream); CHKM(N.msth_medium);
    N.msth_slow       = cu_malloc_init<float>(n, 0.f, stream); CHKM(N.msth_slow);
    N.astrocyte       = cu_malloc_init<float>(n, 0.f, stream); CHKM(N.astrocyte);
    N.health          = cu_malloc_init<float>(n, 1.f, stream); CHKM(N.health);

    if (cudaMalloc(&N.rng, n*sizeof(curandState_t)) != cudaSuccess) return CUNXON_ERR_OUT_OF_MEMORY;
    int tb = 256;
    int nb = (n + tb - 1) / tb;
    k_init_curand<<<nb, tb, 0, stream>>>(N.rng, n, base_seed);

    /* Per-neuron type assignment */
    std::vector<int8_t> h_type(n);
    for (int i = 0; i < sd.n_in;            ++i) h_type[i] = (int8_t)CUNXON_NEURON_INPUT;
    for (int i = sd.n_in; i < sd.n_in + sd.n_hid; ++i) h_type[i] = (int8_t)CUNXON_NEURON_HIDDEN;
    for (int i = sd.n_in + sd.n_hid; i < n; ++i)      h_type[i] = (int8_t)CUNXON_NEURON_OUTPUT;
    cudaMemcpyAsync(N.type, h_type.data(), n*sizeof(int8_t), cudaMemcpyHostToDevice, stream);

    /* ---- Synapses --------------------------------------------------------- */
    SynapseArraysDev& S = sd.S;
    int ns = sd.n_syn;

    auto cuda_copy = [&](auto* d_ptr_addr, const auto& host_vec) {
        using T = typename std::remove_reference<decltype(*host_vec.data())>::type;
        if (cudaMalloc(d_ptr_addr, host_vec.size() * sizeof(T)) != cudaSuccess) return false;
        return cudaMemcpyAsync(*d_ptr_addr, host_vec.data(),
                               host_vec.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };

    cuda_copy(&S.pre_id,    pre);
    cuda_copy(&S.post_id,   post);
    cuda_copy(&S.branch_id, branch);
    cuda_copy(&S.post_offset, post_offset);

    /* Random weight init within the configured ranges (host-side rng).      */
    std::mt19937 rng((uint32_t)(base_seed & 0xFFFFFFFFu) ^ 0x12345);
    std::uniform_real_distribution<float> uf(0.f, 1.f);
    std::vector<float> h_wf(ns), h_ws(ns), h_wm(ns);
    std::vector<int8_t> h_sil(ns, 0), h_mod(ns, 0), h_stype(ns);
    for (int k = 0; k < ns; ++k) {
        h_wf[k] = p.w_fast_init_min + (p.w_fast_init_max - p.w_fast_init_min) * uf(rng);
        h_ws[k] = p.w_slow_init_min + (p.w_slow_init_max - p.w_slow_init_min) * uf(rng);
        h_wm[k] = p.w_meta_init_min + (p.w_meta_init_max - p.w_meta_init_min) * uf(rng);
        h_sil[k] = (uf(rng) < 0.1f) ? 1 : 0;
        h_mod[k] = (uf(rng) < 0.2f) ? 1 : 0;
        if (h_sil[k])                                  h_stype[k] = CUNXON_SYN_SILENT;
        else if (h_mod[k])                             h_stype[k] = CUNXON_SYN_METABOTROPIC;
        else if (std::fabs(h_wf[k]) > std::fabs(h_ws[k])) h_stype[k] = CUNXON_SYN_IONOTROPIC_FAST;
        else                                           h_stype[k] = CUNXON_SYN_IONOTROPIC_SLOW;
    }
    cuda_copy(&S.w_fast, h_wf);
    cuda_copy(&S.w_slow, h_ws);
    cuda_copy(&S.w_meta, h_wm);
    cuda_copy(&S.is_silent, h_sil);
    cuda_copy(&S.is_modulatory, h_mod);
    cuda_copy(&S.synapse_type, h_stype);

    S.integrity        = cu_malloc_init<float>(ns, 1.f, stream); CHKM(S.integrity);
    S.pre_trace        = cu_malloc_init<float>(ns, 0.f, stream); CHKM(S.pre_trace);
    S.post_trace       = cu_malloc_init<float>(ns, 0.f, stream); CHKM(S.post_trace);
    S.chrono_fast_trace= cu_malloc_init<float>(ns, 0.f, stream); CHKM(S.chrono_fast_trace);
    S.chrono_slow_trace= cu_malloc_init<float>(ns, 0.f, stream); CHKM(S.chrono_slow_trace);
    S.chrono_omega     = cu_malloc_init<float>(ns, 0.5f, stream); CHKM(S.chrono_omega);
    S.eligibility      = cu_malloc_init<float>(ns, 0.f, stream); CHKM(S.eligibility);

    /* ---- Oscillator bank -------------------------------------------------- */
    OscillatorBankDev& O = sd.O;
    O.pac_strength = p.osc_pac_strength;
    std::vector<float> h_freq(CUNXON_BAND_COUNT), h_amp(CUNXON_BAND_COUNT, p.osc_amplitude);
    std::vector<float> h_phase(CUNXON_BAND_COUNT, 0.f);
    for (int b = 0; b < CUNXON_BAND_COUNT; ++b) h_freq[b] = p.osc_freq[b];
    cuda_copy(&O.phase,  h_phase);
    cuda_copy(&O.freq_hz, h_freq);
    cuda_copy(&O.amp,    h_amp);

    /* ---- Neuromodulator field -------------------------------------------- */
    NeuromodFieldDev& Mf = sd.M;
    std::vector<float> h_tonic(4), h_phasic(4, 0.f);
    h_tonic[0] = p.dopamine_baseline;
    h_tonic[1] = p.serotonin_baseline;
    h_tonic[2] = p.acetylcholine_baseline;
    h_tonic[3] = p.norepinephrine_baseline;
    cuda_copy(&Mf.tonic,   h_tonic);
    cuda_copy(&Mf.phasic,  h_phasic);
    std::vector<float> h_rec(9, 0.f);
    cuda_copy(&Mf.recept,  h_rec);

    /* ---- Params on device ------------------------------------------------- */
    if (cudaMalloc(&sd.p_dev, sizeof(cunxonNetworkParameters_t)) != cudaSuccess)
        return CUNXON_ERR_OUT_OF_MEMORY;
    cudaMemcpyAsync(sd.p_dev, &p, sizeof(p), cudaMemcpyHostToDevice, stream);

    /* ---- External input buffer (length n_total: input ports occupy slot 0..n_in) */
    sd.ext_in = cu_malloc_init<float>(n, 0.f, stream); CHKM(sd.ext_in);

    /* ---- Energy accumulator --------------------------------------------- */
    sd.d_energy = cu_malloc_init<float>(1, 0.f, stream); CHKM(sd.d_energy);

    /* ---- Default ports ---------------------------------------------------- */
    sd.port_in_sensory = nullptr;   sd.n_port_in_sensory = 0;
    sd.port_in_relay   = nullptr;   sd.n_port_in_relay   = 0;
    sd.port_out_relay  = nullptr;   sd.n_port_out_relay  = 0;
    sd.port_out_readout= nullptr;   sd.n_port_out_readout= 0;

    cudaStreamSynchronize(stream);
    return CUNXON_OK;
}

void free_sphere(SphereDev& sd)
{
    auto F = [](void* p){ if (p) cudaFree(p); };
    F(sd.N.type); F(sd.N.is_active); F(sd.N.U); F(sd.N.adaptation);
    F(sd.N.autoreceptor); F(sd.N.s); F(sd.N.s_prev); F(sd.N.complement_h);
    F(sd.N.s_tilde); F(sd.N.ctsn_phi_gain); F(sd.N.ctsn_phi_bias);
    F(sd.N.dsn_kernel); F(sd.N.dsn_buffer); F(sd.N.dsn_head); F(sd.N.dsn_alpha);
    F(sd.N.branch_pot); F(sd.N.branch_sum); F(sd.N.modulatory_pot);
    F(sd.N.firing_rate_avg);
    F(sd.N.msth_ultrafast); F(sd.N.msth_fast); F(sd.N.msth_medium); F(sd.N.msth_slow);
    F(sd.N.astrocyte); F(sd.N.health); F(sd.N.rng);

    F(sd.S.pre_id); F(sd.S.post_id); F(sd.S.branch_id);
    F(sd.S.w_fast); F(sd.S.w_slow); F(sd.S.w_meta);
    F(sd.S.is_silent); F(sd.S.is_modulatory); F(sd.S.synapse_type);
    F(sd.S.integrity); F(sd.S.pre_trace); F(sd.S.post_trace);
    F(sd.S.chrono_fast_trace); F(sd.S.chrono_slow_trace); F(sd.S.chrono_omega);
    F(sd.S.eligibility); F(sd.S.post_offset);

    F(sd.O.phase); F(sd.O.freq_hz); F(sd.O.amp);
    F(sd.M.tonic); F(sd.M.phasic); F(sd.M.recept);

    F(sd.p_dev);
    F(sd.ext_in);
    F(sd.d_energy);

    F(sd.port_in_sensory); F(sd.port_in_relay);
    F(sd.port_out_relay);  F(sd.port_out_readout);

    if (sd.stream) cudaStreamDestroy(sd.stream);
    sd.stream = 0;
}


/* =============================================================================
 *  Link allocation
 * ============================================================================= */
cunxonStatus_t allocate_link(LinkDev& ld,
                             int n_src_ports, int n_dst_ports,
                             const cunxonLinkParameters_t& p,
                             uint64_t seed)
{
    ld.p           = p;
    ld.n_src_ports = n_src_ports;
    ld.n_dst_ports = n_dst_ports;

    size_t W_count = (size_t)n_dst_ports * n_src_ports;
    if (cudaMalloc(&ld.W, W_count * sizeof(float)) != cudaSuccess) return CUNXON_ERR_OUT_OF_MEMORY;

    /* Initialise W with stronger range than per-synapse weights:
     * inter-sphere fan-in is typically modest (≤ ~30 source ports) and each
     * source state is trinary {-1,0,+1}, so with sparse activity (~25%) the
     * effective sum has std ≈ σ_W · √(N·p).  σ = 0.5 keeps salient signals
     * comfortably above the input-neuron quantisation threshold (0.20) while
     * still allowing noise-dominated patterns to settle near 0.                */
    std::mt19937 rng((uint32_t)(seed & 0xFFFFFFFFu) ^ 0xABCDEF1u);
    std::uniform_real_distribution<float> uf(0.f, 0.5f);
    std::vector<float> hW(W_count);
    for (size_t k = 0; k < W_count; ++k) {
        float w = uf(rng);
        if (p.allow_negative_weights && (rng() & 1)) w = -w;
        if (p.topology == CUNXON_TOPO_SPARSE) {
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            if (u01(rng) > p.sparse_prob) w = 0.f;
        }
        hW[k] = w;
    }
    /* Optional row-wise L1 normalisation */
    if (p.normalize_rows) {
        for (int d = 0; d < n_dst_ports; ++d) {
            float s = 0.f;
            for (int s_p = 0; s_p < n_src_ports; ++s_p) s += std::fabs(hW[d*n_src_ports + s_p]);
            if (s > 1e-8f)
                for (int s_p = 0; s_p < n_src_ports; ++s_p) hW[d*n_src_ports + s_p] /= s;
        }
    }
    cudaMemcpy(ld.W, hW.data(), W_count*sizeof(float), cudaMemcpyHostToDevice);

    int depth = std::max(1, p.delay_steps + 1);
    ld.delay_head = 0;
    if (cudaMalloc(&ld.delay_ring, (size_t)depth * n_src_ports * sizeof(int8_t)) != cudaSuccess)
        return CUNXON_ERR_OUT_OF_MEMORY;
    cudaMemset(ld.delay_ring, 0, (size_t)depth * n_src_ports * sizeof(int8_t));

    if (cudaMalloc(&ld.g_ctc, sizeof(float)) != cudaSuccess) return CUNXON_ERR_OUT_OF_MEMORY;
    float one = 1.f;
    cudaMemcpy(ld.g_ctc, &one, sizeof(float), cudaMemcpyHostToDevice);

    if (cudaMalloc(&ld.contrib, n_dst_ports * sizeof(float)) != cudaSuccess) return CUNXON_ERR_OUT_OF_MEMORY;
    cudaMemset(ld.contrib, 0, n_dst_ports * sizeof(float));

    return CUNXON_OK;
}

void free_link(LinkDev& ld)
{
    auto F = [](void* p){ if (p) cudaFree(p); };
    F(ld.W); F(ld.delay_ring); F(ld.g_ctc); F(ld.contrib); F(ld.dst_port_ids);
}

}  /* namespace cunxon_internal */


/* =============================================================================
 *  Public network configuration
 * ============================================================================= */
extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkAddSphere(cunxonNetwork_t net,
                                      const char* sphere_name,
                                      cunxonSphereKind_t kind,
                                      const cunxonNetworkParameters_t* params,
                                      int* out_id)
{
    if (!net || !params || !out_id) return CUNXON_ERR_INVALID_ARGUMENT;
    if (net->finalized) return CUNXON_ERR_ALREADY_FINALIZED;

    SphereDev sd{};
    sd.kind = kind;
    cudaStreamCreateWithFlags(&sd.stream, cudaStreamNonBlocking);

    /* Build initial CSR topology on host */
    std::vector<int> pre, post, branch, post_offset;
    int sphere_seed = (int)net->spheres.size() + 1;
    cunxon_internal::build_initial_topology(*params, sphere_seed,
                                            pre, post, branch, post_offset);

    uint64_t base_seed = net->ctx->seed + params->random_seed_offset + sphere_seed;
    cunxonStatus_t st = cunxon_internal::allocate_sphere(sd, *params, pre, post,
                                                         branch, post_offset,
                                                         base_seed, sd.stream);
    if (st != CUNXON_OK) {
        cunxon_internal::free_sphere(sd);
        return st;
    }

    net->sphere_params_host.push_back(*params);
    net->sphere_names_host.emplace_back(sphere_name ? sphere_name : "");
    *out_id = (int)net->spheres.size();
    net->spheres.push_back(sd);
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkSetSphereInterface(cunxonNetwork_t net,
                                               int sphere_id,
                                               const int* sensory_input_ids, int n_si,
                                               const int* relay_input_ids,   int n_ri,
                                               const int* relay_output_ids,  int n_ro,
                                               const int* readout_output_ids,int n_rd)
{
    if (!net) return CUNXON_ERR_INVALID_ARGUMENT;
    if (sphere_id < 0 || sphere_id >= (int)net->spheres.size())
        return CUNXON_ERR_INVALID_SPHERE;
    SphereDev& sd = net->spheres[sphere_id];

    auto upload_idx = [&](const int* h, int n, int** d_out, int* n_out) {
        if (*d_out) { cudaFree(*d_out); *d_out = nullptr; }
        *n_out = n;
        if (n <= 0 || !h) return;
        cudaMalloc(d_out, n*sizeof(int));
        cudaMemcpy(*d_out, h, n*sizeof(int), cudaMemcpyHostToDevice);
    };
    upload_idx(sensory_input_ids,  n_si, &sd.port_in_sensory,  &sd.n_port_in_sensory);
    upload_idx(relay_input_ids,    n_ri, &sd.port_in_relay,    &sd.n_port_in_relay);
    upload_idx(relay_output_ids,   n_ro, &sd.port_out_relay,   &sd.n_port_out_relay);
    upload_idx(readout_output_ids, n_rd, &sd.port_out_readout, &sd.n_port_out_readout);
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkAddLink(cunxonNetwork_t net,
                                    int src, int dst,
                                    const cunxonLinkParameters_t* lp,
                                    int* out_id)
{
    if (!net || !lp || !out_id) return CUNXON_ERR_INVALID_ARGUMENT;
    if (src < 0 || src >= (int)net->spheres.size()) return CUNXON_ERR_INVALID_SPHERE;
    if (dst < 0 || dst >= (int)net->spheres.size()) return CUNXON_ERR_INVALID_SPHERE;

    LinkDev ld{};
    ld.src = src;
    ld.dst = dst;
    int n_src = net->spheres[src].n_port_out_relay;
    int n_dst = net->spheres[dst].n_port_in_relay
              + net->spheres[dst].n_port_in_sensory;
    if (n_src <= 0 || n_dst <= 0) return CUNXON_ERR_INVALID_ARGUMENT;

    uint64_t seed = net->ctx->seed ^ (uint64_t)(src*1000003 + dst*99991 + (int)net->links.size());
    cunxonStatus_t st = cunxon_internal::allocate_link(ld, n_src, n_dst, *lp, seed);
    if (st != CUNXON_OK) { cunxon_internal::free_link(ld); return st; }

    /* Build the combined destination port id array:
     *   first  n_port_in_relay     entries from dst.port_in_relay
     *   then   n_port_in_sensory   entries from dst.port_in_sensory
     * This lets inject reach sensory-only spheres (e.g., thalamic feedback).*/
    {
        SphereDev& sd = net->spheres[dst];
        std::vector<int> h_ids(n_dst);
        if (sd.n_port_in_relay > 0)
            cudaMemcpy(h_ids.data(), sd.port_in_relay,
                       sd.n_port_in_relay * sizeof(int),
                       cudaMemcpyDeviceToHost);
        if (sd.n_port_in_sensory > 0)
            cudaMemcpy(h_ids.data() + sd.n_port_in_relay, sd.port_in_sensory,
                       sd.n_port_in_sensory * sizeof(int),
                       cudaMemcpyDeviceToHost);
        if (cudaMalloc(&ld.dst_port_ids, n_dst * sizeof(int)) != cudaSuccess) {
            cunxon_internal::free_link(ld);
            return CUNXON_ERR_OUT_OF_MEMORY;
        }
        cudaMemcpy(ld.dst_port_ids, h_ids.data(), n_dst * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    *out_id = (int)net->links.size();
    net->links.push_back(ld);
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkFinalize(cunxonNetwork_t net)
{
    if (!net) return CUNXON_ERR_INVALID_ARGUMENT;
    if (net->finalized) return CUNXON_ERR_ALREADY_FINALIZED;
    net->finalized = 1;
    return CUNXON_OK;
}


/* =============================================================================
 *  STEP orchestration
 *
 *  For each sphere s (on its own stream):
 *      (A) clear branch_sum and ext_in
 *      (B) k_oscillator_advance(O_s)
 *      (C) k_chrono_warp_and_isyn  (over its synapses)
 *      (D) k_dendritic_gather
 *
 *  Inter-sphere (over net->ctx->default_stream after intra A-D synced):
 *      (E) for each link L: k_ctc_gate, k_intersphere_project,
 *                            k_intersphere_inject
 *
 *  For each sphere s:
 *      (F) k_sphere_activity_stats -> small host download or pinned readout
 *      (G) k_neuromod_update
 *      (H) k_membrane_dsn_ctsn_emit  (uses ext_in possibly modified by links)
 *
 *  If training:
 *      (I) k_plasticity_stdp
 *      (J) k_plasticity_agmp
 *      (K) k_structural_prune  (optional, less frequent)
 *      (L) for each link: k_proj_plasticity
 *
 *  Note: for true scalability we'd batch the activity-stats reductions to
 *  device-resident scalars instead of host roundtrips, but for clarity we
 *  perform the reduction in-place via atomicAdd into a device float.
 * ============================================================================= */

#define KCFG(N) dim3((N) + 255) / 256, dim3(256)

static cunxonStatus_t step_impl(cunxonNetworkImpl_* net,
                                const float* const* ext_inputs,
                                float dt_ms,
                                int training)
{
    if (!net) return CUNXON_ERR_INVALID_ARGUMENT;
    if (!net->finalized) return CUNXON_ERR_NOT_FINALIZED;

    /* (A) clear scratch buffers and copy external inputs ------------------ */
    for (size_t s = 0; s < net->spheres.size(); ++s) {
        SphereDev& sd = net->spheres[s];
        cudaMemsetAsync(sd.N.branch_sum, 0,
                        sd.n_total * sd.n_branches * sizeof(float), sd.stream);
        cudaMemsetAsync(sd.N.modulatory_pot, 0,
                        sd.n_total * sizeof(float), sd.stream);
        cudaMemsetAsync(sd.ext_in, 0, sd.n_total * sizeof(float), sd.stream);
        /* clear per-step energy accumulator */
        cudaMemsetAsync(sd.d_energy, 0, sizeof(float), sd.stream);
        /* Place external inputs onto the SENSORY input port slots */
        if (ext_inputs && ext_inputs[s] && sd.n_port_in_sensory > 0) {
            /* host_data is dense [n_port_in_sensory] -> scatter via a tiny kernel */
            /* For brevity, copy to a scratch buffer and use a 1-thread scatter.   */
            std::vector<int> h_ids(sd.n_port_in_sensory);
            cudaMemcpy(h_ids.data(), sd.port_in_sensory,
                       sd.n_port_in_sensory * sizeof(int),
                       cudaMemcpyDeviceToHost);
            for (int k = 0; k < sd.n_port_in_sensory; ++k) {
                float v = ext_inputs[s][k];
                cudaMemcpyAsync(&sd.ext_in[h_ids[k]], &v, sizeof(float),
                                cudaMemcpyHostToDevice, sd.stream);
            }
        }
    }

    /* (B) advance oscillators ----------------------------------------------- */
    for (auto& sd : net->spheres)
        k_oscillator_advance<<<1, 1, 0, sd.stream>>>(sd.O, dt_ms);

    /* (C) ChronoPlastic warp + Isyn scatter into branches ------------------ */
    for (auto& sd : net->spheres) {
        int n_syn = sd.n_syn;
        if (n_syn == 0) continue;
        int tb = 256, nb = (n_syn + tb - 1)/tb;
        k_chrono_warp_and_isyn<<<nb, tb, 0, sd.stream>>>(
            sd.N, sd.S, n_syn, sd.p_dev, dt_ms, sd.d_energy);
    }

    /* (D) dendritic gather --------------------------------------------------- */
    for (auto& sd : net->spheres) {
        int n = sd.n_total;
        int tb = 256, nb = (n + tb - 1)/tb;
        k_dendritic_gather<<<nb, tb, 0, sd.stream>>>(
            sd.N, sd.S, n, sd.n_branches, sd.p_dev);
    }

    /* sync per-sphere streams before inter-sphere phase */
    for (auto& sd : net->spheres) cudaStreamSynchronize(sd.stream);

    /* (E) inter-sphere CTC + projection + inject ---------------------------- */
    cudaStream_t inter_stream = net->ctx->default_stream;
    for (auto& ld : net->links) {
        SphereDev& src = net->spheres[ld.src];
        SphereDev& dst = net->spheres[ld.dst];
        k_ctc_gate<<<1, 1, 0, inter_stream>>>(
            src.O, dst.O, (int)ld.p.coherence_band, ld.p.coherence_strength, ld.g_ctc);
        int tb = 128;
        k_intersphere_project<<<ld.n_dst_ports, tb, 0, inter_stream>>>(
            src.N, src.port_out_relay, ld.n_src_ports,
            ld.W, ld.n_dst_ports, ld.g_ctc, ld.p.gain, ld.p.bias,
            ld.p.transmission_threshold, ld.contrib);
        int nb = (ld.n_dst_ports + tb - 1) / tb;
        /* dst ports = relay_in + sensory_in (concatenated indices on host) */
        /* We need a flat list of dst port ids:                              */
        /* For simplicity inject directly into ext_in via dst.port_in_relay  */
        k_intersphere_inject<<<nb, tb, 0, inter_stream>>>(
            dst.ext_in, ld.dst_port_ids, ld.n_dst_ports, ld.contrib);
    }
    cudaStreamSynchronize(inter_stream);

    /* (F+G+H) Activity stats -> neuromod -> membrane (per sphere) ---------- */
    for (auto& sd : net->spheres) {
        /* tiny device buffer for the 3 reductions */
        float* d_stats; cudaMalloc(&d_stats, 3*sizeof(float));
        cudaMemsetAsync(d_stats, 0, 3*sizeof(float), sd.stream);

        int tb = 256, nb = std::max(1, (sd.n_total + tb - 1)/tb);
        k_sphere_activity_stats<<<nb, tb, 0, sd.stream>>>(
            sd.N, sd.n_total, d_stats, d_stats+1, d_stats+2);
        float h_stats[3] = {0,0,0};
        cudaMemcpyAsync(h_stats, d_stats, 3*sizeof(float),
                        cudaMemcpyDeviceToHost, sd.stream);
        cudaStreamSynchronize(sd.stream);
        cudaFree(d_stats);
        float mean_abs = (sd.n_total > 0) ? h_stats[0] / sd.n_total : 0.f;
        float exc_frac = (sd.n_total > 0) ? h_stats[1] / sd.n_total : 0.f;
        float chg_rate = (sd.n_total > 0) ? h_stats[2] / sd.n_total : 0.f;

        k_neuromod_update<<<1, 1, 0, sd.stream>>>(
            sd.M, sd.p_dev, dt_ms, mean_abs, exc_frac, chg_rate);

        int n = sd.n_total;
        int nb2 = (n + tb - 1)/tb;
        k_membrane_dsn_ctsn_emit<<<nb2, tb, 0, sd.stream>>>(
            sd.N, sd.M, sd.O, n, sd.p_dev, sd.ext_in, sd.n_in, dt_ms);
    }

    if (training) {
        for (auto& sd : net->spheres) {
            int n_syn = sd.n_syn;
            if (n_syn == 0) continue;
            int tb = 256, nb = (n_syn + tb - 1)/tb;
            k_plasticity_stdp<<<nb, tb, 0, sd.stream>>>(sd.S, sd.N, n_syn, sd.M,
                                                       sd.p_dev, dt_ms);
            /* Associative-neighbour diffusion reads eligibility (= dw_stdp)
             * before AGMP overwrites it with its own running e-trace.       */
            k_plasticity_associative<<<nb, tb, 0, sd.stream>>>(
                sd.S, sd.n_total, n_syn, sd.p_dev, dt_ms);
            k_plasticity_agmp<<<nb, tb, 0, sd.stream>>>(sd.S, sd.N, n_syn, sd.M,
                                                       sd.p_dev, dt_ms);
            /* Structural prune+death+formation: stochastic, RNG-seeded
             * by (step_index, sphere_id) so behaviour is reproducible.    */
            uint64_t step_seed = (net->ctx ? net->ctx->seed : 0xC0FFEEULL)
                               ^ (net->step_index * 0x9E3779B97F4A7C15ULL)
                               ^ ((uint64_t)(&sd - &net->spheres[0])
                                  * 0xBF58476D1CE4E5B9ULL);
            k_structural_prune<<<nb, tb, 0, sd.stream>>>(
                sd.S, n_syn, sd.p_dev, dt_ms, step_seed);
        }
        /* Inter-sphere projection plasticity */
        for (auto& ld : net->links) {
            if (ld.p.plasticity_rate <= 0.f) continue;
            SphereDev& src = net->spheres[ld.src];
            SphereDev& dst = net->spheres[ld.dst];
            dim3 block(16, 16);
            dim3 grid((ld.n_src_ports + 15)/16, (ld.n_dst_ports + 15)/16);
            k_proj_plasticity<<<grid, block, 0, net->ctx->default_stream>>>(
                ld.W, src.N, src.port_out_relay, ld.n_src_ports,
                dst.N, ld.dst_port_ids, ld.n_dst_ports,
                ld.p.plasticity_rate, ld.p.weight_decay, ld.p.weight_clip);
        }
    }

    /* Sync everything before returning so the user can inspect readouts. */
    cudaDeviceSynchronize();

    /* Aggregate per-sphere energy into host accumulator (one float per
     * sphere, copied back after synchronisation).                          */
    for (auto& sd : net->spheres) {
        float e_step = 0.f;
        cudaMemcpy(&e_step, sd.d_energy, sizeof(float), cudaMemcpyDeviceToHost);
        net->energy_total += (double)e_step;
    }

    net->step_index++;
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkStepInfer(cunxonNetwork_t net,
                                      const float* const* ext_inputs,
                                      float dt_ms)
{
    return step_impl(net, ext_inputs, dt_ms, /*training=*/0);
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkStepTrain(cunxonNetwork_t net,
                                      const float* const* ext_inputs,
                                      float dt_ms)
{
    return step_impl(net, ext_inputs, dt_ms, /*training=*/1);
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkRun(cunxonNetwork_t net, int n_steps,
                                const float* const* ext_inputs_per_step,
                                const float* reward_per_step,
                                float dt_ms, int training)
{
    if (!net || n_steps <= 0) return CUNXON_ERR_INVALID_ARGUMENT;
    for (int t = 0; t < n_steps; ++t) {
        const float* const* this_inputs = nullptr;
        if (ext_inputs_per_step) {
            this_inputs = (const float* const*)(ext_inputs_per_step
                                                + t * net->spheres.size());
        }
        if (reward_per_step) {
            /* phasic DA bias for AGMP: inject a small pulse onto sphere 0's
             * neuromodulator field (downstream spheres see it via the
             * inter-sphere CTC paths and global volume transmission).      */
            cunxonNetworkInjectNeuromodulator(net, /*DA=*/0, reward_per_step[t]);
        }
        cunxonStatus_t st = step_impl(net, this_inputs, dt_ms, training);
        if (st != CUNXON_OK) return st;
    }
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkInjectNeuromodulator(cunxonNetwork_t net,
                                                 int nm_index, float amount)
{
    if (!net) return CUNXON_ERR_INVALID_ARGUMENT;
    if (nm_index < 0 || nm_index > 3) return CUNXON_ERR_INVALID_ARGUMENT;
    for (auto& sd : net->spheres) {
        /* Read current phasic, add amount, write back. */
        float v = 0.f;
        cudaMemcpy(&v, sd.M.phasic + nm_index, sizeof(float), cudaMemcpyDeviceToHost);
        v += amount;
        cudaMemcpy(sd.M.phasic + nm_index, &v, sizeof(float), cudaMemcpyHostToDevice);
    }
    return CUNXON_OK;
}


/* =============================================================================
 *  Readout helpers
 * ============================================================================= */
extern "C" CUNXON_API
cunxonStatus_t cunxonSphereGetReadout(cunxonNetwork_t net, int sphere_id,
                                      int8_t* out, int* n_out)
{
    if (!net || !n_out) return CUNXON_ERR_INVALID_ARGUMENT;
    if (sphere_id < 0 || sphere_id >= (int)net->spheres.size())
        return CUNXON_ERR_INVALID_SPHERE;
    SphereDev& sd = net->spheres[sphere_id];

    if (!out) { *n_out = sd.n_port_out_readout; return CUNXON_OK; }

    /* gather sd.s[port_out_readout[i]] for i in [0, n_port_out_readout) */
    std::vector<int> ids(sd.n_port_out_readout);
    cudaMemcpy(ids.data(), sd.port_out_readout,
               sd.n_port_out_readout*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int8_t> tmp(sd.n_total);
    cudaMemcpy(tmp.data(), sd.N.s, sd.n_total*sizeof(int8_t),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < sd.n_port_out_readout; ++i) out[i] = tmp[ids[i]];
    *n_out = sd.n_port_out_readout;
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonSphereSnapshot(cunxonNetwork_t net, int sphere_id,
                                    float* U, float* h, float* st_,
                                    int8_t* s, float* fr, float* asto,
                                    int* n_neurons)
{
    if (!net || !n_neurons) return CUNXON_ERR_INVALID_ARGUMENT;
    if (sphere_id < 0 || sphere_id >= (int)net->spheres.size())
        return CUNXON_ERR_INVALID_SPHERE;
    SphereDev& sd = net->spheres[sphere_id];
    *n_neurons = sd.n_total;
    auto cpy_f = [&](float* host, float* dev){
        if (host) cudaMemcpy(host, dev, sd.n_total*sizeof(float),
                             cudaMemcpyDeviceToHost);
    };
    auto cpy_b = [&](int8_t* host, int8_t* dev){
        if (host) cudaMemcpy(host, dev, sd.n_total*sizeof(int8_t),
                             cudaMemcpyDeviceToHost);
    };
    cpy_f(U,    sd.N.U);
    cpy_f(h,    sd.N.complement_h);
    cpy_f(st_,  sd.N.s_tilde);
    cpy_b(s,    sd.N.s);
    cpy_f(fr,   sd.N.firing_rate_avg);
    cpy_f(asto, sd.N.astrocyte);
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkGetEnergy(cunxonNetwork_t net, double* energy_out)
{
    if (!net || !energy_out) return CUNXON_ERR_INVALID_ARGUMENT;
    *energy_out = net->energy_total;
    return CUNXON_OK;
}


/* =============================================================================
 *  Network reset
 * ============================================================================= */
extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkReset(cunxonNetwork_t net)
{
    if (!net) return CUNXON_ERR_INVALID_ARGUMENT;
    for (auto& sd : net->spheres) {
        int tb = 256, nb = (sd.n_total + tb - 1)/tb;
        if (nb > 0)
            k_reset_neuron_dynamic<<<nb, tb, 0, sd.stream>>>(sd.N, sd.n_total);
        /* neuron scratch */
        cudaMemsetAsync(sd.N.branch_pot, 0,
                        sd.n_total * sd.n_branches * sizeof(float), sd.stream);
        cudaMemsetAsync(sd.N.branch_sum, 0,
                        sd.n_total * sd.n_branches * sizeof(float), sd.stream);
        cudaMemsetAsync(sd.N.modulatory_pot, 0,
                        sd.n_total * sizeof(float), sd.stream);
        /* DSN ring buffer (keep learned kernels, just clear the input history) */
        cudaMemsetAsync(sd.N.dsn_buffer, 0,
                        sd.n_total * sd.dsn_K * sizeof(float), sd.stream);
        cudaMemsetAsync(sd.N.dsn_head, 0, sd.n_total * sizeof(int), sd.stream);
        cudaMemsetAsync(sd.N.dsn_alpha, 0, sd.n_total * sizeof(float), sd.stream);
        /* synapse traces (weights untouched) */
        if (sd.n_syn > 0) {
            size_t bF = (size_t)sd.n_syn * sizeof(float);
            cudaMemsetAsync(sd.S.pre_trace,         0, bF, sd.stream);
            cudaMemsetAsync(sd.S.post_trace,        0, bF, sd.stream);
            cudaMemsetAsync(sd.S.chrono_fast_trace, 0, bF, sd.stream);
            cudaMemsetAsync(sd.S.chrono_slow_trace, 0, bF, sd.stream);
            cudaMemsetAsync(sd.S.eligibility,       0, bF, sd.stream);
        }
        /* oscillator phases back to zero */
        cudaMemsetAsync(sd.O.phase, 0, CUNXON_BAND_COUNT * sizeof(float),
                        sd.stream);
    }
    for (auto& ld : net->links) {
        if (ld.delay_ring && ld.n_src_ports > 0) {
            size_t depth = (size_t)ld.p.delay_steps + 1;
            cudaMemset(ld.delay_ring, 0, depth * ld.n_src_ports * sizeof(int8_t));
        }
        if (ld.contrib && ld.n_dst_ports > 0)
            cudaMemset(ld.contrib, 0, ld.n_dst_ports * sizeof(float));
        ld.delay_head = 0;
    }
    cudaDeviceSynchronize();
    net->energy_total = 0.0;
    return CUNXON_OK;
}


/* =============================================================================
 *  Save / Load   (binary, version-tagged, host-side serialisation)
 *
 *  Format ("CUNXONV1" magic, little-endian, host-byte-order ints and floats):
 *
 *    [magic 8B "CUNXONV1"][n_spheres int32][n_links int32]
 *    for each sphere:
 *        [name_len int32][name char[name_len]]
 *        [kind int32]
 *        [params : cunxonNetworkParameters_t (binary)]
 *        [n_port_in_sensory int32][int32 ids ...]
 *        [n_port_in_relay   int32][int32 ids ...]
 *        [n_port_out_relay  int32][int32 ids ...]
 *        [n_port_out_readout int32][int32 ids ...]
 *        [n_syn int32]
 *        [pre[n_syn] post[n_syn] branch[n_syn]                  : int32]
 *        [w_fast[n_syn] w_slow[n_syn] w_meta[n_syn]             : float32]
 *        [is_silent[n_syn] is_modulatory[n_syn] syn_type[n_syn] : int8]
 *        [integrity[n_syn] chrono_omega[n_syn]                  : float32]
 *        [n_total int32][dsn_K int32]
 *        [dsn_kernel  : n_total * dsn_K float32]
 *        [ctsn_phi_gain[n_total] ctsn_phi_bias[n_total]         : float32]
 *    for each link:
 *        [src int32][dst int32]
 *        [link_params : cunxonLinkParameters_t (binary)]
 *        [n_src_ports int32][n_dst_ports int32]
 *        [W : n_src_ports * n_dst_ports float32]
 * ============================================================================= */

namespace {

template <typename T>
static inline void io_write(std::ostream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
static inline bool io_read(std::istream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return (bool)f;
}
template <typename T>
static inline void io_write_vec(std::ostream& f, const std::vector<T>& v) {
    if (!v.empty()) f.write(reinterpret_cast<const char*>(v.data()),
                            (std::streamsize)(v.size() * sizeof(T)));
}
template <typename T>
static inline bool io_read_vec(std::istream& f, std::vector<T>& v, size_t n) {
    v.resize(n);
    if (n == 0) return (bool)f;
    f.read(reinterpret_cast<char*>(v.data()),
           (std::streamsize)(n * sizeof(T)));
    return (bool)f;
}

template <typename T>
static inline std::vector<T> d2h(const T* d_ptr, size_t n) {
    std::vector<T> h(n);
    if (n) cudaMemcpy(h.data(), d_ptr, n*sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
template <typename T>
static inline void h2d(T* d_ptr, const std::vector<T>& h) {
    if (!h.empty())
        cudaMemcpy(d_ptr, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice);
}

}  /* anon */

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkSave(cunxonNetwork_t net, const char* path)
{
    if (!net || !path) return CUNXON_ERR_INVALID_ARGUMENT;
    if (!net->finalized) return CUNXON_ERR_NOT_FINALIZED;
    std::ofstream f(path, std::ios::binary);
    if (!f) return CUNXON_ERR_FILE_IO;

    const char magic[8] = {'C','U','N','X','O','N','V','1'};
    f.write(magic, 8);
    int ns = (int)net->spheres.size();
    int nl = (int)net->links.size();
    io_write(f, ns);
    io_write(f, nl);

    for (int s = 0; s < ns; ++s) {
        SphereDev& sd = net->spheres[s];
        /* sphere name */
        const std::string& nm = net->sphere_names_host[s];
        int nlen = (int)nm.size();
        io_write(f, nlen);
        if (nlen) f.write(nm.data(), nlen);

        /* kind + params */
        int kind = (int)sd.kind;
        io_write(f, kind);
        io_write(f, net->sphere_params_host[s]);

        /* port interface arrays */
        auto write_ports = [&](const int* d_ids, int n) {
            io_write(f, n);
            if (n > 0) {
                std::vector<int> h = d2h<int>(d_ids, (size_t)n);
                io_write_vec(f, h);
            }
        };
        write_ports(sd.port_in_sensory,  sd.n_port_in_sensory);
        write_ports(sd.port_in_relay,    sd.n_port_in_relay);
        write_ports(sd.port_out_relay,   sd.n_port_out_relay);
        write_ports(sd.port_out_readout, sd.n_port_out_readout);

        /* synapses */
        int n_syn = sd.n_syn;
        io_write(f, n_syn);
        auto pre   = d2h<int>(sd.S.pre_id,    (size_t)n_syn);
        auto post  = d2h<int>(sd.S.post_id,   (size_t)n_syn);
        auto br    = d2h<int>(sd.S.branch_id, (size_t)n_syn);
        auto wf    = d2h<float>(sd.S.w_fast,  (size_t)n_syn);
        auto ws    = d2h<float>(sd.S.w_slow,  (size_t)n_syn);
        auto wm    = d2h<float>(sd.S.w_meta,  (size_t)n_syn);
        auto sil   = d2h<int8_t>(sd.S.is_silent,    (size_t)n_syn);
        auto mod   = d2h<int8_t>(sd.S.is_modulatory,(size_t)n_syn);
        auto styp  = d2h<int8_t>(sd.S.synapse_type, (size_t)n_syn);
        auto integ = d2h<float>(sd.S.integrity,    (size_t)n_syn);
        auto om    = d2h<float>(sd.S.chrono_omega, (size_t)n_syn);
        io_write_vec(f, pre);   io_write_vec(f, post);  io_write_vec(f, br);
        io_write_vec(f, wf);    io_write_vec(f, ws);    io_write_vec(f, wm);
        io_write_vec(f, sil);   io_write_vec(f, mod);   io_write_vec(f, styp);
        io_write_vec(f, integ); io_write_vec(f, om);

        /* learned per-neuron parameters */
        int n_total = sd.n_total;
        int dsn_K   = sd.dsn_K;
        io_write(f, n_total);
        io_write(f, dsn_K);
        auto dsn_k  = d2h<float>(sd.N.dsn_kernel,    (size_t)n_total * (size_t)dsn_K);
        auto ctsn_g = d2h<float>(sd.N.ctsn_phi_gain, (size_t)n_total);
        auto ctsn_b = d2h<float>(sd.N.ctsn_phi_bias, (size_t)n_total);
        io_write_vec(f, dsn_k); io_write_vec(f, ctsn_g); io_write_vec(f, ctsn_b);
    }

    for (auto& ld : net->links) {
        io_write(f, ld.src);
        io_write(f, ld.dst);
        io_write(f, ld.p);
        io_write(f, ld.n_src_ports);
        io_write(f, ld.n_dst_ports);
        size_t W = (size_t)ld.n_src_ports * (size_t)ld.n_dst_ports;
        auto hW = d2h<float>(ld.W, W);
        io_write_vec(f, hW);
    }
    return f.good() ? CUNXON_OK : CUNXON_ERR_FILE_IO;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkLoad(cunxonNetwork_t net, const char* path)
{
    if (!net || !path) return CUNXON_ERR_INVALID_ARGUMENT;
    if (!net->spheres.empty() || !net->links.empty())
        return CUNXON_ERR_ALREADY_FINALIZED;   /* must load into an empty net */

    std::ifstream f(path, std::ios::binary);
    if (!f) return CUNXON_ERR_FILE_IO;

    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "CUNXONV1", 8) != 0)
        return CUNXON_ERR_INCOMPATIBLE_SAVE;

    int ns = 0, nl = 0;
    if (!io_read(f, ns) || !io_read(f, nl) || ns < 0 || nl < 0)
        return CUNXON_ERR_FILE_IO;

    /* === pass 1: rebuild each sphere using the public AddSphere path ===== */
    for (int s = 0; s < ns; ++s) {
        int nlen = 0;
        if (!io_read(f, nlen) || nlen < 0 || nlen > 4096)
            return CUNXON_ERR_FILE_IO;
        std::string sname(nlen, '\0');
        if (nlen) f.read(&sname[0], nlen);   /* &s[0] writable since C++11; .data() only since C++17 */

        int kind_i = 0;
        if (!io_read(f, kind_i)) return CUNXON_ERR_FILE_IO;
        cunxonNetworkParameters_t pp;
        if (!io_read(f, pp))     return CUNXON_ERR_FILE_IO;

        int sid = -1;
        cunxonStatus_t st = cunxonNetworkAddSphere(net, sname.c_str(),
                                                   (cunxonSphereKind_t)kind_i,
                                                   &pp, &sid);
        if (st != CUNXON_OK) return st;
        SphereDev& sd = net->spheres[sid];

        /* ports */
        auto read_ports = [&](int*& d_dst, int& n_dst) -> cunxonStatus_t {
            int n = 0;
            if (!io_read(f, n) || n < 0) return CUNXON_ERR_FILE_IO;
            std::vector<int> ids;
            if (!io_read_vec(f, ids, (size_t)n)) return CUNXON_ERR_FILE_IO;
            /* free any default-allocated ports (AddSphere left them at 0/null) */
            if (d_dst) { cudaFree(d_dst); d_dst = nullptr; }
            if (n > 0) {
                if (cudaMalloc(&d_dst, n*sizeof(int)) != cudaSuccess)
                    return CUNXON_ERR_OUT_OF_MEMORY;
                cudaMemcpy(d_dst, ids.data(), n*sizeof(int), cudaMemcpyHostToDevice);
            }
            n_dst = n;
            return CUNXON_OK;
        };
        st = read_ports(sd.port_in_sensory,   sd.n_port_in_sensory);   if (st) return st;
        st = read_ports(sd.port_in_relay,     sd.n_port_in_relay);     if (st) return st;
        st = read_ports(sd.port_out_relay,    sd.n_port_out_relay);    if (st) return st;
        st = read_ports(sd.port_out_readout,  sd.n_port_out_readout);  if (st) return st;

        /* synapse fan-in (n_syn from file -- discard the auto-built topology) */
        int n_syn_file = 0;
        if (!io_read(f, n_syn_file) || n_syn_file < 0) return CUNXON_ERR_FILE_IO;
        if (n_syn_file != sd.n_syn) {
            /* topology width changed: tear down auto-built synapse arrays and
             * realloc to match the saved fan-in.                            */
            auto free_if = [](void*& p){ if (p) { cudaFree(p); p = nullptr; } };
            void *p;
            #define FREE_SYN_FIELD(field)  p = sd.S.field; free_if(p); sd.S.field = (decltype(sd.S.field))nullptr
            FREE_SYN_FIELD(pre_id);  FREE_SYN_FIELD(post_id); FREE_SYN_FIELD(branch_id);
            FREE_SYN_FIELD(w_fast);  FREE_SYN_FIELD(w_slow);  FREE_SYN_FIELD(w_meta);
            FREE_SYN_FIELD(is_silent); FREE_SYN_FIELD(is_modulatory); FREE_SYN_FIELD(synapse_type);
            FREE_SYN_FIELD(integrity);
            FREE_SYN_FIELD(pre_trace); FREE_SYN_FIELD(post_trace);
            FREE_SYN_FIELD(chrono_fast_trace); FREE_SYN_FIELD(chrono_slow_trace);
            FREE_SYN_FIELD(chrono_omega); FREE_SYN_FIELD(eligibility);
            #undef FREE_SYN_FIELD
            sd.n_syn = n_syn_file;
            auto alloc = [&](void** dd, size_t bytes) {
                return cudaMalloc(dd, bytes) == cudaSuccess;
            };
            if (sd.n_syn > 0) {
                size_t bI = sd.n_syn * sizeof(int);
                size_t bF = sd.n_syn * sizeof(float);
                size_t b8 = sd.n_syn * sizeof(int8_t);
                bool ok =
                    alloc((void**)&sd.S.pre_id, bI)  && alloc((void**)&sd.S.post_id, bI) &&
                    alloc((void**)&sd.S.branch_id, bI) &&
                    alloc((void**)&sd.S.w_fast, bF)  && alloc((void**)&sd.S.w_slow, bF) &&
                    alloc((void**)&sd.S.w_meta, bF)  &&
                    alloc((void**)&sd.S.is_silent, b8) && alloc((void**)&sd.S.is_modulatory, b8) &&
                    alloc((void**)&sd.S.synapse_type, b8) &&
                    alloc((void**)&sd.S.integrity, bF) &&
                    alloc((void**)&sd.S.pre_trace, bF) && alloc((void**)&sd.S.post_trace, bF) &&
                    alloc((void**)&sd.S.chrono_fast_trace, bF) &&
                    alloc((void**)&sd.S.chrono_slow_trace, bF) &&
                    alloc((void**)&sd.S.chrono_omega, bF) &&
                    alloc((void**)&sd.S.eligibility, bF);
                if (!ok) return CUNXON_ERR_OUT_OF_MEMORY;
                /* zero the trace arrays */
                cudaMemset(sd.S.pre_trace, 0, bF);
                cudaMemset(sd.S.post_trace, 0, bF);
                cudaMemset(sd.S.chrono_fast_trace, 0, bF);
                cudaMemset(sd.S.chrono_slow_trace, 0, bF);
                cudaMemset(sd.S.eligibility, 0, bF);
            }
        }

        /* upload synapse fields */
        std::vector<int>    h_pre, h_post, h_br;
        std::vector<float>  h_wf, h_ws, h_wm, h_integ, h_om;
        std::vector<int8_t> h_sil, h_mod, h_styp;
        if (!io_read_vec(f, h_pre,  (size_t)sd.n_syn) ||
            !io_read_vec(f, h_post, (size_t)sd.n_syn) ||
            !io_read_vec(f, h_br,   (size_t)sd.n_syn) ||
            !io_read_vec(f, h_wf,   (size_t)sd.n_syn) ||
            !io_read_vec(f, h_ws,   (size_t)sd.n_syn) ||
            !io_read_vec(f, h_wm,   (size_t)sd.n_syn) ||
            !io_read_vec(f, h_sil,  (size_t)sd.n_syn) ||
            !io_read_vec(f, h_mod,  (size_t)sd.n_syn) ||
            !io_read_vec(f, h_styp, (size_t)sd.n_syn) ||
            !io_read_vec(f, h_integ,(size_t)sd.n_syn) ||
            !io_read_vec(f, h_om,   (size_t)sd.n_syn))
            return CUNXON_ERR_FILE_IO;

        if (sd.n_syn > 0) {
            h2d(sd.S.pre_id, h_pre);   h2d(sd.S.post_id, h_post); h2d(sd.S.branch_id, h_br);
            h2d(sd.S.w_fast, h_wf);    h2d(sd.S.w_slow, h_ws);    h2d(sd.S.w_meta, h_wm);
            h2d(sd.S.is_silent, h_sil);
            h2d(sd.S.is_modulatory, h_mod);
            h2d(sd.S.synapse_type, h_styp);
            h2d(sd.S.integrity, h_integ);
            h2d(sd.S.chrono_omega, h_om);
        }

        /* rebuild CSR post_offset on host (synapses already saved sorted by post) */
        {
            std::vector<int> off(sd.n_total + 1, 0);
            for (int j = 0; j < sd.n_syn; ++j) {
                if (h_post[j] < 0 || h_post[j] >= sd.n_total) return CUNXON_ERR_FILE_IO;
                off[h_post[j] + 1]++;
            }
            for (int i = 1; i <= sd.n_total; ++i) off[i] += off[i-1];
            if (sd.S.post_offset) cudaFree(sd.S.post_offset);
            cudaMalloc(&sd.S.post_offset, (sd.n_total + 1) * sizeof(int));
            cudaMemcpy(sd.S.post_offset, off.data(),
                       (sd.n_total + 1) * sizeof(int), cudaMemcpyHostToDevice);
        }

        /* learned per-neuron kernels */
        int n_total_file = 0, dsn_K_file = 0;
        if (!io_read(f, n_total_file) || !io_read(f, dsn_K_file)) return CUNXON_ERR_FILE_IO;
        if (n_total_file != sd.n_total || dsn_K_file != sd.dsn_K)
            return CUNXON_ERR_INCOMPATIBLE_SAVE;
        std::vector<float> h_dsn, h_cg, h_cb;
        if (!io_read_vec(f, h_dsn, (size_t)sd.n_total * (size_t)sd.dsn_K) ||
            !io_read_vec(f, h_cg,  (size_t)sd.n_total) ||
            !io_read_vec(f, h_cb,  (size_t)sd.n_total))
            return CUNXON_ERR_FILE_IO;
        h2d(sd.N.dsn_kernel, h_dsn);
        h2d(sd.N.ctsn_phi_gain, h_cg);
        h2d(sd.N.ctsn_phi_bias, h_cb);
    }

    /* === pass 2: rebuild links ============================================ */
    for (int l = 0; l < nl; ++l) {
        int src = 0, dst = 0;
        cunxonLinkParameters_t lp{};
        int n_src_ports = 0, n_dst_ports = 0;
        if (!io_read(f, src) || !io_read(f, dst) || !io_read(f, lp) ||
            !io_read(f, n_src_ports) || !io_read(f, n_dst_ports))
            return CUNXON_ERR_FILE_IO;
        if (src < 0 || src >= (int)net->spheres.size() ||
            dst < 0 || dst >= (int)net->spheres.size())
            return CUNXON_ERR_FILE_IO;

        int lid = -1;
        cunxonStatus_t st = cunxonNetworkAddLink(net, src, dst, &lp, &lid);
        if (st != CUNXON_OK) return st;
        LinkDev& ld = net->links[lid];
        if (ld.n_src_ports != n_src_ports || ld.n_dst_ports != n_dst_ports)
            return CUNXON_ERR_INCOMPATIBLE_SAVE;

        size_t W = (size_t)n_src_ports * (size_t)n_dst_ports;
        std::vector<float> hW;
        if (!io_read_vec(f, hW, W)) return CUNXON_ERR_FILE_IO;
        if (W) cudaMemcpy(ld.W, hW.data(), W*sizeof(float), cudaMemcpyHostToDevice);
    }

    /* === finalise ========================================================== */
    return cunxonNetworkFinalize(net);
}


/* =============================================================================
 *  Aigarth hybrid (evolutionary structural plasticity over weights)
 *
 *  One generation:
 *
 *     0. Snapshot baseline weights (synapse w_fast/w_slow/w_meta of every
 *        sphere + projection W of every link) to host.  Also reset dynamic
 *        state once so all clones evaluate from the same starting point.
 *     1. For c in 1..population:
 *          - restore baseline weights to device
 *          - apply Gaussian mutation scaled by (mut_fast, mut_slow, mut_meta)
 *            to every synaptic weight; link W is mutated with the *_meta rate
 *            (it's a slow projection-level signal).  Mutations are clipped.
 *          - reset dynamic state
 *          - score = fitness_fn(net, user_data)
 *          - if score > best_score: copy device weights to a separate "best"
 *            host buffer
 *     2. Upload best weights back to device.  Reset dynamic state once more.
 *
 *  The fitness function receives the *live* network with the candidate weights
 *  already in place and is free to call cunxonNetworkRun/Step* / readout etc.
 *  It must return a higher-is-better float score.
 * ============================================================================= */

/* cuRAND-based ternary/Gaussian mutation kernel (one weight per thread).      */
__global__ static void k_aigarth_mutate_weights(float*  w,
                                                int     n,
                                                float   mut_rate,
                                                float   clip,
                                                uint64_t seed,
                                                uint64_t seq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mut_rate <= 0.f) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, (uint64_t)i, seq, &st);
    float noise = curand_normal(&st) * mut_rate;
    float v = w[i] + noise;
    if (clip > 0.f) {
        if (v > clip)  v = clip;
        if (v < -clip) v = -clip;
    }
    w[i] = v;
}

namespace {

/* Snapshot device->host every mutable weight in the network.                  */
struct AigarthSnapshot {
    std::vector< std::vector<float> > sphere_wf, sphere_ws, sphere_wm;
    std::vector< std::vector<float> > link_W;
};

static void snapshot_weights(const cunxonNetworkImpl_* net, AigarthSnapshot& s)
{
    s.sphere_wf.resize(net->spheres.size());
    s.sphere_ws.resize(net->spheres.size());
    s.sphere_wm.resize(net->spheres.size());
    for (size_t i = 0; i < net->spheres.size(); ++i) {
        const SphereDev& sd = net->spheres[i];
        s.sphere_wf[i] = d2h<float>(sd.S.w_fast, (size_t)sd.n_syn);
        s.sphere_ws[i] = d2h<float>(sd.S.w_slow, (size_t)sd.n_syn);
        s.sphere_wm[i] = d2h<float>(sd.S.w_meta, (size_t)sd.n_syn);
    }
    s.link_W.resize(net->links.size());
    for (size_t i = 0; i < net->links.size(); ++i) {
        const LinkDev& ld = net->links[i];
        size_t W = (size_t)ld.n_src_ports * (size_t)ld.n_dst_ports;
        s.link_W[i] = d2h<float>(ld.W, W);
    }
}

static void restore_weights(cunxonNetworkImpl_* net, const AigarthSnapshot& s)
{
    for (size_t i = 0; i < net->spheres.size(); ++i) {
        SphereDev& sd = net->spheres[i];
        if (sd.n_syn) {
            h2d(sd.S.w_fast, s.sphere_wf[i]);
            h2d(sd.S.w_slow, s.sphere_ws[i]);
            h2d(sd.S.w_meta, s.sphere_wm[i]);
        }
    }
    for (size_t i = 0; i < net->links.size(); ++i) {
        LinkDev& ld = net->links[i];
        if (!s.link_W[i].empty())
            h2d(ld.W, s.link_W[i]);
    }
}

static void reset_dynamic_state(cunxonNetworkImpl_* net)
{
    for (auto& sd : net->spheres) {
        int tb = 256, nb = (sd.n_total + tb - 1) / tb;
        if (nb > 0)
            k_reset_neuron_dynamic<<<nb, tb, 0, sd.stream>>>(sd.N, sd.n_total);
        cudaMemsetAsync(sd.N.branch_pot, 0,
                        sd.n_total * sd.n_branches * sizeof(float), sd.stream);
        cudaMemsetAsync(sd.N.branch_sum, 0,
                        sd.n_total * sd.n_branches * sizeof(float), sd.stream);
        cudaMemsetAsync(sd.N.modulatory_pot, 0,
                        sd.n_total * sizeof(float), sd.stream);
        /* synapse traces: zero them so plasticity has no carry-over          */
        if (sd.n_syn > 0) {
            size_t bF = (size_t)sd.n_syn * sizeof(float);
            cudaMemsetAsync(sd.S.pre_trace,        0, bF, sd.stream);
            cudaMemsetAsync(sd.S.post_trace,       0, bF, sd.stream);
            cudaMemsetAsync(sd.S.chrono_fast_trace,0, bF, sd.stream);
            cudaMemsetAsync(sd.S.chrono_slow_trace,0, bF, sd.stream);
            cudaMemsetAsync(sd.S.eligibility,      0, bF, sd.stream);
        }
        /* link delay rings and contribution buffers                          */
    }
    /* clear inter-sphere delay rings + contributions too                     */
    for (auto& ld : net->links) {
        if (ld.delay_ring && ld.n_src_ports > 0) {
            size_t depth = (size_t)ld.p.delay_steps + 1;
            cudaMemset(ld.delay_ring, 0, depth * ld.n_src_ports * sizeof(int8_t));
        }
        if (ld.contrib && ld.n_dst_ports > 0)
            cudaMemset(ld.contrib, 0, ld.n_dst_ports * sizeof(float));
        ld.delay_head = 0;
    }
    for (auto& sd : net->spheres) cudaStreamSynchronize(sd.stream);
}

static void apply_mutation(cunxonNetworkImpl_* net,
                           float mf, float ms, float mm,
                           float wclip,
                           uint64_t seed,
                           uint64_t generation)
{
    for (size_t i = 0; i < net->spheres.size(); ++i) {
        SphereDev& sd = net->spheres[i];
        if (sd.n_syn <= 0) continue;
        int tb = 256, nb = (sd.n_syn + tb - 1) / tb;
        uint64_t seq = generation * 7 + i * 13;
        k_aigarth_mutate_weights<<<nb, tb, 0, sd.stream>>>(
            sd.S.w_fast, sd.n_syn, mf, wclip, seed ^ 0xA1A1A1A1u, seq + 1);
        k_aigarth_mutate_weights<<<nb, tb, 0, sd.stream>>>(
            sd.S.w_slow, sd.n_syn, ms, wclip, seed ^ 0xB2B2B2B2u, seq + 2);
        k_aigarth_mutate_weights<<<nb, tb, 0, sd.stream>>>(
            sd.S.w_meta, sd.n_syn, mm, wclip, seed ^ 0xC3C3C3C3u, seq + 3);
    }
    for (size_t i = 0; i < net->links.size(); ++i) {
        LinkDev& ld = net->links[i];
        size_t W = (size_t)ld.n_src_ports * (size_t)ld.n_dst_ports;
        if (W == 0) continue;
        int tb = 256, nb = (int)((W + tb - 1) / tb);
        uint64_t seq = generation * 11 + i * 19 + 0xD4D4u;
        float lclip = (ld.p.weight_clip > 0.f) ? ld.p.weight_clip : 1.0f;
        k_aigarth_mutate_weights<<<nb, tb>>>(
            ld.W, (int)W, mm, lclip, seed ^ 0xE5E5E5E5u, seq);
    }
    cudaDeviceSynchronize();
}

}  /* anon */

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkAigarthConfig(cunxonNetwork_t net, int pop,
                                          float mf, float ms, float mm)
{
    if (!net || pop <= 0) return CUNXON_ERR_INVALID_ARGUMENT;
    if (mf < 0.f || ms < 0.f || mm < 0.f) return CUNXON_ERR_INVALID_ARGUMENT;
    net->aigarth_population = pop;
    net->aigarth_mut_fast = mf;
    net->aigarth_mut_slow = ms;
    net->aigarth_mut_meta = mm;
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkAigarthStep(cunxonNetwork_t net,
                                        cunxonFitnessFn_t fn,
                                        void* user)
{
    if (!net || !fn) return CUNXON_ERR_INVALID_ARGUMENT;
    if (!net->finalized) return CUNXON_ERR_NOT_FINALIZED;
    if (net->aigarth_population <= 0)
        return CUNXON_ERR_INVALID_ARGUMENT;   /* call AigarthConfig first */

    /* 0. baseline snapshot ------------------------------------------------- */
    AigarthSnapshot baseline, best;
    snapshot_weights(net, baseline);
    best = baseline;                       /* deep copy via std::vector       */
    reset_dynamic_state(net);

    /* w_clip: use the first sphere's w_clip-equivalent (no global field);
     * fall back to a sane bound that matches the typical |w| <= 1 range.   */
    float wclip = 1.0f;
    if (!net->sphere_params_host.empty()) {
        const auto& p0 = net->sphere_params_host.front();
        wclip = std::max({p0.w_fast_init_max, p0.w_slow_init_max,
                          p0.w_meta_init_max, 1e-3f});
    }

    /* 1. evaluate baseline (generation 0) --------------------------------- */
    float best_score = fn(net, user);

    /* 2. mutate / evaluate / select -------------------------------------- */
    uint64_t seed = net->ctx ? net->ctx->seed ^ 0xA1F00D1ULL : 0xA1F00D1ULL;
    for (int g = 1; g <= net->aigarth_population; ++g) {
        restore_weights(net, baseline);
        apply_mutation(net, net->aigarth_mut_fast, net->aigarth_mut_slow,
                       net->aigarth_mut_meta, wclip, seed, (uint64_t)g);
        reset_dynamic_state(net);

        float score = fn(net, user);
        if (score > best_score) {
            best_score = score;
            snapshot_weights(net, best);
        }
    }

    /* 3. install winner --------------------------------------------------- */
    restore_weights(net, best);
    reset_dynamic_state(net);
    return CUNXON_OK;
}


/* =============================================================================
 *  Sphere Layers  (metadata-only hierarchical grouping; paper §P4)
 * ============================================================================= */
extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkAddLayer(cunxonNetwork_t net,
                                     const char* layer_name,
                                     int depth, int* out_layer_id)
{
    if (!net || !out_layer_id) return CUNXON_ERR_INVALID_ARGUMENT;
    SphereLayerHost L;
    L.name  = layer_name ? layer_name : "";
    L.depth = depth;
    *out_layer_id = (int)net->layers.size();
    net->layers.push_back(std::move(L));
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkAddSphereToLayer(cunxonNetwork_t net,
                                             int layer_id, int sphere_id)
{
    if (!net) return CUNXON_ERR_INVALID_ARGUMENT;
    if (layer_id  < 0 || layer_id  >= (int)net->layers.size())
        return CUNXON_ERR_INVALID_ARGUMENT;
    if (sphere_id < 0 || sphere_id >= (int)net->spheres.size())
        return CUNXON_ERR_INVALID_SPHERE;
    auto& ids = net->layers[layer_id].sphere_ids;
    for (int x : ids) if (x == sphere_id) return CUNXON_OK;  /* idempotent */
    ids.push_back(sphere_id);
    return CUNXON_OK;
}

extern "C" CUNXON_API
int cunxonNetworkNumLayers(cunxonNetwork_t net)
{
    return net ? (int)net->layers.size() : -1;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkGetLayer(cunxonNetwork_t net, int layer_id,
                                     char* name_out, int name_buf_len,
                                     int* depth_out,
                                     int* sphere_ids, int* n_spheres)
{
    if (!net || !n_spheres) return CUNXON_ERR_INVALID_ARGUMENT;
    if (layer_id < 0 || layer_id >= (int)net->layers.size())
        return CUNXON_ERR_INVALID_ARGUMENT;
    const SphereLayerHost& L = net->layers[layer_id];
    if (name_out && name_buf_len > 0) {
        size_t n = std::min((size_t)(name_buf_len - 1), L.name.size());
        std::memcpy(name_out, L.name.data(), n);
        name_out[n] = '\0';
    }
    if (depth_out) *depth_out = L.depth;
    if (!sphere_ids) {
        *n_spheres = (int)L.sphere_ids.size();
        return CUNXON_OK;
    }
    int cap = *n_spheres;
    int k = std::min((int)L.sphere_ids.size(), cap);
    for (int i = 0; i < k; ++i) sphere_ids[i] = L.sphere_ids[i];
    *n_spheres = (int)L.sphere_ids.size();
    return CUNXON_OK;
}


/* =============================================================================
 *  Pattern application layer (Algorithm 8)
 * ============================================================================= */
namespace {

/* Drive a single sphere's sensory inputs for n_steps, optionally training. */
static cunxonStatus_t drive_pattern(cunxonNetworkImpl_* net,
                                    int sphere_id,
                                    const float* pattern, int pattern_len,
                                    int n_steps, float dt_ms, int training)
{
    if (sphere_id < 0 || sphere_id >= (int)net->spheres.size())
        return CUNXON_ERR_INVALID_SPHERE;
    const SphereDev& sd = net->spheres[sphere_id];
    if (sd.n_port_in_sensory != pattern_len) return CUNXON_ERR_INVALID_ARGUMENT;

    /* Build the per-sphere ext_inputs array: pattern for the chosen sphere,
     * nullptr for all others.                                                */
    std::vector<const float*> per_sphere(net->spheres.size(), nullptr);
    per_sphere[sphere_id] = pattern;

    cunxonStatus_t st = CUNXON_OK;
    for (int t = 0; t < n_steps && st == CUNXON_OK; ++t) {
        if (training)
            st = cunxonNetworkStepTrain(net, per_sphere.data(), dt_ms);
        else
            st = cunxonNetworkStepInfer(net, per_sphere.data(), dt_ms);
    }
    return st;
}

}  /* anon */

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkStorePattern(cunxonNetwork_t net,
                                         int sphere_id,
                                         const char* pattern_name,
                                         const float* pattern,
                                         int pattern_len,
                                         int n_present_steps,
                                         float dt_ms)
{
    if (!net || !pattern_name || !pattern || pattern_len <= 0
        || n_present_steps < 0)
        return CUNXON_ERR_INVALID_ARGUMENT;
    if (!net->finalized) return CUNXON_ERR_NOT_FINALIZED;

    cunxonNetworkImpl_::PatternEntry pe;
    pe.sphere_id = sphere_id;
    pe.values.assign(pattern, pattern + pattern_len);
    net->patterns[std::string(pattern_name)] = std::move(pe);

    /* Consolidate via training presentation.                                 */
    if (n_present_steps > 0)
        return drive_pattern(net, sphere_id, pattern, pattern_len,
                             n_present_steps, dt_ms, /*training=*/1);
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkRecallPattern(cunxonNetwork_t net,
                                          int sphere_id,
                                          const char* pattern_name,
                                          int pattern_len,
                                          float mask_fraction,
                                          int n_settle_steps,
                                          float dt_ms,
                                          int8_t* readout_out,
                                          int* n_readout)
{
    if (!net || !pattern_name || !n_readout) return CUNXON_ERR_INVALID_ARGUMENT;
    if (!net->finalized) return CUNXON_ERR_NOT_FINALIZED;

    auto it = net->patterns.find(pattern_name);
    if (it == net->patterns.end()) return CUNXON_ERR_INVALID_ARGUMENT;
    const auto& pe = it->second;
    if ((int)pe.values.size() != pattern_len) return CUNXON_ERR_INVALID_ARGUMENT;
    if (pe.sphere_id != sphere_id)            return CUNXON_ERR_INVALID_SPHERE;

    /* Build masked cue (zero out a random `mask_fraction` of elements).      */
    std::vector<float> cue(pe.values);
    if (mask_fraction > 0.f) {
        std::mt19937_64 rng((uint64_t)net->patterns.size() * 0xC0FFEE + 0xDEAD);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        for (auto& v : cue) if (u01(rng) < mask_fraction) v = 0.f;
    }

    /* Inference-only settle.                                                 */
    cunxonStatus_t st = drive_pattern(net, sphere_id, cue.data(), pattern_len,
                                      n_settle_steps, dt_ms, /*training=*/0);
    if (st != CUNXON_OK) return st;
    return cunxonSphereGetReadout(net, sphere_id, readout_out, n_readout);
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkTrainSequence(cunxonNetwork_t net,
                                          int sphere_id,
                                          const float* const* patterns,
                                          int n_patterns,
                                          int pattern_len,
                                          int n_repetitions,
                                          int n_steps_per_pattern,
                                          float dt_ms)
{
    if (!net || !patterns || n_patterns <= 0 || pattern_len <= 0
        || n_repetitions <= 0 || n_steps_per_pattern <= 0)
        return CUNXON_ERR_INVALID_ARGUMENT;
    if (!net->finalized) return CUNXON_ERR_NOT_FINALIZED;
    for (int r = 0; r < n_repetitions; ++r) {
        for (int p = 0; p < n_patterns; ++p) {
            cunxonStatus_t st = drive_pattern(net, sphere_id, patterns[p],
                                              pattern_len,
                                              n_steps_per_pattern, dt_ms,
                                              /*training=*/1);
            if (st != CUNXON_OK) return st;
        }
    }
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkListPatterns(cunxonNetwork_t net,
                                         char* names_out, int names_buf_len,
                                         int* n_patterns)
{
    if (!net || !n_patterns) return CUNXON_ERR_INVALID_ARGUMENT;
    *n_patterns = (int)net->patterns.size();
    if (!names_out) return CUNXON_OK;
    /* Newline-separated, NUL-terminated. */
    int written = 0;
    for (auto& kv : net->patterns) {
        int need = (int)kv.first.size() + 1;       /* +1 for '\n' */
        if (written + need + 1 > names_buf_len) break;
        std::memcpy(names_out + written, kv.first.data(), kv.first.size());
        written += (int)kv.first.size();
        names_out[written++] = '\n';
    }
    names_out[written < names_buf_len ? written : names_buf_len - 1] = '\0';
    return CUNXON_OK;
}

extern "C" CUNXON_API
cunxonStatus_t cunxonNetworkClearPatterns(cunxonNetwork_t net)
{
    if (!net) return CUNXON_ERR_INVALID_ARGUMENT;
    net->patterns.clear();
    return CUNXON_OK;
}
