/* =============================================================================
 *  cuNxon_intersphere.cu
 *
 *  Inter-sphere communication kernels implementing Multi-Neuraxon's P3:
 *  Frequency-gated transmission via communication-through-coherence (CTC).
 *
 *      g_CTC(t) = (1 - c) + c * 0.5 * (1 + cos(phi_src_b - phi_dst_b))
 *
 *      I_inter_j(t) = sum_p  g_p(t) * w_p * s_out_p.src(t)
 *
 *  Pipeline per link:
 *      1. k_ctc_gate              -- scalar gate value from src/dst phases
 *      2. k_intersphere_project   -- matmul of W * s_src_ports * g_ctc + bias
 *                                    (with optional trinary quantisation)
 *      3. k_intersphere_inject    -- atomic-add into dst's external-input slot
 *      4. k_proj_plasticity       -- Hebbian projection plasticity:
 *                                    dW_ij = eta * s_src_i * s_dst_j
 *
 *  Delays are handled host-side by writing into a delay ring buffer of past
 *  src-port states and reading the head - delay_steps entry.
 * ============================================================================= */

#include "cuNxon_internal.cuh"


/* ----------------------------------------------------------------------------
 *  k_ctc_gate
 *
 *      g = (1 - c) + c * 0.5 * (1 + cos(phi_src - phi_dst))
 * ---------------------------------------------------------------------------- */
__global__ void k_ctc_gate(const OscillatorBankDev Osrc,
                           const OscillatorBankDev Odst,
                           int band,
                           float coherence,
                           float* g_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float ps = Osrc.phase[band];
    float pd = Odst.phase[band];
    float dphi = ps - pd;
    *g_out = (1.f - coherence) + coherence * 0.5f * (1.f + cosf(dphi));
}


/* ----------------------------------------------------------------------------
 *  k_intersphere_project
 *
 *  Inputs:
 *    Nsrc           - source sphere's neuron SoA (read s[] only)
 *    src_port_ids   - which neuron indices in src are output-ports
 *    W              - row-major [n_dst_ports x n_src_ports]
 *    g_ctc          - scalar gate (read by all threads)
 *    gain, bias     - link scalar params
 *
 *  Output:
 *    contrib_out[d] = g_ctc * gain * (sum_p W[d, p] * s_src[port_p]) + bias
 *
 *  One block per dst port; threads cooperatively reduce along src ports.
 * ---------------------------------------------------------------------------- */
__global__ void k_intersphere_project(const NeuronArraysDev Nsrc,
                                      const int* src_port_ids,
                                      int n_src_ports,
                                      const float* W,
                                      int n_dst_ports,
                                      const float* g_ctc,
                                      float gain,
                                      float bias,
                                      float transmission_threshold,
                                      float* contrib_out)
{
    int d = blockIdx.x;
    if (d >= n_dst_ports) return;

    __shared__ float sh_acc[256];
    int tid = threadIdx.x;
    sh_acc[tid] = 0.f;

    const float* Wrow = W + (size_t)d * n_src_ports;

    for (int p = tid; p < n_src_ports; p += blockDim.x) {
        int neuron_id = src_port_ids[p];
        int8_t s = Nsrc.s[neuron_id];
        sh_acc[tid] += Wrow[p] * (float)s;
    }
    __syncthreads();

    /* tree reduction */
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) sh_acc[tid] += sh_acc[tid + off];
        __syncthreads();
    }

    if (tid == 0) {
        float g = *g_ctc;
        float v = g * gain * sh_acc[0] + bias;
        /* Paper Eq.2: trinary quantisation at the input boundary —
         * sub-threshold transmissions are suppressed.                       */
        if (transmission_threshold > 0.f && fabsf(v) < transmission_threshold)
            v = 0.f;
        contrib_out[d] = v;
    }
}


/* ----------------------------------------------------------------------------
 *  k_intersphere_inject
 *
 *  Atomically add this link's per-dst-port contribution into the dst sphere's
 *  per-neuron external-input buffer.  Multiple incoming links sum here.
 *
 *  We additionally apply trinary quantisation later in the membrane kernel via
 *  N.s[] for input-port neurons; here we just sum continuous contributions.
 * ---------------------------------------------------------------------------- */
__global__ void k_intersphere_inject(float* dst_ext_in,
                                     const int* dst_port_ids,
                                     int n_dst_ports,
                                     const float* contrib)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_dst_ports) return;

    int target = dst_port_ids[d];
    /* atomicAdd in case multiple links feed the same port. */
    atomicAdd(&dst_ext_in[target], contrib[d]);
}


/* ----------------------------------------------------------------------------
 *  k_proj_plasticity
 *
 *  Hebbian update for the inter-sphere projection matrix W:
 *      dW[d, p] = eta * s_src[port_p] * s_dst[port_d]
 *      W       <- (1 - decay) * (W + dW), clipped to [-clip, +clip]
 *
 *  One thread per (d, p) entry.
 * ---------------------------------------------------------------------------- */
__global__ void k_proj_plasticity(float* W,
                                  const NeuronArraysDev Nsrc,
                                  const int* src_port_ids,
                                  int n_src_ports,
                                  const NeuronArraysDev Ndst,
                                  const int* dst_port_ids,
                                  int n_dst_ports,
                                  float eta,
                                  float decay,
                                  float clip)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= n_src_ports || d >= n_dst_ports) return;

    int sid = src_port_ids[p];
    int tid = dst_port_ids[d];

    float ss = (float)Nsrc.s[sid];
    float ds_ = (float)Ndst.s[tid];

    size_t off = (size_t)d * n_src_ports + p;
    float w = W[off];
    w = (1.f - decay) * (w + eta * ss * ds_);
    if (w >  clip) w =  clip;
    if (w < -clip) w = -clip;
    W[off] = w;
}
