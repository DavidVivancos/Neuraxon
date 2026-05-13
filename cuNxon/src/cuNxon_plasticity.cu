/* =============================================================================
 *  cuNxon_plasticity.cu
 *
 *  Synaptic plasticity kernels.  All kernels operate one thread per synapse
 *  and read post/pre neuron arrays read-only.
 *
 *  Mirrors Algorithm 1 lines 62-72 and the structural-plasticity block.
 *
 *      k_plasticity_stdp     -- STDP traces + DA-gated (D1 LTP / D2 LTD)
 *                                + Hebbian boost + associative neighbour
 *                                  plasticity (within-branch sharing).
 *      k_plasticity_agmp     -- Eligibility-trace * DA_phasic * astrocyte
 *                                + integration into w_fast/w_slow/w_meta with
 *                                  multi-timescale leak (tau_fast/slow/meta).
 *      k_structural_prune    -- Mark integrity-low / weight-low synapses
 *                                as silent (soft removal).
 * ============================================================================= */

#include "cuNxon_internal.cuh"


__device__ __forceinline__ float dev_clamp_f(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}


/* ============================================================================
 *  k_plasticity_stdp
 *
 *  For each synapse j -> i:
 *      A_plus  += dt * (-A_plus  / tau_stdp + I[s_j == +1])
 *      A_minus += dt * (-A_minus / tau_stdp + I[s_i == +1])
 *
 *      dw_stdp = eta * (A_plus  * I[s_i == +1] * [D1]
 *                     - A_minus * I[s_j == +1] * [D2])
 *
 *  Hebbian boost / penalty on simultaneous trinary coincidences:
 *      (+1, +1) -> +0.5 * eta * [D1]
 *      (+1, -1) -> -0.5 * eta * [D2]
 *      ( 0,  0) -> 0.1x scaling
 *
 *  Associative neighbour-plasticity contribution (within same branch on same
 *  post neuron):  this is approximated by looking up the running mean
 *  dw of the post neuron's synapses, stored in S.eligibility briefly during
 *  the AGMP kernel.  (For simplicity, we omit per-synapse distance weighting
 *  here and use a uniform neighbour share.)
 *
 *  Final integration into w_fast and w_slow happens here as well (mirroring
 *  the Python reference); see Alg 1 lines 69-71.  Modulatory (w_meta) is
 *  updated only when the synapse is flagged modulatory.
 * ============================================================================ */
__global__ void k_plasticity_stdp(SynapseArraysDev S, const NeuronArraysDev N,
                                  int n_syn,
                                  const NeuromodFieldDev M,
                                  const cunxonNetworkParameters_t* p,
                                  float dt_ms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_syn) return;
    if (S.is_silent[idx]) return;

    int pre  = S.pre_id [idx];
    int post = S.post_id[idx];

    int8_t sj = N.s[pre];
    int8_t si = N.s[post];

    float A_plus  = S.pre_trace [idx];
    float A_minus = S.post_trace[idx];

    /* Signed STDP traces: capture both +1 and -1 firing.  The old rule
     * `+ (sj == 1 ? 1 : 0)` made -1 invisible to plasticity, biasing
     * the network to only learn "+1 → +1" patterns and never "-1 → -1".
     * Symmetric trinary requires signed traces.                          */
    float tau = p->tau_stdp;
    A_plus  += dt_ms * (-A_plus  / tau + (float)sj);
    A_minus += dt_ms * (-A_minus / tau + (float)si);

    /* Enforce the biological STDP window: traces saturate at the window-mass
     * implied by exponential decay (window / tau).  Symmetric clip since
     * traces are now signed.                                              */
    float trace_cap = fmaxf(1.f, p->stdp_window / fmaxf(tau, 1e-3f));
    A_plus  = dev_clamp_f(A_plus,  -trace_cap, trace_cap);
    A_minus = dev_clamp_f(A_minus, -trace_cap, trace_cap);

    S.pre_trace [idx] = A_plus;
    S.post_trace[idx] = A_minus;

    float D1 = M.recept[0];
    float D2 = M.recept[1];

    float eta = p->learning_rate;
    /* Symmetric trinary STDP: dw is proportional to signed-trace × current-
     * signed-firing.  Same-sign correlations potentiate; opposite-sign
     * correlations depress; either-side-rest produces no contribution.    */
    float dw  = eta * A_plus  * (float)si * D1
              - eta * A_minus * (float)sj * D2;

    /* Trinary coincidence boosts (now symmetric for ±1 firing) */
    if (sj ==  1 && si ==  1)  dw += 0.5f * eta * D1;    /* same-sign LTP */
    if (sj == -1 && si == -1)  dw += 0.5f * eta * D1;    /* same-sign LTP (was missing) */
    if (sj ==  1 && si == -1)  dw -= 0.5f * eta * D2;    /* opposite-sign LTD */
    if (sj == -1 && si ==  1)  dw -= 0.5f * eta * D2;    /* opposite-sign LTD (was missing) */
    if (sj ==  0 && si ==  0)  dw *= 0.1f;               /* dampen quiet-quiet */

    /* 5HT2A scales modulatory weight updates ------------------------------- */
    float fiveHT2A = M.recept[3];

    /* Apply to fast/slow/meta with multi-timescale leak (Alg 1 lines 69-71). */
    float wf = S.w_fast[idx];
    float ws = S.w_slow[idx];
    float wm = S.w_meta[idx];

    wf += dt_ms / p->tau_fast * (-wf + 0.3f * dw);
    ws += dt_ms / p->tau_slow * (-ws + 0.1f * dw);
    if (S.is_modulatory[idx])
        wm += dt_ms / p->tau_meta * (-wm + 0.05f * dw * (0.5f + fiveHT2A));

    wf = dev_clamp_f(wf, -1.f, 1.f);
    ws = dev_clamp_f(ws, -1.f, 1.f);
    wm = dev_clamp_f(wm, -0.5f, 0.5f);

    S.w_fast[idx] = wf;
    S.w_slow[idx] = ws;
    S.w_meta[idx] = wm;

    /* Maintain integrity (decays toward 1 if synapse is "useful") --------- */
    float integ = S.integrity[idx];
    float u_mag = sqrtf(wf * wf + ws * ws);
    integ = dev_clamp_f(integ + (u_mag > 0.05f ? 0.001f : -0.001f), 0.f, 1.f);
    S.integrity[idx] = integ;

    /* Stash dw briefly in eligibility for AGMP / associative passes ------- */
    S.eligibility[idx] = dw;
}


/* ============================================================================
 *  k_plasticity_agmp
 *
 *  Eligibility-trace AGMP (Alg 1 lines 67-71):
 *      e_t      <- lambda_e * e_t + sign(s_j) * sign(s_i)
 *      dw_agmp  <- eta * DA_phasic * a_post * e_t
 *      w_fast  <- clip(w_fast + dt/tau_fast * (-w_fast + 0.3 * dw_agmp), +-1)
 *      w_slow  <- clip(w_slow + dt/tau_slow * (-w_slow + 0.1 * dw_agmp), +-1)
 *
 *  Note this kernel uses the STDP eligibility hash left over by the previous
 *  kernel: w_fast/w_slow are already updated for STDP; AGMP adds on top.
 * ============================================================================ */
__global__ void k_plasticity_agmp(SynapseArraysDev S, const NeuronArraysDev N,
                                  int n_syn,
                                  const NeuromodFieldDev M,
                                  const cunxonNetworkParameters_t* p,
                                  float dt_ms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_syn) return;
    if (S.is_silent[idx]) return;
    if (!p->agmp_enabled) return;

    int pre  = S.pre_id [idx];
    int post = S.post_id[idx];

    int8_t sj = N.s[pre];
    int8_t si = N.s[post];

    float hebb_sgn = (float)sj * (float)si;        /* in {-1, 0, +1}        */

    float e = S.eligibility[idx];                  /* now holds dw_stdp     */
    float e_new = p->agmp_lambda_e * e + hebb_sgn;
    S.eligibility[idx] = e_new;

    float DA_phasic = M.phasic[0];
    float a_post    = N.astrocyte[post];

    float dw_agmp = p->agmp_eta * DA_phasic * a_post * e_new;

    float wf = S.w_fast[idx];
    float ws = S.w_slow[idx];

    wf += dt_ms / p->tau_fast * (-wf + 0.3f * dw_agmp);
    ws += dt_ms / p->tau_slow * (-ws + 0.1f * dw_agmp);

    wf = dev_clamp_f(wf, -1.f, 1.f);
    ws = dev_clamp_f(ws, -1.f, 1.f);

    S.w_fast[idx] = wf;
    S.w_slow[idx] = ws;
}


/* ============================================================================
 *  k_plasticity_associative
 *
 *  Paper Eq.:   Δw_i += α · Σ_{j ∈ N(i)} (Δw_j − Δw_i) / d_ij
 *
 *  We take the natural CUDA-friendly definition of "neighbour":  two synapses
 *  are neighbours iff they share the same post-neuron (i.e., they are in the
 *  same CSR slice via post_offset).  Distance is the absolute branch-index
 *  difference + 1, so synapses on the same dendritic branch are closer than
 *  ones on different branches.
 *
 *  Reads:   S.eligibility[idx] holds the latest Δw_stdp (stashed by k_plasticity_stdp).
 *  Writes:  S.w_fast / S.w_slow with the diffusive update, clipped.
 *
 *  One thread per synapse.  Loop over the (typically small, ≤ fan_in) neighbour
 *  set held in the same post bucket.
 * ============================================================================ */
__global__ void k_plasticity_associative(SynapseArraysDev S,
                                         int n_total, int n_syn,
                                         const cunxonNetworkParameters_t* p,
                                         float dt_ms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_syn) return;
    if (S.is_silent[idx]) return;
    float alpha = p->associative_alpha;
    if (alpha <= 0.f) return;

    int post = S.post_id[idx];
    int b_i  = S.branch_id[idx];
    if (post < 0 || post >= n_total) return;

    int beg = S.post_offset[post];
    int end = S.post_offset[post + 1];
    /* if there's only one synapse on this post, no neighbours to diffuse with */
    if (end - beg <= 1) return;

    float my_dw = S.eligibility[idx];
    float acc   = 0.f;
    int   cnt   = 0;
    for (int j = beg; j < end; ++j) {
        if (j == idx) continue;
        if (S.is_silent[j]) continue;
        int b_j  = S.branch_id[j];
        float d  = (float)(abs(b_i - b_j) + 1);
        acc += (S.eligibility[j] - my_dw) / d;
        cnt++;
    }
    if (cnt == 0) return;
    float diffuse = alpha * acc / (float)cnt;

    float wf = S.w_fast[idx] + 0.7f * diffuse * dt_ms;
    float ws = S.w_slow[idx] + 0.3f * diffuse * dt_ms;
    S.w_fast[idx] = dev_clamp_f(wf, -1.f, 1.f);
    S.w_slow[idx] = dev_clamp_f(ws, -1.f, 1.f);
}


/* ============================================================================
 *  k_structural_prune
 *
 *  Three-way structural plasticity per call:
 *    1. Stochastic death  — active synapses with weak weights get a small
 *       chance per step (synapse_death_prob * dt) of going silent.
 *    2. Integrity prune   — active synapses with |w|^2 below threshold are
 *       immediately silenced (deterministic).
 *    3. Resurrection/form — silent synapses have a small chance per step
 *       (synapse_formation_prob * dt) of waking with a random small weight.
 *
 *  Each thread handles one synapse and uses a Philox stream keyed on
 *  (idx, step_seed) so behaviour is reproducible.
 * ============================================================================ */
__global__ void k_structural_prune(SynapseArraysDev S, int n_syn,
                                   const cunxonNetworkParameters_t* p,
                                   float dt_ms,
                                   uint64_t step_seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_syn) return;

    curandStatePhilox4_32_10_t rng;
    curand_init(step_seed, (uint64_t)idx, 0, &rng);

    float death = p->synapse_death_prob    * dt_ms;
    float form  = p->synapse_formation_prob * dt_ms;
    if (death > 1.f) death = 1.f;
    if (form  > 1.f) form  = 1.f;

    if (S.is_silent[idx]) {
        /* resurrection: silent → active with small random weight             */
        if (form > 0.f && curand_uniform(&rng) < form) {
            float w = (curand_uniform(&rng) - 0.5f) * 0.2f;
            S.is_silent[idx]  = 0;
            S.w_fast[idx]     = w;
            S.w_slow[idx]     = 0.5f * w;
            S.integrity[idx]  = 1.f;
        }
        return;
    }

    /* active path */
    float wf = S.w_fast[idx];
    float ws = S.w_slow[idx];
    float mag = wf * wf + ws * ws;
    float eps = p->synapse_integrity_threshold;

    /* Deterministic integrity prune --- gated on structural plasticity
     * being enabled (death_prob > 0).  This way users wanting fixed-
     * topology networks can fully disable structural changes by setting
     * synapse_death_prob = 0 and synapse_formation_prob = 0.               */
    if (death > 0.f && mag < eps * eps) {
        S.is_silent[idx] = 1;
        S.integrity[idx] = 0.f;
        return;
    }
    /* Stochastic death (weighted by weakness: weaker syns more likely)       */
    if (death > 0.f) {
        float weak = 1.f - fminf(1.f, mag / (4.f * eps * eps));
        if (curand_uniform(&rng) < death * weak) {
            S.is_silent[idx] = 1;
            S.integrity[idx] = 0.f;
        }
    }
}
