/* =============================================================================
 *  cuNxon_kernels.cu  -  Algorithm 1 forward-pipeline CUDA kernels
 *
 *  Implements (per simulation step):
 *      Step 1: ChronoPlastic synaptic time warping  (k_chrono_warp_and_isyn)
 *      Step 2: dendritic-branch gather              (k_dendritic_gather)
 *      Step 3: MSTH 4-loop update                   (fused in k_membrane_*)
 *      Step 4: DSN dynamic decay alpha_t            (fused)
 *      Step 5: CTSN complement & s_tilde            (fused)
 *      Step 6: trinary readout                      (fused)
 *      Step 7: oscillator bank phase advance        (k_oscillator_advance)
 *      Step 8: neuromodulator field & receptors     (k_neuromod_update)
 *      Step 9: activity reductions                  (k_sphere_activity_stats)
 *
 *  All kernels assume Structure-of-Arrays device layout (see cuNxon_internal.cuh)
 *  and operate over a single sphere.  Multi-sphere parallelism is achieved by
 *  launching these kernels on per-sphere CUDA streams.
 * ============================================================================= */

#include "cuNxon_internal.cuh"
#include <math_constants.h>


/* ----------------------------------------------------------------------------
 *  Device helpers
 * ---------------------------------------------------------------------------- */
__device__ __forceinline__ float dev_clamp(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ float dev_sigmoid(float x) {
    /* Numerically stable */
    if (x >= 0.f) {
        float e = __expf(-x);
        return 1.f / (1.f + e);
    } else {
        float e = __expf(x);
        return e / (1.f + e);
    }
}

__device__ __forceinline__ int   dev_sign_i(float x) {
    return (x > 0.f) - (x < 0.f);
}

/* Trinary readout helper -- maps state to -1,0,+1 from continuous value. */
__device__ __forceinline__ int8_t dev_trinary(float v, float th_pos, float th_neg) {
    if (v > th_pos) return  1;
    if (v < th_neg) return -1;
    return 0;
}


/* ============================================================================
 *  k_oscillator_advance
 *
 *      phi_b <- phi_b + 2*pi*freq_b * dt_s
 *
 *  PAC: slow band phase modulates gamma amplitude via amp_gamma *= (1 + pac *
 *  cos(phi_theta))   (mirrors the cross-frequency coupling described in the
 *  Neuraxon v2.0 paper, Sec. A "Cross-Frequency Coupling").
 * ============================================================================ */
__global__ void k_oscillator_advance(OscillatorBankDev O, float dt_ms) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;        /* tiny kernel */
    const float dt_s = dt_ms * 1e-3f;
    const float two_pi = 6.28318530717958647692f;

    #pragma unroll
    for (int b = 0; b < CUNXON_BAND_COUNT; ++b) {
        float p_ = O.phase[b];
        p_ += two_pi * O.freq_hz[b] * dt_s;
        if (p_ >  CUDART_PI_F * 8.f) p_ -= CUDART_PI_F * 8.f;
        O.phase[b] = p_;
    }
    /* PAC: theta phase -> gamma amplitude */
    float pac_mod = 1.f + O.pac_strength * cosf(O.phase[CUNXON_BAND_THETA]);
    O.amp[CUNXON_BAND_GAMMA] = fmaxf(0.f, pac_mod);
}


/* ============================================================================
 *  k_neuromod_update
 *
 *  Global volume-transmission field (one per sphere) with:
 *      - tonic relaxation toward baseline (tau_tonic, slow)
 *      - phasic decay toward 0           (tau_phasic, fast)
 *      - activity-driven release         (rate proportional to network stats)
 *      - 9 receptor subtypes with logistic activation curves
 *
 *  Modulators: 0=DA, 1=5HT, 2=ACh, 3=NA
 *  Receptors : 0=D1 1=D2 2=5HT1A 3=5HT2A 4=5HT4 5=M1 6=M2 7=beta1 8=alpha2
 * ============================================================================ */
__device__ __forceinline__ float dev_receptor_activation(
        float concentration, float threshold, float gain, float slope_)
{
    float k = slope_;
    float x = -k * (concentration - threshold);
    x = dev_clamp(x, -50.f, 50.f);
    return gain / (1.f + __expf(x));
}

__global__ void k_neuromod_update(NeuromodFieldDev M,
                                  const cunxonNetworkParameters_t* p,
                                  float dt_ms,
                                  float mean_activity,
                                  float exc_fraction,
                                  float state_change_rate)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float baseline[4] = {
        p->dopamine_baseline, p->serotonin_baseline,
        p->acetylcholine_baseline, p->norepinephrine_baseline
    };

    /* 1. Tonic / phasic dynamics ------------------------------------------- */
    for (int i = 0; i < 4; ++i) {
        float tonic  = M.tonic[i];
        float phasic = M.phasic[i];
        tonic  += (dt_ms / p->tau_tonic ) * (baseline[i] - tonic);
        phasic += (dt_ms / p->tau_phasic) * (0.f         - phasic);
        M.tonic[i]  = tonic;
        M.phasic[i] = phasic;
    }

    /* 2. Activity-driven release ------------------------------------------- */
    const float rr = p->neuromod_release_rate;
    M.phasic[0]  += rr * state_change_rate * dt_ms;       /* DA  phasic */
    M.tonic [1]  += rr * mean_activity     * dt_ms;       /* 5HT tonic  */
    /* ACh release is gated DOWN by DA phasic (paper Algorithm 1 line 10):
     *   AChphasic ← (AChphasic + rate·fexc·dt) × max(0, 1 − 0.1·DAphasic)   */
    {
        float da_gate = fmaxf(0.f, 1.f - 0.1f * M.phasic[0]);
        M.phasic[2] += rr * exc_fraction * dt_ms;
        M.phasic[2] *= da_gate;
    }
    M.phasic[3]  += rr * state_change_rate * dt_ms;       /* NA  phasic */

    /* 3. Compute total concentrations and clip ----------------------------- */
    float conc[4];
    for (int i = 0; i < 4; ++i)
        conc[i] = dev_clamp(M.tonic[i] + M.phasic[i], 0.f, p->receptor_concentration_cap);

    /* 4. Receptor activations (9 subtypes).
     *    Thresholds / gains chosen to match the Python reference defaults.   */
    /* DA -> D1 (high-affinity, tonic) and D2 (low-affinity, phasic) */
    M.recept[0] = dev_receptor_activation(conc[0], 0.10f, 1.0f, 20.f);
    M.recept[1] = dev_receptor_activation(conc[0], 0.35f, 1.0f, 10.f);
    /* 5HT -> 5HT1A (tonic), 5HT2A (phasic), 5HT4 (mid) */
    M.recept[2] = dev_receptor_activation(conc[1], 0.05f, 1.0f, 25.f);
    M.recept[3] = dev_receptor_activation(conc[1], 0.40f, 1.0f, 10.f);
    M.recept[4] = dev_receptor_activation(conc[1], 0.20f, 1.0f, 15.f);
    /* ACh -> M1 (tonic), M2 (phasic) */
    M.recept[5] = dev_receptor_activation(conc[2], 0.15f, 1.0f, 18.f);
    M.recept[6] = dev_receptor_activation(conc[2], 0.45f, 1.0f, 10.f);
    /* NA -> beta1 (phasic), alpha2 (tonic) */
    M.recept[7] = dev_receptor_activation(conc[3], 0.30f, 1.0f, 12.f);
    M.recept[8] = dev_receptor_activation(conc[3], 0.08f, 1.0f, 22.f);
}


/* ============================================================================
 *  k_sphere_activity_stats
 *      Reductions over the trinary states of a sphere needed by the
 *      neuromodulator field:
 *          mean(|s|), exc_fraction = #(s==1) / n_total,
 *          state_change_rate = #(s != s_prev) / n_total
 * ============================================================================ */
__global__ void k_sphere_activity_stats(const NeuronArraysDev N, int n_total,
                                        float* out_mean_abs,
                                        float* out_exc_frac,
                                        float* out_change_rate)
{
    __shared__ float sh_abs[256];
    __shared__ float sh_exc[256];
    __shared__ float sh_chg[256];

    int tid = threadIdx.x;
    sh_abs[tid] = 0.f;
    sh_exc[tid] = 0.f;
    sh_chg[tid] = 0.f;

    for (int i = blockIdx.x * blockDim.x + tid; i < n_total;
         i += blockDim.x * gridDim.x)
    {
        int8_t s  = N.s[i];
        int8_t sp = N.s_prev[i];
        sh_abs[tid] += (s != 0) ? 1.f : 0.f;
        sh_exc[tid] += (s == 1) ? 1.f : 0.f;
        sh_chg[tid] += (s != sp) ? 1.f : 0.f;
    }
    __syncthreads();

    /* tree reduction */
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            sh_abs[tid] += sh_abs[tid + off];
            sh_exc[tid] += sh_exc[tid + off];
            sh_chg[tid] += sh_chg[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out_mean_abs,   sh_abs[0]);
        atomicAdd(out_exc_frac,   sh_exc[0]);
        atomicAdd(out_change_rate, sh_chg[0]);
    }
}


/* ============================================================================
 *  k_chrono_warp_and_isyn   --   Algorithm 1 STEP 1
 *
 *  For each active synapse j -> i:
 *      spre   = clamp(N.s[j], -1, +1)
 *      z_gate = tanh(z_trace / gate_norm)
 *      raw    = 2*spre + 1.5*z_gate - 1                          (Eq 7)
 *      omega  = clamp(sigmoid(clamp(raw,-clip,clip)),
 *                     omega_min, omega_max)                       (Alg 1 line 24)
 *      omega  = (1 - beta_ema) * omega_prev + beta_ema * omega    (line 25)
 *      f_trace = clip(alpha_f * f_trace + spre, -Tclip, +Tclip)   (Eq 6)
 *      z_trace = clip((alpha_s ^ omega) * z_trace + spre, ...)    (Eq 7, line 27)
 *      Isyn    = (w_fast + w_slow)*spre
 *              + lambda_f * w_fast * f_trace
 *              + lambda_s * w_slow * z_trace
 *
 *  Isyn is atomicAdded into N.branch_sum[i * n_branches + branch_id].
 *  This is the only place where post-side scatter happens in the forward pipe.
 * ============================================================================ */
__global__ void k_chrono_warp_and_isyn(NeuronArraysDev N, SynapseArraysDev S,
                                       int n_syn,
                                       const cunxonNetworkParameters_t* p,
                                       float dt_ms,
                                       float* d_energy)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_syn) return;
    if (S.is_silent[idx]) return;

    int pre  = S.pre_id [idx];
    int post = S.post_id[idx];
    int br   = S.branch_id[idx];

    float spre = (float)N.s[pre];
    spre = dev_clamp(spre, -1.f, 1.f);

    float w_fast = S.w_fast[idx];
    float w_slow = S.w_slow[idx];

    float Isyn;
    if (p->chrono_enabled) {
        float f_tr = S.chrono_fast_trace[idx];
        float z_tr = S.chrono_slow_trace[idx];
        float om_prev = S.chrono_omega[idx];

        /* --- omega controller --- */
        float z_gate;
        if (p->chrono_gate_norm > 0.f) {
            z_gate = tanhf(z_tr / p->chrono_gate_norm);
        } else {
            z_gate = z_tr;
        }
        float raw = 2.f * spre + 1.5f * z_gate - 1.f;
        if (p->chrono_raw_clip > 0.f)
            raw = dev_clamp(raw, -p->chrono_raw_clip, p->chrono_raw_clip);
        float om_new = dev_sigmoid(raw);
        om_new = dev_clamp(om_new, p->chrono_omega_min, p->chrono_omega_max);
        /* EMA smoothing */
        float beta_ema = p->chrono_omega_smoothing;
        float omega = (1.f - beta_ema) * om_prev + beta_ema * om_new;
        S.chrono_omega[idx] = omega;

        /* --- traces --- */
        float alpha_f = p->chrono_alpha_f;
        float alpha_s = p->chrono_alpha_s;
        /* alpha_s ^ omega  via expf(omega * logf(alpha_s)) ; alpha_s in (0,1) */
        float alpha_s_pow = __expf(omega * __logf(alpha_s));

        float Tclip = p->chrono_trace_clip;
        f_tr = alpha_f     * f_tr + spre;
        z_tr = alpha_s_pow * z_tr + spre;
        if (Tclip > 0.f) {
            f_tr = dev_clamp(f_tr, -Tclip, Tclip);
            z_tr = dev_clamp(z_tr, -Tclip, Tclip);
        }
        S.chrono_fast_trace[idx] = f_tr;
        S.chrono_slow_trace[idx] = z_tr;

        Isyn = (w_fast + w_slow) * spre
             + p->chrono_lambda_f * w_fast * f_tr
             + p->chrono_lambda_s * w_slow * z_tr;
    } else {
        Isyn = (w_fast + w_slow) * spre;
    }

    /* Modulatory (metabotropic) synapses contribute slow neuromodulatory drive,
     * not direct ionotropic current; they bypass the dendritic-branch gather
     * and are scattered into a dedicated per-neuron accumulator that the
     * membrane kernel reads as a threshold/gain shift.                       */
    if (S.is_modulatory[idx]) {
        atomicAdd(&N.modulatory_pot[post], S.w_meta[idx] * spre);
        return;
    }

    /* Scatter Isyn into the appropriate branch sum (live, this step only).   */
    int nb_per_n = /* n_branches (= p->num_dendritic_branches) */
                   p->num_dendritic_branches;
    atomicAdd(&N.branch_sum[post * nb_per_n + br], Isyn);
    /* Energy: |Isyn| * (|w_fast|+|w_slow|+|w_meta|).  Sums per-step over
     * all active synapses; host then accumulates this scalar across steps. */
    if (d_energy) {
        float w_tot = fabsf(S.w_fast[idx]) + fabsf(S.w_slow[idx])
                    + fabsf(S.w_meta[idx]);
        atomicAdd(d_energy, fabsf(Isyn) * w_tot);
    }
}


/* ============================================================================
 *  k_dendritic_gather  --  Algorithm 1 STEP 2 (supralinear branch summation)
 *
 *  For each neuron i:
 *      For each branch b in [0, B-1]:
 *          sigma_b = N.branch_sum[i * B + b]
 *          if (|sigma_b| > theta_d):
 *              contrib_b = sign(sigma_b) * |sigma_b|^gamma_supra
 *          else:
 *              contrib_b = sigma_b
 *      D_raw[i] = sum_b contrib_b
 *
 *  D_raw is written back into N.branch_pot[i * B + 0] (reused as scratch).
 *  The membrane kernel reads it as the "dendritic drive".
 * ============================================================================ */
__global__ void k_dendritic_gather(NeuronArraysDev N, const SynapseArraysDev S,
                                   int n_total, int n_branches,
                                   const cunxonNetworkParameters_t* p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    if (!N.is_active[i]) return;

    float theta_d   = p->dendritic_spike_threshold;
    float gamma_sup = p->dendritic_supralinear_gamma;

    float D = 0.f;
    for (int b = 0; b < n_branches; ++b) {
        float sigma_b = N.branch_sum[i * n_branches + b];
        float c;
        if (fabsf(sigma_b) > theta_d) {
            float sgn = (sigma_b > 0.f) ? 1.f : -1.f;
            c = sgn * powf(fabsf(sigma_b), gamma_sup);
        } else {
            c = sigma_b;
        }
        D += c;
        /* zero out scratch for next step */
        N.branch_sum[i * n_branches + b] = 0.f;
    }
    /* Save D into branch_pot[i*B + 0] for membrane stage (slot 0 reused) */
    N.branch_pot[i * n_branches + 0] = D;
}


/* ============================================================================
 *  k_membrane_dsn_ctsn_emit   --   Algorithm 1 STEPS 3-7 (fused)
 *
 *  Pipeline per neuron i (skipping input neurons, which take ext_in directly):
 *
 *   (Step 3) MSTH four loops:
 *       MSTH_uf  <- (1 - dt/tau_uf) * MSTH_uf  + (dt/tau_uf) * |s_prev|
 *       MSTH_f   <- (1 - dt/tau_f ) * MSTH_f   + (dt/tau_f ) * |s_prev|
 *       e_rate   <- |s_prev| - r_tgt
 *       MSTH_m   <- clamp(MSTH_m - dt/tau_m * (eta_m * e_rate * MSTH_m),
 *                          0.5, 2.0)
 *       MSTH_s   <- (1 - dt/tau_s ) * MSTH_s   + (dt/tau_s ) * |e_rate|
 *
 *   (Step 4) DSN dynamic decay:
 *       buf[head] <- D_raw_input    (raw drive incl. ext + osc - adapt)
 *       conv      <- sum_k K[k] * buf[(head - k) mod K]   + bdsn
 *       alpha_t   <- sigmoid(-conv)
 *       (optional local kernel update if dsn_learn_enabled)
 *
 *   (Step 5) Membrane:
 *       g_NA  <- 1 + 0.5*[beta1] + 0.2*[alpha2]
 *       U     <- alpha_t * U + (1-alpha_t) * (D_raw * MSTH_m * g_NA)
 *       if MSTH_uf > ceiling: U *= 0.5
 *
 *   (Step 6) CTSN complement:
 *       phi   <- tanh(gamma_i * D_raw + beta_i)
 *       h     <- rho * h + (1 - rho) * phi
 *       s_til <- U + h
 *
 *   (Step 7) Trinary readout with modulated thresholds:
 *       d_th_m  <- k * tanh(sum_meta + 0.3*[M1] - 0.2*[M2])
 *       th_pos  <- theta_exc - d_th_m + homeo + MSTH_f_shift - 0.1*autorecep
 *       th_neg  <- theta_inh - d_th_m + homeo + MSTH_f_shift + 0.1*autorecep
 *       s       <- 1 if s_til > th_pos else (-1 if s_til < th_neg else 0)
 *
 *   (AGMP astrocyte update line 59):
 *       a <- lambda_a * a + (1 - lambda_a) * |s_til|
 *
 *  For input neurons (type == INPUT): s = trinary(ext_in[i], theta_in, -theta_in).
 * ============================================================================ */
__global__ void k_membrane_dsn_ctsn_emit(NeuronArraysDev N,
                                         const NeuromodFieldDev M,
                                         OscillatorBankDev O,
                                         int n_total,
                                         const cunxonNetworkParameters_t* p,
                                         const float* ext_in,
                                         int  n_in,
                                         float dt_ms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    if (!N.is_active[i]) return;

    /* Save previous state before overwriting */
    N.s_prev[i] = N.s[i];

    int nb = p->num_dendritic_branches;

    /* --- INPUT neurons: pure-state pass-through ------------------------- *
     * Threshold 0.20 chosen for SNR balance:
     *   - rejects typical sensory noise (~0.20 magnitude) cleanly
     *   - accepts properly-scaled inter-sphere contribs (~0.3+ when W init
     *     is Xavier-tuned), see allocate_link W initialisation.            */
    if (N.type[i] == (int8_t)CUNXON_NEURON_INPUT) {
        float x = (i < n_in) ? ext_in[i] : 0.f;
        N.U[i]        = x;
        N.s_tilde[i]  = x;
        N.complement_h[i] = 0.f;
        N.s[i] = dev_trinary(x, 0.20f, -0.20f);
        return;
    }

    /* Dendritic gather (k_dendritic_gather) already SUMS across all
     * branches and stores total D in branch_pot[i * nb + 0].  Slots 1..nb-1
     * are unused.  Reading slot 0 is correct.                              */
    float D_raw = N.branch_pot[i * nb + 0];

    /* === STEP 3: MSTH ============================================== */
    float s_abs_prev = fabsf((float)N.s_prev[i]);

    float msth_uf = N.msth_ultrafast[i];
    float msth_f  = N.msth_fast[i];
    float msth_m  = N.msth_medium[i];
    float msth_s  = N.msth_slow[i];

    float a_uf = dt_ms / fmaxf(p->msth_ultrafast_tau, 1e-3f);
    float a_f  = dt_ms / fmaxf(p->msth_fast_tau,      1e-3f);
    float a_m  = dt_ms / fmaxf(p->msth_medium_tau,    1e-3f);
    float a_s  = dt_ms / fmaxf(p->msth_slow_tau,      1e-3f);

    msth_uf = (1.f - a_uf) * msth_uf + a_uf * s_abs_prev;
    msth_f  = (1.f - a_f ) * msth_f  + a_f  * s_abs_prev;
    float e_rate = s_abs_prev - p->target_firing_rate;
    msth_m = dev_clamp(msth_m - a_m * (p->msth_medium_gain * e_rate * msth_m),
                       0.5f, 2.0f);
    msth_s = (1.f - a_s) * msth_s + a_s * fabsf(e_rate);

    N.msth_ultrafast[i] = msth_uf;
    N.msth_fast[i]      = msth_f;
    N.msth_medium[i]    = msth_m;
    N.msth_slow[i]      = msth_s;

    /* === Oscillator drive at this neuron (paper Algorithm 1 lines 16-19) =
     * Per-neuron spatial phase offset, θ-gated γ via cross-frequency
     * coupling, plus additive slow and infraslow contributions.            */
    float Phi_i = 2.f * CUDART_PI_F * ((float)i / fmaxf(1.f, (float)n_total));
    float osc_drive = 0.f;
    {
        /* band index convention (matches cunxonBand_t):
         *   0: INFRASLOW, 1: SLOW, 2: THETA, 3: ALPHA, 4: BETA, 5: GAMMA   */
        float phi_infra = O.phase[0];
        float phi_slow  = O.phase[1];
        float phi_theta = O.phase[2];
        float phi_gamma = O.phase[5];
        float amp_gamma = O.amp[5];
        float gate_theta = fmaxf(0.f, cosf(phi_theta + Phi_i));
        float drive_gamma = amp_gamma * gate_theta
                          * sinf(phi_gamma + 2.f * Phi_i);
        osc_drive = O.pac_strength * (drive_gamma
                                    + 0.5f * sinf(phi_slow)
                                    + 0.3f * sinf(phi_infra));
    }

    /* === Spontaneous firing (alpha2-modulated) ========================== */
    float alpha2_act = M.recept[8];
    float spont = 0.f;
    {
        float spont_rate = p->spontaneous_firing_rate + 0.3f * alpha2_act;
        curandState_t st = N.rng[i];
        if (curand_uniform(&st) < spont_rate * dt_ms) {
            spont = curand_normal(&st) * 0.3f;
        }
        N.rng[i] = st;
    }

    /* === NA-modulated gain g_NA ======================================= */
    float beta1_act = M.recept[7];
    float g_NA = 1.f + 0.5f * beta1_act + 0.2f * alpha2_act;

    /* === Raw input stream X(t) for DSN buffer ======================== */
    float ext = 0.f;
    if (N.type[i] == (int8_t)CUNXON_NEURON_INPUT && i < n_in)
        ext = ext_in[i];
    float D_scaled = D_raw * msth_m;
    float raw_in = g_NA * D_scaled + ext + osc_drive - N.adaptation[i] + spont;

    /* === STEP 4: DSN dynamic decay alpha_t ============================= */
    float alpha_t;
    if (p->dsn_enabled) {
        int K = p->dsn_kernel_size;
        int head = N.dsn_head[i];
        /* shift ring: replace oldest */
        head = (head + 1) % K;
        N.dsn_buffer[i * K + head] = raw_in;
        N.dsn_head[i] = head;

        /* causal conv: K[k] * buf[(head - k) mod K], k = 0..K-1 (newest first) */
        float conv = p->dsn_bias;
        for (int k = 0; k < K; ++k) {
            int idx_b = (head - k + K) % K;
            conv += N.dsn_kernel[i * K + k] * N.dsn_buffer[i * K + idx_b];
        }
        alpha_t = dev_sigmoid(-conv);          /* matches Python: sigmoid(-conv) */

        /* Optional local kernel learning (Alg 1 lines 44-46).
         * target alpha = sigmoid(bias_t - sens * |X_t - X_{t-1}|), update by
         *   Delta K_k = lr * (alpha - alpha_tgt) * alpha(1-alpha) * buf[k]
         * Followed by L1 renormalisation, clip first.                       */
        if (p->dsn_learn_enabled) {
            int prev_idx = (head - 1 + K) % K;
            float dx = fabsf(N.dsn_buffer[i*K + head]
                           - N.dsn_buffer[i*K + prev_idx]);
            float tgt_raw = p->dsn_target_bias - p->dsn_target_sensitivity * dx;
            tgt_raw = dev_clamp(tgt_raw, -50.f, 50.f);
            float alpha_tgt = 1.f / (1.f + __expf(-tgt_raw));
            float err = alpha_t - alpha_tgt;
            float dsig = alpha_t * (1.f - alpha_t);
            float common = err * dsig;

            /* update kernel weights with clip + L1 renormalisation */
            float l1 = 0.f;
            for (int k = 0; k < K; ++k) {
                int idx_b = (head - k + K) % K;
                float w = N.dsn_kernel[i*K + k];
                w -= p->dsn_learn_lr * common * N.dsn_buffer[i*K + idx_b];
                w = dev_clamp(w, -p->dsn_kernel_clip, p->dsn_kernel_clip);
                N.dsn_kernel[i*K + k] = w;
                l1 += fabsf(w);
            }
            if (l1 > 1e-8f) {
                for (int k = 0; k < K; ++k)
                    N.dsn_kernel[i*K + k] /= l1;
            }
        }
    } else {
        alpha_t = 0.5f;
    }
    N.dsn_alpha[i] = alpha_t;

    /* === STEP 5: Membrane update ====================================== */
    float U = N.U[i];
    if (p->dsn_enabled) {
        /* DSN's α_t is the dynamic decay; it implicitly encodes the membrane
         * time constant via the learned causal-conv kernel.                 */
        U = alpha_t * U + (1.f - alpha_t) * (D_raw * msth_m * g_NA);
    } else {
        /* Plain leaky-integrate fallback with explicit membrane_time_constant. */
        float tau_m = fmaxf(p->membrane_time_constant, 1e-3f);
        float a = dt_ms / tau_m;
        if (a > 1.f) a = 1.f;
        U = (1.f - a) * U + a * (D_raw * msth_m * g_NA);
    }
    if (msth_uf > p->msth_ultrafast_ceiling) U *= 0.5f;
    N.U[i] = U;

    /* === STEP 6: CTSN complement h(t) ================================= */
    float h = N.complement_h[i];
    float gamma_i = N.ctsn_phi_gain[i];
    float beta_i  = N.ctsn_phi_bias[i];
    float phi     = tanhf(gamma_i * D_raw + beta_i);
    float rho     = p->ctsn_rho;
    if (p->ctsn_enabled)
        h = rho * h + (1.f - rho) * phi;
    else
        h = 0.f;
    N.complement_h[i] = h;

    /* Optional CTSN online learning: drive both phi_gain and phi_bias toward
     * the target firing rate.  These per-neuron parameters become learned
     * via the gradient of the trinary readout w.r.t. (gamma, beta), which
     * is proportional to (1 - phi^2) for the tanh activation.              */
    if (p->ctsn_enabled && p->ctsn_learn_enabled) {
        float r_i = N.firing_rate_avg[i];
        float r_tgt = p->target_firing_rate;
        float err   = (r_tgt - r_i);
        float gate  = (1.f - phi * phi) * (1.f - rho);
        float d_gamma = p->ctsn_learn_lr * err * gate * D_raw;
        float d_beta  = p->ctsn_learn_lr * err * gate;
        float gnew = gamma_i + d_gamma;
        float bnew = beta_i  + d_beta;
        gnew = dev_clamp(gnew, -p->ctsn_phi_gain_clip, p->ctsn_phi_gain_clip);
        bnew = dev_clamp(bnew, -p->ctsn_phi_bias_clip, p->ctsn_phi_bias_clip);
        N.ctsn_phi_gain[i] = gnew;
        N.ctsn_phi_bias[i] = bnew;
    }

    float s_til = U + h;
    N.s_tilde[i] = s_til;

    /* === Modulated thresholds + STEP 7: trinary readout ================
     * Two classes of threshold modulation:
     *   (a) FACILITATIVE   (d_th_m) — neuromodulator-driven (DA, ACh M1/M2).
     *       Bring thresholds TOWARD zero: lower |th_pos|, raise |th_neg|
     *       → firing easier in BOTH directions.
     *   (b) HOMEOSTATIC    (homeo, th_shift_fast, autor) — track over-firing.
     *       Push thresholds APART: raise th_pos, lower th_neg
     *       → firing HARDER in BOTH directions, restoring a rest zone.
     *
     *  The previous version used same-sign shifts for all terms, which
     *  squeezed the rest band [th_neg, th_pos] toward zero and produced
     *  the "no quiet neurons" pathology (every neuron forced to ±1).      */
    float M_pot   = N.modulatory_pot[i];
    float M_gate  = tanhf(0.5f * M_pot);            /* in (-1, +1)         */
    float M1 = M.recept[5];
    float M2 = M.recept[6];
    float raw_mod = 0.3f * M1 - 0.2f * M2 + 0.4f * M_gate;
    float d_th_m = p->threshold_mod_k * tanhf(raw_mod);
    float homeo = p->homeostatic_rate * (N.firing_rate_avg[i] - p->target_firing_rate);
    float th_shift_fast = p->msth_fast_gain * (msth_f - p->target_firing_rate);
    float autor = N.autoreceptor[i];

    /* th_pos:  facilitative LOWERS it, homeostatic RAISES it. */
    float th_pos = p->firing_threshold_excitatory
                 - d_th_m
                 + homeo
                 + th_shift_fast
                 + 0.1f * autor;
    /* th_neg:  facilitative RAISES it (toward zero), homeostatic LOWERS it. */
    float th_neg = p->firing_threshold_inhibitory
                 + d_th_m
                 - homeo
                 - th_shift_fast
                 - 0.1f * autor;

    int8_t snew = dev_trinary(s_til, th_pos, th_neg);
    N.s[i] = snew;

    /* Firing-rate running average (Eq 1) */
    float fr = N.firing_rate_avg[i];
    fr += p->firing_rate_alpha * (fabsf((float)snew) - fr) * dt_ms;
    N.firing_rate_avg[i] = fr;

    /* AGMP astrocyte state */
    float la = p->agmp_lambda_a;
    N.astrocyte[i] = la * N.astrocyte[i] + (1.f - la) * fabsf(s_til);

    /* Adaptation / autoreceptor leak */
    N.adaptation[i]  += (-N.adaptation[i]  / p->adaptation_tau    + fabsf((float)snew)) * dt_ms;
    N.autoreceptor[i] += (-N.autoreceptor[i] / p->autoreceptor_tau + fabsf((float)snew)) * dt_ms;

    /* Health: decays with sustained over-activation OR sustained silence;
     * the slow MSTH loop tracks long-horizon accumulated rate error, scaled
     * by msth_slow_gain.  A neuron whose health falls below
     * neuron_death_threshold is marked inactive and stops firing.           */
    float health = N.health[i];
    float deviation = fabsf(fr - p->target_firing_rate);
    float over = fmaxf(0.f, deviation - 2.f * p->target_firing_rate);
    float slow_press = p->msth_slow_gain * msth_s;     /* long-horizon term  */
    health -= (p->neuron_health_decay * over + 0.1f * slow_press) * dt_ms;
    health  = dev_clamp(health, 0.f, 1.f);
    N.health[i] = health;
    if (health < p->neuron_death_threshold) {
        N.is_active[i] = 0;
        N.s[i] = 0;                       /* dead neurons emit nothing       */
    }
}


/* ============================================================================
 *  k_reset_neuron_dynamic
 *      Reset all dynamic state (U, traces, MSTH, complement_h) keeping
 *      learned weights and per-neuron DSN/CTSN kernels intact.
 * ============================================================================ */
__global__ void k_reset_neuron_dynamic(NeuronArraysDev N, int n_total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    N.U[i] = 0.f;
    N.adaptation[i] = 0.f;
    N.autoreceptor[i] = 0.f;
    N.s[i] = 0;
    N.s_prev[i] = 0;
    N.complement_h[i] = 0.f;
    N.s_tilde[i] = 0.f;
    N.msth_ultrafast[i] = 0.f;
    N.msth_fast[i] = 0.f;
    N.msth_medium[i] = 1.f;
    N.msth_slow[i] = 0.f;
    N.astrocyte[i] = 0.f;
    N.modulatory_pot[i] = 0.f;
    /* Revive any neurons killed by health decay (this is dynamic state). */
    N.is_active[i] = 1;
    N.health[i]    = 1.f;
    /* keep dsn_kernel and ctsn_phi_* (learned parameters) */
}
