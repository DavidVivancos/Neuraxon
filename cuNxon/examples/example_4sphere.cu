/* ============================================================================
 *  example_4sphere.cu
 *  ---------------------------------------------------------------------------
 *  Minimal end-to-end demo of cuNxon: a 4-sphere Multi-Neuraxon "brain"
 *  inspired by the NAS-optimised topology in the Multi-Neuraxon AGI'26 paper.
 *
 *      VIS  (sensory, visual)  ----FF/gamma---\
 *                                                \
 *      AUD  (sensory, audio)   ----FF/gamma----->  ASC  (association)  --FF/gamma-->  MTR  (motor)
 *                                                /       ^   |
 *      VIS <----Lat/theta----> AUD              /         \  |
 *                                              /           \ +----FB/beta----MTR
 *      ASC ----Thalamic/theta-->  VIS  and  AUD  (top-down attention)
 *
 *  At each step we drive VIS and AUD with toy stimuli, supply a reward signal
 *  (the network is trained to fire MTR's first readout neuron whenever VIS
 *  contains a "salient" pattern), then read MTR's trinary output.
 *
 *  Build (example):
 *      nvcc -O3 -std=c++14 -arch=sm_70 -I../include \
 *           example_4sphere.cu ../src/*.cu -lcurand -o example_4sphere
 *  or use the provided CMakeLists.txt:
 *      cmake -S .. -B ../build && cmake --build ../build -j
 *      ../build/example_4sphere
 *
 *  (c) 2026  cuNxon authors -- MIT licence
 * ========================================================================== */

#include <cuNxon.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

/* ---- tiny helper ----------------------------------------------------------- */
#define CHECK(call) do {                                                       \
    cunxonStatus_t _s = (call);                                                \
    if (_s != CUNXON_OK) {                                                     \
        std::fprintf(stderr, "[cuNxon error] %s: %s (%s) at %s:%d\n",          \
                     #call, cunxonGetStatusString(_s),                         \
                     cunxonGetLastError(), __FILE__, __LINE__);                \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

/* ---- per-sphere helpers ---------------------------------------------------- */
static void fill_sphere_params(cunxonNetworkParameters_t* p,
                               int n_in, int n_hid, int n_out,
                               uint64_t seed_offset)
{
    cunxonGetDefaultParameters(p);
    p->num_input_neurons       = n_in;
    p->num_hidden_neurons      = n_hid;
    p->num_output_neurons      = n_out;
    p->random_seed_offset      = seed_offset;
}

static void fill_link_params(cunxonLinkParameters_t* lp,
                             cunxonLinkKind_t kind, cunxonBand_t band,
                             float gain, int delay, float coherence)
{
    std::memset(lp, 0, sizeof(*lp));
    lp->kind                    = kind;
    lp->coherence_band          = band;
    lp->gain                    = gain;
    lp->delay_steps             = delay;
    lp->transmission_threshold  = 0.0f;
    lp->coherence_strength      = coherence;
    lp->topology                = CUNXON_TOPO_DENSE;
    lp->sparse_prob             = 0.30f;
    lp->allow_negative_weights  = 1;
    lp->plasticity_rate         = 1e-3f;
    lp->weight_decay            = 1e-5f;
    lp->weight_clip             = 1.0f;
    lp->normalize_rows          = 0;
    lp->bias                    = 0.0f;
}

/* ---- demo ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    /* ===== 0. CLI ========================================================== */
    int n_train_steps  = (argc > 1) ? std::atoi(argv[1]) : 3000;
    int n_infer_steps  = (argc > 2) ? std::atoi(argv[2]) : 400;
    int device_id      = (argc > 3) ? std::atoi(argv[3]) : 0;
    uint64_t base_seed = 0xC0FFEE2026ULL;

    std::printf("=== cuNxon 4-sphere demo ===\n");
    std::printf("  device_id        : %d\n", device_id);
    std::printf("  warmup steps     : %d\n", n_train_steps);
    std::printf("  testing steps    : %d   (plasticity stays on per paper)\n", n_infer_steps);

    /* ===== 1. context ====================================================== */
    cunxonContext_t ctx = nullptr;
    CHECK(cunxonCreateContext(&ctx, device_id, base_seed, /*flags=*/0));

    char dev_name[256] = {0};
    int  cc = 0; size_t total_mem = 0;
    cunxonContextGetProperty(ctx, CUNXON_PROP_DEVICE_NAME,
                             dev_name, sizeof(dev_name));
    cunxonContextGetProperty(ctx, CUNXON_PROP_COMPUTE_CAPABILITY,
                             &cc, sizeof(cc));
    cunxonContextGetProperty(ctx, CUNXON_PROP_TOTAL_GLOBAL_MEM,
                             &total_mem, sizeof(total_mem));
    std::printf("  device           : %s (cc %d.%d, %.1f GB)\n",
                dev_name, cc/10, cc%10, total_mem/1.0e9);

    /* ===== 2. network ====================================================== */
    cunxonNetwork_t net = nullptr;
    CHECK(cunxonNetworkCreate(ctx, &net, "demo_4sphere"));

    /* ----- per-sphere parameter sets -------------------------------------- *
     * Sizes inspired by the NAS-optimised AGI'26 topology, scaled down for
     * a quick demo. Pattern: sensory spheres small & specialised;
     * association sphere larger with rich recurrence; motor sphere narrow.   */
    cunxonNetworkParameters_t pVIS, pAUD, pASC, pMTR;
    fill_sphere_params(&pVIS,  6, 44, 14, 1);
    fill_sphere_params(&pAUD,  6, 44, 14, 2);
    fill_sphere_params(&pASC, 30, 51, 17, 3);   /* association: larger */
    fill_sphere_params(&pMTR, 19, 21,  5, 4);   /* motor: narrow */

    /* gently tweak ASC for richer dynamics */
    pASC.ws_k        = 8;
    pASC.ws_beta     = 0.15f;

    /* === DEMO-TIMESCALE TUNING ============================================= *
     * Paper defaults use biological timescales (msth_medium_tau = 5 min,
     * msth_slow_tau = 1 hour).  With dt=1ms and 1500 training steps we
     * simulate only 1.5 seconds — homeostasis can't engage and the network
     * runs away to saturation (every neuron fires every step).               *
     * Compress timescales 1000x and strengthen homeostatic gain so the
     * `target_firing_rate=0.2` constraint actually pulls activity down.      *
     * Also disable structural plasticity for the demo: synapse death runs
     * faster than STDP can re-strengthen, killing the network underneath.   */
    auto tune_for_demo = [](cunxonNetworkParameters_t& p) {
        p.msth_fast_tau         = 2.0f;        /* was 2000   */
        p.msth_medium_tau       = 300.0f;      /* was 300000 */
        p.msth_slow_tau         = 3600.0f;     /* was 3600000 */
        p.msth_medium_gain      = 1.0f;        /* was 0.001 — 1000x stronger */
        /* Disable structural growth/death during the demo */
        p.synapse_death_prob    = 0.0f;
        p.synapse_formation_prob= 0.0f;
        /* Reduce dendritic supralinear amplification (1.3 was positive-fb) */
        p.dendritic_supralinear_gamma = 1.0f;  /* linear, no amplification */
        /* Use library-default intra-sphere weight ranges (symmetric:
         * wf=±0.8, ws=±0.4, wm=±0.3 — the model author's design).
         * With our other fixes in place (threshold sign, fast MSTH,
         * oscillator subthreshold, structural plasticity off), saturation
         * no longer occurs and we don't need to over-suppress weights.    */
        /* Stronger plasticity rate to make 1500 steps count for learning */
        p.learning_rate         = 0.05f;       /* was 0.01 — 5x faster STDP */
        /* Lower firing thresholds so signal can propagate through multiple
         * sphere layers.  With default 0.4, drive degrades by sqrt() per
         * hop and MTR's output layer struggles to fire from the residual.
         * 0.25 lets propagation reach the readout reliably.                */
        p.firing_threshold_excitatory =  0.25f;
        p.firing_threshold_inhibitory = -0.25f;
        /* Plasticity weight time constants.  Library defaults are 5/50/1000 ms
         * — wf relaxes to 0.3*dw with τ=5ms, so when dw≈0 (no DA), wf decays
         * to zero within 25 steps.  All synapses go dead before learning
         * can happen.  Stretch the time constants 100x so initial weights
         * persist over the 1500-step demo window.                          */
        p.tau_fast  =   500.f;     /* was 5    */
        p.tau_slow  =  5000.f;     /* was 50   */
        p.tau_meta  = 50000.f;     /* was 1000 */
    };
    tune_for_demo(pVIS);
    tune_for_demo(pAUD);
    tune_for_demo(pASC);
    tune_for_demo(pMTR);

    int vis_id=-1, aud_id=-1, asc_id=-1, mtr_id=-1;
    CHECK(cunxonNetworkAddSphere(net, "VIS", CUNXON_SPHERE_SENSORY,     &pVIS, &vis_id));
    CHECK(cunxonNetworkAddSphere(net, "AUD", CUNXON_SPHERE_SENSORY,     &pAUD, &aud_id));
    CHECK(cunxonNetworkAddSphere(net, "ASC", CUNXON_SPHERE_ASSOCIATION, &pASC, &asc_id));
    CHECK(cunxonNetworkAddSphere(net, "MTR", CUNXON_SPHERE_MOTOR,       &pMTR, &mtr_id));

    /* ----- interface definitions ------------------------------------------ *
     * For each sphere we say which input neurons are driven by the outside
     * world (sensory_input_ids) vs. by relays from other spheres
     * (relay_input_ids), and which output neurons act as relays
     * (relay_output_ids) vs. final readouts (readout_output_ids).             */

    auto fill_range = [](std::vector<int>& v, int a, int b) {
        v.clear(); v.reserve(b - a);
        for (int i = a; i < b; ++i) v.push_back(i);
    };
    std::vector<int> sin, rin, rout, rdo;

    /* Index convention: input [0, n_in), hidden [n_in, n_in+n_hid),
     *                   output [n_in+n_hid, n_total).
     * Relay-out and readout MUST point to OUTPUT neurons (the last block)
     * — those are the ones with learning dynamics. Pointing them at
     * indices 0..N would select INPUT neurons which are just pass-through. */

    /* VIS: all inputs are external; last 14 (output) neurons are relays to ASC */
    {
        int v_out_base = pVIS.num_input_neurons + pVIS.num_hidden_neurons;
        int v_out_top  = v_out_base + pVIS.num_output_neurons;
        fill_range(sin,  0, pVIS.num_input_neurons);
        fill_range(rin,  0, 0);
        fill_range(rout, v_out_base, v_out_top - 2);       /* first 12 outputs go to other spheres */
        fill_range(rdo,  v_out_top - 2, v_out_top);        /* last 2 outputs are readouts */
        CHECK(cunxonNetworkSetSphereInterface(net, vis_id,
                                              sin.data(),  (int)sin.size(),
                                              rin.data(),  (int)rin.size(),
                                              rout.data(), (int)rout.size(),
                                              rdo.data(),  (int)rdo.size()));
    }

    /* AUD: same shape as VIS */
    {
        int a_out_base = pAUD.num_input_neurons + pAUD.num_hidden_neurons;
        int a_out_top  = a_out_base + pAUD.num_output_neurons;
        fill_range(sin,  0, pAUD.num_input_neurons);
        fill_range(rin,  0, 0);
        fill_range(rout, a_out_base, a_out_top - 2);
        fill_range(rdo,  a_out_top - 2, a_out_top);
        CHECK(cunxonNetworkSetSphereInterface(net, aud_id,
                                              sin.data(),  (int)sin.size(),
                                              rin.data(),  (int)rin.size(),
                                              rout.data(), (int)rout.size(),
                                              rdo.data(),  (int)rdo.size()));
    }

    /* ASC: inputs are relay-receivers; outputs route to MTR + 3 readouts */
    {
        int c_out_base = pASC.num_input_neurons + pASC.num_hidden_neurons;
        int c_out_top  = c_out_base + pASC.num_output_neurons;
        fill_range(sin,  0, 0);                              /* no external sensors */
        fill_range(rin,  0, pASC.num_input_neurons);         /* every input is a relay */
        fill_range(rout, c_out_base, c_out_top - 3);         /* 14 outputs go to MTR (and feedback to VIS/AUD) */
        fill_range(rdo,  c_out_top - 3, c_out_top);          /* last 3 are host-visible readouts */
        CHECK(cunxonNetworkSetSphereInterface(net, asc_id,
                                              sin.data(),  (int)sin.size(),
                                              rin.data(),  (int)rin.size(),
                                              rout.data(), (int)rout.size(),
                                              rdo.data(),  (int)rdo.size()));
    }

    /* MTR: 5 output neurons serve double duty — readouts AND efference-copy
     * relay-outs that feed back to ASC.  Same neurons, both port arrays.    */
    {
        int m_out_base = pMTR.num_input_neurons + pMTR.num_hidden_neurons;
        int m_out_top  = m_out_base + pMTR.num_output_neurons;
        fill_range(sin,  0, 0);
        fill_range(rin,  0, pMTR.num_input_neurons);
        fill_range(rout, m_out_base, m_out_top);
        fill_range(rdo,  m_out_base, m_out_top);
        CHECK(cunxonNetworkSetSphereInterface(net, mtr_id,
                                              sin.data(),  (int)sin.size(),
                                              rin.data(),  (int)rin.size(),
                                              rout.data(), (int)rout.size(),
                                              rdo.data(),  (int)rdo.size()));
    }

    /* ----- inter-sphere links --------------------------------------------- */
    cunxonLinkParameters_t lp;
    int  link_id = -1;

    /* VIS -> ASC  (feedforward, gamma) */
    fill_link_params(&lp, CUNXON_LINK_FEEDFORWARD, CUNXON_BAND_GAMMA, 1.0f, 1, 0.6f);
    CHECK(cunxonNetworkAddLink(net, vis_id, asc_id, &lp, &link_id));

    /* AUD -> ASC  (feedforward, gamma) */
    fill_link_params(&lp, CUNXON_LINK_FEEDFORWARD, CUNXON_BAND_GAMMA, 1.0f, 1, 0.6f);
    CHECK(cunxonNetworkAddLink(net, aud_id, asc_id, &lp, &link_id));

    /* ASC -> MTR  (feedforward, gamma) */
    fill_link_params(&lp, CUNXON_LINK_FEEDFORWARD, CUNXON_BAND_GAMMA, 1.2f, 1, 0.7f);
    CHECK(cunxonNetworkAddLink(net, asc_id, mtr_id, &lp, &link_id));

    /* MTR -> ASC  (feedback, beta) -- efference copy */
    fill_link_params(&lp, CUNXON_LINK_FEEDBACK,    CUNXON_BAND_BETA,  0.4f, 2, 0.5f);
    CHECK(cunxonNetworkAddLink(net, mtr_id, asc_id, &lp, &link_id));

    /* ASC -> VIS  (thalamic-like top-down attention, theta) */
    fill_link_params(&lp, CUNXON_LINK_THALAMIC,    CUNXON_BAND_THETA, 0.3f, 2, 0.4f);
    CHECK(cunxonNetworkAddLink(net, asc_id, vis_id, &lp, &link_id));

    /* ASC -> AUD  (thalamic-like top-down attention, theta) */
    fill_link_params(&lp, CUNXON_LINK_THALAMIC,    CUNXON_BAND_THETA, 0.3f, 2, 0.4f);
    CHECK(cunxonNetworkAddLink(net, asc_id, aud_id, &lp, &link_id));

    /* VIS <-> AUD  (lateral cross-sensory binding, theta) */
    fill_link_params(&lp, CUNXON_LINK_LATERAL,     CUNXON_BAND_THETA, 0.25f, 1, 0.35f);
    CHECK(cunxonNetworkAddLink(net, vis_id, aud_id, &lp, &link_id));
    CHECK(cunxonNetworkAddLink(net, aud_id, vis_id, &lp, &link_id));

    /* ----- finalise ------------------------------------------------------- */
    CHECK(cunxonNetworkFinalize(net));
    std::printf("  network finalised : %d spheres\n", cunxonNetworkNumSpheres(net));

    /* ===== 3. inputs ======================================================= *
     * VIS gets a "stimulus" pattern; AUD gets random noise; reward is +1 when
     * the stimulus is "salient" (first 3 channels jointly high).             */
    std::vector<float> visBuf(pVIS.num_input_neurons, 0.0f);
    std::vector<float> audBuf(pAUD.num_input_neurons, 0.0f);
    const float* ext_inputs[4] = { nullptr, nullptr, nullptr, nullptr };
    ext_inputs[vis_id] = visBuf.data();
    ext_inputs[aud_id] = audBuf.data();

    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> unif(-1.0f, 1.0f);
    std::bernoulli_distribution            coin(0.50f);
    /* Balanced classes: lazy "always X" strategy gives 0.0 expected reward,
     * so any genuine discrimination must beat 0.0.  The 30/70 variant gives
     * the network a +0.40 safe harbor at "always -1" — easy to lock into. */

    /* ===== 4. training loop ================================================ */
    std::printf("\n--- warmup (%d steps) ---\n", n_train_steps);
    double cum_reward = 0.0;
    for (int t = 0; t < n_train_steps; ++t) {
        /* build stimulus */
        bool salient = coin(rng);
        for (int i = 0; i < (int)visBuf.size(); ++i) {
            visBuf[i] = (salient && i < 3) ? 0.85f + 0.10f * unif(rng)
                                           : 0.20f * unif(rng);
        }
        for (int i = 0; i < (int)audBuf.size(); ++i)
            audBuf[i] = 0.30f * unif(rng);

        /* train step */
        CHECK(cunxonNetworkStepTrain(net, ext_inputs, /*dt_ms=*/1.0f));

        /* Read MTR's full readout population, compute majority vote.
         * Population code: a single neuron's noisy decision is replaced by
         * the sign of the sum across all 5 motor readouts.                 */
        int n_states = 0;
        CHECK(cunxonSphereGetReadout(net, mtr_id, nullptr, &n_states));
        std::vector<int8_t> readout(n_states);
        CHECK(cunxonSphereGetReadout(net, mtr_id, readout.data(), &n_states));

        int vote_sum = 0;
        for (auto v : readout) vote_sum += v;        /* range: -n_states .. +n_states */
        bool committed = (std::abs(vote_sum) >= 2);
        int decision = committed ? (vote_sum > 0 ? +1 : -1) : (unif(rng) > 0.f ? +1 : -1);

        /* Symmetric ±1 reward: lazy "always answer X" gives 0 average,
         * so to score positive the network must actually discriminate.
         *
         * Only drive plasticity (inject DA) when the network actually
         * committed to a decision (non-zero vote_sum).  Coin-flip ties
         * shouldn't shape weights — that would be plasticity noise.        */
        int target_class = salient ? +1 : -1;
        float reward = (decision == target_class) ? +1.0f : -1.0f;
        cum_reward  += reward;
        if (committed) {
            CHECK(cunxonNetworkInjectNeuromodulator(net, /*DA=*/0, reward));
        }

        if ((t + 1) % 100 == 0) {
            double e = 0.0; cunxonNetworkGetEnergy(net, &e);
            /* Probe each sphere's activity so we can see exactly where the
             * signal chain stops propagating.                              */
            auto sphere_active_frac = [&](int sid) -> float {
                int n = 0;
                cunxonSphereSnapshot(net, sid, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, &n);
                std::vector<int8_t> ss(n);
                cunxonSphereSnapshot(net, sid, nullptr, nullptr, nullptr,
                                     ss.data(), nullptr, nullptr, &n);
                int fc = 0;
                for (auto v : ss) if (v != 0) ++fc;
                return n > 0 ? (float)fc / (float)n : 0.f;
            };
            float fv = sphere_active_frac(vis_id);
            float fa = sphere_active_frac(aud_id);
            float fc = sphere_active_frac(asc_id);
            float fm = sphere_active_frac(mtr_id);
            std::printf("  step %4d  rwd=%+.3f  E=%6.0f  "
                        "VIS=%4.1f%% AUD=%4.1f%% ASC=%4.1f%% MTR=%4.1f%%\n",
                        t + 1, cum_reward / (t + 1), e,
                        100.0f * fv, 100.0f * fa, 100.0f * fc, 100.0f * fm);

            /* Detailed VIS internals at key checkpoints */
            if (t + 1 == 100 || t + 1 == 500 || t + 1 == 1500) {
                int n_vis = 0;
                cunxonSphereSnapshot(net, vis_id, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, &n_vis);
                std::vector<float>  U_v(n_vis), h_v(n_vis), st_v(n_vis), fr_v(n_vis), ast_v(n_vis);
                std::vector<int8_t> s_v(n_vis);
                cunxonSphereSnapshot(net, vis_id, U_v.data(), h_v.data(), st_v.data(),
                                     s_v.data(), fr_v.data(), ast_v.data(), &n_vis);
                int n_in_v = pVIS.num_input_neurons;
                int n_hid_v = pVIS.num_hidden_neurons;
                /* Classify firing by type */
                int fire_in=0, fire_hid=0, fire_out=0;
                float u_max_hid = 0, u_mean_hid = 0, fr_max_hid = 0;
                int hid_start = n_in_v, hid_end = n_in_v + n_hid_v;
                for (int k = 0; k < n_vis; ++k) {
                    if (s_v[k] != 0) {
                        if      (k < n_in_v)             fire_in++;
                        else if (k < hid_end)            fire_hid++;
                        else                             fire_out++;
                    }
                    if (k >= hid_start && k < hid_end) {
                        float u = fabsf(U_v[k]);
                        if (u > u_max_hid) u_max_hid = u;
                        u_mean_hid += u;
                        if (fr_v[k] > fr_max_hid) fr_max_hid = fr_v[k];
                    }
                }
                u_mean_hid /= n_hid_v;
                std::printf("     VIS detail: input_fire=%d/%d, hidden_fire=%d/%d, output_fire=%d/%d\n",
                            fire_in, n_in_v, fire_hid, n_hid_v,
                            fire_out, (int)pVIS.num_output_neurons);
                std::printf("                 VIS-hidden  |U|_max=%.3f  |U|_mean=%.3f  fr_max=%.3f\n",
                            u_max_hid, u_mean_hid, fr_max_hid);
            }
        }
    }

    /* ===== 5. continuous online testing ===================================
     * Per the Neuraxon paper: "Continuous processing enables real-time
     * adjustments WITHOUT discrete training phases."  Plasticity is an
     * intrinsic property of the model, not a training-only mechanism.
     * So this phase keeps StepTrain (plasticity active) and continues
     * injecting reward — the network keeps learning while we measure
     * accuracy on each new stimulus.  This matches the paper's continuous
     * non-stationary learning scenario.                                    */
    std::printf("\n--- online testing (%d steps, continued plasticity) ---\n",
                n_infer_steps);
    int hits = 0, total_salient = 0;
    int true_neg = 0, total_nonsalient = 0;
    /* Track per-window accuracy so we can see online learning progress */
    int window_hits = 0, window_total = 0;
    for (int t = 0; t < n_infer_steps; ++t) {
        bool salient = coin(rng);
        for (int i = 0; i < (int)visBuf.size(); ++i) {
            visBuf[i] = (salient && i < 3) ? 0.85f + 0.10f * unif(rng)
                                           : 0.20f * unif(rng);
        }
        for (int i = 0; i < (int)audBuf.size(); ++i)
            audBuf[i] = 0.30f * unif(rng);

        /* Continued learning: StepTrain, not StepInfer.  The Neuraxon model
         * doesn't separate train and inference — plasticity is always on.  */
        CHECK(cunxonNetworkStepTrain(net, ext_inputs, 1.0f));

        int n_states = 0;
        cunxonSphereGetReadout(net, mtr_id, nullptr, &n_states);
        std::vector<int8_t> readout(n_states);
        cunxonSphereGetReadout(net, mtr_id, readout.data(), &n_states);

        /* Same population vote as training */
        int vote_sum = 0;
        for (auto v : readout) vote_sum += v;
        bool committed = (std::abs(vote_sum) >= 2);
        int decision = committed ? (vote_sum > 0 ? +1 : -1) : (unif(rng) > 0.f ? +1 : -1);

        /* Continued reward injection — supports online learning.
         * Skip on tied votes for the same reason as training.              */
        int target_class = salient ? +1 : -1;
        float reward = (decision == target_class) ? +1.0f : -1.0f;
        if (committed) {
            CHECK(cunxonNetworkInjectNeuromodulator(net, /*DA=*/0, reward));
        }

        if (salient) {
            total_salient++;
            if (decision == +1) { hits++; window_hits++; }
        } else {
            total_nonsalient++;
            if (decision == -1) { true_neg++; window_hits++; }
        }
        window_total++;

        if ((t + 1) % 100 == 0) {
            std::printf("  step %4d   window_accuracy=%.1f%%\n",
                        t + 1, 100.0 * window_hits / fmaxf(1.f, (float)window_total));
            window_hits = 0;
            window_total = 0;
        }
    }
    std::printf("\nTesting period summary:\n");
    std::printf("  hit-rate  (salient detection) : %d / %d  (%.1f%%)\n",
                hits, total_salient,
                total_salient ? 100.0 * hits / total_salient : 0.0);
    std::printf("  reject-rate (non-salient)     : %d / %d  (%.1f%%)\n",
                true_neg, total_nonsalient,
                total_nonsalient ? 100.0 * true_neg / total_nonsalient : 0.0);
    double overall_acc = (hits + true_neg) / (double)n_infer_steps;
    std::printf("  overall accuracy              : %.1f%%  (chance = 50.0%%)\n",
                100.0 * overall_acc);

    /* ===== 6. snapshot one sphere ========================================== */
    int n_total = pASC.num_input_neurons + pASC.num_hidden_neurons
                + pASC.num_output_neurons;
    std::vector<float>  U(n_total), h(n_total), st(n_total), fr(n_total), ast(n_total);
    std::vector<int8_t> s(n_total);
    cunxonSphereSnapshot(net, asc_id,
                         U.data(), h.data(), st.data(), s.data(),
                         fr.data(), ast.data(), &n_total);
    int n_pos = 0, n_neg = 0, n_zero = 0;
    for (auto v : s) (v > 0 ? ++n_pos : v < 0 ? ++n_neg : ++n_zero);
    std::printf("\nASC trinary distribution: +1=%d  0=%d  -1=%d  (of %d)\n",
                n_pos, n_zero, n_neg, n_total);

    /* ===== 7. save & clean up ============================================== */
    CHECK(cunxonNetworkSave(net, "demo_4sphere.cunxon"));
    std::printf("saved network -> demo_4sphere.cunxon\n");

    cunxonNetworkDestroy(net);
    cunxonDestroyContext(ctx);
    std::printf("done.\n");
    return 0;
}
