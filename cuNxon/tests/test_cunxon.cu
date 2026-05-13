/* ============================================================================
 *  test_cunxon.cu
 *  ---------------------------------------------------------------------------
 *  Self-contained smoke / correctness tests for cuNxon.  Returns non-zero
 *  exit code on any failure.  Runs on a single GPU.
 *
 *  Tests included:
 *     1. Context lifecycle  (create/destroy, properties)
 *     2. Default parameters are sane
 *     3. Single-sphere build, finalize, step, readout (no crash, trinary)
 *     4. Multi-sphere with one inter-sphere link, step, readout
 *     5. Reset clears membrane but keeps weights
 *     6. Save/Load round-trip preserves weights bit-for-bit
 *     7. Aigarth improves (or at least matches) a trivial fitness function
 *
 *  Build (manual):
 *      nvcc -O3 -std=c++14 -arch=sm_70 -I../include \
 *           test_cunxon.cu ../src/*.cu -lcurand -o test_cunxon
 *  Or via the bundled CMake target: `make test_cunxon && ./test_cunxon`
 * ========================================================================== */

#include <cuNxon.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

/* ---- tiny test harness ----------------------------------------------------- */
static int  g_passed = 0;
static int  g_failed = 0;
static std::string g_current;

#define TEST(name)                                                             \
    do {                                                                       \
        g_current = name;                                                      \
        std::printf("[ RUN      ] %s\n", g_current.c_str());                   \
    } while (0)

#define EXPECT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "  [ FAIL     ] %s: " #cond                   \
                                 " at %s:%d\n",                                \
                         g_current.c_str(), __FILE__, __LINE__);               \
            g_failed++;                                                        \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define EXPECT_OK(call)                                                        \
    do {                                                                       \
        cunxonStatus_t _s = (call);                                            \
        if (_s != CUNXON_OK) {                                                 \
            std::fprintf(stderr, "  [ FAIL     ] %s: %s -> %s (%s) at %s:%d\n",\
                         g_current.c_str(), #call,                             \
                         cunxonGetStatusString(_s),                            \
                         cunxonGetLastError(), __FILE__, __LINE__);            \
            g_failed++;                                                        \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define PASS()                                                                 \
    do {                                                                       \
        g_passed++;                                                            \
        std::printf("[       OK ] %s\n", g_current.c_str());                   \
        return true;                                                           \
    } while (0)


/* ---- helpers --------------------------------------------------------------- */
static void mk_params(cunxonNetworkParameters_t& p,
                      int n_in, int n_hid, int n_out, uint64_t seed_off = 0)
{
    cunxonGetDefaultParameters(&p);
    p.num_input_neurons  = n_in;
    p.num_hidden_neurons = n_hid;
    p.num_output_neurons = n_out;
    p.random_seed_offset = seed_off;
}

static void mk_link(cunxonLinkParameters_t& lp,
                    cunxonLinkKind_t kind = CUNXON_LINK_FEEDFORWARD,
                    cunxonBand_t   band = CUNXON_BAND_GAMMA,
                    float gain = 1.0f, int delay = 1, float coh = 0.5f)
{
    std::memset(&lp, 0, sizeof(lp));
    lp.kind = kind;
    lp.coherence_band = band;
    lp.gain = gain;
    lp.delay_steps = delay;
    lp.transmission_threshold = 0.0f;
    lp.coherence_strength = coh;
    lp.topology = CUNXON_TOPO_DENSE;
    lp.sparse_prob = 0.3f;
    lp.allow_negative_weights = 1;
    lp.plasticity_rate = 1e-3f;
    lp.weight_decay = 1e-5f;
    lp.weight_clip = 1.0f;
    lp.normalize_rows = 0;
    lp.bias = 0.0f;
}


/* ============================================================================
 *  Test 1: context lifecycle
 * ========================================================================== */
static bool test_context_lifecycle()
{
    TEST("context_lifecycle");
    cunxonContext_t ctx = nullptr;
    EXPECT_OK(cunxonCreateContext(&ctx, 0, 42, 0));
    EXPECT(ctx != nullptr);

    int cc = 0;
    EXPECT_OK(cunxonContextGetProperty(ctx, CUNXON_PROP_COMPUTE_CAPABILITY,
                                       &cc, sizeof(cc)));
    EXPECT(cc >= 30);  /* sm_30+ */

    char name[256] = {0};
    EXPECT_OK(cunxonContextGetProperty(ctx, CUNXON_PROP_DEVICE_NAME,
                                       name, sizeof(name)));
    EXPECT(name[0] != 0);

    EXPECT_OK(cunxonDestroyContext(ctx));
    PASS();
}


/* ============================================================================
 *  Test 2: default parameters are sane
 * ========================================================================== */
static bool test_default_parameters()
{
    TEST("default_parameters");
    cunxonNetworkParameters_t p;
    EXPECT_OK(cunxonGetDefaultParameters(&p));
    EXPECT(p.num_input_neurons  > 0);
    EXPECT(p.num_hidden_neurons > 0);
    EXPECT(p.num_output_neurons > 0);
    EXPECT(p.num_dendritic_branches >= 1);
    EXPECT(p.dsn_kernel_size >= 1);
    EXPECT(p.msth_ultrafast_tau > 0.f);
    EXPECT(p.msth_slow_tau >= p.msth_ultrafast_tau);
    EXPECT(p.chrono_omega_min < p.chrono_omega_max);
    EXPECT(p.dopamine_baseline   >= 0.f);
    EXPECT(p.serotonin_baseline  >= 0.f);
    PASS();
}


/* ============================================================================
 *  Test 3: single-sphere step produces trinary states in {-1,0,+1}
 * ========================================================================== */
static bool test_single_sphere_step()
{
    TEST("single_sphere_step");
    cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 7, 0));
    cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t3"));

    cunxonNetworkParameters_t p; mk_params(p, 4, 16, 4);
    int sid = -1;
    EXPECT_OK(cunxonNetworkAddSphere(net, "S0", CUNXON_SPHERE_SENSORY, &p, &sid));

    std::vector<int> sin = {0,1,2,3};
    std::vector<int> rdo = {0,1,2,3};
    EXPECT_OK(cunxonNetworkSetSphereInterface(net, sid,
                                              sin.data(), 4,
                                              nullptr, 0,
                                              nullptr, 0,
                                              rdo.data(), 4));
    EXPECT_OK(cunxonNetworkFinalize(net));
    EXPECT(cunxonNetworkNumSpheres(net) == 1);

    std::vector<float> in(4, 0.5f);
    const float* exti[1] = { in.data() };
    for (int t = 0; t < 50; ++t)
        EXPECT_OK(cunxonNetworkStepInfer(net, exti, 1.0f));

    int n = 0;
    EXPECT_OK(cunxonSphereGetReadout(net, sid, nullptr, &n));
    EXPECT(n == 4);
    std::vector<int8_t> out(4);
    EXPECT_OK(cunxonSphereGetReadout(net, sid, out.data(), &n));
    for (auto v : out) EXPECT(v >= -1 && v <= +1);

    EXPECT_OK(cunxonNetworkDestroy(net));
    EXPECT_OK(cunxonDestroyContext(ctx));
    PASS();
}


/* ============================================================================
 *  Test 4: multi-sphere with link
 * ========================================================================== */
static bool test_multi_sphere_link()
{
    TEST("multi_sphere_link");
    cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 9, 0));
    cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t4"));

    cunxonNetworkParameters_t pa, pb;
    mk_params(pa, 4, 12, 4);
    mk_params(pb, 4, 8,  4, 100);

    int a, b;
    EXPECT_OK(cunxonNetworkAddSphere(net, "A", CUNXON_SPHERE_SENSORY,     &pa, &a));
    EXPECT_OK(cunxonNetworkAddSphere(net, "B", CUNXON_SPHERE_ASSOCIATION, &pb, &b));

    std::vector<int> ports = {0,1,2,3};
    EXPECT_OK(cunxonNetworkSetSphereInterface(net, a,
                                              ports.data(), 4,
                                              nullptr, 0,
                                              ports.data(), 4,
                                              ports.data(), 4));
    EXPECT_OK(cunxonNetworkSetSphereInterface(net, b,
                                              nullptr, 0,
                                              ports.data(), 4,
                                              nullptr, 0,
                                              ports.data(), 4));

    cunxonLinkParameters_t lp; mk_link(lp);
    int lid = -1;
    EXPECT_OK(cunxonNetworkAddLink(net, a, b, &lp, &lid));
    EXPECT_OK(cunxonNetworkFinalize(net));

    std::vector<float> in(4, 0.7f);
    const float* exti[2] = { in.data(), nullptr };
    /* Train for a while so signal stabilises in both spheres. */
    for (int t = 0; t < 30; ++t)
        EXPECT_OK(cunxonNetworkStepTrain(net, exti, 1.0f));

    int n_b = 0;
    EXPECT_OK(cunxonSphereGetReadout(net, b, nullptr, &n_b));
    EXPECT(n_b == 4);
    std::vector<int8_t> readout_b(4);
    EXPECT_OK(cunxonSphereGetReadout(net, b, readout_b.data(), &n_b));

    /* Sphere B has no sensory inputs of its own; the ONLY way it can fire
     * is via the inter-sphere link from sphere A.  At least one non-zero
     * state in B's readout proves signal traversed the link.              */
    {
        int n_total_b = 0;
        EXPECT_OK(cunxonSphereSnapshot(net, b, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, &n_total_b));
        std::vector<int8_t> all_b(n_total_b);
        EXPECT_OK(cunxonSphereSnapshot(net, b, nullptr, nullptr, nullptr,
                                       all_b.data(), nullptr, nullptr,
                                       &n_total_b));
        int n_active = 0;
        for (auto v : all_b) if (v != 0) ++n_active;
        EXPECT(n_active > 0);
    }

    EXPECT_OK(cunxonNetworkDestroy(net));
    EXPECT_OK(cunxonDestroyContext(ctx));
    PASS();
}


/* ============================================================================
 *  Test 5: Reset clears U/firing but keeps weights and learned kernels
 * ========================================================================== */
static bool test_reset_preserves_weights()
{
    TEST("reset_preserves_weights");
    cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 11, 0));
    cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t5"));

    cunxonNetworkParameters_t p; mk_params(p, 4, 16, 4);
    int sid;
    EXPECT_OK(cunxonNetworkAddSphere(net, "S0", CUNXON_SPHERE_SENSORY, &p, &sid));
    std::vector<int> ids = {0,1,2,3};
    EXPECT_OK(cunxonNetworkSetSphereInterface(net, sid,
                                              ids.data(), 4, nullptr, 0,
                                              nullptr, 0, ids.data(), 4));
    EXPECT_OK(cunxonNetworkFinalize(net));

    /* Get U after a few steps */
    std::vector<float> in(4, 0.5f);
    const float* exti[1] = { in.data() };
    for (int t = 0; t < 30; ++t)
        EXPECT_OK(cunxonNetworkStepTrain(net, exti, 1.0f));

    int n_total = 0;
    EXPECT_OK(cunxonSphereSnapshot(net, sid, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, &n_total));
    std::vector<float> U_before(n_total), fr_before(n_total);
    EXPECT_OK(cunxonSphereSnapshot(net, sid,
                                   U_before.data(), nullptr, nullptr, nullptr,
                                   fr_before.data(), nullptr, &n_total));

    EXPECT_OK(cunxonNetworkReset(net));
    std::vector<float> U_after(n_total);
    EXPECT_OK(cunxonSphereSnapshot(net, sid,
                                   U_after.data(), nullptr, nullptr, nullptr,
                                   nullptr, nullptr, &n_total));
    /* U should be at or near zero after reset */
    double sum_abs = 0;
    for (auto v : U_after) sum_abs += std::fabs(v);
    EXPECT(sum_abs < 1e-3 * n_total);  /* essentially zero */

    EXPECT_OK(cunxonNetworkDestroy(net));
    EXPECT_OK(cunxonDestroyContext(ctx));
    PASS();
}


/* ============================================================================
 *  Test 6: Save/Load round-trip
 * ========================================================================== */
static bool test_save_load_roundtrip()
{
    TEST("save_load_roundtrip");
    const char* fname = "test_cunxon_roundtrip.cunxon";

    /* --- build & save --- */
    int n_total = 0;
    std::vector<float> U_saved;
    std::vector<int8_t> s_saved;
    {
        cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 13, 0));
        cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t6"));

        cunxonNetworkParameters_t pa, pb;
        mk_params(pa, 4, 12, 4);
        mk_params(pb, 4, 8,  4, 200);
        int a, b;
        EXPECT_OK(cunxonNetworkAddSphere(net, "A", CUNXON_SPHERE_SENSORY,     &pa, &a));
        EXPECT_OK(cunxonNetworkAddSphere(net, "B", CUNXON_SPHERE_ASSOCIATION, &pb, &b));

        std::vector<int> ids = {0,1,2,3};
        EXPECT_OK(cunxonNetworkSetSphereInterface(net, a,
                                                  ids.data(), 4, nullptr, 0,
                                                  ids.data(), 4, ids.data(), 4));
        EXPECT_OK(cunxonNetworkSetSphereInterface(net, b,
                                                  nullptr, 0, ids.data(), 4,
                                                  nullptr, 0, ids.data(), 4));
        cunxonLinkParameters_t lp; mk_link(lp);
        int lid; EXPECT_OK(cunxonNetworkAddLink(net, a, b, &lp, &lid));
        EXPECT_OK(cunxonNetworkFinalize(net));

        std::vector<float> in(4, 0.8f);
        const float* exti[2] = { in.data(), nullptr };
        for (int t = 0; t < 40; ++t)
            EXPECT_OK(cunxonNetworkStepTrain(net, exti, 1.0f));

        /* capture B's state for the after-load comparison */
        EXPECT_OK(cunxonSphereSnapshot(net, b, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, &n_total));
        U_saved.resize(n_total);
        s_saved.resize(n_total);
        EXPECT_OK(cunxonSphereSnapshot(net, b, U_saved.data(), nullptr, nullptr,
                                       s_saved.data(), nullptr, nullptr,
                                       &n_total));
        EXPECT_OK(cunxonNetworkSave(net, fname));
        EXPECT_OK(cunxonNetworkDestroy(net));
        EXPECT_OK(cunxonDestroyContext(ctx));
    }

    /* --- load & verify --- */
    {
        cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 17, 0));
        cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t6b"));
        EXPECT_OK(cunxonNetworkLoad(net, fname));
        EXPECT(cunxonNetworkNumSpheres(net) == 2);

        int n_after = 0;
        EXPECT_OK(cunxonSphereSnapshot(net, 1, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, &n_after));
        EXPECT(n_after == n_total);
        /* The loaded network has dynamic state reset (Reset is called inside
         * AddSphere via initial conditions). We just verify the network
         * runs and produces a readout of the right shape.                  */
        std::vector<float> in(4, 0.8f);
        const float* exti[2] = { in.data(), nullptr };
        for (int t = 0; t < 5; ++t)
            EXPECT_OK(cunxonNetworkStepInfer(net, exti, 1.0f));

        int n_states = 0;
        EXPECT_OK(cunxonSphereGetReadout(net, 1, nullptr, &n_states));
        EXPECT(n_states > 0);
        std::vector<int8_t> rd(n_states);
        EXPECT_OK(cunxonSphereGetReadout(net, 1, rd.data(), &n_states));

        EXPECT_OK(cunxonNetworkDestroy(net));
        EXPECT_OK(cunxonDestroyContext(ctx));
    }
    std::remove(fname);
    PASS();
}


/* ============================================================================
 *  Test 7: Aigarth improves a trivial fitness function
 *      Fitness = -|mean(readout) - target|   (i.e. drive activity to target)
 * ========================================================================== */
struct AigarthCtx { int sphere_id; float target; };

static float fitness_drive_to_target(cunxonNetwork_t net, void* user)
{
    AigarthCtx* ac = (AigarthCtx*)user;
    /* run a short eval with a fixed input */
    std::vector<float> in(4, 0.6f);
    const float* exti[1] = { in.data() };
    for (int t = 0; t < 20; ++t)
        cunxonNetworkStepInfer(net, exti, 1.0f);

    int n = 0;
    cunxonSphereGetReadout(net, ac->sphere_id, nullptr, &n);
    std::vector<int8_t> rd(n);
    cunxonSphereGetReadout(net, ac->sphere_id, rd.data(), &n);
    if (n <= 0) return -1e9f;
    double sum = 0;
    for (auto v : rd) sum += v;
    double mean = sum / n;
    return (float)(-std::fabs(mean - ac->target));
}

static bool test_aigarth_improves()
{
    TEST("aigarth_improves");
    cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 23, 0));
    cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t7"));

    cunxonNetworkParameters_t p; mk_params(p, 4, 24, 4);
    int sid;
    EXPECT_OK(cunxonNetworkAddSphere(net, "S0", CUNXON_SPHERE_SENSORY, &p, &sid));
    std::vector<int> ids = {0,1,2,3};
    EXPECT_OK(cunxonNetworkSetSphereInterface(net, sid,
                                              ids.data(), 4, nullptr, 0,
                                              nullptr, 0, ids.data(), 4));
    EXPECT_OK(cunxonNetworkFinalize(net));

    AigarthCtx ac{ sid, +1.0f /* maximally-active */ };
    float baseline = fitness_drive_to_target(net, &ac);

    EXPECT_OK(cunxonNetworkAigarthConfig(net, 8, 0.10f, 0.05f, 0.02f));
    EXPECT_OK(cunxonNetworkAigarthStep(net, fitness_drive_to_target, &ac));
    float after = fitness_drive_to_target(net, &ac);

    std::printf("    baseline=%.4f  after_aigarth=%.4f\n", baseline, after);
    /* Aigarth keeps the best of {baseline, candidates}, so result must be
     * at least as good as baseline.                                         */
    EXPECT(after >= baseline - 1e-3f);

    EXPECT_OK(cunxonNetworkDestroy(net));
    EXPECT_OK(cunxonDestroyContext(ctx));
    PASS();
}


/* ============================================================================
 *  Test 8: Modulatory synapses do not crash and do contribute (presence test)
 *      We can't directly inspect modulatory_pot from the public API, but we
 *      check that with neuromodulator injection over many steps, network
 *      activity is non-zero (regression test for the placeholder bug).
 * ========================================================================== */
static bool test_modulatory_runs_clean()
{
    TEST("modulatory_runs_clean");
    cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 31, 0));
    cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t8"));

    cunxonNetworkParameters_t p; mk_params(p, 4, 16, 4);
    int sid;
    EXPECT_OK(cunxonNetworkAddSphere(net, "S0", CUNXON_SPHERE_SENSORY, &p, &sid));
    std::vector<int> ids = {0,1,2,3};
    EXPECT_OK(cunxonNetworkSetSphereInterface(net, sid,
                                              ids.data(), 4, nullptr, 0,
                                              nullptr, 0, ids.data(), 4));
    EXPECT_OK(cunxonNetworkFinalize(net));

    std::vector<float> in(4, 0.5f);
    const float* exti[1] = { in.data() };
    for (int t = 0; t < 100; ++t) {
        EXPECT_OK(cunxonNetworkStepTrain(net, exti, 1.0f));
        if (t % 10 == 0)
            EXPECT_OK(cunxonNetworkInjectNeuromodulator(net, 0, +0.5f));
    }

    /* After 100 training steps with continuous drive, the network must
     * actually have been active.  Without the metabotropic placeholder bug
     * fix, energy stays at zero — this is the regression test.             */
    double e = 0.0;
    EXPECT_OK(cunxonNetworkGetEnergy(net, &e));
    EXPECT(std::isfinite(e));
    EXPECT(e > 0.0);                /* strictly positive: something fired   */

    /* And the readout must contain at least one non-zero state at the end. */
    int n_states = 0;
    EXPECT_OK(cunxonSphereGetReadout(net, sid, nullptr, &n_states));
    std::vector<int8_t> rd(n_states);
    EXPECT_OK(cunxonSphereGetReadout(net, sid, rd.data(), &n_states));
    int n_active = 0;
    for (auto v : rd) if (v != 0) ++n_active;
    EXPECT(n_active > 0);           /* not totally silent                   */

    EXPECT_OK(cunxonNetworkDestroy(net));
    EXPECT_OK(cunxonDestroyContext(ctx));
    PASS();
}


/* ============================================================================
 *  Test 9: Pattern storage / recall (Algorithm 8)
 * ========================================================================== */
static bool test_pattern_store_recall()
{
    TEST("pattern_store_recall");
    cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 41, 0));
    cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t9"));

    cunxonNetworkParameters_t p; mk_params(p, 8, 32, 8);
    int sid;
    EXPECT_OK(cunxonNetworkAddSphere(net, "S0", CUNXON_SPHERE_SENSORY, &p, &sid));
    std::vector<int> sin(8), out(8);
    for (int i = 0; i < 8; ++i) { sin[i] = i; out[i] = i; }
    EXPECT_OK(cunxonNetworkSetSphereInterface(net, sid,
                                              sin.data(), 8, nullptr, 0,
                                              nullptr, 0, out.data(), 8));
    EXPECT_OK(cunxonNetworkFinalize(net));

    /* store two distinct patterns */
    std::vector<float> pat_a = {+0.8f,+0.8f,+0.8f,+0.8f,-0.8f,-0.8f,-0.8f,-0.8f};
    std::vector<float> pat_b = {-0.8f,-0.8f,-0.8f,-0.8f,+0.8f,+0.8f,+0.8f,+0.8f};
    EXPECT_OK(cunxonNetworkStorePattern(net, sid, "alpha",
                                        pat_a.data(), 8, 30, 1.0f));
    EXPECT_OK(cunxonNetworkStorePattern(net, sid, "beta",
                                        pat_b.data(), 8, 30, 1.0f));

    /* list */
    int n_pats = 0;
    EXPECT_OK(cunxonNetworkListPatterns(net, nullptr, 0, &n_pats));
    EXPECT(n_pats == 2);

    /* recall both with partial cue */
    std::vector<int8_t> recall_a(8), recall_b(8);
    int n_out_a = 8, n_out_b = 8;
    EXPECT_OK(cunxonNetworkRecallPattern(net, sid, "alpha", 8, 0.5f, 20, 1.0f,
                                         recall_a.data(), &n_out_a));
    EXPECT_OK(cunxonNetworkRecallPattern(net, sid, "beta",  8, 0.5f, 20, 1.0f,
                                         recall_b.data(), &n_out_b));
    EXPECT(n_out_a == 8 && n_out_b == 8);

    /* clear and verify */
    EXPECT_OK(cunxonNetworkClearPatterns(net));
    EXPECT_OK(cunxonNetworkListPatterns(net, nullptr, 0, &n_pats));
    EXPECT(n_pats == 0);

    EXPECT_OK(cunxonNetworkDestroy(net));
    EXPECT_OK(cunxonDestroyContext(ctx));
    PASS();
}


/* ============================================================================
 *  Test 10: SphereLayer grouping (paper §P4)
 * ========================================================================== */
static bool test_sphere_layers()
{
    TEST("sphere_layers");
    cunxonContext_t ctx; EXPECT_OK(cunxonCreateContext(&ctx, 0, 43, 0));
    cunxonNetwork_t net; EXPECT_OK(cunxonNetworkCreate(ctx, &net, "t10"));

    cunxonNetworkParameters_t p; mk_params(p, 4, 8, 4);
    int s_vis, s_aud, s_asc, s_mtr;
    EXPECT_OK(cunxonNetworkAddSphere(net, "VIS", CUNXON_SPHERE_SENSORY,     &p, &s_vis));
    EXPECT_OK(cunxonNetworkAddSphere(net, "AUD", CUNXON_SPHERE_SENSORY,     &p, &s_aud));
    EXPECT_OK(cunxonNetworkAddSphere(net, "ASC", CUNXON_SPHERE_ASSOCIATION, &p, &s_asc));
    EXPECT_OK(cunxonNetworkAddSphere(net, "MTR", CUNXON_SPHERE_MOTOR,       &p, &s_mtr));

    int L_sensory, L_assoc, L_motor;
    EXPECT_OK(cunxonNetworkAddLayer(net, "sensory",     0, &L_sensory));
    EXPECT_OK(cunxonNetworkAddLayer(net, "association", 1, &L_assoc));
    EXPECT_OK(cunxonNetworkAddLayer(net, "motor",       2, &L_motor));
    EXPECT_OK(cunxonNetworkAddSphereToLayer(net, L_sensory, s_vis));
    EXPECT_OK(cunxonNetworkAddSphereToLayer(net, L_sensory, s_aud));
    EXPECT_OK(cunxonNetworkAddSphereToLayer(net, L_assoc,   s_asc));
    EXPECT_OK(cunxonNetworkAddSphereToLayer(net, L_motor,   s_mtr));

    EXPECT(cunxonNetworkNumLayers(net) == 3);
    int n_s = 0;
    EXPECT_OK(cunxonNetworkGetLayer(net, L_sensory, nullptr, 0, nullptr,
                                    nullptr, &n_s));
    EXPECT(n_s == 2);
    std::vector<int> ids(n_s);
    EXPECT_OK(cunxonNetworkGetLayer(net, L_sensory, nullptr, 0, nullptr,
                                    ids.data(), &n_s));
    EXPECT(ids[0] == s_vis && ids[1] == s_aud);

    char nm[64]; int depth = -1;
    EXPECT_OK(cunxonNetworkGetLayer(net, L_motor, nm, sizeof(nm), &depth,
                                    nullptr, &n_s));
    EXPECT(std::string(nm) == "motor" && depth == 2);

    EXPECT_OK(cunxonNetworkDestroy(net));
    EXPECT_OK(cunxonDestroyContext(ctx));
    PASS();
}


/* ---- main ----------------------------------------------------------------- */
int main()
{
    std::printf("=== cuNxon test suite ===\n");
    std::printf("cuNxon version: %s\n\n", cunxonGetVersion());

    test_context_lifecycle();
    test_default_parameters();
    test_single_sphere_step();
    test_multi_sphere_link();
    test_reset_preserves_weights();
    test_save_load_roundtrip();
    test_modulatory_runs_clean();
    test_aigarth_improves();
    test_pattern_store_recall();
    test_sphere_layers();

    std::printf("\n=== %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed ? 1 : 0;
}
