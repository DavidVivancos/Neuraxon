/* ============================================================================
 *  example_aigarth.cu
 *  ---------------------------------------------------------------------------
 *  Demonstrates cuNxon's evolutionary plasticity (Aigarth Intelligent Tissue).
 *
 *  Task: a small sphere must learn to distinguish two patterns by EVOLUTION
 *  alone, without any gradient or STDP-based weight updates.  We define a
 *  fitness function over the network's trinary readout and let
 *  cunxonNetworkAigarthStep mutate weights across a small population, keeping
 *  the best candidate each generation.
 *
 *  Compile (via CMake):  make example_aigarth
 *  Run:                  ./example_aigarth [n_generations]
 * ========================================================================== */

#include <cuNxon.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#define CHECK(call) do {                                                       \
    cunxonStatus_t _s = (call);                                                \
    if (_s != CUNXON_OK) {                                                     \
        std::fprintf(stderr, "[cuNxon] %s -> %s (%s) at %s:%d\n",              \
                     #call, cunxonGetStatusString(_s),                         \
                     cunxonGetLastError(), __FILE__, __LINE__);                \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

/* ============================================================================
 *  Fitness: classify two input patterns
 *
 *      pattern A (positive class): inputs ~ +0.8 on [0..3]
 *      pattern B (negative class): inputs ~ -0.8 on [0..3]
 *
 *  For each pattern we run a short eval (drives the network, settles dynamics),
 *  read the trinary output, and score how well the average matches the target
 *  (+1 for A, -1 for B).
 * ========================================================================== */
struct EvalCtx {
    int sphere_id;
    int n_inputs;
};

static float run_and_mean(cunxonNetwork_t net, int sphere_id, int n_inputs,
                          float input_val, int n_steps)
{
    std::vector<float> in(n_inputs, input_val);
    const float* exti[16] = { nullptr };
    /* assume sphere_id < 16 in this demo */
    exti[sphere_id] = in.data();
    /* dummy slots for other spheres (none here) */
    cunxonNetworkReset(net);
    for (int t = 0; t < n_steps; ++t)
        cunxonNetworkStepInfer(net, exti, 1.0f);

    int n_out = 0;
    cunxonSphereGetReadout(net, sphere_id, nullptr, &n_out);
    if (n_out <= 0) return 0.f;
    std::vector<int8_t> rd(n_out);
    cunxonSphereGetReadout(net, sphere_id, rd.data(), &n_out);
    double sum = 0;
    for (auto v : rd) sum += v;
    return (float)(sum / n_out);
}

static float fitness_classify(cunxonNetwork_t net, void* user)
{
    EvalCtx* ec = (EvalCtx*)user;
    float mean_pos = run_and_mean(net, ec->sphere_id, ec->n_inputs, +0.85f, 25);
    float mean_neg = run_and_mean(net, ec->sphere_id, ec->n_inputs, -0.85f, 25);
    /* margin: how far the two classes separate (range [-2, +2]) */
    return mean_pos - mean_neg;
}


/* ============================================================================
 *  main
 * ========================================================================== */
int main(int argc, char** argv)
{
    int n_generations = (argc > 1) ? std::atoi(argv[1]) : 10;
    int pop_per_gen   = (argc > 2) ? std::atoi(argv[2]) : 16;

    std::printf("=== cuNxon Aigarth evolution demo ===\n");
    std::printf("  generations    : %d\n", n_generations);
    std::printf("  pop per gen    : %d\n", pop_per_gen);

    cunxonContext_t ctx;
    CHECK(cunxonCreateContext(&ctx, 0, /*seed=*/123, 0));

    cunxonNetwork_t net;
    CHECK(cunxonNetworkCreate(ctx, &net, "aigarth_demo"));

    cunxonNetworkParameters_t p;
    cunxonGetDefaultParameters(&p);
    p.num_input_neurons  = 4;
    p.num_hidden_neurons = 32;
    p.num_output_neurons = 8;
    p.random_seed_offset = 1;

    int sid = -1;
    CHECK(cunxonNetworkAddSphere(net, "evolve", CUNXON_SPHERE_SENSORY, &p, &sid));

    std::vector<int> ports = {0,1,2,3};
    std::vector<int> readout_ids(p.num_output_neurons);
    for (int i = 0; i < p.num_output_neurons; ++i) readout_ids[i] = i;
    CHECK(cunxonNetworkSetSphereInterface(net, sid,
                                          ports.data(), 4,
                                          nullptr, 0,
                                          nullptr, 0,
                                          readout_ids.data(),
                                          (int)readout_ids.size()));
    CHECK(cunxonNetworkFinalize(net));

    EvalCtx ec{ sid, p.num_input_neurons };

    /* baseline */
    float baseline = fitness_classify(net, &ec);
    std::printf("\nbaseline classification margin: %+.4f\n", baseline);

    /* Run Aigarth generations.  We taper mutation rates: explore widely at
     * first, then refine.                                                    */
    for (int g = 0; g < n_generations; ++g) {
        float frac = (float)g / std::max(1, n_generations - 1);
        float mf = 0.15f * (1.0f - 0.6f * frac);
        float ms = 0.07f * (1.0f - 0.6f * frac);
        float mm = 0.03f * (1.0f - 0.6f * frac);
        CHECK(cunxonNetworkAigarthConfig(net, pop_per_gen, mf, ms, mm));
        CHECK(cunxonNetworkAigarthStep(net, fitness_classify, &ec));
        float score = fitness_classify(net, &ec);
        std::printf("  gen %3d   mut=(%.3f,%.3f,%.3f)   margin=%+.4f\n",
                    g + 1, mf, ms, mm, score);
    }

    float final_score = fitness_classify(net, &ec);
    std::printf("\nfinal classification margin: %+.4f   (improvement: %+.4f)\n",
                final_score, final_score - baseline);

    CHECK(cunxonNetworkDestroy(net));
    CHECK(cunxonDestroyContext(ctx));
    std::printf("done.\n");
    return 0;
}
