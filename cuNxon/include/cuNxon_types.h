/* =============================================================================
 *  cuNxon_types.h  -  Types, enums, and parameter structs for cuNxon
 * ============================================================================= */
#ifndef CUNXON_TYPES_H
#define CUNXON_TYPES_H

#include <stdint.h>

/* ---- DLL export / import decoration ---------------------------------------- */
/* On Windows, every public symbol must be marked dllexport when building the
 * DLL and dllimport when used by a consumer.  Static builds and non-Windows
 * platforms get nothing.                                                       */
#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(CUNXON_BUILDING_DLL) || defined(cunxon_EXPORTS)
    #define CUNXON_API __declspec(dllexport)
  #elif defined(CUNXON_STATIC)
    #define CUNXON_API
  #else
    #define CUNXON_API __declspec(dllimport)
  #endif
#else
  #if __GNUC__ >= 4
    #define CUNXON_API __attribute__((visibility("default")))
  #else
    #define CUNXON_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opaque handles -------------------------------------------------------- */
typedef struct cunxonContextImpl_*  cunxonContext_t;
typedef struct cunxonNetworkImpl_*  cunxonNetwork_t;

/* ---- Status codes ---------------------------------------------------------- */
typedef enum {
    CUNXON_OK                       = 0,
    CUNXON_ERR_INVALID_ARGUMENT     = 1,
    CUNXON_ERR_OUT_OF_MEMORY        = 2,
    CUNXON_ERR_NOT_FINALIZED        = 3,
    CUNXON_ERR_ALREADY_FINALIZED    = 4,
    CUNXON_ERR_INVALID_SPHERE       = 5,
    CUNXON_ERR_INVALID_LINK         = 6,
    CUNXON_ERR_CUDA                 = 7,
    CUNXON_ERR_FILE_IO              = 8,
    CUNXON_ERR_INCOMPATIBLE_SAVE    = 9,
    CUNXON_ERR_NOT_IMPLEMENTED      = 10,
    CUNXON_ERR_UNKNOWN              = 99
} cunxonStatus_t;

/* ---- Context property queries --------------------------------------------- */
typedef enum {
    CUNXON_PROP_DEVICE_ID           = 0,  /* int */
    CUNXON_PROP_DEVICE_NAME         = 1,  /* char[256] */
    CUNXON_PROP_COMPUTE_CAPABILITY  = 2,  /* int (e.g. 86 for 8.6) */
    CUNXON_PROP_TOTAL_GLOBAL_MEM    = 3,  /* size_t */
    CUNXON_PROP_FREE_GLOBAL_MEM     = 4,  /* size_t */
    CUNXON_PROP_MAX_THREADS_PER_BLK = 5,  /* int */
    CUNXON_PROP_SM_COUNT            = 6   /* int */
} cunxonProperty_t;

/* ---- Sphere kind ----------------------------------------------------------- */
typedef enum {
    CUNXON_SPHERE_SENSORY     = 0,
    CUNXON_SPHERE_ASSOCIATION = 1,
    CUNXON_SPHERE_MOTOR       = 2,
    CUNXON_SPHERE_THALAMIC    = 3,
    CUNXON_SPHERE_CUSTOM      = 4
} cunxonSphereKind_t;

/* ---- Link kind / band ----------------------------------------------------- */
typedef enum {
    CUNXON_LINK_FEEDFORWARD   = 0,  /* default band: gamma */
    CUNXON_LINK_FEEDBACK      = 1,  /* default band: beta  */
    CUNXON_LINK_LATERAL       = 2,  /* default band: theta */
    CUNXON_LINK_THALAMIC      = 3,  /* default band: theta */
    CUNXON_LINK_CONTEXT       = 4   /* default band: alpha */
} cunxonLinkKind_t;

typedef enum {
    CUNXON_BAND_INFRASLOW = 0,
    CUNXON_BAND_SLOW      = 1,
    CUNXON_BAND_THETA     = 2,
    CUNXON_BAND_ALPHA     = 3,
    CUNXON_BAND_BETA      = 4,
    CUNXON_BAND_GAMMA     = 5,
    CUNXON_BAND_COUNT     = 6
} cunxonBand_t;

typedef enum {
    CUNXON_TOPO_DENSE        = 0,
    CUNXON_TOPO_SPARSE       = 1,
    CUNXON_TOPO_TOPOGRAPHIC  = 2,
    CUNXON_TOPO_ONE_TO_ONE   = 3
} cunxonTopology_t;

/* ---- Neuron type ---------------------------------------------------------- */
typedef enum {
    CUNXON_NEURON_INPUT  = 0,
    CUNXON_NEURON_HIDDEN = 1,
    CUNXON_NEURON_OUTPUT = 2
} cunxonNeuronType_t;

/* ---- Synapse type --------------------------------------------------------- */
typedef enum {
    CUNXON_SYN_IONOTROPIC_FAST = 0,
    CUNXON_SYN_IONOTROPIC_SLOW = 1,
    CUNXON_SYN_METABOTROPIC    = 2,
    CUNXON_SYN_SILENT          = 3
} cunxonSynapseType_t;


/* ===========================================================================
 *  NetworkParameters  - one per sphere
 *  Mirrors MultiNeuraxon2.py's NetworkParameters dataclass.
 *  All values are in SI-ish units; dt is in ms.
 * =========================================================================== */
typedef struct {
    /* ---- Architecture ---------------------------------------------------- */
    int   num_input_neurons;
    int   num_hidden_neurons;
    int   num_output_neurons;
    int   num_dendritic_branches;
    float dendritic_spike_threshold;
    float dendritic_supralinear_gamma;

    /* Watts-Strogatz small-world (hidden recurrent topology) */
    int   ws_k;
    float ws_beta;

    /* ---- Neuron / membrane ---------------------------------------------- */
    float membrane_time_constant;          /* tau_m (ms)  */
    float firing_threshold_excitatory;
    float firing_threshold_inhibitory;
    float adaptation_tau;
    float autoreceptor_tau;
    float spontaneous_firing_rate;
    float neuron_health_decay;

    /* ---- Homeostatic plasticity (Eq.1) ---------------------------------- */
    float target_firing_rate;
    float homeostatic_rate;
    float firing_rate_alpha;
    float threshold_mod_k;

    /* ---- MSTH 4 loops ---------------------------------------------------- */
    float msth_ultrafast_tau;       /* ~5 ms     */
    float msth_ultrafast_ceiling;
    float msth_fast_tau;            /* ~2 s      */
    float msth_fast_gain;
    float msth_medium_tau;          /* ~5 min    */
    float msth_medium_gain;
    float msth_slow_tau;            /* ~1-24 h   */
    float msth_slow_gain;

    /* ---- DSN dynamic decay (causal conv on the input stream) ----------- */
    int   dsn_kernel_size;          /* default 4 */
    int   dsn_enabled;              /* bool      */
    float dsn_bias;
    int   dsn_learn_enabled;
    float dsn_learn_lr;
    float dsn_target_sensitivity;
    float dsn_target_bias;
    float dsn_kernel_clip;

    /* ---- CTSN complement -------------------------------------------------- */
    float ctsn_rho;
    int   ctsn_enabled;
    float ctsn_phi_gain;
    float ctsn_phi_bias;
    int   ctsn_learn_enabled;
    float ctsn_learn_lr;
    float ctsn_phi_gain_clip;
    float ctsn_phi_bias_clip;

    /* ---- Synapse time constants & weight init --------------------------- */
    float tau_fast;
    float tau_slow;
    float tau_meta;
    float tau_stdp;
    float w_fast_init_min, w_fast_init_max;
    float w_slow_init_min, w_slow_init_max;
    float w_meta_init_min, w_meta_init_max;

    /* ---- ChronoPlasticity (Eqs 5-7) ------------------------------------- */
    float chrono_alpha_f;
    float chrono_alpha_s;
    float chrono_lambda_f;
    float chrono_lambda_s;
    int   chrono_enabled;
    float chrono_trace_clip;
    float chrono_gate_norm;
    float chrono_raw_clip;
    float chrono_omega_min;
    float chrono_omega_max;
    float chrono_omega_smoothing;

    /* ---- AGMP (Eqs 8-10) ------------------------------------------------- */
    float agmp_lambda_e;
    float agmp_lambda_a;
    float agmp_eta;
    int   agmp_enabled;

    /* ---- Plasticity ------------------------------------------------------ */
    float learning_rate;
    float stdp_window;
    float associative_alpha;
    float synapse_integrity_threshold;
    float synapse_formation_prob;
    float synapse_death_prob;
    float neuron_death_threshold;

    /* ---- Neuromodulator baselines --------------------------------------- */
    float dopamine_baseline;
    float serotonin_baseline;
    float acetylcholine_baseline;
    float norepinephrine_baseline;
    float tau_tonic;
    float tau_phasic;
    float neuromod_release_rate;
    float receptor_concentration_cap;

    /* ---- Oscillator bank (per-band frequency, Hz) ----------------------- */
    float osc_freq[CUNXON_BAND_COUNT];   /* defaults: 0.05, 0.5, 6, 10, 20, 40 */
    float osc_amplitude;                  /* common amplitude weight */
    float osc_pac_strength;               /* phase-amplitude coupling */

    /* ---- Misc ------------------------------------------------------------ */
    uint64_t random_seed_offset;          /* combined with context seed */
} cunxonNetworkParameters_t;


/* ===========================================================================
 *  LinkParameters - one per inter-sphere edge
 * =========================================================================== */
typedef struct {
    cunxonLinkKind_t kind;
    cunxonBand_t     coherence_band;
    float            gain;
    int              delay_steps;
    float            transmission_threshold;
    float            coherence_strength;     /* c in g_CTC = (1-c) + c*0.5*(1+cos dphi) */
    cunxonTopology_t topology;
    float            sparse_prob;
    int              allow_negative_weights;
    float            plasticity_rate;        /* Hebbian projection plasticity rate */
    float            weight_decay;
    float            weight_clip;
    int              normalize_rows;         /* L1-normalise rows of projection mat */
    float            bias;
} cunxonLinkParameters_t;


/* ---- Aigarth fitness callback signature ------------------------------------ */
typedef float (*cunxonFitnessFn_t)(cunxonNetwork_t clone, void* user_data);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUNXON_TYPES_H */
