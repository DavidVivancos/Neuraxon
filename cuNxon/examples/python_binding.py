"""
python_binding.py
==================
Minimal ctypes wrapper for cuNxon's C API, so users of the reference
``MultiNeuraxon2.py`` can drive the GPU library straight from Python.

Build cuNxon first (see ../README.md), then::

    python examples/python_binding.py             # tiny smoke-test
    python examples/python_binding.py --train 500 # full 4-sphere demo

Tested on Linux (``libcunxon.so``) and Windows (``cunxon.dll``).

(c) 2026  cuNxon authors -- MIT licence
"""

from __future__ import annotations

import argparse
import ctypes as C
import os
import platform
import sys
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 1.  load the shared library
# ---------------------------------------------------------------------------
def _default_lib_path() -> str:
    """Look for the built shared lib in the most common places."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if platform.system() == "Windows":
        candidates += [os.path.join(here, "..", "build", "Release", "cunxon.dll"),
                       os.path.join(here, "..", "build", "cunxon.dll"),
                       "cunxon.dll"]
    elif platform.system() == "Darwin":
        candidates += [os.path.join(here, "..", "build", "libcunxon.dylib"),
                       "libcunxon.dylib"]
    else:
        candidates += [os.path.join(here, "..", "build", "libcunxon.so"),
                       "libcunxon.so"]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find libcunxon. Build it first (see README) "
        "or set CUNXON_LIB env var."
    )


_lib_path = os.environ.get("CUNXON_LIB") or _default_lib_path()
_lib = C.CDLL(_lib_path)


# ---------------------------------------------------------------------------
# 2.  enums  (kept in sync with cuNxon_types.h)
# ---------------------------------------------------------------------------
# status
CUNXON_OK = 0

# sphere kinds
SPHERE_SENSORY, SPHERE_ASSOCIATION, SPHERE_MOTOR, SPHERE_THALAMIC, SPHERE_CUSTOM = range(5)

# link kinds
LINK_FEEDFORWARD, LINK_FEEDBACK, LINK_LATERAL, LINK_THALAMIC, LINK_CONTEXT = range(5)

# oscillator bands
BAND_INFRASLOW, BAND_SLOW, BAND_THETA, BAND_ALPHA, BAND_BETA, BAND_GAMMA = range(6)

# topology
TOPO_DENSE, TOPO_SPARSE, TOPO_TOPOGRAPHIC, TOPO_ONE_TO_ONE = range(4)

# context properties
PROP_DEVICE_NAME         = 1
PROP_COMPUTE_CAPABILITY  = 2
PROP_TOTAL_GLOBAL_MEM    = 3
PROP_SM_COUNT            = 6

# neuromodulator indices
NM_DA, NM_5HT, NM_ACH, NM_NA = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# 3.  ctypes struct mirrors
# ---------------------------------------------------------------------------
# The structs below mirror cuNxon_types.h *field-for-field, in order*.
# If you change the C header, regenerate these structs.

CUNXON_BAND_COUNT = 6

class NetworkParameters(C.Structure):
    """Mirrors cunxonNetworkParameters_t from cuNxon_types.h *field-for-field,
    in order*. Keep these two in sync."""
    _fields_ = [
        # ---- Architecture --------------------------------------------------
        ("num_input_neurons",            C.c_int),
        ("num_hidden_neurons",           C.c_int),
        ("num_output_neurons",           C.c_int),
        ("num_dendritic_branches",       C.c_int),
        ("dendritic_spike_threshold",    C.c_float),
        ("dendritic_supralinear_gamma",  C.c_float),
        # Watts-Strogatz
        ("ws_k",                         C.c_int),
        ("ws_beta",                      C.c_float),
        # ---- Neuron / membrane --------------------------------------------
        ("membrane_time_constant",       C.c_float),
        ("firing_threshold_excitatory",  C.c_float),
        ("firing_threshold_inhibitory",  C.c_float),
        ("adaptation_tau",               C.c_float),
        ("autoreceptor_tau",             C.c_float),
        ("spontaneous_firing_rate",      C.c_float),
        ("neuron_health_decay",          C.c_float),
        # ---- Homeostatic plasticity ---------------------------------------
        ("target_firing_rate",           C.c_float),
        ("homeostatic_rate",             C.c_float),
        ("firing_rate_alpha",            C.c_float),
        ("threshold_mod_k",              C.c_float),
        # ---- MSTH 4 loops --------------------------------------------------
        ("msth_ultrafast_tau",           C.c_float),
        ("msth_ultrafast_ceiling",       C.c_float),
        ("msth_fast_tau",                C.c_float),
        ("msth_fast_gain",               C.c_float),
        ("msth_medium_tau",              C.c_float),
        ("msth_medium_gain",             C.c_float),
        ("msth_slow_tau",                C.c_float),
        ("msth_slow_gain",               C.c_float),
        # ---- DSN ----------------------------------------------------------
        ("dsn_kernel_size",              C.c_int),
        ("dsn_enabled",                  C.c_int),
        ("dsn_bias",                     C.c_float),
        ("dsn_learn_enabled",            C.c_int),
        ("dsn_learn_lr",                 C.c_float),
        ("dsn_target_sensitivity",       C.c_float),
        ("dsn_target_bias",              C.c_float),
        ("dsn_kernel_clip",              C.c_float),
        # ---- CTSN complement ----------------------------------------------
        ("ctsn_rho",                     C.c_float),
        ("ctsn_enabled",                 C.c_int),
        ("ctsn_phi_gain",                C.c_float),
        ("ctsn_phi_bias",                C.c_float),
        ("ctsn_learn_enabled",           C.c_int),
        ("ctsn_learn_lr",                C.c_float),
        ("ctsn_phi_gain_clip",           C.c_float),
        ("ctsn_phi_bias_clip",           C.c_float),
        # ---- Synapse time constants & weight init -------------------------
        ("tau_fast",                     C.c_float),
        ("tau_slow",                     C.c_float),
        ("tau_meta",                     C.c_float),
        ("tau_stdp",                     C.c_float),
        ("w_fast_init_min",              C.c_float),
        ("w_fast_init_max",              C.c_float),
        ("w_slow_init_min",              C.c_float),
        ("w_slow_init_max",              C.c_float),
        ("w_meta_init_min",              C.c_float),
        ("w_meta_init_max",              C.c_float),
        # ---- ChronoPlasticity (Eqs 5-7) -----------------------------------
        ("chrono_alpha_f",               C.c_float),
        ("chrono_alpha_s",               C.c_float),
        ("chrono_lambda_f",              C.c_float),
        ("chrono_lambda_s",              C.c_float),
        ("chrono_enabled",               C.c_int),
        ("chrono_trace_clip",            C.c_float),
        ("chrono_gate_norm",             C.c_float),
        ("chrono_raw_clip",              C.c_float),
        ("chrono_omega_min",             C.c_float),
        ("chrono_omega_max",             C.c_float),
        ("chrono_omega_smoothing",       C.c_float),
        # ---- AGMP ---------------------------------------------------------
        ("agmp_lambda_e",                C.c_float),
        ("agmp_lambda_a",                C.c_float),
        ("agmp_eta",                     C.c_float),
        ("agmp_enabled",                 C.c_int),
        # ---- Plasticity ---------------------------------------------------
        ("learning_rate",                C.c_float),
        ("stdp_window",                  C.c_float),
        ("associative_alpha",            C.c_float),
        ("synapse_integrity_threshold",  C.c_float),
        ("synapse_formation_prob",       C.c_float),
        ("synapse_death_prob",           C.c_float),
        ("neuron_death_threshold",       C.c_float),
        # ---- Neuromodulator baselines -------------------------------------
        ("dopamine_baseline",            C.c_float),
        ("serotonin_baseline",           C.c_float),
        ("acetylcholine_baseline",       C.c_float),
        ("norepinephrine_baseline",      C.c_float),
        ("tau_tonic",                    C.c_float),
        ("tau_phasic",                   C.c_float),
        ("neuromod_release_rate",        C.c_float),
        ("receptor_concentration_cap",   C.c_float),
        # ---- Oscillator bank ----------------------------------------------
        ("osc_freq",                     C.c_float * CUNXON_BAND_COUNT),
        ("osc_amplitude",                C.c_float),
        ("osc_pac_strength",             C.c_float),
        # ---- Misc ---------------------------------------------------------
        ("random_seed_offset",           C.c_uint64),
    ]


class LinkParameters(C.Structure):
    _fields_ = [
        ("kind",                     C.c_int),   # cunxonLinkKind_t
        ("coherence_band",           C.c_int),   # cunxonBand_t
        ("gain",                     C.c_float),
        ("delay_steps",              C.c_int),
        ("transmission_threshold",   C.c_float),
        ("coherence_strength",       C.c_float),
        ("topology",                 C.c_int),   # cunxonTopology_t
        ("sparse_prob",              C.c_float),
        ("allow_negative_weights",   C.c_int),
        ("plasticity_rate",          C.c_float),
        ("weight_decay",             C.c_float),
        ("weight_clip",              C.c_float),
        ("normalize_rows",           C.c_int),
        ("bias",                     C.c_float),
    ]


# ---------------------------------------------------------------------------
# 4.  prototype bindings
# ---------------------------------------------------------------------------
def _bind(name, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype  = restype
    fn.argtypes = argtypes
    return fn


_lib.cunxonGetStatusString.restype  = C.c_char_p
_lib.cunxonGetStatusString.argtypes = [C.c_int]
_lib.cunxonGetLastError.restype     = C.c_char_p
_lib.cunxonGetLastError.argtypes    = []
_lib.cunxonGetVersion.restype       = C.c_char_p
_lib.cunxonGetVersion.argtypes      = []

CtxP = C.c_void_p
NetP = C.c_void_p

cunxonCreateContext        = _bind("cunxonCreateContext",
                                   C.c_int, [C.POINTER(CtxP), C.c_int, C.c_uint64, C.c_uint32])
cunxonDestroyContext       = _bind("cunxonDestroyContext",  C.c_int, [CtxP])
cunxonContextSync          = _bind("cunxonContextSync",     C.c_int, [CtxP])
cunxonContextGetProperty   = _bind("cunxonContextGetProperty",
                                   C.c_int, [CtxP, C.c_int, C.c_void_p, C.c_size_t])

cunxonNetworkCreate        = _bind("cunxonNetworkCreate",
                                   C.c_int, [CtxP, C.POINTER(NetP), C.c_char_p])
cunxonNetworkDestroy       = _bind("cunxonNetworkDestroy",  C.c_int, [NetP])

cunxonGetDefaultParameters = _bind("cunxonGetDefaultParameters",
                                   C.c_int, [C.POINTER(NetworkParameters)])

cunxonNetworkAddSphere     = _bind("cunxonNetworkAddSphere",
                                   C.c_int,
                                   [NetP, C.c_char_p, C.c_int,
                                    C.POINTER(NetworkParameters), C.POINTER(C.c_int)])

cunxonNetworkSetSphereInterface = _bind("cunxonNetworkSetSphereInterface",
                                   C.c_int,
                                   [NetP, C.c_int,
                                    C.POINTER(C.c_int), C.c_int,
                                    C.POINTER(C.c_int), C.c_int,
                                    C.POINTER(C.c_int), C.c_int,
                                    C.POINTER(C.c_int), C.c_int])

cunxonNetworkAddLink       = _bind("cunxonNetworkAddLink",
                                   C.c_int,
                                   [NetP, C.c_int, C.c_int,
                                    C.POINTER(LinkParameters), C.POINTER(C.c_int)])

cunxonNetworkFinalize      = _bind("cunxonNetworkFinalize", C.c_int, [NetP])
cunxonNetworkReset         = _bind("cunxonNetworkReset",    C.c_int, [NetP])
cunxonNetworkNumSpheres    = _bind("cunxonNetworkNumSpheres", C.c_int, [NetP])

cunxonNetworkStepInfer     = _bind("cunxonNetworkStepInfer",
                                   C.c_int,
                                   [NetP, C.POINTER(C.POINTER(C.c_float)), C.c_float])
cunxonNetworkStepTrain     = _bind("cunxonNetworkStepTrain",
                                   C.c_int,
                                   [NetP, C.POINTER(C.POINTER(C.c_float)), C.c_float])

cunxonNetworkInjectNeuromodulator = _bind("cunxonNetworkInjectNeuromodulator",
                                   C.c_int, [NetP, C.c_int, C.c_float])

cunxonSphereGetReadout     = _bind("cunxonSphereGetReadout",
                                   C.c_int,
                                   [NetP, C.c_int, C.POINTER(C.c_int8), C.POINTER(C.c_int)])

cunxonNetworkGetEnergy     = _bind("cunxonNetworkGetEnergy",
                                   C.c_int, [NetP, C.POINTER(C.c_double)])
cunxonNetworkSave          = _bind("cunxonNetworkSave",
                                   C.c_int, [NetP, C.c_char_p])
cunxonNetworkLoad          = _bind("cunxonNetworkLoad",
                                   C.c_int, [NetP, C.c_char_p])

# --- Aigarth ---
CunxonFitnessFn = C.CFUNCTYPE(C.c_float, NetP, C.c_void_p)
cunxonNetworkAigarthConfig = _bind("cunxonNetworkAigarthConfig",
                                   C.c_int, [NetP, C.c_int, C.c_float,
                                             C.c_float, C.c_float])
cunxonNetworkAigarthStep   = _bind("cunxonNetworkAigarthStep",
                                   C.c_int, [NetP, CunxonFitnessFn, C.c_void_p])

# --- Sphere layers (paper §P4) ---
cunxonNetworkAddLayer      = _bind("cunxonNetworkAddLayer",
                                   C.c_int,
                                   [NetP, C.c_char_p, C.c_int, C.POINTER(C.c_int)])
cunxonNetworkAddSphereToLayer = _bind("cunxonNetworkAddSphereToLayer",
                                      C.c_int, [NetP, C.c_int, C.c_int])
cunxonNetworkNumLayers     = _bind("cunxonNetworkNumLayers",
                                   C.c_int, [NetP])
cunxonNetworkGetLayer      = _bind("cunxonNetworkGetLayer",
                                   C.c_int,
                                   [NetP, C.c_int, C.c_char_p, C.c_int,
                                    C.POINTER(C.c_int),
                                    C.POINTER(C.c_int), C.POINTER(C.c_int)])

# --- Pattern application layer (Algorithm 8) ---
cunxonNetworkStorePattern  = _bind("cunxonNetworkStorePattern",
                                   C.c_int,
                                   [NetP, C.c_int, C.c_char_p,
                                    C.POINTER(C.c_float), C.c_int,
                                    C.c_int, C.c_float])
cunxonNetworkRecallPattern = _bind("cunxonNetworkRecallPattern",
                                   C.c_int,
                                   [NetP, C.c_int, C.c_char_p, C.c_int,
                                    C.c_float, C.c_int, C.c_float,
                                    C.POINTER(C.c_int8), C.POINTER(C.c_int)])
cunxonNetworkTrainSequence = _bind("cunxonNetworkTrainSequence",
                                   C.c_int,
                                   [NetP, C.c_int,
                                    C.POINTER(C.POINTER(C.c_float)),
                                    C.c_int, C.c_int, C.c_int, C.c_int,
                                    C.c_float])
cunxonNetworkListPatterns  = _bind("cunxonNetworkListPatterns",
                                   C.c_int,
                                   [NetP, C.c_char_p, C.c_int,
                                    C.POINTER(C.c_int)])
cunxonNetworkClearPatterns = _bind("cunxonNetworkClearPatterns",
                                   C.c_int, [NetP])


# ---------------------------------------------------------------------------
# 5.  thin Pythonic wrapper
# ---------------------------------------------------------------------------
def _check(status: int, where: str) -> None:
    if status != CUNXON_OK:
        msg = _lib.cunxonGetStatusString(status).decode("utf-8", errors="replace")
        last = (_lib.cunxonGetLastError() or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"{where}: {msg} ({last})")


class Context:
    def __init__(self, device_id: int = 0, seed: int = 42, flags: int = 0):
        self._handle = CtxP()
        _check(cunxonCreateContext(C.byref(self._handle), device_id, seed, flags),
               "cunxonCreateContext")

    def close(self):
        if self._handle:
            cunxonDestroyContext(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def device_name(self) -> str:
        buf = C.create_string_buffer(256)
        cunxonContextGetProperty(self._handle, PROP_DEVICE_NAME, buf, 256)
        return buf.value.decode()

    def compute_capability(self) -> int:
        v = C.c_int()
        cunxonContextGetProperty(self._handle, PROP_COMPUTE_CAPABILITY, C.byref(v), 4)
        return int(v.value)


class Network:
    def __init__(self, ctx: Context, name: str = "net"):
        self._ctx    = ctx
        self._handle = NetP()
        _check(cunxonNetworkCreate(ctx._handle, C.byref(self._handle), name.encode()),
               "cunxonNetworkCreate")
        self._n_sensory_inputs: List[int] = []

    def close(self):
        if self._handle:
            cunxonNetworkDestroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    # ---- topology -----------------------------------------------------
    @staticmethod
    def default_params() -> NetworkParameters:
        p = NetworkParameters()
        _check(cunxonGetDefaultParameters(C.byref(p)), "cunxonGetDefaultParameters")
        return p

    def add_sphere(self, name: str, kind: int, params: NetworkParameters) -> int:
        sid = C.c_int()
        _check(cunxonNetworkAddSphere(self._handle, name.encode(), kind,
                                      C.byref(params), C.byref(sid)),
               "cunxonNetworkAddSphere")
        # track sensory input width per sphere id, default to num_input_neurons
        while len(self._n_sensory_inputs) <= sid.value:
            self._n_sensory_inputs.append(0)
        self._n_sensory_inputs[sid.value] = params.num_input_neurons
        return int(sid.value)

    def set_interface(self, sphere_id: int,
                      sensory_inputs:  Sequence[int],
                      relay_inputs:    Sequence[int],
                      relay_outputs:   Sequence[int],
                      readout_outputs: Sequence[int]) -> None:
        def to_arr(s):
            a = (C.c_int * len(s))(*s) if s else (C.c_int * 0)()
            return a, len(s)
        a1, n1 = to_arr(sensory_inputs)
        a2, n2 = to_arr(relay_inputs)
        a3, n3 = to_arr(relay_outputs)
        a4, n4 = to_arr(readout_outputs)
        _check(cunxonNetworkSetSphereInterface(self._handle, sphere_id,
                                               a1, n1, a2, n2, a3, n3, a4, n4),
               "cunxonNetworkSetSphereInterface")
        self._n_sensory_inputs[sphere_id] = n1

    def add_link(self, src: int, dst: int, lp: LinkParameters) -> int:
        lid = C.c_int()
        _check(cunxonNetworkAddLink(self._handle, src, dst, C.byref(lp), C.byref(lid)),
               "cunxonNetworkAddLink")
        return int(lid.value)

    def finalize(self) -> None:
        _check(cunxonNetworkFinalize(self._handle), "cunxonNetworkFinalize")

    def reset(self) -> None:
        _check(cunxonNetworkReset(self._handle), "cunxonNetworkReset")

    @property
    def num_spheres(self) -> int:
        return cunxonNetworkNumSpheres(self._handle)

    # ---- execution ----------------------------------------------------
    def _pack_inputs(self,
                     ext_inputs: Sequence[Optional[np.ndarray]]
                     ) -> "C.Array":
        """Build a (float*)[num_spheres] argv-style array of device pointers."""
        n_spheres = self.num_spheres
        assert len(ext_inputs) == n_spheres, \
            f"need {n_spheres} ext_inputs entries (one per sphere, may be None)"
        ptr_array = (C.POINTER(C.c_float) * n_spheres)()
        self._keepalive = []   # prevent GC
        for i, x in enumerate(ext_inputs):
            if x is None:
                ptr_array[i] = C.POINTER(C.c_float)()
                continue
            arr = np.ascontiguousarray(x, dtype=np.float32)
            assert arr.size == self._n_sensory_inputs[i], \
                f"sphere {i}: expected {self._n_sensory_inputs[i]} inputs, got {arr.size}"
            self._keepalive.append(arr)
            ptr_array[i] = arr.ctypes.data_as(C.POINTER(C.c_float))
        return ptr_array

    def step_infer(self, ext_inputs: Sequence[Optional[np.ndarray]],
                   dt_ms: float = 1.0) -> None:
        a = self._pack_inputs(ext_inputs)
        _check(cunxonNetworkStepInfer(self._handle, a, dt_ms),
               "cunxonNetworkStepInfer")

    def step_train(self, ext_inputs: Sequence[Optional[np.ndarray]],
                   dt_ms: float = 1.0) -> None:
        a = self._pack_inputs(ext_inputs)
        _check(cunxonNetworkStepTrain(self._handle, a, dt_ms),
               "cunxonNetworkStepTrain")

    def inject_neuromod(self, which: int, amount: float) -> None:
        _check(cunxonNetworkInjectNeuromodulator(self._handle, which, amount),
               "cunxonNetworkInjectNeuromodulator")

    def get_readout(self, sphere_id: int) -> np.ndarray:
        n = C.c_int(0)
        _check(cunxonSphereGetReadout(self._handle, sphere_id, None, C.byref(n)),
               "cunxonSphereGetReadout(size)")
        out = np.zeros(n.value, dtype=np.int8)
        _check(cunxonSphereGetReadout(self._handle, sphere_id,
                                      out.ctypes.data_as(C.POINTER(C.c_int8)),
                                      C.byref(n)),
               "cunxonSphereGetReadout(data)")
        return out

    def energy(self) -> float:
        e = C.c_double(0.0)
        cunxonNetworkGetEnergy(self._handle, C.byref(e))
        return float(e.value)

    def save(self, path: str) -> None:
        _check(cunxonNetworkSave(self._handle, path.encode()), "cunxonNetworkSave")


# ---------------------------------------------------------------------------
# 6.  optional 4-sphere demo (mirrors example_4sphere.cu)
# ---------------------------------------------------------------------------
def _demo(n_train: int = 500, n_infer: int = 200, device: int = 0) -> None:
    print(f"=== cuNxon Python demo (lib: {_lib_path}) ===")
    ctx = Context(device_id=device, seed=0xC0FFEE2026)
    print(f"  device : {ctx.device_name()} (cc {ctx.compute_capability()/10:.1f})")

    net = Network(ctx, "py_demo")

    def mkparams(n_in, n_hid, n_out, offset):
        p = Network.default_params()
        p.num_input_neurons  = n_in
        p.num_hidden_neurons = n_hid
        p.num_output_neurons = n_out
        p.random_seed_offset = offset
        return p

    pVIS = mkparams( 6, 44, 14, 1)
    pAUD = mkparams( 6, 44, 14, 2)
    pASC = mkparams(30, 51, 17, 3); pASC.ws_k = 8; pASC.ws_beta = 0.15
    pMTR = mkparams(19, 21,  5, 4)

    vis = net.add_sphere("VIS", SPHERE_SENSORY,     pVIS)
    aud = net.add_sphere("AUD", SPHERE_SENSORY,     pAUD)
    asc = net.add_sphere("ASC", SPHERE_ASSOCIATION, pASC)
    mtr = net.add_sphere("MTR", SPHERE_MOTOR,       pMTR)

    # interfaces (same convention as the C++ demo)
    net.set_interface(vis,
                      sensory_inputs=list(range(pVIS.num_input_neurons)),
                      relay_inputs=[],
                      relay_outputs=list(range(pVIS.num_output_neurons - 2)),
                      readout_outputs=list(range(pVIS.num_output_neurons - 2,
                                                  pVIS.num_output_neurons)))
    net.set_interface(aud,
                      sensory_inputs=list(range(pAUD.num_input_neurons)),
                      relay_inputs=[],
                      relay_outputs=list(range(pAUD.num_output_neurons - 2)),
                      readout_outputs=list(range(pAUD.num_output_neurons - 2,
                                                  pAUD.num_output_neurons)))
    net.set_interface(asc,
                      sensory_inputs=[],
                      relay_inputs=list(range(pASC.num_input_neurons)),
                      relay_outputs=list(range(pASC.num_output_neurons - 3)),
                      readout_outputs=list(range(pASC.num_output_neurons - 3,
                                                  pASC.num_output_neurons)))
    net.set_interface(mtr,
                      sensory_inputs=[],
                      relay_inputs=list(range(pMTR.num_input_neurons)),
                      relay_outputs=[],
                      readout_outputs=list(range(pMTR.num_output_neurons)))

    def mklink(kind, band, gain, delay, c):
        lp = LinkParameters(kind=kind, coherence_band=band, gain=gain,
                            delay_steps=delay, transmission_threshold=0.0,
                            coherence_strength=c, topology=TOPO_DENSE,
                            sparse_prob=0.3, allow_negative_weights=1,
                            plasticity_rate=1e-3, weight_decay=1e-5,
                            weight_clip=1.0, normalize_rows=0, bias=0.0)
        return lp

    net.add_link(vis, asc, mklink(LINK_FEEDFORWARD, BAND_GAMMA, 1.0, 1, 0.6))
    net.add_link(aud, asc, mklink(LINK_FEEDFORWARD, BAND_GAMMA, 1.0, 1, 0.6))
    net.add_link(asc, mtr, mklink(LINK_FEEDFORWARD, BAND_GAMMA, 1.2, 1, 0.7))
    net.add_link(mtr, asc, mklink(LINK_FEEDBACK,    BAND_BETA,  0.4, 2, 0.5))
    net.add_link(asc, vis, mklink(LINK_THALAMIC,    BAND_THETA, 0.3, 2, 0.4))
    net.add_link(asc, aud, mklink(LINK_THALAMIC,    BAND_THETA, 0.3, 2, 0.4))
    net.add_link(vis, aud, mklink(LINK_LATERAL,     BAND_THETA, 0.25, 1, 0.35))
    net.add_link(aud, vis, mklink(LINK_LATERAL,     BAND_THETA, 0.25, 1, 0.35))

    net.finalize()
    print(f"  network finalised : {net.num_spheres} spheres")

    rng = np.random.default_rng(0xBEEF)

    def make_inputs():
        salient = rng.random() < 0.30
        if salient:
            vis_in = np.concatenate([0.85 + 0.10 * rng.uniform(-1, 1, 3),
                                     0.20 * rng.uniform(-1, 1, pVIS.num_input_neurons - 3)])
        else:
            vis_in = 0.20 * rng.uniform(-1, 1, pVIS.num_input_neurons)
        aud_in = 0.30 * rng.uniform(-1, 1, pAUD.num_input_neurons)
        return vis_in.astype(np.float32), aud_in.astype(np.float32), salient

    # ---- training ----
    cum_reward = 0.0
    for t in range(n_train):
        vis_in, aud_in, salient = make_inputs()
        ext = [vis_in, aud_in, None, None]   # one slot per sphere index
        net.step_train(ext)
        readout = net.get_readout(mtr)
        target  = +1 if salient else -1
        reward  = 1.0 if int(readout[0]) == target else -0.2
        cum_reward += reward
        net.inject_neuromod(NM_DA, reward)
        if (t + 1) % 100 == 0:
            print(f"  step {t+1:4d}   avg_reward={cum_reward / (t+1):+.3f}   "
                  f"energy={net.energy():g}")

    # ---- evaluation ----
    hits = 0; n_salient = 0
    for _ in range(n_infer):
        vis_in, aud_in, salient = make_inputs()
        net.step_infer([vis_in, aud_in, None, None])
        readout = net.get_readout(mtr)
        if salient:
            n_salient += 1
            if int(readout[0]) == +1:
                hits += 1
    pct = 100.0 * hits / max(n_salient, 1)
    print(f"detection hit-rate on salient stimuli: {hits} / {n_salient} ({pct:.1f}%)")

    net.save("demo_4sphere_py.cunxon")
    print("saved -> demo_4sphere_py.cunxon")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",   type=int, default=500)
    ap.add_argument("--infer",   type=int, default=200)
    ap.add_argument("--device",  type=int, default=0)
    args = ap.parse_args()
    _demo(n_train=args.train, n_infer=args.infer, device=args.device)
