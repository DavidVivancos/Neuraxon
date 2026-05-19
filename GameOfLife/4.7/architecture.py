# Neuraxon Game of Life v.4.79 architecture loader (Internal version 171)
# v155 introduces JSON-driven architecture parameterisation. See
# architectures/README.md for usage.
import json
import os
import sys
from typing import Any, Dict, Optional


# Built-in recognised keys per section. Anything not in these sets is
# accepted (with a warning) for forward compatibility.
KNOWN_KEYS = {
    "biology": {
        "metabolic_ramp_per_sec",
        "max_atrophy",
        "metabolic_rate_abs_cap_multiple",
        "start_food_default",
        "food_respawn_default",
        "food_sources_default",
        "mate_cooldown_seconds",
        "circadian_cycle_ticks",
        "idle_explore_seconds",
        "explore_probability",
    },
    "neural": {
        "num_input_neurons",
        "num_output_neurons",
        "num_hidden_neurons_default",
        "connection_probability",
        "afferent_synapse_strength",
        "proprioceptive_afferent_gain",
        "sensory_input_gain",
        "firing_threshold_excitatory",
        "firing_threshold_inhibitory",
        "spontaneous_firing_rate",
        "intrinsic_timescale_default",
        "resting_potential_decay",
        "sensorimotor_coupling",   # v164
        "symmetric_stdp",          # v169 (v4.77) — MultiNeuraxon2 Bug #3 opt-in
        "refractory_period_ticks", # v171 (v4.79) — state 0 buffer after firing
    },
    "operating_ranges": {
        "learning_rate",
        "plasticity_threshold",
        "adaptation_tau_ticks",
        "adaptation_target_excitatory_multiplier",
        "adaptation_target_inhibitory_multiplier",
        "autoreceptor_coefficient",
        "autoreceptor_tau_ticks",
        "autoreceptor_rate_coeff",
        "sensory_boost_function",
        "sensory_boost_scale",
        "plasticity_brake_threshold",
        "plasticity_brake_slope",
        "plasticity_brake_floor",
    },
    "healthy_bands": set(),  # any metric_key:[lo,hi] pair is valid
}


# Module-level cache of the loaded architecture. Lazily initialised on
# first call to get_architecture() / load_architecture(). Subsequent reads
# return the same dict.
_ARCH: Optional[Dict[str, Any]] = None
_ARCH_PATH: Optional[str] = None


def _resolve_path(explicit_path: Optional[str]) -> str:
    """Find which architecture JSON file to load.

    Priority (first hit wins):
      1. Explicit path argument
      2. --architecture CLI flag in sys.argv
      3. NEURAXON_ARCH environment variable
      4. architectures/default.json next to this module
    """
    if explicit_path:
        return explicit_path
    # CLI flag
    if '--architecture' in sys.argv:
        try:
            idx = sys.argv.index('--architecture')
            if idx + 1 < len(sys.argv):
                return sys.argv[idx + 1]
        except ValueError:
            pass
    # Env var
    env_path = os.environ.get('NEURAXON_ARCH')
    if env_path:
        return env_path
    # Default
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, 'architectures', 'default.json')


def load_architecture(path: Optional[str] = None,
                       verbose: bool = True) -> Dict[str, Any]:
    """Load an architecture JSON file, validate it, and cache it.

    Returns a dict with the same shape as default.json. Unknown keys are
    kept (with a warning printed) so forward-compatible configs don't
    crash older code.

    If loading fails for any reason, returns an empty dict — callers
    should treat missing keys as "use the code default".
    """
    global _ARCH, _ARCH_PATH
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        if verbose:
            print(f"[ARCHITECTURE] No architecture file at {resolved!r} — "
                  f"using code defaults throughout")
        _ARCH = {}
        _ARCH_PATH = None
        return _ARCH
    try:
        with open(resolved, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as exc:
        if verbose:
            print(f"[ARCHITECTURE] Failed to load {resolved}: {exc} — "
                  f"using code defaults")
        _ARCH = {}
        _ARCH_PATH = None
        return _ARCH
    if not isinstance(data, dict):
        if verbose:
            print(f"[ARCHITECTURE] {resolved}: top-level must be an object — "
                  f"using code defaults")
        _ARCH = {}
        return _ARCH
    # Strip the _meta and _doc fields, count overrides per section,
    # warn about unrecognised keys.
    cleaned: Dict[str, Dict[str, Any]] = {}
    for section, known in KNOWN_KEYS.items():
        section_data = data.get(section, {})
        if not isinstance(section_data, dict):
            if verbose:
                print(f"[ARCHITECTURE] section {section!r} must be an object, "
                      f"skipping")
            continue
        kept: Dict[str, Any] = {}
        for k, v in section_data.items():
            if k.startswith('_'):
                continue  # documentation / meta keys
            if known and k not in known:
                if verbose:
                    print(f"[ARCHITECTURE] WARNING: unknown key "
                          f"{section}.{k} (forward-compatible — keeping)")
            kept[k] = v
        cleaned[section] = kept
    _ARCH = cleaned
    _ARCH_PATH = resolved
    if verbose:
        print(f"[ARCHITECTURE] Loaded {resolved}")
        for section in KNOWN_KEYS:
            n = len(cleaned.get(section, {}))
            note = "" if n > 0 else " (using code defaults)"
            print(f"[ARCHITECTURE]   {section}: {n} overrides{note}")
    return _ARCH


def get_architecture() -> Dict[str, Any]:
    """Return the cached architecture dict, loading if needed."""
    if _ARCH is None:
        load_architecture()
    return _ARCH or {}


def get_param(section: str, key: str, default: Any = None) -> Any:
    """Look up architecture.<section>.<key>, returning the code default if
    not specified in the JSON.

    Usage from game code:
        from architecture import get_param
        ramp = get_param('biology', 'metabolic_ramp_per_sec', default=10.0)
    """
    arch = get_architecture()
    section_data = arch.get(section, {})
    if key in section_data:
        return section_data[key]
    return default


def get_healthy_band(metric_key: str) -> Optional[list]:
    """Return [lo, hi] for a metric, or None if not in the architecture
    healthy_bands. Callers should fall back to a hard-coded default."""
    arch = get_architecture()
    bands = arch.get('healthy_bands', {})
    band = bands.get(metric_key)
    if isinstance(band, list) and len(band) == 2:
        return band
    return None


def get_loaded_path() -> Optional[str]:
    """Return the absolute path of the architecture file that was loaded,
    or None if no file was found / using pure defaults."""
    return _ARCH_PATH
