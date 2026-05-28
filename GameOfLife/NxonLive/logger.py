# Multi Neuraxon Game of Life 5 — minimal logger stub  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# The v184 research build shipped a 3,374-line DataLogger that records
# dozens of diagnostic time-series, parquet exports, dashboards, etc.
# The GoL5 server keeps ONLY the fundamentals (world dynamics + neural
# substrate + g + all-time ranking), so the heavy logger is replaced
# by this no-op shim.
#
# The neural substrate (neuraxon/{neuron,network,components,multisphere})
# calls a small, fixed set of logger members:
#   .log_level                 gate for expensive logging branches
#   .current_step_data         scratch dict (per-step)
#   .time_series               dict (left empty — no recording)
#   .paused                    bool
#   ._latest_input_saturation  float read by components.py
#   ._spont_count_pending      int  scratch
#   .log_*( ... )              ~14 event hooks — all no-ops here
#
# log_level is pinned to 0 so every `if logger.log_level >= N:` branch
# in the substrate short-circuits → no diagnostic work is done at all,
# which is also the fastest configuration.
# ===================================================================


class _NullLogger:
    __slots__ = ("log_level", "current_step_data", "time_series",
                 "paused", "_latest_input_saturation",
                 "_spont_count_pending")

    def __init__(self):
        self.log_level = 0          # 0 ⇒ substrate skips all logging work
        self.current_step_data = {}
        self.time_series = {}
        self.paused = False
        self._latest_input_saturation = 0.0
        self._spont_count_pending = 0

    # Every log_* event hook the substrate may call resolves to this
    # no-op. Using __getattr__ keeps the shim future-proof if the
    # substrate gains new log_* calls.
    def _noop(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        # Only reached for attributes not in __slots__ (i.e. the
        # log_* methods). Return a callable no-op.
        return self._noop


_LOGGER = _NullLogger()


def get_data_logger():
    return _LOGGER
