"""
logger.py  —  Lightweight no-op data-logger stub for MultiNxonArcAGI3.

The vendored ``neuraxon`` package (taken verbatim from the offline
Game-of-Life v195 build) expects a rich analytics/data logger via
``from logger import get_data_logger``.  In the GoL project that module is a
~3800-line instrument that records every spike, plasticity event and time
series to disk for offline analysis.

For *playing ARC-AGI-3* and for running the NAS we do **not** want that
overhead — we run thousands of headless brain steps across many worker
processes, and per-event disk logging would dominate runtime.  So we ship a
tiny drop-in replacement that satisfies the exact surface the brain touches
while doing nothing.

The complete surface the ``neuraxon`` package uses (verified by grep over
neuraxon/*.py) is:

  * ``get_data_logger()._latest_input_saturation``  -> numeric (read)
  * ``logger.log_level``                            -> numeric, compared ``>= 2``
  * ``logger.time_series``                          -> dict (``in`` / ``[k]=`` / ``setdefault().append()``)
  * ``logger.current_step_data``                    -> dict (``[k] = v``)
  * ``logger._spont_count_pending``                 -> int (``+= 1``)
  * ``logger.paused``                               -> bool
  * ``logger.log_*( ... )``                         -> a family of event-logging methods (no-op)
  * ``logger._log_tick_level2( ... )``              -> analytics hook (no-op)

Everything else resolves through ``__getattr__`` to a no-op callable, so the
stub stays robust even if a rarely-hit code path references another
``log_*`` helper.
"""

from __future__ import annotations

from typing import Any


def _noop(*_args: Any, **_kwargs: Any) -> None:
    """A callable that accepts anything and does nothing."""
    return None


class _NullDataLogger:
    """A do-nothing stand-in for the GoL ``DataLogger``.

    Value-style attributes that the brain *reads as data* (and in some cases
    mutates with ``+=`` or item assignment) are declared as real instance
    attributes so they behave like genuine numbers / dicts.  Any other
    attribute access (chiefly the ``log_*`` event methods) falls through to
    :meth:`__getattr__`, which returns a universal no-op callable.
    """

    # Disable per-step analytics entirely. The brain guards its expensive
    # logging branches behind ``if logger.log_level >= 2``; keeping this at 0
    # means those branches are never taken.
    LOG_LEVEL_OFF = 0

    def __init__(self) -> None:
        # --- numeric state (read and/or incremented by the brain) ---
        self.log_level: int = self.LOG_LEVEL_OFF
        self._latest_input_saturation: float = 0.0
        self._spont_count_pending: int = 0
        self._log_tick_level: int = 0

        # --- container state (item-assigned / appended to by the brain) ---
        self.time_series: dict = {}
        self.current_step_data: dict = {}

        # --- flags ---
        # The real logger toggles ``paused`` itself to throttle the metrics
        # worker; for the stub we are simply never paused and never logging.
        self.paused: bool = False

    # The full set of event methods the brain calls. They are explicitly
    # listed (rather than relying solely on __getattr__) both as living
    # documentation of the contract and so that attribute *lookups* of these
    # names are cheap and unambiguous.
    log_associativity_event = _noop
    log_autoreceptor_event = _noop
    log_dendritic_spike_event = _noop
    log_driven_firing = _noop
    log_homeostatic_event = _noop
    log_itu_evolution_event = _noop
    log_neuromodulator_event = _noop
    log_plasticity_event = _noop
    log_silent_synapse_event = _noop
    log_spontaneous_event = _noop
    log_subthreshold_event = _noop
    log_threshold_modulation_event = _noop
    log_weight_evolution_event = _noop
    _log_tick_level2 = _noop

    # Catch-all: any other attribute (e.g. an unanticipated ``log_*`` helper)
    # resolves to a no-op callable. Because all *value* attributes above are
    # set as real instance/class attributes, this is only ever reached for
    # method-like access, so returning a callable is safe.
    def __getattr__(self, _name: str):  # noqa: D401 - simple stub
        return _noop

    # A couple of lifecycle methods the wider codebase sometimes calls. Kept
    # as explicit no-ops for clarity; harmless if never used.
    def flush(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def reset(self, *_args: Any, **_kwargs: Any) -> None:
        self.time_series.clear()
        self.current_step_data.clear()
        self._spont_count_pending = 0


# Process-wide singleton, mirroring the real module's accessor semantics.
_DATA_LOGGER: _NullDataLogger | None = None


def get_data_logger() -> _NullDataLogger:
    """Return the process-wide null data-logger singleton."""
    global _DATA_LOGGER
    if _DATA_LOGGER is None:
        _DATA_LOGGER = _NullDataLogger()
    return _DATA_LOGGER


def configure_logging(*_args: Any, **_kwargs: Any) -> _NullDataLogger:
    """Compatibility shim. Accepts any arguments, returns the null logger."""
    return get_data_logger()


def reset_data_logger() -> None:
    """Drop the singleton so the next ``get_data_logger`` rebuilds it fresh."""
    global _DATA_LOGGER
    _DATA_LOGGER = None
