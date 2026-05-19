# Neuraxon Game of Life v.4.79 metrics_worker (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 171
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   "Multi-Neuraxon: Emergent Specialization, Modular, Frequency-Gated Neural Dynamics" by David Vivancos & Jose Sanchez
"""
neuraxon/metrics_worker.py  (NEW in v147 / v4.55)
==================================================
Runs the M1-M10 paper-fidelity metrics computation on a background thread so
the main game loop doesn't pay the ~5-15 ms compute cost on every full-
analytics tick.

Threading model
---------------
Single producer (main game thread, calls `enqueue()` once per analytics tick),
single consumer (the worker thread). A `queue.Queue` decouples them with a
small fixed depth — if metrics fall behind, older items are dropped on a
"latest snapshot wins" basis (see `enqueue()` for the policy). The worker:
  1. Pulls a snapshot off the queue.
  2. Calls `research_probes.compute_all_metrics()` exactly as the inline path
     would have.
  3. Acquires `logger_lock`, appends results into `time_series`, releases.
  4. Loops.

Race tolerance
--------------
The worker reads live game state (alive_nxers list, neurons, networks). It
does NOT take a deep copy — that would defeat the threading speedup for big
populations.  Instead it relies on:

  * The CPython GIL making single attribute reads atomic. Reading
    `neuron.trinary_state` or `neuron.is_active` in the worker while the main
    thread mutates them is safe at the bytecode level — we get a slightly
    stale value but never a crash or torn float.
  * All M1-M10 metrics are AGGREGATES over thousands of values (not single
    pointwise checks), so a handful of stale reads don't change the
    aggregate-level conclusions.
  * Structural mutations (births, deaths, neuron / synapse pruning) are rare
    relative to ~10-tick analytics intervals. The biggest risk —  iterating
    over `alive_nxers` while `__delitem__` is happening — is avoided
    because the producer enqueues a *list snapshot* of the alive_nxers,
    which is a fresh list (the items inside are still live references).

The only thing that needs explicit locking is the final time_series append.
Multiple-thread reads of a list are safe under the GIL; mixed read/write is
safe for individual `append`s but the dashboard's `_refresh_cache` does
`list(ts.get(k, []))` which makes a copy — that copy can race with append in
weird ways. Hence `logger_lock`: held briefly during writes, briefly during
the dashboard's snapshot.

Disabling the worker
--------------------
If `THREADED_METRICS_ENABLED = False` (config flag), the logger falls back to
the v146 inline behaviour. Useful for debugging or for environments where
the GIL provides enough serialization that threading overhead exceeds the
gain.

API
---
    worker = MetricsWorker(probe_state, logger_lock)
    worker.start()
    ...
    worker.enqueue(payload)         # called from main thread
    ...
    worker.stop(timeout_s=2.0)      # called at shutdown

`payload` is a dict with the same keys that `compute_all_metrics()` consumes
plus a 'tick' field — the worker calls compute_all_metrics with **payload.
"""
import os
import queue
import threading
import time
import traceback
from typing import Optional, Callable

# ============================================================================
# CONFIG
# ============================================================================

THREADED_METRICS_ENABLED: bool = True
QUEUE_MAX_DEPTH: int = 4              # if main thread runs ahead, keep latest
WORKER_JOIN_TIMEOUT_S: float = 2.0    # graceful shutdown deadline


# ============================================================================
# Stats
# ============================================================================

class WorkerStats:
    """Lightweight stats so the dashboard / debugger can see how the worker
    is doing — backlog, drop count, mean compute time."""
    __slots__ = ('jobs_processed', 'jobs_dropped', 'last_compute_ms',
                  'mean_compute_ms', 'queue_depth', 'last_error',
                  'last_error_at')

    def __init__(self):
        self.jobs_processed = 0
        self.jobs_dropped = 0
        self.last_compute_ms = 0.0
        self.mean_compute_ms = 0.0
        self.queue_depth = 0
        self.last_error: Optional[str] = None
        self.last_error_at: float = 0.0


# ============================================================================
# Worker
# ============================================================================

class MetricsWorker:
    """Background thread that computes M1-M10 metrics off the main game loop.
    
    Lifecycle:  __init__ → start() → enqueue() many times → stop()
    """

    def __init__(self,
                 compute_fn: Callable,
                 result_writer: Callable,
                 logger_lock: threading.Lock,
                 daemon: bool = True):
        """
        compute_fn       — the function to call per snapshot, e.g.
                            research_probes.compute_all_metrics. Receives
                            **payload and returns a dict.
        result_writer    — a function taking (tick, metrics_dict) that
                            performs the locked time_series append. We
                            inject this so the worker doesn't depend on
                            the logger's internals.
        logger_lock      — shared lock with the logger (used by the writer).
        daemon           — if True, thread won't keep the process alive.
        """
        self._compute_fn = compute_fn
        self._result_writer = result_writer
        self._logger_lock = logger_lock
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=QUEUE_MAX_DEPTH)
        self._thread: Optional[threading.Thread] = None
        self._daemon = daemon
        self._running = threading.Event()
        self.stats = WorkerStats()
        self._compute_history: list = []   # last 32 compute durations for mean
        self._enabled = THREADED_METRICS_ENABLED
        self._paused: bool = False         # v148 (v4.56)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._thread is not None and self._thread.is_alive()

    @property
    def thread_id(self) -> Optional[int]:
        return self._thread.ident if self._thread else None
    
    def set_paused(self, paused: bool):
        """v148 (v4.56) — propagate pause state from the game loop. While
        paused, enqueue() short-circuits to avoid flooding the queue with
        identical snapshots, and the worker loop's queue.get blocks so it
        consumes no CPU. Note: the logger sets data_logger.paused itself,
        which is the primary gate; this is a belt-and-suspenders signal."""
        self._paused = bool(paused)

    # ---- lifecycle ----
    def start(self):
        if not self._enabled:
            print("[MetricsWorker] disabled (THREADED_METRICS_ENABLED=False) — running inline")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run,
                                          name="MetricsWorker",
                                          daemon=self._daemon)
        self._thread.start()
        print(f"[MetricsWorker] started (tid={self._thread.ident}, "
              f"queue depth={QUEUE_MAX_DEPTH})")

    def stop(self, timeout_s: float = WORKER_JOIN_TIMEOUT_S):
        if self._thread is None:
            return
        self._running.clear()
        # poke the queue so the blocking get() returns
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # if queue full, the worker will pick up the existing items and
            # check self._running between iterations
            pass
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            print(f"[MetricsWorker] WARNING: failed to join within {timeout_s}s")
        else:
            print(f"[MetricsWorker] stopped cleanly (processed={self.stats.jobs_processed},"
                  f" dropped={self.stats.jobs_dropped})")
        self._thread = None

    # ---- producer side ----
    def enqueue(self, payload: dict) -> bool:
        """Called from the main game thread. Returns True if the job was
        accepted, False if the queue was full (and we dropped an older item
        to make room — "latest snapshot wins" policy)."""
        if not self.enabled:
            return False
        # v148 — when paused, drop the snapshot at the door. Saves the
        # GIL hand-off and prevents the queue from filling up with
        # identical snapshots.
        if self._paused:
            return False
        try:
            self._queue.put_nowait(payload)
            self.stats.queue_depth = self._queue.qsize()
            return True
        except queue.Full:
            # Drop the oldest pending job and put this fresher one in.
            # We prefer fresh metrics over comprehensive metrics.
            try:
                _ = self._queue.get_nowait()
                self.stats.jobs_dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
                self.stats.queue_depth = self._queue.qsize()
                return True
            except queue.Full:
                self.stats.jobs_dropped += 1
                return False

    def is_idle(self) -> bool:
        """True if the worker has nothing pending. Useful in tests."""
        return self.enabled and self._queue.empty()

    # ---- consumer side ----
    def _run(self):
        """Worker thread main loop."""
        while self._running.is_set():
            try:
                # Block up to 0.25s waiting for a payload. The timeout lets
                # us periodically check self._running for graceful shutdown.
                payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if payload is None:                  # shutdown sentinel
                break
            try:
                t0 = time.perf_counter()
                metrics_dict = self._compute_fn(**payload)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                # Write back. The result_writer (DataLogger._metrics_result_writer)
                # acquires self.metrics_lock itself — we MUST NOT also acquire
                # _logger_lock here because they're the same Lock instance and
                # threading.Lock is non-reentrant → instant deadlock.
                # (Bug fixed in v147 development; was the cause of a 2s join
                # timeout on shutdown and unprocessed queue items.)
                tick = payload.get('tick', -1)
                self._result_writer(tick, metrics_dict)
                # update stats
                self.stats.jobs_processed += 1
                self.stats.last_compute_ms = dt_ms
                self._compute_history.append(dt_ms)
                if len(self._compute_history) > 32:
                    self._compute_history.pop(0)
                self.stats.mean_compute_ms = (
                    sum(self._compute_history) / len(self._compute_history))
                self.stats.queue_depth = self._queue.qsize()
            except BaseException as exc:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                self.stats.last_error_at = time.time()
                # Don't print every exception — that would flood. Keep the
                # most-recent for the dashboard / debug.
                if (self.stats.jobs_processed + self.stats.jobs_dropped) % 50 == 0:
                    print(f"[MetricsWorker] exception in compute: {self.stats.last_error}")
                    traceback.print_exc()
        print("[MetricsWorker] worker loop exiting.")


# ============================================================================
# Single global worker — created lazily by the logger on first use
# ============================================================================

_global_worker: Optional[MetricsWorker] = None
_global_worker_lock = threading.Lock()

def get_worker(compute_fn, result_writer, logger_lock) -> MetricsWorker:
    """Return the singleton worker, creating it on first call. The compute_fn
    and result_writer arguments are only used on the very first call — once
    a worker exists, those bindings are fixed for its lifetime."""
    global _global_worker
    with _global_worker_lock:
        if _global_worker is None or not _global_worker.enabled:
            _global_worker = MetricsWorker(compute_fn, result_writer, logger_lock)
            _global_worker.start()
        return _global_worker

def shutdown_worker(timeout_s: float = WORKER_JOIN_TIMEOUT_S):
    """Stop the singleton worker. Call from the main thread at shutdown."""
    global _global_worker
    with _global_worker_lock:
        if _global_worker is not None:
            _global_worker.stop(timeout_s=timeout_s)
            _global_worker = None
