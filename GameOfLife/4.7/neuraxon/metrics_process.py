# Neuraxon Game of Life v.4.79 metrics_process (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 171
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   "Multi-Neuraxon: Emergent Specialization, Modular, Frequency-Gated Neural Dynamics" by David Vivancos & Jose Sanchez
"""
neuraxon/metrics_process.py  (NEW in v148 / v4.56)
===================================================
Process-based variant of the metrics worker (v147's `metrics_worker.py`
runs on a thread, which shares the GIL with the main thread on CPU-bound
work). When `THREADED_METRICS_USE_PROCESS = True` in config, the logger
spawns a child *process* instead — that process runs on a separate
physical core and pays no GIL contention with the main game loop.

Why both flavours
-----------------
The threaded worker is fine for I/O-bound or short bursts. For real
multicore the child must be a separate process. But process IPC requires
**pickling** payloads — and pickling neuron / synapse / NxEr objects is
slow + lossy (live references go through pickle by reference, but our
worker can't access another process's memory). Therefore the process
worker takes a **MetricsSnapshot** (small, all-primitive-fields) instead
of references. The snapshot is built on the main thread once per
analytics tick.

This is a pragmatic compromise:
- Pros: real CPU multicore, true "different physical core" parallelism.
- Cons: snapshot build cost on the main thread (~O(N_neurons) of cheap
  attribute reads, not iteration of synapses), and pickle/unpickle cost
  per tick (~few KB).

For populations < 30 NxErs the threading worker is faster.  For
populations > 50 the process worker pulls ahead.  Toggle via the config
flag in metrics_worker.py.

API
---
    proc = MetricsProcess(result_writer, logger_lock)
    proc.start()
    proc.enqueue(snapshot)        # snapshot is a MetricsSnapshot dataclass
    ...
    proc.stop(timeout_s=2.0)
"""
import os
import time
import multiprocessing as mp
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# Config — change to disable the process variant globally
THREADED_METRICS_USE_PROCESS: bool = False  # set to True to use multiprocessing
PROC_QUEUE_MAX_DEPTH: int = 4
PROC_JOIN_TIMEOUT_S: float = 3.0


# ============================================================================
# MetricsSnapshot — a small, picklable bundle of just the fields the metrics
# computation needs. Built on the main thread; consumed in the worker process.
# ============================================================================

@dataclass
class _NeuronFields:
    """Per-neuron fields the metrics need. Only primitives — no references."""
    trinary_state: int          # M1
    is_active: bool             # M5/M10
    intrinsic_timescale: float  # M6
    
@dataclass
class _NxErFields:
    """Per-NxEr fields the metrics need."""
    id: int
    alive: bool
    branching_ratio: float
    fitness_score: float
    last_inputs: Tuple[int, ...]  # inputs[4] is sight, inputs[9] is song (for M8/M9)
    last_outputs: Tuple[int, ...]
    motor_amp_proxy: float        # mean abs(trinary) over output port neurons
    dead_neuron_fraction: float   # for M10 lesion bins
    neurons: List[_NeuronFields]  # per-neuron fields (for M1, M6 ACW, M10 lesion)
    has_brain: bool

@dataclass
class MetricsSnapshot:
    """The complete payload sent to the worker process every analytics tick."""
    tick: int
    nxers: List[_NxErFields]
    sample_oscillator_low: float
    sample_oscillator_mid: float
    sample_oscillator_high: float
    sample_phases: Tuple[float, float, float]
    spont_count: int
    driven_count: int
    weight_means: Dict[str, float]


def build_snapshot(tick: int,
                    alive_nxers: list,
                    sample_oscillator_low: float,
                    sample_oscillator_mid: float,
                    sample_oscillator_high: float,
                    sample_phases: Tuple[float, float, float],
                    spont_count: int, driven_count: int,
                    weight_means: Dict[str, float]) -> MetricsSnapshot:
    """Build a MetricsSnapshot from live game state on the main thread.
    
    Cheap — O(N_neurons) of attribute reads, no synapse iteration. The
    expensive metric computation happens in the worker process from this
    snapshot alone.
    """
    nx_fields = []
    for a in alive_nxers:
        if not getattr(a, 'alive', False):
            continue
        net = getattr(a, 'net', None)
        if net is None or not getattr(net, 'all_neurons', None):
            continue
        all_n = net.all_neurons
        n_total = len(all_n)
        n_dead = sum(1 for n in all_n if not n.is_active)
        # Per-neuron fields (only the primitives metrics need)
        ns = [_NeuronFields(
                trinary_state=int(n.trinary_state),
                is_active=bool(n.is_active),
                intrinsic_timescale=float(getattr(n, 'intrinsic_timescale', 0.0)),
              ) for n in all_n]
        # Motor amplitude proxy
        out_neurons = [n for n in net.output_neurons if n.is_active]
        motor_amp = (sum(abs(n.trinary_state) for n in out_neurons) /
                     len(out_neurons)) if out_neurons else 0.0
        nx_fields.append(_NxErFields(
            id=int(a.id),
            alive=True,
            branching_ratio=float(getattr(net, 'branching_ratio', 0.0)),
            fitness_score=float(a.stats.fitness_score) if getattr(a, 'stats', None) else 0.0,
            last_inputs=tuple(int(round(v)) for v in (getattr(a, 'last_inputs', ()) or ())),
            last_outputs=tuple(int(round(v)) for v in (getattr(a, 'last_outputs', ()) or ())),
            motor_amp_proxy=motor_amp,
            dead_neuron_fraction=(n_dead / n_total) if n_total > 0 else 0.0,
            neurons=ns,
            has_brain=getattr(a, 'brain', None) is not None,
        ))
    return MetricsSnapshot(
        tick=int(tick),
        nxers=nx_fields,
        sample_oscillator_low=float(sample_oscillator_low),
        sample_oscillator_mid=float(sample_oscillator_mid),
        sample_oscillator_high=float(sample_oscillator_high),
        sample_phases=tuple(float(x) for x in sample_phases),
        spont_count=int(spont_count),
        driven_count=int(driven_count),
        weight_means={k: float(v) for k, v in (weight_means or {}).items()},
    )


# ============================================================================
# Compute function — top-level (must be picklable for spawn)
# ============================================================================

def compute_from_snapshot(snapshot: MetricsSnapshot,
                           probe_state_dict: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Worker-process compute path. Returns (metrics_dict, updated_probe_state).
    
    Re-implements the cheaper subset of M1/M5/M6/M9/M10 from the snapshot
    alone. Heavier metrics (M2, M3 with sliding windows) are skipped for
    the process path — the threaded worker covers those in the main process.
    
    The probe_state_dict carries the small sliding-window state across
    process invocations (we cannot share Python objects across processes,
    so we serialise + restore between calls)."""
    import math
    out: Dict[str, float] = {}
    
    # ---- M1 trinary distribution ----
    all_neurons = [n for nx in snapshot.nxers for n in nx.neurons if n.is_active]
    n = len(all_neurons)
    if n > 0:
        e = sum(1 for x in all_neurons if x.trinary_state == 1) / n
        i = sum(1 for x in all_neurons if x.trinary_state == -1) / n
        nt = sum(1 for x in all_neurons if x.trinary_state == 0) / n
        out['M1_excitatory_fraction'] = e
        out['M1_inhibitory_fraction'] = i
        out['M1_neutral_fraction'] = nt
        out['M1_deviation_from_target'] = abs(e - 0.22) + abs(i - 0.10) + abs(nt - 0.68)
    
    # ---- M5 branching ratio ----
    ratios = [nx.branching_ratio for nx in snapshot.nxers if nx.branching_ratio > 0]
    if ratios:
        m = sum(ratios) / len(ratios)
        out['M5_branching_ratio'] = m
        out['M5_distance_from_critical'] = abs(m - 1.0)
    
    # ---- M6 spontaneous ----
    total_fires = max(1, snapshot.spont_count + snapshot.driven_count)
    out['M6_spontaneous_fraction'] = snapshot.spont_count / total_fires
    out['M6_driven_fraction'] = snapshot.driven_count / total_fires
    timescales = [n.intrinsic_timescale for n in all_neurons if n.intrinsic_timescale > 0]
    if timescales:
        out['M6_acw_mean'] = sum(timescales) / len(timescales)
        m = out['M6_acw_mean']
        out['M6_acw_heterogeneity'] = math.sqrt(
            sum((t - m) ** 2 for t in timescales) / max(1, len(timescales) - 1))
    
    # ---- M10 lesion curve (compact version) ----
    healthy = []
    bins = {0.25: [], 0.50: [], 0.75: []}
    for nx in snapshot.nxers:
        if nx.dead_neuron_fraction < 0.10:
            healthy.append(nx.motor_amp_proxy)
        else:
            for t in (0.25, 0.50, 0.75):
                if nx.dead_neuron_fraction >= t and nx.dead_neuron_fraction < (t + 0.25 if t < 0.75 else 1.01):
                    bins[t].append(nx.motor_amp_proxy)
                    break
    base = (sum(healthy) / len(healthy)) if healthy else 0.0
    if base < 0.05:
        out['M10_lesion_retention_50'] = 1.0
        out['M10_lesion_retention_75'] = 1.0
    else:
        m50 = bins[0.50]
        out['M10_lesion_retention_50'] = ((sum(m50) / len(m50)) / base) if m50 else 1.0
        m75 = bins[0.75]
        out['M10_lesion_retention_75'] = ((sum(m75) / len(m75)) / base) if m75 else 1.0
    
    # NOTE: M2 (gate dynamics needs link refs), M3 (PAC sliding window), M4
    # (weight history), M7/M8/M9 (per-NxEr stateful probes) are NOT computed
    # in this snapshot mode — they continue to live in the threaded worker
    # because they hold cross-tick state that doesn't survive process IPC
    # gracefully. This is documented; it's the tradeoff for true multicore.
    return out, probe_state_dict


# ============================================================================
# Worker-process top-level entry. Must be defined at module top so spawn
# can pickle and re-import it in the child.
# ============================================================================

def _worker_main(in_queue, out_queue, ready_event):
    """Child process main loop. Pulls snapshots, computes metrics,
    pushes results. Reads None as the shutdown sentinel."""
    ready_event.set()
    probe_state_dict: Dict[str, Any] = {}
    while True:
        try:
            payload = in_queue.get(timeout=0.5)
        except Exception:
            continue
        if payload is None:
            break
        try:
            t0 = time.perf_counter()
            metrics, probe_state_dict = compute_from_snapshot(payload, probe_state_dict)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            out_queue.put({'tick': payload.tick, 'metrics': metrics,
                            'compute_ms': dt_ms})
        except Exception as exc:
            out_queue.put({'tick': payload.tick, 'metrics': {}, 'error': str(exc)})


# ============================================================================
# MetricsProcess — the public face used by the logger
# ============================================================================

class MetricsProcess:
    """Process-based metrics worker. Drop-in replacement for MetricsWorker
    when CPU multicore is needed."""

    def __init__(self, result_writer: Callable, logger_lock: threading.Lock):
        self._result_writer = result_writer
        self._logger_lock = logger_lock
        ctx = mp.get_context('spawn')   # cross-platform safe; slower than fork
        self._in_queue = ctx.Queue(maxsize=PROC_QUEUE_MAX_DEPTH)
        self._out_queue = ctx.Queue(maxsize=64)
        self._ready_event = ctx.Event()
        self._proc: Optional[mp.Process] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_running = threading.Event()
        self._stats = {
            'jobs_processed': 0, 'jobs_dropped': 0,
            'last_compute_ms': 0.0, 'mean_compute_ms': 0.0,
            'queue_depth': 0,
        }
        self._compute_history: list = []
        self._paused: bool = False

    @property
    def enabled(self) -> bool:
        return (THREADED_METRICS_USE_PROCESS and self._proc is not None
                and self._proc.is_alive())

    def set_paused(self, paused: bool):
        self._paused = bool(paused)

    def start(self):
        if not THREADED_METRICS_USE_PROCESS:
            return
        if self._proc is not None and self._proc.is_alive():
            return
        ctx = mp.get_context('spawn')
        self._proc = ctx.Process(target=_worker_main,
                                  args=(self._in_queue, self._out_queue, self._ready_event),
                                  daemon=True, name='MetricsProcess')
        self._proc.start()
        # spawn a small reader thread on the main process to drain results
        # and call the result_writer.  Lives only while process is alive.
        self._reader_running.set()
        self._reader_thread = threading.Thread(target=self._reader_loop,
                                                 daemon=True, name='MetricsProcessReader')
        self._reader_thread.start()
        # Wait for child to signal ready (small timeout — if it doesn't,
        # the reader will just spin and we'll mark process unavailable).
        self._ready_event.wait(timeout=3.0)
        print(f"[MetricsProcess] started (pid={self._proc.pid})")

    def stop(self, timeout_s: float = PROC_JOIN_TIMEOUT_S):
        if self._proc is None:
            return
        # signal shutdown via sentinel
        try:
            self._in_queue.put_nowait(None)
        except Exception:
            pass
        self._reader_running.clear()
        self._proc.join(timeout=timeout_s)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        print(f"[MetricsProcess] stopped (processed={self._stats['jobs_processed']})")
        self._proc = None
        self._reader_thread = None

    def enqueue(self, snapshot: MetricsSnapshot) -> bool:
        if not self.enabled or self._paused:
            return False
        try:
            self._in_queue.put_nowait(snapshot)
            self._stats['queue_depth'] = self._in_queue.qsize()
            return True
        except Exception:
            self._stats['jobs_dropped'] += 1
            return False

    def _reader_loop(self):
        """Pull results from the child and write them back through the
        result_writer. Runs as a daemon thread on the main process."""
        while self._reader_running.is_set():
            try:
                result = self._out_queue.get(timeout=0.25)
            except Exception:
                continue
            try:
                with self._logger_lock:
                    self._result_writer(result.get('tick', -1),
                                          result.get('metrics', {}))
                # update stats
                self._stats['jobs_processed'] += 1
                dt = float(result.get('compute_ms', 0.0))
                self._stats['last_compute_ms'] = dt
                self._compute_history.append(dt)
                if len(self._compute_history) > 32:
                    self._compute_history.pop(0)
                self._stats['mean_compute_ms'] = (
                    sum(self._compute_history) / len(self._compute_history))
                self._stats['queue_depth'] = self._in_queue.qsize()
            except Exception:
                pass

    def get_stats(self) -> dict:
        return {
            'enabled': self.enabled,
            'pid': self._proc.pid if self._proc else None,
            **self._stats,
        }


# ============================================================================
# Singleton accessor
# ============================================================================

_global_proc: Optional[MetricsProcess] = None
_global_proc_lock = threading.Lock()

def get_process_worker(result_writer, logger_lock) -> Optional[MetricsProcess]:
    if not THREADED_METRICS_USE_PROCESS:
        return None
    global _global_proc
    with _global_proc_lock:
        if _global_proc is None or not _global_proc.enabled:
            _global_proc = MetricsProcess(result_writer, logger_lock)
            _global_proc.start()
        return _global_proc

def shutdown_process_worker(timeout_s: float = PROC_JOIN_TIMEOUT_S):
    global _global_proc
    with _global_proc_lock:
        if _global_proc is not None:
            _global_proc.stop(timeout_s=timeout_s)
            _global_proc = None
