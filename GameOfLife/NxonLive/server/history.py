# Neuraxon Game of Life v.5.0 (Research Version) — Science history logger
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
#
# v1.34 / GoL Server V 1.054 — persistent SCIENCE history, fully decoupled
# from the crash-recovery snapshot.
#
# Design goals:
#   * NEVER block the engine thread. Producers enqueue dict records; a
#     single daemon thread drains the queue and appends to JSONL files.
#     A bounded queue drops (and counts) records under extreme backlog
#     rather than applying back-pressure to the simulation.
#   * Append-only, line-delimited JSON (one record per line) so the files
#     can be tailed live, parsed incrementally, and never need rewriting.
#   * Size-bounded with rotation so a multi-week run cannot fill the disk.
#
# Streams (all under <state>/history/):
#   timeseries.jsonl  — periodic population sample (the scientific core:
#                       trinary distribution, g, heritable-trait means,
#                       birth/death/mating rates, population).
#   obituaries.jsonl  — one compact record per NxEr death.
#   lineage.jsonl     — one record per birth (id, parents, founder root).
#   provenance.jsonl  — one record per boot / config change (arch + cfg).
#   events.jsonl      — notable world events (booms, crashes, records).
import os
import json
import time
import queue
import threading


class HistoryLogger:
    def __init__(self, base_dir, *, enabled=True, max_mb_per_stream=200,
                 keep_rotations=4, queue_max=20000):
        self.enabled = bool(enabled)
        self.base_dir = base_dir
        self.max_bytes = int(max_mb_per_stream) * 1024 * 1024
        self.keep_rotations = int(keep_rotations)
        self._q = queue.Queue(maxsize=int(queue_max))
        self._stop = threading.Event()
        self._fh = {}            # stream name -> open file handle
        self._sizes = {}         # stream name -> current bytes
        self.dropped = 0         # records dropped under backlog (visible in admin)
        self.written = 0
        self._t = None
        if self.enabled:
            try:
                os.makedirs(self.base_dir, exist_ok=True)
                self._t = threading.Thread(
                    target=self._run, name="history", daemon=True)
                self._t.start()
            except Exception as e:
                print("[History] disabled — could not start:", repr(e))
                self.enabled = False

    # ---- producer API (called from engine / server; never blocks) ----
    def write(self, stream, record):
        if not self.enabled:
            return
        try:
            # attach a wall-clock stamp once, here, cheaply
            if "t_unix" not in record:
                record["t_unix"] = round(time.time(), 2)
            self._q.put_nowait((stream, record))
        except queue.Full:
            self.dropped += 1

    # convenience wrappers (semantic clarity at call sites)
    def sample(self, record):
        self.write("timeseries", record)

    def obituary(self, record):
        self.write("obituaries", record)

    def lineage(self, record):
        self.write("lineage", record)

    def provenance(self, record):
        self.write("provenance", record)

    def event(self, record):
        self.write("events", record)

    def stats(self):
        return {"enabled": self.enabled, "queued": self._q.qsize(),
                "written": self.written, "dropped": self.dropped}

    # ---- background writer thread --------------------------------------
    def _path(self, stream):
        return os.path.join(self.base_dir, stream + ".jsonl")

    def _open(self, stream):
        fh = self._fh.get(stream)
        if fh is None:
            p = self._path(stream)
            try:
                self._sizes[stream] = (os.path.getsize(p)
                                       if os.path.exists(p) else 0)
                fh = open(p, "a", encoding="utf-8")
                self._fh[stream] = fh
            except Exception as e:
                print("[History] open failed for", stream, repr(e))
                return None
        return fh

    def _rotate(self, stream):
        # rename current -> .1, shifting older rotations; drop the oldest
        try:
            fh = self._fh.pop(stream, None)
            if fh:
                fh.close()
            base = self._path(stream)
            oldest = "%s.%d" % (base, self.keep_rotations)
            if os.path.exists(oldest):
                os.remove(oldest)
            for i in range(self.keep_rotations - 1, 0, -1):
                src = "%s.%d" % (base, i)
                if os.path.exists(src):
                    os.replace(src, "%s.%d" % (base, i + 1))
            if os.path.exists(base):
                os.replace(base, "%s.1" % base)
            self._sizes[stream] = 0
        except Exception as e:
            print("[History] rotate failed for", stream, repr(e))

    def _run(self):
        while not self._stop.is_set():
            try:
                stream, record = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            fh = self._open(stream)
            if fh is None:
                continue
            try:
                line = json.dumps(record, separators=(",", ":")) + "\n"
                fh.write(line)
                fh.flush()
                self.written += 1
                self._sizes[stream] = self._sizes.get(stream, 0) + len(line)
                if self._sizes[stream] >= self.max_bytes:
                    self._rotate(stream)
            except Exception as e:
                print("[History] write failed for", stream, repr(e))

    def close(self):
        self._stop.set()
        if self._t is not None:
            try:
                self._t.join(timeout=2.0)
            except Exception:
                pass
        for fh in self._fh.values():
            try:
                fh.close()
            except Exception:
                pass
        self._fh.clear()
