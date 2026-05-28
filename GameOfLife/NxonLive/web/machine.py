# Multi Neuraxon Game of Life 5 — machine resource sampler  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# htop-style CPU + memory for the admin console, with ZERO extra
# dependencies — parses /proc directly (Linux/Ubuntu target).
# CPU% is computed from the delta between two /proc/stat reads, so it
# reflects load since the previous admin poll.
# ===================================================================
import os
import time


class MachineStats:
    def __init__(self):
        self._prev = None          # (total, idle, ts)
        self._ncpu = os.cpu_count() or 1

    def _read_cpu(self):
        try:
            with open("/proc/stat", "r") as f:
                parts = f.readline().split()
            vals = [float(x) for x in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
            total = sum(vals)
            return total, idle
        except Exception:
            return None

    def _read_mem(self):
        info = {}
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    info[k] = float(v.strip().split()[0])  # kB
        except Exception:
            return None
        total = info.get("MemTotal", 0.0)
        avail = info.get("MemAvailable",
                         info.get("MemFree", 0.0))
        used = max(0.0, total - avail)
        return {
            "mem_total_mb": round(total / 1024, 1),
            "mem_used_mb": round(used / 1024, 1),
            "mem_pct": round(100.0 * used / total, 1) if total else 0.0,
        }

    def _loadavg(self):
        try:
            with open("/proc/loadavg", "r") as f:
                a = f.read().split()[:3]
            return [float(x) for x in a]
        except Exception:
            return [0.0, 0.0, 0.0]

    def sample(self):
        out = {"cpu_pct": 0.0, "ncpu": self._ncpu,
               "load_avg": self._loadavg(),
               "mem_total_mb": 0.0, "mem_used_mb": 0.0,
               "mem_pct": 0.0}
        cur = self._read_cpu()
        if cur is not None:
            now = time.time()
            if self._prev is not None:
                pt, pi, _ = self._prev
                dt = cur[0] - pt
                di = cur[1] - pi
                if dt > 0:
                    out["cpu_pct"] = round(
                        100.0 * (1.0 - di / dt), 1)
            self._prev = (cur[0], cur[1], now)
        m = self._read_mem()
        if m:
            out.update(m)
        return out
