# Multi Neuraxon Game of Life 5 — crash-safe JSON persistence  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# Snapshots are written atomically (temp file + os.replace) so a crash
# mid-write can never corrupt the recovery file. Two slots are kept
# (snapshot.json + snapshot.prev.json) so even a corrupt latest can
# fall back one generation.
# ===================================================================
import os
import json
import tempfile


class Persistence:
    def __init__(self, state_dir):
        self.dir = state_dir
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "snapshot.json")
        self.prev = os.path.join(self.dir, "snapshot.prev.json")
        # v1.66 — a run stopped in the middle of a save (V 1.085 ended
        # this way at tick 751,434) leaves a partial tmp*.tmp behind.
        # It is never read; remove it so they cannot pile up.
        try:
            for fn in os.listdir(self.dir):
                if fn.startswith("tmp") and fn.endswith(".tmp"):
                    try:
                        os.remove(os.path.join(self.dir, fn))
                    except OSError:
                        pass
        except OSError:
            pass

    def save(self, state):
        # v1.66 — write the new snapshot COMPLETELY first, then rotate
        # latest -> prev, then move the new one into place. Before, the
        # rotation came first, so a stop mid-write left no snapshot.json
        # at all (only prev + a partial tmp).
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            try:
                if os.path.exists(self.path):
                    os.replace(self.path, self.prev)
            except OSError:
                pass
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def load(self):
        for p in (self.path, self.prev):
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
        return None

    def clear(self):
        """Used by a REBOOT (fresh world)."""
        for p in (self.path, self.prev):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
