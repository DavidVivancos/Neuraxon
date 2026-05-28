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

    def save(self, state):
        # rotate latest -> prev (best-effort)
        try:
            if os.path.exists(self.path):
                os.replace(self.path, self.prev)
        except OSError:
            pass
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
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
