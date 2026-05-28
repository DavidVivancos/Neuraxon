# Multi Neuraxon Game of Life 5 — unique name allocator  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# Names are A, B, ... Z, AA, AB, ... never repeated for the lifetime of
# a WORLD. A "restart" (crash recovery) restores the counter from the
# snapshot so names continue without collision. A "reboot" (admin: new
# world) starts the counter from 0 again — fresh name space.
# ===================================================================


class NameAllocator:
    def __init__(self, start=0):
        self._n = int(start)
        import threading
        self._lk = threading.Lock()

    def _name_for(self, idx):
        # 0->A .. 25->Z, 26->AA, 27->AB ... (bijective base-26)
        s = ""
        idx += 1
        while idx > 0:
            idx, r = divmod(idx - 1, 26)
            s = chr(65 + r) + s
        return s

    def next_name(self):
        with self._lk:                 # thread-safe: called from the
            nm = self._name_for(self._n)   # web thread (register) and
            self._n += 1               # the game loop concurrently
        return nm

    def state(self):
        return {"counter": self._n}

    @classmethod
    def from_state(cls, st):
        return cls(start=int((st or {}).get("counter", 0)))
