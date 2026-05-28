# Multi Neuraxon Game of Life 5 — session caps  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# Two independent, admin-tunable ceilings:
#   * max_viewers           — concurrent connected clients (any kind)
#   * max_registered_users  — concurrent authenticated owners
# A token is minted on connect; closing the socket frees the slot.
# ===================================================================
import time
import secrets
import threading


class SessionManager:
    def __init__(self, max_viewers, max_registered):
        self._lock = threading.RLock()
        self.max_viewers = int(max_viewers)
        self.max_registered = int(max_registered)
        self._viewers = {}      # token -> {kind, nxer, ts}
        self._registered = set()  # tokens that are authenticated owners

    def set_caps(self, max_viewers, max_registered):
        with self._lock:
            self.max_viewers = int(max_viewers)
            self.max_registered = int(max_registered)

    def open_viewer(self):
        with self._lock:
            if len(self._viewers) >= self.max_viewers:
                return None
            tok = secrets.token_urlsafe(16)
            self._viewers[tok] = {"kind": "anon", "nxer": None,
                                  "ts": time.time()}
            return tok

    def promote_registered(self, token, nxer_name):
        with self._lock:
            if token not in self._viewers:
                return False
            if (token not in self._registered
                    and len(self._registered) >= self.max_registered):
                return False
            self._viewers[token]["kind"] = "owner"
            self._viewers[token]["nxer"] = nxer_name
            self._registered.add(token)
            return True

    def promote_god(self, token):
        with self._lock:
            if token in self._viewers:
                self._viewers[token]["kind"] = "god"
                return True
            return False

    def get(self, token):
        with self._lock:
            return self._viewers.get(token)

    def close(self, token):
        with self._lock:
            self._viewers.pop(token, None)
            self._registered.discard(token)

    def counts(self):
        with self._lock:
            return {
                "viewers": len(self._viewers),
                "max_viewers": self.max_viewers,
                "registered": len(self._registered),
                "max_registered": self.max_registered,
            }
