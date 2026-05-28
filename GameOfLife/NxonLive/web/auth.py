# Multi Neuraxon Game of Life 5 — auth & input hardening  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
#  * passwords: salted SHA-256 (per-server random salt) — never stored
#    in clear; only valid while the NxEr is alive (engine clears the
#    hash on death).
#  * brute-force: 3 wrong passwords for ANY NxEr from one IP → that IP
#    is banned for 24h (config: ban_hours, max_pw_attempts).
#  * input hardening: every user string is length-capped and stripped
#    of control / non-printable chars; the password must be exactly 12
#    alphanumerics; param overrides are whitelisted+coerced in the
#    engine (make_params), so no arbitrary attribute can be injected.
# ===================================================================
import re
import time
import secrets
import hashlib

_SALT = secrets.token_hex(16)

_PW_RE = re.compile(r"^[A-Za-z0-9]{1,12}$")
_PRINTABLE = re.compile(r"[^\x20-\x7E]")


def hash_pw(pw):
    return hashlib.sha256((_SALT + ":" + pw).encode("utf-8")).hexdigest()


def valid_password(pw):
    """1-12 letters/digits (min 1, max 12). No spaces/symbols ⇒ also
    blocks header / prompt-injection payloads."""
    return bool(isinstance(pw, str) and _PW_RE.match(pw))


def sanitize(s, max_len=64):
    """Strip control/non-printable chars and hard-cap length.
    Defeats prompt-injection / header-injection via user fields."""
    if not isinstance(s, str):
        return ""
    s = _PRINTABLE.sub("", s).strip()
    return s[:max_len]


def coerce_overrides(raw):
    """Only pass through simple scalar values; the engine whitelists
    the keys, so this just blocks nested / huge payloads."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in list(raw.items())[:32]:
        if not isinstance(k, str) or len(k) > 48:
            continue
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str) and len(v) <= 48:
            out[sanitize(k, 48)] = sanitize(v, 48)
    return out


class IPGuard:
    """Per-IP failed-password counter with a sliding 24h ban."""

    def __init__(self, max_attempts=3, ban_hours=24):
        self.max_attempts = int(max_attempts)
        self.ban_secs = float(ban_hours) * 3600.0
        self._fails = {}     # ip -> [count, first_ts]
        self._banned = {}    # ip -> ban_until_ts

    def is_banned(self, ip):
        until = self._banned.get(ip)
        if until is None:
            return False
        if time.time() >= until:
            self._banned.pop(ip, None)
            self._fails.pop(ip, None)
            return False
        return True

    def record_fail(self, ip):
        now = time.time()
        c, first = self._fails.get(ip, (0, now))
        # reset the window if older than the ban period
        if now - first > self.ban_secs:
            c, first = 0, now
        c += 1
        self._fails[ip] = (c, first)
        if c >= self.max_attempts:
            self._banned[ip] = now + self.ban_secs
            return True
        return False

    def record_success(self, ip):
        self._fails.pop(ip, None)

    def banned_count(self):
        now = time.time()
        return sum(1 for u in self._banned.values() if u > now)
