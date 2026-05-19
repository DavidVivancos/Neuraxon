# Neuraxon Game of Life v4.79 Audio Engine (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 171
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
"""
Audio Engine (NEW v4.5)
-----------------------
Renders the ongoing vocalisations of nearby NxErs as inline sound.

Design constraints (explicitly requested):
  * OFF by default; toggle with the M key.
  * Must NOT add measurable overhead when disabled.
  * Scientifically grounded (additive sine synthesis from the same 6-tone
    voice genome used by the behavioural subsystem — not a decorative SFX).
  * Only audible when the camera is zoomed in close — distance attenuation
    tracks the renderer's zoom factor and screen distance to the NxEr.

v4.5 bugfix (channel/sample-rate detection):
  `pygame.init()` in the renderer opens the mixer with pygame's default
  stereo / 44 100 Hz config. Our earlier build synthesised mono buffers at
  22 050 Hz, and `pygame.sndarray.make_sound()` silently rejected the
  mismatched shape (nothing played). This file now reads the mixer's
  actual config at first toggle-on and shapes every buffer to match —
  mono stays mono, stereo gets a duplicated 2-channel interleave.

Key implementation decisions:
  * Hard ceiling on active voices: AUDIO_MAX_VOICES (8).
  * Hard ceiling on NEW voices allocated per frame: AUDIO_MAX_NEW_PER_FRAME (4).
  * If the audio toggle is OFF, update() returns immediately on the
    first branch — zero numpy work, zero channel operations.
"""

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    import numpy as np
except Exception:   # pragma: no cover
    np = None

if TYPE_CHECKING:
    import pygame
    from simulation.entities import NxEr

# ============================================================================
# TUNABLE CONSTANTS  (overhead-critical — keep conservative)
# ============================================================================

AUDIO_BUFFER_SECS: float = 0.7       # Pre-rendered loop length per voice
AUDIO_MAX_VOICES: int = 8            # Hard ceiling on simultaneous channels
AUDIO_MAX_NEW_PER_FRAME: int = 4     # Throttle: bounds numpy work per tick
AUDIO_CACHE_SIZE: int = 64           # LRU cap on pre-rendered waveforms
AUDIO_MIN_ZOOM: float = 6.0          # Below this zoom, audio is silent
AUDIO_MAX_SCREEN_DIST: float = 700.0 # Pixel radius beyond which voice fades out
AUDIO_MASTER_GAIN: float = 0.40      # Global volume cap (0..1)

# Requested mixer config used only when we need to init the mixer ourselves
# (i.e. pygame.init() hasn't already opened it). Kept compatible with pygame's
# own defaults so re-using an already-open mixer is trivially safe.
_PREFERRED_FREQ: int = 44100
_PREFERRED_CHANNELS: int = 2  # stereo — what pygame.init() opens by default


def _quantize_base_freq(f: float) -> int:
    """Quantize base_freq to 1-semitone bins so similar voices share cache."""
    # Convert to semitones from A3 (220 Hz), round to nearest int
    return int(round(12.0 * math.log2(max(1.0, f) / 220.0)))


# ============================================================================
# AUDIO ENGINE
# ============================================================================

class AudioEngine:
    """Renders NxEr voices as additive-sine audio. Bounded compute.

    Public methods:
        toggle()       - flip enabled state (bound to M key)
        is_enabled()   - cheap bool
        update(...)    - per-frame update with a pre-filtered singer list
        stop_all()     - hard stop (called on pause / exit)
    """

    __slots__ = (
        'enabled', '_initialised', '_sound_cache', '_cache_order',
        '_active', '_channel_pool', '_frames_since_gc',
        '_warned_unavailable',
        # v4.5 bugfix: runtime mixer config snapshot + first-play diagnostic
        '_mixer_freq', '_mixer_channels', '_mixer_fmt',
        '_warned_synth_error', '_first_play_logged',
    )

    def __init__(self, enabled: bool = False):
        # Start OFF by default (spec requirement).
        self.enabled: bool = bool(enabled)
        self._initialised: bool = False
        # Cache: key -> pygame.mixer.Sound. _cache_order is an MRU list.
        self._sound_cache: Dict[Tuple, object] = {}
        self._cache_order: List[Tuple] = []
        # Active voices: nxer_id -> {'channel': Channel, 'key': cache_key, 'gain': float}
        self._active: Dict[int, dict] = {}
        self._channel_pool: List[object] = []
        self._frames_since_gc: int = 0
        self._warned_unavailable: bool = False
        # Mixer config filled in by _lazy_init
        self._mixer_freq: int = _PREFERRED_FREQ
        self._mixer_channels: int = _PREFERRED_CHANNELS
        self._mixer_fmt: int = -16
        self._warned_synth_error: bool = False
        self._first_play_logged: bool = False

    # ------------------------------------------------------------------
    def _lazy_init(self):
        """Defer mixer setup until the user actually toggles audio ON.
        Many environments (headless, CI, Linux without ALSA) fail to open
        a mixer — if we init at construction time we'd crash unrelated
        unit tests.

        v4.5 bugfix: this method must also pick up the case where
        ``pygame.init()`` has already opened the mixer — previously we
        skipped the whole block in that branch and left our synthesis
        parameters (22 050 Hz mono) desynced from the actual mixer
        (44 100 Hz stereo), causing every make_sound() call to raise
        silently. Now we ALWAYS read the actual mixer config and store
        it for the synth to match.
        """
        if self._initialised:
            return
        try:
            import pygame
            mix_info = pygame.mixer.get_init()
            if mix_info is None:
                # Truly uninitialised — open it ourselves with pygame-default-
                # compatible settings (stereo / 44.1 kHz).
                try:
                    pygame.mixer.init(
                        frequency=_PREFERRED_FREQ,
                        size=-16,
                        channels=_PREFERRED_CHANNELS,
                        buffer=1024,
                    )
                except pygame.error as e:
                    # Last-resort: let pygame pick whatever it wants.
                    try:
                        pygame.mixer.init()
                    except pygame.error:
                        print(f"[AUDIO] Mixer unavailable; singing disabled. ({e})")
                        self.enabled = False
                        self._warned_unavailable = True
                        return
                mix_info = pygame.mixer.get_init()
            if mix_info is None:
                print("[AUDIO] Mixer still not ready after init; disabling.")
                self.enabled = False
                return
            self._mixer_freq = int(mix_info[0])
            self._mixer_fmt = int(mix_info[1])
            self._mixer_channels = int(mix_info[2])
            try:
                pygame.mixer.set_num_channels(AUDIO_MAX_VOICES)
            except pygame.error:
                pass
            self._initialised = True
            print(
                f"[AUDIO] Mixer ready — {self._mixer_freq} Hz, "
                f"{self._mixer_channels} ch, fmt={self._mixer_fmt}, "
                f"{AUDIO_MAX_VOICES} voice slots. "
                f"Audible when zoom >= {AUDIO_MIN_ZOOM} and NxErs are within "
                f"{int(AUDIO_MAX_SCREEN_DIST)} px of the camera centre."
            )
        except Exception as e:
            if not self._warned_unavailable:
                print(f"[AUDIO] Mixer unavailable; singing disabled. ({e})")
                self._warned_unavailable = True
            self.enabled = False

    # ------------------------------------------------------------------
    def toggle(self) -> bool:
        """Flip enabled state. Returns new state."""
        self.enabled = not self.enabled
        if self.enabled:
            self._lazy_init()
            if not self._initialised:
                self.enabled = False
                return False
            # Reset first-play diagnostic so user sees confirmation each
            # time they toggle audio back on.
            self._first_play_logged = False
        else:
            self.stop_all()
        return self.enabled

    def is_enabled(self) -> bool:
        return self.enabled and self._initialised

    # ------------------------------------------------------------------
    def _synthesize_sound(self, key: Tuple, base_freq: float,
                          tones: Tuple[int, ...], harmonicity: float):
        """Additive-sine synthesis of a single short looping buffer.
        ~6 sine generators over AUDIO_BUFFER_SECS. Runs only when a
        brand-new voice enters the near-field.

        v4.5 bugfix: PCM shape now matches the mixer's actual channel
        count (read in _lazy_init) instead of a hard-coded mono layout.
        """
        if np is None or not self._initialised:
            return None
        try:
            import pygame
            freq = int(self._mixer_freq)
            n_channels = int(self._mixer_channels)
            n_samples = int(freq * AUDIO_BUFFER_SECS)
            if n_samples < 64:
                return None
            t = np.arange(n_samples, dtype=np.float32) / freq
            wave = np.zeros(n_samples, dtype=np.float32)
            # Partial gains decay 1/k — matches the natural harmonic
            # series roll-off of a physical vocal tract
            for i, semi in enumerate(tones):
                f = base_freq * (2.0 ** (semi / 12.0))
                if f <= 0 or f >= freq * 0.45:
                    continue
                amp = 1.0 / (i + 1)
                wave += amp * np.sin(2.0 * np.pi * f * t, dtype=np.float32)
            # Simple attack/release envelope to avoid click artefacts at loop
            fade = min(int(freq * 0.03), n_samples // 4)
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                wave[:fade] *= ramp
                wave[-fade:] *= ramp[::-1]
            # Harmonicity-weighted amplitude — dissonant voices are quieter
            peak = float(np.max(np.abs(wave))) or 1.0
            gain = (0.5 + 0.5 * harmonicity) / peak
            pcm_mono = (wave * gain * 32760.0).astype(np.int16)
            # Match mixer channel count — this is the v4.5 fix.
            if n_channels <= 1:
                pcm = pcm_mono
            elif n_channels == 2:
                # Duplicate L/R, interleaved via column_stack -> (N, 2)
                pcm = np.column_stack((pcm_mono, pcm_mono))
                pcm = np.ascontiguousarray(pcm)
            else:
                # Surround (rare): broadcast mono across all channels
                pcm = np.tile(pcm_mono[:, None], (1, n_channels)).astype(np.int16)
                pcm = np.ascontiguousarray(pcm)
            sound = pygame.sndarray.make_sound(pcm)
            return sound
        except Exception as e:
            if not self._warned_synth_error:
                print(
                    f"[AUDIO] Synthesis error (first occurrence, silencing further "
                    f"messages): {type(e).__name__}: {e}. "
                    f"mixer_channels={getattr(self, '_mixer_channels', '?')}, "
                    f"mixer_freq={getattr(self, '_mixer_freq', '?')}"
                )
                self._warned_synth_error = True
            return None

    # ------------------------------------------------------------------
    def _get_or_create_sound(self, voice, allocations_this_frame: List[int]):
        """Look up or lazily create a pygame.mixer.Sound for a Voice."""
        key = (_quantize_base_freq(voice.base_freq),
               tuple(sorted(voice.voice_tones)))
        sound = self._sound_cache.get(key)
        if sound is not None:
            # Promote in MRU list
            try:
                self._cache_order.remove(key)
            except ValueError:
                pass
            self._cache_order.append(key)
            return sound, key

        # Throttle allocations per frame
        if allocations_this_frame[0] >= AUDIO_MAX_NEW_PER_FRAME:
            return None, None

        sound = self._synthesize_sound(
            key, voice.base_freq,
            tuple(voice.voice_tones), voice.harmonicity,
        )
        if sound is None:
            return None, None

        allocations_this_frame[0] += 1
        self._sound_cache[key] = sound
        self._cache_order.append(key)

        # LRU eviction
        while len(self._cache_order) > AUDIO_CACHE_SIZE:
            evict_key = self._cache_order.pop(0)
            self._sound_cache.pop(evict_key, None)

        return sound, key

    # ------------------------------------------------------------------
    def update(self, singers: List[dict]):
        """Per-frame audio update.

        Args:
            singers: pre-filtered list of dicts with keys:
                id, voice, sing_level (-1/0/+1), screen_dist (pixels),
                zoom (float), pitch_shift_semi (float, for log only here)
                Singers SHOULD already be truncated to ~top-K by the caller.
        """
        if not self.enabled:
            # Cheapest possible early-exit — the spec's "no overhead when off"
            return
        if not self._initialised:
            self._lazy_init()
            if not self._initialised:
                return
        try:
            import pygame
        except Exception:
            self.enabled = False
            return

        allocations_this_frame = [0]
        still_active: Dict[int, dict] = {}

        # Sort by proximity and take up to AUDIO_MAX_VOICES closest
        singers_sorted = sorted(singers, key=lambda s: s['screen_dist'])[:AUDIO_MAX_VOICES]
        active_ids = {s['id'] for s in singers_sorted}

        # Stop channels whose NxEr is no longer in the near-field
        for nid, rec in list(self._active.items()):
            if nid not in active_ids:
                ch = rec.get('channel')
                if ch is not None:
                    try:
                        ch.stop()
                    except Exception:
                        pass
            else:
                still_active[nid] = rec

        # (Re-)start / update each nearby singer
        for s in singers_sorted:
            voice = s.get('voice')
            if voice is None or s.get('sing_level', 0) <= 0:
                # sing_level -1 (silent) or 0 (hum): no playback (hum would
                # still cost a channel, we disable it to stay within budget).
                continue

            # Distance attenuation: 1.0 at centre, 0.0 at MAX_SCREEN_DIST
            dist = float(s.get('screen_dist', 0.0))
            zoom = float(s.get('zoom', 1.0))
            dist_gain = max(0.0, 1.0 - dist / AUDIO_MAX_SCREEN_DIST)
            # Zoom gain: below AUDIO_MIN_ZOOM hard-cut; above, smooth ramp
            # starting at 0.3 so voices are audible the moment you clear
            # the threshold (the old ramp started at 0 exactly at threshold
            # and needed another full doubling of zoom before becoming loud).
            if zoom < AUDIO_MIN_ZOOM:
                zoom_gain = 0.0
            else:
                zoom_gain = min(1.0, 0.3 + 0.7 * (zoom - AUDIO_MIN_ZOOM) / AUDIO_MIN_ZOOM)
            # Sing level 1 = full voice, 2 (if ever) = louder
            level_gain = 1.0 if s['sing_level'] == 1 else 0.6
            gain = AUDIO_MASTER_GAIN * dist_gain * zoom_gain * level_gain
            if gain <= 0.01:
                continue

            sound, key = self._get_or_create_sound(voice, allocations_this_frame)
            if sound is None:
                continue

            rec = still_active.get(s['id'])
            try:
                if rec is not None and rec.get('key') == key and rec.get('channel') is not None:
                    # Same voice still playing — just update gain
                    rec['channel'].set_volume(gain)
                else:
                    # Grab or allocate a channel
                    if rec is not None and rec.get('channel') is not None:
                        try:
                            rec['channel'].stop()
                        except Exception:
                            pass
                    ch = pygame.mixer.find_channel(force=False)
                    if ch is None:
                        # All channels busy — skip (budget protection)
                        continue
                    ch.play(sound, loops=-1)
                    ch.set_volume(gain)
                    still_active[s['id']] = {'channel': ch, 'key': key, 'gain': gain}
                    # Friendly first-play confirmation so the user knows audio
                    # is actually running (not just enabled-but-silent).
                    if not self._first_play_logged:
                        self._first_play_logged = True
                        print(
                            f"[AUDIO] Now singing — first voice rendered at "
                            f"zoom={zoom:.1f}, dist={dist:.0f} px, gain={gain:.2f}. "
                            f"{len(singers_sorted)} singer(s) in near-field."
                        )
            except Exception as e:
                if not self._warned_synth_error:
                    print(f"[AUDIO] Playback error: {type(e).__name__}: {e}")
                    self._warned_synth_error = True
                continue

        self._active = still_active

    # ------------------------------------------------------------------
    def stop_all(self):
        """Stop every playing voice (for pause / toggle-off / exit)."""
        if not self._initialised:
            self._active.clear()
            return
        try:
            for rec in self._active.values():
                ch = rec.get('channel')
                if ch is not None:
                    try:
                        ch.stop()
                    except Exception:
                        pass
        finally:
            self._active.clear()
