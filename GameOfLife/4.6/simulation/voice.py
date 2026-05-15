# Neuraxon Game of Life v4.68 Voice/Song System (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 160
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1)
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
"""
Voice / Song Subsystem (NEW v4.5)
---------------------------------
Each NxEr carries a "voice" consisting of:
  * base_freq:    fundamental pitch in Hz (species-wide range)
  * voice_tones:  a 6-element list of semitone offsets (integers, 0..23)
                  relative to the base pitch — the "repertoire" it sings from
  * harmonicity:  a scalar in [0,1] measuring how consonant the 6-tone set is
                  to the human ear (derived, not stored separately)
  * clan_signature: a stable fingerprint (sorted tuple) used for tonal affinity

BIOINSPIRED RATIONALE
---------------------
We use 12-TET (twelve-tone equal temperament) because:
  - It is the reference grid used in every psychoacoustic consonance study
    of the last century (Plomp & Levelt 1965; Kameoka & Kuriyagawa 1969).
  - Integer semitone arithmetic makes the cost per NxEr O(1) — no FFTs,
    no per-frame re-synthesis of the genome itself.
  - Consonance rankings are a static lookup table derived from the
    Helmholtz/Plomp–Levelt dissonance curve evaluated at sensory bandwidth
    ~24% of critical band.

Mating biology:
  Children inherit 4 of their 6 active tones from the combined 12-tone
  parental pool (biased toward the fitter parent), then add 2 new tones
  drawn from the same harmonic pool around a blended base_freq. This
  models cultural song transmission (Mithen 2005) rather than pure
  genetic encoding.

Tone-copy-at-mating:
  Each mating event causes each partner to copy ONE tone they didn't
  already know. This grows individual repertoire slowly and lets
  "musical ideas" propagate through a population — but the active
  6-slot genome remains fixed in size so total compute is bounded.

NO PER-TICK AUDIO SYNTHESIS HAPPENS IN THIS MODULE — this file is pure
math / data structures. The actual sound generation lives in ui/audio.py
and runs only when the audio toggle is ON and the camera is zoomed in.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence

# ============================================================================
# PSYCHOACOUSTIC CONSONANCE TABLE (Plomp & Levelt derived)
# ============================================================================
# Keys: interval size in semitones mod 12 (so "octaves are equivalent").
# Values: consonance score in [0,1]  (1 = perfectly consonant, 0 = maximally
# rough). Numbers are taken from the averaged ranking across Plomp–Levelt,
# Kameoka–Kuriyagawa, and Helmholtz's Sensations of Tone (ch. 10/11).
# This is a static lookup — NOT computed per tick.

_SEMITONE_CONSONANCE = {
    0:  1.00,   # unison
    1:  0.08,   # minor 2nd  (very dissonant)
    2:  0.25,   # major 2nd
    3:  0.70,   # minor 3rd
    4:  0.82,   # major 3rd
    5:  0.85,   # perfect 4th
    6:  0.15,   # tritone    (most dissonant)
    7:  0.95,   # perfect 5th
    8:  0.68,   # minor 6th
    9:  0.80,   # major 6th
    10: 0.45,   # minor 7th
    11: 0.12,   # major 7th
}

# Harmonic-pool: we draw individual tones from this semitone set, weighted
# toward consonant choices. Intentionally spans two octaves (0..23) so we
# get inversion variety (octave + third, octave + fifth, etc.).
_HARMONIC_POOL: List[int] = [0, 3, 4, 5, 7, 9, 12, 15, 16, 17, 19, 21]
_POOL_WEIGHTS: List[float] = [
    _SEMITONE_CONSONANCE[t % 12] for t in _HARMONIC_POOL
]

# Dissonant pool — used rarely (~10% of slots) to allow chromatic colour.
_DISSONANT_POOL: List[int] = [1, 2, 6, 11, 13, 14, 18]

# Tone slot count (kept intentionally small — bounds compute & JSON size)
VOICE_TONES_N: int = 6

# Base frequency range (Hz). 82.4 Hz = E2 (low male voice floor),
# 659 Hz = E5 (soprano high). Keeps all output within audible band
# and below Nyquist for the 22 050 Hz mixer sample rate.
BASE_FREQ_MIN: float = 82.4
BASE_FREQ_MAX: float = 659.0

# Consonance threshold above which a song is labelled "harmonic"
# (used as the song-input trinary +1 branch). Calibrated so that
# a random pair of neighbours has ~30% chance of being "harmonic".
HARMONIC_THRESHOLD: float = 0.55

# Dissonance threshold below which a song is labelled "dissonant"
# (song-input trinary -1 branch).
DISSONANT_THRESHOLD: float = 0.30


# ============================================================================
# VOICE DATACLASS
# ============================================================================

@dataclass
class Voice:
    """Per-NxEr tonal genome + its derived acoustic fingerprint.

    Stored separately from NxEr so it round-trips cleanly via asdict()
    in save/load, and so the audio renderer can treat it as an
    immutable-ish snapshot.
    """
    base_freq: float = 220.0              # Hz (A3)
    voice_tones: List[int] = field(default_factory=list)   # 6 semitone offsets
    # Learned repertoire (tones copied from mates, superset of voice_tones).
    # Bounded to 24 entries — far cheaper than full history.
    repertoire: List[int] = field(default_factory=list)
    harmonicity: float = 0.5              # derived, cached after each change

    def __post_init__(self):
        if not self.voice_tones:
            self.voice_tones = _sample_harmonic_tones(VOICE_TONES_N)
        if not self.repertoire:
            self.repertoire = list(self.voice_tones)
        self.harmonicity = compute_harmonicity(self.voice_tones)

    # ------------------------------------------------------------------
    def signature(self) -> Tuple[int, ...]:
        """Stable, sorted tone set — used for clan fingerprinting and
        O(1) set-intersection affinity. Not mutated by pitch-shift."""
        return tuple(sorted(self.voice_tones))

    def effective_frequencies(self, pitch_shift_semi: float = 0.0) -> List[float]:
        """Return the 6 current sung frequencies in Hz, shifted by
        pitch_shift_semi semitones (e.g. food-elevation or joy spike)."""
        shift = pitch_shift_semi
        f0 = self.base_freq
        return [f0 * (2.0 ** ((t + shift) / 12.0)) for t in self.voice_tones]

    def learn_tone(self, tone: int) -> bool:
        """Add a tone to the lifetime repertoire (NOT to active 6).
        Returns True iff something was added. O(|repertoire|) with
        a hard 24-entry cap."""
        if tone in self.repertoire:
            return False
        if len(self.repertoire) >= 24:
            # Replace a random non-active slot to keep size bounded
            candidates = [t for t in self.repertoire if t not in self.voice_tones]
            if not candidates:
                return False
            self.repertoire.remove(random.choice(candidates))
        self.repertoire.append(int(tone))
        return True

    def to_dict(self) -> dict:
        return {
            "base_freq": float(self.base_freq),
            "voice_tones": [int(t) for t in self.voice_tones],
            "repertoire": [int(t) for t in self.repertoire],
            "harmonicity": float(self.harmonicity),
        }

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Voice":
        if not d:
            return Voice()
        v = Voice(
            base_freq=float(d.get("base_freq", 220.0)),
            voice_tones=[int(t) for t in d.get("voice_tones", [])] or _sample_harmonic_tones(VOICE_TONES_N),
            repertoire=[int(t) for t in d.get("repertoire", [])],
        )
        if not v.repertoire:
            v.repertoire = list(v.voice_tones)
        v.harmonicity = compute_harmonicity(v.voice_tones)
        return v


# ============================================================================
# PURE FUNCTIONS — tone sampling & consonance math
# ============================================================================

def _sample_harmonic_tones(n: int = VOICE_TONES_N) -> List[int]:
    """Draw n tones from the harmonic pool with a small (~10%) chance
    of one dissonant tone for chromatic colour. Unique within the set."""
    pool = list(_HARMONIC_POOL)
    weights = list(_POOL_WEIGHTS)
    picked: List[int] = []
    # Weighted sampling without replacement (small n, manual loop is fast)
    for _ in range(n):
        if not pool:
            break
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for idx, w in enumerate(weights):
            acc += w
            if r <= acc:
                picked.append(pool[idx])
                pool.pop(idx)
                weights.pop(idx)
                break
    # 10% chance of replacing one slot with a dissonant tone
    if random.random() < 0.10 and picked:
        picked[random.randrange(len(picked))] = random.choice(_DISSONANT_POOL)
    # Pad if we ran out (shouldn't happen with pool size 12)
    while len(picked) < n:
        picked.append(random.choice(_HARMONIC_POOL))
    return picked[:n]


def _interval_consonance(a: int, b: int) -> float:
    """Consonance of the interval between two semitone offsets, octave-
    equivalent. O(1) dictionary lookup."""
    d = abs(a - b) % 12
    return _SEMITONE_CONSONANCE.get(d, 0.5)


def compute_harmonicity(tones: Sequence[int]) -> float:
    """Harmonicity = mean pairwise consonance across all C(n,2) pairs.
    Reduces to a single scalar in [0,1]. Cheap: ~15 lookups for n=6."""
    tones = list(tones)
    k = len(tones)
    if k < 2:
        return 1.0
    acc = 0.0
    pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            acc += _interval_consonance(tones[i], tones[j])
            pairs += 1
    return acc / pairs if pairs else 1.0


def tone_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """Similarity between two voices in [0,1]. Combines:
      - set overlap  (how many tones they literally share, octave-equiv)
      - mutual consonance of the merged set
    This is the clan-affinity primitive used by the movement bias.
    """
    if not a or not b:
        return 0.0
    sa = {t % 12 for t in a}
    sb = {t % 12 for t in b}
    overlap = len(sa & sb) / max(1, len(sa | sb))
    merged = list(sa | sb)
    cons = compute_harmonicity(merged) if len(merged) > 1 else 1.0
    # 60% set-overlap, 40% consonance of the merged set.
    return 0.6 * overlap + 0.4 * cons


def song_classification(
    listener: Voice,
    nearby_voices: Sequence[Voice],
) -> int:
    """Return a trinary song-input value for the Song sensory channel:
        +1 = a harmonic (pleasant, same-clan-likely) song is nearby
         0 = nothing notable is being sung nearby
        -1 = a dissonant song is nearby (attention grabber, not attractor)
    Compute cost: O(len(nearby_voices)) — intended to receive a PRUNED
    list of at most a handful of voices, never the whole population.
    """
    if not nearby_voices:
        return 0
    # Average blended consonance across the near-field
    total = 0.0
    n = 0
    for v in nearby_voices:
        if v is None:
            continue
        total += tone_similarity(listener.voice_tones, v.voice_tones)
        n += 1
    if n == 0:
        return 0
    avg = total / n
    if avg >= HARMONIC_THRESHOLD:
        return 1
    if avg <= DISSONANT_THRESHOLD:
        return -1
    return 0


# ============================================================================
# INHERITANCE  (used by neuraxon.genetics.Inheritance + first-gen spawn)
# ============================================================================

def inherit_voice(
    father_voice: "Voice",
    mother_voice: "Voice",
    fitter_is_father: bool = True,
) -> "Voice":
    """Produce a child Voice by blending two parents.

    Biology modelled:
      * base_freq: 70/30 blend toward fitter parent + ±5% jitter
      * 4 tones drawn from the combined 12-tone parental pool
        (weighted toward the fitter parent, unique)
      * 2 new tones sampled from the harmonic pool
      * repertoire: union of both parents' repertoires, capped at 24
    """
    # --- base_freq ---
    if fitter_is_father:
        blended = 0.7 * father_voice.base_freq + 0.3 * mother_voice.base_freq
    else:
        blended = 0.7 * mother_voice.base_freq + 0.3 * father_voice.base_freq
    child_base = blended * random.uniform(0.95, 1.05)
    child_base = max(BASE_FREQ_MIN, min(BASE_FREQ_MAX, child_base))

    # --- tone set (4 inherited + 2 novel) ---
    # Combined pool with weights biased toward the fitter parent
    pool_father = list(father_voice.voice_tones)
    pool_mother = list(mother_voice.voice_tones)
    fitter_pool = pool_father if fitter_is_father else pool_mother
    weaker_pool = pool_mother if fitter_is_father else pool_father
    combined = fitter_pool * 2 + weaker_pool  # bias 2:1 toward fitter

    picked: List[int] = []
    seen: set = set()
    # Shuffle once; take uniques until we have 4 (or pool exhausted)
    random.shuffle(combined)
    for t in combined:
        key = t % 12  # octave-equivalent uniqueness to avoid duplicates at diff octaves
        if key in seen:
            continue
        seen.add(key)
        picked.append(int(t))
        if len(picked) >= 4:
            break

    # Fill remaining two slots with fresh harmonic draws
    while len(picked) < VOICE_TONES_N:
        cand = random.choice(_HARMONIC_POOL)
        if (cand % 12) in seen and len(seen) < 12:
            continue
        seen.add(cand % 12)
        picked.append(cand)

    picked = picked[:VOICE_TONES_N]

    # --- repertoire = union, capped ---
    rep = list(set(father_voice.repertoire) | set(mother_voice.repertoire) | set(picked))
    if len(rep) > 24:
        rep = random.sample(rep, 24)

    child = Voice(
        base_freq=child_base,
        voice_tones=picked,
        repertoire=rep,
    )
    return child


def exchange_tone_on_mate(voice_a: Voice, voice_b: Voice) -> None:
    """Each partner learns ONE tone the other knows but they don't.
    Mutates voices in place. O(1) effective cost."""
    # a learns from b
    b_only = [t for t in voice_b.voice_tones if t not in voice_a.repertoire]
    if b_only:
        voice_a.learn_tone(random.choice(b_only))
    # b learns from a
    a_only = [t for t in voice_a.voice_tones if t not in voice_b.repertoire]
    if a_only:
        voice_b.learn_tone(random.choice(a_only))


# ============================================================================
# PITCH-SHIFT HELPERS  (food, discovery)
# ============================================================================

def food_pitch_shift(food: float, start_food: float) -> float:
    """Well-fed NxErs sing HIGHER. Returns a semitone shift in [0, +3].
    BIOINSPIRED: many bird species pitch-modulate begging/courtship
    calls with body condition — high-energy juveniles sing at the top
    of their range (Searcy & Nowicki 2005)."""
    if start_food <= 0:
        return 0.0
    frac = max(0.0, min(2.0, food / start_food))
    # Map food fraction [0, 1.5] linearly to semitone shift [0, +3]
    return 3.0 * min(1.0, frac / 1.5)


def discovery_pitch_spike(ticks_since_discovery: int, duration_ticks: int = 30) -> float:
    """Transient +2 semitone spike during the first 30 ticks after
    discovering a brand-new food source. Linear decay."""
    if ticks_since_discovery < 0 or ticks_since_discovery >= duration_ticks:
        return 0.0
    decay = 1.0 - (ticks_since_discovery / float(duration_ticks))
    return 2.0 * decay
