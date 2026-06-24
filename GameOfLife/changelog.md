# Neuraxon Game of Life Researh Version<br>
CHANGE LOG:<br>

June 24th 2026:<br>
Neuraxon Game of Life — v5.10 / Internal version 196 , Evolutionary NAS improvements

June 19th 2026:<br>
Neuraxon Live v  v1.52 / GoL Server V 1.072** -> NxonKaleido Brain viz update

June 10th 2026:<br>
Neuraxon Game of Life v.5.10 (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 195 ( serveral updates NAS & Trinary  expanded ) + New Brain base model

May 30th 2026:<br>
Multi Neuraxon Game of Life 5 — client / server ersion 5-v1.29 / GoL Server V 1.048 > Including Editor, Extend world map upto 1200x 2400 raw , + new brain base model based on latest Game Of Life research v192

May 28th 2026:<br>
First Release of Nxon Live = Nxon GoL Server + Client

May 25th 2026:<br>
Game of Life Research v5.0 / Internal version 185 -> g intelligence factor introduced

May 19th 2026:<br>
Neuraxon Game of Life — v4.79 / Internal version 171 -> Survival normalization fix, State 0 buffer refractory 

May 18th 2026:<br>
Neuraxon Game of Life — v4.73 / Internal version 165 -> New Evolutionary strategies

May 15th 2026:<br>
Neuraxon Game of Life — v4.68 / Internal version 160 -> NAS

May 12th 2026:<br>
Neuraxon Game of Life — v4.63 / Internal version 155 -> Firt External architectures loader

May 11th 2026:<br>
Neuraxon Game of Life v.4.53 Update Login System - First RealTime Panel Release

May 10th 2026:<br>
Neuraxon Game of Life v.4.52 udpate UX optimizations

May 8th 2026:<br>
Neuraxon Game of Life v.4.51 udpate speed optimizations

April 23rd 2026:<br>
Neuraxon Game of Life v.4.5  Multi-Nxon 2.0 (Research Version): For Neuraxon 2.0 Big update <br>
Adds a bio-inspired tonal-voice subsystem on top of the existing Multi-Neuraxon
2.0 architecture. 

### New scientific primitives
- **`simulation/voice.py`** — every NxEr owns a `Voice` genome:
  - `base_freq` (Hz, one per NxEr, 82–659 Hz range)
  - `voice_tones`: 6 semitone offsets drawn from a harmonic pool, weighted by
    Plomp–Levelt / Kameoka–Kuriyagawa consonance rankings (12-TET grid)
  - `harmonicity`: mean pairwise consonance across the 6-tone set, in [0,1]
  - `repertoire`: lifetime list of known tones (capped at 24), grown at mating
- Children inherit **4 of 6** tones from the combined 2:1-fitness-weighted
  parental pool + **2 novel** harmonic draws; `base_freq` is a 70/30 blend
  toward the fitter parent with ±5 % jitter.
- Each successful mating also triggers a mutual **one-tone exchange** between
  parents so "musical ideas" propagate through a population.

### New neural I/O channels (minimum possible)
- **Input 9 — Song/Hearing** (trinary, routed to the sensory sphere):
  `+1` harmonic song nearby, `0` silent, `-1` dissonant song nearby.
  Classification uses tone-set overlap + merged-set consonance.
- **Output 6 — Sing** (trinary, read from the motor sphere):
  `+1` sing full voice, `0` hum, `-1` silent.
  Brain output is metabolically gated — starving NxErs cannot sing.
- Total: **10 inputs / 7 outputs** (was 9 / 6).

### New behavioural couplings
- Tone-similar singing neighbours exert a **clan-cohesion pull** on idle
  movement (weight × similarity × stochastic nudge).
- When ≥3 harmonic neighbours are within hearing, the listener receives a
  small **serotonin choir-boost** (mood consolidation).
- `base_freq` effectively pitch-shifts **up with food** (0 → +3 semitones)
  and receives a transient **+2 semitone spike** when a brand-new food
  source is discovered (30-tick decay).
- `_is_parent_child` and clan-merging on mating are unchanged.

### Optional audio renderer
- **`ui/audio.py`** — additive-sine synthesis via `pygame.sndarray`,
  22 050 Hz mono, LRU-cached 0.7-second loops keyed on
  `(quantized_base_freq, sorted_tones)` so similar-voiced NxErs share buffers.
- Hard budgets: **8 simultaneous voices**, **4 new syntheses per frame**,
  **64-entry cache**. Inaudible below zoom 8.0 or beyond 450 px from camera.
- **OFF by default**; toggle with **`M`** in-game. First branch of
  `update()` is a no-op early-return, so cost is zero when disabled.

### Save/load, logging, JSON output
- `Voice` and the new `last_new_food_tick`, `last_sing_level`,
  `known_food_ids` fields are persisted in all `_serialize_nxer` outputs,
  all three load paths (`load_state`, `load_nxer_from_file`, champion
  carry-over), and the NxVizer exports.
- Legacy v4.0 (9/6) save files auto-upgrade to (10/7) on load — verified by
  unit test; missing voices are materialised via fresh `Voice()` draws.
- `logger.py` gains four per-NxEr series: `voice_harmonicity`,
  `voice_base_freq`, `sing_level`, `song_input`.
- End-of-game JSON (`GameOfLife()` return value) gains a
  `voice_summary` block: mean/min/max harmonicity, mod-12 tone histogram,
  per-clan mean harmonicity, and the audio-toggle state.

### Controls (in-game)
| Key     | Action                                              |
|---------|-----------------------------------------------------|
| SPACE   | Pause/unpause                                       |
| V       | Toggle world visuals                                |
| **M**   | **Toggle NxEr singing audio (OFF by default)**     |
| S / L   | Save / Load                                         |
| Q / E   | Rotate camera                                       |
| Wheel   | Zoom in/out                                         |
| ESC     | Quit                                                |

### Compute-overhead notes
- `song_classification` scans at most `voice_max_listeners` (default 6)
  grid-bucketed neighbours — O(1) with respect to population size.
- Clan-cohesion bias runs only when the brain's motor output is idle
  (`dx == dy == 0`) and is capped at ~6 similarity checks per NxEr-per-tick.
- Voice genome is 6 integers + 24 integers max; serialisation is tiny.
- Audio engine never runs numpy code when disabled; mixer only initialised
  on first toggle-on.

March 26th 2026:<br>
Neuraxon Game of Life v.4.0  Multi-Nxon 2.0 (Research Version): For Neuraxon 2.0 <br>
March 17th 2026:<br>
Neuraxon Game of Life v.3.5  Headless (Research Version): For Neuraxon 2.0 (No Game GUI)<br>
Used to capture the dataset: https://huggingface.co/datasets/DavidVivancos/Neuraxon2LifeTS<br>
March 13th 2026:<br>
V 2.5 Versions for Active & Passive Binary comparisons needed for the paper The Neutral Buffer State: Trinary Logic Advantage in Branching Ratio Stability for Continuous-Time Networks<br>
March 11th 2026: v3.34 & v.3.35 & v 3.5 <br>
v 3.5 (Neuraxon 2.0 Compliant) Internal version 104<br>
v.3.34 > movement bias fix <br>
v.3.35 > fix for Neuron Health Unbounded Collapse  <br>

February 13th 2026: v3.32 & v3.33<br>

-> enzymatic clearance + autoreceptor feedback for neuromodulator homeostasis<br>
-> fixed temperature -> circadian correlation & active_to_silent transitions<br>
February 12th 2026: v3.31<br>

-> added meta_influence_gain & meta_da_boost & save_state() fixes<br>
February 10th 2026: v3.3<br>

-> Core plasticity rewrite LTP &LTD Fast/slow differentiation & Body/circadian fixes<br>
February 2nd 2026: v3.2<br>

-> Several updates for the new senses & circadian cycle<br>
January 31st 2026: v3.01<br>

-> Updated Input and Output Neurons<br>
-> NUM_INPUT_NEURONS = 9 # Movement, Terrain, TerrainType, Hunger, Sight, Smell, DayNight, Temperature, Proprioception<br>
-> NUM_OUTPUT_NEURONS = 6 # MoveX, MoveY, Social, MateIntent, GiveFood, Resting<br>

January 29th 2026: v3.0<br>

-> 🌗 Circadian Rhythms: Implemented bio-inspired Day/Night cycles affecting metabolism, hormone levels (Melatonin/Serotonin), and behavior.<br>
-> 🌡️ Thermodynamics System: Added body temperature regulation, environmental heat exchange, and social huddling mechanics.<br>
-> 🧱 Proprioception: Agents now possess "body awareness" to detect collision history and escape movement loops.<br>
-> 🧠 Expanded Architecture: Network updated to 9 Inputs / 6 Outputs to process environmental and somatic data.<br>
-> 📁 Modular Refactoring: Codebase restructured into a scalable Python package format.<br>

Jan 29th 2026:<br>
-> v.3.0 bio-Physical Coupling: Circadian Rhythms, Thermodynamics & Proprioceptionts<br>
Jan 26th 2026:<br>
-> v.2.5 Updated new structure and several other improvements<br>
Jan 24th 2026:<br>
-> v2.38: Biologically Plausible Spontaneous/Driven Activity Ratio<br>
-> v2.37: E/I Balance Fix <br>
Jan 23rd 2026:<br>
-> v2.36: Bioinspired Trinary State Rebalancing<br>
-> v2.35: Properly caps intrinsic timescale<br>
Jan 22nd 2026:<br>
-> v2.34: Inherit Synaptic Weights update<br>
Jan 21st 2026:<br>
-> v2.33: code and performance optimizations<br>
-> v2.32: Autoreceptor Negative Feedback Fix<br>
Jan 20th 2026:<br>
-> v.2.31: Synaptic Weight Homeostasis<br>
-> v.2.30: Energy-Aware Firing Threshold<br>
Jan 17th 2026:<br>
-> v2.29: Global NeuroModulator updates<br>
-> v.2.30 Energy-Aware Firing Threshold <br>
Jan 16th 2026:<br>
-> v2.28: Dopamine Dynamics update <br>
-> v2.27: Serotonin update  <br>
Jan 15th 2026:<br>
-> v2.26: Neuromodulators update<br>
Jan 12th 2026:<br>
-> v2.25: Log Mode 3 enabled for deep detailed timeseries at non agragated Nxer level<br>
Jan 10th 2026:<br>
-> v2.24: Sparse comrpesing some Timeseries data to reduce memory usage and improve performance<br>
Jan 9th 2026:<br>
-> v2.23: God mode disabled and improved biological parameters  <br>
Jan 8th 2026:<br>
-> v2.21: New Nxrs Naming convention for Long Game Tracking Through sesions no more duplicate names in next rounds<br>
-> v2.22: Extra Logging enabled up to 1000s timesteps configurable<br>
Jan 7th 2026 v 2.2 Research update. Enhanced Full Fledged Inheritance <br>
Jan 4th 2026 v 2.1 Research update. and new HF Dataset https://huggingface.co/datasets/DavidVivancos/NeuraxonLife2.1-TimeSeries<br>
