# Neuraxon Game of Life v.4.68 entities (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 160
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# Type Checking import to avoid circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neuraxon.network import NeuraxonNetwork
    from neuraxon.multisphere import NeuraxonMultiSphere

from .voice import Voice

# ============================================================================
# VERSION CONSTANTS (UPDATED v4.5 Multi-Sphere + Voice)
# ============================================================================
VERSION = "4.68"
NUM_INPUT_NEURONS = 10   # + Song (was 9 in v4.0)
NUM_OUTPUT_NEURONS = 7   # + Sing (was 6 in v4.0)

# ============================================================================
# PROPRIOCEPTRON (NEW v3.0)
# ============================================================================
# BIOINSPIRED: Proprioception is the sense of body position and movement.
# Real organisms use proprioceptive feedback to avoid obstacles and adjust
# movement strategies. This tracks collision history to prevent "stuck" behavior.

@dataclass 
class Proprioceptron:
    """
    Tracks sensory feedback about the NxEr's own body state and collisions.
    BIOINSPIRED: Mimics proprioceptive neurons that detect body position,
    movement, and contact with obstacles.
    """
    rock_hit_history: List[int] = field(default_factory=list)  # Recent headings that hit rocks
    consecutive_blocked: int = 0  # How many consecutive moves were blocked
    last_successful_heading: int = 0  # Last heading that resulted in movement
    total_rock_hits: int = 0  # Lifetime rock collision count
    forced_turn_count: int = 0  # How many times we forced a direction change
    brain_warning_count: int = 0  # How many times the brain received a pre-motor warning
    brain_avoidance_turn_count: int = 0  # Brain changed heading before reflex override
    last_warning_tick: int = -1  # De-duplicate warning counting within the same tick
    successful_move_streak: int = 0  # Consecutive successful moves (NEW v3.1)
    last_move_result: int = 0  # -1=blocked, 0=neutral, 1=success (NEW v3.1)
    
    def record_rock_hit(self, heading: int, memory_size: int = 5):
        """Record a rock collision at the given heading."""
        self.rock_hit_history.append(heading)
        if len(self.rock_hit_history) > memory_size:
            self.rock_hit_history.pop(0)
        self.consecutive_blocked += 1
        self.total_rock_hits += 1
        self.successful_move_streak = 0  # Reset streak on block
        self.last_move_result = -1
    
    def record_successful_move(self, heading: int):
        """Record a successful movement."""
        self.consecutive_blocked = 0
        self.last_successful_heading = heading
        self.successful_move_streak += 1
        self.last_move_result = 1
    
    def should_force_turn(self, current_heading: int, threshold: int = 3) -> bool:
        """
        Determine if we should force a direction change based on collision history.
        BIOINSPIRED: After repeated collisions in the same direction, organisms
        learn to try different approaches (spatial learning, path integration).
        """
        if self.consecutive_blocked >= threshold:
            return True
        # Check if current heading is frequently blocked
        recent_hits_at_heading = sum(1 for h in self.rock_hit_history if h == current_heading)
        return recent_hits_at_heading >= threshold
    
    def should_warn_brain(self, current_heading: int, threshold: int = 3) -> bool:
        """
        v107 FIX: Determine if brain should receive an early warning BEFORE forced turn.
        Returns True one tick before should_force_turn would trigger.
        BIOINSPIRED: Nociceptive/proprioceptive signals reach the brain faster
        than motor correction — the brain gets a "warning" before reflexive override.
        """
        # Warning fires when we're ONE collision away from forced turn threshold
        if self.consecutive_blocked >= max(1, threshold - 1):
            return True
        recent_hits = sum(1 for h in self.rock_hit_history if h == current_heading)
        return recent_hits >= max(1, threshold - 1)
    
    def register_brain_warning(self, tick: int) -> bool:
        """Count at most one warning per tick; returns True if a new warning was counted."""
        if self.last_warning_tick == tick:
            return False
        self.last_warning_tick = tick
        self.brain_warning_count += 1
        return True

    def register_brain_avoidance_turn(self):
        """Record a brain-initiated heading change that happened before reflex override."""
        self.brain_avoidance_turn_count += 1

    def get_proprioception_signal(self, clear_threshold: int = 5) -> int:
        """
        Get trinary proprioception signal for brain input (NEW v3.1).
        BIOINSPIRED: Provides body awareness of movement success/failure.
        
        Returns:
            -1: Repeatedly blocked (obstacle ahead)
             0: Normal (mixed history)
             1: Clear path (consistent successful movement)
        """
        if self.consecutive_blocked >= 2:
            return -1
        if self.successful_move_streak >= clear_threshold:
            return 1
        return 0
    
    def get_suggested_heading(self, current_heading: int, num_directions: int = 8) -> int:
        """
        Suggest a new heading based on collision history.
        BIOINSPIRED: Avoid recently blocked directions, prefer successful ones.
        """
        # Count hits per direction
        hit_counts = {}
        for h in self.rock_hit_history:
            hit_counts[h] = hit_counts.get(h, 0) + 1
        
        # Find least-blocked direction
        candidates = []
        for h in range(num_directions):
            if h != current_heading:
                candidates.append((hit_counts.get(h, 0), h))
        
        candidates.sort(key=lambda x: x[0])
        
        # Prefer directions near the last successful one if available
        if candidates:
            # Add some randomness to avoid deterministic loops
            import random
            if random.random() < 0.3 and len(candidates) > 1:
                return random.choice(candidates[:3])[1]
            return candidates[0][1]
        
        # Fallback: opposite direction
        return (current_heading + 4) % num_directions

@dataclass
class NxErStats:
    """A dataclass to store performance statistics for a single NxEr agent."""
    food_found: float = 0.0
    food_taken: float = 0.0
    explored: int = 0
    time_lived_s: float = 0.0
    mates_performed: int = 0
    energy_efficiency: float = 0.0
    temporal_sync_score: float = 0.0
    fitness_score: float = 0.0

@dataclass
class NxEr:
    """
    Represents a single agent (a "Neuraxon-enabled life form" or NxEr) in the simulation.
    """
    id: int
    name: str
    color: Tuple[int, int, int]
    pos: Tuple[int, int]
    can_land: bool
    can_sea: bool
    net: 'NeuraxonNetwork'
    food: float
    is_male: bool
    alive: bool = True
    born_ts: float = field(default_factory=time.time)
    died_ts: Optional[float] = None
    # UPDATED v4.5: 10 inputs, 7 outputs
    last_inputs: Tuple[float, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    last_outputs: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0)
    ticks_per_action: int = 1
    tick_accum: int = 0
    harvesting: Optional[int] = None
    mating_with: Optional[int] = None
    mating_end_tick: Optional[int] = None
    stats: NxErStats = None
    visited: set = None
    dopamine_boost_ticks: int = 0
    _last_O4: int = 0
    mating_intent_until_tick: int = 0
    parents: Tuple[Optional[str], Optional[str]] = (None, None)  # Parent names
    parent_ids: Tuple[Optional[int], Optional[int]] = (None, None)  # Stable parent IDs for lineage metrics
    ancestors: List[str] = field(default_factory=list)  # Full lineage (ancestor names)
    rounds_survived: int = 0  # How many rounds this NxEr has survived
    clan_id: Optional[int] = None  # Clan Heritage ID
    mate_cooldown_until_tick: int = 0
    last_move_tick: int = 0
    last_pos: Tuple[int, int] = (0, 0)
    
    # --- NEW PARAMETERS in V2.0---
    vision_range: int = 5     # Random 2 to 15
    smell_radius: int = 3     # Random 2 to 5
    heading: int = 0          # 0=NW, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W
    
    # --- NEW v3.0: Circadian & Temperature ---
    body_temperature: float = 37.0  # Current body temperature
    circadian_phase: float = 0.0    # Current position in day/night cycle [0, 1)
    is_resting: bool = False        # Whether NxEr is in rest/sleep mode
    last_activity_tick: int = 0     # For activity-based temperature
    
    # --- Proprioceptron (UPDATED v3.1) ---
    proprioceptron: Proprioceptron = field(default_factory=Proprioceptron)
    brain_movement_weight: float = 0.5  # How much brain outputs influence movement vs instinct
    _consecutive_successful_moves: int = 0  # Track for proprioception input (NEW v3.1)

    # v3.2: Inherited metabolic preferences
    temperature_tolerance_cold: float = 35.5  # Individual cold threshold
    temperature_tolerance_hot: float = 38.5   # Individual hot threshold
    resting_metabolism_multiplier: float = 0.3  # Individual resting efficiency

    # --- Neuraxon v2.0 per-NxEr state ---
    receptor_activations: Dict = field(default_factory=dict)   # Last receptor activation snapshot
    astrocyte_activity: float = 0.0    # Global astrocyte-like state

    # --- Multi-Sphere Architecture (Paper Sections 7-8) ---
    # The `brain` holds the full NeuraxonMultiSphere graph.
    # The `net` field remains as the primary/motor network for backward compat.
    # When multisphere is enabled, `net` points to the motor sphere's network.
    brain: Optional['NeuraxonMultiSphere'] = None  # Multi-sphere brain (None = single-net mode)
    brain_topology: str = "sensory_association_motor"  # Brain topology identifier

    # --- NEW v4.5: Voice / Song / Hearing ---
    # Per-NxEr tonal genome. See simulation/voice.py for the consonance model.
    voice: Voice = field(default_factory=Voice)
    last_new_food_tick: int = -9999          # Tick of most recent brand-new food source hit — drives discovery pitch spike
    last_sing_level: int = 0                 # Cached O[6] output (-1 silent, 0 hum, +1 full voice)
    known_food_ids: set = field(default_factory=set)  # Food sources this NxEr has ever eaten from — used for "first-time" novelty detection
    
    @property
    def is_multisphere(self) -> bool:
        """Check if this NxEr uses a multi-sphere brain."""
        return self.brain is not None

    @property 
    def sensory_net(self) -> Optional['NeuraxonNetwork']:
        """Get the sensory sphere's network (if multi-sphere)."""
        if self.brain and 'sensory' in self.brain.spheres:
            return self.brain.spheres['sensory'].network
        return None

    @property
    def association_net(self) -> Optional['NeuraxonNetwork']:
        """Get the association sphere's network (if multi-sphere)."""
        if self.brain and 'association' in self.brain.spheres:
            return self.brain.spheres['association'].network
        return None

@dataclass
class Food:
    """Represents a source of food in the world."""
    id: int
    anchor: Tuple[int, int] # The original spawn location, used for respawning nearby.
    pos: Tuple[int, int] # The current location.
    alive: bool = True
    respawn_at_tick: Optional[int] = None # The simulation tick at which this food will respawn.
    remaining: int = 25 # How much food is left at this source.
    progress: Dict[int, int] = field(default_factory=dict) # Tracks harvesting progress by different NxErs.
