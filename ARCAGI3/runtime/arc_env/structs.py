"""
MultiNxonArcAGI3 — ARC-AGI-3 compatible data structures.

This module mirrors the public surface of the official ARC-AGI engine
(``arcengine`` / ``arc_agi``) so that the rest of this project can be written
once against a single interface and run unchanged whether it is driving:

  * the REAL local engine          (``arc_agi.Arcade(OperationMode.OFFLINE)``)
  * the built-in offline fallback  (``arc_env.mock_games``)

The official contract (verified against docs.arcprize.org and the
``arc-agi-3`` PyPI package, Dec 2025) is:

    GameState : NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER
    GameAction: RESET, ACTION1..ACTION7
                - ACTION1..5, ACTION7  -> "simple"  (no coordinates)
                - ACTION6              -> "complex" (carries x, y in 0..63)
    FrameData : game_id, frame (3D list: list of 2D colour grids),
                state (GameState), score (int 0..254),
                available_actions (list[GameAction])

The official 64x64 click space (ACTION6) and 0..15 colour palette are
honoured here.  Grids may be SMALLER than 64x64 (ARC grids vary in size);
the encoder treats whatever it receives as the full observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


# ARC palette is 16 colours (0..15); 0 is conventionally the background.
NUM_COLORS = 16
MAX_GRID = 64          # official ACTION6 coordinate space is 0..63
MAX_SCORE = 254        # official score range is 0..254


class GameState(IntEnum):
    """Lifecycle state of a game, matching the official engine."""
    NOT_PLAYED = 0
    NOT_FINISHED = 1
    WIN = 2
    GAME_OVER = 3


class GameAction(IntEnum):
    """Action set matching the official engine.

    ACTION6 is the only *complex* action — it carries (x, y) click
    coordinates in 0..63.  All other actions are *simple*.

    We deliberately do NOT store mutable per-member coordinate state on the
    enum (the official library does, which is convenient for single-threaded
    agents but is a shared-mutable-singleton footgun).  In this project the
    decoder returns an ``(action, data)`` pair instead, and the environment's
    ``step`` takes ``data`` explicitly.
    """
    RESET = 0
    ACTION1 = 1     # simple — conventionally Up / W
    ACTION2 = 2     # simple — conventionally Down / S
    ACTION3 = 3     # simple — conventionally Left / A
    ACTION4 = 4     # simple — conventionally Right / D
    ACTION5 = 5     # simple — conventionally Enter / Space / interact
    ACTION6 = 6     # complex — Click / Point at (x, y)
    ACTION7 = 7     # simple

    # -- helpers mirroring the official GameAction API ----------------------
    def is_simple(self) -> bool:
        return self not in (GameAction.RESET, GameAction.ACTION6)

    def is_complex(self) -> bool:
        return self is GameAction.ACTION6

    @property
    def value_name(self) -> str:
        return self.name


# The actions an agent may emit during normal play (RESET is handled by the
# loop on NOT_PLAYED / GAME_OVER, never chosen as a "move").
PLAYABLE_ACTIONS: List[GameAction] = [
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6,
    GameAction.ACTION7,
]


@dataclass
class FrameData:
    """A single observation, matching the official FrameData fields.

    ``frame`` is a 3D list: a list of 2D colour grids (layers).  Most games
    use a single layer, so ``frame[0]`` is the H x W grid of colour indices.
    """
    game_id: str = ""
    frame: List[List[List[int]]] = field(default_factory=list)
    state: GameState = GameState.NOT_PLAYED
    score: int = 0
    available_actions: List[GameAction] = field(default_factory=list)
    # convenience extras (also present on the official raw frame)
    levels_completed: int = 0
    # v1.015: within-level progress in [0,1] — the fraction of the CURRENT level
    # that has been achieved. The real ARC-AGI-3 engine exposes no within-level
    # score (score only moves at level completion), so this is the gradient the
    # NAS was previously blind to. Offline games compute it EXACTLY (distance to
    # goal / match fraction / etc.); for real games it stays -1.0 (unknown) and
    # the loop derives a game-agnostic proxy from grid dynamics instead.
    level_progress: float = -1.0

    # ---- ergonomic accessors used by the encoder --------------------------
    @property
    def grid(self) -> List[List[int]]:
        """The primary (top) 2D grid, or an empty grid if none."""
        if self.frame and isinstance(self.frame[0], list):
            return self.frame[0]
        return []

    def grid_size(self) -> tuple[int, int]:
        g = self.grid
        if not g:
            return (0, 0)
        return (len(g), len(g[0]) if g[0] else 0)


@dataclass
class EnvironmentInfo:
    """Static metadata about an environment, matching the official ``env.info``."""
    game_id: str = ""
    title: str = ""
    tags: List[str] = field(default_factory=list)


def empty_grid(h: int, w: int, fill: int = 0) -> List[List[int]]:
    return [[int(fill)] * int(w) for _ in range(int(h))]


def grid_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    if a is b:
        return True
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True
