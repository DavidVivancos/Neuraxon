"""ARC-AGI-3 environment layer for MultiNxonArcAGI3."""
from .structs import (
    FrameData, GameState, GameAction, EnvironmentInfo,
    NUM_COLORS, MAX_GRID, MAX_SCORE, PLAYABLE_ACTIONS,
)
from .adapter import (
    make_env, list_games, real_engine_available,
    real_game_ids, real_env_status,
)

__all__ = [
    "FrameData", "GameState", "GameAction", "EnvironmentInfo",
    "NUM_COLORS", "MAX_GRID", "MAX_SCORE", "PLAYABLE_ACTIONS",
    "make_env", "list_games", "real_engine_available",
    "real_game_ids", "real_env_status",
]
