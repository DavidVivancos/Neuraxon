"""
MultiNxonArcAGI3 — built-in OFFLINE ARC-AGI-3-style games.

These are NOT the proprietary ARC-AGI-3 games.  They are small, original,
grid-based puzzle environments that implement the *exact same interface* as
the official local engine (see ``arc_env.structs``), so the Neuraxon brain,
the play loop, and the NAS all run end-to-end on any machine — no API key,
no proprietary engine, no GPU.  When the real ``arc_agi`` engine is
installed, ``arc_env.adapter`` transparently uses the official games instead
and this module is simply not used.

Each game honours the official contract:
  * observation = ``FrameData`` with a 3D ``frame`` (one HxW colour grid),
    ``state``, ``score`` (0..254), ``available_actions``, ``levels_completed``
  * actions     = ACTION1..ACTION7 (+ RESET); ACTION6 carries (x, y) in 0..63
  * the 0..63 click space is mapped onto the (possibly smaller) grid by
    integer scaling, so the 64x64 ACTION6 contract is preserved.

Games
-----
reach   — navigate a cursor (ACTION1..4 = up/down/left/right) to a goal cell.
          Exercises directional actions + relative-position perception.
switch  — click (ACTION6) the single lit button among distractors.
          Exercises the complex click action + cursor steering + perception.
match   — toggle (ACTION6) bottom-canvas cells to match a top target pattern.
          Exercises construction + comparison; dense score gradient.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .structs import (
    FrameData, GameState, GameAction, EnvironmentInfo,
    empty_grid, grid_equal, MAX_SCORE,
)


# --------------------------------------------------------------------------- #
#  Base environment
# --------------------------------------------------------------------------- #
class _OfflineGame:
    """Base class implementing the official ``env`` surface.

    Subclasses implement ``_reset()`` (builds the first frame) and
    ``_apply(action, data)`` (mutates state, returns (delta_score_event)).
    Everything else — FrameData assembly, action space, info, lifecycle —
    is handled here.
    """

    game_id: str = "offline"
    title: str = "Offline Game"
    tags: List[str] = []
    # which actions the game advertises every step (mirrors a fixed real set)
    advertised_actions: List[GameAction] = [
        GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
        GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6,
        GameAction.ACTION7,
    ]
    grid_h: int = 16
    grid_w: int = 16
    levels_to_win: int = 5
    episode_step_budget: int = 600     # GAME_OVER if exceeded (loop also caps)

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._grid: List[List[int]] = empty_grid(self.grid_h, self.grid_w)
        self._score: int = 0
        self._state: GameState = GameState.NOT_PLAYED
        self._levels: int = 0
        self._steps: int = 0

    # ---- coordinate mapping: official 0..63 click space -> grid cell ------
    def _click_to_cell(self, x: int, y: int) -> Tuple[int, int]:
        x = max(0, min(63, int(x)))
        y = max(0, min(63, int(y)))
        col = (x * self.grid_w) // 64
        row = (y * self.grid_h) // 64
        col = max(0, min(self.grid_w - 1, col))
        row = max(0, min(self.grid_h - 1, row))
        return row, col

    # ---- public env API (matches arc_agi EnvironmentWrapper) --------------
    def reset(self) -> FrameData:
        self.rng = random.Random(self.rng.randint(0, 2 ** 31 - 1))
        self._score = 0
        self._levels = 0
        self._steps = 0
        self._state = GameState.NOT_FINISHED
        self._reset()
        return self._frame()

    def step(self, action: GameAction, data: Optional[Dict[str, Any]] = None,
             reasoning: Optional[Any] = None) -> FrameData:
        if self._state in (GameState.NOT_PLAYED,):
            # mirror real engine: must reset first
            return self._frame()
        if action is GameAction.RESET:
            return self.reset()
        if self._state in (GameState.WIN, GameState.GAME_OVER):
            return self._frame()

        self._steps += 1
        self._apply(action, data or {})

        # lifecycle transitions
        if self._levels >= self.levels_to_win:
            self._state = GameState.WIN
        elif self._steps >= self.episode_step_budget:
            self._state = GameState.GAME_OVER
        self._score = max(0, min(MAX_SCORE, int(self._score)))
        return self._frame()

    @property
    def action_space(self) -> List[GameAction]:
        if self._state in (GameState.WIN, GameState.GAME_OVER, GameState.NOT_PLAYED):
            return [GameAction.RESET]
        return list(self.advertised_actions)

    @property
    def observation_space(self) -> FrameData:
        return self._frame()

    @property
    def info(self) -> EnvironmentInfo:
        return EnvironmentInfo(game_id=self.game_id, title=self.title, tags=list(self.tags))

    # ---- frame assembly ---------------------------------------------------
    def _frame(self) -> FrameData:
        return FrameData(
            game_id=self.game_id,
            frame=[[row[:] for row in self._grid]],   # single-layer 3D frame
            state=self._state,
            score=int(self._score),
            available_actions=self.action_space,
            levels_completed=int(self._levels),
            level_progress=float(self.level_progress()),
        )

    # ---- v1.015: exact within-level progress in [0,1] ---------------------
    def level_progress(self) -> float:
        """Fraction of the CURRENT level achieved (0 = just started, ~1 = about
        to complete). Subclasses override with the exact, game-specific measure.
        The default is uninformative so any game without an override is simply
        treated as having no within-level signal."""
        return 0.0

    # ---- to be implemented by subclasses ----------------------------------
    def _reset(self) -> None:
        raise NotImplementedError

    def _apply(self, action: GameAction, data: Dict[str, Any]) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Game A: reach  (directional navigation)
# --------------------------------------------------------------------------- #
class ReachGame(_OfflineGame):
    game_id = "off_reach"
    title = "Reach the Goal"
    tags = ["navigation", "offline"]
    grid_h = 20
    grid_w = 20
    levels_to_win = 6
    episode_step_budget = 800

    C_PLAYER = 4
    C_GOAL = 3
    C_WALL = 8
    MAX_REACH = 7        # goal placed within this Manhattan distance of player

    def _place_goal(self) -> None:
        while True:
            gy = self.rng.randint(0, self.grid_h - 1)
            gx = self.rng.randint(0, self.grid_w - 1)
            d = abs(gy - self.py) + abs(gx - self.px)
            if 2 <= d <= self.MAX_REACH and (gy, gx) != (self.py, self.px):
                if self._grid[gy][gx] != self.C_WALL:
                    self.gy, self.gx = gy, gx
                    return

    def _reset(self) -> None:
        self._grid = empty_grid(self.grid_h, self.grid_w)
        # a few random walls
        for _ in range(self.rng.randint(6, 14)):
            wy = self.rng.randint(0, self.grid_h - 1)
            wx = self.rng.randint(0, self.grid_w - 1)
            self._grid[wy][wx] = self.C_WALL
        # player at centre
        self.py, self.px = self.grid_h // 2, self.grid_w // 2
        self._grid[self.py][self.px] = 0
        self._place_goal()
        self._render()

    def _render(self) -> None:
        # keep walls; clear player/goal cells then repaint
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if self._grid[y][x] in (self.C_PLAYER, self.C_GOAL):
                    self._grid[y][x] = 0
        self._grid[self.gy][self.gx] = self.C_GOAL
        self._grid[self.py][self.px] = self.C_PLAYER

    def _apply(self, action: GameAction, data: Dict[str, Any]) -> None:
        dy, dx = 0, 0
        if action is GameAction.ACTION1:      # up
            dy = -1
        elif action is GameAction.ACTION2:    # down
            dy = 1
        elif action is GameAction.ACTION3:    # left
            dx = -1
        elif action is GameAction.ACTION4:    # right
            dx = 1
        else:
            return  # ACTION5/6/7 do nothing in this game
        ny, nx = self.py + dy, self.px + dx
        if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w and self._grid[ny][nx] != self.C_WALL:
            self.py, self.px = ny, nx
        if (self.py, self.px) == (self.gy, self.gx):
            self._score += 12
            self._levels += 1
            self._place_goal()
        self._render()

    def level_progress(self) -> float:
        # exact: 1 - normalised Manhattan distance from player to goal. The
        # player starts up to MAX_REACH away, so a brain that steers the cursor
        # toward the goal climbs smoothly from ~0 to ~1 as it closes in — the
        # within-level gradient that completing the level (distance 0) caps.
        if not hasattr(self, "py") or not hasattr(self, "gy"):
            return 0.0
        d = abs(self.py - self.gy) + abs(self.px - self.gx)
        return max(0.0, 1.0 - d / float(self.MAX_REACH))


# --------------------------------------------------------------------------- #
#  Game B: switch  (complex click action)
# --------------------------------------------------------------------------- #
class SwitchGame(_OfflineGame):
    game_id = "off_switch"
    title = "Press the Lit Button"
    tags = ["click", "offline"]
    grid_h = 16
    grid_w = 16
    levels_to_win = 8
    episode_step_budget = 600

    C_LIT = 6
    C_DIM = 2
    N_BUTTONS = 5

    def _new_buttons(self) -> None:
        self._grid = empty_grid(self.grid_h, self.grid_w)
        cells = set()
        while len(cells) < self.N_BUTTONS:
            cells.add((self.rng.randint(1, self.grid_h - 2),
                       self.rng.randint(1, self.grid_w - 2)))
        self.buttons = list(cells)
        for (by, bx) in self.buttons:
            self._grid[by][bx] = self.C_DIM
        self.lit_idx = self.rng.randrange(len(self.buttons))
        ly, lx = self.buttons[self.lit_idx]
        self._grid[ly][lx] = self.C_LIT
        self._last_click = None        # v1.015: track last click for progress

    def level_progress(self) -> float:
        # Switch is click-to-target: there is no cursor to track, so progress is
        # how close the most recent click landed to the lit button (1 = on it,
        # which completes the level). Before any click, 0.
        last = getattr(self, "_last_click", None)
        if last is None or not hasattr(self, "buttons"):
            return 0.0
        ly, lx = self.buttons[self.lit_idx]
        d = abs(last[0] - ly) + abs(last[1] - lx)
        dmax = self.grid_h + self.grid_w
        return max(0.0, 1.0 - d / float(dmax))

    def _reset(self) -> None:
        self._new_buttons()

    def _apply(self, action: GameAction, data: Dict[str, Any]) -> None:
        if action is not GameAction.ACTION6:
            return  # only clicking matters here
        row, col = self._click_to_cell(data.get("x", 0), data.get("y", 0))
        self._last_click = (row, col)   # v1.015: for within-level progress
        ly, lx = self.buttons[self.lit_idx]
        if (row, col) == (ly, lx):
            self._score += 6
            self._levels += 1
            self._new_buttons()
        # wrong clicks: no reward, no penalty below zero (score is sparse)


# --------------------------------------------------------------------------- #
#  Game C: match  (construction + comparison)
# --------------------------------------------------------------------------- #
class MatchGame(_OfflineGame):
    game_id = "off_match"
    title = "Match the Pattern"
    tags = ["construction", "offline"]
    grid_h = 12          # top half = target, bottom half = canvas
    grid_w = 10
    levels_to_win = 1    # "win" = full match
    episode_step_budget = 700

    C_ON = 5

    def _reset(self) -> None:
        self.half = self.grid_h // 2
        self._grid = empty_grid(self.grid_h, self.grid_w)
        # random target in the TOP half
        self.target = empty_grid(self.half, self.grid_w)
        for y in range(self.half):
            for x in range(self.grid_w):
                if self.rng.random() < 0.45:
                    self.target[y][x] = self.C_ON
        # canvas in the BOTTOM half starts blank
        self.canvas = empty_grid(self.half, self.grid_w)
        self._render()
        self._recount()

    def _render(self) -> None:
        for y in range(self.half):
            for x in range(self.grid_w):
                self._grid[y][x] = self.target[y][x]
                self._grid[self.half + y][x] = self.canvas[y][x]

    def _recount(self) -> None:
        matches = 0
        total = self.half * self.grid_w
        for y in range(self.half):
            for x in range(self.grid_w):
                if self.canvas[y][x] == self.target[y][x]:
                    matches += 1
        self._score = matches               # dense gradient: 0..total
        self._match_frac = matches / float(total) if total else 0.0
        if matches >= total:
            self._levels = self.levels_to_win

    def level_progress(self) -> float:
        # exact: fraction of canvas cells matching the target. Match already has
        # a dense internal score, but exposing it as level_progress puts every
        # game on the SAME [0,1] within-level scale for the NAS.
        return float(getattr(self, "_match_frac", 0.0))

    def _apply(self, action: GameAction, data: Dict[str, Any]) -> None:
        if action is not GameAction.ACTION6:
            return
        row, col = self._click_to_cell(data.get("x", 0), data.get("y", 0))
        if row < self.half:
            return  # clicking the target half does nothing
        cy = row - self.half
        self.canvas[cy][col] = 0 if self.canvas[cy][col] == self.C_ON else self.C_ON
        self._render()
        self._recount()


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
_GAMES = {
    ReachGame.game_id: ReachGame,
    SwitchGame.game_id: SwitchGame,
    MatchGame.game_id: MatchGame,
}

# short aliases so callers can say "reach" / "switch" / "match"
_ALIASES = {
    "reach": ReachGame.game_id,
    "switch": SwitchGame.game_id,
    "match": MatchGame.game_id,
}


def list_offline_games() -> List[str]:
    return list(_GAMES.keys())


def make_offline_env(game_id: str = "off_reach", seed: int = 0) -> _OfflineGame:
    gid = _ALIASES.get(game_id, game_id)
    if gid not in _GAMES:
        raise KeyError(f"Unknown offline game {game_id!r}. "
                       f"Available: {list(_GAMES.keys())} (aliases: {list(_ALIASES.keys())})")
    return _GAMES[gid](seed=seed)
