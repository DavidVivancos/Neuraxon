# Neuraxon Game of Life v.4.68 ui renderer (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 160
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import pygame
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import constants and utilities
from config import T_SEA, T_LAND, T_ROCK
from utils import _clamp, _rot

# Type Checking import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation.world import World
    from simulation.entities import NxEr, Food

class Renderer:
    """Handles all Pygame-based rendering and user input for the main simulation window."""
    def __init__(self, world: 'World', textures: Dict[str, Optional[str]], textures_alpha: float):
        pygame.init()
        pygame.display.set_caption("Neuraxon Game of Life v 4.68 (Research Version) - By David Vivancos & Dr Jose Sanchez for Qubic Science")
        self.screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.world = world
        # Camera state variables.
        self.zoom = max(2.0, 800.0 / world.N)
        self.pan = [world.N * 0.5, world.N * 0.5]
        self.rot = 0.0
        self.dt = 1 / 60.0
        self.dragging = False
        self.drag_start = (0, 0)
        self.textures_alpha = _clamp(int(textures_alpha * 255) if textures_alpha <= 1 else int(textures_alpha), 0, 255)
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("consolas", 20, bold=True)
        self._load_textures(textures)
        # Rectangles for clickable UI elements.
        self.button_rects = {}
        self.overlay_buttons = {}
        self.selected_nxer_id: Optional[int] = None # The ID of the currently selected NxEr for the detail view.
        # v157 (v4.65) — auto-follow toggle. When True, draw_world pans the
        # camera each frame so the selected NxEr stays centred. Off by
        # default. Toggled with the F key in game_loop's KEYDOWN handler.
        self.follow_selected: bool = False
        self.detail_buttons: Dict[str, pygame.Rect] = {}
        self.ranking_click_areas: List[Tuple[pygame.Rect, int]] = []
        self.visual_mode = False  # NEW: Visual mode flag, set to off for speed with V Key
        # v4.5: Audio engine reference (set by game_loop). Renderer uses it
        # only to read is_enabled() for the status hint — no synthesis here.
        self.audio_engine = None
        # v146 (v4.54): live metrics dashboard, toggled with the L key.
        # Lazy-imported so the rest of the renderer keeps working even if the
        # dashboard module is broken in some environment.
        try:
            from .dashboard import MetricsDashboard
            self.metrics_dashboard = MetricsDashboard(self.screen)
        except Exception as _e:
            print(f"[RENDERER] MetricsDashboard unavailable: {_e}")
            self.metrics_dashboard = None

        # v4.52 PERF (#HUD-cache): cached name→color map + its signature so we
        # only rebuild the dict when the NxEr population actually changes.
        # The dict comprehension `{a.name: a.color for a in nxers.values()}`
        # was running every single frame (60 Hz) regardless of whether any
        # agent was born/died. For large populations that's the HUD's single
        # biggest cost.
        self._name2color_cache: Dict[str, Tuple[int, int, int]] = {}
        self._name2color_signature: int = -1
        
    def _load_textures(self, tex):
        """Loads optional image files to be used as textures for world elements."""
        def load_one(path):
            if not path or str(path).lower() == "none": return None
            try:
                s = pygame.image.load(path).convert_alpha()
                s.set_alpha(self.textures_alpha)
                return s
            except: return None
        self.tex_land = load_one(tex.get("TextureLand"))
        self.tex_sea = load_one(tex.get("TextureSea"))
        self.tex_rock = load_one(tex.get("TextureRock"))
        self.tex_food = load_one(tex.get("TextureFood"))
        self.tex_nxer = load_one(tex.get("TextureNxEr"))
    
    def world_to_screen(self, x, y):
        """Converts world coordinates to screen coordinates, applying camera pan, zoom, and rotation."""
        cx, cy = self.pan
        dx, dy = (x - cx), (y - cy)
        rx, ry = _rot(dx, dy, self.rot)
        return (int(self.screen.get_width() / 2 + rx * self.zoom), int(self.screen.get_height() / 2 + ry * self.zoom))
    
    def screen_to_world(self, sx, sy) -> Tuple[float, float]:
        """Converts screen coordinates back to world coordinates, reversing the camera transform."""
        cx, cy = self.pan
        rx = (sx - self.screen.get_width() / 2) / self.zoom
        ry = (sy - self.screen.get_height() / 2) / self.zoom
        wx, wy = _rot(rx, ry, -self.rot)
        return (cx + wx, cy + wy)
    
    def _draw_effects(self, effects: List[dict], step_tick: int, GlobalTimeSteps: int):
        """Renders temporary visual effects like hearts for mating or skulls for death."""
        for ef in effects:
            age = step_tick - ef['start']
            if age < 0 or age >= GlobalTimeSteps: continue
            rise_px = int(-40 * (age / max(1, GlobalTimeSteps)))
            sx, sy = self.world_to_screen(ef['pos'][0], ef['pos'][1])
            sy += rise_px
            if ef['kind'] == 'heart':
                r = max(6, int(self.zoom * 0.5))
                pygame.draw.circle(self.screen, (220, 40, 60), (sx - r // 2, sy - r // 4), r // 2)
                pygame.draw.circle(self.screen, (220, 40, 60), (sx + r // 2, sy - r // 4), r // 2)
                pygame.draw.polygon(self.screen, (220, 40, 60), [(sx - r, sy), (sx + r, sy), (sx, sy + r)])
            elif ef['kind'] == 'skull':
                r = max(6, int(self.zoom * 0.45))
                pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), r)
                pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(sx - r // 2, sy, r, r // 2), border_radius=3)
                eye_r = max(2, r // 5)
                pygame.draw.circle(self.screen, (200, 200, 200), (sx - r // 3, sy - r // 4), eye_r)
                pygame.draw.circle(self.screen, (200, 200, 200), (sx + r // 3, sy - r // 4), eye_r)
    
    def _draw_restart_modal(self):
        """Renders the "Game Over" modal dialog with options to restart or quit."""
        self.overlay_buttons = {}
        W, H = self.screen.get_size()
        bw, bh = 200, 48
        rect = pygame.Rect(W // 2 - 260, H // 2 - 140, 520, 280)
        srf = pygame.Surface((W, H), pygame.SRCALPHA)
        srf.fill((0, 0, 0, 160))
        self.screen.blit(srf, (0, 0))
        pygame.draw.rect(self.screen, (15, 15, 18), rect, border_radius=12)
        pygame.draw.rect(self.screen, (90, 90, 100), rect, 2, border_radius=12)
        title = self.big.render("All NxErs have perished.", True, (235, 235, 240))
        subtitle = self.small.render("Restart? (Will do in 10 seconds if no response)", True, (220, 220, 220))
        self.screen.blit(title, (rect.x + (rect.w - title.get_width()) // 2, rect.y + 28))
        self.screen.blit(subtitle, (rect.x + (rect.w - subtitle.get_width()) // 2, rect.y + 60))
        y = rect.y + 130
        yes_rect = pygame.Rect(rect.x + 40, y, bw - 30, bh)
        no_rect = pygame.Rect(rect.x + rect.w - 60 - bw, y, bw - 30, bh)
        for r, lab, key in [(yes_rect, "Yes", "restart_yes"), (no_rect, "No", "restart_no")]:
            pygame.draw.rect(self.screen, (35, 35, 45), r, border_radius=10)
            pygame.draw.rect(self.screen, (110, 110, 130), r, 2, border_radius=10)
            tx = self.big.render(lab, True, (235, 235, 240))
            self.screen.blit(tx, (r.x + (bw - tx.get_width()) // 2, r.y + (bh - tx.get_height()) // 2))
            self.overlay_buttons[key] = r
    
    def _draw_survivability_hud(self, sd: dict):
        """v147 (v4.55) — persistent population-health HUD strip rendered
        in the top-left, just below the audio status. Always visible —
        gives the player an at-a-glance answer to 'are my NxErs OK?'
        without needing to open the L dashboard.
        
        Shows: composite score % with health colour, alive count, motion
        rate, stuck percentage, and births/deaths over the last window.
        Width adapts so it never overlaps the right-side ranking panel.
        """
        # Position: top-left, under the M/V status text
        x = 20
        y = 78 if not self.visual_mode else 44
        # Compute width and content
        score_pct = sd.get('survivability_score', 0.5) * 100.0
        col = sd.get('health_colour', (200, 200, 200))
        label = sd.get('health_label', '...')
        # Compose strings — kept compact
        score_str   = f"survivability {score_pct:5.1f}%  ({label})"
        alive_str   = f"alive: {sd.get('alive_count', 0)}"
        motion_str  = f"moving: {sd.get('mean_motion_rate', 0)*100:.0f}%"
        stuck_str   = f"stuck: {sd.get('stuck_fraction', 0)*100:.0f}%"
        # v149 — N@+1 = neurons stuck at +1. Direct indicator of M1
        # lock-in. Healthy = < 30%; collapsing = > 50%.
        n_at_pos1 = sd.get('stuck_fraction_at_pos1', 0.0)
        n_stuck_str = f"N@+1: {n_at_pos1*100:.0f}%"
        # v150 — S→M corr = correlation between sensory input activity
        # and motor output activity. > 0.20 = healthy responsiveness.
        # Near 0 = NxErs ignoring environment (the v149 sample run
        # pathology that caused population crash 30 → 4).
        sm_corr = sd.get('sensory_motor_corr', 0.0)
        sm_str = f"S→M: {sm_corr:+.2f}"
        bd_str      = f"B/D: {sd.get('births_window', 0)}/{sd.get('deaths_window', 0)}"
        food_str    = f"food: {sd.get('mean_food', 0):.1f}"
        # Background panel
        parts = [score_str, alive_str, motion_str, stuck_str, n_stuck_str, sm_str, bd_str, food_str]
        widths = [self.small.size(p)[0] for p in parts]
        gap = 14
        total_w = sum(widths) + gap * (len(parts) - 1) + 24
        h = 32
        bg = pygame.Surface((total_w, h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, col,
                         pygame.Rect(x, y, total_w, h), 1, border_radius=6)
        # Score block (coloured)
        cur_x = x + 12
        score_surf = self.small.render(score_str, True, col)
        self.screen.blit(score_surf, (cur_x, y + (h - score_surf.get_height()) // 2))
        cur_x += score_surf.get_width() + gap
        # Other blocks (white, with red/green tints on the alarming ones)
        for label_text, status_col in [
            (alive_str,  (220, 220, 220)),
            (motion_str, (90, 215, 130) if sd.get('mean_motion_rate', 0) >= 0.30 else
                          (235, 95, 95) if sd.get('mean_motion_rate', 0) < 0.10 else
                          (235, 200, 90)),
            (stuck_str,  (90, 215, 130) if sd.get('stuck_fraction', 0) < 0.30 else
                          (235, 95, 95) if sd.get('stuck_fraction', 0) >= 0.50 else
                          (235, 200, 90)),
            (n_stuck_str,(90, 215, 130) if n_at_pos1 < 0.30 else
                          (235, 95, 95) if n_at_pos1 >= 0.50 else
                          (235, 200, 90)),
            (sm_str,     (90, 215, 130) if sm_corr >= 0.20 else
                          (235, 95, 95) if sm_corr < 0.05 else
                          (235, 200, 90)),
            (bd_str,     (90, 215, 130) if sd.get('births_window', 0) > sd.get('deaths_window', 0) else
                          (235, 95, 95) if sd.get('births_window', 0) * 2 < sd.get('deaths_window', 0) else
                          (220, 220, 220)),
            (food_str,   (220, 220, 220)),
        ]:
            ts = self.small.render(label_text, True, status_col)
            self.screen.blit(ts, (cur_x, y + (h - ts.get_height()) // 2))
            cur_x += ts.get_width() + gap
    
    def draw_world(self, foods: Dict[int, 'Food'], nxers: Dict[int, 'NxEr'], hud: Dict[str, List[Tuple[str, str]]], alive_count: int, dead_count: int, born_count: int, paused: bool, effects: List[dict], step_tick: int, GlobalTimeSteps: int, game_over: bool, game_index: int, best_scores: Optional[Dict[str, float]] = None):
        """The main rendering function, called once per frame to draw the entire scene."""
        # v157 (v4.65) — auto-follow selected NxEr if follow_selected is on
        # ------------------------------------------------------------------
        # When follow mode is on, snap the camera centre to the selected
        # NxEr's world position before any rendering. Soft-snap (lerp 0.25)
        # so the camera feels less jittery as the NxEr makes small moves.
        # If the selected NxEr died or doesn't exist, follow turns off.
        if self.follow_selected and self.selected_nxer_id is not None:
            a = nxers.get(self.selected_nxer_id)
            if a is not None and a.alive:
                target_x, target_y = a.pos[0] + 0.5, a.pos[1] + 0.5
                # soft snap (25% per frame ≈ exponential ease)
                self.pan[0] += (target_x - self.pan[0]) * 0.25
                self.pan[1] += (target_y - self.pan[1]) * 0.25
            else:
                # Selected NxEr died — disable follow but keep selection so
                # the user sees the death panel
                self.follow_selected = False
        self.screen.fill((0, 0, 0))
        w, h = self.screen.get_size()
        cx, cy = self.pan
        
        # --- Draw World Terrain (ONLY IF VISUAL MODE IS ON) ---
        if self.visual_mode:
            # Calculate the visible portion of the world to avoid drawing off-screen tiles.
            radius = max(w, h) / self.zoom * 1.5
            x0 = int(max(0, cx - radius)); x1 = int(min(self.world.N, cx + radius))
            y0 = int(max(0, cy - radius)); y1 = int(min(self.world.N, cy + radius))
            
            # Use Level of Detail (LOD) to speed up rendering when zoomed out.
            lod = 1
            tile = max(2, int(self.zoom))
            if tile < 4: lod = 3
            elif tile < 2: lod = 6
            
            for y in range(y0, y1, lod):
                for x in range(x0, x1, lod):
                    t = self.world.grid[y][x]
                    base = (40, 180, 60) if t == T_LAND else ((25, 100, 200) if t == T_SEA else (110, 110, 110))
                    height = 0 if t == T_SEA else (2 if t == T_ROCK else 1)
                    c = tuple(_clamp(int(b * (0.85 + 0.08 * height)), 0, 255) for b in base)
                    sx, sy = self.world_to_screen(x, y)
                    if sx < -tile or sx > w + tile or sy < -tile or sy > h + tile: continue
                    pygame.draw.rect(self.screen, c, pygame.Rect(sx, sy, int(self.zoom * lod) + 1, int(self.zoom * lod) + 1))
            
            # --- Draw Objects (Food and NxErs) ---
            for f in foods.values():
                if not f.alive: continue
                sx, sy = self.world_to_screen(f.pos[0], f.pos[1])
                if sx < -50 or sx > w + 50 or sy < -50 or sy > h + 50: continue
                s = max(6, int(self.zoom * 0.8))
                pygame.draw.polygon(self.screen, (220, 40, 40), [(sx, sy - s), (sx - s // 2, sy), (sx + s // 2, sy)])
            for a in nxers.values():
                if not a.alive: continue
                sx, sy = self.world_to_screen(a.pos[0], a.pos[1])
                if sx < -50 or sx > w + 50 or sy < -50 or sy > h + 50: continue
                rad = max(4, int(self.zoom * 0.45))
                pygame.draw.circle(self.screen, a.color, (sx, sy), rad)
                pygame.draw.circle(self.screen, (20, 20, 20), (sx, sy), rad, 1)
                # v157 (v4.65) — gold selection ring around the currently
                # selected NxEr. Pulses slightly so it's visible at any
                # zoom level. The ring is drawn AFTER the body but BEFORE
                # the energy/note glyphs so it sits behind them.
                if a.id == self.selected_nxer_id:
                    pulse = 1.0 + 0.18 * abs(((step_tick % 60) / 30.0) - 1.0)
                    ring_r = max(rad + 4, int(rad * 1.6 * pulse))
                    pygame.draw.circle(self.screen, (255, 215, 0),
                                       (sx, sy), ring_r, 2)
                    if self.follow_selected:
                        # Extra crosshair when actively following
                        pygame.draw.line(self.screen, (255, 215, 0),
                                         (sx - ring_r - 4, sy),
                                         (sx - ring_r + 2, sy), 1)
                        pygame.draw.line(self.screen, (255, 215, 0),
                                         (sx + ring_r - 2, sy),
                                         (sx + ring_r + 4, sy), 1)
                        pygame.draw.line(self.screen, (255, 215, 0),
                                         (sx, sy - ring_r - 4),
                                         (sx, sy - ring_r + 2), 1)
                        pygame.draw.line(self.screen, (255, 215, 0),
                                         (sx, sy + ring_r - 2),
                                         (sx, sy + ring_r + 4), 1)
                # Draw an inner yellow circle representing the agent's energy level.
                if hasattr(a.net, 'get_energy_status'):
                    energy = a.net.get_energy_status().get('average_energy', 0.0)
                    energy_rad = max(2, int(rad * energy / 100.0))
                    pygame.draw.circle(self.screen, (255, 255, 0), (sx, sy), energy_rad, 1)
                # v4.5: Musical-note glyph above any NxEr currently singing.
                # Only drawn at moderate-plus zoom to avoid clutter when zoomed out.
                if self.zoom >= 6.0 and getattr(a, 'last_sing_level', 0) >= 1:
                    note_r = max(2, int(self.zoom * 0.22))
                    nx_x, nx_y = sx + rad, sy - rad - note_r
                    # Harmonic (well-liked by the listener) colour tint
                    harm = getattr(getattr(a, 'voice', None), 'harmonicity', 0.5)
                    col = (60, 220, 120) if harm >= 0.55 else (230, 200, 80)
                    # Filled note-head
                    pygame.draw.ellipse(
                        self.screen, col,
                        pygame.Rect(nx_x - note_r, nx_y - note_r // 2, note_r * 2, note_r),
                    )
                    # Stem
                    pygame.draw.line(
                        self.screen, col,
                        (nx_x + note_r - 1, nx_y),
                        (nx_x + note_r - 1, nx_y - note_r * 3),
                        max(1, note_r // 3),
                    )
            
            self._draw_effects(effects, step_tick, GlobalTimeSteps)
        
        # --- Draw Heads-Up Display (HUD) Side Panel (ALWAYS VISIBLE) ---
        panel_w = 300
        x = self.screen.get_width() - panel_w - 16; y = 12
        rows = 1
        for _, lst in hud.items(): rows += 1 + min(3, len(lst))
        rows += 11
        panel_h = 26 + rows * 18 + 24
        base_rect = pygame.Rect(x - 10, y - 8, panel_w + 20, panel_h-50)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), base_rect, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 60), base_rect, 1, border_radius=8)
        round_text = self.big.render(f"Game Metrics: Round #{game_index}", True, (230, 230, 230))
        self.screen.blit(round_text, (x, y)); y += 28
        
        # Draw Rankings.
        # v4.52 PERF (#HUD-cache): `nxers` dict population changes only on
        # birth/death, which is rare relative to 60-fps frame rate. A cheap
        # length signature catches the common case; renaming/color changes
        # (never) would need a deeper invariant. Colors can't change for an
        # existing NxEr id in this codebase, so length is sufficient.
        _sig = len(nxers)
        if _sig != self._name2color_signature:
            self._name2color_cache = {a.name: a.color for a in nxers.values()}
            self._name2color_signature = _sig
        name2color = self._name2color_cache
        # Ranking click areas must still be rebuilt per frame because the
        # y-coordinates depend on the laid-out HUD.
        self.ranking_click_areas = []
        # v157 (v4.65) — map ranking titles → hotkey numbers (matches the
        # K_1..K_6 handler in game_loop). Shown as small "[N]" hint next
        # to each title so the user discovers them. ORDER matches HUD,
        # which comes from rankings() in game_loop — Food found (2),
        # Food taken (3), World explored (5), Time lived (6), Mates (4),
        # Fitness (1).
        title_to_hotkey = {
            "Food found":     "2",
            "Food taken":     "3",
            "World explored": "5",
            "Time lived (s)": "6",
            "Mates":          "4",
            "Fitness":        "1",
        }
        for title, rows in hud.items():
            display_title = title
            score = best_scores.get(title) if best_scores else None
            if score is not None: display_title = f"{title} ({score:.2f})" if isinstance(score, float) else f"{title} ({int(score)})"
            # v157 — append hotkey hint
            hotkey = title_to_hotkey.get(title)
            if hotkey:
                display_title = f"{display_title}  [{hotkey}]"
            self.screen.blit(self.small.render(display_title, True, (180, 180, 180)), (x, y)); y += 18
            for row_idx, (name, val) in enumerate(rows[:3]):
                base_name = name.replace(" [Die]", "")
                base_name = base_name.split(" [", 1)[0].strip()   #now the round is emmbedd in the name in hud
                dot_c = name2color.get(base_name, (200, 200, 200))
                pygame.draw.circle(self.screen, dot_c, (x + 8, y + 8), 6)
                # v157 — mark top-row champion with a star, plus highlight the
                # currently selected NxEr's row in gold so it's easy to track
                # which one you've selected across categories.
                prefix = "★ " if row_idx == 0 else "  "
                is_selected_row = (self.selected_nxer_id is not None
                                    and base_name in name2color
                                    and self.selected_nxer_id is not None
                                    and any(getattr(_a, 'name', None) == base_name
                                            and getattr(_a, 'id', None) == self.selected_nxer_id
                                            for _a in nxers.values()))
                name_color = (255, 215, 0) if is_selected_row else (230, 230, 230)
                name_text = self.small.render(f"{prefix}{name}", True, name_color)
                val_text = self.small.render(f"{val}", True, (220, 220, 220))
                name_rect = name_text.get_rect(topleft=(x + 20, y))
                val_rect = val_text.get_rect(topleft=(x + 180, y))
                # v4.52 PERF (#HUD-cache): replaced per-row O(N) scan
                # `for nxer_obj in nxers.values(): if nxer_obj.name == base_name`
                # with O(1) membership check against the cached name→color
                # dict. Identical: both resolve "does an NxEr named base_name
                # currently exist?". Used only to gate whether this row is
                # clickable — no field of nxer_obj is read beyond name.
                clicked_nxer_name = base_name if base_name in name2color else None
                if clicked_nxer_name: # Store the clickable area for this ranking entry.
                    combined_rect = name_rect.union(val_rect)
                    self.ranking_click_areas.append((combined_rect, clicked_nxer_name))
                self.screen.blit(name_text, (x + 20, y))
                self.screen.blit(val_text, (x + 180, y))
                y += 16
        y += 6
        
        # Draw general statistics.
        self.screen.blit(self.small.render(f"Alive: {alive_count}", True, (220, 220, 220)), (x, y)); y += 18
        self.screen.blit(self.small.render(f"Dead : {dead_count}", True, (220, 220, 220)), (x, y)); y += 18
        self.screen.blit(self.small.render(f"Born : {born_count}", True, (220, 220, 220)), (x, y)); y += 24
        
        # Draw aggregate network statistics.
        if nxers:
            alive_nxers = [a for a in nxers.values() if a.alive]            
            if alive_nxers:
                avg_energy = np.mean([a.net.get_energy_status().get('average_energy', 0.0) for a in alive_nxers])
                avg_branching = np.mean([a.net.branching_ratio for a in alive_nxers])
                self.screen.blit(self.small.render(f"Avg Energy: {avg_energy:.1f}", True, (200, 200, 0)), (x, y)); y += 18
                self.screen.blit(self.small.render(f"Branching: {avg_branching:.2f}", True, (180, 180, 180)), (x, y)); y += 24
        
        # Draw control buttons.
        self.button_rects = {}
        button_rows = [[("playpause", "Pause" if not paused else "Play"), ("exit", "Exit Game")], [("save", "Save Game"), ("load", "Load Game")], [("save_best", "Save Bests")]]
        bx, by, bw, bh, pad = x, y, 120, 28, 8
        for row in button_rows:
            row_x = bx
            for key, lab in row:
                r = pygame.Rect(row_x, by, bw, bh)
                pygame.draw.rect(self.screen, (35, 35, 40), r, border_radius=6)
                pygame.draw.rect(self.screen, (90, 90, 100), r, 1, border_radius=6)
                tx = self.small.render(lab, True, (230, 230, 230))
                self.screen.blit(tx, (r.x + (bw - tx.get_width()) // 2, r.y + (bh - tx.get_height()) // 2))
                self.button_rects[key] = r
                row_x += bw + pad
            by += bh + pad
        
        # --- Draw Detail Panel for Selected NxEr (ALWAYS AVAILABLE WHEN PAUSED) ---
        self.detail_buttons = {}
        if paused and self.selected_nxer_id is not None and self.selected_nxer_id in nxers:
            a = nxers[self.selected_nxer_id]
            px, py, pw, ph = x, by + 12, panel_w, 340
            rect = pygame.Rect(px - 10, py - 8, pw + 20, ph+50)
            pygame.draw.rect(self.screen, (0, 0, 0, 200), rect, border_radius=8)
            pygame.draw.rect(self.screen, (80, 80, 80), rect, 1, border_radius=8)
            
            gender_str = "Male" if a.is_male else "Female"
            title = f"{a.name} (id {a.id}) - {gender_str}"
            self.screen.blit(self.big.render(title, True, (230, 230, 230)), (px, py)); py += 28
            
            # Display detailed stats for the selected agent.
            terrain_type = "Land" if a.can_land and not a.can_sea else ("Sea" if a.can_sea and not a.can_land else "Both")
            energy_status = a.net.get_energy_status() if hasattr(a.net, 'get_energy_status') else {}
            facts = [f"Color : {a.color}", f"Pos : {a.pos} Food : {a.food:.1f}", f"Alive : {a.alive} Terr: {terrain_type} Lived : {a.stats.time_lived_s:.1f}s", f"Found : {a.stats.food_found:.1f} Taken: {a.stats.food_taken:.1f}", f"Mates : {a.stats.mates_performed} Explr : {a.stats.explored}", f"Energy: {energy_status.get('average_energy', 0):.1f} Fitness: {a.stats.fitness_score:.3f}", f"Branching: {energy_status.get('branching_ratio', 0):.2f}"]
            # v4.5: Voice/Song info — pitch, tones, harmonicity, current sing level
            v = getattr(a, 'voice', None)
            if v is not None:
                tones_str = "/".join(str(int(t)) for t in v.voice_tones)
                sing_sym = {-1: "silent", 0: "hum", 1: "SING"}.get(int(getattr(a, 'last_sing_level', 0)), "?")
                facts.append(f"Voice : {v.base_freq:6.1f}Hz harm={v.harmonicity:.2f} [{sing_sym}]")
                facts.append(f"Tones : {tones_str}  rep={len(v.repertoire)}")
            for line in facts:
                self.screen.blit(self.small.render(line, True, (220, 220, 220)), (px, py)); py += 18
            py += 6
            
            # Display key parameters of the agent's neural network.
            self.screen.blit(self.small.render("Network params:", True, (200, 200, 200)), (px, py)); py += 18
            P = a.net.params
            # v4.52 FIX (unrelated to perf): When a multi-sphere brain is active,
            # `a.net` points at the MOTOR sphere's sub-network (see make_nxer in
            # game_loop.py: "The motor sphere's network becomes the primary net
            # for backward compat"). The motor sphere's num_input_neurons counts
            # its relay ports from the association sphere, NOT the 10 sensory
            # channels. For the stats panel we want the sensory sphere's input
            # count (= world-facing channels), and aggregate hidden/outputs and
            # ITU circles across all spheres.
            brain = getattr(a, 'brain', None)
            if brain is not None and getattr(brain, 'spheres', None):
                spheres = brain.spheres
                sensory = spheres.get('sensory')
                # Inputs: prefer sensory sphere (has the 10 world channels).
                n_inputs = sensory.network.params.num_input_neurons if sensory else P.num_input_neurons
                # Outputs: prefer motor sphere (has the 7 behavior channels) —
                # which happens to be what a.net already points at, but be explicit.
                motor = spheres.get('motor')
                n_outputs = motor.network.params.num_output_neurons if motor else P.num_output_neurons
                # Hidden + circles: sum across all spheres for a faithful brain picture.
                n_hidden = sum(s.network.params.num_hidden_neurons for s in spheres.values())
                n_circles = sum(len(s.network.itu_circles) for s in spheres.values())
                # Sphere summary line — e.g. "Brain: 3 spheres [sensory:6 association:10 motor:6]"
                sphere_parts = []
                for sid in ('sensory', 'association', 'motor'):
                    if sid in spheres:
                        sphere_parts.append(f"{sid}:{spheres[sid].network.params.num_hidden_neurons}")
                # Also include any other user-defined spheres not in the default trio.
                for sid, s in spheres.items():
                    if sid not in ('sensory', 'association', 'motor'):
                        sphere_parts.append(f"{sid}:{s.network.params.num_hidden_neurons}")
                brain_line = f"Brain: {len(spheres)} spheres [{' '.join(sphere_parts)}]"
            else:
                n_inputs = P.num_input_neurons
                n_outputs = P.num_output_neurons
                n_hidden = P.num_hidden_neurons
                n_circles = len(a.net.itu_circles)
                brain_line = None

            main_params = []
            if brain_line is not None:
                main_params.append(brain_line)
            main_params += [
                f"inputs={n_inputs} hidden={n_hidden} outputs={n_outputs}",
                f"conn_prob={P.connection_probability:.2f} steps={P.simulation_steps}",
                f"tau_fast={P.tau_fast:.2f} slow={P.tau_slow:.2f} meta={P.tau_meta:.2f}",
                f"thr_exc={P.firing_threshold_excitatory:.3f} thr_inh={P.firing_threshold_inhibitory:.3f}",
                f"learn={P.learning_rate:.3f} stdp_win={P.stdp_window:.3f}",
                f"dopamine={P.dopamine_baseline:.3f} serotonin={P.serotonin_baseline:.3f}",
                f"energy_cost={P.firing_energy_cost:.1f} meta_rate={P.metabolic_rate:.2f}",
                f"circles={n_circles} evolve_int={P.evolution_interval}",
            ]
            for line in main_params:
                self.screen.blit(self.small.render(line, True, (210, 210, 210)), (px, py)); py += 16
            py += 10
            
            # Draw buttons specific to the detail panel (e.g., save this specific NxEr).
            bw2, bh2, pad2 = 120, 26, 8
            detail_button_rows = [[("save_nxer", "Save NxEr"), ("load_nxer", "Load NxEr")], [("save_nxvizer", "Save NxVizer"), ("load_nxvizer", "Load NxVizer")]]
            for row in detail_button_rows:
                row_x = px
                for key, lab in row:
                    r = pygame.Rect(row_x, py, bw2, bh2)
                    pygame.draw.rect(self.screen, (35, 35, 40), r, border_radius=6)
                    pygame.draw.rect(self.screen, (90, 90, 100), r, 1, border_radius=6)
                    tx = self.small.render(lab, True, (230, 230, 230))
                    self.screen.blit(tx, (r.x + (bw2 - tx.get_width()) // 2, r.y + (bh2 - tx.get_height()) // 2))
                    self.detail_buttons[key] = r
                    row_x += bw2 + pad2
                py += bh2 + pad2
        
        # --- Draw Visual Mode Indicator ---
        if not self.visual_mode:
            indicator = self.big.render("(V to view)", True, (255, 0, 0))
            self.screen.blit(indicator, (20, 20))

        # v4.5: Audio status hint. Always visible in the corner so the user
        # knows "M" toggles singing. The indicator stays off when audio is off.
        audio_on = bool(getattr(self, 'audio_engine', None) and self.audio_engine.is_enabled())
        audio_label = f"(M to {'mute' if audio_on else 'hear'})"
        audio_col = (60, 220, 120) if audio_on else (180, 180, 180)
        audio_surf = self.small.render(audio_label, True, audio_col)
        self.screen.blit(audio_surf, (20, 54 if not self.visual_mode else 20))
        
        # v147 (v4.55) — persistent survivability HUD strip (top-left,
        # always visible whether or not the dashboard is open). Surfaces
        # the "NxErs stop moving and die" pathology without requiring the
        # user to press L. Cheap — one small text render per frame.
        try:
            from logger import get_data_logger as _gdl
            _log = _gdl()
            if _log is not None and hasattr(_log, 'get_survivability_dashboard'):
                sd = _log.get_survivability_dashboard()
                if sd['alive_count'] > 0 or sd['survivability_score'] != 0.5:
                    self._draw_survivability_hud(sd)
        except Exception:
            pass
                
        if game_over: self._draw_restart_modal()
        # v146 (v4.54): metrics-dashboard overlay (toggled with L). Drawn LAST
        # so it sits on top of everything else, but BEFORE display.flip so it
        # appears in the same frame.
        if self.metrics_dashboard is not None and self.metrics_dashboard.visible:
            try:
                from logger import get_data_logger
                import config as _cfg
                # v158 (v4.66) — pass current nxers so the dashboard can
                # build the per-champion combo box and render per-NxEr views.
                self.metrics_dashboard.set_nxers(nxers)
                self.metrics_dashboard.draw(get_data_logger(), _cfg._game_id or "unknown")
            except Exception as _exc_dash:
                # Render a tiny error ribbon, never crash the renderer
                err_msg = self.small.render(
                    f"[dashboard error] {type(_exc_dash).__name__}: {_exc_dash}",
                    True, (255, 100, 100))
                self.screen.blit(err_msg, (20, self.screen.get_height() - 20))
        pygame.display.flip()
    
    def handle_input(self, dt):
        """Handles continuous keyboard input for camera panning."""
        if not self.visual_mode: return  # NEW: Skip input handling when visual mode is off
        keys = pygame.key.get_pressed()
        pstep = (50.0 / self.zoom) * dt
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: self.pan[0] -= pstep
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.pan[0] += pstep
        if keys[pygame.K_w] or keys[pygame.K_UP]: self.pan[1] -= pstep
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: self.pan[1] += pstep
    
    def event_zoom_rotate_pan(self, ev):
        """Handles discrete user input events for camera control (zoom, rotation, drag-pan)."""
        if not self.visual_mode: return  # NEW: Skip event handling when visual mode is off
        if ev.type == pygame.MOUSEWHEEL:
            self.zoom *= 1.1 if ev.y > 0 else 0.9
            self.zoom = _clamp(self.zoom, 0.5, 64.0)
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_q: self.rot -= 0.04
            elif ev.key == pygame.K_e: self.rot += 0.04
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3: # Right mouse button for drag-pan.
            self.dragging = True
            self.drag_start = ev.pos
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            dx = ev.pos[0] - self.drag_start[0]
            dy = ev.pos[1] - self.drag_start[1]
            self.drag_start = ev.pos
            wx, wy = _rot(dx / self.zoom, dy / self.zoom, -self.rot)
            self.pan[0] -= wx
            self.pan[1] -= wy
    
    def button_clicked(self, pos) -> Optional[str]:
        """Checks if a click position collides with any of the main UI buttons."""
        for k, r in self.button_rects.items():
            if r.collidepoint(pos): return k
        for k, r in self.overlay_buttons.items():
            if r.collidepoint(pos): return k
        return None
    
    def detail_button_clicked(self, pos) -> Optional[str]:
        """Checks if a click position collides with any buttons in the detail panel."""
        for k, r in self.detail_buttons.items():
            if r.collidepoint(pos): return k
        return None
    
    def ranking_clicked(self, pos) -> Optional[int]:
        """Checks if a click position collides with any of the names in the ranking list."""
        for rect, name in self.ranking_click_areas:
            if rect.collidepoint(pos): return name
        return None
    
    def clear_detail(self):
        """Deselects the current NxEr and clears the detail panel."""
        self.selected_nxer_id = None
        self.detail_buttons = {}
    
    def tick(self, fps_cap=60):
        """Advances the Pygame clock, enforces an FPS cap, and returns the frame's delta time."""
        self.dt = self.clock.tick(fps_cap) / 1000.0
        return self.dt
