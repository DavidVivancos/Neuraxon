# Neuraxon Game of Life v.4.53 ui renderer (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 145
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
        pygame.display.set_caption("Neuraxon Game of Life v 4.52 (Research Version) - By David Vivancos & Dr Jose Sanchez for Qubic Science")
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
        self.detail_buttons: Dict[str, pygame.Rect] = {}
        self.ranking_click_areas: List[Tuple[pygame.Rect, int]] = []
        self.visual_mode = False  # NEW: Visual mode flag, set to off for speed with V Key
        # v4.5: Audio engine reference (set by game_loop). Renderer uses it
        # only to read is_enabled() for the status hint — no synthesis here.
        self.audio_engine = None

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
    
    def draw_world(self, foods: Dict[int, 'Food'], nxers: Dict[int, 'NxEr'], hud: Dict[str, List[Tuple[str, str]]], alive_count: int, dead_count: int, born_count: int, paused: bool, effects: List[dict], step_tick: int, GlobalTimeSteps: int, game_over: bool, game_index: int, best_scores: Optional[Dict[str, float]] = None):
        """The main rendering function, called once per frame to draw the entire scene."""
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
        for title, rows in hud.items():
            display_title = title
            score = best_scores.get(title) if best_scores else None
            if score is not None: display_title = f"{title} ({score:.2f})" if isinstance(score, float) else f"{title} ({int(score)})"
            self.screen.blit(self.small.render(display_title, True, (180, 180, 180)), (x, y)); y += 18
            for name, val in rows[:3]:
                base_name = name.replace(" [Die]", "")
                base_name = base_name.split(" [", 1)[0].strip()   #now the round is emmbedd in the name in hud
                dot_c = name2color.get(base_name, (200, 200, 200))
                pygame.draw.circle(self.screen, dot_c, (x + 8, y + 8), 6)
                name_text = self.small.render(f"{name}", True, (230, 230, 230))
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
                
        if game_over: self._draw_restart_modal()
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
