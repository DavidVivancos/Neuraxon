# Neuraxon Game of Life v4.79 UI Widgets
# Based on the Paper "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos https://vivancos.com/  & Dr. Jose Sanchez  https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/397331336_Neuraxon
# Play the Lite Version of the Game of Life at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import pygame
import math
from utils import _clamp

class Slider:
    """A simple UI slider widget implemented with Pygame for the configuration screen."""
    def __init__(self, rect: pygame.Rect, min_val: float, max_val: float, default_val: float, label: str, is_int: bool = True):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.is_int = is_int
        self.label = label
        self.handle_radius = 10
        self.dragging = False
        range_size = max_val - min_val
        self.normalized_pos = (default_val - min_val) / range_size if range_size != 0 else 0.5
        self.normalized_pos = _clamp(self.normalized_pos, 0.0, 1.0)
        track_y = rect.centery
        self.track_left = rect.x + self.handle_radius
        self.track_right = rect.x + rect.width - self.handle_radius
        self.track_top = track_y
        self.track_bottom = track_y
        self.handle_x = self.track_left + self.normalized_pos * (self.track_right - self.track_left)
        self.handle_y = track_y
    
    def handle_event(self, event: pygame.event.Event):
        """Handles mouse input for dragging the slider."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos
            if math.hypot(mouse_x - self.handle_x, mouse_y - self.handle_y) <= self.handle_radius:
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x, _ = event.pos
            self.handle_x = _clamp(mouse_x, self.track_left, self.track_right)
            self.normalized_pos = (self.handle_x - self.track_left) / (self.track_right - self.track_left)
            return True
        return False
    
    def get_value(self) -> float:
        """Returns the current numerical value of the slider."""
        value = self.min_val + self.normalized_pos * (self.max_val - self.min_val)
        return int(round(value)) if self.is_int else float(value)
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        """Renders the slider onto a Pygame surface."""
        track_y = self.rect.centery
        pygame.draw.line(surface, (100, 100, 100), (self.track_left, track_y), (self.track_right, track_y), 3)
        pygame.draw.circle(surface, (200, 200, 200), (int(self.handle_x), int(self.handle_y)), self.handle_radius)
        pygame.draw.circle(surface, (150, 150, 150), (int(self.handle_x), int(self.handle_y)), self.handle_radius, 2)
        label_surf = font.render(self.label, True, (220, 220, 220))
        label_x = self.rect.x + self.rect.width // 2 - label_surf.get_width() // 2
        surface.blit(label_surf, (label_x, self.rect.y - 20))
        value = self.get_value()
        value_str = str(int(value)) if self.is_int else f"{value:.2f}"
        value_surf = font.render(value_str, True, (255, 255, 0))
        surface.blit(value_surf, (self.rect.x + self.rect.width + 10, self.rect.y))


class Checkbox:
    """v153 — simple checkbox widget for the configuration screen.
    
    Renders a labelled box that toggles on click. Used for opt-in/opt-out
    settings that don't need a slider (e.g., 'Save full game logs').
    """
    def __init__(self, rect: pygame.Rect, label: str, default: bool = False):
        self.rect = rect
        self.label = label
        self.checked = bool(default)
    
    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
    
    def get_value(self) -> bool:
        return self.checked
    
    def draw(self, surface, font):
        # Outer box
        box_size = self.rect.height
        box_rect = pygame.Rect(self.rect.x, self.rect.y, box_size, box_size)
        pygame.draw.rect(surface, (40, 50, 60), box_rect, border_radius=4)
        border_col = (90, 215, 130) if self.checked else (140, 150, 160)
        pygame.draw.rect(surface, border_col, box_rect, 2, border_radius=4)
        # Check mark
        if self.checked:
            pad = 5
            # Draw simple checkmark — diagonal lines
            pygame.draw.line(surface, (90, 215, 130),
                              (box_rect.x + pad, box_rect.y + box_size // 2),
                              (box_rect.x + box_size // 2 - 1, box_rect.y + box_size - pad), 3)
            pygame.draw.line(surface, (90, 215, 130),
                              (box_rect.x + box_size // 2 - 1, box_rect.y + box_size - pad),
                              (box_rect.x + box_size - pad, box_rect.y + pad), 3)
        # Label to the right of the box
        label_surf = font.render(self.label, True, (220, 220, 220))
        surface.blit(label_surf, (box_rect.right + 10,
                                   box_rect.centery - label_surf.get_height() // 2))


class ComboBox:
    """v158 (v4.66) — dropdown combo box for the realtime Key Metrics
    dashboard. The user clicks the closed box to open it; clicks an option
    to select; clicks outside or selects to close.
    
    Designed for "in-overlay" use: rendered on top of the dashboard panel,
    above all other content when open so the option list isn't occluded.
    
    Each option is a (label, value) tuple. `value` can be any hashable —
    typically a string like "aggregate" or an NxEr id (int). The set of
    options is mutable (use `set_options()` to refresh as champions change
    each frame).
    """
    
    def __init__(self, rect: pygame.Rect, options=None,
                  default_value=None, label: str = ""):
        self.rect = rect
        self.options = list(options) if options else []
        self.label = label
        self.is_open = False
        self._option_row_height = max(20, rect.height)
        # Pick the index of the option matching default_value (or 0)
        self.selected_index = 0
        if default_value is not None:
            for i, (_, v) in enumerate(self.options):
                if v == default_value:
                    self.selected_index = i
                    break
    
    def set_options(self, options, preserve_value=True):
        """Replace the options list. If preserve_value=True, try to keep
        the same selected value (by equality) across the refresh. If the
        previous value is no longer in the list, select index 0 (the
        most semantically meaningful fallback — usually "Overall aggregate"
        or whatever the caller put first)."""
        old_value = self.get_value() if self.options else None
        self.options = list(options)
        if preserve_value and old_value is not None:
            for i, (_, v) in enumerate(self.options):
                if v == old_value:
                    self.selected_index = i
                    return
        # Old value not found (or no preserve) — reset to 0
        self.selected_index = 0
    
    def get_value(self):
        if not self.options:
            return None
        return self.options[self.selected_index][1]
    
    def get_label(self):
        if not self.options:
            return ""
        return self.options[self.selected_index][0]
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if the event was consumed (don't propagate)."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                # Click on closed box → toggle open
                self.is_open = not self.is_open
                return True
            elif self.is_open:
                # Click while open: check each option row
                for i in range(len(self.options)):
                    option_rect = pygame.Rect(
                        self.rect.x,
                        self.rect.y + (i + 1) * self._option_row_height,
                        self.rect.width, self._option_row_height)
                    if option_rect.collidepoint(event.pos):
                        self.selected_index = i
                        self.is_open = False
                        return True
                # Click outside the open dropdown — close without changing
                self.is_open = False
                # Don't consume — let click pass to underlying widget
                return False
        # Escape key closes if open
        if event.type == pygame.KEYDOWN and self.is_open:
            if event.key == pygame.K_ESCAPE:
                self.is_open = False
                return True
        return False
    
    def draw_closed(self, surface, font):
        """Draw just the closed box (call this first, normal z-order)."""
        # Background
        pygame.draw.rect(surface, (35, 45, 55), self.rect, border_radius=4)
        pygame.draw.rect(surface, (110, 120, 130), self.rect, 1, border_radius=4)
        # Selected label
        label = self.get_label()
        # Truncate to fit width
        max_w = self.rect.width - 24  # leave room for arrow
        text = label
        text_surf = font.render(text, True, (230, 230, 230))
        if text_surf.get_width() > max_w:
            # Trim characters until it fits
            while text and font.size(text + "…")[0] > max_w:
                text = text[:-1]
            text = text + "…"
            text_surf = font.render(text, True, (230, 230, 230))
        surface.blit(text_surf, (self.rect.x + 8,
                                  self.rect.centery - text_surf.get_height() // 2))
        # Down-arrow indicator
        ax = self.rect.right - 14
        ay = self.rect.centery - 3
        pygame.draw.polygon(surface, (190, 195, 200),
                             [(ax, ay), (ax + 10, ay), (ax + 5, ay + 6)])
        # Optional label above box
        if self.label:
            lbl = font.render(self.label, True, (170, 175, 180))
            surface.blit(lbl, (self.rect.x, self.rect.y - 16))
    
    def draw_open(self, surface, font):
        """Draw the open dropdown list (call this AFTER all other widgets
        so it sits on top in z-order)."""
        if not self.is_open or not self.options:
            return
        # Backdrop for the dropdown
        list_h = len(self.options) * self._option_row_height
        list_rect = pygame.Rect(self.rect.x,
                                 self.rect.y + self._option_row_height,
                                 self.rect.width, list_h)
        pygame.draw.rect(surface, (25, 35, 45), list_rect, border_radius=4)
        pygame.draw.rect(surface, (140, 150, 160), list_rect, 1, border_radius=4)
        # Each option row
        mouse_pos = pygame.mouse.get_pos()
        for i, (label, _value) in enumerate(self.options):
            opt_rect = pygame.Rect(
                self.rect.x,
                self.rect.y + (i + 1) * self._option_row_height,
                self.rect.width, self._option_row_height)
            # Highlight hovered option
            is_hovered = opt_rect.collidepoint(mouse_pos)
            is_selected = (i == self.selected_index)
            if is_hovered:
                pygame.draw.rect(surface, (55, 75, 95), opt_rect)
            elif is_selected:
                pygame.draw.rect(surface, (45, 60, 75), opt_rect)
            # Truncate label
            max_w = opt_rect.width - 16
            text = label
            text_surf = font.render(text, True, (235, 235, 235))
            if text_surf.get_width() > max_w:
                while text and font.size(text + "…")[0] > max_w:
                    text = text[:-1]
                text = text + "…"
                text_surf = font.render(text, True, (235, 235, 235))
            surface.blit(text_surf, (opt_rect.x + 8,
                                      opt_rect.centery - text_surf.get_height() // 2))
    
    def draw(self, surface, font):
        """Convenience: draw both closed + open in correct order. For
        proper z-order (dropdown on top of everything else), prefer
        draw_closed() at normal point + draw_open() at end of frame."""
        self.draw_closed(surface, font)
        self.draw_open(surface, font)
