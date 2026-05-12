# Neuraxon Game of Life v4.63 UI Widgets
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
