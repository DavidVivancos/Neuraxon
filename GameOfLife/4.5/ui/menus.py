# Neuraxon Game of Life v.4.53 menus (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 145
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 3 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import pygame
from typing import Optional, Dict, Any

# Local imports
from .widgets import Slider
from logger import set_data_logger_level

def run_config_screen() -> Optional[Dict[str, Any]]:
    """
    Displays a pre-simulation configuration screen using Pygame, allowing the user
    to adjust key parameters with sliders before starting the game.
    """
    # Import TestMode locally to avoid circular dependency with game_loop
    # (since game_loop imports ui.renderer)
    from game_loop import TestMode

    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
    pygame.display.set_caption("Neuraxon Game Of Life v 4.5 (Research Version) By David Vivancos & Dr Jose Sanchez for Qubic Science - Configuration")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16); title_font = pygame.font.SysFont("consolas", 32, bold=True)
    
    # Define the parameters that will be configurable via sliders.
    # (Label, Min, Max, Default, IsInt, ConversionFunc)
    param_specs = [
        ("World Size", 30, 150, 40, True, lambda x: x), 
        ("Sea Percentage", 20, 80, 55, True, lambda x: x / 100.0), 
        ("Rock Percentage", 1, 10, 2, True, lambda x: x / 100.0),
        ("Starting NxErs", 1, 100, 30, True, lambda x: x), 
        ("Food Sources", 25, 300, 50, True, lambda x: x), 
        ("Food Respawn", 200, 600, 400, True, lambda x: x),
        ("Start Food", 25, 200, 25, True, lambda x: float(x)), 
        ("Max Neurons", 5, 50, 50, True, lambda x: x), 
        ("Global Time Steps", 30, 90, 60, True, lambda x: x),
        # NEW v3.0: Circadian cycle length
        ("Day/Night Cycle (ticks)", 600, 4800, 600, True, lambda x: x),
        ("Mate Cooldown (sec)", 6, 20, 12, True, lambda x: x), 
        ("Log Level (1-3)", 1, 3, 3, True, lambda x: x),
        # --- NEW SETTINGS ---
        ("Max Rounds (0=Inf)", 0, 100, 5, True, lambda x: None if x == 0 else x),
        ("Round Time Limit Mins (0=Inf)", 0, 60, 0, True, lambda x: None if x == 0 else x)
    ]
    
    screen_width, screen_height = screen.get_size()
    slider_container_width = 700; slider_width = 600
    slider_start_x = (screen_width - slider_container_width) // 2 + (slider_container_width - slider_width) // 2
    start_y = 130; slider_height = 50
    sliders = []
    
    for i, (label, min_val, max_val, default_val, is_int, _) in enumerate(param_specs):
        rect = pygame.Rect(slider_start_x, start_y + i * slider_height, slider_width, 20)
        sliders.append(Slider(rect, min_val, max_val, default_val, label, is_int))
    
    play_button_width = 250; play_button_height = 50
    play_button_x = (screen_width - play_button_width) // 2
    
    # Test Mode Sliders
    test_slider_y_start = 900
    
    # Start Game Button (above test sliders)
    play_button_y = 800
    play_button_rect = pygame.Rect(play_button_x, play_button_y, play_button_width, play_button_height)
    
    # Test Mode Button (below test sliders)
    test_button_rect = pygame.Rect(play_button_x, test_slider_y_start + slider_height + 70, play_button_width, play_button_height)
    test_games_slider_rect = pygame.Rect(slider_start_x, test_slider_y_start, slider_width, 20)
    test_time_slider_rect = pygame.Rect(slider_start_x, test_slider_y_start + slider_height, slider_width, 20)
    test_games_slider = Slider(test_games_slider_rect, 1, 50, 10, "Test Mode: Number of Games", True)
    test_time_slider = Slider(test_time_slider_rect, 1, 60, 20, "Test Mode: Max Minutes per Game", True)
    test_sliders = [test_games_slider, test_time_slider]
    
    running = True
    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return None
            for slider in sliders: slider.handle_event(event)
            for slider in test_sliders: slider.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if play_button_rect.collidepoint(event.pos):
                    params = {}
                    for i, slider in enumerate(sliders): # Collect values from all sliders.
                        raw_value = slider.get_value()
                        conversion_func = param_specs[i][5]
                        param_name = [
                    "NxWorldSize", "NxWorldSea", "NxWorldRocks", "StartingNxErs", 
                    "MaxFood", "FoodRespan", "StartFood", "MaxNeurons", 
                    "GlobalTimeSteps", "DayNightCycle", "MateCooldownSeconds", 
                    "LogLevel", "max_rounds", "round_time_limit"
                ][i]
                        params[param_name] = conversion_func(raw_value)
                    log_level = params.pop("LogLevel", 2)
                    set_data_logger_level(log_level)
                    return params # Return the dictionary of parameters to the main function.
                elif test_button_rect.collidepoint(event.pos):
                    # Trigger Test Mode with slider values
                    games_count = int(test_games_slider.get_value())
                    max_minutes = int(test_time_slider.get_value())
                    TestMode(games_count=games_count, max_minutes=max_minutes)
                    return None # Exit config screen after test mode finishes

        screen.fill((15, 15, 18))
        title_surf = title_font.render("Neuraxon Game Of Life 4.5 (Research Version) - World Configuration", True, (235, 235, 240))
        screen.blit(title_surf, (screen.get_width() // 2 - title_surf.get_width() // 2, 50))
        # v4.5: In-game key-hint subtitle
        hint_font = pygame.font.SysFont("consolas", 14)
        hint = hint_font.render(
            "In-game keys:  SPACE = pause   V = toggle visuals   M = toggle NxEr singing (audio off by default)",
            True, (160, 200, 180),
        )
        screen.blit(hint, (screen.get_width() // 2 - hint.get_width() // 2, 90))
        for slider in sliders: slider.draw(screen, font)
        
        # Draw Play Button (above test sliders)
        pygame.draw.rect(screen, (35, 180, 60), play_button_rect, border_radius=8)
        pygame.draw.rect(screen, (60, 220, 90), play_button_rect, 2, border_radius=8)
        play_text = font.render("Start Game", True, (255, 255, 255))
        screen.blit(play_text, (play_button_rect.x + (play_button_rect.width - play_text.get_width()) // 2, play_button_rect.y + (play_button_rect.height - play_text.get_height()) // 2))
        
        # Draw Test Mode Sliders
        for slider in test_sliders: slider.draw(screen, font)
        
        # Draw Test Mode Button
        pygame.draw.rect(screen, (180, 60, 35), test_button_rect, border_radius=8)
        pygame.draw.rect(screen, (220, 90, 60), test_button_rect, 2, border_radius=8)
        games_count = int(test_games_slider.get_value())
        max_minutes = int(test_time_slider.get_value())
        test_text = font.render(f"Run Test Mode", True, (255, 255, 255))
        screen.blit(test_text, (test_button_rect.x + (test_button_rect.width - test_text.get_width()) // 2, test_button_rect.y + (test_button_rect.height - test_text.get_height()) // 2))
        
        pygame.display.flip()
