# Neuraxon Game of Life v5.05 Simulation World (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 190
# Based on the Paper "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos https://vivancos.com/  & Dr. Jose Sanchez  https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/397331336_Neuraxon
# Play the Lite Version of the Game of Life at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import math
import random
from typing import List, Tuple, Optional

# Import constants
from config import T_SEA, T_LAND, T_ROCK

class World:
    """Generates and holds the state of the 2D grid-based environment."""
    def __init__(self, N: int, sea_pct: float, rock_pct: float, rnd_seed=None):
        self.N = N
        if rnd_seed is not None: random.seed(rnd_seed)
        # Use multiple layers of a simple noise function to generate natural-looking terrain.
        self.noise_offsets = [(random.random() * 100, random.random() * 100) for _ in range(3)]
        self.grid = [[T_LAND for _ in range(N)] for _ in range(N)]
        self._gen(sea_pct, rock_pct)
    
    def _noise(self, x, y, s, offset_idx):
        """A simple noise function based on sine and cosine waves."""
        ox, oy = self.noise_offsets[offset_idx]
        x_off, y_off = x + ox, y + oy
        return (math.sin(x_off * 0.15 * s) + math.cos(y_off * 0.13 * s) + math.sin((x_off + y_off) * 0.07 * s)) * 0.333
    
    def _gen(self, sea_pct, rock_pct):
        """Procedurally generates the world map with land, sea, and rocks."""
        N = self.N
        values = [[0.0] * N for _ in range(N)]
        # Generate a heightmap using layered noise, with a radial gradient to form an island.
        for y in range(N):
            for x in range(N):
                r = math.hypot(x - N / 2, y - N / 2) / (N * 0.7)
                v = (self._noise(x, y, 0.5, 0) + self._noise(x, y, 1.0, 1) + self._noise(x, y, 1.8, 2)) / 3.0 - r * 0.9
                values[y][x] = v
        
        # Determine the sea level threshold based on the desired percentage of sea.
        flat = sorted(v for row in values for v in row)
        k = int(len(flat) * sea_pct)
        sea_thresh = flat[k] if 0 <= k < len(flat) else min(flat)
        
        # Assign terrain types based on height relative to the sea level.
        for y in range(N):
            for x in range(N):
                self.grid[y][x] = T_SEA if values[y][x] <= sea_thresh else T_LAND
                
        # Randomly place rocks on a percentage of the land tiles.
        land = [(x, y) for y in range(N) for x in range(N) if self.grid[y][x] == T_LAND]
        num_rocks = int(len(land) * rock_pct)
        if num_rocks > 0:
            for (x, y) in random.sample(land, k=min(num_rocks, len(land))):
                self.grid[y][x] = T_ROCK
    
    def in_bounds(self, p):
        """Checks if a coordinate is within the world boundaries."""
        x, y = p
        return 0 <= x < self.N and 0 <= y < self.N
    
    def terrain(self, p):
        """Returns the terrain type at a given coordinate, with toroidal (wrapping) boundaries."""
        x, y = p
        return self.grid[y % self.N][x % self.N]
