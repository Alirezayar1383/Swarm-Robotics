import numpy as np
from typing import List, Tuple
from collections import deque

class MapGenerator:
    """
    Generates connected grid maps with obstacles.
    Ensures the free space is a single connected component.
    """
    
    def __init__(self, width: int, height: int, num_maps: int = 10,
                 obstacle_density: float = 0.15, base_seed: int = 42):
        self.width = width
        self.height = height
        self.num_maps = num_maps
        self.obstacle_density = obstacle_density
        self.base_seed = base_seed
    
    def generate(self) -> List[np.ndarray]:
        maps = []
        rng = np.random.RandomState(self.base_seed)
        
        for i in range(self.num_maps):
            seed = self.base_seed + i
            rng.seed(seed)
            grid = self._generate_connected_map(rng)
            maps.append(grid)
        return maps
    
    def _generate_connected_map(self, rng: np.random.RandomState) -> np.ndarray:
        """Generate a single map with fully connected free space."""
        # Start with all free cells
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        
        total_cells = self.width * self.height
        target_obstacles = int(self.obstacle_density * total_cells)
        
        # Get list of all coordinates and shuffle them
        all_coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        rng.shuffle(all_coords)
        
        obstacles_placed = 0
        for x, y in all_coords:
            if obstacles_placed >= target_obstacles:
                break
            # Temporarily place obstacle
            grid[y, x] = 1
            # Check if free space is still connected
            if self._is_connected(grid):
                obstacles_placed += 1
            else:
                # Revert: obstacle would disconnect the free space
                grid[y, x] = 0
        
        return grid
    
    def _is_connected(self, grid: np.ndarray) -> bool:
        """Check if all free cells (0) form a single connected component."""
        # Find first free cell
        start = None
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == 0:
                    start = (x, y)
                    break
            if start is not None:
                break
        if start is None:
            # No free cells (shouldn't happen with our density)
            return False
        
        # BFS to count reachable free cells
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if grid[ny, nx] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        # Count all free cells in the grid
        total_free = np.sum(grid == 0)
        return len(visited) == total_free