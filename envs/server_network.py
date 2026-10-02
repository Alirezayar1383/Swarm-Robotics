import numpy as np
from collections import defaultdict

class ServerNetwork:
    def __init__(self, width: int, height: int, num_servers: int = 3,
                 server_positions: list = None, radius: int = 5):
        self.width = width
        self.height = height
        self.num_servers = num_servers
        self.radius = radius          # 10×10 window (radius=5)

        if server_positions is not None:
            self.server_positions = server_positions
        else:
            self.server_positions = self._default_positions()

        # Remove sections – we don't need them anymore
        self.map_sections = None

        # Each server maintains its own visited set
        self.server_visited = [set() for _ in range(num_servers)]
        self.server_free_cells = [0] * num_servers   # will be computed from grid

        # Worker-to-server mapping (agent_id -> server_id or -1 if unassigned)
        self.worker_server = {}  # value = server_id, or -1 for serverless

    def _default_positions(self):
        # fallback if no positions provided
        positions = []
        if self.num_servers == 1:
            positions.append((self.width // 2, self.height // 2))
        elif self.num_servers == 2:
            positions.append((self.width // 4, self.height // 2))
            positions.append((3 * self.width // 4, self.height // 2))
        elif self.num_servers >= 3:
            positions.append((0, 0))
            positions.append((self.width // 2, self.height // 2))
            positions.append((self.width - 1, self.height - 1))
        while len(positions) < self.num_servers:
            positions.append(positions[-1])
        return positions[:self.num_servers]

    def assign_worker(self, agent_id, position, grid):
        """Assign worker to the nearest server within radius, or -1 if none."""
        x, y = position
        best_dist = 999
        best_server = -1
        for s, (sx, sy) in enumerate(self.server_positions):
            dist = max(abs(x - sx), abs(y - sy))  # Chebyshev distance
            if dist <= self.radius and dist < best_dist:
                best_dist = dist
                best_server = s
        self.worker_server[agent_id] = best_server
        return best_server

    def get_worker_server(self, agent_id):
        return self.worker_server.get(agent_id, -1)

    def get_relative_position(self, agent_id, position):
        """Return (dx, dy) relative to the agent's assigned server, or (0,0) if serverless."""
        x, y = position
        server_id = self.get_worker_server(agent_id)
        if server_id == -1 or server_id >= len(self.server_positions):
            return (0, 0)   # no server, relative position meaningless
        sx, sy = self.server_positions[server_id]
        return (x - sx, y - sy)

    def get_server_coverage(self, server_id):
        if self.server_free_cells[server_id] == 0:
            return 1.0
        return len(self.server_visited[server_id]) / self.server_free_cells[server_id]

    def get_global_coverage(self):
        total_visited = sum(len(v) for v in self.server_visited)
        total_free = sum(self.server_free_cells)
        if total_free == 0:
            return 1.0
        return total_visited / total_free

    def update_visited(self, agent_id, position, grid):
        """Add a cell to the server's visited set if within its radius."""
        x, y = position
        owner = self._get_cell_owner(x, y)
        if owner is not None:
            self.server_visited[owner].add((x, y))

    def _get_cell_owner(self, x, y):
        """Return the server whose radius contains (x,y), or None if none."""
        for s, (sx, sy) in enumerate(self.server_positions):
            if max(abs(x - sx), abs(y - sy)) <= self.radius:
                return s
        return None

    def get_global_visited_set(self):
        visited = set()
        for v in self.server_visited:
            visited.update(v)
        return visited
