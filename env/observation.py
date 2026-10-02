import numpy as np
from collections import deque

class ObservationBuilder:
    def __init__(self, width, height, max_agents, local_radius):
        self.width = width
        self.height = height
        self.max_agents = max_agents
        self.local_radius = local_radius
        self.patch_size = 2 * local_radius + 1
        self.num_channels = 12 + (max_agents - 1)
        # +1 for battery
        self.scalar_dim = 15 + max_agents + 1

    def build(self, world, agent_id, last_action):
        x, y = world.agent_positions[agent_id]
        patch_size = self.patch_size
        r = self.local_radius

        grid_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        agent_mask = np.zeros((patch_size, patch_size), dtype=np.float32)
        pheromone_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        teammate_pheromone_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        coverage_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        shared_coverage_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        breadcrumb_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        agent_positions_patch = np.zeros((self.max_agents - 1, patch_size, patch_size), dtype=np.float32)

        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                gx, gy = x + dx, y + dy
                py = dy + r
                px = dx + r

                if 0 <= gx < self.width and 0 <= gy < self.height:
                    grid_patch[py, px] = world.grid[gy, gx]
                    pheromone_patch[py, px] = world.pheromone[agent_id][gy, gx]
                    teammate_pheromone_patch[py, px] = world.teammate_pheromone[agent_id][gy, gx]
                    coverage_patch[py, px] = 1.0 if (gx, gy) in world.global_visited else 0.0
                    shared_coverage_patch[py, px] = world.shared_coverage[agent_id][gy, gx]
                    breadcrumb_patch[py, px] = world.breadcrumb[agent_id][gy, gx]
                    if (gx, gy) == (x, y):
                        agent_mask[py, px] = 1.0
                    else:
                        for i, pos in enumerate(world.agent_positions):
                            if i != agent_id and pos == (gx, gy):
                                if i < self.max_agents - 1:
                                    agent_positions_patch[i, py, px] = 1.0
                else:
                    grid_patch[py, px] = 1.0

        channels = np.zeros((self.num_channels, self.patch_size, self.patch_size), dtype=np.float32)
        channels[0] = grid_patch
        channels[1] = agent_mask
        channels[2] = pheromone_patch
        channels[3] = teammate_pheromone_patch
        channels[4] = coverage_patch
        channels[5] = shared_coverage_patch
        channels[6] = breadcrumb_patch

        for i in range(self.max_agents - 1):
            channels[7 + i] = agent_positions_patch[i]

        scalars = np.zeros(self.scalar_dim, dtype=np.float32)
        scalars[0] = x / self.width
        scalars[1] = y / self.height
        scalars[2] = 1.0 if world.agent_mode[agent_id] == 'survival' else 0.0
        scalars[3] = world.get_global_coverage()
        scalars[4] = last_action / 4.0
        scalars[5] = world.idle_counters[agent_id] / world.idle_threshold
        # ---- Battery scalar (normalized) ----
        scalars[6] = world.battery[agent_id] / world.battery_capacity

        return {
            'map': channels.transpose(1, 2, 0),
            'scalars': scalars,
            'coverage': np.array([scalars[3]], dtype=np.float32),
            'agent_pos': np.array([scalars[0], scalars[1]], dtype=np.float32)
        }
