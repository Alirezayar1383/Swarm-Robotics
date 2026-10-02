import sys, os, yaml
import numpy as np
import random
import heapq
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv
from envs.grid_world import GridWorld

# ---- Load config ----
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase9_config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

MAX_AGENTS = cfg['max_agents']
WORKER_RADIUS = cfg['worker_local_radius']
SERVER_RADIUS = cfg['server_local_radius']
AGENT_COUNTS = cfg['agent_counts_for_tests']
NUM_TRIALS = 5
OBSTACLE_DENSITY = cfg['obstacle_density']
MAX_STEPS_FACTOR = 4

test_sizes = [20, 25, 28, 30, 35, 40]
test_agents = [3, 5, 8, 10]

# ---- Helper functions for A* and Dijkstra ----
def get_neighbors(x, y, grid):
    neighbors = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0] and grid[ny, nx] == 0:
            neighbors.append((nx, ny))
    return neighbors

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, grid):
    if start == goal:
        return [start]
    open_set = [(0, start)]
    g_score = {start: 0}
    came_from = {}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for neighbor in get_neighbors(current[0], current[1], grid):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                priority = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current
    return []

def dijkstra(start, goal, grid):
    if start == goal:
        return [start]
    open_set = [(0, start)]
    g_score = {start: 0}
    came_from = {}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for neighbor in get_neighbors(current[0], current[1], grid):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))
                came_from[neighbor] = current
    return []

# ---- Baseline policies ----
class RandomWalkPolicy:
    def __init__(self, env):
        self.env = env

    def get_action(self, agent_id):
        x, y = self.env.world.agent_positions[agent_id]
        actions = list(range(5))
        random.shuffle(actions)
        for act in actions:
            nx, ny = x, y
            if act == 0: ny -= 1
            elif act == 1: nx += 1
            elif act == 2: ny += 1
            elif act == 3: nx -= 1
            if 0 <= nx < self.env.world.width and 0 <= ny < self.env.world.height and self.env.world.grid[ny, nx] == 0:
                return act
        return 4

class GreedyFrontierPolicy:
    def __init__(self, env):
        self.env = env

    def get_action(self, agent_id):
        x, y = self.env.world.agent_positions[agent_id]
        best_dist = 999
        best_action = 4
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.env.world.width and 0 <= ny < self.env.world.height \
                   and self.env.world.grid[ny, nx] == 0 \
                   and (nx, ny) not in self.env.world.global_visited:
                    dist = abs(dx) + abs(dy)
                    if dist < best_dist:
                        best_dist = dist
                        if dx < 0: best_action = 3
                        elif dx > 0: best_action = 1
                        elif dy < 0: best_action = 0
                        elif dy > 0: best_action = 2
        return best_action

class LawnmowerPolicy:
    def __init__(self, env, num_agents):
        self.env = env
        self.num_agents = num_agents
        self.directions = [1] * num_agents

    def get_action(self, agent_id, env):
        x, y = env.world.agent_positions[agent_id]
        h = env.world.height
        strip_height = h / self.num_agents

        strip_idx = int(y // strip_height)
        strip_idx = min(strip_idx, self.num_agents - 1)
        y_start = int(strip_idx * strip_height)
        y_end = int((strip_idx + 1) * strip_height) if strip_idx < self.num_agents - 1 else h

        if y < y_start:
            return 2 if env.world.grid[y+1, x] == 0 else 4
        if y >= y_end:
            return 0 if env.world.grid[y-1, x] == 0 else 4

        if self.directions[agent_id] == 1:
            nx, ny = x + 1, y
            if 0 <= nx < env.world.width and env.world.grid[ny, nx] == 0 and y_start <= ny < y_end:
                return 1
            else:
                for dy in [1, -1]:
                    ny = y + dy
                    if y_start <= ny < y_end and env.world.grid[ny, x] == 0:
                        self.directions[agent_id] = -1
                        return 2 if dy > 0 else 0
                return 4
        else:
            nx, ny = x - 1, y
            if 0 <= nx < env.world.width and env.world.grid[ny, nx] == 0 and y_start <= ny < y_end:
                return 3
            else:
                for dy in [1, -1]:
                    ny = y + dy
                    if y_start <= ny < y_end and env.world.grid[ny, x] == 0:
                        self.directions[agent_id] = 1
                        return 2 if dy > 0 else 0
                return 4

class AStarPlanner:
    def __init__(self, env):
        self.env = env

    def get_action(self, agent_id):
        x, y = self.env.world.agent_positions[agent_id]
        best_dist = 999
        best_goal = None
        for gy in range(self.env.world.height):
            for gx in range(self.env.world.width):
                if self.env.world.grid[gy, gx] == 0 and (gx, gy) not in self.env.world.global_visited:
                    dist = abs(gx - x) + abs(gy - y)
                    if dist < best_dist:
                        best_dist = dist
                        best_goal = (gx, gy)
        if best_goal is None:
            return 4
        path = a_star((x, y), best_goal, self.env.world.grid)
        if len(path) < 2:
            return 4
        nx, ny = path[1]
        if nx == x and ny == y - 1: return 0
        if nx == x + 1 and ny == y: return 1
        if nx == x and ny == y + 1: return 2
        if nx == x - 1 and ny == y: return 3
        return 4

class DijkstraPlanner:
    def __init__(self, env):
        self.env = env

    def get_action(self, agent_id):
        x, y = self.env.world.agent_positions[agent_id]
        best_dist = 999
        best_goal = None
        for gy in range(self.env.world.height):
            for gx in range(self.env.world.width):
                if self.env.world.grid[gy, gx] == 0 and (gx, gy) not in self.env.world.global_visited:
                    dist = abs(gx - x) + abs(gy - y)
                    if dist < best_dist:
                        best_dist = dist
                        best_goal = (gx, gy)
        if best_goal is None:
            return 4
        path = dijkstra((x, y), best_goal, self.env.world.grid)
        if len(path) < 2:
            return 4
        nx, ny = path[1]
        if nx == x and ny == y - 1: return 0
        if nx == x + 1 and ny == y: return 1
        if nx == x and ny == y + 1: return 2
        if nx == x - 1 and ny == y: return 3
        return 4

# ---- Evaluation function ----
def evaluate_policy(policy_class, size, num_agents, trials=NUM_TRIALS, max_steps_factor=MAX_STEPS_FACTOR):
    temp_world = GridWorld(width=size, height=size, obstacle_density=OBSTACLE_DENSITY)
    fixed_grid = temp_world.grid

    if size <= 10:
        num_servers = 1
    else:
        num_servers = 2
        if size > 25:
            num_servers = 3

    # Simple server placement
    server_positions = []
    if num_servers == 1:
        server_positions = [(size//2, size//2)]
    elif num_servers == 2:
        server_positions = [(size//4, size//2), (3*size//4, size//2)]
    else:
        server_positions = [(size//4, size//2), (size//2, size//2), (3*size//4, size//2)]

    env_config = {
        'width': size, 'height': size,
        'max_steps': size * size * max_steps_factor,
        'obstacle_density': OBSTACLE_DENSITY,
        'idle_threshold': cfg['idle_threshold'],
        'd_pheromone': cfg['d_pheromone'],
        'd_comm': cfg['d_comm'],
        'tau_pheromone': cfg['tau_pheromone'],
        'survival_policy': cfg['survival_policy'],
        'battery_capacity': cfg.get('battery_capacity', 1000),
        'energy_move': cfg.get('energy_move', 1.0),
        'energy_stay': cfg.get('energy_stay', 0.5),
        'energy_penalty': cfg.get('energy_penalty', 0.01),
    }

    coverages = []
    lams = []
    overlaps = []
    battery_end = []
    active_end = []

    for trial in range(trials):
        env = CoverageEnv(
            env_config,
            max_agents=MAX_AGENTS,
            active_agents=num_agents,
            num_servers=num_servers,
            server_positions=server_positions,
            obstacle_map=fixed_grid,
            worker_local_radius=WORKER_RADIUS
        )
        obs_tuple, _ = env.reset()
        if policy_class == LawnmowerPolicy:
            policy = LawnmowerPolicy(env, num_agents)
        else:
            policy = policy_class(env)
        done = False
        while not done:
            actions = []
            for i in range(num_agents):
                if policy_class == LawnmowerPolicy:
                    act = policy.get_action(i, env)
                else:
                    act = policy.get_action(i)
                actions.append(act)
            while len(actions) < MAX_AGENTS:
                actions.append(4)
            obs_tuple, _, term, trunc, info = env.step(actions, training=False)
            done = term or trunc
        cov = info['coverage']
        T = info['steps']
        C0 = env.world.free_cells
        lam = T / C0 if C0 > 0 else 0
        overlap = (T - C0 / num_agents) / (C0 / num_agents) if C0 > 0 else 0
        coverages.append(cov)
        lams.append(lam)
        overlaps.append(overlap)
        battery_end.append(np.mean(info.get('battery', [0])))
        active_end.append(info.get('active_agents', num_agents))
        print(f"    {size}×{size} | {num_agents} agents | trial {trial+1}: cov={cov:.1%}, λ={lam:.3f}, O={overlap:.3f}")

    return {
        'cov_mean': np.mean(coverages), 'cov_std': np.std(coverages),
        'lam_mean': np.mean(lams), 'lam_std': np.std(lams),
        'overlap_mean': np.mean(overlaps), 'overlap_std': np.std(overlaps),
        'battery_mean': np.mean(battery_end),
        'active_mean': np.mean(active_end),
    }

# ---- Main ----
if __name__ == "__main__":
    print("Running baselines with A*, Dijkstra, and others...")
    baselines = {
        'RandomWalk': RandomWalkPolicy,
        'GreedyFrontier': GreedyFrontierPolicy,
        'Lawnmower': LawnmowerPolicy,
        'AStar': AStarPlanner,
        'Dijkstra': DijkstraPlanner,
    }
    results = defaultdict(lambda: defaultdict(dict))

    for size in test_sizes:
        print(f"\n{'='*60}")
        print(f"Baselines on {size}×{size} maps")
        for agents in test_agents:
            if agents > MAX_AGENTS:
                continue
            print(f"  {agents} agents:")
            for name, policy_class in baselines.items():
                print(f"    {name}:")
                res = evaluate_policy(policy_class, size, agents)
                results[size][agents][name] = res

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    for size in test_sizes:
        for agents in test_agents:
            if agents > MAX_AGENTS:
                continue
            print(f"\n{size}×{size}, {agents} agents:")
            print(f"{'Method':<15} | {'Coverage':>12} | {'λ':>12} | {'O':>12} | {'Batt':>10} | {'Active':>6} | {'Steps':>8}")
            print("-" * 75)
            for name in baselines.keys():
                r = results[size][agents][name]
                print(f"{name:<15} | {r['cov_mean']:>11.1%} ±{r['cov_std']:.1%} | "
                      f"{r['lam_mean']:>11.3f} ±{r['lam_std']:.3f} | "
                      f"{r['overlap_mean']:>11.3f} ±{r['overlap_std']:.3f} | "
                      f"{r['battery_mean']:>10.0f} | {r['active_mean']:>6.1f}")
    print("=" * 80)
    print("Baseline evaluation complete.")
