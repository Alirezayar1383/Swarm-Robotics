import sys, os, yaml
import numpy as np
import random
import heapq
from collections import defaultdict
import matplotlib

# ✅ FIXED: Try interactive backend, fall back to Agg
try:
    matplotlib.use('TkAgg')
    INTERACTIVE = True
except Exception:
    matplotlib.use('Agg')
    INTERACTIVE = False
import matplotlib.pyplot as plt

# ✅ FIXED: sys.path before project imports
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
OBSTACLE_DENSITY = cfg['obstacle_density']

# ✅ FIXED: Read test configuration from config file (not hardcoded)
test_sizes = cfg.get('final_tests', [20, 25, 28, 30, 35, 40])
test_agents = cfg.get('agent_counts_for_tests', [1, 3, 5, 8, 10])

# ✅ FIXED: Per-size configuration (same as final_evaluation.py)
NUM_SERVERS_PER_SIZE = cfg.get('num_servers_per_size', {})
TEST_MAX_STEPS_FACTOR_PER_SIZE = cfg.get('test_max_steps_factor_per_size', {})
NUM_TRIALS_PER_SIZE = cfg.get('num_trials_per_size', {})
LARGE_MAP_THRESHOLD = cfg.get('large_map_threshold', 50)
AGENT_COUNTS_LARGE = cfg.get('agent_counts_for_large_maps', test_agents)

DEFAULT_TRIALS = 5
DEFAULT_MAX_STEPS_FACTOR = cfg.get('test_max_steps_factor', 8)


def get_num_servers_for_size(size):
    """Get server count for a given map size (per-size if available)."""
    return NUM_SERVERS_PER_SIZE.get(size, cfg.get('num_servers', 3))


def get_max_steps_factor_for_size(size):
    """Get max_steps_factor for a given map size."""
    return TEST_MAX_STEPS_FACTOR_PER_SIZE.get(size, DEFAULT_MAX_STEPS_FACTOR)


def get_num_trials_for_size(size):
    """Get number of trials for a given map size."""
    return NUM_TRIALS_PER_SIZE.get(size, DEFAULT_TRIALS)


def get_agent_counts_for_size(size):
    """Reduce agent counts for very large maps to save runtime."""
    if size >= LARGE_MAP_THRESHOLD:
        return AGENT_COUNTS_LARGE
    return test_agents


# ============================================================================
# A* helpers
# ============================================================================
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


# ============================================================================
# Baseline policies
# ============================================================================
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
            if 0 <= nx < self.env.world.width and 0 <= ny < self.env.world.height \
                    and self.env.world.grid[ny, nx] == 0:
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
                nx, ny = x + dx, y + dy
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
            return 2 if env.world.grid[y + 1, x] == 0 else 4
        if y >= y_end:
            return 0 if env.world.grid[y - 1, x] == 0 else 4

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


# ============================================================================
# ✅ FIXED: Greedy server placement — scales to any number of servers
# ============================================================================
def place_servers_greedy(width, height, grid, num_servers, radius=SERVER_RADIUS):
    if num_servers == 0:
        return []
    chosen_positions = []
    covered = np.zeros_like(grid, dtype=bool)
    for _ in range(num_servers):
        best_score = -1
        best_pos = None
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 0 and (x, y) not in chosen_positions:
                    score = 0
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height \
                                    and grid[ny, nx] == 0 and not covered[ny, nx]:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_pos = (x, y)
        if best_pos is None:
            break
        chosen_positions.append(best_pos)
        bx, by = best_pos
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = bx + dx, by + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0:
                    covered[ny, nx] = True
    return chosen_positions


# ============================================================================
# Evaluation function
# ============================================================================
def evaluate_policy(policy_class, size, num_agents, trials=None, max_steps_factor=None):
    if trials is None:
        trials = get_num_trials_for_size(size)
    if max_steps_factor is None:
        max_steps_factor = get_max_steps_factor_for_size(size)

    temp_world = GridWorld(width=size, height=size, obstacle_density=OBSTACLE_DENSITY)
    fixed_grid = temp_world.grid

    # ✅ FIXED: Use per-size server count
    num_servers = get_num_servers_for_size(size)

    # ✅ FIXED: Use greedy placement (scales to any number of servers)
    server_positions = place_servers_greedy(size, size, fixed_grid, num_servers)

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
    steps_end = []

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
        # ✅ FIXED: Use MAX_AGENTS for consistency with training overlap
        overlap = (T - C0 / MAX_AGENTS) / (C0 / MAX_AGENTS) if C0 > 0 else 0

        coverages.append(cov)
        lams.append(lam)
        overlaps.append(overlap)
        battery_end.append(np.mean(info.get('battery', [0])))
        active_end.append(info.get('active_agents', num_agents))
        steps_end.append(T)

        print(f"    {size}×{size} | {num_agents} agents | trial {trial + 1}/{trials}: "
              f"cov={cov:.1%}, λ={lam:.3f}, O={overlap:.3f}, steps={T}")

    return {
        'cov_mean': np.mean(coverages), 'cov_std': np.std(coverages),
        'lam_mean': np.mean(lams), 'lam_std': np.std(lams),
        'overlap_mean': np.mean(overlaps), 'overlap_std': np.std(overlaps),
        'battery_mean': np.mean(battery_end),
        'active_mean': np.mean(active_end),
        'steps_mean': np.mean(steps_end),
    }


# ============================================================================
# Main entry point
# ============================================================================
if __name__ == "__main__":
    print("Running baselines with A*, Dijkstra, and others...")
    print(f"Test sizes: {test_sizes}")
    print(f"Test agents: {test_agents}")
    print()

    baselines = {
        'RandomWalk': RandomWalkPolicy,
        'GreedyFrontier': GreedyFrontierPolicy,
        'Lawnmower': LawnmowerPolicy,
        'AStar': AStarPlanner,
        'Dijkstra': DijkstraPlanner,
    }
    results = defaultdict(lambda: defaultdict(dict))

    for size in test_sizes:
        agent_counts = get_agent_counts_for_size(size)
        num_servers = get_num_servers_for_size(size)
        num_trials = get_num_trials_for_size(size)
        max_steps_factor = get_max_steps_factor_for_size(size)

        print(f"\n{'=' * 60}")
        print(f"Baselines on {size}×{size} maps "
              f"(servers={num_servers}, trials={num_trials}, factor={max_steps_factor})")
        print(f"{'=' * 60}")

        for agents in agent_counts:
            if agents > MAX_AGENTS:
                continue
            print(f"  {agents} agents:")
            for name, policy_class in baselines.items():
                print(f"    {name}:")
                res = evaluate_policy(policy_class, size, agents)
                results[size][agents][name] = res

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 100)
    for size in test_sizes:
        agent_counts = get_agent_counts_for_size(size)
        for agents in agent_counts:
            if agents > MAX_AGENTS:
                continue
            if not results[size][agents]:
                continue
            print(f"\n{size}×{size}, {agents} agents:")
            print(f"{'Method':<15} | {'Coverage':>12} | {'λ':>12} | "
                  f"{'O':>12} | {'Batt':>10} | {'Active':>6} | {'Steps':>8}")
            print("-" * 90)
            for name in baselines.keys():
                if name not in results[size][agents]:
                    continue
                r = results[size][agents][name]
                print(f"{name:<15} | {r['cov_mean']:>11.1%} ±{r['cov_std']:.1%} | "
                      f"{r['lam_mean']:>11.3f} ±{r['lam_std']:.3f} | "
                      f"{r['overlap_mean']:>11.3f} ±{r['overlap_std']:.3f} | "
                      f"{r['battery_mean']:>10.0f} | {r['active_mean']:>6.1f} | "
                      f"{r['steps_mean']:>8.0f}")
    print("=" * 100)
    print("Baseline evaluation complete.")