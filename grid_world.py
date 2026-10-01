import numpy as np
from collections import deque, defaultdict
from envs.server_network import ServerNetwork

class GridWorld:
    def __init__(self, width: int, height: int, num_agents: int = 5,
                 obstacle_density: float = 0.0, obstacle_map: np.ndarray = None,
                 min_spawn_dist: int = None, idle_threshold: int = 8,
                 d_pheromone: int = 5, d_comm: int = 10,
                 tau_pheromone: float = 0.95, active_agents: int = None,
                 num_servers: int = 3, server_positions: list = None,
                 survival_policy: str = 'greedy_frontier',
                 battery_capacity: float = 1000.0,
                 energy_move: float = 1.0,
                 energy_stay: float = 0.5,
                 energy_penalty: float = 0.01,
                 enforce_collisions: bool = True,
                 apply_fallback: bool = True):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.active_agents = active_agents if active_agents is not None else num_agents
        self.min_spawn_dist = min_spawn_dist if min_spawn_dist is not None else max(2, min(width, height)//4)
        self.idle_threshold = idle_threshold
        self.d_pheromone = d_pheromone
        self.d_comm = d_comm
        self.tau_pheromone = tau_pheromone
        self.survival_policy = survival_policy
        self._rl_survival_callback = None
        self.enforce_collisions = enforce_collisions
        self.apply_fallback = apply_fallback

        self.battery_capacity = battery_capacity
        self.energy_move = energy_move
        self.energy_stay = energy_stay
        self.energy_penalty = energy_penalty

        if obstacle_map is not None:
            self.grid = obstacle_map.copy()
        else:
            self.grid = np.zeros((height, width), dtype=np.int32)
            self._add_random_obstacles(obstacle_density)

        self.free_cells = int((self.grid == 0).sum())
        self.num_servers = num_servers
        self.server_network = ServerNetwork(width, height, num_servers, server_positions, radius=5)
        self.use_servers = (num_servers > 0)

        self.agent_mode = ['normal'] * num_agents
        self.agent_positions = []
        self.global_visited = set()
        self.visit_counts = defaultdict(int)

        self.action_histories = []
        self.pheromone = []
        self.teammate_pheromone = []
        self.coverage_patches = []
        self.shared_coverage = []
        self.breadcrumb = []
        self.recent_positions = []
        self.idle_counters = [0] * num_agents
        self.process_order = list(range(num_agents))
        self.last_action = [4] * self.num_agents
        self.last_action_success = [1] * self.num_agents
        self.fallback_target = [None] * self.num_agents
        self.fallback_commit_steps = [0] * self.num_agents

        self.belief_visited = [set() for _ in range(num_agents)]
        self.agent_origins = []
        self.local_positions = []

        self.battery = [battery_capacity] * num_agents
        self.agent_active = [True] * num_agents

        # Policy usage counters
        self.rl_actions_used = 0
        self.fallback_actions_used = 0
        self.survival_bfs_actions_used = 0
        self.survival_rl_actions_used = 0
        self.normal_steps = 0
        self.survival_steps = 0

    def set_rl_survival_callback(self, callback):
        self._rl_survival_callback = callback

    def _add_random_obstacles(self, density: float):
        num_obstacles = int(self.width * self.height * density)
        placed, attempts = 0, 0
        while placed < num_obstacles and attempts < num_obstacles * 100:
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if self.grid[y, x] == 0:
                self.grid[y, x] = 1
                if self._is_connected():
                    placed += 1
                else:
                    self.grid[y, x] = 0
            attempts += 1

    def _is_connected(self) -> bool:
        free = [(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y, x] == 0]
        if not free:
            return False
        seen, q = {free[0]}, deque([free[0]])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 0 and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return len(seen) == len(free)

    # ---- Mode switching ----
    def switch_agent_to_survival(self, agent_id):
        self.agent_mode[agent_id] = 'survival'

    def get_agent_mode(self, agent_id):
        return self.agent_mode[agent_id]

    # ---- Server helpers ----
    def get_worker_server(self, agent_id):
        if self.agent_mode[agent_id] == 'survival' or not self.agent_active[agent_id]:
            return -1
        if not self.use_servers or self.server_network.num_servers == 0:
            return -1
        return self.server_network.get_worker_server(agent_id)

    def get_relative_position(self, agent_id):
        if self.agent_mode[agent_id] == 'survival' or not self.agent_active[agent_id]:
            lx, ly = self.local_positions[agent_id]
            return (lx, ly)
        if self.use_servers and self.server_network.num_servers > 0:
            server_id = self.server_network.get_worker_server(agent_id)
            if server_id == -1:
                return (0, 0)
            return self.server_network.get_relative_position(agent_id, self.agent_positions[agent_id])
        return (0, 0)

    def get_server_coverage(self, server_id):
        if not self.use_servers or self.server_network.num_servers == 0:
            return self.get_global_coverage()
        return self.server_network.get_server_coverage(server_id)

    def get_global_coverage(self):
        if self.free_cells == 0:
            return 1.0
        return len(self.global_visited) / self.free_cells

    def get_global_visited(self):
        return self.global_visited

    def get_belief_visited(self, agent_id):
        return self.belief_visited[agent_id]

    def reassign_normal_agents(self):
        if not self.use_servers:
            return
        self.use_servers = (self.server_network.num_servers > 0)
        if not self.use_servers:
            return
        for i in range(self.num_agents):
            if self.agent_mode[i] == 'normal' and self.agent_active[i]:
                server_id = self.server_network.assign_worker(i, self.agent_positions[i], self.grid)
                if server_id == -1:
                    self.agent_mode[i] = 'survival'

    # ---- Reset ----
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        free_cells = [(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y, x] == 0]
        if len(free_cells) < self.active_agents:
            raise RuntimeError("Not enough free cells for all active agents")
        np.random.shuffle(free_cells)

        self.agent_positions = []
        self.global_visited = set()
        self.visit_counts = defaultdict(int)
        self.action_histories = [[] for _ in range(self.num_agents)]
        self.pheromone = [np.zeros((self.height, self.width), dtype=np.float32) for _ in range(self.num_agents)]
        self.teammate_pheromone = [np.zeros((self.height, self.width), dtype=np.float32) for _ in range(self.num_agents)]
        self.coverage_patches = [np.zeros((5,5), dtype=bool) for _ in range(self.num_agents)]
        self.shared_coverage = [np.zeros((self.height, self.width), dtype=bool) for _ in range(self.num_agents)]
        self.breadcrumb = [np.zeros((self.height, self.width), dtype=bool) for _ in range(self.num_agents)]
        self.recent_positions = [deque(maxlen=7) for _ in range(self.num_agents)]
        self.idle_counters = [0] * self.num_agents
        self.process_order = list(range(self.num_agents))
        self.last_action = [4] * self.num_agents
        self.last_action_success = [1] * self.num_agents
        self.fallback_target = [None] * self.num_agents
        self.fallback_commit_steps = [0] * self.num_agents

        self.belief_visited = [set() for _ in range(self.num_agents)]
        self.agent_origins = []
        self.local_positions = []
        self.agent_mode = ['normal'] * self.num_agents
        self.battery = [self.battery_capacity] * self.num_agents
        self.agent_active = [True] * self.num_agents

        # Reset policy counters
        self.rl_actions_used = 0
        self.fallback_actions_used = 0
        self.survival_bfs_actions_used = 0
        self.survival_rl_actions_used = 0
        self.normal_steps = 0
        self.survival_steps = 0

        occupied = set()
        for i in range(self.num_agents):
            if i < self.active_agents:
                placed = False
                for cell in free_cells:
                    if cell in occupied:
                        continue
                    dist_ok = True
                    for pos in self.agent_positions:
                        if max(abs(cell[0]-pos[0]), abs(cell[1]-pos[1])) < self.min_spawn_dist:
                            dist_ok = False
                            break
                    if dist_ok:
                        self.agent_positions.append(cell)
                        self.agent_origins.append(cell)
                        self.local_positions.append((0, 0))
                        occupied.add(cell)
                        self.global_visited.add(cell)
                        self.belief_visited[i].add(cell)
                        placed = True
                        break
                if not placed:
                    for cell in free_cells:
                        if cell not in occupied:
                            self.agent_positions.append(cell)
                            self.agent_origins.append(cell)
                            self.local_positions.append((0, 0))
                            occupied.add(cell)
                            self.global_visited.add(cell)
                            self.belief_visited[i].add(cell)
                            placed = True
                            break
                if not placed:
                    raise RuntimeError(f"Could not place agent {i}")
            else:
                for cell in free_cells:
                    if cell not in occupied:
                        self.agent_positions.append(cell)
                        self.agent_origins.append(cell)
                        self.local_positions.append((0, 0))
                        occupied.add(cell)
                        self.global_visited.add(cell)
                        self.belief_visited[i].add(cell)
                        break

        if self.use_servers:
            for i in range(self.num_agents):
                server_id = self.server_network.assign_worker(i, self.agent_positions[i], self.grid)
                if server_id == -1:
                    self.agent_mode[i] = 'survival'
                else:
                    self.agent_mode[i] = 'normal'
            self.server_network.server_visited = [set() for _ in range(self.server_network.num_servers)]
            for i in range(self.active_agents):
                if self.agent_mode[i] != 'survival':
                    self.server_network.update_visited(i, self.agent_positions[i], self.grid)

        for i, (x, y) in enumerate(self.agent_positions):
            self.pheromone[i][y, x] = 1.0
            self.breadcrumb[i][y, x] = 1
            self.recent_positions[i].append((x, y))

        return self.agent_positions, self.get_global_coverage()

    # ---- BFS survival fallback ----
    def _bfs_survival_action(self, agent_id):
        if not self.agent_active[agent_id]:
            return 4
        lx, ly = self.local_positions[agent_id]
        ox, oy = self.agent_origins[agent_id]
        start = (ox + lx, oy + ly)
        belief = self.belief_visited[agent_id]
        if len(belief) >= self.free_cells:
            return 4
        q = deque()
        q.append(start)
        parent = {start: None}
        while q:
            cx, cy = q.popleft()
            if self.grid[cy, cx] == 0 and (cx, cy) not in belief:
                path = []
                cur = (cx, cy)
                while cur != start:
                    path.append(cur)
                    cur = parent[cur]
                first_step = path[-1]
                nx, ny = first_step
                if nx == start[0] and ny == start[1] - 1: return 0
                if nx == start[0] + 1 and ny == start[1]: return 1
                if nx == start[0] and ny == start[1] + 1: return 2
                if nx == start[0] - 1 and ny == start[1]: return 3
                return 4
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny, nx] == 0 and (nx, ny) not in parent:
                        occupied = False
                        for j in range(self.active_agents):
                            if j == agent_id or not self.agent_active[j]:
                                continue
                            if self.agent_positions[j] == (nx, ny):
                                occupied = True
                                break
                        if not occupied:
                            parent[(nx, ny)] = (cx, cy)
                            q.append((nx, ny))
        return 4

    def _survival_fallback_action(self, agent_id):
        if not self.agent_active[agent_id]:
            return 4
        if self.survival_policy == 'rl' and self._rl_survival_callback is not None:
            return self._rl_survival_callback(agent_id)
        else:
            return self._bfs_survival_action(agent_id)

    # ---- Local movement ----
    def _move_local(self, agent_id, action):
        if not self.agent_active[agent_id]:
            return self.local_positions[agent_id][0], self.local_positions[agent_id][1], False
        lx, ly = self.local_positions[agent_id]
        if action == 0:    nx, ny = lx, ly-1
        elif action == 1:  nx, ny = lx+1, ly
        elif action == 2:  nx, ny = lx, ly+1
        elif action == 3:  nx, ny = lx-1, ly
        else:              nx, ny = lx, ly
        ox, oy = self.agent_origins[agent_id]
        gx, gy = ox + nx, oy + ny
        if not (0 <= gx < self.width and 0 <= gy < self.height and self.grid[gy, gx] == 0):
            return lx, ly, False
        for j in range(self.active_agents):
            if j == agent_id or not self.agent_active[j]:
                continue
            if self.agent_positions[j] == (gx, gy):
                return lx, ly, False
        return nx, ny, True

    def _is_looping(self, agent_id):
        if not self.agent_active[agent_id]:
            return False
        recent = list(self.recent_positions[agent_id])
        if len(recent) < 3:
            return False
        return recent[-1] in recent[:-1]

    # ---- Normal fallback helpers ----
    def _steer_toward(self, x, y, target):
        tx, ty = target
        dx, dy = tx - x, ty - y
        if dx == 0 and dy == 0:
            return 4, None
        if abs(dx) >= abs(dy):
            primary = 1 if dx > 0 else 3
            secondary = (2 if dy > 0 else (0 if dy < 0 else None))
        else:
            primary = 2 if dy > 0 else 0
            secondary = (1 if dx > 0 else (3 if dx < 0 else None))
        return primary, secondary

    def _action_is_walkable(self, x, y, action):
        if action == 0:    nx, ny = x, y-1
        elif action == 1:  nx, ny = x+1, y
        elif action == 2:  nx, ny = x, y+1
        elif action == 3:  nx, ny = x-1, y
        else:               return True
        return 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 0

    def _bfs_path_to_nearest_unvisited(self, agent_id):
        if not self.agent_active[agent_id]:
            return 4
        x, y = self.agent_positions[agent_id]
        if self.agent_mode[agent_id] == 'survival':
            return self._survival_fallback_action(agent_id)
        visited = self.global_visited
        if len(visited) >= self.free_cells:
            return 4
        q = deque()
        q.append((x, y))
        parent = {(x, y): None}
        while q:
            cx, cy = q.popleft()
            if self.grid[cy, cx] == 0 and (cx, cy) not in visited:
                path = []
                cur = (cx, cy)
                while cur != (x, y):
                    path.append(cur)
                    cur = parent[cur]
                first_step = path[-1]
                nx, ny = first_step
                if nx == x and ny == y - 1: return 0
                if nx == x + 1 and ny == y: return 1
                if nx == x and ny == y + 1: return 2
                if nx == x - 1 and ny == y: return 3
                return 4
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny, nx] == 0 and (nx, ny) not in parent:
                        parent[(nx, ny)] = (cx, cy)
                        q.append((nx, ny))
        return 4

    def _committed_fallback_action(self, agent_id, x, y):
        if not self.agent_active[agent_id]:
            return 4
        if self.agent_mode[agent_id] == 'survival':
            return self._survival_fallback_action(agent_id)
        if self.fallback_commit_steps[agent_id] > 0 and self.fallback_target[agent_id] is not None:
            tx, ty = self.fallback_target[agent_id]
            if (tx, ty) in self.global_visited or self.grid[ty, tx] == 1:
                self.fallback_commit_steps[agent_id] = 0
                self.fallback_target[agent_id] = None
            else:
                primary, secondary = self._steer_toward(x, y, (tx, ty))
                if primary == 4:
                    self.fallback_commit_steps[agent_id] = 0
                    self.fallback_target[agent_id] = None
                    return 4
                if self._action_is_walkable(x, y, primary):
                    self.fallback_commit_steps[agent_id] -= 1
                    return primary
                elif secondary is not None and self._action_is_walkable(x, y, secondary):
                    self.fallback_commit_steps[agent_id] -= 1
                    return secondary
                else:
                    self.fallback_commit_steps[agent_id] = 0
                    self.fallback_target[agent_id] = None

        best_action = self._bfs_path_to_nearest_unvisited(agent_id)
        if best_action != 4:
            gx, gy = self.agent_positions[agent_id]
            if best_action == 0:    target = (gx, gy-1)
            elif best_action == 1:  target = (gx+1, gy)
            elif best_action == 2:  target = (gx, gy+1)
            elif best_action == 3:  target = (gx-1, gy)
            else:                   target = None
            if target is not None and self._action_is_walkable(gx, gy, best_action):
                self.fallback_target[agent_id] = target
                self.fallback_commit_steps[agent_id] = 8
                return best_action
            else:
                self.fallback_target[agent_id] = None
                self.fallback_commit_steps[agent_id] = 0
                return self._pheromone_wander(agent_id, gx, gy)

        self.fallback_target[agent_id] = None
        self.fallback_commit_steps[agent_id] = 0
        return self._pheromone_wander(agent_id, x, y)

    def _pheromone_wander(self, agent_id, x, y):
        if not self.agent_active[agent_id] or self.agent_mode[agent_id] == 'survival':
            return 4
        heat = self.pheromone[agent_id]
        dir_avg = []
        for direction, (dx_range, dy_range) in enumerate([
            (range(-2, 0), range(-2, 3)),
            (range(1, 3),  range(-2, 3)),
            (range(-2, 3), range(-2, 0)),
            (range(-2, 3), range(1, 3))
        ]):
            total = 0.0
            count = 0
            for dy in dy_range:
                for dx in dx_range:
                    gx, gy = x+dx, y+dy
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        total += heat[gy, gx]
                        count += 1
            if count > 0:
                dir_avg.append((total/count, direction))
        if dir_avg:
            dir_avg.sort()
            for _, best_dir in dir_avg:
                if best_dir == 0: candidate = 3
                elif best_dir == 1: candidate = 1
                elif best_dir == 2: candidate = 0
                else: candidate = 2
                if self._action_is_walkable(x, y, candidate):
                    return candidate
        return 4

    def _detect_oscillation(self, agent_id):
        if not self.agent_active[agent_id]:
            return False
        hist = self.action_histories[agent_id]
        if len(hist) < 4:
            return False
        a1, a2, a3, a4 = hist[-4:]
        if a1 == a3 and a2 == a4 and a1 != a2:
            OPPOSITES = {0:2, 2:0, 1:3, 3:1}
            if (a1 in OPPOSITES and OPPOSITES[a1] == a2) or (a2 in OPPOSITES and OPPOSITES[a2] == a1):
                return True
        return False

    # ---- Local coordination (added after fallback) ----
    def _local_coordination(self, actions):
        """
        Performs local coordination after fallback to prevent collisions.
        Runs only when enforce_collisions is False (i.e., we are relying
        on this mechanism for safety instead of a hard physics filter).
        """
        if not self.enforce_collisions:
            active_indices = [i for i in range(self.active_agents) if self.agent_active[i]]
            positions = [self.agent_positions[i] for i in active_indices]
            current_actions = [actions[i] for i in active_indices]

            for _ in range(10):
                changed = False
                proposed = []
                for idx in range(len(active_indices)):
                    x, y = positions[idx]
                    act = current_actions[idx]
                    if act == 0:   ny = y - 1; nx = x
                    elif act == 1: nx = x + 1; ny = y
                    elif act == 2: ny = y + 1; nx = x
                    elif act == 3: nx = x - 1; ny = y
                    else:          nx, ny = x, y
                    proposed.append((nx, ny))

                staying_positions = set()
                for idx in range(len(active_indices)):
                    if current_actions[idx] == 4:
                        staying_positions.add(positions[idx])

                for idx in range(len(active_indices)):
                    if current_actions[idx] != 4 and proposed[idx] in staying_positions:
                        current_actions[idx] = 4
                        changed = True

                # Recompute proposed after changes
                proposed = []
                for idx in range(len(active_indices)):
                    x, y = positions[idx]
                    act = current_actions[idx]
                    if act == 0:   ny = y - 1; nx = x
                    elif act == 1: nx = x + 1; ny = y
                    elif act == 2: ny = y + 1; nx = x
                    elif act == 3: nx = x - 1; ny = y
                    else:          nx, ny = x, y
                    proposed.append((nx, ny))

                staying_positions = set()
                for idx in range(len(active_indices)):
                    if current_actions[idx] == 4:
                        staying_positions.add(positions[idx])

                cell_to_agents = {}
                for idx, (nx, ny) in enumerate(proposed):
                    if current_actions[idx] != 4:
                        cell_to_agents.setdefault((nx, ny), []).append(idx)

                for cell, agents in cell_to_agents.items():
                    if len(agents) > 1:
                        agents.sort(key=lambda idx: abs(positions[idx][0] - cell[0]) + abs(positions[idx][1] - cell[1]))
                        for loser in agents[1:]:
                            current_actions[loser] = 4
                            changed = True

                for i in range(len(active_indices)):
                    if current_actions[i] == 4:
                        continue
                    for j in range(i+1, len(active_indices)):
                        if current_actions[j] == 4:
                            continue
                        if proposed[i] == positions[j] and proposed[j] == positions[i]:
                            current_actions[i] = 4
                            current_actions[j] = 4
                            changed = True

                if not changed:
                    break

            for idx, i in enumerate(active_indices):
                actions[i] = current_actions[idx]

        return actions

    # ---- STEP (MAIN) ----
    def step(self, actions, training=True):
        # Pheromone decay
        for i in range(self.active_agents):
            if self.agent_mode[i] != 'survival' and self.agent_active[i]:
                self.pheromone[i] *= self.tau_pheromone

        # Save original actions before fallback
        original_actions = list(actions)

        # Fallback override (if enabled)
        final_actions = list(actions)
        if self.apply_fallback:
            for i in range(self.active_agents):
                if self.agent_active[i] and (self.idle_counters[i] >= self.idle_threshold or self._is_looping(i)):
                    x, y = self.agent_positions[i]
                    final_actions[i] = self._committed_fallback_action(i, x, y)
        actions = final_actions

        # ====== Local coordination after fallback ======
        # Ensures safety even without hard physics filter
        actions = self._local_coordination(actions)
        # ==============================================

        # Policy usage counting
        for i in range(self.active_agents):
            if self.agent_mode[i] == 'survival':
                self.survival_steps += 1
                if self.survival_policy == 'rl' and self._rl_survival_callback is not None:
                    self.survival_rl_actions_used += 1
                else:
                    self.survival_bfs_actions_used += 1
            else:
                self.normal_steps += 1
                if actions[i] != original_actions[i]:
                    self.fallback_actions_used += 1
                else:
                    self.rl_actions_used += 1

        proposed = []
        for i in range(self.active_agents):
            x, y = self.agent_positions[i]
            action = actions[i]
            self.action_histories[i].append(action)
            if len(self.action_histories[i]) > 20:
                self.action_histories[i].pop(0)
            if action == 0:    nx, ny = x, y-1
            elif action == 1:  nx, ny = x+1, y
            elif action == 2:  nx, ny = x, y+1
            elif action == 3:  nx, ny = x-1, y
            else:              nx, ny = x, y
            proposed.append((nx, ny, action, x, y))

        # ---- Collision resolution (if enforced) ----
        occupied_this_step = set()
        for i in range(self.active_agents):
            if self.agent_active[i]:
                occupied_this_step.add(self.agent_positions[i])

        obstacle_avoidances = 0
        agent_agent_collisions = 0
        final_positions = [None] * self.num_agents
        agent_collision_events = [0] * self.num_agents

        order = [i for i in range(self.active_agents) if self.agent_active[i]]
        np.random.shuffle(order)

        for idx in order:
            i = idx
            nx, ny, action, x, y = proposed[i]

            if self.enforce_collisions:
                blocked = (nx < 0 or nx >= self.width or ny < 0 or ny >= self.height
                           or self.grid[ny, nx] == 1)
                if blocked:
                    obstacle_avoidances += 1
                    nx, ny = x, y
                    agent_collision_events[i] = 1
                else:
                    if (nx, ny) in occupied_this_step and (nx, ny) != (x, y):
                        agent_agent_collisions += 1
                        nx, ny = x, y
                        agent_collision_events[i] = 1
            else:
                # If not enforcing, local coordination already prevented collisions,
                # but we still keep agents within bounds.
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    nx, ny = x, y

            occupied_this_step.add((nx, ny))
            final_positions[i] = (nx, ny)

        for i in range(self.active_agents, self.num_agents):
            final_positions[i] = self.agent_positions[i]

        # Record last action success
        for i in range(self.active_agents):
            if not self.agent_active[i]:
                continue
            intended = (proposed[i][0], proposed[i][1])
            actual = final_positions[i]
            self.last_action[i] = actions[i]
            self.last_action_success[i] = 1 if intended == actual else 0

        # Update positions, battery, coverage
        global_visited_before = self.global_visited.copy()
        for i in range(self.active_agents):
            if not self.agent_active[i]:
                continue
            x, y = final_positions[i]
            self.agent_positions[i] = (x, y)

            action = actions[i]
            energy = self.energy_move if action != 4 else self.energy_stay
            self.battery[i] -= energy
            if self.battery[i] <= 0:
                self.battery[i] = 0.0
                self.agent_active[i] = False
                continue

            if self.agent_mode[i] == 'survival':
                ox, oy = self.agent_origins[i]
                lx, ly = x - ox, y - oy
                self.local_positions[i] = (lx, ly)
                self.belief_visited[i].add((x, y))

            if self.agent_mode[i] != 'survival':
                self.pheromone[i][y, x] += 1.0

            self.recent_positions[i].append((x, y))
            self.breadcrumb[i].fill(0)
            for (bx, by) in list(self.recent_positions[i])[-5:]:
                if 0 <= bx < self.width and 0 <= by < self.height:
                    self.breadcrumb[i][by, bx] = 1

            if self.grid[y, x] == 0:
                if (x, y) not in global_visited_before:
                    self.global_visited.add((x, y))
                    self.idle_counters[i] = 0
                    self.fallback_commit_steps[i] = 0
                    self.fallback_target[i] = None
                else:
                    self.idle_counters[i] += 1
            else:
                self.idle_counters[i] += 1

            self.visit_counts[(x, y)] += 1

            if self.agent_mode[i] != 'survival' and self.use_servers and self.server_network.num_servers > 0:
                self.server_network.update_visited(i, (x, y), self.grid)

        # Pheromone sharing
        for i in range(self.active_agents):
            if not self.agent_active[i] or self.agent_mode[i] == 'survival':
                continue
            self.teammate_pheromone[i].fill(0)
            for j in range(self.active_agents):
                if i == j or not self.agent_active[j] or self.agent_mode[j] == 'survival':
                    continue
                xi, yi = self.agent_positions[i]
                xj, yj = self.agent_positions[j]
                if max(abs(xi-xj), abs(yi-yj)) <= self.d_pheromone:
                    self.teammate_pheromone[i] = np.maximum(self.teammate_pheromone[i], self.pheromone[j])

        # Coverage patches
        for i in range(self.active_agents):
            if not self.agent_active[i]:
                self.coverage_patches[i].fill(0)
                continue
            x, y = self.agent_positions[i]
            patch = np.zeros((5,5), dtype=bool)
            visited_source = self.belief_visited[i] if self.agent_mode[i] == 'survival' else self.global_visited
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    gx, gy = x+dx, y+dy
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        patch[dy+2, dx+2] = (gx, gy) in visited_source
            self.coverage_patches[i] = patch

        # Shared coverage
        for i in range(self.active_agents):
            if not self.agent_active[i] or self.agent_mode[i] == 'survival':
                self.shared_coverage[i].fill(0)
                continue
            merged = np.zeros((self.height, self.width), dtype=bool)
            for j in range(self.active_agents):
                if j == i or not self.agent_active[j] or self.agent_mode[j] == 'survival':
                    continue
                xi, yi = self.agent_positions[i]
                xj, yj = self.agent_positions[j]
                if max(abs(xi-xj), abs(yi-yj)) <= self.d_comm:
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            gx, gy = xj+dx, yj+dy
                            if 0 <= gx < self.width and 0 <= gy < self.height:
                                if self.coverage_patches[j][dy+2, dx+2]:
                                    merged[gy, gx] = True
            self.shared_coverage[i] = merged

        # Gossip (survival agents only)
        survival_ids = [i for i in range(self.active_agents) if self.agent_mode[i] == 'survival' and self.agent_active[i]]
        for idx_i, i in enumerate(survival_ids):
            for idx_j in range(idx_i+1, len(survival_ids)):
                j = survival_ids[idx_j]
                xi, yi = self.agent_positions[i]
                xj, yj = self.agent_positions[j]
                if max(abs(xi-xj), abs(yi-yj)) <= self.d_comm:
                    merged = self.belief_visited[i] | self.belief_visited[j]
                    self.belief_visited[i] = merged.copy()
                    self.belief_visited[j] = merged.copy()

        # Reward
        team_new_cells = len(self.global_visited) - len(global_visited_before)
        K = 0.25
        k_n = 1.0
        k_ts = -1.0
        k_c = -2.0

        energy_cost = 0.0
        for i in range(self.active_agents):
            if not self.agent_active[i]:
                continue
            action = actions[i]
            energy = self.energy_move if action != 4 else self.energy_stay
            energy_cost += energy * self.energy_penalty

        rewards = []
        for i in range(self.active_agents):
            x, y = self.agent_positions[i]
            e_n = 1.0 if (x, y) in self.global_visited and self.visit_counts[(x, y)] == 1 else 0.0
            e_c = agent_collision_events[i]
            if self.active_agents > 1:
                others_new = (team_new_cells - e_n) / (self.active_agents - 1)
            else:
                others_new = 0.0
            r_j = (1 - K) * k_n * e_n + k_ts + K * k_n * others_new + k_c * e_c
            if self._detect_oscillation(i):
                r_j -= 0.01
            rewards.append(r_j)
        team_reward = sum(rewards) / self.active_agents
        team_reward -= energy_cost

        done = (self.get_global_coverage() >= 1.0)
        info = {
            'coverage': self.get_global_coverage(),
            'agent_agent_collisions': agent_agent_collisions,
            'obstacle_avoidances': obstacle_avoidances,
            'actual_agent_hits': 0,
            'actual_obstacle_hits': 0,
            'new_cells': team_new_cells,
            'agent_positions': self.agent_positions.copy(),
            'battery': self.battery[:self.active_agents].copy(),
            'active_agents': sum(self.agent_active[:self.active_agents]),
            'policy_usage': {
                'rl_actions': self.rl_actions_used,
                'fallback_actions': self.fallback_actions_used,
                'survival_bfs_actions': self.survival_bfs_actions_used,
                'survival_rl_actions': self.survival_rl_actions_used,
                'normal_steps': self.normal_steps,
                'survival_steps': self.survival_steps,
            },
        }
        if self.use_servers and self.server_network.num_servers > 0:
            info['server_coverages'] = [self.get_server_coverage(s) for s in range(self.server_network.num_servers)]
        return team_reward, done, info

    # ---- Getters ----
    def get_coverage(self) -> float:
        return self.get_global_coverage()

    def get_agent_heatmap(self, agent_id):
        if not self.agent_active[agent_id] or self.agent_mode[agent_id] == 'survival':
            return np.zeros((self.height, self.width), dtype=np.float32)
        return self.pheromone[agent_id].copy()

    def get_teammate_heatmap(self, agent_id):
        if not self.agent_active[agent_id] or self.agent_mode[agent_id] == 'survival':
            return np.zeros((self.height, self.width), dtype=np.float32)
        return self.teammate_pheromone[agent_id].copy()

    def get_shared_coverage(self, agent_id):
        if not self.agent_active[agent_id]:
            return np.zeros((self.height, self.width), dtype=float)
        return self.shared_coverage[agent_id].astype(float)

    def get_breadcrumb(self, agent_id):
        if not self.agent_active[agent_id]:
            return np.zeros((self.height, self.width), dtype=float)
        return self.breadcrumb[agent_id].astype(float)