import sys, os, yaml
import numpy as np
import matplotlib
# تلاش برای بک‌اند تعاملی
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv
from envs.grid_world import GridWorld
from agents.agent_wrapper import AgentWrapper
from models.dqn_network import QNetwork

config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase9_config.yaml')
if not os.path.exists(config_path):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase4_2_config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_AGENTS = cfg['max_agents']
WORKER_RADIUS = cfg['worker_local_radius']
SERVER_RADIUS = cfg['server_local_radius']
IN_CHANNELS = 12 + (MAX_AGENTS - 1)
SCALAR_DIM = 15 + MAX_AGENTS + 1

def load_model():
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', cfg['checkpoint_dir'], cfg['best_model_name'])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    shared_q_net = QNetwork(action_dim=5, scalar_dim=SCALAR_DIM, in_channels=IN_CHANNELS).to(device)
    shared_q_net.load_state_dict(checkpoint['agent_net'])
    shared_q_net.eval()
    return shared_q_net

def place_servers_optimally(width, height, grid, num_servers, radius=SERVER_RADIUS):
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
                    for dy in range(-radius, radius+1):
                        for dx in range(-radius, radius+1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0 and not covered[ny, nx]:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_pos = (x, y)
        if best_pos is None:
            break
        chosen_positions.append(best_pos)
        bx, by = best_pos
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx, ny = bx + dx, by + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0:
                    covered[ny, nx] = True
    return chosen_positions

# ================== Improved Filter 2 (Iterative) ==================
def improved_filter2(actions, positions, active):
    actions = actions.copy()
    num_agents = len(actions)
    for i in range(num_agents):
        if not active[i]:
            actions[i] = 4
    for _ in range(10):
        changed = False
        proposed = []
        for i in range(num_agents):
            x, y = positions[i]
            act = actions[i]
            if act == 0:   ny = y - 1; nx = x
            elif act == 1: nx = x + 1; ny = y
            elif act == 2: ny = y + 1; nx = x
            elif act == 3: nx = x - 1; ny = y
            else:          nx, ny = x, y
            proposed.append((nx, ny))
        staying_positions = set()
        for i in range(num_agents):
            if (not active[i]) or actions[i] == 4:
                staying_positions.add(positions[i])
        for i in range(num_agents):
            if active[i] and actions[i] != 4:
                if proposed[i] in staying_positions:
                    actions[i] = 4
                    changed = True
        # Recompute
        proposed = []
        for i in range(num_agents):
            x, y = positions[i]
            act = actions[i]
            if act == 0:   ny = y - 1; nx = x
            elif act == 1: nx = x + 1; ny = y
            elif act == 2: ny = y + 1; nx = x
            elif act == 3: nx = x - 1; ny = y
            else:          nx, ny = x, y
            proposed.append((nx, ny))
        staying_positions = set()
        for i in range(num_agents):
            if (not active[i]) or actions[i] == 4:
                staying_positions.add(positions[i])
        cell_to_agents = {}
        for i, (nx, ny) in enumerate(proposed):
            if active[i] and actions[i] != 4:
                cell_to_agents.setdefault((nx, ny), []).append(i)
        for cell, agents in cell_to_agents.items():
            if len(agents) > 1:
                agents.sort(key=lambda i: abs(positions[i][0] - cell[0]) + abs(positions[i][1] - cell[1]))
                for loser in agents[1:]:
                    actions[loser] = 4
                    changed = True
        # Block swaps
        for i in range(num_agents):
            if not active[i] or actions[i] == 4:
                continue
            for j in range(i+1, num_agents):
                if not active[j] or actions[j] == 4:
                    continue
                if proposed[i] == positions[j] and proposed[j] == positions[i]:
                    actions[i] = 4
                    actions[j] = 4
                    changed = True
        if not changed:
            break
    return actions
# =====================================================================

def run_scenario(map_size, num_agents, num_servers, mask_enabled, filter2_enabled, trials=3):
    """
    Runs a scenario with given settings.
    Returns: total_agent_collisions, total_obstacle_collisions, total_steps, total_coverage
    """
    temp_world = GridWorld(width=map_size, height=map_size, obstacle_density=cfg['obstacle_density'])
    fixed_grid = temp_world.grid

    # Simple server placement
    if num_servers == 1:
        server_positions = [(map_size//2, map_size//2)]
    elif num_servers == 2:
        server_positions = [(map_size//4, map_size//2), (3*map_size//4, map_size//2)]
    else:
        server_positions = [(map_size//4, map_size//2), (map_size//2, map_size//2), (3*map_size//4, map_size//2)]

    env_config = {
        'width': map_size, 'height': map_size,
        'max_steps': map_size * map_size * 8,
        'obstacle_density': cfg['obstacle_density'],
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

    # Disable filter3 (enforce_collisions=False) and fallback (apply_fallback=True)
    env = CoverageEnv(
        env_config,
        max_agents=MAX_AGENTS,
        active_agents=num_agents,
        num_servers=num_servers,
        server_positions=server_positions,
        obstacle_map=fixed_grid,
        worker_local_radius=WORKER_RADIUS,
        enforce_collisions=False,
        apply_fallback=True
    )

    shared_q_net = load_model()
    agent_wrappers = [AgentWrapper(i, shared_q_net, device, mask_enabled=mask_enabled) for i in range(num_agents)]

    total_agent_collisions = 0
    total_obstacle_collisions = 0
    total_steps = 0
    total_coverage = 0

    for _ in range(trials):
        obs_tuple, _ = env.reset()
        obs_list = list(obs_tuple)
        for a in agent_wrappers:
            a.reset_hidden()

        done = False
        step = 0
        while not done and step < env.max_steps:
            actions = []
            for i in range(num_agents):
                if env.world.agent_active[i]:
                    act, _ = agent_wrappers[i].select_action(obs_list[i], training=False)
                    actions.append(act)
                else:
                    actions.append(4)
            while len(actions) < MAX_AGENTS:
                actions.append(4)

            # Apply filter2 if requested
            if filter2_enabled:
                filtered = improved_filter2(
                    actions[:num_agents],
                    env.world.agent_positions[:num_agents],
                    env.world.agent_active[:num_agents]
                )
                actions[:num_agents] = filtered

            obs_tuple, _, term, trunc, info = env.step(actions, training=False)
            done = term or trunc
            obs_list = list(obs_tuple)
            step += 1

            # Count collisions (agent-obstacle, agent-agent)
            positions = env.world.agent_positions[:num_agents]
            for i in range(num_agents):
                x, y = positions[i]
                if env.world.grid[y, x] == 1:
                    total_obstacle_collisions += 1
            for i in range(num_agents):
                for j in range(i+1, num_agents):
                    if positions[i] == positions[j]:
                        total_agent_collisions += 1

        total_steps += step
        total_coverage += info['coverage']

    return total_agent_collisions, total_obstacle_collisions, total_steps, total_coverage

if __name__ == "__main__":
    print("=== Filter Contribution Analysis (without filter 3) ===\n")

    map_size = 60
    num_agents = 10
    num_servers = 10
    trials = 3

    # Scenario A: no filter1, no filter2
    print("Running Scenario A: no filters...")
    aa_A, ao_A, steps_A, cov_A = run_scenario(map_size, num_agents, num_servers,
                                               mask_enabled=False, filter2_enabled=False, trials=trials)
    print(f"  Agent-Agent collisions: {aa_A}, Agent-Obstacle collisions: {ao_A}, Steps: {steps_A}, Coverage: {cov_A/trials*100:.1f}%")

    # Scenario B: only filter1
    print("Running Scenario B: filter1 only...")
    aa_B, ao_B, steps_B, cov_B = run_scenario(map_size, num_agents, num_servers,
                                               mask_enabled=True, filter2_enabled=False, trials=trials)
    print(f"  Agent-Agent collisions: {aa_B}, Agent-Obstacle collisions: {ao_B}, Steps: {steps_B}, Coverage: {cov_B/trials*100:.1f}%")

    # Scenario C: filter1 + filter2
    print("Running Scenario C: filter1 + filter2...")
    aa_C, ao_C, steps_C, cov_C = run_scenario(map_size, num_agents, num_servers,
                                               mask_enabled=True, filter2_enabled=True, trials=trials)
    print(f"  Agent-Agent collisions: {aa_C}, Agent-Obstacle collisions: {ao_C}, Steps: {steps_C}, Coverage: {cov_C/trials*100:.1f}%")

    # Compute prevented counts
    total_potential_aa = aa_A
    total_potential_ao = ao_A
    filter1_prev_aa = aa_A - aa_B
    filter1_prev_ao = ao_A - ao_B
    filter2_prev_aa = aa_B - aa_C
    filter2_prev_ao = ao_B - ao_C
    remaining_aa = aa_C
    remaining_ao = ao_C

    print("\n=== Summary ===")
    print(f"Total potential collisions (no filters): AA={total_potential_aa}, AO={total_potential_ao}")
    print(f"Filter1 prevented: AA={filter1_prev_aa}, AO={filter1_prev_ao}")
    print(f"Filter2 prevented: AA={filter2_prev_aa}, AO={filter2_prev_ao}")
    print(f"Remaining after both filters: AA={remaining_aa}, AO={remaining_ao}")

    # Plot 1: Bar chart of contributions
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Potential', 'Filter1 Prevented', 'Filter2 Prevented', 'Remaining']
    aa_vals = [total_potential_aa, filter1_prev_aa, filter2_prev_aa, remaining_aa]
    ao_vals = [total_potential_ao, filter1_prev_ao, filter2_prev_ao, remaining_ao]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, aa_vals, width, label='Agent-Agent', color='red')
    ax.bar(x + width/2, ao_vals, width, label='Agent-Obstacle', color='orange')

    ax.set_xlabel('Filter Contribution')
    ax.set_ylabel('Collision Count')
    ax.set_title(f'Filter Contribution Analysis (Map {map_size}×{map_size}, {num_agents} agents, {num_servers} servers, {trials} trials)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'videos', 'filter_contribution.png')
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()

    # Plot 2: Line chart showing step and coverage per scenario (optional)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    scenarios = ['No Filters', 'Filter1 Only', 'Filters 1+2']
    steps = [steps_A/trials, steps_B/trials, steps_C/trials]
    covs = [cov_A/trials*100, cov_B/trials*100, cov_C/trials*100]
    ax2.bar(scenarios, steps, color='blue', alpha=0.6, label='Avg Steps')
    ax2.set_ylabel('Steps', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax3 = ax2.twinx()
    ax3.plot(scenarios, covs, 'ro-', linewidth=2, label='Avg Coverage (%)')
    ax3.set_ylabel('Coverage (%)', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    ax2.set_title('Steps and Coverage across Scenarios')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path2 = os.path.join(os.path.dirname(__file__), '..', 'videos', 'filter_contribution_steps_coverage.png')
    plt.savefig(save_path2, dpi=150)
    print(f"Second plot saved to {save_path2}")
    plt.show()