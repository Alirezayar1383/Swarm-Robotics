import sys, os, yaml
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # برای نمایش پنجره
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

# ========== Improved Filter 2 (Iterative) ==========
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
# ======================================================

def run_episode_and_collect(map_size=20, num_agents=5, num_servers=3, max_steps_factor=8, failure_step=0, servers_to_fail=0):
    temp_world = GridWorld(width=map_size, height=map_size, obstacle_density=cfg['obstacle_density'])
    fixed_grid = temp_world.grid

    server_positions = place_servers_optimally(map_size, map_size, fixed_grid, num_servers)

    env_config = {
        'width': map_size, 'height': map_size,
        'max_steps': map_size * map_size * max_steps_factor,
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

    env = CoverageEnv(env_config, max_agents=MAX_AGENTS, active_agents=num_agents,
                      num_servers=num_servers, server_positions=server_positions,
                      obstacle_map=fixed_grid, worker_local_radius=WORKER_RADIUS)

    shared_q_net = load_model()
    agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(num_agents)]

    obs_tuple, _ = env.reset()
    obs_list = list(obs_tuple)
    for a in agent_wrappers:
        a.reset_hidden()

    done = False
    step = 0
    total_aa_about = 0
    total_ao_about = 0
    total_aa_actual = 0
    total_ao_actual = 0
    failure_triggered = False
    active_server_count = num_servers

    while not done and step < env.max_steps:
        # Server failure
        if failure_step > 0 and step == failure_step and not failure_triggered and active_server_count > 0:
            to_remove = min(servers_to_fail, active_server_count)
            print(f"\n*** SERVER FAILURE at step {step}: Removing {to_remove} server(s) ***")
            failure_triggered = True
            if hasattr(env.world, 'server_network'):
                failing_server_ids = list(range(to_remove))
                affected_agents = []
                for i in range(num_agents):
                    server_id = env.world.server_network.get_worker_server(i)
                    if server_id in failing_server_ids:
                        affected_agents.append(i)
                env.world.server_network.num_servers = active_server_count - to_remove
                env.world.server_network.server_positions = server_positions[to_remove:]
                active_server_count = env.world.server_network.num_servers
                for agent_id in affected_agents:
                    env.world.switch_agent_to_survival(agent_id)
                env.world.reassign_normal_agents()

        actions = []
        for i in range(num_agents):
            act, _ = agent_wrappers[i].select_action(obs_list[i], training=False)
            actions.append(act)
        while len(actions) < MAX_AGENTS:
            actions.append(4)

        # Apply improved filter2
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

        total_aa_about += info.get('agent_agent_collisions', 0)
        total_ao_about += info.get('obstacle_avoidances', 0)
        total_aa_actual += info.get('actual_agent_hits', 0)
        total_ao_actual += info.get('actual_obstacle_hits', 0)

    usage = info['policy_usage']
    final_cov = info['coverage'] * 100

    print(f"\n=== Policy Usage Analysis ===")
    print(f"Map size: {map_size}×{map_size}, Agents: {num_agents}, Servers: {num_servers}, Steps: {step}")
    if failure_triggered:
        print(f"Failure at step {failure_step} – Removed {servers_to_fail} servers")
    print(f"Final Coverage: {final_cov:.1f}%")
    print(f"AA About: {total_aa_about} | AO About: {total_ao_about}")
    print(f"AA Actual: {total_aa_actual} | AO Actual: {total_ao_actual}")
    print(f"RL actions used:           {usage['rl_actions']}")
    print(f"Fallback actions used:     {usage['fallback_actions']}")
    print(f"Survival BFS actions used: {usage['survival_bfs_actions']}")
    print(f"Survival RL actions used:  {usage['survival_rl_actions']}")
    print(f"Normal agent-steps:        {usage['normal_steps']}")
    print(f"Survival agent-steps:      {usage['survival_steps']}")

    total_actions = (usage['rl_actions'] + usage['fallback_actions'] +
                     usage['survival_bfs_actions'] + usage['survival_rl_actions'])
    if total_actions > 0:
        print(f"\nPercentages:")
        print(f"  RL actions:               {usage['rl_actions']/total_actions*100:.1f}%")
        print(f"  Fallback actions:         {usage['fallback_actions']/total_actions*100:.1f}%")
        print(f"  Survival BFS actions:     {usage['survival_bfs_actions']/total_actions*100:.1f}%")
        print(f"  Survival RL actions:      {usage['survival_rl_actions']/total_actions*100:.1f}%")

    # Plot with title including steps, servers, coverage, collisions (both types)
    labels = ['RL Actions', 'Normal Fallback', 'Survival BFS', 'Survival RL']
    values = [usage['rl_actions'], usage['fallback_actions'], usage['survival_bfs_actions'], usage['survival_rl_actions']]
    colors = ['green', 'orange', 'blue', 'red']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel('Count')
    title_lines = [
        f'Policy Usage Distribution\n'
        f'Map {map_size}x{map_size} | Agents {num_agents} | Servers {num_servers} | Steps: {step}'
    ]
    if failure_triggered:
        title_lines.append(f'Failure at step {failure_step} (Removed {servers_to_fail} servers)')
    title_lines.append(f'Final Coverage: {final_cov:.1f}% | AA About: {total_aa_about} | AA Actual: {total_aa_actual} | AO About: {total_ao_about} | AO Actual: {total_ao_actual}')
    plt.title('\n'.join(title_lines), fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'videos', 'policy_usage.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.show()

if __name__ == "__main__":

    run_episode_and_collect(map_size=50, num_agents=10, num_servers=10, failure_step=430, servers_to_fail=7)
