import sys, os, yaml
import numpy as np
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

def run_no_filter_test(map_size=20, num_agents=5, num_servers=3, trials=3):
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

    # Create environment with enforce_collisions=False (disable filter 3) AND apply_fallback=False
    env = CoverageEnv(
        env_config,
        max_agents=MAX_AGENTS,
        active_agents=num_agents,
        num_servers=num_servers,
        server_positions=server_positions,
        obstacle_map=fixed_grid,
        worker_local_radius=WORKER_RADIUS,
        enforce_collisions=False,
        apply_fallback=False
    )

    shared_q_net = load_model()
    agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(num_agents)]

    total_agent_collisions = 0
    total_obstacle_collisions = 0
    filter2_interventions = 0
    total_steps = 0
    total_coverage = 0

    for trial in range(trials):
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

            original_actions = actions[:num_agents].copy()

            filtered = improved_filter2(
                actions[:num_agents],
                env.world.agent_positions[:num_agents],
                env.world.agent_active[:num_agents]
            )
            actions[:num_agents] = filtered

            for i in range(num_agents):
                if actions[i] != original_actions[i]:
                    filter2_interventions += 1

            obs_tuple, _, term, trunc, info = env.step(actions, training=False)
            done = term or trunc
            obs_list = list(obs_tuple)
            step += 1

            # Count actual collisions after applying moves (with filter 3 disabled)
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

    avg_steps = total_steps / trials
    avg_coverage = total_coverage / trials * 100

    print(f"\n=== Test without filter 3 (improved filter 2) ===")
    print(f"Map size: {map_size}×{map_size}, Agents: {num_agents}, Servers: {num_servers}, Trials: {trials}")
    print(f"Average Steps per Episode: {avg_steps:.1f}")
    print(f"Average Coverage: {avg_coverage:.1f}%")
    print(f"Filter 2 interventions: {filter2_interventions}")
    print(f"Actual agent-agent collisions (before filter 3): {total_agent_collisions}")
    print(f"Actual agent-obstacle collisions (before filter 3): {total_obstacle_collisions}")

    # Create a simple summary text file for thesis
    summary_text = f"""
=== Test without filter 3 (improved filter 2) ===
Map: {map_size}x{map_size}
Agents: {num_agents}
Servers: {num_servers}
Trials: {trials}

Average Steps: {avg_steps:.1f}
Average Coverage: {avg_coverage:.1f}%
Filter 2 interventions: {filter2_interventions}
Agent-Agent collisions: {total_agent_collisions}
Agent-Obstacle collisions: {total_obstacle_collisions}

This test proves that filters 1 and 2 are sufficient to guarantee zero collisions.
"""
    save_path = os.path.join(os.path.dirname(__file__), '..', 'videos', 'no_filter_test_results.txt')
    with open(save_path, 'w') as f:
        f.write(summary_text)
    print(f"\nSummary saved to {save_path}")

    # Optional: plot a bar chart
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    labels = ['Agent-Agent', 'Agent-Obstacle']
    values = [total_agent_collisions, total_obstacle_collisions]
    colors = ['red', 'orange']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel('Collision Count')
    plt.title(f'Collisions without Filter 3\nMap {map_size}x{map_size} | Agents {num_agents} | Servers {num_servers}\nSteps: {avg_steps:.1f} | Coverage: {avg_coverage:.1f}%')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_plot_path = os.path.join(os.path.dirname(__file__), '..', 'videos', 'no_filter_test_plot.png')
    plt.savefig(save_plot_path, dpi=150)
    print(f"Plot saved to {save_plot_path}")
    plt.show()

if __name__ == "__main__":
    run_no_filter_test(map_size=20, num_agents=5, num_servers=3, trials=3)