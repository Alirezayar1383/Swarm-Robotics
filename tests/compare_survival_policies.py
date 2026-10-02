import sys, os, yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv
from envs.grid_world import GridWorld
from agents.agent_wrapper import AgentWrapper
from models.dqn_network import QNetwork

# ---- Load config ----
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
    state_dict = checkpoint['agent_net']
    model_state = shared_q_net.state_dict()
    
    # Adapt if scalar dimension changed (battery added)
    if ('scalar_fc.weight' in state_dict and 
        state_dict['scalar_fc.weight'].shape != model_state['scalar_fc.weight'].shape):
        print("⚠️  Scalar dimension mismatch – adapting checkpoint to new size.")
        old_weight = state_dict['scalar_fc.weight']
        old_bias = state_dict['scalar_fc.bias']
        new_weight = model_state['scalar_fc.weight']
        new_bias = model_state['scalar_fc.bias']
        new_weight[:, :old_weight.shape[1]] = old_weight
        new_bias[:old_bias.shape[0]] = old_bias
        state_dict['scalar_fc.weight'] = new_weight
        state_dict['scalar_fc.bias'] = new_bias
    
    shared_q_net.load_state_dict(state_dict)
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

# ---------- Improved Filter 2 (iterative) ----------
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

# ---------- Run Test ----------
def run_test(map_size, num_agents, num_servers, failure_step, servers_to_fail=1,
             max_steps_factor=8, obstacle_map=None, survival_policy='bfs', agent_wrappers=None):
    optimal_server_positions = place_servers_optimally(map_size, map_size, obstacle_map, num_servers)

    env_config = {
        'width': map_size, 'height': map_size,
        'max_steps': map_size * map_size * max_steps_factor,
        'obstacle_density': cfg['obstacle_density'],
        'idle_threshold': cfg['idle_threshold'],
        'd_pheromone': cfg['d_pheromone'],
        'd_comm': cfg['d_comm'],
        'tau_pheromone': cfg['tau_pheromone'],
        'survival_policy': survival_policy,
        'battery_capacity': cfg.get('battery_capacity', 1000),
        'energy_move': cfg.get('energy_move', 1.0),
        'energy_stay': cfg.get('energy_stay', 0.5),
        'energy_penalty': cfg.get('energy_penalty', 0.01),
    }

    env = CoverageEnv(env_config, max_agents=MAX_AGENTS,
                      active_agents=num_agents,
                      num_servers=num_servers,
                      server_positions=optimal_server_positions,
                      obstacle_map=obstacle_map,
                      worker_local_radius=WORKER_RADIUS)

    if survival_policy == 'rl' and agent_wrappers is not None:
        def rl_survival_callback(agent_id):
            obs = env.obs_builder.build(env.world, agent_id, env.last_actions[agent_id] if env.last_actions[agent_id] is not None else -1)
            act, _ = agent_wrappers[agent_id].select_action(obs, training=False)
            return act
        env.world.set_rl_survival_callback(rl_survival_callback)

    obs_tuple, _ = env.reset()
    obs_list = list(obs_tuple)
    for a in agent_wrappers:
        a.reset_hidden()
        a.q_net.eval()

    done = False
    step = 0
    failure_triggered = False
    coverage_history = []
    server_coverage_history = [[] for _ in range(num_servers)]
    agent_positions_history = []
    active_server_count = num_servers

    total_aa_about = 0
    total_ao_about = 0
    total_aa_actual = 0
    total_ao_actual = 0
    battery_history = []
    active_history = []

    while not done and step < env.max_steps:
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
                env.world.server_network.server_positions = optimal_server_positions[to_remove:]
                active_server_count = env.world.server_network.num_servers
                for agent_id in affected_agents:
                    env.world.switch_agent_to_survival(agent_id)
                env.world.reassign_normal_agents()

        actions = []
        for i, wrapper in enumerate(agent_wrappers):
            act, _ = wrapper.select_action(obs_list[i], training=False)
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
        step += 1
        obs_list = list(obs_tuple)

        total_aa_about += info.get('agent_agent_collisions', 0)
        total_ao_about += info.get('obstacle_avoidances', 0)
        total_aa_actual += info.get('actual_agent_hits', 0)
        total_ao_actual += info.get('actual_obstacle_hits', 0)
        battery_history.append(info.get('battery', [0.0] * num_agents))
        active_history.append(info.get('active_agents', num_agents))

        coverage_history.append(info['coverage'])
        agent_positions_history.append(info['agent_positions'].copy())
        if 'server_coverages' in info:
            for s, scov in enumerate(info['server_coverages']):
                if s < len(server_coverage_history):
                    server_coverage_history[s].append(scov)

        if step % 200 == 0:
            print(f"Step {step}: Coverage = {info['coverage']*100:.1f}%")

    final_coverage = info['coverage'] if info else 0.0
    print(f"\nFinal coverage: {final_coverage*100:.1f}% in {step} steps.")

    return {
        'coverage_history': coverage_history,
        'server_coverage_history': server_coverage_history,
        'agent_positions_history': agent_positions_history,
        'final_coverage': final_coverage,
        'total_steps': step,
        'failure_triggered': failure_triggered,
        'success': final_coverage >= 0.95,
        'total_aa_about': total_aa_about,
        'total_ao_about': total_ao_about,
        'total_aa_actual': total_aa_actual,
        'total_ao_actual': total_ao_actual,
        'battery_history': battery_history,
        'active_history': active_history,
    }

# ---------- Plot ----------
def plot_comparison(results_bfs, results_rl, map_size, num_agents, num_servers, failure_step, servers_to_fail):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle(f'Survival Policy Comparison – {map_size}×{map_size}, {num_agents} agents, {num_servers} servers\nFailure at step {failure_step}, {servers_to_fail} servers removed', fontsize=14)

    # Coverage over time
    ax = axes[0, 0]
    if results_bfs:
        steps_bfs = np.arange(len(results_bfs['coverage_history']))
        ax.plot(steps_bfs, np.array(results_bfs['coverage_history']) * 100, 'b-', label='BFS Survival', linewidth=2)
    if results_rl:
        steps_rl = np.arange(len(results_rl['coverage_history']))
        ax.plot(steps_rl, np.array(results_rl['coverage_history']) * 100, 'r-', label='RL Survival', linewidth=2)
    ax.axvline(failure_step, color='black', linestyle='--', alpha=0.5, label='Failure')
    ax.set_xlabel('Step')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage over time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final coverage bar
    ax = axes[0, 1]
    labels, values = [], []
    if results_bfs:
        labels.append('BFS')
        values.append(results_bfs['final_coverage'] * 100)
    if results_rl:
        labels.append('RL')
        values.append(results_rl['final_coverage'] * 100)
    ax.bar(labels, values, color=['blue', 'red'])
    ax.set_ylabel('Final Coverage (%)')
    ax.set_title('Final Coverage')
    ax.grid(axis='y', alpha=0.3)

    # Collision metrics bar
    ax = axes[0, 2]
    metrics = ['A-A About', 'A-O About', 'A-A Actual', 'A-O Actual']
    b_values = [results_bfs['total_aa_about'], results_bfs['total_ao_about'], results_bfs['total_aa_actual'], results_bfs['total_ao_actual']]
    r_values = [results_rl['total_aa_about'], results_rl['total_ao_about'], results_rl['total_aa_actual'], results_rl['total_ao_actual']]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, b_values, width, label='BFS', color='blue')
    ax.bar(x + width/2, r_values, width, label='RL', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Collision / Avoidance Events')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Battery over time
    ax = axes[1, 0]
    for res, color, label in zip([results_bfs, results_rl], ['blue', 'red'], ['BFS', 'RL']):
        batt_avg = [np.mean(b) / 1000 * 100 for b in res['battery_history']]
        ax.plot(np.arange(len(batt_avg)), batt_avg, color=color, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg Battery (%)')
    ax.set_title('Battery Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Active agents over time
    ax = axes[1, 1]
    for res, color, label in zip([results_bfs, results_rl], ['blue', 'red'], ['BFS', 'RL']):
        ax.plot(np.arange(len(res['active_history'])), res['active_history'], color=color, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Active Agents')
    ax.set_title('Active Agents Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_lines = [
        "COMPARISON SUMMARY", "="*40,
        f"BFS Survival:",
        f"  Final Coverage:  {results_bfs['final_coverage']*100:.1f}%",
        f"  Total Steps:     {results_bfs['total_steps']}",
        f"  A-A About:       {results_bfs['total_aa_about']}",
        f"  A-O About:       {results_bfs['total_ao_about']}",
        f"  A-A Actual Hit:  {results_bfs['total_aa_actual']}",
        f"  A-O Actual Hit:  {results_bfs['total_ao_actual']}",
        "",
        f"RL Survival:",
        f"  Final Coverage:  {results_rl['final_coverage']*100:.1f}%",
        f"  Total Steps:     {results_rl['total_steps']}",
        f"  A-A About:       {results_rl['total_aa_about']}",
        f"  A-O About:       {results_rl['total_ao_about']}",
        f"  A-A Actual Hit:  {results_rl['total_aa_actual']}",
        f"  A-O Actual Hit:  {results_rl['total_ao_actual']}",
        "",
        f"Success (>=95%): BFS={'✅' if results_bfs['success'] else '❌'} | RL={'✅' if results_rl['success'] else '❌'}",
    ]
    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'videos', 'survival_policy_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")

# ---------- Main ----------
def main():
    print("\n=== COMPARE SURVIVAL POLICIES (BFS vs RL) ===\n")
    try:
        map_size = int(input("Enter map size (e.g., 20): "))
    except:
        map_size = 20
    try:
        num_agents = int(input("Enter number of agents (e.g., 5): "))
        if num_agents > MAX_AGENTS:
            print(f"⚠️ Capping to MAX_AGENTS={MAX_AGENTS}")
            num_agents = MAX_AGENTS
    except:
        num_agents = 5
    try:
        num_servers = int(input("Enter total number of servers (default 3): ") or 3)
    except:
        num_servers = 3
    try:
        servers_to_fail = int(input(f"How many servers should fail (1 to {num_servers})? (default 1): ") or 1)
        servers_to_fail = min(max(servers_to_fail, 1), num_servers)
    except:
        servers_to_fail = 1
    try:
        failure_step = int(input("Enter failure step (0 for no failure, default half of max_steps): ") or 0)
    except:
        failure_step = 0

    shared_q_net = load_model()
    agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(num_agents)]

    print("Generating obstacle map...")
    temp_grid_world = GridWorld(width=map_size, height=map_size, obstacle_density=cfg['obstacle_density'])
    fixed_grid = temp_grid_world.grid

    print("\n>> Running with BFS survival policy...")
    results_bfs = run_test(map_size, num_agents, num_servers, failure_step, servers_to_fail,
                           obstacle_map=fixed_grid, survival_policy='bfs', agent_wrappers=agent_wrappers)

    print("\n>> Running with RL survival policy...")
    results_rl = run_test(map_size, num_agents, num_servers, failure_step, servers_to_fail,
                          obstacle_map=fixed_grid, survival_policy='rl', agent_wrappers=agent_wrappers)

    plot_comparison(results_bfs, results_rl, map_size, num_agents, num_servers, failure_step, servers_to_fail)

if __name__ == "__main__":
    main()
