import sys, os, yaml
import numpy as np
import matplotlib

# ✅ FIXED: Try interactive backend, fall back to Agg for headless environments
try:
    matplotlib.use('TkAgg')
    INTERACTIVE = True
except Exception:
    matplotlib.use('Agg')
    INTERACTIVE = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

# ✅ FIXED: sys.path must be set BEFORE project imports
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
WORKER_RADIUS = cfg['worker_local_radius']   # 5x5 window (radius=2)
SERVER_RADIUS = cfg['server_local_radius']   # 11x11 window (radius=5)
IN_CHANNELS = 12 + (MAX_AGENTS - 1)
SCALAR_DIM = 15 + MAX_AGENTS + 1   # +1 for battery

# ✅ FIXED: Paths for both checkpoints
STAGE_FINAL_PATH = os.path.join(
    os.path.dirname(__file__), '..', cfg['checkpoint_dir'], 'stage_final.pt'
)
BEST_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', cfg['checkpoint_dir'], cfg['best_model_name']
)


# ============================================================================
# ✅ FIXED: load_model() prefers stage_final.pt (exact end-of-curriculum
#    network) over best_ctde.pt.
# ============================================================================
def load_model():
    if os.path.exists(STAGE_FINAL_PATH):
        checkpoint_path = STAGE_FINAL_PATH
        print(f"✅ Loading model from END-OF-CURRICULUM: {checkpoint_path}")
    else:
        checkpoint_path = BEST_MODEL_PATH
        print(f"⚠️  stage_final.pt not found, falling back to best_ctde.pt: {checkpoint_path}")

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


# ============================================================================
# Server placement — greedy optimal
# ============================================================================
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


def get_server_positions(map_size, num_servers, grid=None):
    if grid is None:
        if num_servers == 1:
            return [(map_size // 2, map_size // 2)]
        elif num_servers == 2:
            return [(map_size // 4, map_size // 2), (3 * map_size // 4, map_size // 2)]
        else:
            positions = []
            for i in range(num_servers):
                x = (i + 1) * map_size // (num_servers + 1)
                y = map_size // 2
                positions.append((x, y))
            return positions
    return place_servers_optimally(map_size, map_size, grid, num_servers)


# ============================================================================
# Improved Filter 2 (iterative local coordination)
# ============================================================================
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
        # Block head-on swaps
        for i in range(num_agents):
            if not active[i] or actions[i] == 4:
                continue
            for j in range(i + 1, num_agents):
                if not active[j] or actions[j] == 4:
                    continue
                if proposed[i] == positions[j] and proposed[j] == positions[i]:
                    actions[i] = 4
                    actions[j] = 4
                    changed = True
        if not changed:
            break
    return actions


# ============================================================================
# Run a single episode and record the trajectory for later animation
# ============================================================================
def run_episode_and_record(map_size, num_agents, num_servers, failure_step,
                            servers_to_fail=1, max_steps_factor=8, obstacle_map=None):
    optimal_server_positions = get_server_positions(map_size, num_servers, obstacle_map)

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

    shared_q_net = load_model()
    agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(num_agents)]

    env = CoverageEnv(env_config, max_agents=MAX_AGENTS,
                      active_agents=num_agents,
                      num_servers=num_servers,
                      server_positions=optimal_server_positions,
                      obstacle_map=obstacle_map,
                      worker_local_radius=WORKER_RADIUS)

    obs_tuple, _ = env.reset()
    obs_list = list(obs_tuple)
    for a in agent_wrappers:
        a.reset_hidden()
        a.q_net.eval()

    done = False
    step = 0
    failure_triggered = False
    active_server_count = num_servers

    positions_history = []
    coverage_history = []
    server_positions_history = []
    aa_collision_history = []
    ao_collision_history = []
    visited_history = []
    battery_history = []
    active_history = []
    battery_off_history = []

    while not done and step < env.max_steps:
        # ---- Server failure event ----
        if failure_step > 0 and step == failure_step and not failure_triggered and active_server_count > 0:
            to_remove = min(servers_to_fail, active_server_count)
            print(f"*** SERVER FAILURE at step {step}: Removing {to_remove} server(s) ***")
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

        # ---- Collect actions ----
        actions = []
        for i, wrapper in enumerate(agent_wrappers):
            act, _ = wrapper.select_action(obs_list[i], training=False)
            actions.append(act)
        while len(actions) < MAX_AGENTS:
            actions.append(4)

        # ---- Apply Filter 2 (Local Coordination) ----
        filtered = improved_filter2(
            actions[:num_agents],
            env.world.agent_positions[:num_agents],
            env.world.agent_active[:num_agents]
        )
        actions[:num_agents] = filtered

        # ---- Step environment ----
        obs_tuple, _, term, trunc, info = env.step(actions, training=False)
        done = term or trunc
        step += 1
        obs_list = list(obs_tuple)

        positions_history.append(env.world.agent_positions[:num_agents].copy())
        coverage_history.append(info['coverage'])
        aa_collision_history.append(info.get('agent_agent_collisions', 0))
        ao_collision_history.append(info.get('obstacle_avoidances', 0))
        visited_history.append(env.world.global_visited.copy())
        battery_history.append(info.get('battery', [0.0] * num_agents))
        active_history.append(info.get('active_agents', num_agents))
        battery_off_history.append(sum(1 for b in info.get('battery', []) if b <= 0))
        if hasattr(env.world, 'server_network'):
            server_positions_history.append(env.world.server_network.server_positions.copy())
        else:
            server_positions_history.append([])

    final_coverage = info['coverage'] if info else 0.0
    print(f"Episode finished: Coverage {final_coverage * 100:.1f}% in {step} steps.")

    return {
        'positions_history': positions_history,
        'coverage_history': coverage_history,
        'server_positions_history': server_positions_history,
        'aa_collision_history': aa_collision_history,
        'ao_collision_history': ao_collision_history,
        'visited_history': visited_history,
        'final_coverage': final_coverage,
        'total_steps': step,
        'failure_triggered': failure_triggered,
        'failure_step': failure_step if failure_triggered else None,
        'grid': env.world.grid.copy(),
        'battery_history': battery_history,
        'active_history': active_history,
        'battery_off_history': battery_off_history,
    }


# ============================================================================
# Draw a single frame on the given axes
# ============================================================================
def draw_map(ax, grid, data, frame, agent_colors, server_color,
             server_radius=SERVER_RADIUS):
    h, w = grid.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    visited = data['visited_history'][frame] if frame < len(data['visited_history']) else set()

    for (x, y) in visited:
        if 0 <= x < w and 0 <= y < h and grid[y, x] == 0:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                       facecolor='lightgreen', edgecolor='none', alpha=0.5))

    for y in range(h):
        for x in range(w):
            if grid[y, x] == 1:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                           facecolor='black', edgecolor='none'))

    if frame < len(data['positions_history']):
        positions = data['positions_history'][frame]
        battery = data['battery_history'][frame] if frame < len(data['battery_history']) else []
        for i, (x, y) in enumerate(positions):
            if 0 <= x < w and 0 <= y < h and grid[y, x] == 0:
                ax.add_patch(plt.Circle((x, y), 0.35,
                                        color=agent_colors[i % len(agent_colors)],
                                        ec='white', lw=1.5))
                ax.text(x, y - 0.45, str(i), color='black', fontsize=8,
                        ha='center', va='center', weight='bold')
                if i < len(battery):
                    batt = battery[i] / 1000
                    ax.add_patch(plt.Rectangle((x - 0.25, y + 0.35), 0.5, 0.08,
                                               facecolor='gray', edgecolor='none'))
                    ax.add_patch(plt.Rectangle((x - 0.25, y + 0.35), 0.5 * batt, 0.08,
                                               facecolor='green' if batt > 0.3 else 'red',
                                               edgecolor='none'))

    if frame < len(data['server_positions_history']):
        servers = data['server_positions_history'][frame]
        for sx, sy in servers:
            if 0 <= sx < w and 0 <= sy < h:
                ax.add_patch(plt.Rectangle((sx - server_radius - 0.5, sy - server_radius - 0.5),
                                           2 * server_radius + 1, 2 * server_radius + 1,
                                           facecolor='none', edgecolor='gold',
                                           linestyle='--', lw=1, alpha=0.6))
                ax.add_patch(plt.Rectangle((sx - 0.5, sy - 0.5), 1, 1,
                                           facecolor=server_color, edgecolor='black',
                                           lw=1, alpha=0.8))
                ax.text(sx, sy, 'S', color='black', fontsize=8,
                        ha='center', va='center')


# ============================================================================
# Animation: side-by-side comparison (no-failure vs with-failure)
# ============================================================================
def animate_comparison(map_size, num_agents, num_servers, servers_to_fail,
                        failure_step, max_steps_factor=8):
    print("Generating obstacle map...")
    temp_grid_world = GridWorld(width=map_size, height=map_size,
                                 obstacle_density=cfg['obstacle_density'])
    fixed_grid = temp_grid_world.grid

    print("Running baseline (no failure)...")
    data_no_fail = run_episode_and_record(map_size, num_agents, num_servers, 0, 0,
                                           max_steps_factor, obstacle_map=fixed_grid)
    print("Running with failure...")
    data_fail = run_episode_and_record(map_size, num_agents, num_servers,
                                        failure_step, servers_to_fail,
                                        max_steps_factor, obstacle_map=fixed_grid)

    max_steps = max(len(data_no_fail['positions_history']),
                    len(data_fail['positions_history']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Robustness Comparison – {map_size}×{map_size}, "
                 f"{num_agents} agents, {num_servers} servers", fontsize=14)

    grid = fixed_grid
    agent_colors = ['red', 'blue', 'green', 'orange', 'purple',
                    'cyan', 'magenta', 'brown', 'lime', 'pink']
    server_color = 'gold'

    # Mutable counters (updated inside closure)
    cum_aa_no = [0]
    cum_ao_no = [0]
    cum_aa_fail = [0]
    cum_ao_fail = [0]

    def update(frame):
        ax1.clear()
        ax2.clear()

        # ---- LEFT: Without Failure ----
        ax1.set_title("Without Failure", fontsize=12)
        draw_map(ax1, grid, data_no_fail, frame, agent_colors, server_color)
        cov_no = data_no_fail['coverage_history'][frame] if frame < len(data_no_fail['coverage_history']) else 1.0
        active_no = data_no_fail['active_history'][frame] if frame < len(data_no_fail['active_history']) else num_agents
        batt_no = np.mean(data_no_fail['battery_history'][frame]) / 1000 * 100 if frame < len(data_no_fail['battery_history']) else 100
        batt_off_no = data_no_fail['battery_off_history'][frame] if frame < len(data_no_fail['battery_off_history']) else 0

        if frame < len(data_no_fail['aa_collision_history']):
            cum_aa_no[0] += data_no_fail['aa_collision_history'][frame]
        if frame < len(data_no_fail['ao_collision_history']):
            cum_ao_no[0] += data_no_fail['ao_collision_history'][frame]

        ax1.text(0.02, 0.98, f"Cov: {cov_no * 100:.1f}%",
                 transform=ax1.transAxes, va='top', ha='left', fontsize=10, color='blue')
        ax1.text(0.02, 0.92, f"Step: {frame}",
                 transform=ax1.transAxes, va='top', ha='left', fontsize=9)
        ax1.text(0.02, 0.86, f"A-A Coll: {cum_aa_no[0]}",
                 transform=ax1.transAxes, va='top', ha='left', fontsize=9, color='red')
        ax1.text(0.02, 0.80, f"Obs Coll: {cum_ao_no[0]}",
                 transform=ax1.transAxes, va='top', ha='left', fontsize=9, color='orange')
        ax1.text(0.02, 0.74, f"Active: {active_no}/{num_agents}",
                 transform=ax1.transAxes, va='top', ha='left', fontsize=9, color='green')
        ax1.text(0.02, 0.68, f"Batt: {batt_no:.0f}%",
                 transform=ax1.transAxes, va='top', ha='left', fontsize=9, color='purple')
        ax1.text(0.02, 0.62, f"Dead: {batt_off_no}",
                 transform=ax1.transAxes, va='top', ha='left', fontsize=9, color='red')

        # ---- RIGHT: With Failure ----
        ax2.set_title("With Failure", fontsize=12)
        draw_map(ax2, grid, data_fail, frame, agent_colors, server_color)
        cov_fail = data_fail['coverage_history'][frame] if frame < len(data_fail['coverage_history']) else 1.0
        active_fail = data_fail['active_history'][frame] if frame < len(data_fail['active_history']) else num_agents
        batt_fail = np.mean(data_fail['battery_history'][frame]) / 1000 * 100 if frame < len(data_fail['battery_history']) else 100
        batt_off_fail = data_fail['battery_off_history'][frame] if frame < len(data_fail['battery_off_history']) else 0

        if frame < len(data_fail['aa_collision_history']):
            cum_aa_fail[0] += data_fail['aa_collision_history'][frame]
        if frame < len(data_fail['ao_collision_history']):
            cum_ao_fail[0] += data_fail['ao_collision_history'][frame]

        ax2.text(0.02, 0.98, f"Cov: {cov_fail * 100:.1f}%",
                 transform=ax2.transAxes, va='top', ha='left', fontsize=10, color='blue')
        ax2.text(0.02, 0.92, f"Step: {frame}",
                 transform=ax2.transAxes, va='top', ha='left', fontsize=9)
        ax2.text(0.02, 0.86, f"A-A Coll: {cum_aa_fail[0]}",
                 transform=ax2.transAxes, va='top', ha='left', fontsize=9, color='red')
        ax2.text(0.02, 0.80, f"Obs Coll: {cum_ao_fail[0]}",
                 transform=ax2.transAxes, va='top', ha='left', fontsize=9, color='orange')
        ax2.text(0.02, 0.74, f"Active: {active_fail}/{num_agents}",
                 transform=ax2.transAxes, va='top', ha='left', fontsize=9, color='green')
        ax2.text(0.02, 0.68, f"Batt: {batt_fail:.0f}%",
                 transform=ax2.transAxes, va='top', ha='left', fontsize=9, color='purple')
        ax2.text(0.02, 0.62, f"Dead: {batt_off_fail}",
                 transform=ax2.transAxes, va='top', ha='left', fontsize=9, color='red')

        if data_fail['failure_triggered'] and frame >= data_fail['failure_step']:
            ax2.text(0.5, 0.05, "FAILURE", transform=ax2.transAxes, ha='center', va='bottom',
                     color='red', fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7))

        return ax1, ax2

    ani = FuncAnimation(fig, update, frames=max_steps, interval=100, blit=False)
    plt.tight_layout()
    gif_path = os.path.join(os.path.dirname(__file__), '..', 'videos',
                             'robustness_animation.gif')
    ani.save(gif_path, writer='pillow', fps=10)
    print(f"Animation saved to {gif_path}")

    # ✅ FIXED: Only show interactively if GUI is available
    if INTERACTIVE:
        plt.show()
    else:
        print("(Non-interactive backend — GIF saved but not shown)")

    plt.close(fig)


# ============================================================================
# Main entry point
# ============================================================================
if __name__ == "__main__":
    try:
        map_size = int(input("Enter map size (e.g., 20): "))
    except Exception:
        map_size = 20
    try:
        num_agents = int(input("Enter number of agents (e.g., 5): "))
    except Exception:
        num_agents = 5
    try:
        num_servers = int(input("Enter total number of servers (default 3): ") or 3)
    except Exception:
        num_servers = 3
    try:
        servers_to_fail = int(input(f"How many servers should fail (1 to {num_servers})? (default 1): ") or 1)
        servers_to_fail = min(max(servers_to_fail, 1), num_servers)
    except Exception:
        servers_to_fail = 1
    try:
        failure_step = int(input("Enter failure step (0 for no failure, default half of max_steps): ") or 0)
    except Exception:
        failure_step = 0
    if failure_step == 0:
        print("Failure step 0 – no failure will occur.")
    print(f"Animating: {map_size}×{map_size} map, {num_agents} agents, "
          f"{num_servers} servers, fail {servers_to_fail} at step {failure_step}.")
    animate_comparison(map_size, num_agents, num_servers, servers_to_fail, failure_step)