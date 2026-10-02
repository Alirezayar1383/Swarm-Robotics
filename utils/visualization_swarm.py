import sys, os
import matplotlib
matplotlib.use('TkAgg')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv
from envs.grid_world import GridWorld
from agents.agent_wrapper import AgentWrapper
from models.dqn_network import QNetwork
from utils.visualization import SwarmVisualizerSimple
import torch
import yaml
import numpy as np

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

def load_model():
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', cfg['checkpoint_dir'], cfg['best_model_name'])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create a new model with the current scalar dimension (including battery)
    shared_q_net = QNetwork(action_dim=5, scalar_dim=SCALAR_DIM, in_channels=IN_CHANNELS).to(device)
    
    # Get the state dict from checkpoint and the current model
    state_dict = checkpoint['agent_net']
    model_state = shared_q_net.state_dict()
    
    # Check if scalar_fc weight shapes differ – if so, adapt
    if ('scalar_fc.weight' in state_dict and 
        state_dict['scalar_fc.weight'].shape != model_state['scalar_fc.weight'].shape):
        print("⚠️  Scalar dimension mismatch – adapting checkpoint to new size.")
        # Copy old weights into the new layer (first 25 dims)
        old_weight = state_dict['scalar_fc.weight']
        old_bias = state_dict['scalar_fc.bias']
        new_weight = model_state['scalar_fc.weight']
        new_bias = model_state['scalar_fc.bias']
        
        # Copy the old weights (first columns) and keep the new column randomly initialised
        new_weight[:, :old_weight.shape[1]] = old_weight
        new_bias[:old_bias.shape[0]] = old_bias
        
        # Update the state dict
        state_dict['scalar_fc.weight'] = new_weight
        state_dict['scalar_fc.bias'] = new_bias
    
    shared_q_net.load_state_dict(state_dict)
    shared_q_net.eval()
    return shared_q_net

def main():
    print("\n=== SWARM VISUALIZATION (with optional server failure) ===\n")
    try:
        map_size = int(input("Enter map size (e.g., 20): "))
    except:
        map_size = 20

    if map_size <= 10:
        suggested = 1
    elif map_size <= 25:
        suggested = 2
    else:
        suggested = 3
    print(f"\n💡 Suggested number of servers for this map size: {suggested}")

    try:
        num_agents = int(input("Enter number of agents to show (e.g., 5): "))
        if num_agents > MAX_AGENTS:
            print(f"⚠️ Capping to MAX_AGENTS={MAX_AGENTS}")
            num_agents = MAX_AGENTS
    except:
        num_agents = 5

    try:
        num_servers = int(input(f"Enter total number of servers (default {suggested}): ") or suggested)
    except:
        num_servers = suggested

    try:
        servers_to_fail = int(input(f"How many servers should fail (0 to {num_servers})? (default 0): ") or 0)
        servers_to_fail = min(max(servers_to_fail, 0), num_servers)
    except:
        servers_to_fail = 0

    try:
        failure_step = int(input("Enter failure step (0 for no failure, default half of max_steps): ") or 0)
    except:
        failure_step = 0

    print("Generating obstacle map...")
    temp_world = GridWorld(width=map_size, height=map_size, obstacle_density=cfg['obstacle_density'])
    fixed_grid = temp_world.grid

    server_positions = place_servers_optimally(map_size, map_size, fixed_grid, num_servers)

    shared_q_net = load_model()
    agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(num_agents)]

    env_config = {
        'width': map_size, 'height': map_size,
        'max_steps': map_size * map_size * cfg.get('max_steps_factor', 4),
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

    env = CoverageEnv(
        env_config,
        max_agents=MAX_AGENTS,
        active_agents=num_agents,
        num_servers=num_servers,
        server_positions=server_positions,
        obstacle_map=fixed_grid,
        worker_local_radius=WORKER_RADIUS
    )

    if failure_step > 0 and servers_to_fail > 0:
        print(f"\n👉 Server failure will occur at step {failure_step} (removing {servers_to_fail} servers).")
        print("   The visualizer will show the swarm switching to survival mode.\n")

    # ---- Custom visualizer with failure handling (already implemented inside SwarmVisualizerSimple? Not yet; we'll keep simple) ----
    # We'll just run the basic visualizer; failure can be added via subclassing if needed.
    viz = SwarmVisualizerSimple(
        env, agent_wrappers, max_steps=300, display_servers=True,
        draw_server_fov=True, server_radius=SERVER_RADIUS
    )
    viz.run()

if __name__ == "__main__":
    main()
