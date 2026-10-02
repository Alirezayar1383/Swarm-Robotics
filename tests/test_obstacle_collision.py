import sys, os, yaml
import numpy as np

# ---- Add project root to path ----
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv

# ---- Load config (adjust path if needed) ----
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase9_config.yaml')
if not os.path.exists(config_path):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase4_2_config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

def test_no_agent_on_obstacle():
    map_size = 20
    num_agents = 5
    env_config = {
        'width': map_size, 'height': map_size,
        'max_steps': map_size * map_size * 8,
        'obstacle_density': cfg['obstacle_density'],
        'idle_threshold': cfg['idle_threshold'],
        'd_pheromone': cfg['d_pheromone'],
        'd_comm': cfg['d_comm'],
        'tau_pheromone': cfg['tau_pheromone'],
        'battery_capacity': cfg.get('battery_capacity', 1000),
        'energy_move': cfg.get('energy_move', 1.0),
        'energy_stay': cfg.get('energy_stay', 0.5),
        'energy_penalty': cfg.get('energy_penalty', 0.01),
    }
    # Create environment with correct parameter name
    env = CoverageEnv(
        env_config,
        max_agents=cfg['max_agents'],
        active_agents=num_agents,
        num_servers=3,
        worker_local_radius=cfg['worker_local_radius']  # <-- FIX
    )
    env.reset()
    for step in range(200):
        actions = [np.random.randint(0,5) for _ in range(cfg['max_agents'])]
        obs, r, d, t, info = env.step(actions)
        for i, (x, y) in enumerate(info['agent_positions']):
            if i < num_agents:
                assert env.world.grid[y, x] == 0, f"Agent {i} inside obstacle at ({x},{y})"
    print("✅ All agents remained on free cells for 200 steps.")

if __name__ == "__main__":
    test_no_agent_on_obstacle()
