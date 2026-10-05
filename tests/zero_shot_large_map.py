"""
Fast zero-shot evaluation for LARGE maps only.

Speeds up the evaluation by:
    1. Testing only large map sizes (default: 50, 55, 60, 65).
    2. Using fewer trials (default: 2 instead of 3).
    3. Skipping failure-tolerance tests entirely.
    4. Early-terminating episodes when all agents run out of battery
       (a huge speedup for large maps where agents die before finishing).
    5. Running all agent counts [1, 3, 5, 8, 10] regardless of the
       `large_map_threshold` restriction in config.

Usage:
    python zero_shot_large_maps.py
    python zero_shot_large_maps.py --trials 1 --sizes 60 65
    python zero_shot_large_maps.py --turbo           # fastest mode
"""

import sys, os, yaml, time, argparse, torch, numpy as np
import matplotlib

try:
    matplotlib.use('TkAgg')
    INTERACTIVE = True
except Exception:
    matplotlib.use('Agg')
    INTERACTIVE = False
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv
from envs.grid_world import GridWorld
from models.dqn_network import QNetwork
from agents.agent_wrapper import AgentWrapper


# ============================================================================
# Command-line arguments
# ============================================================================
parser = argparse.ArgumentParser(description='Fast zero-shot eval for large maps.')
parser.add_argument('--sizes', type=int, nargs='+', default=[50, 55, 60, 65],
                    help='Map sizes to test (default: 50 55 60 65)')
parser.add_argument('--agents', type=int, nargs='+', default=[1, 3, 5, 8, 10],
                    help='Agent counts to test (default: 1 3 5 8 10)')
parser.add_argument('--trials', type=int, default=2,
                    help='Number of trials per (size, agents) combo (default: 2)')
parser.add_argument('--turbo', action='store_true',
                    help='Turbo mode: trials=1, agents=[5,10], sizes=all')
args = parser.parse_args()

if args.turbo:
    args.trials = 1
    args.agents = [5, 10]
    print("🚀 TURBO mode: trials=1, agents=[5, 10]")

SIZES_TO_TEST = sorted(args.sizes)
AGENT_COUNTS_TO_TEST = sorted(args.agents)
NUM_TRIALS = args.trials


# ============================================================================
# Tee: write to console AND file
# ============================================================================
class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# Config
# ============================================================================
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase9_config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_AGENTS = cfg['max_agents']
WORKER_RADIUS = cfg['worker_local_radius']
SERVER_RADIUS = cfg['server_local_radius']
IN_CHANNELS = 12 + (MAX_AGENTS - 1)
SCALAR_DIM = 15 + MAX_AGENTS + 1

VIDEO_DIR = os.path.join(os.path.dirname(__file__), '..', cfg['video_dir'])
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', cfg['checkpoint_dir'])
stage_final_path = os.path.join(CHECKPOINT_DIR, 'stage_final.pt')
best_model_path = os.path.join(CHECKPOINT_DIR, cfg['best_model_name'])
os.makedirs(VIDEO_DIR, exist_ok=True)

# Per-size configuration
NUM_SERVERS_PER_SIZE = cfg.get('num_servers_per_size', {})
TEST_MAX_STEPS_FACTOR_PER_SIZE = cfg.get('test_max_steps_factor_per_size', {})
TEST_MAX_STEPS_FACTOR = cfg.get('test_max_steps_factor', 8)

K_S = 5
K_P = 3


def get_num_servers_for_size(size):
    return NUM_SERVERS_PER_SIZE.get(size, cfg.get('num_servers', 3))


def get_max_steps_factor_for_size(size):
    return TEST_MAX_STEPS_FACTOR_PER_SIZE.get(size, TEST_MAX_STEPS_FACTOR)


# ============================================================================
# Server placement
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


def load_model():
    if os.path.exists(stage_final_path):
        checkpoint_path = stage_final_path
        print(f"✅ Loading model from END-OF-CURRICULUM: {checkpoint_path}")
    else:
        checkpoint_path = best_model_path
        print(f"⚠️  stage_final.pt not found, using {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    shared_q_net = QNetwork(action_dim=5, scalar_dim=SCALAR_DIM,
                            in_channels=IN_CHANNELS).to(device)
    shared_q_net.load_state_dict(checkpoint['agent_net'])
    shared_q_net.eval()
    return shared_q_net


# ============================================================================
# ✅ Fast single-episode runner with early termination
# ============================================================================
def run_episode_fast(env, agent_wrappers, actual_agents):
    """Run one episode with early termination when all agents are inactive.

    Returns a dict of aggregated metrics.
    """
    obs_tuple, _ = env.reset()
    obs_list = list(obs_tuple)
    for a in agent_wrappers:
        a.reset_hidden()

    done = False
    step = 0
    forced_stay = [0] * actual_agents

    while not done:
        actions = []
        for i in range(actual_agents):
            act, _ = agent_wrappers[i].select_action(obs_list[i], training=False)
            actions.append(act)
        while len(actions) < MAX_AGENTS:
            actions.append(4)

        # Robustness: idle check
        for i in range(actual_agents):
            if env.world.agent_active[i]:
                if env.world.idle_counters[i] >= K_S or env.world._is_looping(i):
                    x, y = env.world.agent_positions[i]
                    actions[i] = env.world._committed_fallback_action(i, x, y)

        # Filter 2
        proposed = []
        for i in range(actual_agents):
            x, y = env.world.agent_positions[i]
            act = actions[i]
            if act == 0:   ny = y - 1; nx = x
            elif act == 1: nx = x + 1; ny = y
            elif act == 2: ny = y + 1; nx = x
            elif act == 3: nx = x - 1; ny = y
            else:          nx, ny = x, y
            proposed.append((nx, ny))

        staying = set()
        for i in range(actual_agents):
            if actions[i] == 4:
                staying.add(env.world.agent_positions[i])

        for i in range(actual_agents):
            if actions[i] != 4 and proposed[i] in staying:
                actions[i] = 4
                forced_stay[i] += 1

        cell_to_agents = {}
        for i, (nx, ny) in enumerate(proposed):
            cell_to_agents.setdefault((nx, ny), []).append(i)

        for cell, agents in cell_to_agents.items():
            if len(agents) > 1:
                agents.sort(key=lambda i: (
                    abs(env.world.agent_positions[i][0] - cell[0]) +
                    abs(env.world.agent_positions[i][1] - cell[1]),
                    -forced_stay[i], i))
                for loser in agents[1:]:
                    actions[loser] = 4
                    forced_stay[loser] += 1

        for i in range(actual_agents):
            for j in range(i + 1, actual_agents):
                if (proposed[i] == env.world.agent_positions[j] and
                        proposed[j] == env.world.agent_positions[i]):
                    actions[i] = 4
                    actions[j] = 4
                    forced_stay[i] += 1
                    forced_stay[j] += 1

        obs_tuple, _, term, trunc, info = env.step(actions, training=False)
        done = term or trunc
        obs_list = list(obs_tuple)
        step += 1

        # ✅ HUGE SPEEDUP: early stop when all agents are inactive (battery dead)
        if not any(env.world.agent_active[i] for i in range(actual_agents)):
            # coverage has plateaued; nothing more will happen
            break

    return {
        'coverage': info['coverage'],
        'steps': step,
        'battery': np.mean(info.get('battery', [0])),
        'active': info.get('active_agents', actual_agents),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    # Log file
    sizes_str = '_'.join(str(s) for s in SIZES_TO_TEST)
    log_path = os.path.join(VIDEO_DIR, f'zero_shot_large_sizes{sizes_str}.txt')
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print(f"Device: {device}")
        print(f"Sizes to test:       {SIZES_TO_TEST}")
        print(f"Agent counts:        {AGENT_COUNTS_TO_TEST}")
        print(f"Trials per combo:    {NUM_TRIALS}")
        print(f"Total combinations:  {len(SIZES_TO_TEST)} × {len(AGENT_COUNTS_TO_TEST)} × {NUM_TRIALS} "
              f"= {len(SIZES_TO_TEST) * len(AGENT_COUNTS_TO_TEST) * NUM_TRIALS} episodes")
        print("=" * 70)

        # Load model once
        shared_q_net = load_model()
        agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(MAX_AGENTS)]

        results = {}

        t_start_global = time.time()

        for size in SIZES_TO_TEST:
            size_max_steps = get_max_steps_factor_for_size(size)
            num_servers = get_num_servers_for_size(size)

            print(f"\n{'=' * 70}")
            print(f"▶  MAP {size}×{size}  |  servers={num_servers}  |  max_steps_factor={size_max_steps}")
            print(f"{'=' * 70}")

            env_config = {
                'width': size, 'height': size,
                'max_steps': size * size * size_max_steps,
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

            print(f"   Generating obstacle grid...")
            temp_world = GridWorld(width=size, height=size,
                                    obstacle_density=cfg['obstacle_density'])
            fixed_grid = temp_world.grid

            print(f"   Placing {num_servers} servers...")
            server_positions = place_servers_optimally(size, size, fixed_grid, num_servers)

            results[size] = {}

            for num_agents in AGENT_COUNTS_TO_TEST:
                actual_agents = min(num_agents, MAX_AGENTS)
                t_start_agents = time.time()

                coverages, lams, overlaps = [], [], []
                batteries, actives, steps_list = [], [], []

                print(f"\n   → {num_agents} agents:")

                for trial in range(NUM_TRIALS):
                    t_start_trial = time.time()

                    env = CoverageEnv(
                        env_config, max_agents=MAX_AGENTS,
                        active_agents=actual_agents,
                        num_servers=num_servers,
                        server_positions=server_positions,
                        obstacle_map=fixed_grid,
                        worker_local_radius=WORKER_RADIUS,
                    )
                    r = run_episode_fast(env, agent_wrappers, actual_agents)

                    cov = r['coverage']
                    T = r['steps']
                    C0 = env.world.free_cells
                    J = actual_agents
                    lam = T / C0 if C0 > 0 else 0
                    overlap = (T - C0 / J) / (C0 / J) if C0 > 0 else 0

                    coverages.append(cov)
                    lams.append(lam)
                    overlaps.append(overlap)
                    batteries.append(r['battery'])
                    actives.append(r['active'])
                    steps_list.append(T)

                    dt = time.time() - t_start_trial
                    print(f"      trial {trial+1}/{NUM_TRIALS}: "
                          f"cov={cov:.1%}, λ={lam:.3f}, O={overlap:.3f}, "
                          f"steps={T}, batt={r['battery']:.0f}, "
                          f"active={r['active']}/{num_agents}  ({dt:.1f}s)")

                results[size][num_agents] = {
                    'cov_mean': np.mean(coverages),
                    'cov_std':  np.std(coverages),
                    'lam_mean': np.mean(lams),
                    'lam_std':  np.std(lams),
                    'overlap_mean': np.mean(overlaps),
                    'overlap_std':  np.std(overlaps),
                    'battery_mean': np.mean(batteries),
                    'active_mean':  np.mean(actives),
                    'steps_mean':   np.mean(steps_list),
                }

                dt_agents = time.time() - t_start_agents
                print(f"      → Mean coverage = {results[size][num_agents]['cov_mean']:.1%} "
                      f"(total: {dt_agents:.1f}s)")

        dt_total = time.time() - t_start_global

        # ====================================================================
        # Summary table
        # ====================================================================
        print(f"\n{'=' * 100}")
        print("SUMMARY")
        print(f"{'=' * 100}")
        print(f"{'Map':>8} | {'Agents':>6} | {'Coverage':>14} | {'λ':>14} | "
              f"{'O':>14} | {'Steps':>8} | {'Batt':>6} | {'Active':>6}")
        print("-" * 110)
        for size in SIZES_TO_TEST:
            for agents in AGENT_COUNTS_TO_TEST:
                r = results[size][agents]
                print(f"{size}×{size:>3} | {agents:>6} | "
                      f"{r['cov_mean']:>7.1%} ±{r['cov_std']:.1%} | "
                      f"{r['lam_mean']:>7.3f} ±{r['lam_std']:.3f} | "
                      f"{r['overlap_mean']:>7.3f} ±{r['overlap_std']:.3f} | "
                      f"{r['steps_mean']:>8.0f} | "
                      f"{r['battery_mean']:>6.0f} | "
                      f"{r['active_mean']:>6.1f}")
        print(f"{'=' * 100}")
        print(f"✅ Total time: {dt_total:.1f}s ({dt_total/60:.1f} min)")

        # ====================================================================
        # Plot
        # ====================================================================
        print("\n📊 Generating plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(SIZES_TO_TEST))
        width = 0.8 / len(AGENT_COUNTS_TO_TEST)

        for idx, agents in enumerate(AGENT_COUNTS_TO_TEST):
            covs = [results[s][agents]['cov_mean'] * 100 for s in SIZES_TO_TEST]
            stds = [results[s][agents]['cov_std'] * 100 for s in SIZES_TO_TEST]
            ax.bar(x + idx * width, covs, width, yerr=stds,
                   label=f'{agents} agents', capsize=3)

        ax.set_xticks(x + width * (len(AGENT_COUNTS_TO_TEST) - 1) / 2)
        ax.set_xticklabels([f'{s}×{s}' for s in SIZES_TO_TEST])
        ax.set_xlabel('Map size')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'Zero-shot coverage on large maps (trials={NUM_TRIALS})')
        ax.set_ylim(0, 105)
        ax.axhline(100, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(VIDEO_DIR, f'zero_shot_large_sizes{sizes_str}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"   ✅ Saved: {plot_path}")

        print(f"\n✅ Log: {log_path}")

    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()