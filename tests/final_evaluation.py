"""
Final evaluation script — zero-shot tests on unseen map sizes.

Features:
    - Prefers stage_final.pt (end-of-curriculum) over best_ctde.pt
    - Per-size configuration (servers, max_steps, trials)
    - Extended map range including 50, 55, 60, 65
    - Failure-tolerance tests with proper trigger step
    - Auto-saves full log to text file + saves plots to disk
    - Faster: uses per-size reduced agent counts and trials for large maps
"""

import sys, os, yaml, torch, numpy as np, random
import matplotlib

# ✅ Try interactive backend, fall back to Agg for headless
try:
    matplotlib.use('TkAgg')
    INTERACTIVE = True
except Exception:
    matplotlib.use('Agg')
    INTERACTIVE = False
import matplotlib.pyplot as plt
from collections import defaultdict

# ✅ FIXED: sys.path before project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv
from envs.grid_world import GridWorld
from models.dqn_network import QNetwork
from agents.agent_wrapper import AgentWrapper


# ============================================================================
# Tee stdout to both console and text file
# ============================================================================
class Tee:
    """Write to both stdout and a text file simultaneously."""
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


# ---- Load config ----
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase9_config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_AGENTS = cfg['max_agents']
WORKER_RADIUS = cfg['worker_local_radius']
SERVER_RADIUS = cfg['server_local_radius']
IN_CHANNELS = 12 + (MAX_AGENTS - 1)
SCALAR_DIM = 15 + MAX_AGENTS + 1

# ✅ FIXED: Extended map range including 65
FINAL_TESTS = cfg.get('final_tests', [20, 25, 28, 30, 35, 40, 50, 55, 60])
if 65 not in FINAL_TESTS:
    FINAL_TESTS = sorted(set(list(FINAL_TESTS) + [65]))

AGENT_COUNTS = cfg.get('agent_counts_for_tests', [1, 3, 5, 8, 10])
VIDEO_DIR = os.path.join(os.path.dirname(__file__), '..', cfg['video_dir'])
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', cfg['checkpoint_dir'])
BEST_MODEL = cfg['best_model_name']
stage_final_path = os.path.join(CHECKPOINT_DIR, 'stage_final.pt')
best_model_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL)
os.makedirs(VIDEO_DIR, exist_ok=True)

TEST_MAX_STEPS_FACTOR = cfg.get('test_max_steps_factor', 8)
FAILURE_AGENT_COUNTS = cfg.get('failure_agent_counts', [3, 5, 8, 10])

# ✅ Per-size configuration
NUM_SERVERS_PER_SIZE = cfg.get('num_servers_per_size', {})
TEST_MAX_STEPS_FACTOR_PER_SIZE = cfg.get('test_max_steps_factor_per_size', {})
NUM_TRIALS_PER_SIZE = cfg.get('num_trials_per_size', {})
LARGE_MAP_THRESHOLD = cfg.get('large_map_threshold', 50)
AGENT_COUNTS_LARGE = cfg.get('agent_counts_for_large_maps', [5, 10])

SAVE_GIFS = False
DISPLAY_LIVE = False

K_S = 5
K_P = 3


# ============================================================================
# Per-size configuration helpers
# ============================================================================
def get_num_servers_for_size(size):
    return NUM_SERVERS_PER_SIZE.get(size, cfg.get('num_servers', 3))


def get_max_steps_factor_for_size(size):
    return TEST_MAX_STEPS_FACTOR_PER_SIZE.get(size, TEST_MAX_STEPS_FACTOR)


def get_num_trials_for_size(size):
    return NUM_TRIALS_PER_SIZE.get(size, 5)


def get_agent_counts_for_size(size):
    if size >= LARGE_MAP_THRESHOLD:
        return AGENT_COUNTS_LARGE
    return AGENT_COUNTS


def get_failure_agent_counts_for_size(size):
    """Reduce failure test agent counts for large maps to save time."""
    if size >= LARGE_MAP_THRESHOLD:
        return [5, 10]  # fewer combos for big maps
    return FAILURE_AGENT_COUNTS


# ============================================================================
# Server placement (scales to any number of servers)
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
                            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0 and not covered[ny, nx]:
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
# Load model (prefers stage_final.pt)
# ============================================================================
def load_model():
    if os.path.exists(stage_final_path):
        checkpoint_path = stage_final_path
        print(f"✅ Loading model from END-OF-CURRICULUM: {checkpoint_path}")
    else:
        checkpoint_path = best_model_path
        print(f"⚠️  stage_final.pt not found, falling back to: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    shared_q_net = QNetwork(action_dim=5, scalar_dim=SCALAR_DIM, in_channels=IN_CHANNELS).to(device)
    shared_q_net.load_state_dict(checkpoint['agent_net'])
    shared_q_net.eval()
    return shared_q_net


# ============================================================================
# Single evaluation trial (one episode) — returns aggregated metrics
# ============================================================================
def run_eval_episode(env, agent_wrappers, actual_agents):
    obs_tuple, _ = env.reset()
    obs_list = list(obs_tuple)
    for a in agent_wrappers:
        a.reset_hidden()

    done = False
    episode_forced_stay_count = [0] * actual_agents
    aa_about = ao_about = aa_actual = ao_actual = 0
    episode_steps = 0

    while not done:
        actions = []
        for i in range(actual_agents):
            act, _ = agent_wrappers[i].select_action(obs_list[i], training=False)
            actions.append(act)
        while len(actions) < MAX_AGENTS:
            actions.append(4)

        # Robustness monitor (idle check)
        for i in range(actual_agents):
            if env.world.agent_active[i]:
                idle = env.world.idle_counters[i]
                recent = list(env.world.recent_positions[i])
                stuck_long = (idle >= K_S)
                stuck_short = False
                if len(recent) >= K_P:
                    dist = abs(recent[-1][0] - recent[-K_P][0]) + abs(recent[-1][1] - recent[-K_P][1])
                    stuck_short = (idle >= K_P) and (dist <= 1)
                if stuck_long or stuck_short:
                    x, y = env.world.agent_positions[i]
                    actions[i] = env.world._committed_fallback_action(i, x, y)

        # Filter 2 (Local Coordination)
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

        staying_positions = set()
        for i in range(actual_agents):
            if actions[i] == 4:
                staying_positions.add(env.world.agent_positions[i])

        for i in range(actual_agents):
            if actions[i] != 4 and proposed[i] in staying_positions:
                actions[i] = 4
                episode_forced_stay_count[i] += 1

        cell_to_agents = {}
        for i, (nx, ny) in enumerate(proposed):
            cell_to_agents.setdefault((nx, ny), []).append(i)

        for cell, agents in cell_to_agents.items():
            if len(agents) > 1:
                agents.sort(key=lambda i: (
                    abs(env.world.agent_positions[i][0] - cell[0]) +
                    abs(env.world.agent_positions[i][1] - cell[1]),
                    -episode_forced_stay_count[i], i))
                for loser in agents[1:]:
                    actions[loser] = 4
                    episode_forced_stay_count[loser] += 1

        for i in range(actual_agents):
            for j in range(i + 1, actual_agents):
                if (proposed[i] == env.world.agent_positions[j] and
                        proposed[j] == env.world.agent_positions[i]):
                    actions[i] = 4
                    actions[j] = 4
                    episode_forced_stay_count[i] += 1
                    episode_forced_stay_count[j] += 1

        obs_tuple, _, term, trunc, info = env.step(actions, training=False)
        done = term or trunc
        obs_list = list(obs_tuple)
        episode_steps += 1

        aa_about += info.get('agent_agent_collisions', 0)
        ao_about += info.get('obstacle_avoidances', 0)
        aa_actual += info.get('actual_agent_hits', 0)
        ao_actual += info.get('actual_obstacle_hits', 0)

    return {
        'coverage': info['coverage'],
        'steps': episode_steps,
        'aa_about': aa_about, 'ao_about': ao_about,
        'aa_actual': aa_actual, 'ao_actual': ao_actual,
        'battery': np.mean(info.get('battery', [0])),
        'active': info.get('active_agents', actual_agents),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    # ✅ Save full output to text file
    log_path = os.path.join(VIDEO_DIR, 'final_eval_results.txt')
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print(f"Device: {device}")
        print(f"Extended test sizes: {FINAL_TESTS}")

        shared_q_net = load_model()
        agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(MAX_AGENTS)]

        results = defaultdict(dict)
        failure_results = defaultdict(dict)

        for size in FINAL_TESTS:
            print(f"\n{'=' * 60}")
            print(f"Evaluating on {size}×{size} maps")

            size_max_steps = get_max_steps_factor_for_size(size)
            num_servers = get_num_servers_for_size(size)
            agent_counts_to_test = get_agent_counts_for_size(size)
            num_trials_for_size = get_num_trials_for_size(size)

            print(f"  → Servers: {num_servers} | max_steps_factor: {size_max_steps} "
                  f"| Trials: {num_trials_for_size} | Agents: {agent_counts_to_test}")

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

            temp_world = GridWorld(width=size, height=size, obstacle_density=cfg['obstacle_density'])
            fixed_grid = temp_world.grid
            server_positions = place_servers_optimally(size, size, fixed_grid, num_servers)

            # ---- Normal zero-shot evaluation ----
            for num_agents in agent_counts_to_test:
                actual_agents = min(num_agents, MAX_AGENTS)
                if actual_agents < num_agents:
                    print(f"  ⚠️  Capping {num_agents} agents to MAX_AGENTS={MAX_AGENTS}")

                coverages, lams, overlaps = [], [], []
                final_batteries, active_agents_end, total_steps_list = [], [], []
                aa_about_list, ao_about_list, aa_actual_list, ao_actual_list = [], [], [], []

                for trial in range(num_trials_for_size):
                    env = CoverageEnv(
                        env_config, max_agents=MAX_AGENTS,
                        active_agents=actual_agents,
                        num_servers=num_servers,
                        server_positions=server_positions,
                        obstacle_map=fixed_grid,
                        worker_local_radius=WORKER_RADIUS,
                    )
                    r = run_eval_episode(env, agent_wrappers, actual_agents)

                    cov = r['coverage']
                    T = r['steps']
                    C0 = env.world.free_cells
                    J = actual_agents
                    lam = T / C0 if C0 > 0 else 0
                    overlap = (T - C0 / J) / (C0 / J) if C0 > 0 else 0

                    coverages.append(cov)
                    lams.append(lam)
                    overlaps.append(overlap)
                    final_batteries.append(r['battery'])
                    active_agents_end.append(r['active'])
                    total_steps_list.append(T)
                    aa_about_list.append(r['aa_about'])
                    ao_about_list.append(r['ao_about'])
                    aa_actual_list.append(r['aa_actual'])
                    ao_actual_list.append(r['ao_actual'])

                    print(f"    {size}×{size} | {num_agents} agents | trial {trial+1}/{num_trials_for_size}: "
                          f"cov={cov:.1%}, λ={lam:.3f}, O={overlap:.3f}, steps={T}")

                results[size][num_agents] = {
                    'cov_mean': np.mean(coverages), 'cov_std': np.std(coverages),
                    'lam_mean': np.mean(lams), 'lam_std': np.std(lams),
                    'overlap_mean': np.mean(overlaps), 'overlap_std': np.std(overlaps),
                    'battery_mean': np.mean(final_batteries),
                    'active_mean': np.mean(active_agents_end),
                    'steps_mean': np.mean(total_steps_list),
                    'aa_about_mean': np.mean(aa_about_list),
                    'ao_about_mean': np.mean(ao_about_list),
                    'aa_actual_mean': np.mean(aa_actual_list),
                    'ao_actual_mean': np.mean(ao_actual_list),
                }

            # ---- Failure tolerance tests ----
            print(f"\n  Failure tests on {size}×{size}")
            failure_agents_to_test = get_failure_agent_counts_for_size(size)

            for init_agents in failure_agents_to_test:
                if init_agents > MAX_AGENTS:
                    continue
                remove_count = min(2, init_agents - 1)
                remaining_agents = init_agents - remove_count

                env = CoverageEnv(
                    env_config, max_agents=MAX_AGENTS,
                    active_agents=init_agents,
                    num_servers=num_servers,
                    server_positions=server_positions,
                    obstacle_map=fixed_grid,
                    worker_local_radius=WORKER_RADIUS,
                )

                obs_tuple, _ = env.reset()
                obs_list = list(obs_tuple)
                for a in agent_wrappers:
                    a.reset_hidden()

                done = False
                step = 0
                failure_triggered = False
                before_cov = 0.0
                after_cov = 0.0
                after_covs = []
                episode_forced_stay_count = [0] * MAX_AGENTS
                info = None

                # ✅ FIXED: Failure step proportional to expected episode length
                #    Use a fraction of max_steps that's likely to trigger within the episode.
                #    For 20×20 with max_steps=3200, normal episode is ~200 steps.
                #    Set failure_step = 40% of typical episode length:
                failure_step = max(20, int(size * 4))

                while not done:
                    if step == failure_step and not failure_triggered:
                        before_cov = info['coverage'] if info is not None else 0.0
                        env.active_agents = remaining_agents
                        env.world.active_agents = remaining_agents
                        failure_triggered = True
                        print(f"    {init_agents}→{remaining_agents} at step {step}, "
                              f"coverage before: {before_cov:.1%}")

                    actions = []
                    actual_active = env.active_agents
                    for i in range(actual_active):
                        act, _ = agent_wrappers[i].select_action(obs_list[i], training=False)
                        actions.append(act)
                    while len(actions) < MAX_AGENTS:
                        actions.append(4)

                    # Robustness monitor
                    for i in range(actual_active):
                        if env.world.agent_active[i]:
                            idle = env.world.idle_counters[i]
                            recent = list(env.world.recent_positions[i])
                            stuck_long = (idle >= K_S)
                            stuck_short = False
                            if len(recent) >= K_P:
                                dist = abs(recent[-1][0] - recent[-K_P][0]) + abs(recent[-1][1] - recent[-K_P][1])
                                stuck_short = (idle >= K_P) and (dist <= 1)
                            if stuck_long or stuck_short:
                                x, y = env.world.agent_positions[i]
                                actions[i] = env.world._committed_fallback_action(i, x, y)

                    # Filter 2
                    proposed = []
                    for i in range(actual_active):
                        x, y = env.world.agent_positions[i]
                        act = actions[i]
                        if act == 0:   ny = y - 1; nx = x
                        elif act == 1: nx = x + 1; ny = y
                        elif act == 2: ny = y + 1; nx = x
                        elif act == 3: nx = x - 1; ny = y
                        else:          nx, ny = x, y
                        proposed.append((nx, ny))

                    staying_positions = set()
                    for i in range(actual_active):
                        if actions[i] == 4:
                            staying_positions.add(env.world.agent_positions[i])

                    for i in range(actual_active):
                        if actions[i] != 4 and proposed[i] in staying_positions:
                            actions[i] = 4

                    cell_to_agents = {}
                    for i, (nx, ny) in enumerate(proposed):
                        cell_to_agents.setdefault((nx, ny), []).append(i)

                    for cell, agents in cell_to_agents.items():
                        if len(agents) > 1:
                            agents.sort(key=lambda i: (
                                abs(env.world.agent_positions[i][0] - cell[0]) +
                                abs(env.world.agent_positions[i][1] - cell[1]),
                                -episode_forced_stay_count[i], i))
                            for loser in agents[1:]:
                                actions[loser] = 4
                                episode_forced_stay_count[loser] += 1

                    for i in range(actual_active):
                        for j in range(i + 1, actual_active):
                            if (proposed[i] == env.world.agent_positions[j] and
                                    proposed[j] == env.world.agent_positions[i]):
                                actions[i] = 4
                                actions[j] = 4
                                episode_forced_stay_count[i] += 1
                                episode_forced_stay_count[j] += 1

                    obs_tuple, _, term, trunc, info = env.step(actions, training=False)
                    done = term or trunc
                    obs_list = list(obs_tuple)
                    step += 1
                    if failure_triggered:
                        after_covs.append(info['coverage'])

                if not failure_triggered:
                    before_cov = info['coverage'] if info is not None else 0.0
                    after_cov = before_cov
                    print(f"    ⚠️  Episode ended before removal step at {failure_step}.")
                else:
                    after_cov = after_covs[-1] if after_covs else info['coverage'] if info else 0.0

                failure_results[size][init_agents] = (before_cov, after_cov)
                print(f"    {init_agents}→{remaining_agents}: before = {before_cov:.1%}, "
                      f"after = {after_cov:.1%}")

        # ====================================================================
        # Plots
        # ====================================================================
        print(f"\n{'=' * 60}")
        print("Generating plots...")

        plot_agent_counts = sorted(set(
            a for a in AGENT_COUNTS if any(a in results[s] for s in FINAL_TESTS)
        ))

        # ---- Plot 1: Zero-shot coverage ----
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(FINAL_TESTS))
        width = 0.8 / max(len(plot_agent_counts), 1)
        for idx, agents in enumerate(plot_agent_counts):
            covs = [results[s][agents]['cov_mean'] * 100 if agents in results[s] else 0
                    for s in FINAL_TESTS]
            stds = [results[s][agents]['cov_std'] * 100 if agents in results[s] else 0
                    for s in FINAL_TESTS]
            ax.bar(x + idx * width, covs, width, yerr=stds,
                   label=f'{agents} agents', capsize=3)
        ax.set_xticks(x + width * (len(plot_agent_counts) - 1) / 2)
        ax.set_xticklabels([f'{s}×{s}' for s in FINAL_TESTS])
        ax.set_xlabel('Map size')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('Zero-shot coverage for different swarm sizes')
        ax.set_ylim(0, 105)
        ax.axhline(100, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VIDEO_DIR, 'zero_shot_coverage.png'), dpi=150)
        plt.close(fig)
        print("  ✅ Saved: zero_shot_coverage.png")

        # ---- Plot 2: λ vs Map Size ----
        fig, ax = plt.subplots(figsize=(12, 6))
        for agents in plot_agent_counts:
            lam_means = [results[s][agents]['lam_mean'] if agents in results[s] else np.nan
                         for s in FINAL_TESTS]
            lam_stds = [results[s][agents]['lam_std'] if agents in results[s] else 0
                        for s in FINAL_TESTS]
            ax.errorbar(FINAL_TESTS, lam_means, yerr=lam_stds, marker='o',
                        label=f'{agents} agents', capsize=3)
        ax.set_xlabel('Map size')
        ax.set_ylabel('λ (Time Save Factor)')
        ax.set_title('λ vs map size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VIDEO_DIR, 'lambda_vs_size.png'), dpi=150)
        plt.close(fig)
        print("  ✅ Saved: lambda_vs_size.png")

        # ---- Plot 3: Overlap vs Map Size ----
        fig, ax = plt.subplots(figsize=(12, 6))
        for agents in plot_agent_counts:
            overlap_means = [results[s][agents]['overlap_mean'] if agents in results[s] else np.nan
                             for s in FINAL_TESTS]
            overlap_stds = [results[s][agents]['overlap_std'] if agents in results[s] else 0
                            for s in FINAL_TESTS]
            ax.errorbar(FINAL_TESTS, overlap_means, yerr=overlap_stds, marker='s',
                        label=f'{agents} agents', capsize=3)
        ax.set_xlabel('Map size')
        ax.set_ylabel('O (Overlap)')
        ax.set_title('Overlap vs map size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VIDEO_DIR, 'overlap_vs_size.png'), dpi=150)
        plt.close(fig)
        print("  ✅ Saved: overlap_vs_size.png")

        # ---- Plot 4: Steps summary ----
        fig, ax = plt.subplots(figsize=(12, 6))
        for agents in plot_agent_counts:
            steps_means = [results[s][agents]['steps_mean'] if agents in results[s] else np.nan
                           for s in FINAL_TESTS]
            ax.plot(FINAL_TESTS, steps_means, marker='o', label=f'{agents} agents')
        ax.set_xlabel('Map size')
        ax.set_ylabel('Steps')
        ax.set_title('Average steps per episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VIDEO_DIR, 'steps_summary.png'), dpi=150)
        plt.close(fig)
        print("  ✅ Saved: steps_summary.png")

        # ---- Plot 5: Battery & Active summary ----
        fig, ax = plt.subplots(figsize=(12, 6))
        for agents in plot_agent_counts:
            batt_means = [results[s][agents]['battery_mean'] if agents in results[s] else np.nan
                          for s in FINAL_TESTS]
            active_means = [results[s][agents]['active_mean'] if agents in results[s] else np.nan
                            for s in FINAL_TESTS]
            ax.plot(FINAL_TESTS, batt_means, marker='o',
                    label=f'{agents} agents (battery)')
            ax.plot(FINAL_TESTS, active_means, marker='s', linestyle='--',
                    label=f'{agents} agents (active)')
        ax.set_xlabel('Map size')
        ax.set_ylabel('Value')
        ax.set_title('Battery and active agents at episode end')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VIDEO_DIR, 'battery_active_summary.png'), dpi=150)
        plt.close(fig)
        print("  ✅ Saved: battery_active_summary.png")

        # ---- Plot 6: Failure tolerance ----
        for init_agents in FAILURE_AGENT_COUNTS:
            if init_agents > MAX_AGENTS:
                continue
            sizes_with_data = sorted(s for s in failure_results if init_agents in failure_results[s])
            if not sizes_with_data:
                continue
            before = [failure_results[s][init_agents][0] * 100 for s in sizes_with_data]
            after = [failure_results[s][init_agents][1] * 100 for s in sizes_with_data]
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(sizes_with_data))
            width = 0.35
            ax.bar(x - width / 2, before, width, label='Before failure')
            ax.bar(x + width / 2, after, width, label='After failure')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{s}×{s}' for s in sizes_with_data], rotation=45)
            ax.set_ylabel('Coverage (%)')
            ax.set_title(f'Failure tolerance: {init_agents} → {init_agents - 2} agents')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(VIDEO_DIR, f'failure_tolerance_{init_agents}agents.png'), dpi=150)
            plt.close(fig)
        print("  ✅ Saved: failure_tolerance_*.png")

        # ====================================================================
        # Summary table
        # ====================================================================
        print("\n" + "=" * 120)
        print("FINAL RESULTS SUMMARY (Zero-shot evaluation)")
        print("=" * 120)
        print(f"{'Map':>8} | {'Agents':>6} | {'Coverage':>14} | {'λ':>16} | "
              f"{'O':>16} | {'Steps':>8} | {'Batt':>6} | {'Active':>6}")
        print("-" * 140)
        for size in FINAL_TESTS:
            for agents in sorted(results[size].keys()):
                r = results[size][agents]
                print(f"{size}×{size:>3} | {agents:>6} | "
                      f"{r['cov_mean']:>7.1%} ±{r['cov_std']:.1%} | "
                      f"{r['lam_mean']:>8.3f} ±{r['lam_std']:.3f} | "
                      f"{r['overlap_mean']:>8.3f} ±{r['overlap_std']:.3f} | "
                      f"{r['steps_mean']:>8.0f} | "
                      f"{r['battery_mean']:>6.0f} | "
                      f"{r['active_mean']:>6.1f}")

        print("\n" + "=" * 120)
        print("Failure tolerance results (before → after):")
        for size in FINAL_TESTS:
            for init_agents in FAILURE_AGENT_COUNTS:
                if init_agents > MAX_AGENTS:
                    continue
                if init_agents in failure_results[size]:
                    before, after = failure_results[size][init_agents]
                    print(f"  {size}×{size}, {init_agents}→{init_agents - 2}: "
                          f"{before:.1%} → {after:.1%}")
        print("=" * 120)
        print(f"Evaluation complete. Results saved to: {VIDEO_DIR}")
        print(f"Full log: {log_path}")

    finally:
        # ✅ Always restore stdout and close file
        sys.stdout = tee.stdout
        tee.close()
        print(f"\n✅ Log written to: {log_path}")


if __name__ == "__main__":
    main()