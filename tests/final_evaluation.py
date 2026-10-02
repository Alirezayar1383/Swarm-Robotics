import sys, os, yaml, torch, numpy as np, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv
from envs.grid_world import GridWorld
from models.dqn_network import QNetwork
from agents.agent_wrapper import AgentWrapper
from utils.metrics import animate_multi

config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase9_config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

MAX_AGENTS = cfg['max_agents']
WORKER_RADIUS = cfg['worker_local_radius']
SERVER_RADIUS = cfg['server_local_radius']
IN_CHANNELS = 12 + (MAX_AGENTS - 1)
SCALAR_DIM = 15 + MAX_AGENTS + 1

FINAL_TESTS = cfg['final_tests']
AGENT_COUNTS = cfg['agent_counts_for_tests']
NUM_TRIALS = 5
VIDEO_DIR = os.path.join(os.path.dirname(__file__), '..', cfg['video_dir'])
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', cfg['checkpoint_dir'])
BEST_MODEL = cfg['best_model_name']
best_model_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL)
os.makedirs(VIDEO_DIR, exist_ok=True)

TEST_MAX_STEPS_FACTOR = cfg.get('test_max_steps_factor', 8)
FAILURE_AGENT_COUNTS = cfg.get('failure_agent_counts', [3, 5, 8, 10])

SAVE_GIFS = False
DISPLAY_LIVE = False

K_S = 5
K_P = 3

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
    checkpoint = torch.load(best_model_path, map_location=device)
    shared_q_net = QNetwork(action_dim=5, scalar_dim=SCALAR_DIM, in_channels=IN_CHANNELS).to(device)
    shared_q_net.load_state_dict(checkpoint['agent_net'])
    shared_q_net.eval()
    return shared_q_net

shared_q_net = load_model()
agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(MAX_AGENTS)]

results = defaultdict(dict)
failure_results = defaultdict(dict)

for size in FINAL_TESTS:
    print(f"\n{'='*60}")
    print(f"Evaluating on {size}×{size} maps")
    env_config = {
        'width': size, 'height': size,
        'max_steps': size * size * TEST_MAX_STEPS_FACTOR,
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

    num_servers = cfg.get('num_servers', 3)
    server_positions = place_servers_optimally(size, size, fixed_grid, num_servers)

    # --- Normal evaluation for each agent count ---
    for num_agents in AGENT_COUNTS:
        actual_agents = min(num_agents, MAX_AGENTS)
        if actual_agents < num_agents:
            print(f"  Warning: Using {actual_agents} networks for {num_agents} agents test")

        coverages = []
        lams = []
        overlaps = []
        final_batteries = []
        active_agents_end = []
        total_steps_list = []
        aa_about_list = []
        ao_about_list = []
        aa_actual_list = []
        ao_actual_list = []

        for trial in range(NUM_TRIALS):
            env = CoverageEnv(
                env_config,
                max_agents=MAX_AGENTS,
                active_agents=actual_agents,
                num_servers=num_servers,
                server_positions=server_positions,
                obstacle_map=fixed_grid,
                worker_local_radius=WORKER_RADIUS
            )
            obs_tuple, _ = env.reset()
            obs_list = list(obs_tuple)
            for a in agent_wrappers:
                a.reset_hidden()
            done = False
            episode_forced_stay_count = [0] * actual_agents
            aa_about = 0
            ao_about = 0
            aa_actual = 0
            ao_actual = 0
            episode_steps = 0

            while not done:
                actions = []
                for i in range(actual_agents):
                    act, _ = agent_wrappers[i].select_action(obs_list[i], training=False)
                    actions.append(act)
                while len(actions) < MAX_AGENTS:
                    actions.append(4)

                # Robustness monitor (idle check) – only for active agents
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

                # Improved filter2
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
                    if actions[i] != 4:
                        if proposed[i] in staying_positions:
                            actions[i] = 4
                            episode_forced_stay_count[i] += 1

                cell_to_agents = {}
                for i, (nx, ny) in enumerate(proposed):
                    cell_to_agents.setdefault((nx, ny), []).append(i)

                for cell, agents in cell_to_agents.items():
                    if len(agents) > 1:
                        agents.sort(key=lambda i: (
                            abs(env.world.agent_positions[i][0] - cell[0])
                            + abs(env.world.agent_positions[i][1] - cell[1]),
                            -episode_forced_stay_count[i],
                            i
                        ))
                        for loser in agents[1:]:
                            actions[loser] = 4
                            episode_forced_stay_count[loser] += 1

                for i in range(actual_agents):
                    for j in range(i+1, actual_agents):
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

            cov = info['coverage']
            T = info['steps']
            C0 = env.world.free_cells
            J = actual_agents
            lam = T / C0 if C0 > 0 else 0
            overlap = (T - C0/J) / (C0/J) if C0 > 0 else 0

            coverages.append(cov)
            lams.append(lam)
            overlaps.append(overlap)
            final_batteries.append(np.mean(info.get('battery', [0])))
            active_agents_end.append(info.get('active_agents', actual_agents))
            total_steps_list.append(episode_steps)
            aa_about_list.append(aa_about)
            ao_about_list.append(ao_about)
            aa_actual_list.append(aa_actual)
            ao_actual_list.append(ao_actual)

            print(f"    {size}×{size} | {num_agents} agents | trial {trial+1}: cov={cov:.1%}, λ={lam:.3f}, O={overlap:.3f}, steps={episode_steps}")

        avg_cov = np.mean(coverages)
        avg_lam = np.mean(lams)
        avg_overlap = np.mean(overlaps)
        results[size][num_agents] = {
            'cov_mean': avg_cov, 'cov_std': np.std(coverages),
            'lam_mean': avg_lam, 'lam_std': np.std(lams),
            'overlap_mean': avg_overlap, 'overlap_std': np.std(overlaps),
            'battery_mean': np.mean(final_batteries),
            'active_mean': np.mean(active_agents_end),
            'steps_mean': np.mean(total_steps_list),
            'aa_about_mean': np.mean(aa_about_list),
            'ao_about_mean': np.mean(ao_about_list),
            'aa_actual_mean': np.mean(aa_actual_list),
            'ao_actual_mean': np.mean(ao_actual_list),
        }

    # ─── Failure Tolerance Tests ──────────────────────────────
    print(f"\n  Failure tests on {size}×{size}")
    for init_agents in FAILURE_AGENT_COUNTS:
        if init_agents > MAX_AGENTS:
            print(f"    Skipping {init_agents} agents (exceeds MAX_AGENTS={MAX_AGENTS})")
            continue
        remove_count = min(2, init_agents - 1)
        remaining_agents = init_agents - remove_count

        env = CoverageEnv(
            env_config,
            max_agents=MAX_AGENTS,
            active_agents=init_agents,
            num_servers=num_servers,
            server_positions=server_positions,
            obstacle_map=fixed_grid,
            worker_local_radius=WORKER_RADIUS
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

        while not done:
            if step == size * size * 2 and not failure_triggered:
                before_cov = info['coverage'] if info is not None else 0.0
                env.active_agents = remaining_agents
                env.world.active_agents = remaining_agents
                failure_triggered = True
                print(f"    {init_agents}→{remaining_agents} at step {step}, coverage before: {before_cov:.1%}")

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

            # Improved filter2
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
                if actions[i] != 4:
                    if proposed[i] in staying_positions:
                        actions[i] = 4
                        episode_forced_stay_count[i] += 1

            cell_to_agents = {}
            for i, (nx, ny) in enumerate(proposed):
                cell_to_agents.setdefault((nx, ny), []).append(i)

            for cell, agents in cell_to_agents.items():
                if len(agents) > 1:
                    agents.sort(key=lambda i: (
                        abs(env.world.agent_positions[i][0] - cell[0])
                        + abs(env.world.agent_positions[i][1] - cell[1]),
                        -episode_forced_stay_count[i],
                        i
                    ))
                    for loser in agents[1:]:
                        actions[loser] = 4
                        episode_forced_stay_count[loser] += 1

            for i in range(actual_active):
                for j in range(i+1, actual_active):
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
            print(f"    Episode ended before removal step. Final coverage: {before_cov:.1%}")
        else:
            after_cov = after_covs[-1] if after_covs else info['coverage'] if info else 0.0

        failure_results[size][init_agents] = (before_cov, after_cov)
        print(f"    {init_agents}→{remaining_agents}: before = {before_cov:.1%}, after = {after_cov:.1%}")

# ===================== Plotting =====================
# 1. Zero-shot coverage bar chart (title includes servers and steps)
plt.figure(figsize=(14, 6))
x = np.arange(len(FINAL_TESTS))
width = 0.15
for idx, agents in enumerate(AGENT_COUNTS):
    covs = [results[size][agents]['cov_mean'] for size in FINAL_TESTS]
    stds = [results[size][agents]['cov_std'] for size in FINAL_TESTS]
    plt.bar(x + idx*width, covs, width, yerr=stds, label=f'{agents} agents', capsize=3)
plt.xticks(x + width * (len(AGENT_COUNTS)-1)/2, [f'{s}×{s}' for s in FINAL_TESTS])
plt.xlabel('Map size')
plt.ylabel('Coverage')
plt.title(f'Zero‑shot coverage for different swarm sizes (Servers: {num_servers})')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_DIR, 'zero_shot_coverage.png'))
plt.close()

# 2. Lambda vs map size
plt.figure(figsize=(12, 5))
for agents in AGENT_COUNTS:
    lam_means = [results[size][agents]['lam_mean'] for size in FINAL_TESTS]
    lam_stds = [results[size][agents]['lam_std'] for size in FINAL_TESTS]
    plt.errorbar(FINAL_TESTS, lam_means, yerr=lam_stds, marker='o', label=f'{agents} agents')
plt.xlabel('Map size')
plt.ylabel('λ (Time Save Factor)')
plt.title(f'λ vs map size (Servers: {num_servers})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_DIR, 'lambda_vs_size.png'))
plt.close()

# 3. Overlap vs map size
plt.figure(figsize=(12, 5))
for agents in AGENT_COUNTS:
    overlap_means = [results[size][agents]['overlap_mean'] for size in FINAL_TESTS]
    overlap_stds = [results[size][agents]['overlap_std'] for size in FINAL_TESTS]
    plt.errorbar(FINAL_TESTS, overlap_means, yerr=overlap_stds, marker='s', label=f'{agents} agents')
plt.xlabel('Map size')
plt.ylabel('Overlap (O)')
plt.title(f'Path Overlap vs map size (Servers: {num_servers})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_DIR, 'overlap_vs_size.png'))
plt.close()

# 4. Failure tolerance plots
for init_agents in FAILURE_AGENT_COUNTS:
    if init_agents > MAX_AGENTS:
        continue
    plt.figure(figsize=(10, 5))
    sizes = sorted(failure_results.keys())
    before = [failure_results[size][init_agents][0] for size in sizes if init_agents in failure_results[size]]
    after  = [failure_results[size][init_agents][1] for size in sizes if init_agents in failure_results[size]]
    if not before:
        continue
    x = np.arange(len(sizes))
    width = 0.35
    plt.bar(x - width/2, before, width, label='Before failure')
    plt.bar(x + width/2, after, width, label='After failure')
    plt.xticks(x, [f'{s}×{s}' for s in sizes])
    plt.ylabel('Coverage')
    plt.title(f'Failure tolerance: {init_agents} → {init_agents-2} agents (Servers: {num_servers})')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIDEO_DIR, f'failure_tolerance_{init_agents}agents.png'))
    plt.close()

# 5. Battery and active agents summary
plt.figure(figsize=(12, 5))
for agents in AGENT_COUNTS:
    batt_means = [results[size][agents]['battery_mean'] for size in FINAL_TESTS]
    active_means = [results[size][agents]['active_mean'] for size in FINAL_TESTS]
    plt.plot(FINAL_TESTS, batt_means, marker='o', label=f'{agents} agents (battery)')
    plt.plot(FINAL_TESTS, active_means, marker='s', linestyle='--', label=f'{agents} agents (active)')
plt.xlabel('Map size')
plt.ylabel('Value')
plt.title(f'Battery and active agents at episode end (Servers: {num_servers})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_DIR, 'battery_active_summary.png'))
plt.close()

# 6. Steps summary (new plot)
plt.figure(figsize=(12, 5))
for agents in AGENT_COUNTS:
    steps_means = [results[size][agents]['steps_mean'] for size in FINAL_TESTS]
    plt.plot(FINAL_TESTS, steps_means, marker='o', label=f'{agents} agents')
plt.xlabel('Map size')
plt.ylabel('Steps')
plt.title(f'Average Steps per Episode (Servers: {num_servers})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_DIR, 'steps_summary.png'))
plt.close()

# 7. Collision summary (about + actual)
plt.figure(figsize=(14, 6))
for agents in AGENT_COUNTS:
    aa_about = [results[size][agents]['aa_about_mean'] for size in FINAL_TESTS]
    aa_actual = [results[size][agents]['aa_actual_mean'] for size in FINAL_TESTS]
    ao_about = [results[size][agents]['ao_about_mean'] for size in FINAL_TESTS]
    ao_actual = [results[size][agents]['ao_actual_mean'] for size in FINAL_TESTS]
    plt.plot(FINAL_TESTS, aa_about, marker='o', linestyle='--', label=f'{agents} agents (AA about)')
    plt.plot(FINAL_TESTS, aa_actual, marker='o', linestyle='-', label=f'{agents} agents (AA actual)')
    plt.plot(FINAL_TESTS, ao_about, marker='s', linestyle='--', label=f'{agents} agents (AO about)')
    plt.plot(FINAL_TESTS, ao_actual, marker='s', linestyle='-', label=f'{agents} agents (AO actual)')
plt.xlabel('Map size')
plt.ylabel('Collision Count')
plt.title(f'Average Collisions per Episode (Servers: {num_servers})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_DIR, 'collision_summary.png'))
plt.close()

# ===================== Summary Table =====================
print("\n" + "="*100)
print("FINAL RESULTS SUMMARY (Zero-shot evaluation)")
print("="*100)
print(f"{'Map Size':>10} | {'Agents':>6} | {'Coverage':>10} | {'λ':>12} | {'O':>10} | {'Steps':>6} | {'Batt':>10} | {'Active':>6} | {'AA About':>8} | {'AA Act':>6} | {'AO About':>8} | {'AO Act':>6}")
print("-"*130)
for size in FINAL_TESTS:
    for agents in AGENT_COUNTS:
        r = results[size][agents]
        print(f"{size}×{size:>4} | {agents:>6} | {r['cov_mean']:>9.1%} ±{r['cov_std']:.1%} | {r['lam_mean']:>11.3f} ±{r['lam_std']:.3f} | {r['overlap_mean']:>9.3f} ±{r['overlap_std']:.3f} | {r['steps_mean']:>6.1f} | {r['battery_mean']:>9.0f} | {r['active_mean']:>6.1f} | {r['aa_about_mean']:>8.1f} | {r['aa_actual_mean']:>6.1f} | {r['ao_about_mean']:>8.1f} | {r['ao_actual_mean']:>6.1f}")
print("="*100)
print("Failure tolerance results (before → after):")
for size in FINAL_TESTS:
    for init_agents in FAILURE_AGENT_COUNTS:
        if init_agents > MAX_AGENTS:
            continue
        if init_agents in failure_results[size]:
            before, after = failure_results[size][init_agents]
            print(f"  {size}×{size}, {init_agents}→{init_agents-2}: {before:.1%} → {after:.1%}")
print("="*100)
print("Evaluation complete. Results saved to:", VIDEO_DIR)
