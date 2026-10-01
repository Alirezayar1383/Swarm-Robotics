import sys, os, yaml, torch, torch.nn.functional as F, numpy as np, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.coverage_env import CoverageEnv
from models.dqn_network import QNetwork
from models.local_gnn_mixer import LocalGNNMixer
from agents.agent_wrapper import AgentWrapper
from agents.simple_buffer import EpisodicReplayBuffer
from utils.metrics_logger import MetricsLogger
import platform
if platform.system() == 'Windows':
    import msvcrt
else:
    msvcrt = None

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'phase9_config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

MAX_AGENTS = cfg['max_agents']
WORKER_RADIUS = cfg['worker_local_radius']   # 2 (5x5)
SERVER_RADIUS = cfg['server_local_radius']   # 5 (10x10)
IN_CHANNELS = 12 + (MAX_AGENTS - 1)
SCALAR_DIM = 15 + MAX_AGENTS + 1   # +1 for battery
GAMMA = cfg['gamma']
TAU = cfg['tau']
LR_AGENT = cfg['lr_agent']
LR_MIXER = cfg['lr_mixer']
WEIGHT_DECAY = cfg['weight_decay']
BATCH_SIZE = cfg['batch_size']
SEQ_LEN = cfg['sequence_length']
MIN_EPISODES = cfg['min_episodes_before_training']
MAX_EPISODES_IN_BUFFER = cfg['max_episodes_in_buffer']
TARGET_COVERAGE = cfg['target_coverage']
STREAK_TARGET = cfg['streak_for_early_stop']
EVAL_FREQ = cfg['eval_freq']
STAGES = cfg['stages']
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', cfg['checkpoint_dir'])
BEST_MODEL = cfg['best_model_name']
VIDEO_DIR = os.path.join(os.path.dirname(__file__), '..', cfg['video_dir'])
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

best_model_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL)

WARMUP_EPISODES = cfg['warmup_episodes']
PRIORITIZED_UPDATES = cfg['prioritized_updates']
PRIORITIZED_ALPHA = cfg['prioritized_alpha']
CHI = cfg['chi']
SNAPSHOT_FREQ = 20
N_EVAL = 3

ROLLBACK_PATIENCE = cfg.get('rollback_patience', 40)
MAX_ROLLBACKS = cfg.get('max_rollbacks', 3)
PRIORITIZED_ALPHA_FINAL = cfg.get('prioritized_alpha_final', 0.3)

# Battery parameters
BATTERY_CAPACITY = cfg.get('battery_capacity', 1000)
ENERGY_MOVE = cfg.get('energy_move', 1.0)
ENERGY_STAY = cfg.get('energy_stay', 0.5)
ENERGY_PENALTY = cfg.get('energy_penalty', 0.01)

K_S = 5
K_P = 3

def user_wants_early_stop():
    if msvcrt is None:
        return False
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b'\x1b', b'q'):
            return True
    return False

def preprocess_agent_batch(obs_list, device):
    maps = torch.from_numpy(np.array([o['map'] for o in obs_list], dtype=np.float32)).permute(0,3,1,2).float().to(device)
    scalars = torch.from_numpy(np.array([o['scalars'] for o in obs_list], dtype=np.float32)).float().to(device)
    return {'map': maps, 'scalars': scalars}

def update(agent_wrappers, buffer, mixer, target_mixer, mixer_optimizer,
           shared_optimizer, target_q_net, alpha=0.0):
    sequences = buffer.sample_sequences(BATCH_SIZE, SEQ_LEN, alpha=alpha)
    if len(sequences) < BATCH_SIZE:
        return None

    for a in agent_wrappers:
        a.reset_hidden(batch_size=BATCH_SIZE)

    target_hidden = [torch.zeros(1, BATCH_SIZE, target_q_net.gru.hidden_size, device=device)
                     for _ in range(MAX_AGENTS)]

    online_q_all = torch.zeros(BATCH_SIZE, MAX_AGENTS, SEQ_LEN + 1, 5, device=device)
    target_q_all = torch.zeros_like(online_q_all)

    for step in range(SEQ_LEN + 1):
        for i, a in enumerate(agent_wrappers):
            obs_list_i = [s[1][step][i] for s in sequences]
            obs_batch_i = preprocess_agent_batch(obs_list_i, device)
            q_i, _, _ = a.q_values_and_features(obs_batch_i)
            online_q_all[:, i, step, :] = q_i

        for i in range(MAX_AGENTS):
            obs_list_i = [s[1][step][i] for s in sequences]
            obs_batch_i = preprocess_agent_batch(obs_list_i, device)
            with torch.no_grad():
                q_t, new_h = target_q_net(obs_batch_i, target_hidden[i])
                target_q_all[:, i, step, :] = q_t
                target_hidden[i] = new_h

    total_loss = 0.0
    for t in range(SEQ_LEN):
        current_q = online_q_all[:, :, t, :]
        next_q_online = online_q_all[:, :, t+1, :]
        next_q_target = target_q_all[:, :, t+1, :]

        rewards = torch.tensor([s[3][t] for s in sequences], dtype=torch.float, device=device).unsqueeze(1)
        dones = torch.tensor([s[6][t] for s in sequences], dtype=torch.float, device=device).unsqueeze(1)
        actions = torch.tensor([s[2][t] for s in sequences], dtype=torch.long, device=device)

        chosen_q = current_q.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        next_actions = next_q_online.argmax(dim=2)
        next_chosen_q = next_q_target.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)

        q_tot, _ = mixer(
            agent_q_values=chosen_q.unsqueeze(-1),
            agent_positions=torch.zeros(BATCH_SIZE, MAX_AGENTS, 2, device=device),
            agent_features=None
        )

        with torch.no_grad():
            next_q_tot, _ = target_mixer(
                agent_q_values=next_chosen_q.unsqueeze(-1),
                agent_positions=torch.zeros(BATCH_SIZE, MAX_AGENTS, 2, device=device),
                agent_features=None
            )

        td_target = rewards + GAMMA * next_q_tot * (1.0 - dones)
        td_target = torch.clamp(td_target, -50.0, 50.0)

        loss = F.smooth_l1_loss(q_tot, td_target)
        total_loss += loss

    mixer_optimizer.zero_grad()
    shared_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(mixer.parameters(), 5.0)
    torch.nn.utils.clip_grad_norm_(agent_wrappers[0].q_net.parameters(), 2.5)
    mixer_optimizer.step()
    shared_optimizer.step()

    for tp, op in zip(target_mixer.parameters(), mixer.parameters()):
        tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)
    for tp, op in zip(target_q_net.parameters(), agent_wrappers[0].q_net.parameters()):
        tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

    return total_loss.item() / SEQ_LEN

def run_one_eval_episode(env_config, agent_wrappers, num_servers, server_positions):
    eval_env = CoverageEnv(env_config, max_agents=MAX_AGENTS,
                           active_agents=MAX_AGENTS,
                           num_servers=num_servers,
                           server_positions=server_positions,
                           worker_local_radius=WORKER_RADIUS)
    eval_obs, _ = eval_env.reset()
    eval_obs_list = list(eval_obs)
    for a in agent_wrappers:
        a.reset_hidden()
        a.q_net.eval()
    edone = False
    total_q = 0.0
    q_count = 0
    episode_forced_stay_count = [0] * MAX_AGENTS
    einfo = None
    while not edone:
        actions = []
        for i in range(MAX_AGENTS):
            act, q = agent_wrappers[i].select_action(eval_obs_list[i], training=False)
            actions.append(act)
            total_q += np.max(q)
            q_count += 1

        # Robustness monitor (idle check) – only for active agents
        for i in range(MAX_AGENTS):
            if eval_env.world.agent_active[i]:
                if eval_env.world.idle_counters[i] >= K_S or eval_env.world._is_looping(i):
                    x, y = eval_env.world.agent_positions[i]
                    actions[i] = eval_env.world._committed_fallback_action(i, x, y)

        # Improved filter 2 (iterative)
        proposed = []
        for i in range(MAX_AGENTS):
            x, y = eval_env.world.agent_positions[i]
            act = actions[i]
            if act == 0:   ny = y - 1; nx = x
            elif act == 1: nx = x + 1; ny = y
            elif act == 2: ny = y + 1; nx = x
            elif act == 3: nx = x - 1; ny = y
            else:          nx, ny = x, y
            proposed.append((nx, ny))

        staying_positions = set()
        for i in range(MAX_AGENTS):
            if actions[i] == 4:
                staying_positions.add(eval_env.world.agent_positions[i])

        for i in range(MAX_AGENTS):
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
                    abs(eval_env.world.agent_positions[i][0] - cell[0])
                    + abs(eval_env.world.agent_positions[i][1] - cell[1]),
                    -episode_forced_stay_count[i],
                    i
                ))
                for loser in agents[1:]:
                    actions[loser] = 4
                    episode_forced_stay_count[loser] += 1

        for i in range(MAX_AGENTS):
            for j in range(i+1, MAX_AGENTS):
                if (proposed[i] == eval_env.world.agent_positions[j] and
                    proposed[j] == eval_env.world.agent_positions[i]):
                    actions[i] = 4
                    actions[j] = 4
                    episode_forced_stay_count[i] += 1
                    episode_forced_stay_count[j] += 1

        eval_obs, _, eterm, etrunc, einfo = eval_env.step(actions, training=False)
        edone = eterm or etrunc
        eval_obs_list = list(eval_obs)

    cov = einfo['coverage']
    T = einfo['steps']
    C0 = eval_env.world.free_cells
    J = MAX_AGENTS
    lam = T / C0 if C0 > 0 else 0
    overlap = (T - C0/J) / (C0/J) if C0 > 0 else 0
    avg_q = total_q / max(1, q_count)
    fstay = sum(episode_forced_stay_count)

    return {
        'coverage': cov, 'lam': lam, 'overlap': overlap,
        'avg_q': avg_q, 'fstay': fstay,
        'agent_positions': eval_env.world.agent_positions.copy(),
        'world': eval_env.world,
        'server_coverages': einfo.get('server_coverages', [])
    }

def collect_snapshot(env, agent_wrappers, max_steps):
    obs_tuple, _ = env.reset()
    obs_list = list(obs_tuple)
    for a in agent_wrappers:
        a.reset_hidden()
        a.q_net.eval()
    done = False
    step = 0
    traj = []
    pheromone_snaps = []
    teammate_pheromone_snaps = []
    coverage_hist = []
    pos_hist = []
    episode_forced_stay_count = [0] * MAX_AGENTS
    while not done and step < max_steps:
        actions = []
        for i in range(MAX_AGENTS):
            act, _ = agent_wrappers[i].select_action(obs_list[i], training=False)
            actions.append(act)
        # Robustness monitor
        for i in range(MAX_AGENTS):
            if env.world.agent_active[i]:
                if env.world.idle_counters[i] >= K_S or env.world._is_looping(i):
                    x, y = env.world.agent_positions[i]
                    actions[i] = env.world._committed_fallback_action(i, x, y)
        # Improved filter 2
        proposed = []
        for i in range(MAX_AGENTS):
            x, y = env.world.agent_positions[i]
            act = actions[i]
            if act == 0:   ny = y - 1; nx = x
            elif act == 1: nx = x + 1; ny = y
            elif act == 2: ny = y + 1; nx = x
            elif act == 3: nx = x - 1; ny = y
            else:          nx, ny = x, y
            proposed.append((nx, ny))
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
        for i in range(MAX_AGENTS):
            for j in range(i+1, MAX_AGENTS):
                if (proposed[i] == env.world.agent_positions[j] and
                    proposed[j] == env.world.agent_positions[i]):
                    actions[i] = 4
                    actions[j] = 4
                    episode_forced_stay_count[i] += 1
                    episode_forced_stay_count[j] += 1
        obs_tuple, _, term, trunc, info = env.step(actions, training=False)
        done = term or trunc
        obs_list = list(obs_tuple)
        traj.append(env.world.agent_positions.copy())
        pos_hist.append(env.world.agent_positions.copy())
        coverage_hist.append(info['coverage'])
        if step % 50 == 0:
            pheromone_snaps.append([env.world.pheromone[i].copy() for i in range(MAX_AGENTS)])
            teammate_pheromone_snaps.append([env.world.teammate_pheromone[i].copy() for i in range(MAX_AGENTS)])
        step += 1
    territory = np.full((env.height, env.width), -1, dtype=int)
    for y in range(env.height):
        for x in range(env.width):
            if env.world.grid[y, x] == 0 and (x, y) in env.world.get_global_visited():
                min_dist = 999
                closest = 0
                for i, (ax, ay) in enumerate(env.world.agent_positions):
                    d = abs(ax - x) + abs(ay - y)
                    if d < min_dist:
                        min_dist = d
                        closest = i
                territory[y, x] = closest
    comm_graphs = []
    for t_pos in pos_hist[::50]:
        adj = np.zeros((MAX_AGENTS, MAX_AGENTS), dtype=bool)
        for i in range(MAX_AGENTS):
            for j in range(i+1, MAX_AGENTS):
                if max(abs(t_pos[i][0] - t_pos[j][0]), abs(t_pos[i][1] - t_pos[j][1])) <= cfg['d_comm']:
                    adj[i, j] = True
        comm_graphs.append(adj)
    return {
        'trajectory': traj,
        'pheromone_snaps': pheromone_snaps,
        'teammate_pheromone_snaps': teammate_pheromone_snaps,
        'territory_map': territory,
        'comm_graphs': comm_graphs,
        'agent_positions_history': pos_hist,
        'coverage_history': coverage_hist,
        'final_coverage': info['coverage'],
        'steps': step
    }

all_stage_eps = []
all_stage_covs = []
all_stage_qs = []

for stage_idx, (w, h, max_ep) in enumerate(STAGES):
    print(f"\n{'='*50}")
    print(f"Stage {stage_idx+1}: {w}×{h} with up to {MAX_AGENTS} agents, up to {max_ep} episodes")

    # ---- Determine number of servers and positions based on map size ----
    map_size = w  # assume square
    if map_size <= 10:
        num_servers = 1
        server_positions = [(w//2, h//2)]
    else:
        num_servers = 2
        server_positions = [(w//4, h//2), (3*w//4, h//2)]
        if map_size > 25:
            num_servers = 3
            server_positions = [(w//4, h//2), (w//2, h//2), (3*w//4, h//2)]

    env_config = {
        'width': w, 'height': h,
        'max_steps': max(60, w * h * cfg['max_steps_factor']),
        'obstacle_density': cfg['obstacle_density'],
        'idle_threshold': cfg['idle_threshold'],
        'd_pheromone': cfg['d_pheromone'],
        'd_comm': cfg['d_comm'],
        'tau_pheromone': cfg['tau_pheromone'],
        'survival_policy': cfg['survival_policy'],
        'battery_capacity': BATTERY_CAPACITY,
        'energy_move': ENERGY_MOVE,
        'energy_stay': ENERGY_STAY,
        'energy_penalty': ENERGY_PENALTY,
    }

    # Create environment with server parameters and worker_local_radius
    env = CoverageEnv(env_config, max_agents=MAX_AGENTS,
                      active_agents=MAX_AGENTS,
                      num_servers=num_servers,
                      server_positions=server_positions,
                      worker_local_radius=WORKER_RADIUS)

    buffer = EpisodicReplayBuffer(max_episodes=MAX_EPISODES_IN_BUFFER)

    # ---- Network with updated scalar dimension ----
    shared_q_net = QNetwork(action_dim=5, scalar_dim=SCALAR_DIM, in_channels=IN_CHANNELS).to(device)
    target_q_net = QNetwork(action_dim=5, scalar_dim=SCALAR_DIM, in_channels=IN_CHANNELS).to(device)
    target_q_net.load_state_dict(shared_q_net.state_dict())
    shared_optimizer = torch.optim.Adam(shared_q_net.parameters(), lr=LR_AGENT, weight_decay=WEIGHT_DECAY)

    if stage_idx == 0:
        print("Running synthetic observation diff test with real environment...")
        test_obs_tuple, _ = env.reset()
        if MAX_AGENTS > 1:
            obs0 = test_obs_tuple[0]
        else:
            obs0 = test_obs_tuple
        for _ in range(5):
            test_obs_tuple, _, _, _, _ = env.step([1] * MAX_AGENTS)
        if MAX_AGENTS > 1:
            obs1 = test_obs_tuple[0]
        else:
            obs1 = test_obs_tuple
        map_frontier = obs1['map'].copy()
        r = WORKER_RADIUS
        map_frontier[r, r + 2, 0] = 1.0
        map_obstacle = map_frontier.copy()
        map_obstacle[r, r + 2, 0] = 0.0
        map_obstacle[r, r + 2, 2] = 1.0
        map1 = torch.from_numpy(map_frontier).permute(2,0,1).unsqueeze(0).float().to(device)
        map2 = torch.from_numpy(map_obstacle).permute(2,0,1).unsqueeze(0).float().to(device)
        scalars = torch.from_numpy(obs1['scalars']).unsqueeze(0).float().to(device)
        with torch.no_grad():
            q1, _ = shared_q_net({'map': map1, 'scalars': scalars})
            q2, _ = shared_q_net({'map': map2, 'scalars': scalars})
        diff = (q1 - q2).abs().max().item()
        print(f"Max Q difference between frontier-right and obstacle-right: {diff:.4f}")
        if diff < 0.1:
            print("WARNING: Network is barely responding to observation changes (expected pre-training).")
        else:
            print("Network is input-sensitive - proceeding with training.")

    agent_wrappers = [AgentWrapper(i, shared_q_net, device) for i in range(MAX_AGENTS)]

    mixer = LocalGNNMixer(d_comm=cfg['d_comm']).to(device)
    target_mixer = LocalGNNMixer(d_comm=cfg['d_comm']).to(device)
    target_mixer.load_state_dict(mixer.state_dict())
    mixer_optimizer = torch.optim.Adam(mixer.parameters(), lr=LR_MIXER)

    if stage_idx > 0 and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        shared_q_net.load_state_dict(checkpoint['agent_net'])
        mixer.load_state_dict(checkpoint['mixer'])
        target_mixer.load_state_dict(checkpoint['mixer'])
        target_q_net.load_state_dict(checkpoint['agent_net'])
        print("  Loaded best model from previous stage")

    logger = MetricsLogger(save_dir=os.path.join(VIDEO_DIR, f"stage_{stage_idx+1}_logs"))

    best_cov = 0.0
    streak = 0
    update_count = 0
    loss_val = None
    stage_eps, stage_covs, stage_qs = [], [], []

    episodes_since_improvement = 0
    rollback_count = 0

    print("  Press ESC (or 'q') to stop this stage early.")

    for ep in range(max_ep):
        # Sample number of agents
        r = np.random.random()
        if r < 0.3:
            active_agents = 1
        elif r < 0.5:
            active_agents = np.random.randint(2, MAX_AGENTS)
        else:
            active_agents = MAX_AGENTS

        env.active_agents = active_agents
        env.world.active_agents = active_agents
        obs_tuple, _ = env.reset(options={'active_agents': active_agents})
        obs_list = list(obs_tuple)

        for a in agent_wrappers:
            a.reset_hidden()
            a.q_net.train()
            a.q_net.reset_noise()

        done = False
        ep_obs_lists, ep_actions, ep_rewards = [], [], []
        ep_next_obs_lists, ep_dones = [], []

        while not done:
            if ep < WARMUP_EPISODES:
                actions = [np.random.randint(5) for _ in range(MAX_AGENTS)]
            else:
                actions = []
                for i in range(MAX_AGENTS):
                    if i < active_agents:
                        act, _ = agent_wrappers[i].select_action(obs_list[i], training=True)
                        actions.append(act)
                    else:
                        actions.append(4)

            # Improved filter 2
            proposed = []
            for i in range(active_agents):
                x, y = env.world.agent_positions[i]
                act = actions[i]
                if act == 0:   ny = y - 1; nx = x
                elif act == 1: nx = x + 1; ny = y
                elif act == 2: ny = y + 1; nx = x
                elif act == 3: nx = x - 1; ny = y
                else:          nx, ny = x, y
                proposed.append((nx, ny))

            staying_positions = set()
            for i in range(active_agents):
                if actions[i] == 4:
                    staying_positions.add(env.world.agent_positions[i])

            for i in range(active_agents):
                if actions[i] != 4:
                    if proposed[i] in staying_positions:
                        actions[i] = 4

            cell_to_agents = {}
            for i, (nx, ny) in enumerate(proposed):
                cell_to_agents.setdefault((nx, ny), []).append(i)

            for cell, agents in cell_to_agents.items():
                if len(agents) > 1:
                    agents.sort(key=lambda i: abs(env.world.agent_positions[i][0] - cell[0]) + abs(env.world.agent_positions[i][1] - cell[1]))
                    for loser in agents[1:]:
                        actions[loser] = 4

            for i in range(active_agents):
                for j in range(i+1, active_agents):
                    if (proposed[i] == env.world.agent_positions[j] and
                        proposed[j] == env.world.agent_positions[i]):
                        actions[i] = 4
                        actions[j] = 4

            next_obs_tuple, rewards, term, trunc, info = env.step(actions, training=True)
            done = term or trunc
            next_obs_list = list(next_obs_tuple)
            ep_obs_lists.append(obs_list)
            ep_actions.append(actions)
            ep_rewards.append(rewards[0])
            ep_next_obs_lists.append(next_obs_list)
            ep_dones.append(done)
            obs_list = next_obs_list

        # Add episode to buffer with dummy global states (empty lists)
        buffer.add_episode(
            global_states=[],
            obs_lists=ep_obs_lists,
            actions=ep_actions,
            rewards=ep_rewards,
            next_global_states=[],
            next_obs_lists=ep_next_obs_lists,
            dones=ep_dones,
            coverage=info['coverage']
        )

        if ep >= WARMUP_EPISODES and len(buffer) >= MIN_EPISODES:
            if update_count < PRIORITIZED_UPDATES:
                alpha = PRIORITIZED_ALPHA
            else:
                alpha = PRIORITIZED_ALPHA_FINAL

            loss_val = update(agent_wrappers, buffer, mixer, target_mixer, mixer_optimizer,
                              shared_optimizer, target_q_net, alpha=alpha)
            update_count += 1

        if ep % EVAL_FREQ == 0:
            results = [run_one_eval_episode(env_config, agent_wrappers, num_servers, server_positions) for _ in range(N_EVAL)]

            cov = float(np.mean([r['coverage'] for r in results]))
            lam = float(np.mean([r['lam'] for r in results]))
            overlap = float(np.mean([r['overlap'] for r in results]))
            avg_q = float(np.mean([r['avg_q'] for r in results]))
            fstay_mean = float(np.mean([r['fstay'] for r in results]))
            last = results[-1]

            stage_eps.append(ep)
            stage_covs.append(cov)
            stage_qs.append(avg_q)

            if cov > best_cov:
                best_cov = cov
                episodes_since_improvement = 0
                torch.save({
                    'agent_net': shared_q_net.state_dict(),
                    'mixer': mixer.state_dict()
                }, best_model_path)
            else:
                if best_cov >= TARGET_COVERAGE:
                    episodes_since_improvement = 0
                else:
                    episodes_since_improvement += 1

            if (best_cov < TARGET_COVERAGE and
                episodes_since_improvement >= ROLLBACK_PATIENCE and
                rollback_count < MAX_ROLLBACKS):
                print(f"  No improvement for {ROLLBACK_PATIENCE} evals — rolling back to best checkpoint "
                      f"and halving learning rate.")
                checkpoint = torch.load(best_model_path, map_location=device)
                shared_q_net.load_state_dict(checkpoint['agent_net'])
                target_q_net.load_state_dict(checkpoint['agent_net'])
                mixer.load_state_dict(checkpoint['mixer'])
                target_mixer.load_state_dict(checkpoint['mixer'])
                for g in shared_optimizer.param_groups:
                    g['lr'] *= 0.5
                for g in mixer_optimizer.param_groups:
                    g['lr'] *= 0.5
                episodes_since_improvement = 0
                rollback_count += 1

            if cov >= TARGET_COVERAGE:
                streak += 1
            else:
                streak = 0

            print(f"  Ep {ep:4d} | Agents: {active_agents:2d} | Cov: {cov:.1%} (avg of {N_EVAL}) | Best: {best_cov:.1%} | "
                  f"Avg Q: {avg_q:.3f} | λ: {lam:.3f} | O: {overlap:.3f} | FStay: {fstay_mean:.1f}")

            world = last['world']
            frontier_cells = 0
            global_visited = world.get_global_visited()
            for y in range(world.height):
                for x in range(world.width):
                    if world.grid[y,x]==0 and (x,y) in global_visited:
                        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nx,ny = x+dx,y+dy
                            if 0<=nx<world.width and 0<=ny<world.height \
                               and world.grid[ny,nx]==0 \
                               and (nx,ny) not in global_visited:
                                frontier_cells += 1
                                break
            comm_events = 0
            pos = world.agent_positions
            for i in range(MAX_AGENTS):
                for j in range(i+1, MAX_AGENTS):
                    if max(abs(pos[i][0]-pos[j][0]), abs(pos[i][1]-pos[j][1])) <= cfg['d_comm']:
                        comm_events += 1

            logger.log_eval(
                episode=ep,
                coverage=cov,
                avg_q=avg_q,
                loss=loss_val,
                episode_length=info['steps'],
                agent_positions=world.agent_positions.copy(),
                pheromone_maps=[world.pheromone[i].copy() for i in range(MAX_AGENTS)],
                teammate_pheromone_maps=[world.teammate_pheromone[i].copy() for i in range(MAX_AGENTS)],
                comm_events=comm_events,
                frontier_count=frontier_cells,
                corner_time_fraction=0.0,
                idle_counters=world.idle_counters.copy(),
                breadcrumb_maps=[world.breadcrumb[i].copy() for i in range(MAX_AGENTS)]
            )

            if streak >= STREAK_TARGET:
                print(f"  Early stop – {STREAK_TARGET}x >= {TARGET_COVERAGE:.0%} (averaged)")
                break

        if ep % SNAPSHOT_FREQ == 0:
            snap_env = CoverageEnv(env_config, max_agents=MAX_AGENTS,
                                   active_agents=MAX_AGENTS,
                                   num_servers=num_servers,
                                   server_positions=server_positions,
                                   worker_local_radius=WORKER_RADIUS)
            snap_data = collect_snapshot(snap_env, agent_wrappers, env_config['max_steps'])
            logger.save_episode_snapshot(
                episode=ep,
                trajectory=snap_data['trajectory'],
                pheromone_snapshots=snap_data['pheromone_snaps'],
                teammate_pheromone_snapshots=snap_data['teammate_pheromone_snaps'],
                territory_map=snap_data['territory_map'],
                comm_graphs=snap_data['comm_graphs'],
                agent_positions_history=snap_data['agent_positions_history'],
                coverage_history=snap_data['coverage_history']
            )

        if user_wants_early_stop():
            print("  ESC pressed - stopping stage early.")
            break

    all_stage_eps.append(stage_eps)
    all_stage_covs.append(stage_covs)
    all_stage_qs.append(stage_qs)

    logger.save_stage()
    print(f"  Stage {stage_idx+1} completed. Best coverage: {best_cov:.1%}")

# Final summary plots
num_stages = len(STAGES)
fig, axes = plt.subplots(2, num_stages, figsize=(4*num_stages, 8))
if num_stages == 1:
    axes = axes.reshape(2, 1)
for i in range(num_stages):
    axes[0, i].plot(all_stage_eps[i], all_stage_covs[i], 'b-', linewidth=2)
    axes[0, i].set_title(f"Stage {i+1} Coverage")
    axes[0, i].set_xlabel("Episode")
    axes[0, i].set_ylabel("Coverage")
    axes[0, i].grid(True, alpha=0.3)

    axes[1, i].plot(all_stage_eps[i], all_stage_qs[i], 'r-', linewidth=2)
    axes[1, i].set_title(f"Stage {i+1} Avg Q")
    axes[1, i].set_xlabel("Episode")
    axes[1, i].set_ylabel("Avg Q")
    axes[1, i].grid(True, alpha=0.3)

fig.suptitle("Curriculum Training Summary", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_DIR, "curriculum_summary.png"))
plt.close(fig)

print("All stages complete.")