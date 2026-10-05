import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.grid_world import GridWorld
from envs.observation import ObservationBuilder

class CoverageEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, config: dict, max_agents: int = 10,
                 active_agents: int = None,
                 num_servers: int = 3, server_positions: list = None,
                 obstacle_map: np.ndarray = None,
                 worker_local_radius: int = 2,
                 enforce_collisions: bool = True,
                 apply_fallback: bool = True):
        super().__init__()
        self.width = config['width']
        self.height = config['height']
        self.max_steps = config.get('max_steps', 500)
        self.max_agents = max_agents
        self.active_agents = active_agents if active_agents is not None else max_agents
        self.worker_local_radius = worker_local_radius
        self.obstacle_density = config.get('obstacle_density', 0.0)
        self.num_servers = num_servers
        self.server_positions = server_positions
        self.obstacle_map = obstacle_map
        self.enforce_collisions = enforce_collisions
        self.apply_fallback = apply_fallback

        self.world = GridWorld(
            self.width, self.height,
            num_agents=max_agents,
            obstacle_density=self.obstacle_density,
            obstacle_map=self.obstacle_map,
            min_spawn_dist=None,
            idle_threshold=config.get('idle_threshold', 8),
            d_pheromone=config.get('d_pheromone', 5),
            d_comm=config.get('d_comm', 10),
            tau_pheromone=config.get('tau_pheromone', 0.95),
            active_agents=self.active_agents,
            num_servers=self.num_servers,
            server_positions=self.server_positions,
            survival_policy=config.get('survival_policy', 'greedy_frontier'),
            battery_capacity=config.get('battery_capacity', 1000.0),
            energy_move=config.get('energy_move', 1.0),
            energy_stay=config.get('energy_stay', 0.5),
            energy_penalty=config.get('energy_penalty', 0.01),
            enforce_collisions=self.enforce_collisions,
            apply_fallback=self.apply_fallback,
            # ---- NEW: reward coefficients from config ----
            reward_new_cell=config.get('reward_new_cell', 1.0),
            reward_step=config.get('reward_step', -1.0),
            cooperative_factor=config.get('cooperative_factor', 0.25),
            collision_penalty=config.get('collision_penalty', -2.0),
        )

        self.obs_builder = ObservationBuilder(
            self.width, self.height,
            max_agents=max_agents,
            local_radius=self.worker_local_radius
        )

        self.map_size = 2 * self.worker_local_radius + 1
        in_channels = 12 + (max_agents - 1)
        scalar_dim = self.obs_builder.scalar_dim

        single_agent_obs = spaces.Dict({
            'map':       spaces.Box(0, 1, (self.map_size, self.map_size, in_channels), dtype=np.float32),
            'scalars':   spaces.Box(-np.inf, np.inf, (scalar_dim,), dtype=np.float32),
            'coverage':  spaces.Box(0, 1, (1,), dtype=np.float32),
            'agent_pos': spaces.Box(0, 1, (2,), dtype=np.float32),
        })

        if max_agents == 1:
            self.observation_space = single_agent_obs
        else:
            self.observation_space = spaces.Tuple([single_agent_obs] * max_agents)

        if max_agents == 1:
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(5)] * max_agents)

        self.steps = 0
        self.last_actions = [None] * max_agents
        self.episode_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and 'active_agents' in options:
            self.active_agents = options['active_agents']
            self.world.active_agents = self.active_agents
        self.world.reset(seed=seed)
        self.steps = 0
        self.last_actions = [None] * self.max_agents
        self.episode_reward = 0.0
        return self._get_obs(), {}

    def step(self, actions, training=True):
        self.steps += 1
        self.last_actions = list(actions)
        team_reward, done, info = self.world.step(actions, training=training)
        info['steps'] = self.steps
        rewards = [team_reward] * self.max_agents
        self.episode_reward += team_reward
        if self.steps >= self.max_steps:
            done = True
        return self._get_obs(), rewards, done, False, info

    def _get_obs(self):
        obs_list = []
        for i in range(self.max_agents):
            last_act = self.last_actions[i] if self.last_actions[i] is not None else -1
            obs_list.append(self.obs_builder.build(self.world, i, last_act))
        if self.max_agents == 1:
            return obs_list[0]
        else:
            return tuple(obs_list)

    def render(self, mode='human'):
        pass