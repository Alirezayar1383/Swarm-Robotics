import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

AGENT_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'lime', 'pink']
SERVER_COLOR = 'gold'
OBSTACLE_COLOR = 'black'
VISITED_COLOR = (0.2, 0.8, 0.2)
UNVISITED_COLOR = (0.95, 0.95, 0.95)

class SwarmVisualizerSimple:
    def __init__(self, env, agent_wrappers, max_steps=600, display_servers=True,
                 draw_server_fov=True, server_radius=5):
        self.env = env
        self.agent_wrappers = agent_wrappers
        self.max_steps = max_steps
        self.display_servers = display_servers
        self.draw_server_fov = draw_server_fov
        self.server_radius = server_radius
        self.num_agents = len(agent_wrappers)

        n_cols = min(self.num_agents, 4)
        n_rows = (self.num_agents + n_cols - 1) // n_cols

        self.fig = plt.figure(figsize=(8 + 3 * n_cols, 3 + 3 * n_rows))
        gs_main = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])

        self.ax_map = self.fig.add_subplot(gs_main[0, 0])
        self.ax_map.set_title('Swarm Map', fontsize=12, fontweight='bold')

        gs_views = gs_main[0, 1].subgridspec(n_rows, n_cols, hspace=0.1, wspace=0.1)
        self.ax_views = []
        for i in range(self.num_agents):
            row = i // n_cols
            col = i % n_cols
            ax = self.fig.add_subplot(gs_views[row, col])
            ax.set_title(f'Agent {i} View', fontsize=8)
            ax.axis('off')
            self.ax_views.append(ax)

        self.fig.suptitle(f'Swarm Simulation – {env.width}×{env.height}', fontsize=14, fontweight='bold')
        self.info_text = self.fig.text(0.02, 0.02, '', fontsize=10, verticalalignment='bottom',
                                       bbox=dict(facecolor='white', alpha=0.7))

        self.step = 0
        self.done = False
        self.info = None
        self.obs_list = None

        self._init_plots()

    def _init_plots(self):
        w, h = self.env.width, self.env.height
        self.ax_map.set_aspect('equal')
        self.ax_map.set_xlim(w, 0)
        self.ax_map.set_ylim(h, 0)
        self.ax_map.set_xlabel('X (0 at right)')
        self.ax_map.set_ylabel('Y (0 at bottom)')
        self.ax_map.grid(True, linestyle='--', alpha=0.2)

        for ax in self.ax_views:
            ax.clear()
            ax.axis('off')

        self.fig.tight_layout()

    def update(self, frame):
        if self.done or self.step >= self.max_steps:
            return

        actions = []
        for i, wrapper in enumerate(self.agent_wrappers):
            act, _ = wrapper.select_action(self.obs_list[i], training=False)
            actions.append(act)
        while len(actions) < self.env.max_agents:
            actions.append(4)
        next_obs_tuple, _, term, trunc, info = self.env.step(actions, training=False)
        self.done = term or trunc
        self.step += 1
        self.obs_list = list(next_obs_tuple)[:self.num_agents]
        self.info = info

        self._update_map()
        for i, obs in enumerate(self.obs_list):
            self._update_local_view(i, obs)

        cov = self.info['coverage'] * 100
        active = self.info.get('active_agents', self.num_agents)
        # Show battery info
        battery_avg = np.mean(self.info['battery']) / self.env.world.battery_capacity * 100 if 'battery' in self.info else 0
        self.info_text.set_text(f'Step: {self.step}  |  Cov: {cov:.1f}%  |  Active: {active}/{self.num_agents}  |  Avg Batt: {battery_avg:.0f}%')
        self.fig.suptitle(f'Swarm Simulation – {self.env.width}×{self.env.height}  |  Step {self.step}', fontsize=14)

        return []

    def _update_map(self):
        ax = self.ax_map
        ax.clear()

        world = self.env.world
        grid = world.grid
        h, w = grid.shape

        for y in range(h):
            for x in range(w):
                if grid[y, x] == 1:
                    color = OBSTACLE_COLOR
                else:
                    visited = (x, y) in world.get_global_visited()
                    color = VISITED_COLOR if visited else UNVISITED_COLOR
                rect = Rectangle((x, y), 1, 1, facecolor=color, edgecolor='gray', linewidth=0.1)
                ax.add_patch(rect)

        for i, pos in enumerate(world.agent_positions[:self.num_agents]):
            x, y = pos
            if world.agent_active[i]:
                circle = Circle((x+0.5, y+0.5), 0.3, color=AGENT_COLORS[i % len(AGENT_COLORS)],
                                edgecolor='black', linewidth=1)
                ax.add_patch(circle)
                # Battery bar (small)
                batt = world.battery[i] / world.battery_capacity
                ax.add_patch(Rectangle((x+0.1, y+0.7), 0.8, 0.15, facecolor='gray', edgecolor='none'))
                ax.add_patch(Rectangle((x+0.1, y+0.7), 0.8*batt, 0.15, facecolor='green' if batt > 0.3 else 'red', edgecolor='none'))
            else:
                # Dead agent – grey X
                ax.plot([x+0.2, x+0.8], [y+0.2, y+0.8], color='gray', lw=2)
                ax.plot([x+0.2, x+0.8], [y+0.8, y+0.2], color='gray', lw=2)
            ax.text(x+0.5, y+0.5, str(i), color='white' if world.agent_active[i] else 'gray', fontsize=8, ha='center', va='center')

        if self.display_servers and hasattr(world, 'server_network'):
            for s, (sx, sy) in enumerate(world.server_network.server_positions):
                if self.draw_server_fov:
                    fov = Rectangle((sx - self.server_radius, sy - self.server_radius),
                                    2*self.server_radius + 1, 2*self.server_radius + 1,
                                    facecolor='none', edgecolor='#FFD700', linestyle='--',
                                    linewidth=1.5, alpha=0.5)
                    ax.add_patch(fov)
                rect = Rectangle((sx, sy), 1, 1, facecolor=SERVER_COLOR, edgecolor='black',
                                 linewidth=2, alpha=0.7)
                ax.add_patch(rect)
                ax.text(sx+0.5, sy+0.5, f'S{s}', color='black', fontsize=8, ha='center', va='center')

        ax.set_xlim(w, 0)
        ax.set_ylim(h, 0)
        ax.set_xlabel('X (0 at right)')
        ax.set_ylabel('Y (0 at bottom)')
        ax.grid(True, linestyle='--', alpha=0.2)

    def _update_local_view(self, agent_idx, obs):
        ax = self.ax_views[agent_idx]
        ax.clear()
        map_data = obs['map']

        rgb = np.zeros((map_data.shape[0], map_data.shape[1], 3))

        obstacles = map_data[:, :, 0] > 0.5
        rgb[obstacles] = [0, 0, 0]

        visited = map_data[:, :, 4] > 0.5
        unvisited = (~visited) & (~obstacles)
        rgb[visited] = [0, 1, 0]
        rgb[unvisited] = [0.5, 0.8, 1]

        agent = map_data[:, :, 1] > 0.5
        rgb[agent] = [1, 0, 0]

        ax.imshow(rgb, origin='upper', interpolation='nearest')
        # Show battery scalar (normalized) if available
        if 'scalars' in obs and obs['scalars'].shape[0] > 6:
            batt = obs['scalars'][6]
            ax.set_title(f'Agent {agent_idx} View  Batt: {batt:.0%}', fontsize=8)
        else:
            ax.set_title(f'Agent {agent_idx} View', fontsize=8)
        ax.axis('off')

    def run(self):
        obs_tuple, _ = self.env.reset()
        self.obs_list = list(obs_tuple)[:self.num_agents]
        for wrapper in self.agent_wrappers:
            wrapper.reset_hidden()
            wrapper.q_net.eval()

        ani = FuncAnimation(self.fig, self.update, frames=self.max_steps,
                            interval=500, blit=False, repeat=False)
        plt.show()
        return ani
