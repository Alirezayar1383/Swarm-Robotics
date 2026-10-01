import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # or 'Qt5Agg' for better interactivity – change if needed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import os

AGENT_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
UNVISITED_COLOR = [0.2, 0.4, 1.0]
VISITED_COLOR   = [0.2, 0.8, 0.2]
OBSTACLE_COLOR  = [0.1, 0.1, 0.1]

def local_obs_to_rgb(obs_map):
    h, w, _ = obs_map.shape
    rgb = np.ones((h, w, 3), dtype=np.float32) * 0.95
    unvisited = obs_map[:,:,0] > 0.5
    visited   = obs_map[:,:,1] > 0.5
    obstacle  = obs_map[:,:,2] > 0.5
    agent_pos = obs_map[:,:,3] > 0.5
    rgb[unvisited] = UNVISITED_COLOR
    rgb[visited]   = VISITED_COLOR
    rgb[obstacle]  = OBSTACLE_COLOR
    rgb[agent_pos] = [1.0, 0.2, 0.2]
    return rgb

def animate_multi(env, agent_wrappers, stage_name, max_steps=600,
                  save_video=False, video_path=None, display=False):
    """
    Animate the multi-agent environment.
    - save_video: if True, saves a GIF.
    - display: if True, shows the animation live in a window (blocks until closed).
    If both are True, it saves and also shows (but saving may take time).
    """
    num_anim_agents = len(agent_wrappers)

    obs_tuple, _ = env.reset()
    obs_list = list(obs_tuple)[:num_anim_agents]

    for a in agent_wrappers:
        a.reset_hidden()
        a.q_net.eval()

    done = False
    step = 0
    total_agent_collisions = 0
    total_obstacle_avoidances = 0
    info = {'coverage': 0.0, 'agent_agent_collisions': 0, 'obstacle_avoidances': 0}

    fig = plt.figure(figsize=(14, 3 * num_anim_agents + 2))
    gs = GridSpec(num_anim_agents, 3, width_ratios=[2, 1, 0.6])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_locals, ax_bars = [], []
    for i in range(num_anim_agents):
        ax_loc = fig.add_subplot(gs[i, 1])
        ax_bar = fig.add_subplot(gs[i, 2])
        ax_locals.append(ax_loc)
        ax_bars.append(ax_bar)
    action_names = ['Up', 'Right', 'Down', 'Left', 'Stay']

    # If live display, turn on interactive mode
    if display:
        plt.ion()
        fig.show()

    def update_frame(frame):
        nonlocal done, step, total_agent_collisions, total_obstacle_avoidances, obs_list, info
        if done or step >= max_steps:
            if display:
                plt.ioff()
                plt.close(fig)
            return []

        actions, q_vals = [], []
        for i, a in enumerate(agent_wrappers):
            act, q = a.select_action(obs_list[i], training=False)
            actions.append(act)
            q_vals.append(q)

        # Pad actions to match env.max_agents
        while len(actions) < env.max_agents:
            actions.append(4)

        next_obs_tuple, rewards, term, trunc, info = env.step(actions, training=False)
        done = term or trunc
        step += 1
        total_agent_collisions += info['agent_agent_collisions']
        total_obstacle_avoidances += info['obstacle_avoidances']
        obs_list = list(next_obs_tuple)[:num_anim_agents]

        # Draw full map
        world = env.world
        h, w = world.grid.shape
        map_rgb = np.ones((h, w, 3), dtype=np.float32) * 0.95
        for y in range(h):
            for x in range(w):
                if world.grid[y, x] == 1:
                    map_rgb[y, x] = OBSTACLE_COLOR
                elif (x, y) in world.global_visited:
                    map_rgb[y, x] = VISITED_COLOR
                else:
                    map_rgb[y, x] = UNVISITED_COLOR
        ax_map.clear()
        ax_map.imshow(map_rgb, origin='upper', interpolation='nearest')
        for i, (x, y) in enumerate(world.agent_positions):
            if i < num_anim_agents:
                circle = Circle((x, y), radius=0.3, color=AGENT_COLORS[i], ec='black', lw=1)
                ax_map.add_patch(circle)
        ax_map.set_title(f"Full map – Step {step}\nCollisions: {total_agent_collisions} | Avoid: {total_obstacle_avoidances}")
        ax_map.axis('off')

        # Local observations and Q-bars
        for i, obs in enumerate(obs_list):
            local_rgb = local_obs_to_rgb(obs['map'])
            ax_locals[i].clear()
            ax_locals[i].imshow(local_rgb, origin='upper', interpolation='nearest')
            ax_locals[i].set_title(f"Agent {i+1}")
            ax_locals[i].set_xticks([])
            ax_locals[i].set_yticks([])

            ax_bars[i].clear()
            xpos = np.arange(len(action_names))
            bars = ax_bars[i].bar(xpos, q_vals[i], color=AGENT_COLORS[i])
            ax_bars[i].set_xticks(xpos)
            ax_bars[i].set_xticklabels(action_names, rotation=45, fontsize=7)
            ax_bars[i].set_ylabel('Q', fontsize=7)
            ax_bars[i].set_ylim(min(q_vals[i].min()-0.5, -1), max(q_vals[i].max()+0.5, 1))
            for bar, val in zip(bars, q_vals[i]):
                ax_bars[i].text(bar.get_x()+bar.get_width()/2., bar.get_height(),
                                f'{val:.2f}', ha='center', va='bottom', fontsize=6)

        fig.suptitle(f"{stage_name} | Cov: {info['coverage']:.1%} | Step {step}", fontsize=12)
        fig.tight_layout(rect=[0,0,1,0.95])

        # If live display, redraw and pause briefly
        if display:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)   # adjust speed

        return []

    # Create the animation object (will be used for saving, and also for display if not using ion)
    ani = FuncAnimation(fig, update_frame, frames=max_steps, repeat=False, blit=False)

    if save_video and video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        # If we also want live display, we might need to save after the display loop.
        # We'll handle saving after the display loop if display is on.
        if display:
            print("Live display is active – GIF will be saved after closing the figure.")
            # We need to keep the figure open until user closes it.
            plt.ioff()
            plt.show()   # blocks until figure is closed
            # Then save
            ani.save(video_path, writer='pillow', fps=2)
            print(f"Video saved to {video_path}")
        else:
            ani.save(video_path, writer='pillow', fps=2)
            print(f"Video saved to {video_path}")
    elif display:
        # Just show live, no saving
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)

    print(f"{stage_name} final coverage: {info['coverage']:.1%} in {step} steps.")
    print(f"  Total agent-agent collisions: {total_agent_collisions}")
    print(f"  Total obstacle avoidances:    {total_obstacle_avoidances}")
    return info['coverage'], total_agent_collisions, total_obstacle_avoidances