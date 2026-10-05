import os, pickle
import numpy as np
from collections import defaultdict

class MetricsLogger:
    """Stores training metrics and full episode snapshots for later figure generation."""
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.eval_log = defaultdict(list)
        self.episode_snapshots = []

    def log_eval(self, episode, coverage, avg_q, loss, episode_length,
                 agent_positions, pheromone_maps, teammate_pheromone_maps,
                 comm_events, frontier_count, corner_time_fraction,
                 idle_counters, breadcrumb_maps,
                 lam=None, overlap=None, fstay=None, td_error=None):
        self.eval_log['episode'].append(episode)
        self.eval_log['coverage'].append(coverage)
        self.eval_log['avg_q'].append(avg_q)
        self.eval_log['loss'].append(loss)
        self.eval_log['episode_length'].append(episode_length)
        self.eval_log['agent_positions'].append(agent_positions)
        self.eval_log['pheromone_maps'].append(pheromone_maps)
        self.eval_log['teammate_pheromone_maps'].append(teammate_pheromone_maps)
        self.eval_log['comm_events'].append(comm_events)
        self.eval_log['frontier_count'].append(frontier_count)
        self.eval_log['corner_time_fraction'].append(corner_time_fraction)
        self.eval_log['idle_counters'].append(idle_counters)
        self.eval_log['breadcrumb_maps'].append(breadcrumb_maps)
        self.eval_log['lam'].append(lam if lam is not None else 0.0)
        self.eval_log['overlap'].append(overlap if overlap is not None else 0.0)
        self.eval_log['fstay'].append(fstay if fstay is not None else 0)
        self.eval_log['td_error'].append(td_error if td_error is not None else 0.0)

    def save_episode_snapshot(self, episode, trajectory, pheromone_snapshots,
                              teammate_pheromone_snapshots, territory_map, comm_graphs,
                              agent_positions_history, coverage_history):
        self.episode_snapshots.append({
            'episode': episode,
            'trajectory': trajectory,
            'pheromone_snapshots': pheromone_snapshots,
            'teammate_pheromone_snapshots': teammate_pheromone_snapshots,
            'territory_map': territory_map,
            'comm_graphs': comm_graphs,
            'agent_positions_history': agent_positions_history,
            'coverage_history': coverage_history
        })

    def save_stage(self):
        np.savez_compressed(os.path.join(self.save_dir, 'eval_log.npz'), **self.eval_log)
        with open(os.path.join(self.save_dir, 'episode_snapshots.pkl'), 'wb') as f:
            pickle.dump(self.episode_snapshots, f)
        print(f"  Logs saved to {self.save_dir}")