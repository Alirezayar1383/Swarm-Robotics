import random
from collections import deque
import numpy as np

class EpisodicReplayBuffer:
    def __init__(self, max_episodes=400):
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        self.coverages = deque(maxlen=max_episodes)

    def add_episode(self, global_states, obs_lists, actions, rewards,
                    next_global_states, next_obs_lists, dones, coverage=None):
        self.episodes.append({
            'global_states': global_states,
            'obs_lists': obs_lists,
            'actions': actions,
            'rewards': rewards,
            'next_global_states': next_global_states,
            'next_obs_lists': next_obs_lists,
            'dones': dones
        })
        self.coverages.append(coverage if coverage is not None else 0.0)

    def sample_sequences(self, batch_size, seq_len, alpha=0.0):
        """Return sequences of length seq_len+1 (to allow clean unrolling)."""
        if len(self.episodes) == 0:
            return []
        if alpha > 0:
            probs = np.array(self.coverages) ** alpha
            probs = probs / probs.sum()
            episodes = random.choices(self.episodes, weights=probs, k=batch_size)
        else:
            episodes = random.choices(self.episodes, k=batch_size)

        sequences = []
        for ep in episodes:
            T = len(ep['rewards'])
            # Work on a copy
            ep_copy = {
                'global_states': list(ep['global_states']),
                'obs_lists': list(ep['obs_lists']),
                'actions': list(ep['actions']),
                'rewards': list(ep['rewards']),
                'next_global_states': list(ep['next_global_states']),
                'next_obs_lists': list(ep['next_obs_lists']),
                'dones': list(ep['dones']),
            }
            # Need seq_len+1 transitions for clean unrolling
            needed = seq_len + 1
            if T < needed:
                pad_len = needed - T
                self._pad_episode(ep_copy, pad_len)
                T = needed
            start = random.randint(0, T - needed)
            end = start + needed
            seq_global = ep_copy['global_states'][start:end]
            seq_obs = ep_copy['obs_lists'][start:end]
            seq_actions = ep_copy['actions'][start:end-1]   # one fewer action
            seq_rewards = ep_copy['rewards'][start:end-1]
            seq_next_global = ep_copy['next_global_states'][start:end-1]
            seq_next_obs = ep_copy['next_obs_lists'][start:end-1]
            seq_dones = ep_copy['dones'][start:end-1]
            sequences.append((seq_global, seq_obs, seq_actions, seq_rewards,
                              seq_next_global, seq_next_obs, seq_dones))
        return sequences

    def _pad_episode(self, ep, pad_len):
        for _ in range(pad_len):
            ep['global_states'].append(ep['global_states'][-1])
            ep['obs_lists'].append(ep['obs_lists'][-1])
            ep['actions'].append(ep['actions'][-1])
            ep['rewards'].append(0.0)
            ep['next_global_states'].append(ep['next_global_states'][-1])
            ep['next_obs_lists'].append(ep['next_obs_lists'][-1])
            ep['dones'].append(True)

    def __len__(self):
        return len(self.episodes)