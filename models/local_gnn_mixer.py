import torch
import torch.nn as nn

class LocalGNNMixer(nn.Module):
    """
    VDN mixer with learnable bias. Computes and returns a Chebyshev-distance
    adjacency matrix (available for future locality-aware extensions / figures),
    but Q_tot itself is the standard VDN sum: Q_tot = sum_i(Q_i) + bias.
    NOTE: summing q_masked over both (i,j) double/N-multiplies each agent's Q —
    that was the bug in the previous version. Do not reintroduce it.
    """
    def __init__(self, d_comm: float = 10.0):
        super().__init__()
        self.d_comm = d_comm
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, agent_q_values, agent_positions, agent_features=None):
        """
        agent_q_values: (batch, num_agents, 1)
        agent_positions: (batch, num_agents, 2)
        Returns:
            Q_tot: (batch, 1)
            adj:   (batch, num_agents, num_agents)
        """
        batch_size, num_agents, _ = agent_q_values.shape

        delta = torch.abs(agent_positions.unsqueeze(2) - agent_positions.unsqueeze(1))  # (B,N,N,2)
        max_dist = delta.max(dim=-1)[0]  # (B,N,N)
        adj = (max_dist <= self.d_comm).float()

        # Plain VDN sum — each agent's Q counted exactly once.
        q_tot = agent_q_values.sum(dim=1) + self.bias  # (B,1)

        return q_tot, adj
