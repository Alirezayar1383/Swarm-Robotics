import torch
import numpy as np

class AgentWrapper:
    def __init__(self, agent_id, shared_q_network, device, mask_enabled=True):
        self.agent_id = agent_id
        self.q_net = shared_q_network
        self.device = device
        self.hidden_state = None
        self.mask_enabled = mask_enabled   # <-- new flag

    def reset_hidden(self, batch_size=1):
        gru_hidden = self.q_net.gru.hidden_size
        self.hidden_state = torch.zeros(1, batch_size, gru_hidden, device=self.device)

    def select_action(self, obs, training=True):
        map_t = torch.from_numpy(obs['map']).permute(2,0,1).unsqueeze(0).float().to(self.device)
        scal_t = torch.from_numpy(obs['scalars']).unsqueeze(0).float().to(self.device)
        obs_t = {'map': map_t, 'scalars': scal_t}

        if training:
            self.q_net.train()
        else:
            self.q_net.eval()

        with torch.no_grad():
            q_vals, new_hidden = self.q_net(obs_t, self.hidden_state)

            # If mask enabled, apply obstacle mask
            if self.mask_enabled:
                r = obs_t['map'].shape[2] // 2
                obstacle_map = obs_t['map'][:, 0, :, :]   # channel 0 = obstacles
                mask = torch.zeros_like(q_vals)
                if obstacle_map[0, r-1, r] > 0.5: mask[0, 0] = -9999.0 # Up
                if obstacle_map[0, r, r+1] > 0.5: mask[0, 1] = -9999.0 # Right
                if obstacle_map[0, r+1, r] > 0.5: mask[0, 2] = -9999.0 # Down
                if obstacle_map[0, r, r-1] > 0.5: mask[0, 3] = -9999.0 # Left
                q_vals = q_vals + mask

        self.hidden_state = new_hidden
        action = q_vals.argmax(dim=1).item()
        if not training:
            self.q_net.train()
        return action, q_vals.cpu().numpy().flatten()

    def q_values_and_features(self, obs_batch, apply_mask=True):
        q_vals, new_hidden = self.q_net(obs_batch, self.hidden_state)

        if apply_mask and self.mask_enabled:
            r = obs_batch['map'].shape[2] // 2
            obstacle_map = obs_batch['map'][:, 0, :, :]
            mask = torch.zeros_like(q_vals)
            mask[:, 0] = torch.where(obstacle_map[:, r-1, r] > 0.5, torch.tensor(-9999.0, device=self.device), torch.tensor(0.0, device=self.device))
            mask[:, 1] = torch.where(obstacle_map[:, r, r+1] > 0.5, torch.tensor(-9999.0, device=self.device), torch.tensor(0.0, device=self.device))
            mask[:, 2] = torch.where(obstacle_map[:, r+1, r] > 0.5, torch.tensor(-9999.0, device=self.device), torch.tensor(0.0, device=self.device))
            mask[:, 3] = torch.where(obstacle_map[:, r, r-1] > 0.5, torch.tensor(-9999.0, device=self.device), torch.tensor(0.0, device=self.device))
            q_vals = q_vals + mask

        self.hidden_state = new_hidden
        features = self.q_net.features(obs_batch)
        return q_vals, features, new_hidden