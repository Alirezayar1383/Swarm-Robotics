import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('eps_w', torch.empty(out_features, in_features))

        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))
        self.register_buffer('eps_b', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        self.mu_w.data.uniform_(-stdv, stdv)
        self.sigma_w.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.mu_b.data.uniform_(-stdv, stdv)
        self.sigma_b.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.eps_w.copy_(eps_out.outer(eps_in))
        self.eps_b.copy_(eps_out)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.mu_w + self.sigma_w * self.eps_w.to(x.device)
            bias = self.mu_b + self.sigma_b * self.eps_b.to(x.device)
        else:
            weight = self.mu_w
            bias = self.mu_b
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(self, action_dim=5, scalar_dim=23,   # <-- default 23 for 10 agents
                 in_channels=21,                      # <-- default 21 for 10 agents
                 conv_channels=[32, 64, 64],
                 kernel_size=3,
                 hidden_size=512, gru_hidden_size=128):
        super().__init__()
        self.action_dim = action_dim

        layers = []
        for i, out_ch in enumerate(conv_channels):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_ch, kernel_size,
                                        stride=1, padding=1))
            else:
                layers.append(nn.Conv2d(conv_channels[i-1], out_ch, kernel_size,
                                        stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(4)
        conv_out_size = conv_channels[-1] * 4 * 4    # 64*16 = 1024

        self.scalar_fc = nn.Linear(scalar_dim, 32)
        self.scalar_relu = nn.ReLU(inplace=True)

        self.feature_size = conv_out_size + 32       # 1056

        self.gru = nn.GRU(self.feature_size, gru_hidden_size, batch_first=False)

        self.value_stream = nn.Sequential(
            NoisyLinear(gru_hidden_size, hidden_size, sigma_init=0.3),
            nn.ReLU(inplace=True),
            NoisyLinear(hidden_size, 1, sigma_init=0.3)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(gru_hidden_size, hidden_size, sigma_init=0.3),
            nn.ReLU(inplace=True),
            NoisyLinear(hidden_size, action_dim, sigma_init=0.3)
        )

    def features(self, obs):
        map_img = obs['map']
        scalars = obs['scalars']
        x = self.conv(map_img)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        s = self.scalar_relu(self.scalar_fc(scalars))
        return torch.cat([x, s], dim=1)

    def forward(self, obs, hidden_state=None):
        feat = self.features(obs)
        if hidden_state is None:
            hidden_state = torch.zeros(1, feat.size(0), self.gru.hidden_size,
                                       device=feat.device)
        gru_in = feat.unsqueeze(0)
        gru_out, new_hidden = self.gru(gru_in, hidden_state)
        gru_out = gru_out.squeeze(0)

        value = self.value_stream(gru_out)
        advantage = self.advantage_stream(gru_out)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q, new_hidden

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()