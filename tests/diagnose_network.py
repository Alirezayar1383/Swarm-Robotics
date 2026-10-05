import sys, os, yaml, torch, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dqn_network import QNetwork, NoisyLinear
import torch.nn as nn
import torch.nn.functional as F

# ---- Simple CNN (no GRU, no identity, 4 base channels) ----
class SimpleCNN(nn.Module):
    def __init__(self, action_dim=5, scalar_dim=10, in_channels=4, hidden_size=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        conv_out = 64 * 4 * 4
        self.scalar_fc = nn.Linear(scalar_dim, 32)
        self.value = nn.Sequential(NoisyLinear(conv_out+32, hidden_size), nn.ReLU(), NoisyLinear(hidden_size, 1))
        self.advantage = nn.Sequential(NoisyLinear(conv_out+32, hidden_size), nn.ReLU(), NoisyLinear(hidden_size, action_dim))

    def forward(self, obs, hidden=None):
        map_img = obs['map'][:, :4, :, :]   # take only first 4 channels
        scalars = obs['scalars'][:, :10]    # use only base 10 scalars (no identity)
        x = self.pool(self.conv(map_img)).flatten(1)
        s = F.relu(self.scalar_fc(scalars))
        merged = torch.cat([x, s], dim=1)
        v = self.value(merged)
        a = self.advantage(merged)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q, None   # no hidden state

def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create synthetic observations
    obs1 = np.random.randn(5, 5, 4).astype(np.float32)
    obs1[2, 3, 0] = 1.0  # frontier right
    obs2 = obs1.copy()
    obs2[2, 3, 0] = 0.0
    obs2[2, 3, 2] = 1.0  # obstacle right

    map1 = torch.from_numpy(obs1).permute(2,0,1).unsqueeze(0).to(device)
    map2 = torch.from_numpy(obs2).permute(2,0,1).unsqueeze(0).to(device)
    scalars = torch.zeros(1, 10, device=device)

    # Test 1: Simple CNN
    print("\n=== Test 1: Simple CNN (4 base channels, no GRU) ===")
    net = SimpleCNN(in_channels=4, scalar_dim=10).to(device)
    with torch.no_grad():
        q1, _ = net({'map': map1, 'scalars': scalars})
        q2, _ = net({'map': map2, 'scalars': scalars})
    diff = (q1 - q2).abs().max().item()
    print(f"Max Q diff: {diff:.4f}")
    if diff > 0.1:
        print("→ Simple CNN is INPUT‑SENSITIVE. Problem is likely GRU or extra channels.")
    else:
        print("→ Simple CNN is NOT sensitive. Problem is deeper – CNN can't learn from 5×5.")

    # Test 2: Full GRU network with only 4 base channels
    print("\n=== Test 2: Full GRU network, 4 base channels only ===")
    net2 = QNetwork(action_dim=5, scalar_dim=10, in_channels=4).to(device)
    map1_4 = map1[:, :4, :, :]
    map2_4 = map2[:, :4, :, :]
    scalars2 = torch.zeros(1, 10, device=device)
    with torch.no_grad():
        q1, _ = net2({'map': map1_4, 'scalars': scalars2})
        q2, _ = net2({'map': map2_4, 'scalars': scalars2})
    diff2 = (q1 - q2).abs().max().item()
    print(f"Max Q diff: {diff2:.4f}")
    if diff2 > 0.1:
        print("→ GRU network with 4 channels IS sensitive. Extra channels/identity are the problem.")
    else:
        print("→ GRU network with 4 channels is NOT sensitive. GRU or architecture is the problem.")

    # Test 3: Full GRU with current 16 channels + identity
    print("\n=== Test 3: Full GRU network, 16 channels + identity ===")
    net3 = QNetwork(action_dim=5, scalar_dim=21, in_channels=16).to(device)
    map1_full = torch.randn(1, 16, 5, 5, device=device)
    map2_full = map1_full.clone()
    map1_full[0, 0, 2, 3] = 1.0   # frontier right
    map2_full[0, 0, 2, 3] = 0.0
    map2_full[0, 2, 2, 3] = 1.0   # obstacle right
    scalars3 = torch.zeros(1, 21, device=device)
    with torch.no_grad():
        q1, _ = net3({'map': map1_full, 'scalars': scalars3})
        q2, _ = net3({'map': map2_full, 'scalars': scalars3})
    diff3 = (q1 - q2).abs().max().item()
    print(f"Max Q diff: {diff3:.4f}")
    if diff3 > 0.1:
        print("→ Full current network IS sensitive. Training should work.")
    else:
        print("→ Full current network is NOT sensitive. Confirm this matches your training logs.")

if __name__ == "__main__":
    test()