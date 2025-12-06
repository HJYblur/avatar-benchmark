import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPShape(nn.Module):
    def __init__(self, C, out_dim=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(C, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, f):
        out = self.mlp(f)
        scale = F.softplus(out[..., :3])  # scale, keep it positive and stable
        rot = out[..., 3:7]  # quaternion
        rot = rot / rot.norm(dim=-1, keepdim=True)
        return scale, rot


class MLPAppearance(nn.Module):
    def __init__(self, C, out_dim=49):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(C, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, f):
        out = self.mlp(f)
        alpha = torch.sigmoid(out[..., 0])
        sh = out[..., 1:]  # Consider SH DC + rest, 48 dimensions in total
        return alpha, sh

