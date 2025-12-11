import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """Identity encoder that takes the feature map from the backbone and
    applies a linear layer after global average pooling.
    The output is a latent vector of specified dimension.
    """

    def __init__(self, backbone_feat_dim, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(backbone_feat_dim, latent_dim)

    def forward(self, feature_map):
        pooled = feature_map.mean(
            dim=[2, 3]
        )  # global avg pooling, (B, C_local, Hf, Wf) -> (B, C_local)
        return self.fc(pooled)  # (B, C_local) -> (B, D)
