import torch
import torch.nn as nn
import cv2


class IdentityEncoder(nn.Module):
    """Identity encoder that takes the feature map from the backbone and
    applies a linear layer after global average pooling.
    The output is a latent vector of specified dimension.
    """

    def __init__(self, backbone_feat_dim, latent_dim=128, mask=None):
        super().__init__()
        # TODO(Optional): Masked pooling using coord2d to extract only foreground avatar features
        self.mask = mask
        self.fc = nn.Linear(backbone_feat_dim, latent_dim)

    def forward(self, feature_map, preds, img_shape):
        # Feat: Use sampled features from the feature map to pool identity features, the sampling grid is based on the 2D non-param vertex coords
        non_vertex_coords2d = preds["non_vertex_coords2d"][
            :, 0
        ]  # (B, Nv, 2), Nv = 1024
        sampled_feats = self.grid_sample(
            feature_map, non_vertex_coords2d, img_shape
        )  # (B, Nv, C)
        pooled_feats = sampled_feats.mean(dim=1)  # (B, C)
        z_id = self.fc(pooled_feats)  # (B, latent_dim)
        return z_id  # (B, latent_dim)

    def grid_sample(self, feature_map, coord2d, img_shape):
        """Sample features from the feature map using 2D coordinates.

        coord2d: (B, Nv, 2) in pixel space
        Returns: (B, Nv, C_local)
        """
        H, W = img_shape
        # Normalize to [-1, 1]
        x = coord2d[..., 0] / (W - 1) * 2 - 1  # (B, Nv), normalized to [-1,1]
        y = coord2d[..., 1] / (H - 1) * 2 - 1  # (B, Nv), normalized to [-1,1]
        grid = torch.stack([x, y], dim=-1).unsqueeze(1)  # (B, 1, Nv, 2)

        sampled = torch.nn.functional.grid_sample(
            feature_map,  # (B, C, Hf, Wf)
            grid,  # (B, 1, Nv, 2)
            mode="bilinear",
            align_corners=True,
        )  # (B, C, 1, Nv)

        return sampled[:, :, 0, :].permute(0, 2, 1)  # (B, Nv, C_local)
