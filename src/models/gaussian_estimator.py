import torch
import torch.nn as nn
import torch.nn.functional as F


class AvatarGaussianEstimator(nn.Module):
    """
    Feature Sampler features and predicts Gaussian parameters
    """

    def __init__(self, template, feature_map, pred):
        super().__init__()
        self._avatar = template
        self._feature_map = feature_map  # Tensor of shape (B, C, H, W)
        self._vertices2d = pred["vertices2d"][
            0
        ]  # vertices2d[0] -> torch.Size([1, 10475, 2]), (B, N, 2)
        self._vertices3d = pred["vertices3d"][
            0
        ]  # vertices3d[0] -> torch.Size([1, 10475, 3]), (B, N, 3)
        self._gaussian_estimator = GaussianEstimator(feature_dim=feature_map.shape[1])

    def compute_gaussian_coord2d(self):
        """Vectorized computation of per-gaussian 2D centers.

        Returns:
            Tensor of shape (N,2) with Gaussian centers in the same coord space as `self._vertices2d`.
        """
        N = int(self._avatar.total_gaussians_num())
        K = int(self._avatar.num_gaussians())

        # Retrieve parents and barycentric coords from the template.
        parents = self._avatar.parents()
        bary = self._avatar.barycentric_coords()

        device = self._feature_map.device

        # Normalize types and devices
        parents = parents.to(device=device, dtype=torch.long)
        bary = bary.to(device=device, dtype=torch.float32)
        verts2d = self._vertices2d.to(device=device)

        # Create an index that maps each gaussian (0..N-1) to its barycentric row (0..K-1)
        idx = torch.arange(N, device=device) % K  # (N,)
        bary_per_gauss = bary[idx]  # (N,3)

        # parents: (N,3) -> gather vertex coordinates: face_verts (N,3,2)
        face_verts = verts2d[parents]  # (N,3,2)

        # Weighted sum over the 3 parent vertices using barycentric weights -> (N,2)
        centers2d = (face_verts * bary_per_gauss.unsqueeze(-1)).sum(dim=1)

        return centers2d

    def feature_sample(self):
        centers2d = self.compute_gaussian_coord2d()  # (N,2)
        B, C, H, W = self._feature_map.shape
        device = self._feature_map.device

        # Normalize
        x = centers2d[:, 0] / (W - 1) * 2 - 1
        y = centers2d[:, 1] / (H - 1) * 2 - 1
        centers_norm = torch.stack([x, y], dim=-1)  # (N,2)

        # Build grid: (B,1,N,2)
        grid = centers_norm.unsqueeze(0).unsqueeze(1).expand(B, 1, -1, 2)

        # Sample all N points
        # grid sample expects feature map (B, C, H, W) and grid of shape (B, H_out, W_out, 2)
        sampled = F.grid_sample(
            self._feature_map,  # (B,C,H,W)
            grid,  # (B,1,N,2)
            mode="bilinear",
            align_corners=True,
        )

        # Now sampled = (B,C,1,N)
        # Return (B,N,C)
        return sampled[:, :, 0, :].permute(0, 2, 1)

    def forward(self):
        """Predict Gaussian parameters from the sampled features.

        Returns:
            A tensor of shape (N, 1 + 4 + 3 + 3) containing Gaussian parameters:
              - 'scale': Tensor of shape (N, 3)
              - 'rotation': Tensor of shape (N, 4)
              - 'alpha': Tensor of shape (N,)
              - 'sh': Tensor of shape (N, 3)
        """
        batch_features = self.feature_sample()  # (B,N,C)
        # For now, assume batch size 1
        batch_gaussian_params = self._gaussian_estimator(
            batch_features[0]
        )  # (N, 1 + 4 + 3 + 3)
        return batch_gaussian_params


class SingleGaussianEstimator(nn.Module):
    """Lightweight MLP head that predicts batch Gaussian parameters from features."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.shape_mlp = MLPShape(C=feature_dim, out_dim=7)  # scale (3) + rotation (4)
        self.appearance_mlp = MLPAppearance(
            C=feature_dim, out_dim=4
        )  # alpha (1) + SH (3)

    def forward(self, batch_features: torch.Tensor) -> dict:
        """Predict Gaussian parameters from batch input features.

        Args:
            batch_features: Tensor of shape (B, C) where C is the feature dimension.

        Returns:
            A tensor of shape (B, 1 + 4 + 3 + 3) containing Gaussian parameters:
              - 'scale': Tensor of shape (B, 3)
              - 'rotation': Tensor of shape (B, 4)
              - 'alpha': Tensor of shape (B,)
              - 'sh': Tensor of shape (B, 3)
        """
        scale, rotation = self.shape_mlp(batch_features)
        alpha, sh = self.appearance_mlp(batch_features)
        gaussian_params = torch.stack([scale, rotation, alpha.unsqueeze(-1), sh], dim=1)
        return gaussian_params


class MLPShape(nn.Module):
    """MLP head to predict shape parameters for a single Gaussian: scale and rotation.
    (Offset translation is ignored for now)
        Args:
            C: Input feature dimension.
            out_dim: Output dimension (default 7 = 3 for scale + 4 for rotation).

        Returns:
            scale: Tensor of shape (B, 3)
            rotation: Tensor of shape (B, 4) representing a normalized quaternion.
    """

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
    """MLP head to predict appearance parameters for a single Gaussian: alpha and SH coefficients.

    Args:
        C: Input feature dimension.
        out_dim: Output dimension (alpha * 1 + SH DC * 3)
        (can be extended to 49 = 1 for alpha + 48 for SH coefficients).

    Returns:
        alpha: Tensor of shape (B,) with values in [0, 1]
        sh: Tensor of shape (B, 4) representing spherical harmonics coefficients.
    """

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
        sh = out[..., 1:]  # Consider SH DC, 3 dimensions for now
        return alpha, sh
