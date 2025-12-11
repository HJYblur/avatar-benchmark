import torch
import torch.nn as nn


class SingleGaussianEstimator(nn.Module):
    """Lightweight MLP head that predicts batch Gaussian parameters from features.

    Returns a per-sample parameter vector composed of:
      scale(3) + rotation_quaternion(4) + alpha(1) + sh_dc(3) => 11 dims
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.shape_mlp = MLPShape(C=feature_dim, out_dim=7)  # scale (3) + rotation (4)
        self.appearance_mlp = MLPAppearance(
            C=feature_dim, out_dim=4
        )  # alpha (1) + SH (3)

    def forward(self, batch_features: torch.Tensor) -> torch.Tensor:
        """Predict Gaussian parameters from batch input features.

        Args:
            batch_features: Tensor of shape (B, C) where C is the feature dimension.

        Returns:
            A tensor of shape (B, 11) containing concatenated parameters:
              [scale(3), rotation_quat(4), alpha(1), sh_dc(3)]
        """
        scale, rotation = self.shape_mlp(batch_features)
        alpha, sh = self.appearance_mlp(batch_features)
        gaussian_params = torch.cat([scale, rotation, alpha.unsqueeze(-1), sh], dim=-1)
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
        # Normalize quaternion with epsilon to avoid NaNs
        eps = 1e-8
        rot = rot / (rot.norm(dim=-1, keepdim=True) + eps)
        return scale, rot


class MLPAppearance(nn.Module):
    """MLP head to predict appearance parameters for a single Gaussian: alpha and SH coefficients.

    Args:
        C: Input feature dimension.
        out_dim: Output dimension (alpha * 1 + SH DC * 3)
        (can be extended to 49 = 1 for alpha + 48 for SH coefficients).

    Returns:
        alpha: Tensor of shape (B,) with values in [0, 1]
        sh: Tensor of shape (B, 3) representing SH DC coefficients.
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
