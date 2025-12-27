import torch
import torch.nn as nn


class GaussianDecoder(nn.Module):
    """MLP head to predict Gaussian parameters taking in all the encoding.

    Args:
        C: Input feature dimension.
        out_dim: Output dimension (default 11 = 3 for scale(3)+rot(4)+alpha(1)+sh(3)).

    Returns:
        params: Tensor of shape (B, N, out_dim = 11)
    """

    def __init__(
        self, in_dim, hidden=256, out_dim=11
    ):  # scale(3)+rot(4)+alpha(1)+sh(3)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):  # (B, N, in_dim) -> (B, N, out_dim)
        # TODO[run-pipeline]: Split the output head into named fields and apply appropriate
        #   parameterizations, e.g., scales=exp(), rotation=normalize_quat(), alpha=sigmoid(), sh=tanh().
        return self.mlp(x)
