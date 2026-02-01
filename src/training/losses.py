import torch


def L1_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the L1 loss between predictions and targets."""
    return torch.mean(torch.abs(preds - targets))


def L2_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the L2 loss between predictions and targets."""
    return torch.mean((preds - targets) ** 2)


"""
TODO: Implement actual loss functions for training the model, there are three kinds we could consider:
1. Basic reconstruction loss (e.g., L2 loss) between predicted and ground truth images.
2. Smoothness loss on the Gaussian parameters to penalize divergence between neighbors.
3. Scale loss to regularize the size of the Gaussians.
"""
