import torch


def dummy_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """A placeholder loss function that computes mean squared error."""
    return torch.mean((preds - targets) ** 2)


"""
TODO: Implement actual loss functions for training the model, there are three kinds we could consider:
1. Basic reconstruction loss (e.g., L2 loss) between predicted and ground truth images.
2. Smoothness loss on the Gaussian parameters to penalize divergence between neighbors.
3. Scale loss to regularize the size of the Gaussians.
"""
