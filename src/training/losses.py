import torch


def dummy_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """A placeholder loss function that computes mean squared error."""
    return torch.mean((preds - targets) ** 2)
