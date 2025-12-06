import torch


def zeros_like_loss(pred: torch.Tensor) -> torch.Tensor:
    """Placeholder loss that returns a scalar zero tensor."""
    return pred.new_zeros(())


def localization_losses_stub(*_args, **_kwargs):
    """Scaffold placeholder for 2D/3D localization losses."""
    return {
        "loss2d": torch.tensor(0.0),
        "loss3d_rel": torch.tensor(0.0),
        "loss3d_abs": torch.tensor(0.0),
        "total": torch.tensor(0.0),
    }


def gaussian_param_regularizers_stub(*_args, **_kwargs):
    """Scaffold placeholder for gaussian parameter regularization."""
    return {"loss_gauss": torch.tensor(0.0)}

