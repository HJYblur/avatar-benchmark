import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure
from avatar_utils.config import load_config

class LossFunctions(nn.Module):
    def __init__(self, weight_rgb):
        super().__init__()
        self.weight_rgb = weight_rgb if weight_rgb is not None else float(load_config().get("train", {}).get("weight_rgb", 1.0))

    def rgb_loss(self, pred_imgs, gt_imgs):
        l1_loss = nn.L1Loss(mean=True)
        ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)  # Assuming images are normalized to [0,1]
        return 0.8 * l1_loss(pred_imgs, gt_imgs) + 0.2 * (1 - ssim_loss(pred_imgs, gt_imgs))

    def forward(self, pred_imgs, gt_imgs):
        loss = self.weight_rgb * self.rgb_loss(pred_imgs, gt_imgs)
        # TODO: Add more loss components (e.g., regularization on Gaussian parameters) and corresponding weights from config
        return loss
