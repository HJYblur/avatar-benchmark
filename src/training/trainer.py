from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
import lightning as L
from models.nlf_backbone_adapter import NLFBackboneAdapter
from models.gaussian_estimator import AvatarGaussianEstimator
from avatar.avatar_template import AvatarTemplate


class Trainer(L.LightningModule):
    """
    Minimal training scaffold that demonstrates wiring:
    - Loads batches from a DataLoader
    - Encodes features with NLF backbone adapter
    - Predict Gaussian parameters with AvatarGaussianEstimator
    - Placeholder for localization + gaussian estimation + losses
    """

    def __init__(
        self,
        dataloader: DataLoader,
        backbone_adapter: NLFBackboneAdapter,
        gaussian_estimator: SingleGaussianEstimator,
    ):
        super().__init__()
        self.dataloader = dataloader
        self.template = AvatarTemplate().load_avatar_template()
        self.backbone = backbone_adapter
        self.gaussian_estimator = gaussian_estimator
        self.avatar_estimator = AvatarGaussianEstimator(self.template)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        image, _ = batch

        feats, preds = self.backbone.detect_with_features(image, use_half=True)

        gaussian_params = self.avatar_estimator(feats, preds, self.gaussian_estimator)

        # Reconstruct gaussian avatar and compute losses here...
        return {"loss": 0.0}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
