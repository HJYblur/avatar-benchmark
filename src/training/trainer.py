from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
import lightning as L
from encoder.nlf_backbone_adapter import NLFBackboneAdapter
from encoder.gaussian_estimator import AvatarGaussianEstimator
from encoder.identity_encoder import IdentityEncoder
from encoder.avatar_template import AvatarTemplate
from decoder.gaussian_decoder import GaussianDecoder


class Trainer(L.LightningModule):
    """
    Minimal training scaffold that demonstrates wiring:
    - Loads batches from a DataLoader
    - Encodes features with NLF backbone adapter
    - Predict Gaussian parameters with AvatarGaussianEstimator
    - Identity encoding with IdentityEncoder
    """

    def __init__(
        self,
        backbone_adapter: NLFBackboneAdapter,
        identity_encoder: IdentityEncoder,
        decoder: GaussianDecoder,
    ):
        super().__init__()
        # Keep the AvatarTemplate instance (it loads its internal avatar on init)
        self.template = AvatarTemplate()
        self.backbone = backbone_adapter
        self.avatar_estimator = AvatarGaussianEstimator(self.template)
        self.identity_encoder = identity_encoder
        self.decoder = decoder
        # TODO[run-pipeline]: Add args/config to control optimizer, lr, loss weights, renderer, etc.

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        # Be tolerant to different batch formats
        if isinstance(batch, (list, tuple)):
            image = batch[0]
        elif isinstance(batch, dict) and "image" in batch:
            image = batch["image"]
        else:
            # fallback
            image = batch

        # Ensure BCHW order
        if image.dim() == 3:
            if image.shape[0] == 3:
                image = image.unsqueeze(0)
            elif image.shape[-1] == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
        elif image.dim() == 4 and image.shape[1] != 3 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()

        B = image.shape[0]
        H = image.shape[-2]
        W = image.shape[-1]
        feats, preds = self.backbone.detect_with_features(image, use_half=True)

        B_feats, C_local, Hf, Wf = feats.shape
        assert B == B_feats, "Batch size mismatch between image and features"
        N = int(self.template.total_gaussians_num)

        z_id = self.identity_encoder(feature_map=feats, preds=preds, img_shape=(H, W))

        local_feats = self.avatar_estimator.feature_sample(
            feats, preds
        )  # (B, N, C_local)

        coord3d = self.avatar_estimator.compute_gaussian_coord3d(
            feats, preds
        )  # (B, N, 3)

        """
        Decode:
        gaussian_params: Gaussian Params(B, N, C_params)
        """

        z_expanded = z_id.unsqueeze(1).expand(-1, N, -1)  # (B, D) -> (B, N, D)
        combined_feats = torch.cat(
            [z_expanded, local_feats, coord3d], dim=-1
        )  # (B, N, D + C_local + 3)

        gaussian_params = self.decoder(combined_feats)

        # Reconstruct/render TODO handled elsewhere: scales/rots/alpha parameterization and losses to be added.

        loss = gaussian_params.sum() * 0.0
        return loss

    def configure_optimizers(self):
        # TODO[run-pipeline]: Expose LR and optimizer choice via config; add scheduler if needed.
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    # Provide dataloader from an external DataModule or Trainer.fit call
