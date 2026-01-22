from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
import lightning as L
from encoder.nlf_backbone_adapter import NLFBackboneAdapter
from encoder.gaussian_estimator import AvatarGaussianEstimator
from encoder.identity_encoder import IdentityEncoder
from encoder.avatar_template import AvatarTemplate
from decoder.gaussian_decoder import GaussianDecoder
from avatar_utils.ply_loader import reconstruct_gaussian_avatar_as_ply


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

        # Use image feature (float32/float16) for feature extraction
        # Use the original int image: `detect_input` for prediction ops.
        detect_input = image
        if isinstance(batch, dict) and "image_uint8" in batch:
            detect_input = batch["image_uint8"]

        # feats_path = "debug_backbone_features.pt"
        # preds_path = "debug_backbone_preds.pt"
        # if feats_path is not None and preds_path is not None:
        #     try:
        #         # Try to load precomputed features and preds for faster debugging
        #         feats = torch.load(feats_path, map_location=image.device)
        #         preds = torch.load(preds_path, map_location=image.device)
        #         print(
        #             f"[trainer] Loaded backbone features from {feats_path} and preds from {preds_path}"
        #         )
        #     except Exception as exc:
        #         print(
        #             f"[trainer] Unable to load precomputed features/preds: {exc}. Running backbone inference."
        #         )
        #         feats, preds = self.backbone.detect_with_features(
        #             image_feature=image, frame_batch=detect_input, use_half=True
        #         )
        # else:

        #     torch.save(feats, feats_path)
        #     torch.save(preds, preds_path)
        #     print(
        #         f"[trainer] Saved backbone features to {feats_path} and preds to {preds_path}"
        #     )

        feats, preds = self.backbone.detect_with_features(
            image_feature=image, frame_batch=detect_input, use_half=True
        )

        # --- Debug: save a sample image ---
        try:
            # Save the first sample image to disk for visual inspection.
            # Import locally to avoid top-level dependency changes.
            from torchvision.utils import save_image

            sample_img = image[0].detach().cpu()
            # Clamp in case image values are slightly out of [0,1]
            save_image(sample_img.clamp(0.0, 1.0), "debug_sample.png")
            print("[trainer] Saved input sample to debug_sample.png")
        except Exception as exc:  # pragma: no cover - debugging helper
            print(f"[trainer] Unable to save sample image: {exc}")
        # --- end debug ---

        """
        Encode:
        z_id: Identity Latent Vector (B, D)
        local_feats: Local Features sampled at Gaussian centers (B, N, C_local)
        coord3d: Gaussian 3D Coordinates (B, N, 3)
        """
        B_feats, C_local, Hf, Wf = feats.shape
        assert B == B_feats, "Batch size mismatch between image and features"
        N = int(self.template.total_gaussians_num)

        z_id = self.identity_encoder(feature_map=feats)  # (B, D)
        print(f"[trainer] Identity latent vector z_id shape: {z_id.shape}")

        # Pass original image size so gaussian coord normalization uses image pixels
        local_feats = self.avatar_estimator.feature_sample(
            feats, preds, img_shape=(H, W)
        )  # (B, N, C_local)

        coord3d = self.avatar_estimator.compute_gaussian_coord3d(
            feats, preds
        )  # (B, N, 3)

        """
        Decode:
        gaussian_params: Fused gaussian Params(N, C_params)
        """

        combined_feats = torch.cat(
            [local_feats, coord3d], dim=-1
        )  # (B, N, C_local + 3)

        gaussian_params = self.decoder(combined_feats, z_id)  # Fused output

        # Debug check:
        for k, v in gaussian_params.items():
            print(f"[trainer] Decoded gaussian_params[{k}] shape: {v.shape}")

        # Use gaussian_params and gaussian avatar to generate a .ply file as the reconstruction.
        reconstruct_gaussian_avatar_as_ply(
            gaussian_params=gaussian_params,
            template=self.template,
            output_path="debug_reconstructed_avatar.ply",
        )

        # TODO[run-pipeline]: Add render and implement proper loss computation
        loss = torch.tensor(0.0, device=image.device)
        return loss

    def configure_optimizers(self):
        # TODO[run-pipeline]: Expose LR and optimizer choice via config; add scheduler if needed.
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
