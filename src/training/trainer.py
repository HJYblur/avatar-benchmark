from typing import Any, Dict, Optional
import os
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
        train_decoder_only: bool = True,
    ):
        super().__init__()
        self.template = AvatarTemplate()
        self.backbone = backbone_adapter
        self.avatar_estimator = AvatarGaussianEstimator(self.template)
        self.identity_encoder = identity_encoder
        self.decoder = decoder
        self.train_decoder_only = train_decoder_only

        # If True, freeze all parameters except the decoder's so only decoder gets updated.
        if self.train_decoder_only:
            self.identity_encoder.eval()
        # TODO[run-pipeline]: Add args/config to control optimizer, lr, loss weights, renderer, etc.

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        # Extract images from new dataset keys: images_float / images_uint8
        if isinstance(batch, dict) and "images_float" in batch:
            image = batch["images_float"]
            # Default collate with batch_size=1 yields shape [1,V,C,H,W]
            if image.ndim == 5 and image.shape[0] == 1:
                image = image[0]  # [V,C,H,W]
        elif isinstance(batch, (list, tuple)):
            image = batch[0]
        elif isinstance(batch, dict) and "image" in batch:
            image = batch["image"]
        else:
            image = batch

        # Ensure BCHW order
        if image.dim() == 3:
            if image.shape[0] == 3:
                image = image.unsqueeze(0)
            elif image.shape[-1] == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
        elif image.dim() == 4 and image.shape[1] != 3 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()

        # Move image to the trainer/device to ensure later ops use correct device
        try:
            image = image.to(self.device)
        except Exception:
            # If image isn't a tensor (fallback), ignore
            pass

        B = image.shape[0]
        H = image.shape[-2]
        W = image.shape[-1]

        # Use image feature (float32/float16) for feature extraction
        # Use the original int image: `detect_input` for prediction ops.
        detect_input = image
        if isinstance(batch, dict) and "images_uint8" in batch:
            detect_input = batch["images_uint8"]
            if detect_input.ndim == 5 and detect_input.shape[0] == 1:
                detect_input = detect_input[0]  # [V,C,H,W]

        feats_path = "debug_backbone_features.pt"
        preds_path = "debug_backbone_preds.pt"
        if os.path.exists(feats_path) and os.path.exists(preds_path):
            # Try to load precomputed features and preds for faster debugging
            feats = torch.load(feats_path, map_location=self.device)
            preds = torch.load(preds_path, map_location=self.device)
            print(
                f"[trainer] Loaded backbone features from {feats_path} and preds from {preds_path}"
            )
        else:
            feats, preds = self.backbone.detect_with_features(
                image_feature=image, frame_batch=detect_input, use_half=True
            )
            torch.save(feats, feats_path)
            torch.save(preds, preds_path)
            print(
                f"[trainer] Saved backbone features to {feats_path} and preds to {preds_path}"
            )

        # feats, preds = self.backbone.detect_with_features(
        #     image_feature=image, frame_batch=detect_input, use_half=True
        # )

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
            template=self.template.load_avatar_template(mode="test"),
            output_path="debug_reconstructed_avatar.ply",
        )

        # TODO[run-pipeline]: Add render and implement proper loss computation

        # Use a tiny L2 regularization loss on the decoder parameters so the
        # returned loss has a gradient graph and optimizer can step.
        # Ensure there is at least one trainable param in decoder when
        # training only the decoder.
        has_trainable = any(p.requires_grad for p in self.decoder.parameters())
        if getattr(self, "train_decoder_only", False) and not has_trainable:
            raise RuntimeError(
                "train_decoder_only=True but decoder has no trainable parameters"
            )

        # Ensure reg_loss is on the module/device
        reg_loss = torch.tensor(0.0, device=self.device)
        for p in self.decoder.parameters():
            # accumulate squared L2 norm
            reg_loss = reg_loss + p.pow(2).sum()
        # scale down the regularizer so it doesn't dominate when a real loss is used
        loss = reg_loss * 1e-6
        return loss

    def configure_optimizers(self):
        # TODO[run-pipeline]: Expose LR and optimizer choice via config; add scheduler if needed.
        lr = 1e-4
        if getattr(self, "train_decoder_only", False):
            # Only update decoder parameters
            optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
