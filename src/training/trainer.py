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
from avatar_utils.config import get_config


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
        self.debug = bool(get_config().get("sys", {}).get("debug", False))
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
        # Extract and normalize images + detector input (now: img_float and img_uint8)
        img_float, img_uint8, (B, H, W) = self.process_input(batch)

        # --- Debug: load tmp results and save a sample image ---
        if self.debug:
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
                    image_feature=img_float, frame_batch=img_uint8, use_half=True
                )
                torch.save(feats, feats_path)
                torch.save(preds, preds_path)
                print(
                    f"[trainer] Saved backbone features to {feats_path} and preds to {preds_path}"
                )

            try:
                # Save the first sample image to disk for visual inspection.
                # Import locally to avoid top-level dependency changes.
                from torchvision.utils import save_image

                sample_img = img_float[0].detach().cpu()
                # Clamp in case image values are slightly out of [0,1]
                save_image(sample_img.clamp(0.0, 1.0), "debug_sample.png")
                print("[trainer] Saved input sample to debug_sample.png")
            except Exception as exc:  # pragma: no cover - debugging helper
                print(f"[trainer] Unable to save sample image: {exc}")
        # --- end debug ---
        else:
            feats, preds = self.backbone.detect_with_features(
                image_feature=img_float, frame_batch=img_uint8, use_half=True
            )

        """
        Encode:
        z_id: Identity Latent Vector (B, D)
        local_feats: Local Features sampled at Gaussian centers (B, N, C_local)
        coord3d: Gaussian 3D Coordinates (B, N, 3)
        """
        B_feats, C_local, Hf, Wf = feats.shape
        assert B == B_feats, "Batch size mismatch between image and features"
        N = int(self.template.total_gaussians_num)

        z_id = self.identity_encoder(feature_map=feats)  # (1, D)
        print(f"[trainer] Identity latent vector z_id shape: {z_id.shape}")

        # [Structure]: Ignore coord3d as encoder output for now
        local_feats, view_weights, _ = (
            self.avatar_estimator.feature_sample_with_visibility(
                feats, preds, img_shape=(H, W)
            )
        )  # (B, N, C_local), (B, N)

        if local_feats.shape[0] > 1:
            weight_sum = view_weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
            local_feats = (local_feats * view_weights.unsqueeze(-1)).sum(
                dim=0, keepdim=True
            ) / weight_sum.unsqueeze(
                -1
            )  # (1, N, C_local)

        # Free large intermediates early to reduce peak VRAM before decoding
        try:
            del feats
            del preds
            del img_float
            del img_uint8
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        """
        Decode:
        gaussian_params: Fused gaussian Params(N, C_params)
        """

        gaussian_params = self.decoder(local_feats, z_id)

        # Free combined inputs post-decoding
        try:
            del local_feats
            del z_id
            del coord3d
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

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

    def process_input(self, batch):
        """Extract tensors from the dataset batch and normalize shape/device.

        Returns
        -------
        img_float : torch.Tensor
            Float image tensor used for feature extraction, shape (B,C,H,W) on self.device.
        img_uint8 : torch.Tensor
            The original uint8 image used by the detector, shape (B,C,H,W) on self.device.
        (B, H, W) : tuple[int,int,int]
            Spatial dims extracted from `img_float`.
        """
        assert (
            "images_float" in batch and "images_uint8" in batch
        ), "Batch missing 'images_float' or 'images_uint8' key"

        img_float = batch["images_float"]
        # If dataset wrapped a singleton batch dim, unwrap it
        if img_float.ndim == 5 and img_float.shape[0] == 1:
            img_float = img_float[0]

        # Optional uint8 input for detectors
        img_uint8 = batch["images_uint8"]
        if img_uint8.ndim == 5 and img_uint8.shape[0] == 1:
            img_uint8 = img_uint8[0]

        # Move tensors to module device
        img_float = img_float.to(self.device)
        img_uint8 = img_uint8.to(self.device)

        B, _, H, W = img_float.shape

        return img_float, img_uint8, (B, H, W)

    def configure_optimizers(self):
        # TODO[run-pipeline]: Expose LR and optimizer choice via config; add scheduler if needed.
        lr = 1e-4
        if getattr(self, "train_decoder_only", False):
            # Only update decoder parameters
            optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
