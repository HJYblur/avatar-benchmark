import inspect
from typing import Any, Dict

import torch


class NLFBackboneAdapter:
    """Thin wrapper around NLF's crop_model.backbone with feature/prediction split.

    The adapter exposes helpers so callers can access intermediate outputs
    without modifying the NLF package:

    - ``extract_feature_map``: run the backbone only and return the feature map.
    - ``detect_with_features``: convenience wrapper that mirrors the old
      ``detect_smpl_batched`` usage but also returns the intermediate feature map.
    """

    def __init__(self, nlf_model):
        if not hasattr(nlf_model, "crop_model") or not hasattr(
            nlf_model.crop_model, "backbone"
        ):
            raise AttributeError("nlf_model must expose crop_model.backbone")
        if not hasattr(nlf_model, "detector"):
            raise AttributeError("nlf_model must expose detector")
        self.nlf_model = nlf_model

    def __call__(self, image: torch.Tensor, use_half: bool = True) -> torch.Tensor:
        return self.extract_feature_map(image=image, use_half=use_half)

    def extract_feature_map(
        self, image: torch.Tensor, use_half: bool = True, use_heatmap_head: bool = True
    ) -> torch.Tensor:
        """Return the backbone feature map only."""

        x = image.half() if use_half else image
        x = self.nlf_model.crop_model.backbone(x)
        if use_heatmap_head:
            x = self.nlf_model.crop_model.heatmap_head.layer(x)
        return x

    def detect_with_features(
        self,
        frame_batch: torch.Tensor,
        model_name: str = "smplx",
        use_half: bool = True,
        use_heatmap_head: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run detection while exposing the intermediate feature map.

        Returns a dictionary containing both the feature map and the NLF
        detection outputs (including 2D/3D poses as provided by NLF).
        """

        feature_map = self.extract_feature_map(
            image=frame_batch, use_half=use_half, use_heatmap_head=use_heatmap_head
        )
        preds = self.nlf_model.detect_smpl_batched(
            images=frame_batch, model_name=model_name, **kwargs
        )
        return feature_map, preds
