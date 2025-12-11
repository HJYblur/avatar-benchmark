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
        """Return the backbone feature map only.

        If ``use_heatmap_head`` is True and the NLF model exposes a heatmap head,
        apply it as part of feature extraction.

        Outputs:
            feature_map: Tensor of shape (B, C_local, Hf, Wf)
            By default, C_local=512 when using EfficientNetV2.
        """

        x = image.half() if use_half else image
        x = self.nlf_model.crop_model.backbone(x)
        if use_heatmap_head and hasattr(self.nlf_model.crop_model, "heatmap_head"):
            head = self.nlf_model.crop_model.heatmap_head
            # Some NLF variants expose ``heatmap_head.layer``, others are callable modules
            x = getattr(head, "layer", head)(x)
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

        Returns a tuple: (feature_map, preds)
          - feature_map: Tensor of shape (B, C, H, W)
          - preds: Dict with NLF detection outputs.
          E.g.
            Prediction keys: ['boxes', 'pose', 'betas', 'trans', 'vertices3d', 'joints3d', 'vertices2d', 'joints2d', 'vertices3d_nonparam', 'joints3d_nonparam', 'vertices2d_nonparam', 'joints2d_nonparam', 'vertex_uncertainties', 'joint_uncertainties']
            boxes -> torch.Size([Batch_size, num_people, 5])
            pose -> torch.Size([Batch_size, num_people, 165])
            betas -> torch.Size([Batch_size, num_people, 10])
            trans -> torch.Size([Batch_size, num_people, 3])
            vertices3d -> torch.Size([Batch_size, num_people, 10475, 3])
            joints3d -> torch.Size([Batch_size, num_people, 55, 3])
            vertices2d -> torch.Size([Batch_size, num_people, 10475, 2])
            joints2d -> torch.Size([Batch_size, num_people, 55, 2])
            vertices3d_nonparam -> torch.Size([Batch_size, num_people, 1024, 3])
            joints3d_nonparam -> torch.Size([Batch_size, num_people, 55, 3])
            vertices2d_nonparam -> torch.Size([Batch_size, num_people, 1024, 2])
            joints2d_nonparam -> torch.Size([Batch_size, num_people, 55, 2])
            vertex_uncertainties -> torch.Size([Batch_size, num_people, 1024])
            joint_uncertainties -> torch.Size([Batch_size, num_people, 55])
        """

        feature_map = self.extract_feature_map(
            image=frame_batch, use_half=use_half, use_heatmap_head=use_heatmap_head
        )
        preds = self.nlf_model.detect_smpl_batched(
            frame_batch, model_name=model_name, **kwargs
        )
        return feature_map, preds
