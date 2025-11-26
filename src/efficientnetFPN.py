import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class EfficientNetV2FPN(nn.Module):
    def __init__(self, out_channels=256):
        """
        Args:
            out_channels: The number of channels for the final feature map.
                          This will be the input size for your Gaussian MLP.
        """
        super().__init__()

        # 1. Load the Backbone
        weights = EfficientNet_V2_S_Weights.DEFAULT
        base_model = efficientnet_v2_s(weights=weights)

        # 2. Identify the nodes for Stride 4, 8, 16, 32
        # For EfficientNetV2-S specifically:
        # features.2 -> Stride 4
        # features.3 -> Stride 8
        # features.5 -> Stride 16
        # features.7 -> Stride 32
        return_nodes = {
            "features.2": "stride_4",
            "features.3": "stride_8",
            "features.5": "stride_16",
            "features.7": "stride_32",
        }

        # Create the extractor
        self.backbone = create_feature_extractor(base_model, return_nodes=return_nodes)

        # 3. Dynamically find the input channels for these layers
        # (This makes the code safe even if you switch to EffNet-M or L)
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        # 4. Create Lateral Layers (1x1 Convs to project to out_channels)
        self.lateral_convs = nn.ModuleDict()
        self.fpn_convs = nn.ModuleDict()

        for key, value in features.items():
            in_channels = value.shape[1]
            # Project to fixed size (e.g. 1280 -> 256)
            self.lateral_convs[key] = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
            # Smoothing layer (3x3) to clean up aliasing after upsampling
            self.fpn_convs[key] = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            )

    def forward(self, x):
        # 1. Extract Multi-scale features
        # Returns a dict: {'stride_4': tensor, 'stride_32': tensor, ...}
        enc_features = self.backbone(x)

        # 2. Project features to same channel width (Lateral connections)
        proj_features = {}
        for key in enc_features:
            proj_features[key] = self.lateral_convs[key](enc_features[key])

        # 3. Build the Pyramid (Top-Down Pathway)
        # We start from the deepest layer (Stride 32) and work up to Stride 4

        # Stride 32 (Deepest)
        p32 = proj_features["stride_32"]

        # Stride 16 = Projected Stride 16 + Upsampled P32
        # Use explicit target size (from the lateral feature) instead of a fixed
        # scale_factor to avoid off-by-one mismatches when input spatial sizes
        # are not divisible by powers of two (odd dimensions).
        target_size_p16 = proj_features["stride_16"].shape[-2:]
        p16 = proj_features["stride_16"] + F.interpolate(
            p32, size=target_size_p16, mode="nearest"
        )

        # Stride 8 = Projected Stride 8 + Upsampled P16
        target_size_p8 = proj_features["stride_8"].shape[-2:]
        p8 = proj_features["stride_8"] + F.interpolate(
            p16, size=target_size_p8, mode="nearest"
        )

        # Stride 4 = Projected Stride 4 + Upsampled P8
        target_size_p4 = proj_features["stride_4"].shape[-2:]
        p4 = proj_features["stride_4"] + F.interpolate(
            p8, size=target_size_p4, mode="nearest"
        )

        # 4. Apply Smoothing (FPN Standard Practice)
        out_map = self.fpn_convs["stride_4"](p4)

        # We usually return the finest map (Stride 4) for sampling
        # But we could return the global vector (p32 pooled) if needed for your Pose Path
        return out_map, p32
