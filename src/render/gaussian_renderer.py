import torch
from gsplat import rasterization
from avatar_utils.config import get_config
from avatar_utils.camera import load_camera_mapping
from typing import Sequence, Union


class GsplatRenderer:
    def __init__(self):
        pass

    def render(
        self,
        gaussian_3d: torch.Tensor,
        gaussian_params: dict[str, torch.Tensor],
        view_name: Union[str, Sequence[str]],
    ) -> torch.Tensor:
        """
        Render the Gaussian splat representation into 2D images.

        Args:
            gaussian_3d: Tensor of shape (N, 3) representing 3D Gaussian centers.
            gaussian_params: Dictionary containing Gaussian parameters such as scales, rotations, alphas, etc.
            view_name: A single view name (e.g., 'front') or a list of view names
                (e.g., ['front', 'left']).

        Returns:
            Rendered images as a tensor of shape (1, 3, H, W).
        """
        # Load precomputed camera matrices (batched if a list is provided)
        viewmats, Ks = load_camera_mapping(view_name)
        width, height = get_config().get("data", {}).get("image_size", (1024, 1024))
        rendered_imgs, rendered_alphas, meta = rasterization(
            means=gaussian_3d,
            quats=gaussian_params["rotation"],
            scales=gaussian_params["scales"],
            opacities=gaussian_params["alpha"],
            colors=gaussian_params.get("sh", None),  # (N, K), usually K = 3
            viewmats=viewmats,
            Ks=Ks,
            image_width=width,
            image_height=height,
        )
        return rendered_imgs  # (B, 3, H, W) where B=len(view_name) if list
