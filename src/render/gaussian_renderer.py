import os
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
        save_folder_path: str = None,
        render_mode: str = "RGB",
        backgrounds: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Render the Gaussian splat representation into 2D images.

        Args:
            gaussian_3d: Tensor of shape (N, 3) representing 3D Gaussian centers.
            gaussian_params: Dictionary containing Gaussian parameters such as scales, rotations, alphas, etc.
            view_name: A single view name (e.g., 'front') or a list of view names
                (e.g., ['front', 'left']).

        Returns:
            Rendered images as a tensor of shape (B, H, W, 3).
        """
        # Load precomputed camera matrices (batched if a list is provided)
        viewmats, Ks = load_camera_mapping(view_name)  # (B, 4, 4), (B, 3, 3)
        viewmats = viewmats.to(gaussian_3d.device).contiguous()
        Ks = Ks.to(gaussian_3d.device).contiguous()
        width, height = get_config().get("data", {}).get("image_size", (1024, 1024))
        num_cameras = viewmats.shape[0]
        # backgrounds = (
        #     torch.ones((num_cameras, 1), device=gaussian_3d.device)
        #     if backgrounds is None
        #     else backgrounds
        # )
        rendered_imgs, rendered_alphas, meta = rasterization(
            means=gaussian_3d,
            quats=gaussian_params["rotation"],
            scales=gaussian_params["scales"],
            opacities=gaussian_params["alpha"],
            colors=gaussian_params.get("sh", None),  # (N, K), usually K = 3
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            render_mode=render_mode,
            backgrounds=backgrounds,
        )
        # rendered_imgs: (B, H, W, 3)
        if save_folder_path is not None:
            from torchvision.io import write_png
            from torchvision.transforms.functional import convert_image_dtype
            from pathlib import Path as _Path

            # Ensure the output directory exists and build a filename
            out_dir = _Path(save_folder_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "debug.png"

            # Save the first rendered image as PNG for inspection
            sample_img = (
                rendered_imgs[0].permute(2, 0, 1).to(torch.device("cpu"))
            )  # (3, H, W)
            img_to_save = convert_image_dtype(sample_img.clamp(0, 1), dtype=torch.uint8)
            write_png(img_to_save, str(out_file))
        return rendered_imgs  # (B, H, W, 3) where B=len(view_name) if list
