import os
import torch
from gsplat import rasterization
from avatar_utils.config import get_config
from avatar_utils.camera import load_camera_mapping
from typing import Sequence, Union


class GsplatRenderer:
    def __init__(self):
        self.sh_degree = get_config().get("decoder", {}).get("sh_degree", 3)

    def render(
        self,
        gaussian_3d: torch.Tensor,
        gaussian_params: dict[str, torch.Tensor],
        view_name: Union[str, Sequence[str]],
        camera_model: str = "pinhole",  # “pinhole”, “ortho”, “fisheye”, and “ftheta”. Default is “pinhole”
        save_folder_path: str = None,
        render_mode: str = "RGB",
        backgrounds: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Render the Gaussian splat representation into 2D images.

        Args:
            gaussian_3d: Tensor of shape (N, 3) for shared 3D, or (B, N, 3) for per-view 3D
                (each view uses its own means; use when backbone predicts in camera space per view).
            gaussian_params: Dictionary containing Gaussian parameters such as scales, rotations, alphas, etc.
            view_name: A single view name (e.g., 'front') or a list of view names
                (e.g., ['front', 'left']).

        Returns:
            Rendered images as a tensor of shape (B, H, W, 3).
        """
        # Process sh because gsplat expects SH coefficients with shape […, N, K, 3],
        # where K is the number of SH coefficients.
        shs = gaussian_params["sh"]  # (N, K), K = (sh_degree + 1)^2 = 16
        N, K = shs.shape
        assert (
            K == (self.sh_degree + 1) ** 2 * 3
        ), f"We expected SH shape (N, {(self.sh_degree + 1) ** 2 * 3}), got {shs.shape}"
        colors = shs.view(N, -1, 3)  # (N, K//3, 3)

        width, height = get_config().get("data", {}).get("image_size", (1024, 1024))

        view_names_list = [view_name] if isinstance(view_name, str) else list(view_name)
        num_views = len(view_names_list)

        # Per-view 3D: gaussian_3d (B, N, 3) — render each view with its own means
        per_view_3d = gaussian_3d.dim() == 3 and gaussian_3d.shape[0] == num_views and gaussian_3d.shape[1] == N

        if per_view_3d:
            rendered_list = []
            for i in range(num_views):
                viewmats, Ks = load_camera_mapping(view_names_list[i])
                viewmats = viewmats.to(gaussian_3d.device).contiguous()
                Ks = Ks.to(gaussian_3d.device).contiguous()
                if backgrounds is None:
                    bg = torch.ones(3, device=gaussian_3d.device)
                else:
                    bg = backgrounds.to(gaussian_3d.device)
                img, _, _ = rasterization(
                    means=gaussian_3d[i],
                    quats=gaussian_params["rotation"],
                    scales=gaussian_params["scales"],
                    opacities=gaussian_params["alpha"],
                    sh_degree=self.sh_degree,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    camera_model=camera_model,
                    width=width,
                    height=height,
                    render_mode=render_mode,
                    backgrounds=bg,
                )
                rendered_list.append(img)
            rendered_imgs = torch.cat(rendered_list, dim=0)
        else:
            # Single 3D (N, 3) shared across all views
            viewmats, Ks = load_camera_mapping(view_name)
            viewmats = viewmats.to(gaussian_3d.device).contiguous()
            Ks = Ks.to(gaussian_3d.device).contiguous()
            if backgrounds is None:
                backgrounds = torch.ones(3, device=gaussian_3d.device)
            else:
                backgrounds = backgrounds.to(gaussian_3d.device)
            rendered_imgs, _, _ = rasterization(
                means=gaussian_3d,
                quats=gaussian_params["rotation"],
                scales=gaussian_params["scales"],
                opacities=gaussian_params["alpha"],
                sh_degree=self.sh_degree,
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                camera_model=camera_model,
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

            # Ensure the output directory exists
            out_dir = _Path(save_folder_path)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save rendered image for each view
            for idx, vname in enumerate(view_names_list):
                out_file = out_dir / f"debug_{vname}.png"
                sample_img = (
                    rendered_imgs[idx].permute(2, 0, 1).to(torch.device("cpu"))
                )  # (3, H, W)
                img_to_save = convert_image_dtype(sample_img.clamp(0, 1), dtype=torch.uint8)
                write_png(img_to_save, str(out_file))
        return rendered_imgs  # (B, H, W, 3) where B=len(view_name) if list
