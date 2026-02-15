import os
import torch
from gsplat import rasterization
from avatar_utils.config import get_config
from avatar_utils.camera import load_camera_mapping
from typing import Sequence, Union


def _viewmat_opengl_to_gsplat(viewmats: torch.Tensor) -> torch.Tensor:
    """Convert OpenGL-style w2c (camera looks along -Z) to gsplat (camera looks along +Z).
    gsplat expects z_cam positive into the scene; our preprocess uses OpenGL -Z into scene.
    Flip the third row of w2c so transformed points get z_cam negated."""
    out = viewmats.clone()
    out[..., 2, :] = -out[..., 2, :]
    return out


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

        # Always use single shared 3D in world space so all views see the same avatar.
        # Backbone predicts per-view 3D in camera space; we use front view and convert to world.
        if gaussian_3d.dim() == 3 and gaussian_3d.shape[0] == num_views and gaussian_3d.shape[1] == N:
            # Caller passed (V, N, 3); use first view only — trainer will pass world-space (N, 3)
            means = gaussian_3d[0]
        else:
            means = gaussian_3d

        viewmats, Ks = load_camera_mapping(view_name)
        viewmats = viewmats.to(means.device).contiguous()
        Ks = Ks.to(means.device).contiguous()
        viewmats = _viewmat_opengl_to_gsplat(viewmats)

        if backgrounds is None:
            backgrounds = torch.ones(3, device=means.device)
        else:
            backgrounds = backgrounds.to(means.device)
        rendered_imgs, _, _ = rasterization(
            means=means,
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
