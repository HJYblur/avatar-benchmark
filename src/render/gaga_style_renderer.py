"""
GAGAvatar-style camera and rendering (see github.com/xg-chu/GAGAvatar).

- Canonical cameras: distance from origin, look at origin, explicit focal/size.
- build_camera_matrices_canonical() -> world-to-camera 4x4 + intrinsics 3x3.
- render_gaussian_canonical() -> rasterize with gsplat using these cameras.
"""
import math
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from gsplat import rasterization

from avatar_utils.config import get_config


# View name -> (azimuth_deg, elevation_deg). Y up, front = +Z.
# eye = (d*cos(el)*sin(az), d*sin(el), d*cos(el)*cos(az)); az=0 -> +Z, az=90 -> +X (right), az=-90 -> -X (left).
VIEW_AZIMUTH_ELEVATION = {
    "front": (0.0, 0.0),    # (0, 0, d)
    "back": (180.0, 0.0),   # (0, 0, -d)
    "left": (-90.0, 0.0),   # (-d, 0, 0)
    "right": (90.0, 0.0),   # (d, 0, 0)
}


def build_camera_matrices_canonical(
    view_names: Union[str, Sequence[str]],
    image_width: int,
    image_height: int,
    distance: float = 2.0,
    yfov_deg: float = 45.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build world-to-camera (4x4) and intrinsics K (3x3) for canonical views (GAGAvatar-style).
    All cameras look at the world origin; avatar should be centered at origin.

    Returns:
        viewmats: (B, 4, 4) world-to-camera (OpenGL-style: camera looks along -Z).
        Ks: (B, 3, 3) intrinsics with fx, fy, cx, cy from image size and yfov.
    """
    if isinstance(view_names, str):
        view_names = [view_names]
    view_names = list(view_names)
    B = len(view_names)
    device = device or torch.device("cpu")
    dtype = torch.float32

    yfov_rad = math.radians(yfov_deg)
    fy = image_height / (2.0 * math.tan(yfov_rad / 2.0))
    fx = fy
    cx = (image_width - 1) / 2.0
    cy = (image_height - 1) / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    Ks = K.unsqueeze(0).expand(B, 3, 3).contiguous()

    viewmats_list = []
    for vname in view_names:
        if vname not in VIEW_AZIMUTH_ELEVATION:
            raise ValueError(f"Unknown view: {vname}. Use one of {list(VIEW_AZIMUTH_ELEVATION.keys())}")
        az_deg, el_deg = VIEW_AZIMUTH_ELEVATION[vname]
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        # Eye position: spherical then to Cartesian (Y up, Z forward for front).
        # Front (az=0, el=0) -> (0, 0, distance). Back -> (0, 0, -distance). Left -> (-d,0,0). Right -> (d,0,0).
        x = distance * math.cos(el) * math.sin(az)
        y = distance * math.sin(el)
        z = distance * math.cos(el) * math.cos(az)
        eye = torch.tensor([x, y, z], dtype=dtype, device=device)
        center = torch.zeros(3, dtype=dtype, device=device)
        up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        # c2w: camera-to-world (OpenGL look_at: camera looks along -Z in cam space)
        forward = center - eye
        forward = forward / (torch.norm(forward) + 1e-8)
        right = torch.cross(up, forward)
        right = right / (torch.norm(right) + 1e-8)
        up = torch.cross(forward, right)
        up = up / (torch.norm(up) + 1e-8)
        c2w = torch.eye(4, dtype=dtype, device=device)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # OpenGL: camera looks along -Z
        c2w[:3, 3] = eye
        w2c = torch.linalg.inv(c2w)
        viewmats_list.append(w2c)
    viewmats = torch.stack(viewmats_list, dim=0).contiguous()
    return viewmats, Ks


def viewmat_opengl_to_gsplat(viewmats: torch.Tensor) -> torch.Tensor:
    """Convert OpenGL w2c (camera looks along -Z) to gsplat (camera looks along +Z)."""
    out = viewmats.clone()
    out[..., 2, :] = -out[..., 2, :]
    return out


def render_gaussian_canonical(
    means: torch.Tensor,
    gaussian_params: dict,
    view_names: Union[str, Sequence[str]],
    sh_degree: int = 3,
    image_size: Optional[tuple] = None,
    distance: float = 2.0,
    yfov_deg: float = 45.0,
    save_folder_path: Optional[Union[str, Path]] = None,
    backgrounds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Render Gaussians with GAGAvatar-style canonical cameras (all views look at origin).
    Uses gsplat; means should be in world space with avatar centered at origin.

    Args:
        means: (N, 3) world-space positions.
        gaussian_params: dict with "rotation" (N,4), "scales" (N,3), "alpha" (N,), "sh" (N, K*3).
        view_names: e.g. ["front", "back", "left", "right"].
        sh_degree: SH degree for gsplat.
        image_size: (width, height). Default from config.
        distance: Camera distance from origin.
        yfov_deg: Vertical FOV in degrees.
        save_folder_path: If set, save debug_<view>.png per view.
        backgrounds: (3,) or None for white.

    Returns:
        rendered_imgs: (B, H, W, 3).
    """
    if image_size is None:
        image_size = get_config().get("data", {}).get("image_size", (1024, 1024))
    width, height = int(image_size[0]), int(image_size[1])
    view_names_list = [view_names] if isinstance(view_names, str) else list(view_names)
    N = means.shape[0]
    device = means.device

    shs = gaussian_params["sh"]
    K_sh = (sh_degree + 1) ** 2 * 3
    assert shs.shape == (N, K_sh), f"sh shape {shs.shape} vs (N={N}, K={K_sh})"
    colors = shs.view(N, -1, 3)

    viewmats, Ks = build_camera_matrices_canonical(
        view_names_list,
        image_width=width,
        image_height=height,
        distance=distance,
        yfov_deg=yfov_deg,
        device=device,
    )
    viewmats = viewmat_opengl_to_gsplat(viewmats)
    if backgrounds is None:
        backgrounds = torch.ones(3, device=device, dtype=means.dtype)
    else:
        backgrounds = backgrounds.to(device)

    parts = []
    for i in range(len(view_names_list)):
        out, _, _ = rasterization(
            means=means,
            quats=gaussian_params["rotation"],
            scales=gaussian_params["scales"],
            opacities=gaussian_params["alpha"],
            sh_degree=sh_degree,
            colors=colors,
            viewmats=viewmats[i : i + 1],
            Ks=Ks[i : i + 1],
            camera_model="pinhole",
            width=width,
            height=height,
            render_mode="RGB",
            backgrounds=backgrounds,
        )
        parts.append(out)
    rendered_imgs = torch.cat(parts, dim=0)

    if save_folder_path is not None:
        from torchvision.io import write_png
        from torchvision.transforms.functional import convert_image_dtype
        out_dir = Path(save_folder_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, vname in enumerate(view_names_list):
            out_file = out_dir / f"debug_{vname}.png"
            img = (
                rendered_imgs[idx].permute(2, 0, 1).to(torch.device("cpu"))
            )
            img_u8 = convert_image_dtype(img.clamp(0.0, 1.0), dtype=torch.uint8)
            write_png(img_u8, str(out_file))

    return rendered_imgs
