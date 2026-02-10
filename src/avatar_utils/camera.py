import os
import json
import trimesh
import torch
from typing import List, Optional, Sequence, Union
from pathlib import Path
from avatar_utils.config import get_config


def load_normalization(identity: str, processed_root: str = None) -> tuple[torch.Tensor, float]:
    """Load per-identity normalization (center, radius) from processed/<id>/norm.json.
    
    Args:
        identity: Subject identifier (folder name).
        processed_root: Path to processed data root (default from config).
    
    Returns:
        center: (3,) tensor with original bbox center.
        radius: float with original bbox radius.
    """
    if processed_root is None:
        processed_root = get_config().get("data", {}).get("root", "processed")
    norm_path = Path(processed_root) / identity / "norm.json"
    try:
        with open(norm_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        center = torch.tensor(payload["center"], dtype=torch.float32)
        radius = float(payload["radius"])
        return center, radius
    except Exception as e:
        # Fallback: return identity transform (no normalization)
        print(f"Warning: could not load normalization for {identity}: {e}")
        return torch.zeros(3, dtype=torch.float32), 1.0


def apply_normalization(xyz: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """Apply the same normalization to world-space coordinates as used in preprocessing.
    
    Args:
        xyz: (N, 3) or (B, N, 3) tensor of 3D positions.
        center: (3,) tensor with bbox center to subtract.
        radius: scalar bbox radius for scaling.
    
    Returns:
        xyz_norm: normalized coordinates (same shape as input).
    """
    scale = 1.0 / (radius + 1e-8)
    center = center.to(xyz.device)
    return (xyz - center) * scale


def load_camera_mapping(
    view_name: Union[str, Sequence[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load camera matrices from project-local cache with optional batching.

    Accepts a single view name or a list/sequence of view names. Returns
    batched tensors for viewmats (B,4,4) and Ks (B,3,3). If a cached JSON file
    is missing or unreadable, falls back to computed values for that view.

    Expects JSON files under project-root/data/THuman_cameras named
    thuman_<view>.json with keys: K (3x3), viewmat (4x4), and image_size.
    """
    # Resolve project root as two levels up from this file (src/...)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cache_dir = os.path.join(project_root, "data", "THuman_cameras")
    
    # Define device
    device = torch.device(get_config().get("sys", {}).get("device", "cpu"))

    def _load_one(vname: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = os.path.join(cache_dir, f"thuman_{vname}.json")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            K = torch.tensor(payload["K"], dtype=torch.float32)
            viewmat = torch.tensor(payload["viewmat"], dtype=torch.float32)
            # If JSON stores camera-to-world, convert to world-to-camera
            t = payload.get("type")
            if isinstance(t, str) and t.lower() == "c2w":
                try:
                    viewmat = torch.linalg.inv(viewmat)
                except Exception:
                    pass
            # Adjust intrinsics if current image size differs from cached
            try:
                W0, H0 = payload.get("image_size", [None, None])
                W1, H1 = get_config().get("data", {}).get("image_size", (W0, H0))
                if W0 and H0 and W1 and H1 and (W0 != W1 or H0 != H1):
                    sx = float(W1) / float(W0)
                    sy = float(H1) / float(H0)
                    # fx, fy scale with sy (derived from vertical FOV), cx scales with sx, cy with sy
                    K[0, 0] = K[0, 0] * sy
                    K[1, 1] = K[1, 1] * sy
                    K[0, 2] = K[0, 2] * sx
                    K[1, 2] = K[1, 2] * sy
            except Exception:
                pass
            return viewmat, K
        except Exception:
            # Fallback to on-the-fly computation when cache is missing
            vm, k = camera_mapping(vname)
            # camera_mapping already returns batched (1,...), squeeze for stacking
            return vm.squeeze(0), k.squeeze(0)

    if isinstance(view_name, str):
        vm, k = _load_one(view_name)
        return vm.unsqueeze(0).to(device), k.unsqueeze(0).to(device)
    else:
        vms = []
        ks = []
        for v in view_name:
            vm, k = _load_one(str(v))
            vms.append(vm)
            ks.append(k)
        return torch.stack(vms, dim=0).to(device), torch.stack(ks, dim=0).to(device)


def camera_mapping(view_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get camera intrinsics and extrinsics for a given view name.

    Args:
        view_name: Name of the view (e.g., 'front', 'back', 'left', 'right').

    Returns:
        viewmats: Tensor of shape (B, 4, 4) representing the batched camera extrinsic matrix.
        Ks: Tensor of shape (B, 3, 3) representing the batched camera intrinsic matrix.
    """
    # Match preprocess_thuman.py: Perspective camera with yfov=45 degrees and
    # camera pose built via look_at(center + direction * distance, center, up).
    # We assume a canonical scene with center at origin and a fixed distance
    # (since radius is unknown at render time here). This yields consistent
    # camera placement for each named view.

    # Image size used to derive intrinsics (principal point & focal length)
    width, height = get_config().get("data", {}).get("image_size", (1024, 1024))
    W, H = int(width), int(height)
    yfov_deg = 45.0
    yfov_rad = torch.tensor(yfov_deg * 3.141592653589793 / 180.0, dtype=torch.float32)
    # Focal length from vertical FOV: fy = H / (2 * tan(yfov/2)); fx = fy (square pixels)
    fy = H / (2.0 * torch.tan(yfov_rad / 2.0))
    fx = fy
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    K = torch.tensor(
        [[fx.item(), 0.0, cx], [0.0, fy.item(), cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).unsqueeze(
        0
    )  # (1,3,3)

    # Map view name to direction vector (same as VIEWPOINTS in preprocess)
    directions = {
        "front": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        "back": torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        "left": torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32),
        "right": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    }
    if view_name not in directions:
        raise ValueError(f"Unsupported view_name: {view_name}")

    center = torch.zeros(3, dtype=torch.float32)
    direction = directions[view_name]
    # Use a canonical distance matching preprocess_thuman's pattern (radius * 2.5).
    # Without radius, pick distance=2.5.
    distance = 2.5
    eye = center + direction * distance

    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    # If up is parallel to direction, use Z-up
    if torch.allclose(torch.cross(up, direction), torch.zeros(3, dtype=torch.float32)):
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    # Build camera-to-world pose (match look_at in preprocess)
    z = eye - center
    z = z / (torch.norm(z) + 1e-8)
    x = torch.cross(up, z)
    x = x / (torch.norm(x) + 1e-8)
    y = torch.cross(z, x)

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye

    # Extrinsics expected by rasterizer are usually world-to-camera: inverse of c2w
    w2c = torch.linalg.inv(c2w).unsqueeze(0)  # (1,4,4)
    return w2c, K


def intrinsic_matrix_from_field_of_view(
    fov_degrees: float, imshape: List[int], device: Optional[torch.device] = None
):
    imshape = torch.tensor(imshape, dtype=torch.float32, device=device)
    fov_radians = fov_degrees * torch.tensor(
        torch.pi / 180, dtype=torch.float32, device=device
    )
    larger_side = torch.max(imshape)
    focal_length = larger_side / (torch.tan(fov_radians / 2) * 2)
    _0 = torch.tensor(0, dtype=torch.float32, device=device)
    _1 = torch.tensor(1, dtype=torch.float32, device=device)

    # print(torch.stack([focal_length, _0, imshape[1] / 2], dim=-1))
    return (
        torch.stack(
            [
                focal_length,
                _0,
                (imshape[1] - 1) / 2,
                _0,
                focal_length,
                (imshape[0] - 1) / 2,
                _0,
                _0,
                _1,
            ],
            dim=-1,
        )
        .unflatten(-1, (3, 3))
        .unsqueeze(0)
    )
