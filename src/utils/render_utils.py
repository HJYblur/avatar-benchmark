import os
from typing import Tuple, Optional

import math
import pickle
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    RasterizationSettings,
    PointLights,
    BlendParams,
    Materials,
    look_at_view_transform,
)


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues (axis-angle) vector to a rotation matrix.

    r: (3,) axis-angle vector. Returns (3,3) rotation matrix.
    This is a pure NumPy implementation; avoids cv2 dependency.
    """
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3, dtype=float)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R


def camera_loader(camera_pkl_path: str) -> dict:
    """Load and normalize camera.pkl contents into a dictionary with keys:
    - 'R' : (3,3) rotation matrix (numpy)
    - 'T' : (3,) translation vector (numpy)
    - 'f' : (2,) focal lengths (fx, fy)
    - 'c' : (2,) principal point (cx, cy)
    - 'width', 'height' : ints
    - 'raw' : original loaded object

    The loader tries common key names seen in PeopleSnapshot-style camera.pkl files.
    """
    with open(camera_pkl_path, "rb") as f:
        cam = pickle.load(f, encoding="latin1")

    md = {"raw": cam}

    # intrinsics
    if "camera_f" in cam:
        md["f"] = np.asarray(cam["camera_f"]).astype(float)
    elif "camera_focal" in cam:
        md["f"] = np.asarray(cam["camera_focal"]).astype(float)

    if "camera_c" in cam:
        md["c"] = np.asarray(cam["camera_c"]).astype(float)

    if "width" in cam:
        try:
            md["width"] = int(cam["width"])
        except Exception:
            md["width"] = None
    if "height" in cam:
        try:
            md["height"] = int(cam["height"])
        except Exception:
            md["height"] = None

    # translation
    for tkey in ("camera_t", "camera_T", "T", "trans", "translation"):
        if tkey in cam:
            md["T"] = np.asarray(cam[tkey]).astype(float)
            break

    # rotation: either provided as a rotation matrix or an axis-angle (Rodrigues) vector
    if "R" in cam and np.asarray(cam["R"]).shape == (3, 3):
        md["R"] = np.asarray(cam["R"]).astype(float)
    else:
        for rkey in ("camera_rt", "camera_r", "rt", "r", "rotation"):
            if rkey in cam:
                rvec = np.asarray(cam[rkey]).astype(float)
                # if shape (3,), treat as Rodrigues/axis-angle
                if rvec.shape == (3,):
                    md["R"] = _rodrigues(rvec)
                elif rvec.shape == (3, 3):
                    md["R"] = rvec
                break

    return md


def renderer_setup(
    device,
    cam_dist=2.7,
    cam_elev=30.0,
    cam_azim=0.0,
    camera_fov=30.0,
    image_size=512,
    R_cam: Optional[np.ndarray] = None,
    T_cam: Optional[np.ndarray] = None,
    camera_metadata: Optional[dict] = None,
):
    """Create renderer and camera. If camera_metadata is provided it takes precedence.

    camera_metadata may contain keys 'R','T','f','c','width','height'. If 'f' and
    'height' are available we compute vertical fov from fy and image height.
    """
    # If camera metadata supplied, extract R/T and intrinsics
    if camera_metadata is not None:
        if "R" in camera_metadata and camera_metadata["R"] is not None:
            R_cam = camera_metadata["R"]
        if "T" in camera_metadata and camera_metadata["T"] is not None:
            T_cam = camera_metadata["T"]
        # compute fov if focal and height provided
        if "f" in camera_metadata and "height" in camera_metadata:
            f = np.asarray(camera_metadata["f"]).astype(float)
            height = camera_metadata["height"]
            # assume f = [fx, fy] or scalar
            fy = (
                float(f[1]) if getattr(f, "shape", None) and len(f) > 1 else float(f[0])
            )
            # vertical fov in degrees
            camera_fov = float(2.0 * math.degrees(math.atan((height / 2.0) / fy)))

    # Convert provided R_cam/T_cam numpy arrays to torch tensors on device if present
    if R_cam is not None and T_cam is not None:
        if isinstance(R_cam, np.ndarray):
            R_t = torch.from_numpy(R_cam).to(device)
        else:
            R_t = R_cam.to(device)
        if isinstance(T_cam, np.ndarray):
            T_t = torch.from_numpy(T_cam).to(device)
        else:
            T_t = T_cam.to(device)
        # Ensure batch dims
        if R_t.ndim == 2:
            R_t = R_t.unsqueeze(0)
        if T_t.ndim == 1:
            T_t = T_t.unsqueeze(0)
        R_use, T_use = R_t, T_t
    else:
        R_use, T_use = look_at_view_transform(
            dist=cam_dist, elev=cam_elev, azim=cam_azim, device=device
        )

    cameras = FoVPerspectiveCameras(device=device, R=R_use, T=T_use, fov=camera_fov)
    lights = PointLights(device=device, location=T_use)
    materials = Materials(
        device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0
    )

    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )
    blend_params = BlendParams(background_color=(0, 0, 0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            cameras=cameras,
            lights=lights,
            device=device,
            blend_params=blend_params,
            materials=materials,
        ),
    )
    return renderer, cameras


def save_image_with_axes(
    image_np, cameras, device, image_size, axis_scale: float, output_path: str
):
    """Overlay simple XYZ axes on the rendered image and save.

    image_np: numpy array HxWx3 float in [0,1]
    cameras: PyTorch3D camera used to render (so we can project 3D points)
    """
    image_pil = Image.fromarray((image_np * 255).astype("uint8"))
    # axes points in world coordinates
    axes_pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [axis_scale, 0.0, 0.0],
            [0.0, axis_scale, 0.0],
            [0.0, 0.0, axis_scale],
        ],
        device=device,
    ).unsqueeze(0)
    # project to screen
    # PyTorch3D expects image_size to be a tensor of shape (B, 2) or similar.
    # Convert scalar or tuple to a (1,2) tensor on the correct device.
    if isinstance(image_size, int):
        image_size_t = torch.tensor([[image_size, image_size]], device=device)
    elif isinstance(image_size, (tuple, list)):
        image_size_t = torch.tensor([list(image_size)], device=device)
    else:
        # assume already tensor-like; move to device
        image_size_t = torch.as_tensor(image_size, device=device)

    screen_pts = (
        cameras.transform_points_screen(axes_pts, image_size=image_size_t)[0]
        .cpu()
        .numpy()
    )
    coords = [(float(x), float(y)) for x, y in screen_pts[:, :2]]
    draw = ImageDraw.Draw(image_pil)
    origin = coords[0]
    draw.line([origin, coords[1]], fill=(255, 0, 0), width=3)
    draw.line([origin, coords[2]], fill=(0, 255, 0), width=3)
    draw.line([origin, coords[3]], fill=(0, 0, 255), width=3)
    image_pil.save(output_path)


def save_image(image_np, output_path: str):
    """Save an image numpy array (HxWx3 float in [0,1]) to a PNG without overlays."""
    image_pil = Image.fromarray((image_np * 255).astype("uint8"))
    image_pil.save(output_path)
