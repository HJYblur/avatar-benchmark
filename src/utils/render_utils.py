import os
from typing import Tuple

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


def renderer_setup(
    device, cam_dist=2.7, cam_elev=30.0, cam_azim=0.0, camera_fov=30.0, image_size=512
):
    R_cam, T_cam = look_at_view_transform(
        dist=cam_dist, elev=cam_elev, azim=cam_azim, device=device
    )
    cameras = FoVPerspectiveCameras(device=device, R=R_cam, T=T_cam, fov=camera_fov)
    lights = PointLights(device=device, location=T_cam)
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
