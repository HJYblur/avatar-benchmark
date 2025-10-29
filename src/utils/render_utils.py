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


def file_check(
    smpl_folder: str,
    pose_path: str,
    shape_path: str,
    texture_path: str,
    smpl_uv_template: str,
) -> bool:
    if not os.path.exists(os.path.join(smpl_folder, "smpl")):
        print(f"Error: SMPL model directory not found at: {smpl_folder}")
        return False
    if not os.path.exists(pose_path):
        print(f"Error: PeopleSnapshot pose file not found at: {pose_path}")
        return False
    if not os.path.exists(shape_path):
        print(f"Error: PeopleSnapshot shape file not found at: {shape_path}")
        return False
    if not os.path.exists(texture_path):
        print(f"Error: Texture file not found at: {texture_path}")
        return False
    if not os.path.exists(smpl_uv_template):
        print(f"Error: SMPL UV template file not found at: {smpl_uv_template}")
        return False
    return True


def load_sequence(shape_path: str, pose_path: str, load_shape_fn, load_pose_fn):
    """Load betas, poses and translations using provided loader functions.

    The caller typically passes src.utils.data_loader.load_smpl_shape and
    load_smpl_pose.
    """
    shape_data = load_shape_fn(shape_path)
    pose_data = load_pose_fn(pose_path)

    betas = shape_data["betas"].squeeze()
    if betas.shape[0] > 10:
        betas = betas[:10]

    poses = pose_data["pose"]
    trans = pose_data["trans"]
    return betas, poses, trans


def uv_loader(
    device: torch.device, smpl_uv_template: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, faces_data, aux_data = load_obj(smpl_uv_template, device=device)
    verts_uvs = aux_data.verts_uvs
    faces_uvs = faces_data.textures_idx
    return verts_uvs, faces_uvs


def texture_loader(device: torch.device, texture_path: str, verts_uvs, faces_uvs):
    from PIL import Image
    import numpy as np
    import torch

    texture_image_pil = Image.open(texture_path).convert("RGB")
    texture_image_np = np.array(texture_image_pil).astype(np.float32) / 255.0
    texture_image_tensor = (
        torch.tensor(texture_image_np, dtype=torch.float32).unsqueeze(0).to(device)
    )

    faces_uvs_t = faces_uvs.to(torch.int64).unsqueeze(0).to(device)
    verts_uvs_t = verts_uvs.to(torch.float32).unsqueeze(0).to(device)

    from pytorch3d.renderer import TexturesUV

    textures = TexturesUV(
        maps=texture_image_tensor, faces_uvs=faces_uvs_t, verts_uvs=verts_uvs_t
    )
    return textures


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
