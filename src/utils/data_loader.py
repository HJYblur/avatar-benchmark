import os
from typing import Tuple
import numpy as np
import torch
import h5py
from PIL import Image, ImageDraw
from pytorch3d.io import load_obj


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
    if pose_path.endswith(".hdf5") or pose_path.endswith(".h5"):
        pose_data = load_pose_fn(pose_path)
    elif pose_path.endswith(".npz"):
        pose_data = np.load(pose_path)

    betas = shape_data["betas"].squeeze()
    if betas.shape[0] > 10:
        betas = betas[:10]

    pose_key = "pose" if "pose" in pose_data else "poses"
    poses = pose_data[pose_key]
    trans = pose_data["trans"]
    return betas, poses, trans


def mask_loader(mask_path: str) -> np.ndarray:
    with h5py.File(mask_path, "r") as f:
        masks = f["masks"][:]
    return masks  # Shape: (n_frames, H, W)


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
