import argparse
import json
import torch
import smplx
import numpy as np
import os
import pickle
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    RasterizationSettings,
    PointLights,
    BlendParams,
    Materials,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d, euler_angles_to_matrix

from src.utils.config_utils import load_config
from src.utils.data_loader import load_smpl_shape, load_smpl_pose

# Default configuration values
SMPL_FOLDER = None
PEOPLE_SNAPSHOT_FOLDER = None
POSE_FILE_PATH = None
SHAPE_FILE_PATH = None
TEXTURE_FILE_PATH = None
SMPL_UV_TEMPLATE_PATH = None
OUTPUT_IMAGE_FOLDER = None

# --------------------------


def file_check():
    # Check if model directory exists
    if not os.path.exists(os.path.join(SMPL_FOLDER, "smpl")):
        print(f"Error: SMPL model directory not found at: {SMPL_FOLDER}")
        return False

    # Check if data files exist
    if not os.path.exists(POSE_FILE_PATH):
        print(f"Error: PeopleSnapshot pose file not found at: {POSE_FILE_PATH}")
        return False
    if not os.path.exists(SHAPE_FILE_PATH):
        print(f"Error: PeopleSnapshot shape file not found at: {SHAPE_FILE_PATH}")
        return False
    if not os.path.exists(TEXTURE_FILE_PATH):
        print(f"Error: Texture file not found at: {TEXTURE_FILE_PATH}")
        return False
    if not os.path.exists(SMPL_UV_TEMPLATE_PATH):
        print(f"Error: SMPL UV template file not found at: {SMPL_UV_TEMPLATE_PATH}")
        return False
    return True


def smpl_data_loader():
    # Load shape and pose data using the utility functions
    shape_data = load_smpl_shape(SHAPE_FILE_PATH)
    pose_data = load_smpl_pose(POSE_FILE_PATH)

    # Get shape from the shape data
    betas = shape_data["betas"].squeeze()  # Get (10,) array
    if betas.shape[0] > 10:
        betas = betas[:10]  # Ensure it's only 10 shape params

    # Get pose and translation for the whole sequence
    poses = pose_data["pose"]
    trans = pose_data["trans"]
    return betas, poses, trans


def uv_loader(device):
    # Load the SMPL UV template to get UV coordinates
    _, faces_data, aux_data = load_obj(SMPL_UV_TEMPLATE_PATH, device=device)

    verts_uvs = aux_data.verts_uvs  # (V_uv, 2)
    faces_uvs = faces_data.textures_idx  # (F, 3)
    return verts_uvs, faces_uvs


def texture_loader(device, verts_uvs, faces_uvs):
    # Load texture image
    texture_image_pil = Image.open(TEXTURE_FILE_PATH).convert("RGB")
    texture_image_np = np.array(texture_image_pil).astype(np.float32) / 255.0
    texture_image_tensor = (
        torch.tensor(texture_image_np, dtype=torch.float32).unsqueeze(0).to(device)
    )

    # Ensure UV tensors are on the right device and dtype and have a batch dim
    faces_uvs_t = faces_uvs.to(torch.int64).unsqueeze(0).to(device)
    verts_uvs_t = verts_uvs.to(torch.float32).unsqueeze(0).to(device)

    textures = TexturesUV(
        maps=texture_image_tensor,
        faces_uvs=faces_uvs_t,  # (1, F, 3)
        verts_uvs=verts_uvs_t,  # (1, V_uv, 2)
    )
    return textures


def renderer_setup(device, cam_dist=2.7, cam_elev=30.0, cam_azim=0.0, image_size=512):
    R_cam, T_cam = look_at_view_transform(
        dist=cam_dist, elev=cam_elev, azim=cam_azim, device=device
    )
    cameras = FoVPerspectiveCameras(device=device, R=R_cam, T=T_cam, fov=60)
    # Place a light co-located with the camera so the model is lit from the view direction.
    lights = PointLights(device=device, location=T_cam)
    materials = Materials(
        device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
        max_faces_per_bin=1000,
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
    return renderer


if __name__ == "__main__":
    """
    Loads SMPL parameters,
    generates the mesh for the first frame, and visualizes it with texture.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(
        description="Render a subject from PeopleSnapshot using SMPL + PyTorch3D"
    )
    parser.add_argument("--override", type=str, default=False)
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file to override defaults",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Subject folder name to render (e.g. female-3-casual)",
    )
    parser.add_argument(
        "--people_folder",
        type=str,
        default=None,
        help="Top-level PeopleSnapshot folder",
    )
    parser.add_argument(
        "--smpl_folder", type=str, default=None, help="SMPL models folder"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output folder for rendered images"
    )
    parser.add_argument("--camera_dist", type=float, default=None)
    parser.add_argument("--camera_elev", type=float, default=None)
    parser.add_argument("--camera_azim", type=float, default=None)
    args = parser.parse_args()

    if args.override:
        overrides = {
            "subject": args.subject,
            "people_snapshot_folder": args.people_folder,
            "smpl_folder": args.smpl_folder,
            "output_folder": args.output,
            "camera_dist": args.camera_dist,
            "camera_elev": args.camera_elev,
            "camera_azim": args.camera_azim,
        }
        cfg = load_config(args.config, overrides)
    else:
        cfg = load_config(args.config)

    # Set globals from cfg
    SMPL_FOLDER = cfg["smpl_folder"]
    PEOPLE_SNAPSHOT_FOLDER = cfg["people_snapshot_folder"]
    subject = cfg["subject"]
    candidate_base = os.path.join(PEOPLE_SNAPSHOT_FOLDER, subject)
    POSE_FILE_PATH = os.path.join(candidate_base, cfg["pose_filename"])
    SHAPE_FILE_PATH = os.path.join(candidate_base, cfg["shape_filename"])
    TEXTURE_FILE_PATH = os.path.join(candidate_base, cfg["texture_filename"])
    SMPL_UV_TEMPLATE_PATH = cfg["smpl_uv_template"]
    OUTPUT_IMAGE_FOLDER = cfg["output_folder"]

    FRAME_NUM = 0

    file_check()

    # --- 3. Load PeopleSnapshot Data ---
    betas, poses, trans = smpl_data_loader()
    betas_tensor = torch.tensor(betas, dtype=torch.float32).to(device).unsqueeze(0)
    trans_tensor = torch.tensor(trans[FRAME_NUM], dtype=torch.float32).to(device)
    poses_tensor = torch.tensor(poses[FRAME_NUM], dtype=torch.float32).to(device)
    print("Betas shape:", betas_tensor.shape)
    print("Trans shape:", trans_tensor.shape)
    print("Poses shape:", poses_tensor.shape)
    global_orient = poses_tensor[:3].unsqueeze(0)
    body_pose = poses_tensor[3:].unsqueeze(0)
    print("Global orient shape:", global_orient.shape)
    print("Body pose shape:", body_pose.shape)

    # # Set random body pose, betas, and global orientation
    # betas_tensor = torch.zeros(1, 10)  # 10 shape coefficients
    # global_orient = torch.zeros(1, 3)  # Global rotation
    # body_pose = torch.zeros(1, 23 * 3)  # 21 joints * 3 (axis-angle representation)

    # --- Load SMPL Model ---
    model = smplx.create(
        model_path=SMPL_FOLDER,
        model_type="smpl",
        gender=cfg.get("gender", "neutral"),
        batch_size=1,
    ).to(device)

    # --- Generate Mesh (Forward Pass) ---
    print("Generating mesh...")
    with torch.no_grad():
        model_output = model(
            betas=betas_tensor,
            body_pose=body_pose,
            global_orient=global_orient,
            # transl=trans_tensor,
            return_verts=True,
        )

    # Get posed vertices and the model's faces
    R_fix = euler_angles_to_matrix(torch.tensor([[np.pi / 2, np.pi, 0.0]]), "XYZ")[0]
    vertices = model_output.vertices.squeeze(0) @ R_fix.T
    # vertices = model_output.vertices.squeeze(0)  # (6890, 3)
    faces = model.faces.astype(np.int64)  # (F, 3)

    # --- Prepare Texture ---
    verts_uvs, faces_uvs = uv_loader(device)
    texture = texture_loader(device, verts_uvs, faces_uvs)

    mesh = Meshes(
        verts=[vertices.squeeze(0).to(device)],
        faces=[torch.tensor(model.faces.astype(np.int64), device=device)],
        textures=texture,
    )

    # --- Set Up Renderer ---
    renderer = renderer_setup(
        device,
        cam_dist=cfg.get("camera_dist", 2.7),
        cam_elev=cfg.get("camera_elev", 0.0),
        cam_azim=cfg.get("camera_azim", 180.0),
    )

    # --- Render the Mesh ---
    images = renderer(mesh)

    # Save the image
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    output_image_path = os.path.join(OUTPUT_IMAGE_FOLDER, "rendered_frame_0.png")
    image = images[0, ..., :3].cpu().numpy()
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_pil.save(output_image_path)
    print(f"Rendered image saved to: {output_image_path}")
