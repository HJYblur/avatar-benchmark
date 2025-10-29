import torch
import smplx
import numpy as np
import os
import pickle
from PIL import Image
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

# --- Configuration ---
SMPL_FOLDER = "models"
POSE_FILE_PATH = "data/female-3-casual/poses.npz"
SHAPE_FILE_PATH = "data/female-3-casual/consensus.pkl"
TEXTURE_FILE_PATH = "data/female-3-casual/tex-female-3-casual.jpg"
SMPL_UV_TEMPLATE_PATH = "models/smpl/smpl_uv.obj"
OUTPUT_IMAGE_FOLDER = "output"
# --------------------------


def file_check():
    # Check if model directory exists
    if not os.path.exists(SMPL_FOLDER):
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
        print("This .obj file is required to get the correct UV coordinates.")
        return False
    return True


def data_loader():
    try:
        pose_data = np.load(POSE_FILE_PATH, allow_pickle=True)
        with open(SHAPE_FILE_PATH, "rb") as f:
            # Use latin1 encoding for compatibility with Python 2-saved pickles
            shape_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    # Get shape from the .pkl file
    betas = shape_data["betas"].squeeze()  # Get (10,) array
    if betas.shape[0] > 10:
        betas = betas[:10]  # Ensure it's only 10 shape params

    # Get pose and translation for the *first frame* from .npz
    poses = pose_data["thetas"][0]  # First frame pose
    trans = pose_data["transl"][0]  # First frame translation
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


def renderer_setup(device):
    R_cam, T_cam = look_at_view_transform(dist=2.7, elev=30, azim=0.0, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R_cam, T=T_cam, fov=60)
    lights = PointLights(device=device, location=T_cam)
    materials = Materials(
        device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0
    )

    raster_settings = RasterizationSettings(
        image_size=512,
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


def main():
    """
    Loads SMPL parameters,
    generates the mesh for the first frame, and visualizes it with texture.
    """

    # --- 2. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not file_check():
        return

    # --- 3. Load PeopleSnapshot Data ---
    # betas, poses, trans = data_loader()
    # betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device)
    # trans_tensor = torch.tensor(trans, dtype=torch.float32).unsqueeze(0).to(device)
    # poses_tensor = torch.tensor(poses, dtype=torch.float32).unsqueeze(0).to(device)
    # global_orient = poses_tensor[:, :3]
    # body_pose = poses_tensor[:, 3:]

    # Set random body pose, betas, and global orientation
    body_pose = torch.zeros(1, 23 * 3)  # 21 joints * 3 (axis-angle representation)
    betas_tensor = torch.zeros(1, 10)  # 10 shape coefficients
    global_orient = torch.zeros(1, 3)  # Global rotation

    # --- 4. Load SMPL Model ---
    model = smplx.create(
        model_path=SMPL_FOLDER, model_type="smpl", gender="female", batch_size=1
    ).to(device)

    # --- 6. Generate Mesh (Forward Pass) ---
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
    vertices = model_output.vertices  # (V, 3)
    faces = model.faces.astype(np.int64)  # (F, 3)

    # --- 7. Prepare Texture ---
    verts_uvs, faces_uvs = uv_loader(device)
    texture = texture_loader(device, verts_uvs, faces_uvs)

    mesh = Meshes(
        verts=[vertices.squeeze(0).to(device)],
        faces=[torch.tensor(model.faces.astype(np.int64), device=device)],
        textures=texture,
    )

    # --- 8. Set Up Renderer ---
    renderer = renderer_setup(device)

    # --- 9. Render the Mesh ---
    images = renderer(mesh)
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    output_image_path = os.path.join(OUTPUT_IMAGE_FOLDER, "rendered_frame_0.png")
    image = images[0, ..., :3].cpu().numpy()
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_pil.save(output_image_path)
    print(f"Rendered image saved to: {output_image_path}")


if __name__ == "__main__":
    main()
