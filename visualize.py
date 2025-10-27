import torch
import smplx
import numpy as np
import trimesh
import os
import pickle

# --- 1. Configuration ---
# --------------------------
# ⚠️ UPDATE THIS: Path to the directory containing SMPL_MALE.pkl, SMPL_FEMALE.pkl, etc.
SMPL_MODEL_DIR = "models/smpl/"

# ⚠️ UPDATE THIS: Path to a .npz file from the PeopleSnapshot dataset
POSE_FILE_PATH = "data/female-3-casual/poses.npz"
SHAPE_FILE_PATH = "data/female-3-casual/consensus.pkl"
# --------------------------


def visualize_first_frame(model_dir, pose_npz_path, shape_npz_path):
    """
    Loads SMPL parameters from a PeopleSnapshot .npz file,
    generates the mesh for the first frame, and visualizes it.
    """

    # --- 2. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: SMPL model directory not found at: {SMPL_MODEL_DIR}")
        print("Please download the SMPL models from https://smpl-x.is.tue.mpg.de/")
        return

    # Check if data file exists
    if not os.path.exists(pose_npz_path):
        print(f"Error: PeopleSnapshot file not found at: {pose_npz_path}")
        return
    if not os.path.exists(shape_npz_path):
        print(f"Error: PeopleSnapshot file not found at: {shape_npz_path}")
        return

    # --- 3. Load PeopleSnapshot Data ---
    try:
        # shape = pickle.load(open(shape_npz_path, "rb"), encoding="latin")
        pose_data = np.load(pose_npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    # Get gender and ensure it's a string
    gender = "neutral"
    print(f"Loaded data for gender: {gender}")

    # Get parameters for the *first frame*
    betas = pose_data["betas"][:10]  # Take only the first 10 shape parameters
    # print(f"Loaded shape parameters (betas): {betas}")
    poses = pose_data["thetas"][0]  # First frame pose
    # print(f"Loaded first frame pose parameters (thetas): {poses}")
    trans = pose_data["transl"][0]  # First frame translation
    # print(f"Loaded first frame translation (transl): {trans}")

    # --- 4. Load SMPL Model ---
    # We load the model with batch_size=1
    model = smplx.SMPL(model_path=model_dir, gender=gender, batch_size=1).to(device)

    # --- 5. Prepare Tensors ---
    # Add a batch dimension (B=1) and move to device
    betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device)
    trans_tensor = torch.tensor(trans, dtype=torch.float32).unsqueeze(0).to(device)

    # Poses are (B, 72). Split into global orientation (B, 3) and body pose (B, 69)
    poses_tensor = torch.tensor(poses, dtype=torch.float32).unsqueeze(0).to(device)
    global_orient = poses_tensor[:, :3]
    body_pose = poses_tensor[:, 3:]

    # --- 6. Generate Mesh (Forward Pass) ---
    print("Generating mesh...")
    with torch.no_grad():  # We don't need gradients for visualization
        model_output = model(
            betas=betas_tensor,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=trans_tensor,
        )

    # Get vertices and faces
    # .squeeze() removes the batch dimension (B, N, 3) -> (N, 3)
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces  # Faces are a constant numpy array

    print(
        f"Mesh generated with {vertices.shape[0]} vertices and {faces.shape[0]} faces."
    )

    # --- 7. Visualize ---
    print("Launching visualization...")

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Use mesh.show() for a simple interactive window
    mesh.show()

    # --- Optional: For a better view with a ground plane ---
    # scene = trimesh.Scene()
    # scene.add_geometry(mesh)

    # # Add a ground plane
    # ground = trimesh.creation.box(extents=[2.0, 2.0, 0.01])
    # ground.visual.face_colors = [0.8, 0.8, 0.8, 0.5] # Semi-transparent grey
    # scene.add_geometry(ground, transform=trimesh.transformations.translation_matrix([0, 0, -mesh.bounds[0][2]]))

    # scene.show()


if __name__ == "__main__":
    # ⚠️ UPDATE the paths at the top of the script before running
    visualize_first_frame(SMPL_MODEL_DIR, POSE_FILE_PATH, SHAPE_FILE_PATH)
