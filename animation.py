import argparse
import json
import torch
import smplx
import torch
import smplx
import numpy as np
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime
import csv
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
from pytorch3d.transforms import euler_angles_to_matrix
from src.utils.config_utils import load_config
from src.preprocessing.video_preprocessor import VideoPreprocessor
from src.utils.smpl_loader import load_smpl_shape, load_smpl_pose
from src.utils.data_loader import (
    file_check,
    load_sequence,
    uv_loader,
    texture_loader,
)
from src.utils.render_utils import (
    renderer_setup,
    save_image_with_axes,
    save_image,
)
from src.utils.render_utils import camera_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    cfg = load_config("config/animation.yaml")

    # Set globals from cfg
    SMPL_FOLDER = cfg["smpl_folder"]
    PEOPLE_SNAPSHOT_FOLDER = cfg["people_snapshot_folder"]
    subject = cfg["subject"]
    POSE_FILE_PATH = cfg["pose_filename"]
    SHAPE_FILE_PATH = cfg["shape_filename"]
    TEXTURE_FILE_PATH = cfg["texture_filename"]
    SMPL_UV_TEMPLATE_PATH = cfg["smpl_uv_template"]
    OUTPUT_IMAGE_FOLDER = cfg["output_folder"]
    LOG_FILE_PATH = cfg.get("log_folder", "logs")

    # Quick checks
    if not file_check(
        SMPL_FOLDER,
        POSE_FILE_PATH,
        SHAPE_FILE_PATH,
        TEXTURE_FILE_PATH,
        SMPL_UV_TEMPLATE_PATH,
    ):
        raise SystemExit(1)

    # Load sequence data
    betas, poses, trans = load_sequence(
        SHAPE_FILE_PATH, POSE_FILE_PATH, load_smpl_shape, load_smpl_pose
    )
    n_frames = int(poses.shape[0])
    print(f"Sequence length: {n_frames} frames")

    # --- Load SMPL Model once ---
    model = smplx.create(
        model_path=SMPL_FOLDER,
        model_type="smpl",
        gender=cfg.get("gender", "neutral"),
        batch_size=1,
    ).to(device)

    # Precompute UVs and texture
    verts_uvs, faces_uvs = uv_loader(device, SMPL_UV_TEMPLATE_PATH)
    textures = texture_loader(device, TEXTURE_FILE_PATH, verts_uvs, faces_uvs)

    # Setup renderer (returns renderer and cameras)
    image_size = cfg.get("image_size", 512)

    renderer, cameras = renderer_setup(
        device,
        cam_dist=cfg.get("camera_dist", 2.7),
        cam_elev=cfg.get("camera_elev", 0.0),
        cam_azim=cfg.get("camera_azim", 0.0),
        camera_fov=cfg.get("camera_fov", 30.0),
        image_size=image_size,
    )

    # Rotation fix used previously to orient mesh
    R_fix = euler_angles_to_matrix(torch.tensor([[np.pi, 0.0, 0.0]]), "XYZ")[0]

    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    axis_scale = cfg.get("axis_scale", 0.2)

    # Loop over frames and render
    inference_times = []
    os.makedirs(LOG_FILE_PATH, exist_ok=True)
    csv_path = os.path.join(LOG_FILE_PATH, f"{subject}-{datetime.now()}.csv")
    csv_writer = csv.writer(open(csv_path, "a", newline=""))
    csv_writer.writerow(["frame", "inference_time"])
    overall_start = time.perf_counter()
    rendered_frame_paths = []

    for FRAME_NUM in tqdm(range(n_frames)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device)
        poses_frame = torch.tensor(poses[FRAME_NUM], dtype=torch.float32).to(device)
        trans_frame = torch.tensor(trans[FRAME_NUM], dtype=torch.float32).to(device)
        global_orient = poses_frame[:3].unsqueeze(0)
        body_pose = poses_frame[3:].unsqueeze(0)
        transl = trans_frame.unsqueeze(0)

        # Set random body pose, betas, and global orientation
        # body_pose = torch.zeros(1, 23 * 3)  # 21 joints * 3 (axis-angle representation)
        # betas_tensor = torch.zeros(1, 10)  # 10 shape coefficients
        # global_orient = torch.zeros(1, 3)  # Global rotation

        with torch.no_grad():
            model_output = model(
                betas=betas_tensor,
                body_pose=body_pose,
                global_orient=global_orient,
                # transl=transl,
                return_verts=True,
            )

        vertices = model_output.vertices.squeeze(0) @ R_fix.T
        mesh = Meshes(
            verts=[vertices.to(device)],
            faces=[torch.tensor(model.faces.astype(np.int64), device=device)],
            textures=textures,
        )

        images = renderer(mesh)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)
        csv_writer.writerow([FRAME_NUM, t1 - t0])

        image = images[0, ..., :3].cpu().numpy()

        output_image_path = os.path.join(
            OUTPUT_IMAGE_FOLDER, f"rendered_frame_{FRAME_NUM:04d}.png"
        )
        rendered_frame_paths.append(output_image_path)
        save_image(image, output_image_path)

    # Summary
    overall_end = time.perf_counter()
    avg_inf = sum(inference_times) / n_frames if n_frames else 0.0
    print("===== Timing summary =====")
    print(f"Frames processed: {n_frames}")
    print(f"Average inference time / frame: {avg_inf:.4f} s")
    print(f"Total elapsed time            : {overall_end - overall_start:.4f} s")
    csv_writer.writerow(["Total elapsed time", "Average inference time"])
    csv_writer.writerow([overall_end - overall_start, avg_inf])

    video_processor = VideoPreprocessor()
    output_image_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{subject}_animation.mp4")
    print(f"Compiling rendered images into video: {output_image_path}")
    video_processor.images_to_video(
        rendered_frame_paths,
        output_image_path,
        fps=30,
        remove_frames=False,
    )
