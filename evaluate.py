import os
import numpy as np
from PIL import Image
import numpy as np
from src.utils.config_utils import load_config
from src.preprocessing.video_preprocessor import VideoPreprocessor
from src.metrics.evaluator import MetricsCalculator


if __name__ == "__main__":
    # Load configuration
    cfg = load_config("config/default.yaml")
    subject = cfg["subject"]
    people_snapshot_folder = cfg["people_snapshot_folder"]
    frame_folder_name = cfg.get("frame_folder_name", "frames")
    video_filename = cfg.get("video_filename", f"{subject}.mp4")
    output_folder = cfg["output_folder"]

    # Load original images
    frame_folder = os.path.join(people_snapshot_folder, subject, frame_folder_name)
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
        video_processor = VideoPreprocessor()
        video_path = os.path.join(people_snapshot_folder, subject, video_filename)
        print(f"Extracting frames from video: {video_path}")
        video_processor.extract_frames(video_path, frame_folder)
        print(f"Frames extracted to: {frame_folder}")

    original_frames = []
    print(f"Loading frames from: {frame_folder}")
    for frame_file in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame_file)
        image = np.array(Image.open(frame_path))
        original_frames.append(image)
    n_frames = len(original_frames)

    # Load reconstructed images
    reconstructed_frames = []
    print(f"Loading reconstructed frames from: {output_folder}")
    for frame_file in os.listdir(output_folder):
        frame_path = os.path.join(output_folder, frame_file)
        image = np.array(Image.open(frame_path))
        reconstructed_frames.append(image)
    if len(reconstructed_frames) != n_frames:
        print("Warning: Number of reconstructed frames does not match original frames.")
        min_frames = min(len(reconstructed_frames), n_frames)
        original_frames = original_frames[:min_frames]
        reconstructed_frames = reconstructed_frames[:min_frames]
        print(f"Using {min_frames} frames for evaluation.")

    # Metric calculation
    metrics_calculator = MetricsCalculator(
        compute_psnr=True,
        compute_ssim=True,
        compute_lpips=True,
        compute_mesh_error=False,
    )

    # Test first frame
    psnr_value = metrics_calculator.calculate_psnr(
        original_frames[0], reconstructed_frames[0]
    )
    ssim_value = metrics_calculator.calculate_ssim(
        original_frames[0], reconstructed_frames[0]
    )
    lpips_value = metrics_calculator.calculate_lpips(
        original_frames[0], reconstructed_frames[0]
    )
    print(f"Frame 0 - PSNR: {psnr_value}, SSIM: {ssim_value}, LPIPS: {lpips_value}")

    # psnr, ssim, lpips = [], [], []
    # for img1, img2 in zip(original_frames, reconstructed_frames):
    #     psnr_value = metrics_calculator.calculate_psnr(img1, img2)
    #     ssim_value = metrics_calculator.calculate_ssim(img1, img2)
    #     lpips_value = metrics_calculator.calculate_lpips(img1, img2)
    #     psnr.append(psnr_value)
    #     ssim.append(ssim_value)
    #     lpips.append(lpips_value)

    # print(f"PSNR: {np.mean(psnr)}")
    # print(f"SSIM: {np.mean(ssim)}")
    # print(f"LPIPS: {np.mean(lpips)}")
