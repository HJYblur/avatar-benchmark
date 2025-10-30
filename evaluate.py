import os
import re
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config_utils import load_config
from src.utils.data_loader import mask_loader
from src.preprocessing.video_preprocessor import VideoPreprocessor
from src.metrics.evaluator import MetricsCalculator


if __name__ == "__main__":
    # Load configuration
    cfg = load_config("config/default.yaml")
    subject = cfg["subject"]
    people_snapshot_folder = cfg["people_snapshot_folder"]
    frame_folder_name = cfg.get("frame_folder_name", "frames")
    video_filename = cfg.get("video_filename", f"{subject}.mp4")
    mask_filename = cfg.get("mask_filename", "masks.hdf5")
    output_folder = cfg["output_folder"]

    # Load original images
    frame_folder = os.path.join(people_snapshot_folder, subject, frame_folder_name)
    mask_file = os.path.join(people_snapshot_folder, subject, mask_filename)
    if not os.path.exists(frame_folder):
        mask_arr = mask_loader(mask_file)
        video_processor = VideoPreprocessor()
        video_path = os.path.join(people_snapshot_folder, subject, video_filename)
        print(f"Extracting unmasked frames from video: {video_path}")
        video_processor.extract_frames_without_mask(video_path, mask_arr, frame_folder)
        print(f"Frames extracted to: {frame_folder}")

    def numeric_key(fname: str):
        m = re.search(r"(\d+)", fname)
        return int(m.group(1)) if m else float("inf")

    original_frames = []
    original_names = []
    print(f"Loading frames from: {frame_folder}")
    orig_files = sorted(
        [
            f
            for f in os.listdir(frame_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ],
        key=numeric_key,
    )
    for frame_file in orig_files:
        frame_path = os.path.join(frame_folder, frame_file)
        image = np.array(Image.open(frame_path))
        original_frames.append(image)
        original_names.append(frame_file)
    n_frames = len(original_frames)

    # Load reconstructed images
    reconstructed_frames = []
    reconstructed_names = []
    print(f"Loading reconstructed frames from: {output_folder}")
    recon_files = sorted(
        [
            f
            for f in os.listdir(output_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ],
        key=numeric_key,
    )
    for frame_file in recon_files:
        frame_path = os.path.join(output_folder, frame_file)
        image = np.array(Image.open(frame_path))
        reconstructed_frames.append(image)
        reconstructed_names.append(frame_file)
    if len(reconstructed_frames) != n_frames:
        print("Warning: Number of reconstructed frames does not match original frames.")
        min_frames = min(len(reconstructed_frames), n_frames)
        original_frames = original_frames[:min_frames]
        reconstructed_frames = reconstructed_frames[:min_frames]
        print(f"Using {min_frames} frames for evaluation.")

    # Print debug mapping between original and reconstructed filenames for the first few frames
    print("Example filename mapping (original -> reconstructed):")
    for i in range(min(10, len(original_frames))):
        orig_name = original_names[i] if i < len(original_names) else "<missing>"
        recon_name = (
            reconstructed_names[i] if i < len(reconstructed_names) else "<missing>"
        )
        print(f"  {i:03d}: {orig_name}  ->  {recon_name}")

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

    # Load example comparison image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Frame")
    plt.imshow(original_frames[0])
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Frame")
    plt.imshow(reconstructed_frames[0])
    plt.axis("off")
    plt.show()
