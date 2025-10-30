"""
Evaluation metrics for avatar reconstruction quality.
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import trimesh
import lpips


class MetricsCalculator:
    """Calculate evaluation metrics for avatar reconstruction."""

    def __init__(
        self,
        compute_psnr: bool = True,
        compute_ssim: bool = True,
        compute_lpips: bool = True,
        compute_mesh_error: bool = True,
    ):
        """
        Initialize metrics calculator.

        Args:
            compute_psnr: Whether to compute PSNR
            compute_ssim: Whether to compute SSIM
            compute_lpips: Whether to compute LPIPS
            compute_mesh_error: Whether to compute mesh error metrics
        """
        self.compute_psnr = compute_psnr
        self.compute_ssim = compute_ssim
        self.compute_lpips = compute_lpips
        self.compute_mesh_error = compute_mesh_error

        if self.compute_lpips:
            try:
                import lpips

                self.lpips_fn = lpips.LPIPS(net="alex")
            except ImportError:
                print("Warning: lpips not installed, LPIPS metric will not be computed")
                self.compute_lpips = False

    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate PSNR between two images.

        Args:
            img1: First image
            img2: Second image

        Returns:
            PSNR value
        """
        print(img1.shape, img2.shape)
        return psnr(img1, img2, data_range=255)

    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate SSIM between two images.

        Args:
            img1: First image
            img2: Second image

        Returns:
            SSIM value
        """
        # Convert to grayscale if images are color
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2

        return ssim(img1_gray, img2_gray, data_range=255)

    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate LPIPS between two images.

        Args:
            img1: First image (H, W, 3) in BGR format
            img2: Second image (H, W, 3) in BGR format

        Returns:
            LPIPS value
        """
        if not self.compute_lpips:
            return 0.0

        import torch

        # Convert BGR to RGB and normalize to [-1, 1]
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        img1_rgb = (img1_rgb - 0.5) / 0.5
        img2_rgb = (img2_rgb - 0.5) / 0.5

        # Convert to tensors
        img1_tensor = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0)

        # Calculate LPIPS
        with torch.no_grad():
            lpips_val = self.lpips_fn(img1_tensor, img2_tensor)

        return lpips_val.item()

    def calculate_mesh_error(
        self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh
    ) -> Dict[str, float]:
        """
        Calculate mesh error metrics.

        Args:
            mesh1: First mesh
            mesh2: Second mesh (ground truth)

        Returns:
            Dictionary with mesh error metrics
        """
        # Chamfer distance (simplified version)
        vertices1 = mesh1.vertices
        vertices2 = mesh2.vertices

        # Sample points if meshes have different number of vertices
        n_samples = min(len(vertices1), len(vertices2), 1000)

        # Calculate point-to-point distances
        distances = np.linalg.norm(
            vertices1[:n_samples] - vertices2[:n_samples], axis=1
        )

        return {
            "mean_error": np.mean(distances),
            "max_error": np.max(distances),
            "std_error": np.std(distances),
        }

    def evaluate_sequence(
        self, rendered_paths: List[str], ground_truth_paths: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a sequence of rendered images against ground truth.

        Args:
            rendered_paths: List of paths to rendered images
            ground_truth_paths: List of paths to ground truth images

        Returns:
            Dictionary with average metrics
        """
        assert len(rendered_paths) == len(
            ground_truth_paths
        ), "Number of rendered and ground truth images must match"

        psnr_values = []
        ssim_values = []
        lpips_values = []

        for rendered_path, gt_path in zip(rendered_paths, ground_truth_paths):
            # Load images
            rendered = cv2.imread(rendered_path)
            gt = cv2.imread(gt_path)

            # Resize if needed
            if rendered.shape != gt.shape:
                rendered = cv2.resize(rendered, (gt.shape[1], gt.shape[0]))

            # Calculate metrics
            if self.compute_psnr:
                psnr_values.append(self.calculate_psnr(rendered, gt))

            if self.compute_ssim:
                ssim_values.append(self.calculate_ssim(rendered, gt))

            if self.compute_lpips:
                lpips_values.append(self.calculate_lpips(rendered, gt))

        results = {}
        if psnr_values:
            results["avg_psnr"] = np.mean(psnr_values)
            results["std_psnr"] = np.std(psnr_values)

        if ssim_values:
            results["avg_ssim"] = np.mean(ssim_values)
            results["std_ssim"] = np.std(ssim_values)

        if lpips_values:
            results["avg_lpips"] = np.mean(lpips_values)
            results["std_lpips"] = np.std(lpips_values)

        return results

    def save_results(self, results: Dict[str, float], output_path: str):
        """
        Save evaluation results to file.

        Args:
            results: Dictionary with evaluation results
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write("Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")

        print(f"Saved evaluation results to {output_path}")
