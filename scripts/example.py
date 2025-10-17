#!/usr/bin/env python3
"""
Example script demonstrating individual pipeline components.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.video_preprocessor import VideoPreprocessor
from src.inference.pymafx_wrapper import PyMAFXInference
from src.rendering.mesh_generator import SMPLXMeshGenerator
from src.rendering.renderer import MeshRenderer
from src.metrics.evaluator import MetricsCalculator
from src.utils.config_utils import load_config


def example_preprocessing():
    """Example: Video preprocessing."""
    print("\n=== Video Preprocessing Example ===")
    preprocessor = VideoPreprocessor(fps=30, max_frames=10, frame_skip=1)
    print("VideoPreprocessor initialized")
    print(f"  FPS: {preprocessor.fps}")
    print(f"  Max frames: {preprocessor.max_frames}")
    print(f"  Frame skip: {preprocessor.frame_skip}")


def example_inference():
    """Example: PyMAF-X inference."""
    print("\n=== PyMAF-X Inference Example ===")
    pymafx = PyMAFXInference(model_path="models/pymafx", device="cuda")
    print("PyMAFXInference initialized")
    print(f"  Model path: {pymafx.model_path}")
    print(f"  Device: {pymafx.device}")
    
    # Show dummy SMPL-X params structure
    dummy_params = pymafx._get_dummy_smplx_params()
    print("\nSMPL-X parameters structure:")
    for key, value in dummy_params.items():
        print(f"  {key}: shape {value.shape}")


def example_mesh_generation():
    """Example: Mesh generation."""
    print("\n=== Mesh Generation Example ===")
    mesh_gen = SMPLXMeshGenerator(
        model_path="models/smplx",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10
    )
    print("SMPLXMeshGenerator initialized")
    print(f"  Model path: {mesh_gen.model_path}")
    print(f"  Gender: {mesh_gen.gender}")
    print(f"  Num betas: {mesh_gen.num_betas}")
    print(f"  Num expression coeffs: {mesh_gen.num_expression_coeffs}")


def example_rendering():
    """Example: Mesh rendering."""
    print("\n=== Mesh Rendering Example ===")
    renderer = MeshRenderer(
        image_size=(512, 512),
        camera_distance=2.5,
        light_intensity=3.0,
        background_color=(0, 0, 0)
    )
    print("MeshRenderer initialized")
    print(f"  Image size: {renderer.image_size}")
    print(f"  Camera distance: {renderer.camera_distance}")
    print(f"  Light intensity: {renderer.light_intensity}")


def example_metrics():
    """Example: Metrics calculation."""
    print("\n=== Metrics Calculation Example ===")
    evaluator = MetricsCalculator(
        compute_psnr=True,
        compute_ssim=True,
        compute_lpips=False,
        compute_mesh_error=True
    )
    print("MetricsCalculator initialized")
    print(f"  Compute PSNR: {evaluator.compute_psnr}")
    print(f"  Compute SSIM: {evaluator.compute_ssim}")
    print(f"  Compute LPIPS: {evaluator.compute_lpips}")
    print(f"  Compute mesh error: {evaluator.compute_mesh_error}")


def example_config():
    """Example: Configuration loading."""
    print("\n=== Configuration Example ===")
    try:
        config = load_config("configs/default.yaml")
        print("Configuration loaded successfully")
        print("\nConfiguration keys:")
        for key in config.keys():
            print(f"  - {key}")
    except Exception as e:
        print(f"Error loading config: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Avatar Benchmark - Component Examples")
    print("=" * 60)
    
    example_preprocessing()
    example_inference()
    example_mesh_generation()
    example_rendering()
    example_metrics()
    example_config()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
