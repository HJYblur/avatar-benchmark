#!/usr/bin/env python3
"""
Main pipeline script for full-body avatar evaluation.
Pipeline: Video → PyMAF-X → SMPL-X params → Mesh/Render → Metrics
"""
import os
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.video_preprocessor import VideoPreprocessor
from src.inference.pymafx_wrapper import PyMAFXInference
from src.rendering.mesh_generator import SMPLXMeshGenerator
from src.rendering.renderer import MeshRenderer
from src.metrics.evaluator import MetricsCalculator
from src.utils.config_utils import load_config


def main(args):
    """Main pipeline execution."""
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Create output directories
    output_base = Path(config.OUTPUT_PATH) / args.output_name
    frames_dir = output_base / "frames"
    params_dir = output_base / "params"
    meshes_dir = output_base / "meshes"
    renders_dir = output_base / "renders"
    metrics_dir = output_base / "metrics"
    
    for dir_path in [frames_dir, params_dir, meshes_dir, renders_dir, metrics_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_base}")
    
    # Step 1: Video Preprocessing
    print("\n" + "="*50)
    print("Step 1: Video Preprocessing")
    print("="*50)
    
    preprocessor = VideoPreprocessor(
        fps=config.PROCESSING.FPS,
        max_frames=config.PROCESSING.MAX_FRAMES,
        frame_skip=config.PROCESSING.FRAME_SKIP
    )
    
    frame_paths = preprocessor.extract_frames(args.video, str(frames_dir))
    print(f"Extracted {len(frame_paths)} frames")
    
    # Step 2: PyMAF-X Inference
    print("\n" + "="*50)
    print("Step 2: PyMAF-X Inference (SMPL-X Parameter Extraction)")
    print("="*50)
    
    pymafx = PyMAFXInference(
        model_path=config.PYMAFX_MODEL_PATH,
        device=config.PYMAFX.DEVICE
    )
    
    # Load frames for inference
    import cv2
    frames = [cv2.imread(fp) for fp in frame_paths]
    
    # Run inference
    smplx_params = pymafx.infer(frames)
    
    # Save parameters
    params_file = params_dir / "smplx_params.npy"
    pymafx.save_params(smplx_params, str(params_file))
    
    # Step 3: Mesh Generation
    print("\n" + "="*50)
    print("Step 3: Mesh Generation from SMPL-X Parameters")
    print("="*50)
    
    mesh_generator = SMPLXMeshGenerator(
        model_path=config.SMPLX_MODEL_PATH,
        gender=config.SMPLX.GENDER,
        num_betas=config.SMPLX.NUM_BETAS,
        num_expression_coeffs=config.SMPLX.NUM_EXPRESSION_COEFFS
    )
    
    mesh_paths = mesh_generator.generate_meshes_batch(smplx_params, str(meshes_dir))
    print(f"Generated {len(mesh_paths)} meshes")
    
    # Step 4: Rendering
    print("\n" + "="*50)
    print("Step 4: Mesh Rendering")
    print("="*50)
    
    renderer = MeshRenderer(
        image_size=tuple(config.RENDERING.IMAGE_SIZE),
        camera_distance=config.RENDERING.CAMERA_DISTANCE,
        light_intensity=config.RENDERING.LIGHT_INTENSITY,
        background_color=config.RENDERING.BACKGROUND_COLOR
    )
    
    render_paths = renderer.render_meshes_batch(mesh_paths, str(renders_dir))
    print(f"Rendered {len(render_paths)} images")
    
    # Step 5: Evaluation Metrics (if ground truth is provided)
    if args.ground_truth:
        print("\n" + "="*50)
        print("Step 5: Evaluation Metrics")
        print("="*50)
        
        evaluator = MetricsCalculator(
            compute_psnr=config.METRICS.COMPUTE_PSNR,
            compute_ssim=config.METRICS.COMPUTE_SSIM,
            compute_lpips=config.METRICS.COMPUTE_LPIPS,
            compute_mesh_error=config.METRICS.COMPUTE_MESH_ERROR
        )
        
        # Get ground truth frame paths
        gt_preprocessor = VideoPreprocessor(
            fps=config.PROCESSING.FPS,
            max_frames=config.PROCESSING.MAX_FRAMES,
            frame_skip=config.PROCESSING.FRAME_SKIP
        )
        gt_frames_dir = output_base / "gt_frames"
        gt_frames_dir.mkdir(exist_ok=True)
        gt_frame_paths = gt_preprocessor.extract_frames(args.ground_truth, str(gt_frames_dir))
        
        # Evaluate
        results = evaluator.evaluate_sequence(render_paths, gt_frame_paths)
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 30)
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        
        # Save results
        results_file = metrics_dir / "evaluation_results.txt"
        evaluator.save_results(results, str(results_file))
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {output_base}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-body avatar evaluation pipeline")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_name", type=str, default="evaluation",
                       help="Name for output directory")
    parser.add_argument("--ground_truth", type=str, default=None,
                       help="Path to ground truth video for evaluation (optional)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    if args.ground_truth and not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth video not found: {args.ground_truth}")
        sys.exit(1)
    
    main(args)
