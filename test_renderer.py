"""
Test script for GsplatRenderer
Creates Gaussians along the edges of a 3D pyramid to verify rendering functionality.
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Make 'src' importable
sys.path.append(str(Path(__file__).parent / "src"))

from src.render.gaussian_renderer import GsplatRenderer
from avatar_utils.config import load_config
import os


def save_points_as_ply(
    points: torch.Tensor,
    ply_path: str,
    colors: torch.Tensor | None = None,
) -> None:
    """Save point cloud as an ASCII PLY.

    Args:
        points: (N, 3) float tensor (any device)
        ply_path: output file path
        colors: optional (N, 3) uint8 or float tensor in [0,1]/[0,255]
    """
    pts = points.detach().to("cpu").float().contiguous().numpy()
    N = pts.shape[0]

    cols = None
    if colors is not None:
        c = colors.detach().to("cpu").contiguous()
        if c.dtype != torch.uint8:
            c = c.float()
            # Accept either 0..1 or 0..255
            if c.max().item() <= 1.0:
                c = (c * 255.0).round()
            c = c.clamp(0, 255).to(torch.uint8)
        cols = c.numpy()
        if cols.shape != (N, 3):
            raise ValueError(f"colors must have shape (N,3), got {cols.shape}")

    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    has_color = cols is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header += ["end_header"]

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        if has_color:
            for i in range(N):
                x, y, z = pts[i]
                r, g, b = cols[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for i in range(N):
                x, y, z = pts[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def create_pyramid_gaussians(
    apex_height=1.0,
    base_size=1.0,
    num_points_per_edge=10,
    device="cpu"
):
    """
    Create Gaussians along the edges of a 3D pyramid.
    
    The pyramid has:
    - An apex at (0, apex_height, 0)
    - A square base centered at origin in the XZ plane with corners at:
      - (-base_size/2, 0, -base_size/2)
      - (base_size/2, 0, -base_size/2)
      - (base_size/2, 0, base_size/2)
      - (-base_size/2, 0, base_size/2)
    
    Args:
        apex_height: Height of the pyramid apex
        base_size: Size of the square base
        num_points_per_edge: Number of Gaussians to place along each edge
        device: torch device
    
    Returns:
        gaussian_3d: Tensor of shape (N, 3) with Gaussian centers
        gaussian_params: Dictionary with Gaussian parameters
    """
    # Define pyramid vertices
    apex = np.array([0.0, apex_height, 0.0])
    base_corners = np.array([
        [-base_size/2, 0.0, -base_size/2],  # corner 0
        [base_size/2, 0.0, -base_size/2],   # corner 1
        [base_size/2, 0.0, base_size/2],    # corner 2
        [-base_size/2, 0.0, base_size/2],   # corner 3
    ])
    
    points_list = []
    
    # Create points along edges from apex to each base corner (4 edges)
    for i in range(4):
        edge_points = np.linspace(apex, base_corners[i], num_points_per_edge)
        points_list.append(edge_points)
    
    # Create points along base edges (4 edges)
    for i in range(4):
        next_i = (i + 1) % 4
        edge_points = np.linspace(base_corners[i], base_corners[next_i], num_points_per_edge)
        # Exclude last point to avoid duplicates at corners
        points_list.append(edge_points[:-1])
    
    # Combine all points
    all_points = np.vstack(points_list)
    N = all_points.shape[0]
    
    print(f"Created {N} Gaussians along pyramid edges")
    
    # Convert to torch tensor
    gaussian_3d = torch.tensor(all_points, dtype=torch.float32, device=device)
    
    # Create Gaussian parameters
    gaussian_params = create_gaussian_params(N, device, uniform_color=False)
    
    return gaussian_3d, gaussian_params


def create_gaussian_params(N, device="cpu", uniform_color=True):
    """
    Create Gaussian parameters for N Gaussians.
    
    Args:
        N: Number of Gaussians
        device: torch device
        uniform_color: If True, use uniform red color; else random colors
    
    Returns:
        Dictionary with keys: 'rotation', 'scales', 'alpha', 'sh'
    """
    # Rotation (quaternions) - identity rotation
    # Quaternion format: [w, x, y, z]
    rotations = torch.zeros(N, 4, device=device)
    rotations[:, 0] = 1.0  # w=1 for identity rotation
    
    # Scales - small uniform spheres
    scales = torch.ones(N, 3, device=device) * 0.02  # Small Gaussians
    
    # Opacity - fully opaque (1D tensor)
    alphas = torch.rand((N,), device=device)
    
    # Spherical Harmonics coefficients
    # For sh_degree=3, we need (3+1)^2 = 16 coefficients per color channel
    # Total: 16 * 3 = 48 coefficients
    sh_degree = 3
    num_sh_coeffs = (sh_degree + 1) ** 2
    shs = torch.zeros(N, num_sh_coeffs * 3, device=device)
    
    if uniform_color:
        # Red color: DC component (first coefficient) controls base color
        # SH DC component is scaled, so we use a value that gives bright red
        shs[:, 0] = 0.5  # R channel DC
        shs[:, num_sh_coeffs] = -0.5  # G channel DC (negative to reduce green)
        shs[:, 2 * num_sh_coeffs] = -0.5  # B channel DC (negative to reduce blue)
    else:
        # Random colors
        shs[:, 0] = torch.rand(N, device=device) - 0.5  # R DC
        shs[:, num_sh_coeffs] = torch.rand(N, device=device) - 0.5  # G DC
        shs[:, 2 * num_sh_coeffs] = torch.rand(N, device=device) - 0.5  # B DC
    
    return {
        'rotation': rotations,
        'scales': scales,
        'alpha': alphas,
        'sh': shs
    }


def main():
    # Load config
    config_path = "configs/nlfgs_gpu.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Using default config path...")
        config_path = "configs/nlfgs_gpu.yaml"
    
    os.environ["NLFGS_CONFIG"] = config_path
    cfg = load_config(config_path)
    
    # Determine device
    device_str = cfg.get("sys", {}).get("device", "cpu") if isinstance(cfg, dict) else "cpu"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create pyramid Gaussians
    print("\n" + "="*60)
    print("Creating pyramid Gaussians...")
    print("="*60)
    gaussian_3d, gaussian_params = create_pyramid_gaussians(
        apex_height=1.0,
        base_size=1.0,
        num_points_per_edge=20,
        device=device
    )
    
    print(f"\nGaussian 3D centers shape: {gaussian_3d.shape}")
    print(f"Gaussian parameters:")
    for key, value in gaussian_params.items():
        print(f"  {key}: {value.shape}")
    
    # Initialize renderer
    print("\n" + "="*60)
    print("Initializing renderer...")
    print("="*60)
    renderer = GsplatRenderer()
    print(f"Renderer initialized with sh_degree={renderer.sh_degree}")
    
    # Define views to render (only front, back, left, right are supported)
    views = ['front', 'right', 'back', 'left']
    
    # Create output directory
    output_dir = "./output/renderer_test"

    # Save the gaussian centers as a point-cloud PLY for quick inspection in MeshLab/Blender
    ply_path = os.path.join(output_dir, "pyramid_gaussians.ply")
    # Use a uniform red color to match the SH setup
    ply_colors = torch.tensor([255, 0, 0], dtype=torch.uint8, device=gaussian_3d.device).view(1, 3).repeat(
        gaussian_3d.shape[0], 1
    )
    save_points_as_ply(gaussian_3d, ply_path, colors=ply_colors)
    print(f"Saved gaussian centers as PLY: {ply_path}")
    
    print("\n" + "="*60)
    print(f"Rendering {len(views)} views...")
    print("="*60)
    
    # Render the pyramid from multiple views
    try:
        rendered_images = renderer.render(
            gaussian_3d=gaussian_3d,
            gaussian_params=gaussian_params,
            view_name=views,
            camera_model="pinhole",
            save_folder_path=output_dir,
            render_mode="RGB",
            backgrounds=None  # White background
        )
        
        print(f"\n✓ Rendering successful!")
        print(f"  Output shape: {rendered_images.shape}")
        print(f"  Saved images to: {output_dir}/")
        print(f"  Views rendered: {', '.join(views)}")
        
        # Print statistics
        print("\nRendered image statistics:")
        print(f"  Min value: {rendered_images.min().item():.4f}")
        print(f"  Max value: {rendered_images.max().item():.4f}")
        print(f"  Mean value: {rendered_images.mean().item():.4f}")
        
    except Exception as e:
        print(f"\n✗ Rendering failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print("Test completed successfully! ✓")
    print("="*60)
    print(f"\nCheck the output images in: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    exit(main())
