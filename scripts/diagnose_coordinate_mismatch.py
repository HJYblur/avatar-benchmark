#!/usr/bin/env python3
"""
Diagnostic script to check coordinate frame mismatch between:
1. THuman scanned mesh (used in preprocessing)
2. SMPL-X mesh (output by NLF, used for Gaussian positions)
"""

import numpy as np
import trimesh
from pathlib import Path
import json

# Load THuman scanned mesh
def check_thuman_mesh(identity="0010"):
    mesh_path = Path(f"data/THuman_2.0/{identity}/{identity}.obj")
    norm_path = Path(f"processed/{identity}/norm.json")
    
    if not mesh_path.exists():
        print(f"Mesh not found: {mesh_path}")
        return
    
    mesh = trimesh.load(str(mesh_path), process=False)
    vertices = np.array(mesh.vertices)
    
    # Compute bbox
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    radius = np.linalg.norm(bbox_max - bbox_min) / 2.0
    
    print(f"{'='*60}")
    print(f"THUMAN SCANNED MESH ({identity})")
    print(f"{'='*60}")
    print(f"Vertices shape: {vertices.shape}")
    print(f"Bbox min: {bbox_min}")
    print(f"Bbox max: {bbox_max}")
    print(f"Center: {center}")
    print(f"Radius: {radius:.4f}")
    print(f"Size (bbox diagonal): {np.linalg.norm(bbox_max - bbox_min):.4f}")
    
    # Check normalized
    vertices_norm = (vertices - center) / radius
    print(f"\nAfter normalization:")
    print(f"  Min: {vertices_norm.min(axis=0)}")
    print(f"  Max: {vertices_norm.max(axis=0)}")
    print(f"  Center: {vertices_norm.mean(axis=0)}")
    
    # Load saved normalization
    if norm_path.exists():
        with open(norm_path, 'r') as f:
            norm_data = json.load(f)
        print(f"\nSaved normalization (norm.json):")
        print(f"  Center: {norm_data['center']}")
        print(f"  Radius: {norm_data['radius']:.4f}")
        print(f"  Match? {np.allclose(center, norm_data['center']) and np.isclose(radius, norm_data['radius'])}")
    
def check_smplx_template():
    """Check SMPL-X template mesh to understand its canonical frame."""
    # Try to load SMPL-X UV template
    template_path = Path("models/smplx/smplx_uv.obj")
    
    if not template_path.exists():
        print(f"\nSMPL-X template not found at: {template_path}")
        print("Cannot check SMPL-X canonical frame")
        return
    
    mesh = trimesh.load(str(template_path), process=False)
    vertices = np.array(mesh.vertices)
    
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    radius = np.linalg.norm(bbox_max - bbox_min) / 2.0
    
    print(f"\n{'='*60}")
    print(f"SMPL-X TEMPLATE MESH")
    print(f"{'='*60}")
    print(f"Vertices shape: {vertices.shape}")
    print(f"Bbox min: {bbox_min}")
    print(f"Bbox max: {bbox_max}")
    print(f"Center: {center}")
    print(f"Radius: {radius:.4f}")
    print(f"Size (bbox diagonal): {np.linalg.norm(bbox_max - bbox_min):.4f}")
    
    print(f"\nKey Observations:")
    print(f"  - SMPL-X is typically centered around origin in T-pose")
    print(f"  - Y-axis is up (pelvis at origin, head at +Y)")
    print(f"  - Arms extend along X-axis")
    print(f"  - Facing direction is +Y or +Z depending on convention")

if __name__ == "__main__":
    check_thuman_mesh("0010")
    check_smplx_template()
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS")
    print(f"{'='*60}")
    print(f"If THuman center ≠ SMPL-X center, applying THuman normalization")
    print(f"to SMPL-X-based Gaussians will shift them to wrong position!")
    print(f"\nSolution: Use SMPL-X bbox for normalization, not THuman bbox.")
