#!/usr/bin/env python3
"""
Check the actual scale of NLF-predicted SMPL-X vertices vs preprocessing.
This will help us understand if we need to scale cameras or the mesh.
"""

import numpy as np
import json
from pathlib import Path

def compare_scales():
    # THuman preprocessed (normalized)
    norm_path = Path("processed/0004/norm.json")
    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
    
    thuman_center = np.array(norm_data['center'])
    thuman_radius = norm_data['radius']
    
    print("="*60)
    print("SCALE COMPARISON")
    print("="*60)
    
    print("\n1. THuman Preprocessing:")
    print(f"   Original radius: {thuman_radius:.4f} m")
    print(f"   After normalization: fits in unit sphere (radius ≈ 0.76 m at worst)")
    print(f"   Normalization factor: 1/{thuman_radius:.4f} = {1/thuman_radius:.4f}")
    
    print("\n2. SMPL-X Template:")
    print(f"   Canonical T-pose radius: 1.2559 m")
    print(f"   About 2.13× larger than normalized THuman")
    
    print("\n3. Expected from NLF:")
    print(f"   NLF predicts SMPL-X shape/pose parameters")
    print(f"   SMPL-X vertices will be in 'natural' scale (meters)")
    print(f"   Likely similar to original THuman scale (~0.6-1.2 m radius)")
    
    print("\n4. Camera Coverage:")
    distance = 2.5
    yfov_rad = np.deg2rad(45.0)
    height_visible = 2 * distance * np.tan(yfov_rad / 2)
    
    print(f"   Camera distance: {distance} m")
    print(f"   FOV height: {height_visible:.4f} m")
    
    # Calculate how much of the FOV each mesh occupies
    thuman_norm_height = thuman_radius * 2 * (1/thuman_radius)  # After normalization
    thuman_orig_height = thuman_radius * 2
    smplx_height = 1.2559 * 2
    
    print(f"\n5. Avatar Size in Frame:")
    print(f"   THuman (original): {thuman_orig_height:.4f} m → {thuman_orig_height/height_visible*100:.1f}% of FOV")
    print(f"   THuman (normalized): {thuman_norm_height:.4f} m → {thuman_norm_height/height_visible*100:.1f}% of FOV") 
    print(f"   SMPL-X (canonical): {smplx_height:.4f} m → {smplx_height/height_visible*100:.1f}% of FOV")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    print("The preprocessing normalized the mesh, making it smaller.")
    print("But Gaussians are based on unnormalized SMPL-X, making them larger.")
    print("\nSOLUTION: Apply the same normalization to SMPL-X vertices!")
    print(f"  Scale Gaussians by: 1/{thuman_radius:.4f} = {1/thuman_radius:.4f}")
    print(f"  This will make them match the preprocessed scale.")

if __name__ == "__main__":
    compare_scales()
