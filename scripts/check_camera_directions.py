#!/usr/bin/env python3
"""
Test to check if gsplat and pyrender have the same coordinate conventions.
We'll check if flipping Y or negating Z fixes the mismatch.
"""

import numpy as np
import json

def check_view_direction(view_name):
    """Check what direction the camera is looking."""
    with open(f'data/THuman_cameras/thuman_{view_name}.json', 'r') as f:
        data = json.load(f)
    
    viewmat = np.array(data['viewmat'])  # w2c
    c2w = np.linalg.inv(viewmat)
    
    cam_pos = c2w[:3, 3]
    
    # In OpenGL convention, camera looks down -Z axis in camera space
    # So the viewing direction in world space is: -c2w[:, 2]
    view_dir_world = -c2w[:3, 2]
    
    # Forward should point toward origin
    forward_to_origin = -cam_pos / np.linalg.norm(cam_pos)
    
    print(f"\n{view_name.upper()}:")
    print(f"  Camera position: {cam_pos}")
    print(f"  View direction (from c2w): {view_dir_world}")
    print(f"  Should point toward origin: {forward_to_origin}")
    print(f"  Match? {np.allclose(view_dir_world, forward_to_origin, atol=1e-6)}")
    
    # Check camera axes
    cam_right = c2w[:3, 0]
    cam_up = c2w[:3, 1]
    cam_back = c2w[:3, 2]
    
    print(f"  Camera right (+X): {cam_right}")
    print(f"  Camera up (+Y):    {cam_up}")
    print(f"  Camera back (+Z):  {cam_back}")

print("="*60)
print("CHECKING GSPLAT CAMERA ORIENTATIONS")
print("="*60)

for view in ['front', 'back', 'left', 'right']:
    check_view_direction(view)

print("\n" + "="*60)
print("EXPECTED FOR OPENGL/PYRENDER:")
print("="*60)
print("Camera looks down -Z axis, so viewing direction = -c2w[:, 2]")
print("This should point from camera position toward origin (0,0,0)")
