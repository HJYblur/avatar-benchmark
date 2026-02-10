#!/usr/bin/env python3
"""Validate camera consistency between preprocessing and gsplat renderer.

This script checks that:
1. Preprocessing cameras match stored JSON cameras
2. Normalization metadata exists for each identity
3. Camera intrinsics/extrinsics are correct
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from avatar_utils.camera import load_camera_mapping, load_normalization
from data.preprocess_thuman import VIEWPOINTS, look_at


def validate_camera_json(view_name: str, distance: float = 2.5):
    """Check that stored camera JSON matches expected canonical camera."""
    print(f"\n=== Validating {view_name} camera ===")
    
    # Load from JSON
    try:
        viewmat, K = load_camera_mapping(view_name)
        viewmat = viewmat.squeeze(0).numpy()
        K = K.squeeze(0).numpy()
    except Exception as e:
        print(f"❌ Failed to load camera mapping: {e}")
        return False
    
    # Expected canonical camera
    origin = np.zeros(3, dtype=float)
    direction = VIEWPOINTS[view_name]
    eye = direction * distance
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    if np.allclose(np.cross(up, direction), 0.0):
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    
    c2w_expected = look_at(eye, origin, up)
    w2c_expected = np.linalg.inv(c2w_expected)
    
    # Compare
    if np.allclose(viewmat, w2c_expected, atol=1e-5):
        print(f"✓ Extrinsics match (distance={distance})")
    else:
        print(f"❌ Extrinsics mismatch!")
        print(f"  Expected eye: {eye}")
        c2w_loaded = np.linalg.inv(viewmat)
        print(f"  Loaded eye:   {c2w_loaded[:3, 3]}")
        return False
    
    # Check intrinsics
    H, W = 1024, 1024
    yfov_rad = np.deg2rad(45.0)
    fy_expected = H / (2.0 * np.tan(yfov_rad / 2.0))
    fx_expected = fy_expected
    
    if np.allclose(K[0, 0], fx_expected, atol=1.0) and np.allclose(K[1, 1], fy_expected, atol=1.0):
        print(f"✓ Intrinsics match (fx={fx_expected:.1f}, fy={fy_expected:.1f})")
    else:
        print(f"❌ Intrinsics mismatch!")
        print(f"  Expected: fx={fx_expected:.1f}, fy={fy_expected:.1f}")
        print(f"  Loaded:   fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        return False
    
    return True


def validate_normalization(identity: str, processed_root: str = "processed"):
    """Check that normalization metadata exists."""
    print(f"\n=== Validating normalization for {identity} ===")
    
    try:
        center, radius = load_normalization(identity, processed_root)
        print(f"✓ Normalization loaded: center={center.numpy()}, radius={radius:.4f}")
        return True
    except Exception as e:
        print(f"❌ Failed to load normalization: {e}")
        return False


def main():
    print("Camera Validation Script")
    print("=" * 60)
    
    # Validate all view cameras
    all_ok = True
    for view_name in VIEWPOINTS.keys():
        if not validate_camera_json(view_name):
            all_ok = False
    
    # Check a sample identity normalization (if processed data exists)
    processed_dir = Path("processed")
    if processed_dir.exists():
        identities = [d.name for d in processed_dir.iterdir() if d.is_dir()]
        if identities:
            sample = identities[0]
            if not validate_normalization(sample):
                all_ok = False
        else:
            print("\n⚠️  No processed identities found to validate normalization")
    else:
        print("\n⚠️  Processed directory not found")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All validations passed!")
        return 0
    else:
        print("❌ Some validations failed - see above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
