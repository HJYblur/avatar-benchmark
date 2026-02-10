# Camera Rendering Mismatch - Root Cause Analysis

## Problem Summary
The gsplat rendering output shows a different viewing angle than the preprocessed input images, even though:
- Camera matrices are verified correct
- Coordinate systems (OpenGL) match between pyrender and gsplat  
- Camera positions are identical (distance=2.5, looking at origin)

## Root Cause Hypothesis

The issue is **NOT with cameras**, but with **coordinate frame mismatch between two different meshes**:

### Two Different Meshes in Play:

1. **Preprocessing (THuman scanned mesh)**:
   - Original OBJ file from THuman 2.0 dataset
   - Has its own bbox/center/orientation
   - Gets normalized: `(mesh - center) * (1/radius)`
   - Rendered with pyrender → creates training images

2. **Training (SMPL-X parametric mesh)**:
   - NLF backbone outputs SMPL-X parameters
   - SMPL-X creates a parametric body mesh
   - Gaussian positions computed via barycentric interpolation on SMPL-X vertices
   - These vertices are in **SMPL-X canonical space**, NOT THuman scan space!

### The Mismatch:
- **Preprocessing normalization** transforms the *scanned mesh* to origin
- **Gaussians** are positioned based on *SMPL-X vertices* which have a **different canonical frame**
- Even if we apply the same normalization (center, radius) to Gaussians, it won't align because:
  - SMPL-X canonical pose ≠ THuman scan pose
  - SMPL-X orientation ≠ THuman scan orientation  
  - The bbox center/radius computed from scan doesn't apply to SMPL-X

## What's Actually Happening:

```
Preprocessing:
THuman scan → compute bbox → normalize → render with camera → training images

Training:
Input images → NLF → SMPL-X params → SMPL-X mesh → Gaussian positions (in SMPL-X space)
→ Apply THuman normalization (WRONG! Different space!) → Render with camera → mismatch!
```

## The Real Fix Needed:

### Option 1: No Normalization (Simpler)
- Don't normalize meshes in preprocessing
- Don't normalize Gaussians in training  
- Use cameras that account for varying subject sizes (adjust distance based on bbox)
- OR: Ensure SMPL-X and scanned mesh are in same scale/position

### Option 2: Normalize SMPL-X Output (Complex)
- During training, get SMPL-X vertices from NLF
- Compute bbox of SMPL-X mesh
- Normalize Gaussians using SMPL-X bbox (not scan bbox)
- This ensures Gaussians are in same canonical frame as expected

### Option 3: Register Scan to SMPL-X (Most Correct)
- During preprocessing, fit SMPL-X to the scanned mesh
- Use SMPL-X normalization parameters for both scan rendering AND Gaussian positioning
- Ensures perfect alignment between training data and model output

## Immediate Diagnostic:

Check if NLF outputs are in expected range:
```python
# In training_step, before normalization:
print(f"SMPL-X vertices range: {pred['vertices3d'][0][0].min()} to {pred['vertices3d'][0][0].max()}")
print(f"Gaussian centers range: {gaussian_3d[0].min(dim=0)[0]} to {gaussian_3d[0].max(dim=0)[0]}")
print(f"Expected after THuman norm: should be roughly [-1, 1] if normalized to unit sphere")
```

## Why Previous Analysis Was Incomplete:

We verified cameras were correct, but didn't check if the 3D geometry (Gaussians) was in the right coordinate frame to match those cameras. The cameras are looking at origin expecting normalized meshes, but the Gaussians might not be centered/scaled correctly because they come from a different mesh (SMPL-X vs THuman scan).
