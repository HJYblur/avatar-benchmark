# Coordinate Frame Mismatch - SOLUTION

## Root Cause (CONFIRMED)

THuman scanned mesh and SMPL-X have **completely different coordinate frames**:

| Property | THuman Scan (0010) | SMPL-X Template |
|----------|-------------------|-----------------|
| Center Y | +0.045 m | **-0.462 m** |
| Radius | 0.590 m | **1.256 m** |
| Size | 1.180 m | **2.512 m** |

When applying THuman normalization `(xyz - center_thuman) / radius_thuman` to SMPL-X-based Gaussians:
- The Gaussians are shifted and scaled incorrectly
- They end up in wrong positions relative to the cameras
- Result: Different viewing angles despite identical camera matrices

## Solution Implemented

**Removed THuman normalization from training** (in `src/training/trainer.py`):
```python
# Do NOT apply THuman normalization to Gaussians
# Gaussians are based on SMPL-X mesh which has different bbox
gaussian_3d_norm = gaussian_3d[0]  # Use SMPL-X coordinates directly
```

## Next Steps

### Option A: Remove Normalization from Preprocessing (RECOMMENDED)
1. Update `preprocess_thuman.py` to NOT normalize meshes
2. Adjust camera distance based on each mesh's bbox: `distance = radius * 2.5`
3. Cameras will be at different distances per subject, but all will frame the subject similarly

### Option B: Use SMPL-X Normalization Everywhere
1. During preprocessing, fit SMPL-X to each scan
2. Save SMPL-X bbox as normalization parameters
3. Apply same SMPL-X normalization to both scans (preprocessing) and Gaussians (training)

### Option C: Keep Current Setup (QUICK FIX - Current Choice)
1. Leave preprocessing as-is (normalized scans)
2. Don't normalize Gaussians (they stay in SMPL-X space)
3. Accept that preprocessing uses normalized scans while training uses SMPL-X scale
4. This works if NLF correctly maps between the two spaces

## Why Option C Works

The NLF backbone is trained to:
- Input: Images of normalized THuman scans
- Output: SMPL-X parameters

The Gaussians positioned on SMPL-X vertices should already be in a scale/position that makes sense when rendered, because:
1. NLF learns to output SMPL-X params that match the input images
2. SMPL-X vertices are in a standard coordinate frame
3. Cameras can render SMPL-X directly without additional normalization

## Testing

Run training and check if debug images now match input views:
```bash
python train.py
# Check output/[subject]/debug_front.png vs processed/[subject]/[subject]_front.png
```

If views still don't match, we need Option A or B.
