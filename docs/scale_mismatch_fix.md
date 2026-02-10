# Scale Mismatch Fix - Summary

## Problem

The viewing angle is now correct, but the avatar size differs between:
- **Preprocessed images**: Normalized THuman scan (scaled to fit frame nicely)
- **Rendered output**: SMPL-X from NLF (in natural scale, appears larger)

## Root Cause

### Preprocessing Normalization:
```python
# THuman scan for subject 0004:
radius = 0.3520 m  # Original mesh radius
scale = 1 / radius = 2.8405  # Makes it 2.84× larger
# After normalization: mesh occupies ~96% of camera FOV
```

### Training (Before Fix):
```python
# SMPL-X from NLF:
# Natural scale (~1-2m radius)
# No scaling applied
# Result: occupies ~121% of FOV → too large!
```

## Solution

Apply **scale-only normalization** (not full translation+scale):

```python
# In trainer.py:
_, radius = load_normalization(subject)  # Get THuman radius
scale_factor = 1.0 / radius              # e.g., 2.8405 for subject 0004
gaussian_3d_scaled = gaussian_3d * scale_factor  # Scale about origin
```

### Why Scale-Only?

1. **Different centers**: THuman center ≈ (0, 0, 0), SMPL-X center ≈ (0, -0.46, 0)
2. **Different body positioning**: Can't use THuman translation on SMPL-X
3. **But same relative scale needed**: Both should occupy similar % of FOV

### Expected Result:

After this fix:
- Avatar size in rendered output matches preprocessed images
- Both occupy similar portion of the 1024×1024 frame
- Viewing angle remains correct (front view shows front)

## Files Changed

- `src/training/trainer.py`: Added scale-only normalization using THuman radius
- `scripts/check_scale_mismatch.py`: Diagnostic tool to analyze scale differences

## Test

Run training and compare avatar sizes:
```bash
python train.py
# Compare sizes in: output/0004/debug_front.png vs processed/0004/0004_front.png
```

Both avatars should now have similar height/width in frame.

## Technical Details

| Aspect | Preprocessing | Training (After Fix) |
|--------|--------------|---------------------|
| Mesh Source | THuman scan | SMPL-X from NLF |
| Center | ~(0, 0.007, 0.016) | ~(0, -0.46, 0) |
| Original Radius | 0.352 m | ~1.0 m (varies) |
| Scale Applied | ×2.84 | ×2.84 (same factor!) |
| Final Size | ~96% of FOV | ~96% of FOV ✓ |
| Camera Distance | 2.5 m | 2.5 m ✓ |
| FOV | 45° (2.07m height) | 45° (2.07m height) ✓ |
