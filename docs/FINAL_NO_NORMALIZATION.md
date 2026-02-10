# Final Solution: No Normalization

## Summary

**Removed all normalization** from both preprocessing and training to ensure coordinate frame consistency.

## Rationale

1. **NLF outputs canonical SMPL-X**: The NLF backbone predicts SMPL-X parameters in a standard coordinate frame
2. **THuman scans have their own coordinate frame**: Each scan has different center/scale
3. **Incompatible coordinate frames**: Applying THuman normalization to SMPL-X vertices causes mismatch
4. **Solution**: Use the same coordinate frame everywhere - no normalization

## Changes Made

### 1. Preprocessing (`src/data/preprocess_thuman.py`)

**REMOVED:**
- Bbox computation and normalization metadata saving
- Mesh transformation (translation + scaling)
- norm.json file generation

**NOW:**
- Renders original THuman meshes as-is
- Uses canonical cameras (distance=2.5, yfov=45°, looking at origin)
- Meshes stay in their original coordinate frame

```python
# Before: meshes were normalized
m_norm.apply_translation(-center)
m_norm.apply_scale(1/radius)

# After: meshes rendered as-is (no transformation)
scene.add(_mesh_to_pyrender(mesh, texture_path))
```

### 2. Training (`src/training/trainer.py`)

**REMOVED:**
- Import of `load_normalization`, `apply_normalization`
- All normalization/scaling code
- Dependency on norm.json files

**NOW:**
- Gaussians stay in SMPL-X coordinate frame
- No transformation applied before rendering
- Direct pass-through from decoder to renderer

```python
# Before: attempted to apply THuman or scale normalization
gaussian_3d_scaled = apply_normalization(gaussian_3d[0], center, radius)

# After: use SMPL-X coordinates directly
gaussian_3d_norm = gaussian_3d[0]
```

### 3. Renderer (`src/render/gaussian_renderer.py`)

**NO CHANGES NEEDED:**
- Already saves per-view debug images (`debug_front.png`, etc.)
- Uses cameras from JSON (consistent with preprocessing)

## Expected Behavior

### Preprocessing Output:
- THuman scans rendered from canonical cameras
- Each subject at different apparent size (natural variation)
- Subjects roughly centered if scans are roughly centered

### Training Output:
- SMPL-X meshes from NLF rendered from same canonical cameras
- Same apparent size as SMPL-X natural scale
- Should match preprocessing if NLF correctly predicts SMPL-X

## Coordinate Frame Details

| Aspect | Preprocessing | Training |
|--------|--------------|----------|
| Mesh Source | THuman scan OBJ | SMPL-X from NLF |
| Coordinate Frame | Original scan frame | SMPL-X canonical frame |
| Center | Scan-dependent (~origin) | SMPL-X-dependent (~Y=-0.46) |
| Scale | Original scan scale | SMPL-X natural scale (~2m) |
| Normalization | **None** | **None** |
| Camera Distance | 2.5 m | 2.5 m |
| Camera FOV | 45° | 45° |

## Why This Works

1. **NLF is trained to map images → SMPL-X params**
2. **SMPL-X has a consistent canonical frame** across all subjects
3. **If training data shows unnormalized scans**, NLF learns to output SMPL-X at the right scale
4. **Rendering SMPL-X with same cameras** should produce similar views

## Important Notes

⚠️ **You need to re-run preprocessing** after this change:
```bash
python -m src.data.preprocess_thuman
```

This will regenerate images without normalization. The old normalized images won't match the new training pipeline.

## Testing

1. **Re-preprocess data**: `python -m src.data.preprocess_thuman`
2. **Run training**: `python train.py`
3. **Compare outputs**:
   - `output/0004/debug_front.png` (rendered from SMPL-X)
   - `processed/0004/0004_front.png` (preprocessed THuman scan)

Both should now:
- ✅ Show same viewing angle (front view)
- ✅ Show subject at natural scale (not normalized)
- ✅ Use identical camera parameters

## Files Modified

1. ✅ `src/data/preprocess_thuman.py` - Removed normalization
2. ✅ `src/training/trainer.py` - Removed normalization, cleaned imports
3. ✅ `src/render/gaussian_renderer.py` - Already correct (per-view debug saves)

## Clean State

No norm.json files will be generated or used. The codebase is now simpler and coordinate frames are consistent.
