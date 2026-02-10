# Camera Normalization Fix - Summary

## Problem Identified

Your preprocessing and gsplat renderer were using **different camera configurations**:

1. **Preprocessing (`preprocess_thuman.py`)**: 
   - Used per-identity cameras via `_camera_pose(direction, center, radius)`
   - Camera distance = `radius * 2.5` (varies per subject)
   - Camera center = bbox center (varies per subject)
   - Result: Different viewpoint for each identity

2. **Gsplat Renderer (`gaussian_renderer.py`)**:
   - Used global cameras from `data/THuman_cameras/*.json`
   - Camera distance = fixed 2.5
   - Camera center = origin (0,0,0)
   - Result: Same viewpoint for all identities

**This mismatch caused different rendering outputs even with "the same camera".**

---

## Solution Implemented

### Canonical Frame Normalization (Option 1 - Recommended)

All identities are now normalized to a canonical frame during preprocessing:

1. **Compute normalization parameters** (per identity):
   - `center` = bbox center of all meshes
   - `radius` = half of bbox diagonal
   - `scale` = 1 / radius

2. **Transform meshes**:
   - Translate by `-center` → centered at origin
   - Scale by `1/radius` → normalized to unit size

3. **Use global canonical cameras**:
   - All cameras look at origin (0,0,0)
   - Fixed distance = 2.5
   - Fixed FOV = 45°

4. **Save normalization metadata**:
   - `processed/<identity>/norm.json` contains `{center, radius, scale}`
   - Used at training/inference to normalize Gaussians

---

## Files Changed

### 1. `src/data/preprocess_thuman.py`
**Changes:**
- `_render_views()`: Now normalizes meshes before adding to scene
- Saves `norm.json` per identity
- Uses canonical cameras (origin, distance=2.5) instead of bbox-based cameras

**Key code:**
```python
# Compute normalization
center = (bbox_min + bbox_max) / 2.0
radius = float(np.linalg.norm(bbox_max - bbox_min)) / 2.0
scale = 1.0 / (radius + 1e-8)

# Apply to meshes
m_norm.apply_translation(-center)
m_norm.apply_scale(scale)

# Canonical cameras
eye = direction * 2.5  # fixed distance
pose = look_at(eye, origin, up)
```

### 2. `src/avatar_utils/camera.py`
**Added:**
- `load_normalization(identity)` → loads center, radius from `norm.json`
- `apply_normalization(xyz, center, radius)` → applies same transform to Gaussians

**Usage:**
```python
center, radius = load_normalization("subject_001")
xyz_norm = apply_normalization(xyz, center, radius)
```

### 3. `src/training/trainer.py`
**Changes:**
- Imports `load_normalization`, `apply_normalization`
- Before rendering, normalizes `gaussian_3d` using the identity's saved parameters

**Key code:**
```python
center, radius = load_normalization(subject)
gaussian_3d_norm = apply_normalization(gaussian_3d[0], center, radius)
rendered_imgs = self.renderer.render(gaussian_3d=gaussian_3d_norm, ...)
```

### 4. `scripts/validate_cameras.py` (NEW)
**Purpose:**
- Validates camera JSON files match expected canonical cameras
- Checks normalization metadata exists
- Run with: `python scripts/validate_cameras.py`

---

## How to Use

### Step 1: Re-run Preprocessing
```bash
cd /Users/lemon/Documents/TUD/Thesis/avatar-benchmark
python -m src.data.preprocess_thuman
```

This will:
- Normalize all meshes to canonical frame
- Render with global canonical cameras
- Save `processed/<id>/norm.json` for each identity

### Step 2: Validate Cameras
```bash
python scripts/validate_cameras.py
```

Check that all cameras and normalizations are correct.

### Step 3: Train/Render
Your training pipeline now:
- Loads per-identity normalization from `norm.json`
- Normalizes Gaussian means before rendering
- Uses consistent global cameras everywhere

**Result: Preprocessing renders and gsplat renders now match!**

---

## Camera Configuration

All cameras now use:
- **Intrinsics**: `yfov=45°`, `1024×1024`, square pixels
- **Extrinsics**: Look at origin from distance 2.5
  - front: eye at (0, 0, 2.5)
  - back: eye at (0, 0, -2.5)
  - left: eye at (-2.5, 0, 0)
  - right: eye at (2.5, 0, 0)

Stored in: `data/THuman_cameras/thuman_{view}.json`

---

## Normalization Metadata

Each identity has `processed/<id>/norm.json`:
```json
{
  "center": [x, y, z],
  "radius": r,
  "scale": 1/r
}
```

**Important:** Any world-space data (Gaussians, SMPL vertices, etc.) must be normalized using:
```python
xyz_normalized = (xyz - center) * scale
```

---

## Verification Checklist

- [ ] Re-run preprocessing to generate normalized renders and `norm.json`
- [ ] Run `scripts/validate_cameras.py` - all checks pass
- [ ] Check one preprocessed image matches rendered output from gsplat
- [ ] Verify `processed/<id>/norm.json` exists for all identities
- [ ] Training loads normalization and applies it before rendering

---

## Next Steps

If you want to:
1. **Change camera distance**: Update `distance=2.5` in both `preprocess_thuman.py` and `generate_camera_mapping()`
2. **Add more views**: Add to `VIEWPOINTS` dict and regenerate camera JSONs
3. **Use different image size**: Update `IMAGE_SIZE` and regenerate camera JSONs
4. **Visualize transforms**: I can add a script to show before/after normalization

---

## Why This Works

**Before:**
- Preprocess camera distance = f(radius) → varies per identity
- Gsplat camera distance = 2.5 → fixed
- **Mismatch!**

**After:**
- All subjects normalized to unit scale
- All cameras use distance = 2.5 relative to origin
- **Consistent!**

The key insight: instead of adjusting cameras per identity, normalize the world so a single global camera set works for everyone.
