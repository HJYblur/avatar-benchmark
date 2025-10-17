# Pipeline Architecture

## Full Pipeline Flow

```
┌─────────────────────────┐
│    Input Video File     │
│   (data/videos/*.mp4)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Video Preprocessing    │
│  - Extract frames       │
│  - Resize (optional)    │
│  (VideoPreprocessor)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   PyMAF-X Inference     │
│  - Load frames          │
│  - Extract SMPL-X       │
│    parameters           │
│  (PyMAFXInference)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    SMPL-X Parameters    │
│  - Body shape (betas)   │
│  - Body pose            │
│  - Hand poses           │
│  - Expression           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Mesh Generation       │
│  - Create SMPL-X mesh   │
│  - Export to OBJ        │
│  (SMPLXMeshGenerator)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│      3D Meshes          │
│  (outputs/meshes/*.obj) │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Mesh Rendering        │
│  - Setup camera/lights  │
│  - Render to images     │
│  (MeshRenderer)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Rendered Images       │
│ (outputs/renders/*.png) │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Metrics Evaluation     │
│  - PSNR, SSIM, LPIPS   │
│  - Mesh errors          │
│  (MetricsCalculator)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Evaluation Results    │
│ (outputs/metrics/*.txt) │
└─────────────────────────┘
```

## Module Breakdown

### 1. Video Preprocessing (`src/preprocessing/`)

**Input:** Video file (MP4, AVI, etc.)  
**Output:** Extracted frames (PNG)

- Extracts frames at specified FPS
- Supports frame skipping for efficiency
- Resizes frames to target resolution

### 2. PyMAF-X Inference (`src/inference/`)

**Input:** Image frames  
**Output:** SMPL-X parameters (NPY)

- Loads pretrained PyMAF-X model
- Extracts full-body parameters including:
  - Shape parameters (betas)
  - Body pose (21 joints)
  - Hand poses (left/right)
  - Facial parameters (jaw, eyes, expression)

### 3. Mesh Generation (`src/rendering/`)

**Input:** SMPL-X parameters  
**Output:** 3D meshes (OBJ)

- Instantiates SMPL-X model
- Generates mesh vertices and faces
- Exports to standard 3D formats

### 4. Rendering (`src/rendering/`)

**Input:** 3D meshes  
**Output:** Rendered images (PNG)

- Configurable camera parameters
- Adjustable lighting conditions
- Background overlay support

### 5. Metrics Evaluation (`src/metrics/`)

**Input:** Rendered images + Ground truth  
**Output:** Evaluation metrics (TXT)

Supported metrics:
- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index
- **LPIPS** - Learned Perceptual Image Patch Similarity
- **Mesh Error** - Point-to-point distance metrics

## Configuration

The pipeline is controlled via `configs/default.yaml`:

```yaml
# Model paths
SMPLX_MODEL_PATH: 'models/smplx'
PYMAFX_MODEL_PATH: 'models/pymafx'

# Processing settings
PROCESSING:
  FPS: 30
  MAX_FRAMES: null      # null = all frames
  FRAME_SKIP: 1         # Process every Nth frame

# Rendering settings
RENDERING:
  IMAGE_SIZE: [512, 512]
  CAMERA_DISTANCE: 2.5
  LIGHT_INTENSITY: 3.0
  BACKGROUND_COLOR: [0, 0, 0]

# Metrics
METRICS:
  COMPUTE_PSNR: true
  COMPUTE_SSIM: true
  COMPUTE_LPIPS: true
  COMPUTE_MESH_ERROR: true
```

## Usage Examples

### Basic Pipeline

```bash
python scripts/run_pipeline.py \
    --video data/videos/input.mp4 \
    --output_name basic_eval
```

### With Evaluation

```bash
python scripts/run_pipeline.py \
    --video data/videos/input.mp4 \
    --ground_truth data/videos/gt.mp4 \
    --output_name eval_with_metrics
```

### Custom Configuration

```bash
python scripts/run_pipeline.py \
    --video data/videos/input.mp4 \
    --config configs/custom.yaml \
    --output_name custom_eval
```

## Output Structure

```
outputs/<output_name>/
├── frames/              # Extracted frames
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── params/              # SMPL-X parameters
│   └── smplx_params.npy
├── meshes/              # 3D meshes
│   ├── mesh_000000.obj
│   ├── mesh_000001.obj
│   └── ...
├── renders/             # Rendered images
│   ├── render_000000.png
│   ├── render_000001.png
│   └── ...
└── metrics/             # Evaluation results
    └── evaluation_results.txt
```
