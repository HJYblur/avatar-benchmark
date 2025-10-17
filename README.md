# Avatar Benchmark

A comprehensive baseline for full-body avatar evaluation using the following pipeline:

**Video → PyMAF-X → SMPL-X params → Mesh/Render → Metrics**

## Overview

This project provides a complete pipeline for evaluating full-body avatar reconstruction quality. It processes input videos to extract 3D human body parameters using PyMAF-X, generates SMPL-X meshes, renders them, and computes evaluation metrics.

## Pipeline Architecture

```
Input Video
    ↓
[Video Preprocessing] - Extract frames from video
    ↓
[PyMAF-X Inference] - Extract SMPL-X parameters
    ↓
[Mesh Generation] - Generate 3D meshes from SMPL-X params
    ↓
[Rendering] - Render meshes to images
    ↓
[Metrics Evaluation] - Compute quality metrics (PSNR, SSIM, LPIPS)
```

## Project Structure

```
avatar-benchmark/
├── configs/              # Configuration files
│   └── default.yaml     # Default pipeline configuration
├── data/                # Input data
│   ├── videos/          # Input video files
│   └── processed/       # Processed data
├── models/              # Model files
│   ├── smplx/          # SMPL-X model files
│   └── pymafx/         # PyMAF-X checkpoint
├── outputs/             # Pipeline outputs
│   ├── meshes/         # Generated meshes
│   ├── renders/        # Rendered images
│   └── metrics/        # Evaluation results
├── src/                 # Source code
│   ├── preprocessing/  # Video preprocessing modules
│   ├── inference/      # PyMAF-X inference wrapper
│   ├── rendering/      # Mesh generation and rendering
│   ├── metrics/        # Evaluation metrics
│   └── utils/          # Utility functions
├── scripts/             # Main pipeline scripts
│   └── run_pipeline.py # Main pipeline execution script
└── requirements.txt     # Python dependencies
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HJYblur/avatar-benchmark.git
cd avatar-benchmark
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download models

#### SMPL-X Models
1. Visit [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
2. Download SMPL-X model files
3. Place them in `models/smplx/`:
   - `SMPLX_NEUTRAL.npz`
   - `SMPLX_MALE.npz`
   - `SMPLX_FEMALE.npz`

#### PyMAF-X Model
1. Visit [https://github.com/HongwenZhang/PyMAF-X](https://github.com/HongwenZhang/PyMAF-X)
2. Download PyMAF-X checkpoint
3. Place it in `models/pymafx/`

## Usage

### Basic Usage

Run the complete pipeline on a video:

```bash
python scripts/run_pipeline.py --video data/videos/input.mp4 --output_name my_evaluation
```

### With Ground Truth

Evaluate against ground truth video:

```bash
python scripts/run_pipeline.py \
    --video data/videos/input.mp4 \
    --ground_truth data/videos/ground_truth.mp4 \
    --output_name my_evaluation
```

### Custom Configuration

Use a custom configuration file:

```bash
python scripts/run_pipeline.py \
    --video data/videos/input.mp4 \
    --config configs/custom.yaml \
    --output_name my_evaluation
```

## Configuration

Edit `configs/default.yaml` to customize the pipeline:

```yaml
# Model paths
SMPLX_MODEL_PATH: 'models/smplx'
PYMAFX_MODEL_PATH: 'models/pymafx'

# PyMAF-X settings
PYMAFX:
  DEVICE: 'cuda'  # or 'cpu'
  BATCH_SIZE: 1
  IMAGE_SIZE: 224

# Rendering settings
RENDERING:
  IMAGE_SIZE: [512, 512]
  CAMERA_DISTANCE: 2.5
  LIGHT_INTENSITY: 3.0

# Metrics to compute
METRICS:
  COMPUTE_PSNR: true
  COMPUTE_SSIM: true
  COMPUTE_LPIPS: true
```

## Output Format

The pipeline generates the following outputs:

```
outputs/<output_name>/
├── frames/              # Extracted video frames
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── params/              # SMPL-X parameters
│   └── smplx_params.npy
├── meshes/              # Generated 3D meshes
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

## Evaluation Metrics

The pipeline computes the following metrics:

- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality
- **SSIM** (Structural Similarity Index): Measures structural similarity
- **LPIPS** (Learned Perceptual Image Patch Similarity): Measures perceptual similarity
- **Mesh Error**: Point-to-point mesh distance metrics

## Modules

### Video Preprocessing (`src/preprocessing/`)
- Extract frames from videos
- Resize and preprocess frames

### PyMAF-X Inference (`src/inference/`)
- Wrapper for PyMAF-X model
- Extract SMPL-X parameters from images

### Mesh Generation (`src/rendering/`)
- Generate 3D meshes from SMPL-X parameters
- SMPL-X model integration

### Rendering (`src/rendering/`)
- Render 3D meshes to images
- Configurable camera and lighting

### Metrics (`src/metrics/`)
- Image quality metrics (PSNR, SSIM, LPIPS)
- Mesh error metrics

## References

- **SMPL-X**: [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
- **PyMAF-X**: [https://github.com/HongwenZhang/PyMAF-X](https://github.com/HongwenZhang/PyMAF-X)

## License

This project follows the licenses of the respective components:
- SMPL-X model license
- PyMAF-X license

## Citation

If you use this benchmark in your research, please cite the relevant papers for SMPL-X and PyMAF-X.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.