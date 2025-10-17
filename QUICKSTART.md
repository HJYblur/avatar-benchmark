# Quick Start Guide

This guide will help you get started with the avatar evaluation pipeline.

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

## Installation Steps

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/HJYblur/avatar-benchmark.git
cd avatar-benchmark

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

#### SMPL-X Models
1. Register at [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
2. Download SMPL-X v1.1 models
3. Extract and place files:
   ```
   models/smplx/
   ├── SMPLX_NEUTRAL.npz
   ├── SMPLX_MALE.npz
   └── SMPLX_FEMALE.npz
   ```

#### PyMAF-X Checkpoint
1. Visit [https://github.com/HongwenZhang/PyMAF-X](https://github.com/HongwenZhang/PyMAF-X)
2. Download the pretrained checkpoint
3. Place in `models/pymafx/checkpoint.pth`

### 3. Prepare Data

Place your input video in the data directory:
```bash
cp /path/to/your/video.mp4 data/videos/
```

### 4. Verify Installation

Run the structure validation test:
```bash
python scripts/test_structure.py
```

You should see:
```
✓ All checks passed!
```

### 5. Run the Pipeline

Basic usage:
```bash
python scripts/run_pipeline.py --video data/videos/your_video.mp4 --output_name test_run
```

With ground truth for evaluation:
```bash
python scripts/run_pipeline.py \
    --video data/videos/input.mp4 \
    --ground_truth data/videos/ground_truth.mp4 \
    --output_name evaluation_001
```

### 6. View Results

Results are saved in `outputs/<output_name>/`:
- `frames/` - Extracted video frames
- `params/` - SMPL-X parameters
- `meshes/` - Generated 3D meshes (OBJ format)
- `renders/` - Rendered images
- `metrics/` - Evaluation results (if ground truth provided)

## Configuration

Customize the pipeline by editing `configs/default.yaml`:

```yaml
# Example: Change rendering resolution
RENDERING:
  IMAGE_SIZE: [1024, 1024]  # Higher resolution
  CAMERA_DISTANCE: 3.0       # Further camera
  LIGHT_INTENSITY: 4.0       # Brighter lighting
```

Then run with custom config:
```bash
python scripts/run_pipeline.py \
    --video data/videos/input.mp4 \
    --config configs/default.yaml \
    --output_name high_res_test
```

## Troubleshooting

### GPU/CUDA Issues
If CUDA is not available, the pipeline will automatically use CPU:
```yaml
PYMAFX:
  DEVICE: 'cpu'  # Force CPU usage
```

### Memory Issues
Reduce memory usage:
```yaml
PROCESSING:
  MAX_FRAMES: 100    # Limit number of frames
  FRAME_SKIP: 2      # Process every 2nd frame
```

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

1. Check the [main README](README.md) for detailed documentation
2. Explore the example script: `python scripts/example.py`
3. Modify the configuration for your specific use case
4. Integrate the pipeline into your own projects

## Getting Help

- Check the [README](README.md) for detailed information
- Review the source code in `src/` for implementation details
- Open an issue on GitHub for bugs or feature requests
