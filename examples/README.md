# Example Scripts

This directory contains example shell scripts demonstrating different usage patterns of the avatar evaluation pipeline.

## Available Examples

### 1. Basic Pipeline (`run_basic_pipeline.sh`)
Runs the pipeline with default settings on a sample video.

```bash
bash examples/run_basic_pipeline.sh
```

**What it does:**
- Extracts frames from video
- Runs PyMAF-X inference
- Generates meshes
- Renders images
- Saves all outputs

### 2. Evaluation with Ground Truth (`run_with_evaluation.sh`)
Runs the pipeline and computes evaluation metrics against ground truth.

```bash
bash examples/run_with_evaluation.sh
```

**What it does:**
- All steps from basic pipeline
- Computes PSNR, SSIM, LPIPS metrics
- Generates evaluation report

### 3. High-Quality Rendering (`run_high_quality.sh`)
Uses high-quality configuration for best visual results.

```bash
bash examples/run_high_quality.sh
```

**What it does:**
- Renders at Full HD resolution (1920x1080)
- Uses enhanced lighting
- Processes all frames without skipping

### 4. Fast Test (`run_fast_test.sh`)
Quick test for debugging or rapid iteration.

```bash
bash examples/run_fast_test.sh
```

**What it does:**
- Lower resolution (256x256)
- Processes only first 50 frames
- Skips some metrics for speed

## Customization

To customize these examples:

1. **Change Input Video**: Modify the `VIDEO_PATH` variable
2. **Change Output Location**: Modify the `OUTPUT_NAME` variable
3. **Use Different Config**: Modify the `CONFIG` variable

Example:
```bash
VIDEO_PATH="data/videos/my_video.mp4"
OUTPUT_NAME="my_custom_evaluation"
CONFIG="configs/my_config.yaml"
```

## Creating Your Own Examples

Copy an existing example and modify it:

```bash
cp examples/run_basic_pipeline.sh examples/my_example.sh
chmod +x examples/my_example.sh
# Edit my_example.sh with your settings
bash examples/my_example.sh
```

## Batch Processing

To process multiple videos:

```bash
for video in data/videos/*.mp4; do
    basename=$(basename "$video" .mp4)
    python scripts/run_pipeline.py \
        --video "$video" \
        --output_name "$basename"
done
```

## Notes

- Make sure to set up models before running (see main README.md)
- Ensure input videos exist in the specified paths
- Check `outputs/` directory for results
