#!/bin/bash
# Example: Run high-quality rendering

VIDEO_PATH="data/videos/sample.mp4"
CONFIG="configs/high_quality.yaml"
OUTPUT_NAME="high_quality_render"

python scripts/run_pipeline.py \
    --video ${VIDEO_PATH} \
    --config ${CONFIG} \
    --output_name ${OUTPUT_NAME}

echo "High-quality renders saved to: outputs/${OUTPUT_NAME}/renders/"
