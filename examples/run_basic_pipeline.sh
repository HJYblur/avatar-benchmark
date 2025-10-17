#!/bin/bash
# Example: Run basic pipeline on a sample video

VIDEO_PATH="data/videos/sample.mp4"
OUTPUT_NAME="basic_evaluation"

python scripts/run_pipeline.py \
    --video ${VIDEO_PATH} \
    --output_name ${OUTPUT_NAME}

echo "Results saved to: outputs/${OUTPUT_NAME}/"
