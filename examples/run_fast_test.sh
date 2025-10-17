#!/bin/bash
# Example: Quick test with fast configuration

VIDEO_PATH="data/videos/sample.mp4"
CONFIG="configs/fast.yaml"
OUTPUT_NAME="quick_test"

python scripts/run_pipeline.py \
    --video ${VIDEO_PATH} \
    --config ${CONFIG} \
    --output_name ${OUTPUT_NAME}

echo "Quick test results saved to: outputs/${OUTPUT_NAME}/"
