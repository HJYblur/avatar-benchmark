#!/bin/bash
# Example: Run pipeline with ground truth for evaluation

INPUT_VIDEO="data/videos/input.mp4"
GT_VIDEO="data/videos/ground_truth.mp4"
OUTPUT_NAME="evaluation_with_metrics"

python scripts/run_pipeline.py \
    --video ${INPUT_VIDEO} \
    --ground_truth ${GT_VIDEO} \
    --output_name ${OUTPUT_NAME}

echo "Evaluation results saved to: outputs/${OUTPUT_NAME}/metrics/"
