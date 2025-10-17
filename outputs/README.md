# Outputs Directory

This directory contains the outputs from the avatar evaluation pipeline.

## Structure

- `meshes/`: Generated 3D meshes from SMPL-X parameters
- `renders/`: Rendered images from the meshes
- `metrics/`: Evaluation metrics and results

## Output Format

When you run the pipeline, a new subdirectory will be created with the specified output name:

```
outputs/
└── <output_name>/
    ├── frames/           # Extracted video frames
    ├── params/           # SMPL-X parameters (*.npy)
    ├── meshes/           # Generated meshes (*.obj)
    ├── renders/          # Rendered images (*.png)
    └── metrics/          # Evaluation results (*.txt)
```
