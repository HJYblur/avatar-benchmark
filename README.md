# Avatar Benchmark

## 3DGS loading site
[Link](https://superspl.at/editor)

## Intro
This repository contains tools to load SMPL avatars, bind textures, generate per-frame pose-driven animations, render image sequences with dataset cameras, and evaluate reconstructed frames against originals. The main functions are:
- Provide an end-to-end pipeline to render SMPL-based avatars using recorded pose sequences.
- Produce per-frame timing logs and evaluation helpers (PSNR/SSIM/LPIPS).
  

## High-level pipeline
1. Load SMPL/SMPL-X model parameters (folder in `models/smpl` or `models/smplx`).
2. Load per-subject shape and pose sequences (files in `data/<subject>/`) using utilities in `src/utils` and `src/preprocessing`.
3. Create a textured mesh:
   - Load SMPL vertices for given betas/pose.
   - Load UV layout (`smpl_uv.obj`) and texture image (e.g., `tex-{subject}.jpg`).
   - Create PyTorch3D `TexturesUV` and assemble a `Meshes` object.
4. Configure camera and renderer:
   - By default use a look-at FoV camera.
   - *[TODO]* If `camera.pkl` is available for a subject, `camera_loader` will extract rotation/translation and intrinsics and wire them into the renderer so output matches dataset view.
5. Render per-frame images and optionally overlay debug axes.
6. Save images and per-frame timing to CSV (`logs/`).
7. Use `evaluate.py` to numerically sort original vs reconstructed frames and compute basic metrics.

## Example Output
![image](assests/demo.png)

## File / folder structure (tree-map)
```
Root
- config/                     # YAML configuration (default.yaml)
- data/                       # Dataset root (subjects, frames, camera files, etc.)
  +- people_snapshot_public/
     +- <subject>/
        +- camera.pkl          # dataset camera metadata (optional)
        +- poses.npz / poses.pt
        +- keypoints.hdf5
        +- images/             # original extracted frames
        +- masks/              # per-frame masks (optional)
- models/
  +- smpl/                     # SMPL pickles and uv obj
  +- smplx/                    # SMPL-X npz models
- src/
  +- preprocessing/           
  +- utils/
  +- metrics/                  # evaluator / metric computation
- main.py                     # rendering pipeline
- evaluate.py                 # visulization metrics calculation
- requirements.txt            
```


## Quick start
1. Install dependencies (example):

```bash
pip install -r requirements.txt
# If pytorch3d is not in requirements.txt, install it following the instructions for your env version:
# https://pytorch3d.org/
```

For SMPL models, we use the same models as in [https://github.com/JunukCha/textured_smpl]. Please refer to the readme there.

2. Edit `config/default.yaml` to point to your `people_snapshot` data folder and the `subject` you want to render.

3. Run the pipeline (renders into the configured output folder):

```bash
python main.py
```

4. Evaluate / compare frames:

```bash
python evaluate.py
```
Note, there are some strange warnings when running evaluation, thus to get a more clear console output, we can use

```
PYTHONWARNINGS=ignore python -W ignore evaluate.py 2>/dev/null
```

## TODO
1. Add other evaluation metrics (CRQA, Soft-DTW) into pipeline.
2. Incorparate the camera view into rendering