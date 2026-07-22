# NLF-GS

A model for training **personalized 3D avatars** from sparse multi-view images.

A **ResNet50-FPN** backbone extracts image features; Gaussians live on a fixed **SMPL-X**-anchored template and are decoded into 3D Gaussian Splatting parameters, then rendered with **[gsplat](https://github.com/nerfstudio-project/gsplat)**. Training is driven by **[PyTorch Lightning](https://lightning.ai/)** with photometric and regularization losses.

---

## Requirements

NVIDIA GPUs are required for training, inference, and animation (gsplat rasterization). Preprocessing uses **pyrender** and can run on macOS with a display, but Linux headless GPU (EGL) is recommended for large THuman batches.

We recommend **Anaconda** to manage Python environments.

```bash
conda create -n ml python=3.10
conda activate ml
# Install PyTorch with CUDA matching your driver (example for CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Then install other packages
conda env update -f environment.yml
```

Log in to [Weights & Biases](https://wandb.ai/) before training (`train.py` logs to W&B by default):

```bash
wandb login
```

---

## Configuration

All paths and hyperparameters live in YAML configs. Pass `--config` to `train.py`, `inference.py`, and `anim.py`.

| Config | Purpose |
|--------|---------|
| `configs/nlfgs_gpu.yaml` | Full-resolution training / inference (default) |
| `configs/nlfgs_debug.yaml` | Smaller images, fewer views, fast smoke tests |

Important keys:

- `data.processed_root` — preprocessed RGB + masks (default: `processed`)
- `data.train_subject_path` / `data.val_subject_path` — subject splits under `data/` (included in repo; regenerated if missing)
- `inference.checkpoint` — Lightning `.ckpt` for `inference.py`
- `inference.output_dir` — where `{subject}/{subject}.pt` and reconstruction PNGs are written
- `animation.*` — pose sequences and video settings for `anim.py`

### Gaussian density (`k_num_gaussians`)

Gaussians are anchored to the SMPL-X UV mesh: every triangle gets `avatar_template.k_num_gaussians` Gaussians, so the total is `num_faces × k_num_gaussians`. There is no adaptive densification, so `k` is the main knob for how many Gaussians live on each surface. To change it:

1. **Set the count.** Edit `avatar_template.k_num_gaussians` in your config (e.g. `configs/nlfgs_gpu.yaml`) to any integer ≥ 1.
2. **Barycentric layout (optional).** `avatar_template.barycentric_coords` controls where the Gaussians sit within each face. It is **omitted from the shipped configs**, so placement is auto-generated for any `k` and you normally only set `k_num_gaussians`. To pin an explicit layout, add a `barycentric_coords:` list with **exactly** `k_num_gaussians` rows of `[u, v, w]` (a mismatched row count is ignored and auto-generation is used instead).
3. **Regenerate the template PLY.** The total count is baked into `models/avatar_template.ply`, so changing only the config has no effect until you rebuild it. Set `avatar_template.mode: generate` and run once (this rewrites the PLY), then switch back to `mode: default`. Alternatively, delete `models/avatar_template.ply` and it will regenerate on the next run.

```yaml
avatar_template:
  mode: generate          # rebuild the PLY once, then set back to `default`
  path: models/avatar_template.ply
  cano_mesh_path: models/smplx/smplx_uv.obj
  k_num_gaussians: 8      # e.g. double the density from the default of 4
  # barycentric_coords:   # optional; omit to auto-generate for any k
```

Higher `k` increases quality/detail at a roughly linear cost in VRAM and speed (feature sampling, decoder, and rasterization all scale with the total Gaussian count). Existing checkpoints still load (they store network weights only, not Gaussian topology), but a new `k` needs a matching regenerated template and is best paired with retraining.

---

## Model assets

Create a `models/` directory and populate it as follows (large files are gitignored).

```text
models/
├── smplx/
│   ├── SMPLX_FEMALE.npz
│   ├── SMPLX_MALE.npz
│   ├── SMPLX_NEUTRAL.npz
│   └── smplx_uv.obj              # canonical SMPL-X UV mesh (required)
├── torchvision/
│   └── resnet50-0676ba61.pth     # optional; torchvision auto-downloads if missing
├── avatar_template.ply           # auto-generated on first run from smplx_uv.obj
├── checkpoints/
│   └── <your>.ckpt               # set inference.checkpoint or pass --checkpoint
└── lpips/                        # created automatically when running metrics
```

**SMPL-X body models** — download from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/) and place the `.npz` files under `models/smplx/`.

**`smplx_uv.obj`** — canonical UV mesh used to anchor Gaussians (`avatar_template.cano_mesh_path`). Obtain this from project assets or your collaborators; it is separate from the official SMPL-X release.

**ResNet50 weights** (optional local copy):

```bash
mkdir -p models/torchvision
curl -L https://download.pytorch.org/models/resnet50-0676ba61.pth \
  -o models/torchvision/resnet50-0676ba61.pth
```

**Training checkpoint** — after training, point `inference.checkpoint` in the YAML to your `.ckpt`, or pass `--checkpoint path/to.ckpt` to `inference.py`.

---

## Data setup

### Download THuman2.0 dataset

Follow the [original repo](https://github.com/ytrock/THuman2.0-Dataset) to request permission and download the raw OBJ scans and estimated SMPL-X parameters. We use the first **525** subjects. Place scans under `data/THuman_2.0` and SMPL-X fits under `data/THuman_2.0_smplx`.

The final `data/` folder should look like:

```text
data/
├── THuman_2.0/
│   ├── 0001/
│   │   ├── 0001.obj
│   │   ├── material0.jpeg
│   │   └── material0.mtl
│   ├── 0002/
│   │   └── ...
│   └── ...
├── THuman_2.0_smplx/
│   ├── 0001/
│   │   ├── smplx_param.pkl
│   │   └── mesh_smplx.obj
│   ├── 0002/
│   │   └── ...
│   └── ...
├── split_train.txt
├── split_val.txt
└── THuman_cameras/               # created by preprocessing (see below)
    ├── thuman_0.json
    └── ...
```

If macOS creates stray `._*` resource-fork files, remove them with:

```bash
chmod +x scripts/clean_dot_underscore.sh
./scripts/clean_dot_underscore.sh data/THuman_2.0
./scripts/clean_dot_underscore.sh data/THuman_2.0 --delete
```

### Preprocess

Render ground-truth images from the scanned OBJ files:

```bash
python src/data/preprocess_thuman.py
```

Process a subject range only (useful for debugging or resuming):

```bash
python src/data/preprocess_thuman.py --start-subject 0001 --end-subject 0050
```

Outputs are written to `processed/`:

```text
processed/
├── 0001/
│   ├── 0001_0.png
│   ├── 0001_0_mask.png
│   ├── 0001_15.png
│   ├── 0001_15_mask.png
│   │   # … every 15° through …
│   ├── 0001_345.png
│   ├── 0001_345_mask.png
│   └── smplx_param.pkl
├── 0002/
│   └── ...
└── 0525/
    └── ...
```

Camera intrinsics / extrinsics for Gaussian rendering are written under `data/THuman_cameras/` as `thuman_0.json` … `thuman_345.json` (24 azimuth views, 0° = front). These are created when you run `preprocess_thuman.py`.

Train/val splits in `data/split_train.txt` and `data/split_val.txt` are checked in. If either file is missing, the datamodule creates a deterministic random split on first training run.

---

## Usage

### Train

```bash
python train.py --config configs/nlfgs_gpu.yaml
```

Resume from a checkpoint (set `train.resume: true` and `train.ckpt_path` in the config).

### Inference

Runs on subjects listed in `data.val_subject_path`. Requires a trained checkpoint.

```bash
python inference.py --config configs/nlfgs_gpu.yaml --checkpoint models/checkpoints/your.ckpt
```

With `inference.save_reconstruction: true`, writes `reconstructed_<azimuth>.png` under `output/<subject>/reconstruction/` and saves `{subject}.pt` Gaussian bundles.

### Animation

Replay saved Gaussian appearance under new SMPL-X poses (`anim.py`). Configure `animation` in the YAML (pose source, `custom_pose_path`, video vs image mode). Example pose sequences live under `data/anim/`.

```bash
python anim.py --config configs/nlfgs_gpu.yaml --start-subject 0001 --end-subject 0001
```

Requires a prior inference `.pt` under `inference.output_dir/<subject>/`.

### Evaluation

Compare inference renders against ground truth. By default:

- **Targets** (`--target-root`) — preprocessed GT images (`processed/<subject>/<subject>_<view>.png`)
- **Predictions** (`--preds-root`) — inference output (`output/<subject>/reconstruction/reconstructed_<view>.png`)

```bash
python src/evaluation/compute_metrics.py \
  --config-path configs/nlfgs_gpu.yaml \
  --target-root processed \
  --preds-root output \
  --no-mask
```

Use `--test-views` to override which azimuth degrees are scored; otherwise views follow `data.num_views` from the config. LPIPS weights are cached under `models/lpips/`.

#### Logging a run

Pass `--log-dir` (and optionally `--run-name`, `--checkpoint`, `--tag`) to persist a machine-readable record of the evaluation instead of only printing to stdout:

```bash
python src/evaluation/compute_metrics.py \
  --config-path configs/nlfgs_gpu.yaml \
  --target-root processed --preds-root output --no-mask \
  --log-dir eval_logs --run-name my_run --checkpoint models/checkpoints/your.ckpt
```

This writes `eval_logs/my_run.json` (timestamp, checkpoint, config, eval flags, aggregate PSNR/SSIM/LPIPS with mean/std/count, and per-subject metrics) and `eval_logs/my_run.log` (the human-readable console output). Use `--log-json PATH` to write the JSON to an explicit path.

### Ablation study (repeated runs + statistics)

For an ablation where each condition has several repeated runs (e.g. 3 seeds), `scripts/run_ablation.py` automates the full loop: for every checkpoint it runs `inference.py`, then `compute_metrics.py`, and finally aggregates the per-run means **across repeats** of each condition (mean / std / var / min / max, sample std with ddof=1).

1. Edit `scripts/ablation_manifest.yaml` and fill in the real checkpoint paths per condition.
2. Run:

```bash
python scripts/run_ablation.py --manifest scripts/ablation_manifest.yaml
```

Results are written to `ablation_results/<timestamp>/`:

- `summary.json` — full stats plus every per-run record
- `summary.csv` — one row per condition (mean/std/var/min/max per metric)
- `summary.md` — table of `mean ± std` per condition
- `logs/` — per-run inference logs and the `compute_metrics` JSON/log for each repeat

Inference reuses the config's `inference.output_dir` (default `output/`), so repeats run strictly sequentially (inference → eval) before the next checkpoint overwrites the renders. Useful flags: `--dry-run` (print commands only), `--continue-on-error` (skip failed repeats), `--skip-inference` (evaluate existing renders only).
