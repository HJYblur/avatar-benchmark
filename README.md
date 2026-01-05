# Avatar Benchmark

This repo scaffolds an avatar generation pipeline that combines an NLF backbone for SMPL‑X pose prediction with a Gaussian‑splatting avatar representation for appearance. The goal is to enable training personalized interactive 3D avatars from RGB images.


## Pipeline

- Avatar Template
  - Attach `k` Gaussians per SMPL‑X mesh face; total `N = k * |F|`.
  - Each Gaussian `g_i = {x_i^can, r_i, s_i, α_i, c_i^SH}` in canonical space.
- Image Encoding
  - Backbone extracts dense features `F_img ∈ R^{C×Hf×Wf}` from input image.
  - NLF predicts SMPL‑X quantities and 2D/3D vertices used downstream.
- Identity Encoding
  - Pool features over avatar foreground (via projected non‑param vertices) to get `z_id ∈ R^D`.
- Pose‑Conditioned Localization
  - For each Gaussian, compute posed center using barycentric weights over its parent triangle: `x_i^{3D}` and projected `x_i^{2D}` from NLF vertices.
- Local Feature Sampling
  - Bilinear sample `F_img` at `x_i^{2D}` to get per‑Gaussian local features `f_i`.
- Identity‑Conditioned Decoding
  - Concatenate `[z_id, f_i, x_i^{3D}]` and decode `(r_i, s_i, α_i, c_i^SH)` with a shared MLP.
- Rendering & Losses
  - Render posed Gaussians with a differentiable 3DGS renderer; supervise with photometric/perceptual losses and regularizers.
  - See architecture diagram: `assets/pipeline.png`.

## TODO For A Runnable Pipeline

1) Training pipeline
- Unify `train.py` and `src/training/trainer.py` using Lightning.
- Instantiate and wire `IdentityEncoder` and `GaussianDecoder` in `train.py`.
- Add checkpointing and logging (metrics, losses, samples).

2) Data & Preprocessing
- Define a dataset class and the dataset to use.
- Add preprocessing scripts for datasets.
- Add camera extrinsics/poses or derive if available.
- Add masks/crops to restrict supervision to the person region.
- Add train/val/test splits and basic augmentations.

3) NLF Integration
- Decide single‑person vs multi‑person handling and batching.


4) Model Glue
- Verify channel order and tensor shapes between dataset, adapter, and trainer.
- Fix `IdentityEncoder` to use the correct key (`vertices2d_nonparam`).
- Determine `backbone_feat_dim` from the loaded backbone or config.

5) Rendering & Losses
- Integrate a differentiable 3D Gaussian renderer for supervision.
- Define and implement photometric, perceptual, occlusion losses.
- Export predicted Gaussians to PLY for debugging/visualization.

6) Config & Requirements
- Add `lightning` to requirements or remove the dependency.
- Validate required config keys and fail with clear messages.


## Usage (Scaffold)
- Generate or load the avatar template via config (`avatar.template.mode: generate|default|test`).
- Point `data.root` to a directory with RGB images; ensure `nlf.checkpoint_path` points to a TorchScript MultipersonNLF checkpoint.
- Run: `python train.py --config configs/nlfgs_base.yaml`

## 3DGS Loading Site
[Link](https://superspl.at/editor)
