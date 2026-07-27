"""Render the processed input, the estimated SMPL-X mesh, and the NLF-GS output.

For one or more subjects this writes a single comparison PNG per subject where
each row is one camera view and contains three panels:

    [ processed input ]  |  [ estimated SMPL-X mesh ]  |  [ NLF-GS output ]

The input panel is the preprocessed ground-truth image
(``processed/<subject>/<subject>_<view>.png``). The middle panel is the estimated
SMPL-X body shape rasterized with pyrender using the same canonical orbit camera
used everywhere else in the project. The right panel is the reconstructed 3D
Gaussian avatar, either rendered live with gsplat (requires CUDA) or loaded from
previously saved ``reconstructed_<view>.png`` files.

Inputs are taken from an inference ``.pt`` bundle (``output/<subject>/<subject>.pt``)
when available; otherwise inference is run on the fly (requires ``--checkpoint``).

Examples
--------
Use existing inference outputs (bundle + reconstruction PNGs), no GPU needed::

    python scripts/render_smplx_vs_output.py --config configs/nlfgs_gpu.yaml \
        --subject 0007 --output-source png

Run inference and render both panels live (needs CUDA + gsplat)::

    python scripts/render_smplx_vs_output.py --config configs/nlfgs_gpu.yaml \
        --checkpoint models/checkpoints/your.ckpt --subject 0007 \
        --views 0,90,180,270
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

# Make both the repo root and ``src`` importable (mirrors inference.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _configure_pyopengl_platform() -> None:
    """Pick an offscreen OpenGL backend for pyrender (same logic as preprocessing)."""
    if "PYOPENGL_PLATFORM" in os.environ:
        return
    if sys.platform == "darwin":
        return
    is_linux = sys.platform.startswith("linux")
    is_headless = (not os.environ.get("DISPLAY")) and (not os.environ.get("WAYLAND_DISPLAY"))
    if is_linux and is_headless:
        os.environ["PYOPENGL_PLATFORM"] = "egl"


_configure_pyopengl_platform()

from src.avatar_utils.camera import look_at_viewmatrix  # noqa: E402
from src.avatar_utils.config import load_config  # noqa: E402
from src.avatar_utils.view_config import (  # noqa: E402
    MODEL_INPUT_4VIEW_ORDER,
    azimuth_direction,
)
from src.training.nlfgs_builder import (  # noqa: E402
    apply_matmul_precision_for_device,
    build_nlf_gaussian_model,
    device_from_cfg,
    gsplat_renderer_if_cuda,
)

# Imported lazily inside functions to keep import cost / optional deps contained:
#   - inference.run_inference / _load_checkpoint (only when running inference)
#   - smplx model faces
#   - pyrender / cv2 / PIL


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_views(raw: Optional[str]) -> list[str]:
    if not raw:
        return list(MODEL_INPUT_4VIEW_ORDER)
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _parse_subjects(raw: Optional[str], cfg: dict) -> list[str]:
    if raw:
        return [tok.strip() for tok in raw.split(",") if tok.strip()]
    # Fall back to the configured validation split (same source as inference.py).
    from inference import _subjects_from_val_split

    return _subjects_from_val_split(cfg)


def _output_dir(cfg: dict) -> Path:
    inf = cfg.get("inference", {}) or {}
    render = cfg.get("render", {}) or {}
    return _resolve_path(str(inf.get("output_dir", render.get("save_path", "output"))))


def _processed_root(cfg: dict) -> Path:
    data_cfg = cfg.get("data", {}) or {}
    return _resolve_path(str(data_cfg.get("processed_root", "processed")))


def _reconstruction_subdir(cfg: dict) -> str:
    inf = cfg.get("inference", {}) or {}
    return str(inf.get("reconstruction_subdir") or inf.get("canonical_views_subdir") or "reconstruction")


def _reconstruction_prefix(cfg: dict) -> str:
    inf = cfg.get("inference", {}) or {}
    raw = inf.get("reconstruction_save_prefix")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    return "reconstructed"


# ---------------------------------------------------------------------------
# Gaussian bundle / inference
# ---------------------------------------------------------------------------
def _load_bundle(pt_path: Path) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Load an inference ``.pt`` bundle → (gaussian_3d, gaussian_params, vertices3d).

    Scales are returned in **linear** space regardless of how they were stored.
    """
    try:
        bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(pt_path, map_location="cpu")

    gaussian_3d = bundle["gaussian_3d"]
    gaussian_params = {k: v for k, v in bundle["gaussian_params"].items()}
    vertices3d = bundle.get("vertices3d", torch.empty(0, 3))

    if bool(bundle.get("scales_are_log_space", False)) and "scales" in gaussian_params:
        gaussian_params["scales"] = torch.exp(gaussian_params["scales"])
    return gaussian_3d, gaussian_params, vertices3d


def _get_gaussians_and_vertices(
    cfg: dict,
    subject: str,
    *,
    device: torch.device,
    pt_path: Optional[Path],
    checkpoint: Optional[str],
    force_inference: bool,
    shared_model,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    out_dir = _output_dir(cfg)
    default_pt = out_dir / subject / f"{subject}.pt"
    chosen_pt = pt_path or default_pt

    if not force_inference and chosen_pt.is_file():
        print(f"[{subject}] Loading Gaussian bundle: {chosen_pt}")
        return _load_bundle(chosen_pt)

    if not checkpoint:
        raise FileNotFoundError(
            f"[{subject}] No bundle at {chosen_pt} and no --checkpoint given. "
            f"Run inference.py first or pass --checkpoint to generate it on the fly."
        )

    from inference import run_inference

    print(f"[{subject}] Running inference (checkpoint: {checkpoint})")
    gaussian_3d, gaussian_params, vertices3d, _subj, _tmpl, _neural = run_inference(
        cfg, subject, str(checkpoint), device, model=shared_model
    )
    return gaussian_3d.detach().cpu(), {k: v.detach().cpu() for k, v in gaussian_params.items()}, vertices3d


# ---------------------------------------------------------------------------
# SMPL-X mesh rendering (pyrender)
# ---------------------------------------------------------------------------
def _faces_for_vertices(vertices3d: torch.Tensor) -> np.ndarray:
    """Return triangle faces compatible with ``vertices3d``.

    Prefer the standard SMPL-X body topology (matches ``subject_params`` vertices,
    10475 verts). Fall back to the canonical UV-mesh faces when the vertex count
    matches that mesh instead (``smplx_source: canonical_mesh``).
    """
    n_verts = int(vertices3d.shape[0])

    def _valid(faces: np.ndarray) -> bool:
        return faces.size > 0 and int(faces.max()) < n_verts

    # Standard SMPL-X faces.
    try:
        from src.avatar_utils.smplx_loader import _get_smplx_model

        faces = np.asarray(_get_smplx_model().faces, dtype=np.int64)
        if _valid(faces):
            return faces
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load SMPL-X model faces ({exc}); trying canonical UV mesh faces.")

    # Canonical UV-mesh faces (avatar_template.cano_mesh_path).
    try:
        from src.encoder.avatar_template import AvatarTemplate

        faces = AvatarTemplate().mesh_faces.cpu().numpy().astype(np.int64)
        if _valid(faces):
            return faces
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load canonical UV mesh faces ({exc}).")

    raise ValueError(
        f"No triangle topology compatible with vertices3d (n={n_verts}). "
        f"Ensure the SMPL-X model or canonical UV mesh is available under models/."
    )


def _camera_params(cfg: dict) -> tuple[int, int, float, float, list[float]]:
    data_cfg = cfg.get("data", {}) or {}
    cam_cfg = cfg.get("camera", {}) or {}
    w, h = data_cfg.get("image_size", [1024, 1024])
    distance = float(cam_cfg.get("distance", 3.0))
    yfov_deg = float(cam_cfg.get("yfov_deg", 45.0))
    up = list(cam_cfg.get("up", [0.0, 1.0, 0.0]))
    return int(w), int(h), distance, yfov_deg, up


def render_smplx_mesh(
    vertices3d: torch.Tensor,
    faces: np.ndarray,
    view_name: str,
    cfg: dict,
    *,
    bg_color: Sequence[int] = (0, 0, 0),
    base_color=(0.72, 0.72, 0.78, 1.0),
) -> np.ndarray:
    """Rasterize the SMPL-X mesh for one orbit view → RGB uint8 (H, W, 3)."""
    import pyrender
    import trimesh

    width, height, distance, yfov_deg, up = _camera_params(cfg)

    verts = vertices3d.detach().cpu().numpy().astype(np.float64)
    mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=list(base_color), metallicFactor=0.0, roughnessFactor=0.85, alphaMode="OPAQUE"
    )
    mesh = pyrender.Mesh.from_trimesh(mesh_tm, material=material, smooth=True)

    bg = list(bg_color)
    scene = pyrender.Scene(bg_color=[bg[0], bg[1], bg[2], 0], ambient_light=[0.35, 0.35, 0.35])
    scene.add(mesh)

    # Same orbit camera convention as preprocessing: pyrender uses -Z forward.
    direction = np.asarray(azimuth_direction(float(view_name)), dtype=np.float64)
    up_vec = np.asarray(up, dtype=np.float64)
    if np.allclose(np.cross(up_vec, direction), 0.0):
        up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    eye = direction * distance

    _w2c, c2w = look_at_viewmatrix(
        eye=eye, target=np.zeros(3), up=up_vec, device=None, dtype=torch.float32, forward="-z"
    )
    c2w = c2w.detach().cpu().numpy().astype(float)

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(yfov_deg))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(camera, pose=c2w)
    scene.add(light, pose=c2w)

    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _depth = renderer.render(scene)
    finally:
        renderer.delete()
    return np.ascontiguousarray(color[:, :, :3]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Output (gsplat render or saved reconstruction PNG)
# ---------------------------------------------------------------------------
def render_output_gsplat(
    gaussian_3d: torch.Tensor,
    gaussian_params: dict[str, torch.Tensor],
    view_name: str,
    device: torch.device,
) -> np.ndarray:
    """Rasterize the Gaussian avatar for one view with gsplat → RGB uint8 (H, W, 3)."""
    from src.render.gaussian_renderer import GsplatRenderer

    renderer = GsplatRenderer()
    means = gaussian_3d.to(device)
    params = {k: v.to(device) for k, v in gaussian_params.items()}
    imgs = renderer.render(means, params, view_name=view_name)  # (1, H, W, 3)
    img = imgs[0].clamp(0.0, 1.0).detach().cpu().numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)


def load_reconstruction_png(cfg: dict, subject: str, view_name: str) -> Optional[np.ndarray]:
    """Load a previously saved ``reconstructed_<view>.png`` if present → RGB uint8."""
    from PIL import Image

    png = _output_dir(cfg) / subject / _reconstruction_subdir(cfg) / f"{_reconstruction_prefix(cfg)}_{view_name}.png"
    if not png.is_file():
        return None
    return np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)


def load_input_png(cfg: dict, subject: str, view_name: str) -> Optional[np.ndarray]:
    """Load the preprocessed input image ``processed/<subject>/<subject>_<view>.png``."""
    from PIL import Image

    png = _processed_root(cfg) / subject / f"{subject}_{view_name}.png"
    if not png.is_file():
        return None
    return np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def _resize_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    import cv2

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _label_band(width: int, text: str, height: int = 42) -> np.ndarray:
    import cv2

    band = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        band, text, (12, int(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )
    return band


def _placeholder(height: int, width: int, text: str) -> np.ndarray:
    import cv2

    img = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(
        img, text, (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA
    )
    return img


def _panel(img: Optional[np.ndarray], h: int, w: int, label: str, missing: str) -> np.ndarray:
    body = _placeholder(h, w, missing) if img is None else _resize_to(img, h, w)
    return np.vstack([_label_band(w, label), body])


def _compose_row(
    input_img: Optional[np.ndarray],
    mesh_img: np.ndarray,
    out_img: Optional[np.ndarray],
    view_name: str,
) -> np.ndarray:
    """One row: [ input | SMPL-X | output ], each with a label band on top."""
    h, w = mesh_img.shape[:2]
    panels = [
        _panel(input_img, h, w, f"input  (view {view_name})", "input unavailable"),
        _panel(mesh_img, h, w, f"SMPL-X  (view {view_name})", "mesh unavailable"),
        _panel(out_img, h, w, f"output  (view {view_name})", "output unavailable"),
    ]

    sep = np.full((panels[0].shape[0], 4, 3), 64, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i > 0:
            stacked.append(sep)
        stacked.append(p)
    return np.hstack(stacked)


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    if len(rows) == 1:
        return rows[0]
    width = rows[0].shape[1]
    gap = np.full((6, width, 3), 64, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for i, r in enumerate(rows):
        if i > 0:
            stacked.append(gap)
        stacked.append(r)
    return np.vstack(stacked)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the estimated SMPL-X mesh and the NLF-GS output in one image."
    )
    parser.add_argument("--config", type=str, default="configs/nlfgs_gpu.yaml", help="YAML config path.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Lightning .ckpt (only needed when the inference bundle is missing or --force-inference).",
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Subject id, or comma-separated ids. Default: all subjects in data.val_subject_path.",
    )
    parser.add_argument(
        "--views", type=str, default=None,
        help="Comma-separated azimuth views (e.g. 0,90,180,270). Default: cardinal 4 views.",
    )
    parser.add_argument(
        "--pt", type=str, default=None,
        help="Explicit path to a single inference .pt bundle (implies a single subject).",
    )
    parser.add_argument(
        "--output-source", choices=["auto", "render", "png"], default="auto",
        help="Right panel source: 'render' (gsplat, needs CUDA), 'png' (saved reconstruction), or 'auto'.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for comparison PNGs (default: output_dir).")
    parser.add_argument("--out", type=str, default=None, help="Explicit output PNG path (single subject only).")
    parser.add_argument("--force-inference", action="store_true", help="Ignore any existing bundle and run inference.")
    parser.add_argument(
        "--mesh-bg", type=int, nargs=3, default=(0, 0, 0), metavar=("R", "G", "B"),
        help="Background color for the SMPL-X mesh panel (default: 0 0 0).",
    )
    args = parser.parse_args()

    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)
    device = device_from_cfg(cfg)
    apply_matmul_precision_for_device(cfg, device)

    views = _parse_views(args.views)
    subjects = _parse_subjects(args.subject, cfg)
    if args.pt and len(subjects) != 1:
        raise ValueError("--pt applies to a single subject; pass exactly one --subject.")
    if args.out and len(subjects) != 1:
        raise ValueError("--out is for a single subject; use --out-dir for multiple subjects.")

    # Decide the output panel source up front.
    output_source = args.output_source
    if output_source == "render" and device.type != "cuda":
        raise RuntimeError("--output-source render requires CUDA (gsplat). Use 'png' or 'auto'.")
    if output_source == "auto":
        output_source = "render" if device.type == "cuda" else "png"
    print(f"Views: {views} | output panel source: {output_source} | device: {device}")

    checkpoint = args.checkpoint or (cfg.get("inference", {}) or {}).get("checkpoint")

    # Build the model once if we may need to run inference.
    shared_model = None
    need_model = args.force_inference or output_source == "render"
    if need_model and checkpoint:
        from inference import _load_checkpoint

        shared_model = build_nlf_gaussian_model(cfg, device)
        try:
            _load_checkpoint(shared_model, str(_resolve_path(str(checkpoint))), device)
        except FileNotFoundError:
            # Only fatal if we actually need to run inference; loading a bundle is still fine.
            if args.force_inference:
                raise
            shared_model = None

    out_dir_base = _resolve_path(args.out_dir) if args.out_dir else _output_dir(cfg)

    for subject in subjects:
        try:
            gaussian_3d, gaussian_params, vertices3d = _get_gaussians_and_vertices(
                cfg, subject, device=device,
                pt_path=_resolve_path(args.pt) if args.pt else None,
                checkpoint=str(_resolve_path(str(checkpoint))) if checkpoint else None,
                force_inference=args.force_inference, shared_model=shared_model,
            )
        except FileNotFoundError as exc:
            print(f"[{subject}] Skipping: {exc}")
            continue

        if vertices3d is None or int(vertices3d.shape[0]) == 0:
            print(f"[{subject}] Skipping: bundle has no SMPL-X vertices3d.")
            continue

        faces = _faces_for_vertices(vertices3d)

        rows: list[np.ndarray] = []
        for view_name in views:
            mesh_img = render_smplx_mesh(vertices3d, faces, view_name, cfg, bg_color=tuple(args.mesh_bg))

            input_img = load_input_png(cfg, subject, view_name)
            if input_img is None:
                print(
                    f"[{subject}] No processed input for view {view_name} under "
                    f"{_processed_root(cfg) / subject}; leaving input panel blank."
                )

            out_img: Optional[np.ndarray] = None
            if output_source == "render":
                out_img = render_output_gsplat(gaussian_3d, gaussian_params, view_name, device)
            else:
                out_img = load_reconstruction_png(cfg, subject, view_name)
                if out_img is None:
                    print(
                        f"[{subject}] No reconstruction PNG for view {view_name} under "
                        f"{_output_dir(cfg) / subject / _reconstruction_subdir(cfg)}; leaving panel blank."
                    )
            rows.append(_compose_row(input_img, mesh_img, out_img, view_name))

        combined = _stack_rows(rows)

        if args.out:
            out_path = _resolve_path(args.out)
        else:
            out_path = out_dir_base / subject / f"{subject}_smplx_vs_output.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        Image.fromarray(combined).save(out_path)
        print(f"[{subject}] Saved comparison → {out_path.resolve()}")


if __name__ == "__main__":
    main()
