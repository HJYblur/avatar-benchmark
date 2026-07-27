"""Project the estimated SMPL-X vertices onto the input / output as a wireframe.

Instead of rendering the SMPL-X body as a separate panel (see
``render_smplx_vs_output.py``), this overlays the projected SMPL-X mesh directly on
top of the images, so you can visually check how well the fitted body aligns with the
processed input and/or the NLF-GS output.

For one or more subjects it writes a comparison PNG per subject where each row is one
camera view and contains the requested targets, each with the SMPL-X wireframe drawn
on top:

    [ input + wireframe ]  |  [ output + wireframe ]

The vertices are projected with the same camera used everywhere else in the project
(``load_camera_mapping`` + ``vertices_3d_to_2d``). Triangle edges from the SMPL-X
topology are drawn with cv2; back-facing triangles are culled by default so the front
surface reads as a clean wireframe.

Examples
--------
Overlay on both input and output (output from saved reconstruction PNGs, no GPU)::

    python scripts/overlay_smplx_wireframe.py --config configs/nlfgs_gpu.yaml \
        --subject 0007 --views 0,90,180,270 --output-source png

Overlay only on the input, drawing projected vertices as points::

    python scripts/overlay_smplx_wireframe.py --config configs/nlfgs_gpu.yaml \
        --subject 0007 --targets input --mode points
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make repo root, ``src`` and this ``scripts`` dir importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the shared pipeline helpers from the side-by-side script.
import render_smplx_vs_output as base  # noqa: E402

from src.avatar_utils.camera import load_camera_mapping  # noqa: E402
from src.avatar_utils.config import load_config  # noqa: E402
from src.avatar_utils.smplx_loader import vertices_3d_to_2d  # noqa: E402
from src.training.nlfgs_builder import (  # noqa: E402
    apply_matmul_precision_for_device,
    build_nlf_gaussian_model,
    device_from_cfg,
)


# ---------------------------------------------------------------------------
# Projection + edge extraction
# ---------------------------------------------------------------------------
def _unique_edges(faces: torch.Tensor) -> torch.Tensor:
    """Undirected, deduplicated edge list (E, 2) from triangle faces (F, 3)."""
    e = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )  # (3F, 2)
    e = torch.sort(e, dim=1).values
    return torch.unique(e, dim=0)


def project_smplx(
    vertices3d: torch.Tensor,
    faces: np.ndarray,
    view_name: str,
    *,
    cull_backfaces: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project SMPL-X vertices for one view.

    Returns:
        verts2d: (N, 2) float pixel coordinates.
        edges: (E, 2) int vertex-index pairs to draw (front-facing, in-front-of-camera).
        depth: (N,) camera-space z (for optional depth ordering / debugging).
    """
    verts = vertices3d.to(dtype=torch.float32)
    viewmats, Ks = load_camera_mapping(view_name)
    w2c = viewmats[0].to(dtype=torch.float32)
    K = Ks[0].to(dtype=torch.float32)

    verts2d = vertices_3d_to_2d(verts, K, w2c)  # (N, 2)

    R = w2c[:3, :3]
    t = w2c[:3, 3]
    verts_cam = verts @ R.T + t  # (N, 3), camera space (+z forward)
    depth = verts_cam[:, 2]

    faces_t = torch.as_tensor(faces, dtype=torch.long)
    face_z = depth[faces_t]  # (F, 3)
    in_front = (face_z > 0).all(dim=1)

    keep = in_front
    if cull_backfaces:
        v0 = verts_cam[faces_t[:, 0]]
        v1 = verts_cam[faces_t[:, 1]]
        v2 = verts_cam[faces_t[:, 2]]
        normal = torch.linalg.cross(v1 - v0, v2 - v0)
        centroid = (v0 + v1 + v2) / 3.0
        # Camera at origin looking +z: front-facing => normal points back toward camera.
        front_facing = (normal * centroid).sum(dim=1) < 0
        keep = keep & front_facing

    faces_keep = faces_t[keep]
    if faces_keep.numel() == 0:
        # Fall back to depth-only culling if back-face culling removed everything.
        faces_keep = faces_t[in_front]

    edges = _unique_edges(faces_keep).cpu().numpy()
    return verts2d.cpu().numpy(), edges, depth.cpu().numpy()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_overlay(
    base_img: np.ndarray,
    verts2d: np.ndarray,
    edges: np.ndarray,
    *,
    mode: str = "wireframe",
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    point_radius: int = 1,
    alpha: float = 0.6,
) -> np.ndarray:
    """Draw the projected SMPL-X wireframe (or points) over ``base_img`` (RGB uint8)."""
    import cv2

    canvas = base_img.copy()
    h, w = canvas.shape[:2]
    pts = np.rint(verts2d).astype(np.int64)

    if mode == "points":
        for x, y in pts:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (int(x), int(y)), point_radius, color, -1, cv2.LINE_AA)
    else:  # wireframe
        for i, j in edges:
            p0 = pts[i]
            p1 = pts[j]
            cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, thickness, cv2.LINE_AA)

    if alpha >= 1.0:
        return canvas
    return cv2.addWeighted(canvas, alpha, base_img, 1.0 - alpha, 0.0)


# ---------------------------------------------------------------------------
# Base images per target
# ---------------------------------------------------------------------------
def _base_image_size(cfg: dict) -> tuple[int, int]:
    w, h, _d, _f, _up = base._camera_params(cfg)
    return w, h


def get_target_image(
    target: str,
    cfg: dict,
    subject: str,
    view_name: str,
    *,
    gaussian_3d: torch.Tensor,
    gaussian_params: dict[str, torch.Tensor],
    output_source: str,
    device: torch.device,
) -> Optional[np.ndarray]:
    """Fetch the base image (before overlay) for a target ('input' or 'output')."""
    if target == "input":
        return base.load_input_png(cfg, subject, view_name)
    if target == "output":
        if output_source == "render":
            return base.render_output_gsplat(gaussian_3d, gaussian_params, view_name, device)
        return base.load_reconstruction_png(cfg, subject, view_name)
    raise ValueError(f"Unknown target {target!r} (expected 'input' or 'output').")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay the projected SMPL-X wireframe on the input / output images."
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
        "--targets", type=str, default="input,output",
        help="Comma-separated images to overlay on: any of 'input','output'. Default: input,output.",
    )
    parser.add_argument("--pt", type=str, default=None, help="Explicit path to a single inference .pt bundle.")
    parser.add_argument(
        "--output-source", choices=["auto", "render", "png"], default="auto",
        help="Base image for the 'output' target: gsplat 'render' (CUDA), saved 'png', or 'auto'.",
    )
    parser.add_argument("--mode", choices=["wireframe", "points"], default="wireframe", help="Overlay style.")
    parser.add_argument(
        "--color", type=int, nargs=3, default=(0, 255, 0), metavar=("R", "G", "B"),
        help="Overlay color (default: 0 255 0 = green).",
    )
    parser.add_argument("--thickness", type=int, default=1, help="Wireframe line thickness in pixels.")
    parser.add_argument("--point-radius", type=int, default=1, help="Vertex point radius (points mode).")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay opacity in [0,1] (1 = opaque).")
    parser.add_argument("--no-cull", action="store_true", help="Disable back-face culling (draw all front edges).")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for overlay PNGs (default: output_dir).")
    parser.add_argument("--out", type=str, default=None, help="Explicit output PNG path (single subject only).")
    parser.add_argument("--force-inference", action="store_true", help="Ignore any existing bundle and run inference.")
    args = parser.parse_args()

    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)
    device = device_from_cfg(cfg)
    apply_matmul_precision_for_device(cfg, device)

    views = base._parse_views(args.views)
    subjects = base._parse_subjects(args.subject, cfg)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in ("input", "output"):
            raise ValueError(f"--targets entries must be 'input' or 'output', got {t!r}.")
    if not targets:
        raise ValueError("--targets must list at least one of 'input','output'.")

    if args.pt and len(subjects) != 1:
        raise ValueError("--pt applies to a single subject; pass exactly one --subject.")
    if args.out and len(subjects) != 1:
        raise ValueError("--out is for a single subject; use --out-dir for multiple subjects.")

    output_source = args.output_source
    need_output = "output" in targets
    if output_source == "render" and device.type != "cuda":
        raise RuntimeError("--output-source render requires CUDA (gsplat). Use 'png' or 'auto'.")
    if output_source == "auto":
        output_source = "render" if device.type == "cuda" else "png"
    print(f"Views: {views} | targets: {targets} | output source: {output_source} | device: {device}")

    checkpoint = args.checkpoint or (cfg.get("inference", {}) or {}).get("checkpoint")

    shared_model = None
    need_model = args.force_inference or (need_output and output_source == "render")
    if need_model and checkpoint:
        from inference import _load_checkpoint

        shared_model = build_nlf_gaussian_model(cfg, device)
        try:
            _load_checkpoint(shared_model, str(base._resolve_path(str(checkpoint))), device)
        except FileNotFoundError:
            if args.force_inference:
                raise
            shared_model = None

    out_dir_base = base._resolve_path(args.out_dir) if args.out_dir else base._output_dir(cfg)
    bw, bh = _base_image_size(cfg)
    color = tuple(int(c) for c in args.color)

    for subject in subjects:
        try:
            gaussian_3d, gaussian_params, vertices3d = base._get_gaussians_and_vertices(
                cfg, subject, device=device,
                pt_path=base._resolve_path(args.pt) if args.pt else None,
                checkpoint=str(base._resolve_path(str(checkpoint))) if checkpoint else None,
                force_inference=args.force_inference, shared_model=shared_model,
            )
        except FileNotFoundError as exc:
            print(f"[{subject}] Skipping: {exc}")
            continue

        if vertices3d is None or int(vertices3d.shape[0]) == 0:
            print(f"[{subject}] Skipping: bundle has no SMPL-X vertices3d.")
            continue

        faces = base._faces_for_vertices(vertices3d)

        rows: list[np.ndarray] = []
        for view_name in views:
            verts2d, edges, _depth = project_smplx(
                vertices3d, faces, view_name, cull_backfaces=not args.no_cull
            )

            panels: list[np.ndarray] = []
            for target in targets:
                img = get_target_image(
                    target, cfg, subject, view_name,
                    gaussian_3d=gaussian_3d, gaussian_params=gaussian_params,
                    output_source=output_source, device=device,
                )
                if img is None:
                    print(f"[{subject}] No base image for target '{target}' view {view_name}; using placeholder.")
                    panel_body = base._placeholder(bh, bw, f"{target} unavailable")
                else:
                    img = base._resize_to(img, bh, bw)
                    panel_body = draw_overlay(
                        img, verts2d, edges,
                        mode=args.mode, color=color, thickness=args.thickness,
                        point_radius=args.point_radius, alpha=args.alpha,
                    )
                label = f"{target} + SMPL-X  (view {view_name})"
                panels.append(np.vstack([base._label_band(bw, label), panel_body]))

            sep = np.full((panels[0].shape[0], 4, 3), 64, dtype=np.uint8)
            stacked: list[np.ndarray] = []
            for i, p in enumerate(panels):
                if i > 0:
                    stacked.append(sep)
                stacked.append(p)
            rows.append(np.hstack(stacked))

        combined = base._stack_rows(rows)

        if args.out:
            out_path = base._resolve_path(args.out)
        else:
            out_path = out_dir_base / subject / f"{subject}_smplx_wireframe.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        Image.fromarray(combined).save(out_path)
        print(f"[{subject}] Saved wireframe overlay → {out_path.resolve()}")


if __name__ == "__main__":
    main()
