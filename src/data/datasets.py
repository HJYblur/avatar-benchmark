import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AvatarDataset(Dataset):
    """
    Avatar dataset supporting two layouts:

    1) Processed PeopleSnapshot-style data under `processed/<subject>/` with:
       - images/image_XXXX.png
       - masks/mask_XXXX.png (optional)
       - cameras.npz (intrinsic 3x3, extrinsic 4x4, height, width)
       - poses.npz (betas, thetas, transl)
    2) Legacy flat RGB image glob under a root folder.

    Returns per-sample dict containing:
      - image: FloatTensor [3,H,W] in [0,1]
      - mask: FloatTensor [1,H,W] in [0,1] if available
      - intrinsics: FloatTensor [3,3]
      - extrinsics: FloatTensor [4,4] if available
      - smpl: Dict[str, FloatTensor] with 'betas' [B], 'thetas' [T], 'transl' [3] if available
      - [optional] canonical_points: FloatTensor [P,3] from avatar template PLY
      - subject: str subject identifier (if processed layout)
      - frame_idx: int frame index (if processed layout)
      - path: str absolute image path
    """

    def __init__(
        self,
        root: str,
        proc_side: int = 256,
        template_path: str = "models/avatar_template.ply",
        transform: Optional[Any] = None,
    ):
        # Test mode
        self.test_mode = True

        self.root = root
        self.proc_side = int(proc_side)
        self.template_path = template_path
        self.transform = transform

        # Internal indices and caches
        # Store the file names found under processed layout
        self._records: List[Dict[str, Any]] = []
        self._camera_cache: Dict[str, Dict[str, Any]] = {}
        self._pose_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Require processed layout; no fallbacks
        if not self._index_processed_layout(self.root) or len(self._records) == 0:
            raise FileNotFoundError(
                f"No processed subjects found under {root}. Expect cameras.npz and images/ per subject."
            )

        # Canonical points are constant across samples and are handled by the model/template.
        # No need to load them here per-dataset.

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._records[idx]

        # Image
        img = Image.open(rec["image_path"]).convert("RGB")
        img = img.resize((self.proc_side, self.proc_side), Image.BILINEAR)
        img_arr = np.array(img)
        img_tensor = torch.from_numpy(img_arr.astype(np.float32) / 255.0).permute(
            2, 0, 1
        )
        # Also provide a uint8 copy for detectors that expect byte inputs
        img_uint8 = torch.from_numpy(img_arr.astype(np.uint8)).permute(2, 0, 1)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # Mask (optional)
        mask_tensor = None
        if rec.get("mask_path") and os.path.exists(rec["mask_path"]):
            m = (
                Image.open(rec["mask_path"])
                .convert("L")
                .resize((self.proc_side, self.proc_side), Image.NEAREST)
            )
            mask_tensor = torch.from_numpy(np.array(m).astype(np.float32) / 255.0)[
                None, :, :
            ]

        # Camera intrinsics/extrinsics
        if rec.get("camera_subject"):
            cam = self._camera_cache[rec["camera_subject"]]
            K = cam["intrinsic"].copy()
            H_orig, W_orig = int(cam["height"]), int(cam["width"])
            sx = self.proc_side / float(W_orig)
            sy = self.proc_side / float(H_orig)
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy
            K_t = torch.from_numpy(K.astype(np.float32))
            w2c_t = torch.from_numpy(cam["extrinsic"].astype(np.float32))
        else:
            raise RuntimeError(
                "Missing camera_subject in record; processed layout expected."
            )

        # SMPL params (optional)
        smpl_out = None
        if rec.get("pose_subject"):
            poses = self._pose_cache[rec["pose_subject"]]
            f = rec["frame_idx"]
            thetas = poses.get("thetas")
            transl = poses.get("transl")
            betas = poses.get("betas")
            if thetas is not None and transl is not None and betas is not None:
                betas_item = (
                    betas[0] if betas.ndim == 2 and betas.shape[0] == 1 else betas
                )
                smpl_out = {
                    "betas": torch.from_numpy(np.asarray(betas_item, dtype=np.float32)),
                    "thetas": torch.from_numpy(np.asarray(thetas[f], dtype=np.float32)),
                    "transl": torch.from_numpy(np.asarray(transl[f], dtype=np.float32)),
                }

        out: Dict[str, Any] = {
            "image": img_tensor,
            "image_uint8": img_uint8,
            "intrinsics": K_t,
            "path": rec["image_path"],
        }
        if w2c_t is not None:
            out["extrinsics"] = w2c_t
        if mask_tensor is not None:
            out["mask"] = mask_tensor
        if smpl_out is not None:
            out["smpl"] = smpl_out
        out["subject"] = rec["subject"]
        out["frame_idx"] = rec["frame_idx"]
        return out

    def _index_processed_layout(self, root: str) -> bool:
        """Index processed layout under `processed/<subject>/`.

        Returns True if any processed subjects were found.
        """
        found = False
        root_p = Path(root)

        def is_subject_dir(p: Path) -> bool:
            return (p / "cameras.npz").exists() and (p / "images").exists()

        candidate_dirs: List[Path] = []
        if is_subject_dir(root_p):
            candidate_dirs = [root_p]
        elif root_p.is_dir():
            for child in root_p.iterdir():
                if child.is_dir() and is_subject_dir(child):
                    candidate_dirs.append(child)

        for subj_dir in sorted(candidate_dirs):
            subj_name = subj_dir.name
            cam_path = subj_dir / "cameras.npz"
            try:
                cam_npz = np.load(str(cam_path))
                self._camera_cache[subj_name] = {
                    "intrinsic": cam_npz["intrinsic"],
                    "extrinsic": cam_npz["extrinsic"],
                    "height": int(cam_npz["height"]),
                    "width": int(cam_npz["width"]),
                }
                found = True
            except Exception:
                continue

            # Optional poses
            pose_path = subj_dir / "poses.npz"
            if pose_path.exists():
                try:
                    pose_npz = np.load(str(pose_path))
                    self._pose_cache[subj_name] = {
                        "betas": pose_npz.get("betas"),
                        "thetas": pose_npz.get("thetas"),
                        "transl": pose_npz.get("transl"),
                    }
                except Exception:
                    pass

            # Frames
            img_dir = subj_dir / "images"
            mask_dir = subj_dir / "masks"
            img_paths = sorted(img_dir.glob("image_*.png"))
            for p in img_paths:
                stem = p.stem.replace("image_", "")
                try:
                    fidx = int(stem)
                except ValueError:
                    fidx = -1
                mpath = mask_dir / f"mask_{stem}.png"
                self._records.append(
                    {
                        "image_path": str(p),
                        "mask_path": str(mpath) if mpath.exists() else None,
                        "subject": subj_name,
                        "frame_idx": fidx,
                        "camera_subject": subj_name,
                        "pose_subject": (
                            subj_name if subj_name in self._pose_cache else None
                        ),
                    }
                )

        # If in test mode, limit records to the first 5 examples to speed up iteration
        if getattr(self, "test_mode", False):
            if len(self._records) > 5:
                self._records = self._records[:5]

        return found

    # No flat image fallback; processed layout is required


def _sorted_recursive_glob(pattern: str) -> List[str]:
    # Simple sorted glob supporting ** via os.walk
    import glob

    paths = glob.glob(pattern, recursive=True)
    return sorted(paths)
