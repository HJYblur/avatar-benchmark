import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.camera import intrinsic_matrix_from_field_of_view
from utils.ply_loader import load_ply


class AvatarDataset(Dataset):
    """
    Minimal dataset scaffold.
    - Loads images by glob pattern under a root.
    - Provides intrinsics from FOV.
    - Provides canonical_points from the avatar template PLY.

    Notes:
      - This scaffold does not include SMPL-X per-sample fits. Add as needed.
      - Canonical points are taken as stored in the PLY (face-local offsets). If you
        need absolute canonical mesh coords, adjust the loader to add face centers.
    """

    def __init__(
        self,
        root: str,
        image_glob: str = "*.jpg",
        fov_degrees: float = 55.0,
        proc_side: int = 256,
        template_path: str = "models/avatar_template.ply",
        transform: Optional[Any] = None,
    ):
        self.root = root
        self.image_glob = image_glob
        self.fov_degrees = float(fov_degrees)
        self.proc_side = int(proc_side)
        self.template_path = template_path
        self.transform = transform

        self.image_paths = _sorted_recursive_glob(os.path.join(root, image_glob))
        if len(self.image_paths) == 0:
            # Try common subfolders
            for sub in ("images", "rgb", "imgs"):
                self.image_paths = _sorted_recursive_glob(os.path.join(root, sub, image_glob))
                if self.image_paths:
                    break
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found under {root} with pattern {image_glob}")

        # Load canonical points once from template
        gd = load_ply(self.template_path, mode="default")
        # xyz from PLY may be face-local; can be adapted later to absolute canonical coords
        self.canonical_points = torch.from_numpy(gd.xyz.astype(np.float32))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB").resize((self.proc_side, self.proc_side))
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        K = intrinsic_matrix_from_field_of_view(self.fov_degrees, [self.proc_side, self.proc_side])

        return {
            "image": img_tensor,  # [3,H,W]
            "intrinsics": K.squeeze(0),  # [3,3]
            "canonical_points": self.canonical_points,  # [P,3]
            "path": img_path,
        }


def _sorted_recursive_glob(pattern: str) -> List[str]:
    # Simple sorted glob supporting ** via os.walk
    import glob

    paths = glob.glob(pattern, recursive=True)
    return sorted(paths)
