import os
import trimesh
import torch
from typing import List, Optional


def intrinsic_matrix_from_field_of_view(
    fov_degrees: float, imshape: List[int], device: Optional[torch.device] = None
):
    imshape = torch.tensor(imshape, dtype=torch.float32, device=device)
    fov_radians = fov_degrees * torch.tensor(
        torch.pi / 180, dtype=torch.float32, device=device
    )
    larger_side = torch.max(imshape)
    focal_length = larger_side / (torch.tan(fov_radians / 2) * 2)
    _0 = torch.tensor(0, dtype=torch.float32, device=device)
    _1 = torch.tensor(1, dtype=torch.float32, device=device)

    # print(torch.stack([focal_length, _0, imshape[1] / 2], dim=-1))
    return (
        torch.stack(
            [
                focal_length,
                _0,
                (imshape[1] - 1) / 2,
                _0,
                focal_length,
                (imshape[0] - 1) / 2,
                _0,
                _0,
                _1,
            ],
            dim=-1,
        )
        .unflatten(-1, (3, 3))
        .unsqueeze(0)
    )


def load_cano_mesh(cano_mesh_path: str) -> trimesh.Trimesh:
    if not os.path.exists(cano_mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {cano_mesh_path}")
    mesh = trimesh.load(cano_mesh_path, process=False, maintain_order=True)
    print(
        f"Loaded mesh from {cano_mesh_path}, with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces."
    )
    return mesh
