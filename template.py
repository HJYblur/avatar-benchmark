import os
import trimesh
import numpy as np
from src.utils.ply_loader import load_ply, save_ply


class GaussianData:
    def __init__(self, idx, xyz, rots, scales, opacities, shs):
        self.xyz = xyz
        self.rots = rots
        self.scales = scales
        self.opacities = opacities
        self.shs = shs
        self.parent = idx


class AvatarTemplate:
    """
    AvatarTemplate class.

    Manages a template of Gaussian primitives attached to each face of a source mesh.
    The template is initialized from the spmlx_uv mesh (self.mesh_path) with a fixed
    number of Gaussians per face. On construction, if a cached avatar file (.ply) exists
    at avatar_path it is loaded; otherwise the template is generated from the mesh and
    saved to disk for future use.

    Notes:
    - The number of Gaussians per face is currently static.
    - Generation is intended to be done once; subsequent uses load the cached file.
    """

    def __init__(self, avatar_path="./models/avatar_template.ply"):
        self.mesh_path = "models/smplx/smplx_uv.obj"
        self.avatar_path = avatar_path
        self.k_num_gaussians = 4  # Number of Gaussians per face
        self.avatar = self.load_avatar_template()

    def load_avatar_template(self):
        if os.path.exists(self.avatar_path):
            gau_data = load_ply(self.avatar_path)
            print("Loaded avatar template from:", self.avatar_path)
            return gau_data
        else:
            _avatar = self.generate_avatar_template()
            return _avatar

    def generate_avatar_template(self):
        if not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")
        mesh = trimesh.load(self.mesh_path)
        print(
            f"Loaded mesh from {self.mesh_path}, with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.\n"
            f"Generating avatar template..."
        )
        # vertices[0]: [ 0.062714  0.2885   -0.009561]
        # face[0]: [3 1 0]
        vertices = mesh.vertices
        faces = mesh.faces

        all_xyz = []
        all_rots = []
        all_scales = []
        all_opacities = []
        all_shs = []
        all_parents = []

        for idx, face in enumerate(faces):
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            # generate gaussians for each face
            xyzs, rots, scales, opacity, shs = self.generate_gaussians_for_face(
                idx, v0, v1, v2
            )
            all_xyz.append(xyzs)
            all_rots.append(rots)
            all_scales.append(scales)
            all_opacities.append(opacity)
            all_shs.append(shs)
            all_parents.extend([idx] * self.k_num_gaussians)

        if len(all_xyz) == 0:
            raise RuntimeError("No gaussians generated from mesh")

        all_xyz = np.vstack(all_xyz)
        all_rots = np.vstack(all_rots)
        all_scales = np.vstack(all_scales)
        all_opacities = np.vstack(all_opacities).reshape(-1, 1)
        all_shs = np.vstack(all_shs).reshape(-1, 3)
        parents = np.asarray(all_parents, dtype=np.int32)

        data = {
            "xyz": all_xyz,
            "rots": all_rots,
            "scales": all_scales,
            "opacities": all_opacities,
            "shs": all_shs,
            "parent": parents,
        }

        save_ply(data, self.avatar_path)
        return data

    def generate_gaussians_for_face(self, face_id, v0, v1, v2):
        num_gaussians = self.k_num_gaussians
        center = (v0 + v1 + v2) / 3.0

        # Construct a local coordinate system for the face
        e1 = v1 - v0
        e1 = e1 / (np.linalg.norm(e1) + 1e-9)
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        e2 = np.cross(normal, e1)
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)

        R = np.stack([e1, e2, normal], axis=1)  # 3×3 rotation matrix (columns)

        # Define barycentric coordinates for 4 Gaussians per face
        B4 = [
            (1 / 3, 1 / 3, 1 / 3),
            (1 / 2, 1 / 2, 0),
            (1 / 2, 0, 1 / 2),
            (0, 1 / 2, 1 / 2),
        ]

        # Calculate face area to determine Gaussian scale
        face_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
        gaussian_area = face_area / num_gaussians
        r = np.sqrt(gaussian_area / np.pi)
        std_scale = np.array([r, r, 0.05])  # Example scale with small depth

        # Initialize Gaussians (as numpy arrays)
        xyzs = np.zeros((num_gaussians, 3), dtype=np.float32)
        rots = np.zeros((num_gaussians, 3, 3), dtype=np.float32)
        scales = np.zeros((num_gaussians, 3), dtype=np.float32)
        opacity = np.zeros((num_gaussians, 1), dtype=np.float32)
        shs = np.zeros((num_gaussians, 3), dtype=np.float32)
        for i in range(self.k_num_gaussians):
            pt = B4[i][0] * v0 + B4[i][1] * v1 + B4[i][2] * v2
            xyzs[i, :] = pt
            rots[i, :, :] = R
            scales[i, :] = std_scale
            opacity[i, 0] = 0.6
            shs[i, :] = 0.5
        return xyzs, rots, scales, opacity, shs


if __name__ == "__main__":
    avatar_template = AvatarTemplate()
    print("Avatar template generated and saved.")
