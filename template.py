import os
import trimesh
import numpy as np
from src.utils.ply_loader import load_ply, save_ply, matrix_to_quaternion


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
    saved for future use.

    PLY schema per vertex (Gaussian):
        x, y, z -> position
        nx, ny, nz -> normal
        f_dc_0, f_dc_1, f_dc_2 -> SH DC terms
        f_rest_0 .. f_rest_44 -> SH rest terms
        opacity -> opacity
        scale_0, scale_1, scale_2 -> scale
        rot_0, rot_1, rot_2, rot_3 -> rotation

    Notes:
    - The number of Gaussians per face is currently static.
    - Generation is intended to be done once; subsequent uses load the cached file.
    """

    def __init__(self, avatar_path="./models/avatar_template.ply"):
        self.mesh_path = "models/smplx/smplx_uv.obj"
        self.avatar_path = avatar_path
        self.k_num_gaussians = 4  # Number of Gaussians per face
        self.avatar = self.load_avatar_template()

    def load_avatar_template(self, regenerate=True):
        if os.path.exists(self.avatar_path) and not regenerate:
            gau_data = load_ply(self.avatar_path)
            print("Loaded avatar template from:", self.avatar_path)
            return gau_data
        else:
            _avatar = self.generate_avatar_template()
            save_ply(_avatar, self.avatar_path)
            print("Generated and saved avatar template to:", self.avatar_path)
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
        all_normals = []
        all_shs = []
        all_sh_rest = []
        all_opacities = []
        all_scales = []
        all_rots = []
        all_parents = []

        for idx, face in enumerate(faces):
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            # generate gaussians for each face
            xyzs, normals, shs, sh_rest, opacity, scales, rots = (
                self.generate_gaussians_for_face(idx, v0, v1, v2)
            )
            all_xyz.append(xyzs)
            all_normals.append(normals)
            all_shs.append(shs)
            all_sh_rest.append(sh_rest)
            all_opacities.append(opacity)
            all_scales.append(scales)
            all_rots.append(rots)
            all_parents.extend([idx] * self.k_num_gaussians)

        if len(all_xyz) == 0:
            raise RuntimeError("No gaussians generated from mesh")

        all_xyz = np.concatenate(all_xyz, axis=0)
        all_normals = np.concatenate(all_normals, axis=0)
        all_shs = np.concatenate(all_shs, axis=0)
        all_sh_rest = np.concatenate(all_sh_rest, axis=0)
        all_opacities = np.concatenate(all_opacities, axis=0).reshape(-1, 1)
        all_scales = np.concatenate(all_scales, axis=0)
        all_rots = np.concatenate(all_rots, axis=0)
        parents = np.asarray(all_parents, dtype=np.int32)

        data = {
            "xyz": all_xyz,
            "normals": all_normals,
            "shs": all_shs,
            "sh_rest": all_sh_rest,
            "opacities": all_opacities,
            "scales": all_scales,
            "rots": all_rots,
            "parent": parents,
        }

        return data

    def generate_gaussians_for_face(self, face_id, v0, v1, v2):
        num_gaussians = self.k_num_gaussians
        center = (v0 + v1 + v2) / 3.0

        # Construct a local coordinate system for the face
        e1 = v1 - v0
        normal = np.cross(e1, v2 - v0)
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
        std_scale = np.array([r, r, 0.25 * r])  # Example scale with small depth

        # Initialize Gaussians (as numpy arrays)
        xyzs = np.zeros((num_gaussians, 3), dtype=np.float32)
        normals = np.zeros((num_gaussians, 3), dtype=np.float32)
        shs = np.zeros((num_gaussians, 3), dtype=np.float32)
        sh_rest = np.zeros((num_gaussians, 45), dtype=np.float32)
        opacity = np.zeros((num_gaussians, 1), dtype=np.float32)
        scales = np.zeros((num_gaussians, 3), dtype=np.float32)
        rots = np.zeros((num_gaussians, 4), dtype=np.float32)
        for i in range(self.k_num_gaussians):
            pt = B4[i][0] * v0 + B4[i][1] * v1 + B4[i][2] * v2
            xyzs[i, :] = pt
            normals[i, :] = [0, 0, 0]
            shs[i, :] = 0.5
            sh_rest[i, :] = 0.0
            opacity[i, 0] = 0.6
            scales[i, :] = std_scale
            rots[i, :] = matrix_to_quaternion(R)
        return xyzs, normals, shs, sh_rest, opacity, scales, rots


if __name__ == "__main__":
    avatar_template = AvatarTemplate()
    print("Avatar template generated and saved.")
