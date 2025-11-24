import numpy as np
import os
from collections import namedtuple
from plyfile import PlyData, PlyElement


# Simple container for loaded Gaussian data (now includes parent index)
GaussianData = namedtuple(
    "GaussianData", ["xyz", "rots", "scales", "opacities", "shs", "parent"]
)


def matrix_to_quaternion(matrix):
    # Convert a rotation matrix to a quaternion.
    m = matrix.reshape(3, 3)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def load_cano_mesh_face_center(self):
    if not os.path.exists(self.cano_mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {self.cano_mesh_path}")
    mesh = trimesh.load(self.cano_mesh_path)
    vertices, faces = mesh.vertices, mesh.faces

    face_center = np.zeros((len(faces), 3), dtype=np.float32)
    for idx, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        center = (v0 + v1 + v2) / 3.0
        face_center[idx, :] = center
    return face_center


def load_ply(path, mode="default", cano_mesh=None) -> GaussianData:
    """Load Gaussian template PLY saved by save_ply.

    This loader is tolerant: it reads whatever fields are present (DC SH, optional
    extra SH coefficients, scale_*, rot_*, optional parent) and returns a
    GaussianData object. Coordinates are negated back to the original sign
    because save_ply stores x,y,z with flipped signs.
    """
    plydata = PlyData.read(path)
    elem = plydata.elements[0]
    names = elem.data.dtype.names

    # Positions:
    # If in "test" mode, we reload the xyz from face based local coords to world coords and negate it for visualization
    if mode == "test":
        if cano_mesh is None:
            raise ValueError("cano_mesh must be provided in test mode to reload xyz.")
        # Here we would implement the logic to convert local coords to world coords
        vertices, faces = cano_mesh.vertices, cano_mesh.faces

        face_center = np.zeros((len(faces), 3), dtype=np.float32)
        for idx, face in enumerate(faces):
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            center = (v0 + v1 + v2) / 3.0
            face_center[idx, :] = center

        for idx in range(len(elem.data)):
            parent_face_idx = elem.data["parent"][idx]
            cur_face_center = face_center[parent_face_idx]
            elem.data["x"][idx] = -(elem.data["x"][idx] + cur_face_center[0])
            elem.data["y"][idx] = -(elem.data["y"][idx] + cur_face_center[1])
            elem.data["z"][idx] = -(elem.data["z"][idx] + cur_face_center[2])

    # Else in "default" mode, just load it. We need it in local coords for further processing
    xyz = np.stack(
        (np.asarray(elem["x"]), np.asarray(elem["y"]), np.asarray(elem["z"])),
        axis=1,
    ).astype(np.float32)

    # Opacity stored directly (0..1), keep as column vector
    opacities = np.asarray(elem["opacity"])[..., np.newaxis].astype(np.float32)

    # SH DC (f_dc_0..2) if present, otherwise zeros
    dc = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    for i in range(3):
        key = f"f_dc_{i}"
        if key in names:
            dc[:, i] = np.asarray(elem[key]).astype(np.float32)

    # Optional extra SH coefficients (f_rest_*) – read if present and keep order
    extra_f_names = [n for n in names if n.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    if len(extra_f_names) > 0:
        extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
        for idx, k in enumerate(extra_f_names):
            extra[:, idx] = np.asarray(elem[k]).astype(np.float32)
        shs = np.concatenate([dc, extra], axis=1)
    else:
        shs = dc

    # scales (scale_0.. ) if present; default to ones
    scale_names = [n for n in names if n.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    if len(scale_names) > 0:
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, k in enumerate(scale_names):
            scales[:, idx] = np.asarray(elem[k]).astype(np.float32)
    else:
        scales = np.ones((xyz.shape[0], 3), dtype=np.float32)

    # rotations (rot_0..rot_M) if present
    rot_names = [n for n in names if n.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    if len(rot_names) > 0:
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, k in enumerate(rot_names):
            rots[:, idx] = np.asarray(elem[k]).astype(np.float32)
        # normalize
        nrm = np.linalg.norm(rots, axis=-1, keepdims=True)
        nrm[nrm == 0] = 1.0
        rots = (rots / nrm).astype(np.float32)
    else:
        rots = np.zeros((xyz.shape[0], 4), dtype=np.float32)

    # optional parent
    parent = None
    if "parent" in names:
        parent = np.asarray(elem["parent"]).astype(np.int32)

    return GaussianData(xyz, rots, scales, opacities, shs, parent)


def save_ply(data, path: str):
    """Save Gaussian template to a PLY file.

    Expected `data` to be a dict-like or object with attributes:
      data['xyz']       -> (N,3) float
      ----- data['normals']   -> (N,3) float
      data['shs']       -> (N,3) float (DC terms)
      ----- data['sh_rest']   -> (N,45) float (extra SH coeffs)
      data['opacities'] -> (N,1) or (N,) float
      data['scales']    -> (N,3) float
      data['rots']      -> (N,4) float (quaternion)
      data['parent']    -> (N,) int

    The saved PLY will include placeholder SH coefficients (f_rest_0..f_rest_44) set to zero
    so that loaders expecting a fixed number of SH extras can read the file.
    """
    # Normalize accessors
    if hasattr(data, "xyz"):
        xyz = np.asarray(data.xyz)
    elif isinstance(data, dict):
        xyz = np.asarray(data["xyz"])
    else:
        raise ValueError("Unsupported data type for save_ply")

    N = xyz.shape[0]

    def get_field(name, default, shape=None):
        if hasattr(data, name):
            v = np.asarray(getattr(data, name))
        elif isinstance(data, dict) and name in data:
            v = np.asarray(data[name])
        else:
            v = np.full((N,) + (() if shape is None else shape), default)
        return v

    shs = get_field("shs", 0.5, shape=(3,)).reshape(N, -1)  # DC RGB
    opacities = get_field("opacities", 1.0).reshape(N, -1)
    scales = get_field("scales", 1.0, shape=(3,)).reshape(N, -1)
    rots = get_field("rots", 0.0, shape=(4,)).reshape(N, -1)

    # Parent is optional: include only if provided by the data
    parent_present = hasattr(data, "parent") or (
        isinstance(data, dict) and "parent" in data
    )
    parent_arr = None
    if parent_present:
        parent_arr = get_field("parent", -1).reshape(-1)

    # Build structured dtype: position, opacity, parent, SH DC, scales, rot
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
    ]
    # For visualization/debugging purposes we exclude parent
    if parent_present:
        vertex_dtype.append(("parent", "i4"))

    # SH DC
    vertex_dtype += [(f"f_dc_{i}", "f4") for i in range(3)]

    # scales (scale_0..)
    for i in range(scales.shape[1]):
        vertex_dtype.append((f"scale_{i}", "f4"))

    # rotation (rot_0..rot_M)
    for i in range(rots.shape[1]):
        vertex_dtype.append((f"rot_{i}", "f4"))

    # Create structured array and fill fields
    vertices = np.empty(N, dtype=vertex_dtype)
    vertices["x"] = xyz[:, 0].astype(np.float32)
    vertices["y"] = xyz[:, 1].astype(np.float32)
    vertices["z"] = xyz[:, 2].astype(np.float32)
    vertices["opacity"] = opacities.reshape(-1).astype(np.float32)
    if parent_present:
        vertices["parent"] = parent_arr.astype(np.int32)
        # print("Get parent info")

    # SH DC
    vertices["f_dc_0"] = shs[:, 0].astype(np.float32)
    vertices["f_dc_1"] = shs[:, 1].astype(np.float32)
    vertices["f_dc_2"] = shs[:, 2].astype(np.float32)

    # scales
    for i in range(scales.shape[1]):
        vertices[f"scale_{i}"] = scales[:, i].astype(np.float32)

    # rotation
    for i in range(rots.shape[1]):
        vertices[f"rot_{i}"] = rots[:, i].astype(np.float32)

    ply_el = PlyElement.describe(vertices, "vertex")
    PlyData([ply_el], text=False).write(path)
