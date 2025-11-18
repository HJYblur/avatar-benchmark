import numpy as np
import os
from collections import namedtuple
from plyfile import PlyData, PlyElement


# Simple container for loaded Gaussian data (now includes parent index)
GaussianData = namedtuple(
    "GaussianData", ["xyz", "rots", "scales", "opacities", "shs", "parent"]
)


def load_ply(path):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
    )
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1 / (1 + np.exp(-opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate(
        [features_dc.reshape(-1, 3), features_extra.reshape(len(features_dc), -1)],
        axis=-1,
    ).astype(np.float32)
    shs = shs.astype(np.float32)
    # read parent if present
    parent = None
    if "parent" in plydata.elements[0].data.dtype.names:
        parent = np.asarray(plydata.elements[0]["parent"]).astype(np.int32)

    return GaussianData(xyz, rots, scales, opacities, shs, parent)


def save_ply(data, path: str):
    """Save Gaussian template to a PLY file.

    Expected `data` to be a dict-like or object with attributes:
      data['xyz']       -> (N,3) float
      data['rots']      -> (N,9) or (N,3,3) float
      data['scales']    -> (N,3) float
      data['opacities'] -> (N,1) or (N,) float
      data['shs']       -> (N,3) float (DC terms)
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

    rots = get_field("rots", 0.0)
    scales = get_field("scales", 1.0)
    opacities = get_field("opacities", 1.0).reshape(N, -1)
    shs = get_field("shs", 0.5).reshape(N, -1)  # expect DC RGB

    # ensure shapes
    if rots.ndim == 3 and rots.shape[1:] == (3, 3):
        rots_flat = rots.reshape(N, 9)
    else:
        rots_flat = rots.reshape(N, -1)

    scales = np.asarray(scales).reshape(N, -1)

    # Build structured array fields expected by load_ply
    # Basic fields
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("parent", "i4"),
    ]
    # sh DC
    vertex_dtype += [(f"f_dc_{i}", "f4") for i in range(3)]
    # placeholder extra SH coefficients (45 entries for degree <=3 per-channel minus DC)
    num_extra = 3 * (3 + 1) ** 2 - 3  # matches loader expectation
    vertex_dtype += [(f"f_rest_{i}", "f4") for i in range(num_extra)]
    # scales
    for i in range(scales.shape[1]):
        vertex_dtype.append((f"scale_{i}", "f4"))
    # rotation flattened
    for i in range(rots_flat.shape[1]):
        vertex_dtype.append((f"rot_{i}", "f4"))

    vertices = np.empty(N, dtype=vertex_dtype)
    vertices["x"] = xyz[:, 0].astype(np.float32)
    vertices["y"] = xyz[:, 1].astype(np.float32)
    vertices["z"] = xyz[:, 2].astype(np.float32)
    vertices["opacity"] = opacities.reshape(-1).astype(np.float32)
    # parent indices default to -1 (unknown)
    parent_arr = get_field("parent", -1).reshape(-1)
    vertices["parent"] = parent_arr.astype(np.int32)

    # SH DC
    vertices["f_dc_0"] = shs[:, 0].astype(np.float32)
    vertices["f_dc_1"] = shs[:, 1].astype(np.float32)
    vertices["f_dc_2"] = shs[:, 2].astype(np.float32)

    # f_rest placeholders (zeros)
    for i in range(num_extra):
        vertices[f"f_rest_{i}"] = 0.0

    # scales
    for i in range(scales.shape[1]):
        vertices[f"scale_{i}"] = scales[:, i].astype(np.float32)

    # rotation flattened
    for i in range(rots_flat.shape[1]):
        vertices[f"rot_{i}"] = rots_flat[:, i].astype(np.float32)

    ply_el = PlyElement.describe(vertices, "vertex")
    PlyData([ply_el], text=False).write(path)
