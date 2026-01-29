import os
from pathlib import Path
import numpy as np
import trimesh
import pyrender
from PIL import Image


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a 4x4 camera-to-world pose matrix.

    This matches the common OpenGL/pyrender convention where the camera looks
    towards the negative Z axis. The returned matrix places the camera at
    `eye` and orients it to look at `target` with the given `up` vector.

    Args:
        eye: (3,) camera position in world coordinates.
        target: (3,) point the camera looks at.
        up: (3,) up direction for the camera.

    Returns:
        (4,4) homogeneous transformation matrix (camera -> world).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    # forward vector (camera's local Z) points from target to eye
    z = eye - target
    z = z / np.linalg.norm(z)

    # right vector (camera's local X)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        # up was parallel to viewing direction; pick a different up
        up_alt = np.array([0.0, 0.0, 1.0])
        x = np.cross(up_alt, z)
    x = x / np.linalg.norm(x)

    # true up (camera's local Y)
    y = np.cross(z, x)

    mat = np.eye(4, dtype=float)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return mat


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "THuman_2.0"
OUT_ROOT = Path(__file__).resolve().parents[2] / "processed"
IMAGE_SIZE = (1024, 1024)
VIEWPOINTS = {
    "front": np.array([0.0, 0.0, 1.0]),
    "back": np.array([0.0, 0.0, -1.0]),
    "left": np.array([-1.0, 0.0, 0.0]),
    "right": np.array([1.0, 0.0, 0.0]),
}


def _iter_identities(root: Path):
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            obj_files = list(entry.glob("*.obj"))
            if obj_files:
                yield entry.name, obj_files[0]


def _find_texture_for_obj(obj_path: Path) -> Path | None:
    """Best-effort lookup for the per-identity texture image.

    THuman identities typically have a single *.jpeg/*jpg texture next to the OBJ.
    """
    for pattern in ("*.jpeg", "*.jpg", "*.png"):
        matches = sorted(obj_path.parent.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_mesh(obj_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force="scene", process=False)
    if isinstance(mesh, trimesh.Scene):
        # `Scene.dump(concatenate=True)` is deprecated; use `to_geometry()`.
        combined = trimesh.scene.scene.Scene()
        combined.add_geometry(mesh.to_geometry())
        return combined.to_geometry()
    return mesh


def _mesh_to_pyrender(
    mesh: trimesh.Trimesh, texture_path: Path | None
) -> pyrender.Mesh:
    """Convert trimesh to pyrender mesh, applying a texture if provided."""
    if texture_path is not None and texture_path.exists():
        # Create a simple PBR material with the provided baseColor texture.
        tex_img = Image.open(texture_path).convert("RGB")
        # pyrender expects a numpy array or path for the texture source.
        tex_data = np.asarray(tex_img)
        # specify source_channels ('RGB' here) because some pyrender
        # versions require an explicit channel specification.
        tex = pyrender.Texture(source=tex_data, source_channels="RGB")
        material = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=tex,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
        )
        return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    # Fall back to trimesh visuals (vertex colors / embedded materials) if present.
    if getattr(mesh, "visual", None) is not None:
        return pyrender.Mesh.from_trimesh(mesh, smooth=True)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def _camera_pose(
    direction: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    distance = radius * 2.5
    eye = center + direction * distance
    up = np.array([0.0, 1.0, 0.0])
    if np.allclose(np.cross(up, direction), 0.0):
        up = np.array([0.0, 0.0, 1.0])
    return look_at(eye, center, up)


def _render_views(
    mesh: trimesh.Trimesh, out_dir: Path, texture_path: Path | None, identity: str
):
    scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(_mesh_to_pyrender(mesh, texture_path))

    bbox = mesh.bounds
    center = bbox.mean(axis=0)
    radius = np.linalg.norm(bbox[1] - bbox[0]) / 2.0

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    renderer = pyrender.OffscreenRenderer(*IMAGE_SIZE)
    try:
        for name, direction in VIEWPOINTS.items():
            pose = _camera_pose(direction, center, radius)
            cam_node = scene.add(camera, pose=pose)
            light_node = scene.add(light, pose=pose)

            color, depth = renderer.render(scene)
            # Save color image with identity + view in the filename
            Image.fromarray(color).save(out_dir / f"{identity}_{name}.png")

            # Generate a foreground mask from depth (valid, >0)
            if depth is not None:
                mask_bool = np.isfinite(depth) & (depth > 0)
                mask_img = mask_bool.astype(np.uint8) * 255
                Image.fromarray(mask_img).save(out_dir / f"{identity}_{name}_mask.png")

            scene.remove_node(cam_node)
            scene.remove_node(light_node)
    finally:
        renderer.delete()


def preprocess_thuman(data_root: Path = DATA_ROOT, out_root: Path = OUT_ROOT):
    os.makedirs(out_root, exist_ok=True)
    for identity, obj_path in _iter_identities(data_root):
        target_dir = out_root / identity
        target_dir.mkdir(parents=True, exist_ok=True)
        mesh = _load_mesh(obj_path)
        texture_path = _find_texture_for_obj(obj_path)
        _render_views(mesh, target_dir, texture_path, identity)
        print(f"Rendered {identity}")


def main():
    preprocess_thuman()


if __name__ == "__main__":
    main()
