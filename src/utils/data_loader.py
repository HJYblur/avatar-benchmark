import pickle
import h5py


def load_smpl_shape(shape_file_path) -> dict:
    """Load SMPL shape data from specified files.

    Args:
        shape_file_path (str): Path to the shape data file (pickle format).
    Returns:
        dict: Dictionary containing shape data.
    """
    with open(shape_file_path, "rb") as f:
        smpl_params = pickle.load(f, encoding="latin1")
    return smpl_params


def load_smpl_pose(pose_file_path) -> dict:
    """Load SMPL pose data from specified files.

    Args:
        pose_file_path (str): Path to the pose data file (.hdf5 format).
    Returns:
        dict: Dictionary containing pose data.
    """
    poses = {}
    with h5py.File(pose_file_path, "r") as f:
        for k, v in f.items():
            poses[k] = v[:]
    return poses
