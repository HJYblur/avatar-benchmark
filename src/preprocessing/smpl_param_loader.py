import os
import torch
import pickle


def load_pkl(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f, encoding="latin")


def load_smpl_params(file_path: str) -> dict:
    """
    Load SMPL parameters from a .pkl or .pt file.

    Args:
        file_path: Path to the .pkl or .pt file containing SMPL parameters.

    Returns:
        A dictionary containing the SMPL parameters.
    """
    if file_path.endswith(".pkl"):
        pkl_data = load_pkl(file_path)
        return torch.tensor(pkl_data)
    elif file_path.endswith(".pt"):
        return torch.load(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .pkl or .pt file.")
