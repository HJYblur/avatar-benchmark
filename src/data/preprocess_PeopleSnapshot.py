"""
Adapted from SplattingAvatar
"""

import os
import pickle
from pathlib import Path
import numpy as np
import cv2
import h5py
import tqdm
import torch
import json


def load_pkl(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f, encoding="latin")


def load_h5py(fpath):
    return h5py.File(fpath, "r")


def process_people_snapshot(
    root: str | Path | None, subject: str, outdir: str | Path | None = None
):
    """Process a PeopleSnapshot subject and dump cameras, images, masks and poses.

    Defaults:
      - If `root` is None, assume repository root/data/ (calling file parent parents) as data root.
      - If `outdir` is None, write to repository root/processed/<subject>/.

    Returns the output directory Path on success.
    """
    # Resolve paths relative to repository layout: repo_root/data/<subject>
    repo_root = Path(__file__).resolve().parents[2]
    if root is None:
        root = os.path.join(str(repo_root), "data")

    dirpath = os.path.join(str(root), subject)
    if not os.path.exists(dirpath):
        raise FileNotFoundError(f"Cannot open {dirpath}")

    if outdir is None:
        outdir = os.path.join(str(repo_root), "processed", subject)
    os.makedirs(outdir, exist_ok=True)

    # load camera
    camera = load_pkl(os.path.join(dirpath, "camera.pkl"))
    K = np.eye(3)
    K[0, 0] = camera["camera_f"][0]
    K[1, 1] = camera["camera_f"][1]
    K[:2, 2] = camera["camera_c"]
    dist_coeffs = camera["camera_k"]

    H, W = camera["height"], camera["width"]
    w2c = np.eye(4)
    w2c[:3, :3] = cv2.Rodrigues(camera["camera_rt"])[0]
    w2c[:3, 3] = camera["camera_t"]

    camera_path = os.path.join(outdir, "cameras.npz")
    np.savez(
        camera_path,
        **{
            "intrinsic": K,
            "extrinsic": w2c,
            "height": H,
            "width": W,
        },
    )
    print("Write camera to", camera_path)
    with open(camera_path.replace(".npz", ".json"), "w") as fp:
        json.dump(
            {
                "intrinsic": K.tolist(),
                "extrinsic": w2c.tolist(),
                "height": H,
                "width": W,
            },
            fp,
        )
    torch.save(
        {
            "intrinsic": K,
            "extrinsic": w2c,
            "height": H,
            "width": W,
        },
        camera_path.replace(".npz", ".pt"),
    )

    # load images
    image_dir = os.path.join(outdir, "images")
    os.makedirs(image_dir, exist_ok=True)

    print("Write images to", image_dir)
    video_file = os.path.join(dirpath, f"{subject}.mp4")
    if os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm.trange(frame_cnt):
            img_path = os.path.join(image_dir, f"image_{i:04d}.png")
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.undistort(frame, K, dist_coeffs)
            cv2.imwrite(img_path, frame)
    else:
        print(f"No video file {video_file} found — skipping image extraction")

    # load masks
    mask_dir = os.path.join(outdir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    print("Write mask to", mask_dir)
    masks = np.asarray(load_h5py(os.path.join(dirpath, "masks.hdf5"))["masks"]).astype(
        np.uint8
    )
    for i, mask in enumerate(tqdm.tqdm(masks)):
        mask_path = os.path.join(mask_dir, f"mask_{i:04d}.png")
        mask = cv2.undistort(mask, K, dist_coeffs)

        # remove boundary artifact
        alpha = mask * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        alpha = cv2.erode(alpha, kernel)
        alpha = cv2.blur(alpha, (3, 3))
        cv2.imwrite(mask_path, alpha)

    smpl_params = load_h5py(os.path.join(dirpath, "reconstructed_poses.hdf5"))
    smpl_params = {
        "betas": np.asarray(smpl_params["betas"]).astype(np.float32),
        "thetas": np.asarray(smpl_params["pose"]).astype(np.float32),
        "transl": np.asarray(smpl_params["trans"]).astype(np.float32),
    }
    np.savez(os.path.join(outdir, "poses.npz"), **smpl_params)

    smpl_params_t = {
        "betas": torch.tensor(smpl_params["betas"]),
        "thetas": torch.tensor(smpl_params["thetas"]),
        "transl": torch.tensor(smpl_params["transl"]),
    }
    torch.save(smpl_params_t, os.path.join(outdir, "poses.pt"))

    return outdir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="path to the PeopleSnapshotData (defaults to repository 'data/' folder)",
    )
    parser.add_argument(
        "--subject", type=str, default="male-3-casual", help="sequence to process"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="path to output (defaults to repository 'processed')",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = process_people_snapshot(
        args.root, args.subject, os.path.join(args.outdir, args.subject)
    )
    print("Processed to:", out)
