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
    root: str | Path | None,
    subject: str,
    outdir: str | Path | None = None,
    use_mask: bool = False,
    save_mask: bool = True,
):
    """Process a PeopleSnapshot subject and dump cameras, images, masks and poses.

    out = process_people_snapshot(
        args.root,
        args.subject,
        os.path.join(args.outdir, args.subject),
        use_mask=args.use_mask,
        save_mask=args.save_mask,
    )

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

    # load masks (we load early so we can optionally apply them when saving images)
    masks = None
    masks_path = os.path.join(dirpath, "masks.hdf5")
    if os.path.exists(masks_path):
        try:
            masks = np.asarray(load_h5py(masks_path)["masks"]).astype(np.uint8)
            print(f"Loaded {len(masks)} masks from {masks_path}")
        except Exception:
            masks = None

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
            # optionally apply mask to cover background
            if use_mask and masks is not None and i < masks.shape[0]:
                try:
                    mask = masks[i]
                    # create a smoothed alpha like in the mask saving step
                    alpha = (mask * 255).astype(np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    alpha = cv2.erode(alpha, kernel)
                    alpha = cv2.blur(alpha, (3, 3))
                    if alpha.ndim == 2 and frame.ndim == 3:
                        alpha3 = np.stack([alpha] * frame.shape[2], axis=2)
                    else:
                        alpha3 = alpha
                    # set background to pure white where mask is zero
                    mask_f = alpha3.astype(np.float32) / 255.0
                    frame = (
                        frame.astype(np.float32) * mask_f + 255.0 * (1.0 - mask_f)
                    ).astype(np.uint8)
                except Exception:
                    # fallback: write unmasked frame
                    pass
            cv2.imwrite(img_path, frame)
    else:
        print(f"No video file {video_file} found — skipping image extraction")

    # write masks to disk (optional)
    if save_mask:
        mask_dir = os.path.join(outdir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        print("Write mask to", mask_dir)
        # use already-loaded masks if available
        if masks is None:
            if os.path.exists(masks_path):
                masks = np.asarray(load_h5py(masks_path)["masks"]).astype(np.uint8)
            else:
                masks = None

        if masks is not None:
            for i, mask in enumerate(tqdm.tqdm(masks)):
                mask_path = os.path.join(mask_dir, f"mask_{i:04d}.png")
                mask = cv2.undistort(mask, K, dist_coeffs)

                # remove boundary artifact
                alpha = mask * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                alpha = cv2.erode(alpha, kernel)
                alpha = cv2.blur(alpha, (3, 3))
                cv2.imwrite(mask_path, alpha)
        else:
            print(f"No masks found at {masks_path} — skipping mask saving")

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
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=True,
        help="If set, images will be saved with masks applied (background covered).",
    )
    # single flag to control whether mask PNGs get saved (default True)
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=False,
        help="Save mask PNGs to the output folder (default).",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = process_people_snapshot(
        args.root,
        args.subject,
        os.path.join(args.outdir, args.subject),
        use_mask=args.use_mask,
        save_mask=args.save_mask,
    )
    print("Processed to:", out)
