import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make 'src' importable when running as a script
sys.path.append(str(Path(__file__).parent / "src"))

from data.datasets import AvatarDataset
from models.nlf_backbone_adapter import NLFBackboneAdapter
from training.trainer import Trainer
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="NLF-GS Training Scaffold")
    parser.add_argument("--config", type=str, default="configs/nlfgs_base.yaml")
    parser.add_argument(
        "--nlf_model_module",
        type=str,
        default="",
        help="Python import path to a module that exposes 'nlf_model' with crop_model.backbone. If empty, training will not run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Build dataset and dataloader
    ds = AvatarDataset(
        root=cfg["data"]["root"],
        image_glob=cfg["data"]["image_glob"],
        fov_degrees=cfg["data"]["fov_degrees"],
        proc_side=cfg["data"]["proc_side"],
        template_path=cfg["data"]["template_path"],
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=True,
    )

    if not args.nlf_model_module:
        print(
            "Scaffold ready. Provide --nlf_model_module to run (must expose 'nlf_model')."
        )
        return

    # Dynamically import nlf model handle without touching nlf/ package here
    mod = __import__(args.nlf_model_module, fromlist=["nlf_model"])  # type: ignore
    nlf_model = getattr(mod, "nlf_model")
    backbone = NLFBackboneAdapter(nlf_model)

    trainer = Trainer(backbone_adapter=backbone, dataloader=dl)
    for _ in range(cfg["train"]["epochs"]):
        metrics = trainer.train_one_epoch()
        print({k: int(v) if isinstance(v, int) else v for k, v in metrics.items()})


if __name__ == "__main__":
    main()
