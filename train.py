import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as L

# Make 'src' importable when running as a script
sys.path.append(str(Path(__file__).parent / "src"))

from src.encoder.nlf_backbone_adapter import NLFBackboneAdapter
from src.encoder.identity_encoder import IdentityEncoder
from src.decoder.gaussian_decoder import GaussianDecoder
from src.training.trainer import Trainer
from src.utils.config import load_config


def main():
    # Arg parsing
    parser = argparse.ArgumentParser(description="NLF-GS Training Scaffold")
    parser.add_argument("--config", type=str, default="configs/nlfgs_base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # TODO: Build dataset and dataloader
    # ds = Dataset()
    # dl = DataLoader()

    # Import nlf model
    nlf_checkpoint = torch.jit.load(cfg["nlf"]["checkpoint_path"])
    backbone = NLFBackboneAdapter(nlf_checkpoint)

    # Infer backbone feature dimensionality from a sample batch
    sample = next(iter(dl))
    sample_img = sample["image"]  # (B,3,H,W)
    with torch.no_grad():
        feats = backbone.extract_feature_map(sample_img, use_half=True)
    c_local = int(feats.shape[1])

    id_latent_dim = int(cfg["identity_encoder"]["latent_dim"])
    id_encoder = IdentityEncoder(backbone_feat_dim=c_local, latent_dim=id_latent_dim)
    decoder = GaussianDecoder(in_dim=(id_latent_dim + c_local + 3))

    module = Trainer(
        backbone_adapter=backbone,
        identity_encoder=id_encoder,
        decoder=decoder,
    )

    max_epochs = int(cfg["train"]["epochs"]) if "train" in cfg else 1
    trainer = L.Trainer(
        max_epochs=max_epochs, devices=1, accelerator="auto", logger="wandb"
    )
    trainer.fit(module, train_dataloaders=dl)


if __name__ == "__main__":
    main()
