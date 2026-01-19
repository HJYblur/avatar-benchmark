import argparse
import logging
import sys
from pathlib import Path

import torch
import lightning as L

# Make 'src' importable when running as a script
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.datamodule import AvatarDataModule
from src.encoder.nlf_backbone_adapter import NLFBackboneAdapter
from src.encoder.identity_encoder import IdentityEncoder
from src.decoder.gaussian_decoder import GaussianDecoder
from src.training.trainer import Trainer
from src.utils.config import load_config


def main():
    # configure logging
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file)),
        ],
    )
    logger = logging.getLogger("train")
    logger.info("\n\n")
    logger.info("Starting training script")
    # Arg parsing
    parser = argparse.ArgumentParser(description="NLF-GS Training Scaffold")
    parser.add_argument("--config", type=str, default="configs/nlfgs_base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    # Force device to be on cpu
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Build datamodule
    dm = AvatarDataModule(cfg)
    dm.setup("fit")
    logger.info("DataModule setup complete")

    # Infer backbone feature dimensionality from a sample batch
    sample = next(iter(dm.train_dataloader()))
    sample_img = sample["image"]  # (B,3,H,W)
    # move sample to same device as the NLF model
    sample_img = sample_img.to(device)
    logger.info(f"Sample image tensor shape: {tuple(sample_img.shape)}")

    # Import nlf model
    # Load TorchScript model; force it to the chosen device
    nlf_checkpoint = torch.jit.load(
        cfg["nlf"]["checkpoint_path"], map_location=device
    ).eval()
    try:
        nlf_checkpoint.to(device)
    except Exception:
        # Some TorchScript modules may not implement .to(); that's okay
        raise RuntimeError("NLF model does not support .to() method.")

    # Log a small sanity-check about model param device/dtype (works with ScriptModule state_dict)
    try:
        sd = nlf_checkpoint.state_dict()
        first_tensor = next(iter(sd.values()))
        logger.info(
            f"NLF checkpoint params -> device={first_tensor.device}, dtype={first_tensor.dtype}"
        )
    except Exception as _e:
        logger.info(
            "Could not introspect NLF checkpoint state_dict for device/dtype check"
        )

    # Backbone Adapter Initialization
    backbone = NLFBackboneAdapter(nlf_checkpoint)
    logger.info("NLF Backbone Adapter initialized")
    c_local = int(cfg["nlf"].get("latent_dim", 512))
    logger.info(f"Backbone feature map channels: {c_local}")

    # Identity Encoder Initialization
    id_latent_dim = int(cfg["identity_encoder"].get("latent_dim", 64))
    id_encoder = IdentityEncoder(backbone_feat_dim=c_local, latent_dim=id_latent_dim)

    # Decoder Initialization
    decoder = GaussianDecoder(in_dim=(c_local + 3))  # local feature dim + coord3d dim
    logger.info(f"Decoder input dimension: {decoder.in_dim}")  # 512 + 3 = 515

    module = Trainer(
        backbone_adapter=backbone,
        identity_encoder=id_encoder,
        decoder=decoder,
    )

    max_epochs = int(cfg["train"]["epochs"]) if "train" in cfg else 1
    logger.info(f"Training for max_epochs={max_epochs}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        devices=int(cfg.get("trainer", {}).get("devices", 1)),
        accelerator=cfg.get("trainer", {}).get("accelerator", "cpu"),
        logger=False,
    )
    logger.info("Beginning trainer.fit()")
    trainer.fit(module, datamodule=dm)
    logger.info("trainer.fit() finished")


if __name__ == "__main__":
    main()
