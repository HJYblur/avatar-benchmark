import argparse
import logging
import sys
import os
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
from src.avatar_utils.config import load_config


def main():
    # Arg parsing
    parser = argparse.ArgumentParser(description="NLF-GS Training Scaffold")
    parser.add_argument("--config", type=str, default="configs/nlfgs_base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    debug = (
        bool(cfg.get("sys", {}).get("debug", False)) if isinstance(cfg, dict) else False
    )
    logger = setup_logger(debug)
    logger.info("Starting training script")
    logger.info(f"Loaded config: {args.config}")

    # Determine device from config (fallback to cpu)
    device_str = None
    if isinstance(cfg, dict):
        device_str = cfg.get("sys", {}).get("device") if cfg.get("sys") else None
    # Fallback to environment variable or cpu
    if not device_str:
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device from config: {device}")

    # Build datamodule
    dm = AvatarDataModule(cfg)
    dm.setup("fit")
    logger.info("DataModule setup complete")

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
        logger.debug(
            f"NLF checkpoint params -> device={first_tensor.device}, dtype={first_tensor.dtype}"
        )
    except Exception as _e:
        logger.debug(
            "Could not introspect NLF checkpoint state_dict for device/dtype check"
        )

    # Backbone Adapter Initialization
    backbone = NLFBackboneAdapter(nlf_checkpoint)
    logger.info("NLF Backbone Adapter initialized")
    c_local = int(cfg["nlf"].get("latent_dim", 512))
    logger.debug(f"Backbone feature map channels: {c_local}")

    # Identity Encoder Initialization
    id_latent_dim = int(cfg["identity_encoder"].get("latent_dim", 64))
    id_encoder = IdentityEncoder(backbone_feat_dim=c_local, latent_dim=id_latent_dim)

    # Decoder Initialization
    decoder = GaussianDecoder()  # local feature dim + coord3d dim
    logger.info(f"Decoder Initialized.")  # 512 + 3 = 515

    module = Trainer(
        backbone_adapter=backbone,
        identity_encoder=id_encoder,
        decoder=decoder,
        train_decoder_only=True,
    )

    max_epochs = int(cfg["train"]["epochs"]) if "train" in cfg else 1
    logger.info(f"Training for max_epochs={max_epochs}")

    # Trainer precision and accelerator
    precision = cfg.get("train", {}).get("precision")

    # Improve CUDA memory behavior unless user overrides
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    trainer = L.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator=cfg.get("train", {}).get("accelerator", "cpu"),
        precision=precision if precision else None,
        logger=False,
    )
    logger.info("Beginning trainer.fit()")
    trainer.fit(module, datamodule=dm)
    logger.info("trainer.fit() finished")


def setup_logger(debug: bool = False) -> logging.Logger:
    """Initialize and return the project logger.

    If debug is True, set level to DEBUG; otherwise INFO. Writes logs to both stdout
    and a file under ./logs/train.log.
    """
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "train.log"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file)),
        ],
    )
    return logging.getLogger("train")


if __name__ == "__main__":
    main()
