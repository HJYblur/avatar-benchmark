from typing import Optional, Dict, Any

import math
import torch
from torch.utils.data import DataLoader, Subset
import lightning as L

from src.data.datasets import AvatarDataset


class AvatarDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping AvatarDataset with simple train/val split."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_ds: Optional[torch.utils.data.Dataset] = None
        self.val_ds: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None):
        data_cfg = self.cfg.get("data", {})
        train_cfg = self.cfg.get("train", {})

        base_ds = AvatarDataset(
            root=data_cfg.get("root", "processed"),
            proc_side=int(data_cfg.get("proc_side", 256)),
            template_path=data_cfg.get("template_path", "models/avatar_template.ply"),
        )

        n = len(base_ds)
        val_ratio = float(train_cfg.get("val_ratio", 0.0))
        if val_ratio > 0.0 and n > 1:
            n_val = max(1, int(math.floor(n * val_ratio)))
            idx = torch.randperm(n).tolist()
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
            if len(train_idx) == 0:  # fallback to at least one train sample
                train_idx, val_idx = idx[:-1], idx[-1:]
            self.train_ds = Subset(base_ds, train_idx)
            self.val_ds = Subset(base_ds, val_idx)
        else:
            self.train_ds = base_ds
            self.val_ds = None

    def train_dataloader(self) -> DataLoader:
        train_cfg = self.cfg.get("train", {})
        return DataLoader(
            self.train_ds,
            batch_size=int(train_cfg.get("batch_size", 2)),
            num_workers=int(train_cfg.get("num_workers", 2)),
            shuffle=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        train_cfg = self.cfg.get("train", {})
        return DataLoader(
            self.val_ds,
            batch_size=int(train_cfg.get("batch_size", 2)),
            num_workers=int(train_cfg.get("num_workers", 2)),
            shuffle=False,
        )
