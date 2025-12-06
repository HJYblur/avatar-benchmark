from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader


class Trainer:
    """
    Minimal training scaffold that demonstrates wiring:
    - Loads batches from a DataLoader
    - Encodes features with NLF backbone adapter
    - Placeholder for localization + gaussian estimation + losses
    """

    def __init__(
        self,
        backbone_adapter: Any,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> None:
        self.backbone = backbone_adapter
        self.dataloader = dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_one_epoch(self) -> Dict[str, float]:
        metrics = {"batches": 0}
        for batch in self.dataloader:
            image = batch["image"].to(self.device)
            # intrinsics = batch["intrinsics"].to(self.device)  # Reserved for localization
            # canonical_points = batch["canonical_points"].to(self.device)

            with torch.no_grad():
                feats = self.backbone(image, use_half=True)

            # Placeholders for downstream modules
            _ = feats  # noqa: F841

            metrics["batches"] += 1
        return metrics

