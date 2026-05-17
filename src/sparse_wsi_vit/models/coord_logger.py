"""Drop-in replacement for SparseViT5SlideEncoder that only logs coordinate stats.

Usage: swap the net in your config:
    from sparse_wsi_vit.models.coord_logger import CoordLogger
    config.net = LazyConfig(CoordLogger)(out_features=OUT_FEATURES)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CoordLogger(nn.Module):
    def __init__(self, out_features: int = 1, **_) -> None:
        super().__init__()
        self.out_features = out_features
        self.head = nn.Linear(1, out_features)

    def forward(self, x: Tensor, coords: Tensor | None = None) -> dict[str, Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        num_patches = x.shape[1]

        if coords is not None:
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            print(
                f"[CoordLogger] patches={num_patches}"
                f"  sqrt={num_patches**0.5:.0f}"
                f"  x=[{coords[...,0].min().item():.0f}, {coords[...,0].max().item():.0f}]"
                f"  y=[{coords[...,1].min().item():.0f}, {coords[...,1].max().item():.0f}]"
                f"  amax={coords.amax().item():.0f}"
            )

        logits = torch.zeros(batch_size, self.out_features, device=x.device)
        return {"logits": logits}
