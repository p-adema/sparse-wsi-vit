"""Small CNN encoder for HalliGalli synthetic shape patches.

Architecture (input 64×64 RGB, values in [0, 1]):
    4 conv blocks: 3→16→32→64→128, each Conv+BN+ReLU+MaxPool2
    AdaptiveAvgPool2d(1) → 128-d
    Linear(128, embed_dim) ← penultimate (feature vector used at inference)
    Linear(embed_dim, num_classes) ← classification head (dropped at inference)

At inference time, call forward_features() to get the embed_dim-d embedding.
The head is only used during training.
"""

import torch
import torch.nn as nn


class ShapePatchCNN(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # 32×32 → 16×16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # 16×16 → 8×8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # 8×8 → 4×4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(128, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, embed_dim) embedding without the classification head."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.embed(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
