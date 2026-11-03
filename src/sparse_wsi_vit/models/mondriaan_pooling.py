"""Arrangement-sensitive pooling controls for the lossless Mondriaan inputs.

Coordinates, when enabled, are fixed patch-grid coordinates. No object identities,
corners, scene memberships, or labels enter the encoder.
"""

import torch
from torch import nn


class LearnedPooling(nn.Module):
    """A deeper per-token encoder followed by sum/mean and a nonlinear head."""

    def __init__(self, width=768, features=256, aggregation="mean",
                 use_coordinates=False, coordinate_scale=32.0, dropout=0.1):
        super().__init__()
        if aggregation not in ("mean", "sum"):
            raise ValueError("aggregation must be mean or sum")
        if coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive")
        self.aggregation = aggregation
        self.use_coordinates = use_coordinates
        self.coordinate_scale = float(coordinate_scale)
        self.encoder = nn.Sequential(
            nn.Linear(50 if use_coordinates else 48, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, features), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(features, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, 2),
        )

    def forward(self, x, coords, lengths=None):
        if x.ndim != 3 or x.shape[-1] != 48:
            raise ValueError("Expected (batch, tokens, 48) lossless patch pixels")
        if coords.shape != (*x.shape[:2], 2):
            raise ValueError("Coordinates must match the patch grid")
        if lengths is not None:
            if lengths.shape != (len(x),) or (lengths <= 0).any():
                raise ValueError("lengths must contain positive bag lengths")
            if (lengths > x.shape[1]).any():
                raise ValueError("length exceeds padded token dimension")
            mask = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
            denominator = lengths[:, None].to(x.dtype)
        else:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            denominator = x.shape[1]
        foreground = (x != -1).any(dim=-1) & mask
        if self.use_coordinates:
            # Background appearance is constant, and grid coordinates repeat
            # across the batch. Keep exact (batch, coordinate) multiplicities.
            # Coordinates are observed constants, not differentiable inputs.
            background_mask = mask & ~foreground
            pooled = x.new_zeros((len(x), self.encoder[-2].out_features))
            if background_mask.any():
                unique_coords, inverse = torch.unique(
                    coords[background_mask], dim=0, return_inverse=True)
                bg_inputs = torch.cat([x.new_full((len(unique_coords), 48), -1),
                                       unique_coords / self.coordinate_scale], dim=-1)
                indices = background_mask.nonzero()[:, 0] * len(unique_coords) + inverse
                weights = torch.bincount(indices, minlength=len(x) * len(unique_coords))
                weights = weights.reshape(len(x), len(unique_coords)).to(x.dtype)
                pooled = weights @ self.encoder(bg_inputs)
            if foreground.any():
                inputs = torch.cat([x[foreground], coords[foreground] / self.coordinate_scale], dim=-1)
                pooled = pooled.index_add(0, foreground.nonzero()[:, 0], self.encoder(inputs))
        else:
            # All sentinel tokens have exactly the same encoded value. Encode
            # it once, retaining its exact multiplicity, rather than doing the
            # same large MLP computation for every background patch. No pixels
            # or background tokens are dropped by this equivalent evaluation.
            background_counts = (mask & ~foreground).sum(dim=1, keepdim=True)
            background = self.encoder(x.new_full((1, 48), -1))
            pooled = background_counts * background
            if foreground.any():
                batch_indices = foreground.nonzero()[:, 0]
                pooled = pooled.index_add(0, batch_indices, self.encoder(x[foreground]))
        if self.aggregation == "mean":
            pooled = pooled / denominator
        return {"logits": self.classifier(pooled)}


def pooling_parameter_count(width, features=256, use_coordinates=False):
    """Count parameters without allocating a sequence of candidate models."""
    sizes = [(50 if use_coordinates else 48, width), (width, width),
             (width, width), (width, features), (features, width),
             (width, width), (width, 2)]
    return sum((inputs + 1) * outputs for inputs, outputs in sizes)


def comparable_pooling_width(target_parameters=2_381_954, features=256,
                             use_coordinates=False):
    """Choose the nearest width divisible by eight; report actual counts too."""
    return min(range(8, 2049, 8), key=lambda width: abs(
        pooling_parameter_count(width, features, use_coordinates) - target_parameters))
