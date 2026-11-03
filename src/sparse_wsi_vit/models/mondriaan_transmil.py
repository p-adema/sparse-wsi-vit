"""TransMIL on the complete physical grid of lossless Mondriaan patches.

The two 512-dimensional Nyström layers and PPEG follow the official TransMIL
implementation. PPEG sees the real row-major grid, rather than square-padding
an arbitrary bag. Nyström's internal sequence padding remains unchanged.
"""

import torch
from nystrom_attention import NystromAttention
from torch import nn

from sparse_wsi_vit.models.transmil import PPEG


class _TransLayer(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.attn = NystromAttention(
            dim=512, dim_head=64, heads=8, num_landmarks=num_landmarks,
            pinv_iterations=6, residual=True, dropout=0.1,
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class MondriaanTransMIL(nn.Module):
    """Spatial correlated MIL without supplied object or scene membership.

    Coordinates are physical patch top-left corners in (x, y) order. The input
    must cover a complete rectangular grid starting at (0, 0), with spacing
    ``patch_size``. Coordinate/token permutations are canonicalized before any
    order-sensitive operation. The default short-sequence setting of 32
    landmarks is a computational adaptation; the stock setting is 256.
    """

    def __init__(self, in_features=48, out_features=2, num_landmarks=32, patch_size=4):
        super().__init__()
        if in_features != 48 or out_features != 2:
            raise ValueError("Mondriaan TransMIL requires 48 input features and 2 classes")
        if not isinstance(num_landmarks, int) or num_landmarks < 1:
            raise ValueError("num_landmarks must be a positive integer")
        if patch_size != 4:
            raise ValueError("48-value RGB tokens require patch_size=4")
        self.num_landmarks = num_landmarks
        self.patch_size = patch_size
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = _TransLayer(num_landmarks)
        self.layer2 = _TransLayer(num_landmarks)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, out_features)

    def order_grid(self, x, coords, lengths=None):
        """Return physical row-major tokens and (height, width), or reject holes."""
        if x.ndim != 3 or x.shape[-1] != 48 or min(x.shape[:2]) < 1:
            raise ValueError("Expected nonempty (batch, tokens, 48) lossless patches")
        if coords.shape != (*x.shape[:2], 2):
            raise ValueError("Coordinates must match the patch tokens")
        if coords.device != x.device:
            raise ValueError("Coordinates and patch tokens must share a device")
        if lengths is not None:
            if lengths.shape != (len(x),) or not torch.all(lengths == x.shape[1]):
                raise ValueError("TransMIL requires an unpadded complete physical grid")
        if not torch.isfinite(coords).all() or (coords < 0).any():
            raise ValueError("Grid coordinates must be finite and nonnegative")
        cells = coords / self.patch_size
        if not torch.equal(cells, cells.round()):
            raise ValueError("Coordinates must lie on the physical 4-pixel patch grid")
        cells = cells.to(torch.long)
        dimensions = cells.amax(dim=1) + 1
        if not torch.all(dimensions == dimensions[:1]):
            raise ValueError("All samples must have the same physical grid shape")
        width, height = dimensions[0].tolist()
        if width * height != x.shape[1]:
            raise ValueError("Missing patches: a complete rectangular grid is required")
        keys = cells[..., 1] * width + cells[..., 0]
        ordered_keys, order = keys.sort(dim=1)
        expected = torch.arange(x.shape[1], device=x.device)[None].expand_as(keys)
        if not torch.equal(ordered_keys, expected):
            raise ValueError("Duplicate or missing patch coordinates")
        ordered = x.gather(1, order[..., None].expand_as(x))
        return ordered, height, width

    def forward(self, x, coords, lengths=None):
        patches, height, width = self.order_grid(x, coords, lengths)
        h = self._fc1(patches.float())
        h = torch.cat((self.cls_token.expand(len(h), -1, -1), h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, height, width)
        h = self.layer2(h)
        logits = self._fc2(self.norm(h)[:, 0])
        return {"logits": logits, "Y_prob": logits.softmax(dim=1),
                "Y_hat": logits.argmax(dim=1)}
