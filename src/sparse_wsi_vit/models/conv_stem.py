import math

import torch
import torch.nn as nn

from sparse_wsi_vit.experiments.utils.lazy_config import instantiate


class ConvStem(nn.Module):
    """Learned k x k conv (stride 1) on a full square grid of pixel tokens, then `net`.

    Keeps one token per pixel; each token becomes a learned function of its k x k
    neighbourhood. Tokens must be the row-major grid written by extract_mondriaan.py.
    `net` is the lazy config of the model after the stem; like run.py does for
    config.net, ConvStem sets its in_features (to dim) and out_features.
    """

    def __init__(
        self,
        net,
        in_features: int,
        out_features: int,
        dim: int = 256,
        kernel: int = 5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_features, dim, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.net = instantiate(net, in_features=dim, out_features=out_features)
        self.out_features = out_features  # read by the Lightning wrappers

    def forward(self, x: torch.Tensor, *args, **kwargs) -> dict:
        B, N, C = x.shape
        side = math.isqrt(N)
        x = x.transpose(1, 2).reshape(B, C, side, side)
        x = self.stem(x).flatten(2).transpose(1, 2)
        return self.net(x, *args, **kwargs)
