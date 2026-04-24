from math import pi
import torch.nn.functional as F
import math

import torch
from torch import nn

from einops import rearrange, repeat
import numpy as np


def broadcat(freqss, dim=-1):
    num_freqss = len(freqss)
    shape_lens = set(list(map(lambda t: len(t.shape), freqss)))
    if not torch.compiler.is_compiling():
        assert len(shape_lens) == 1, "freqss must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), freqss)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    if not torch.compiler.is_compiling():
        assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (
            "invalid dimensions for broadcastable concatentation"
        )
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_freqss), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    freqss = list(map(lambda t: t[0].expand(*t[1]), zip(freqss, expandable_shapes)))
    return torch.cat(freqss, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            pt_seq_len=14,
            custom_freqs=None,
            freqs_for="lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            coord_high=None,
            dynamic=False,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                    theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.pt_seq_len = pt_seq_len
        self.coord_high = coord_high
        self.dynamic = dynamic
        self.register_buffer("freqs", freqs)

    def forward(self, x, coords: torch.Tensor | None = None):
        if not self.dynamic:
            # Fixed, dense image
            ft_seq_len = int(np.sqrt(x.shape[1]))
            t = torch.arange(ft_seq_len).cuda() / ft_seq_len * self.pt_seq_len

            freqs = torch.einsum("..., f -> ... f", t, self.freqs)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)  # 14*32
            freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)  # 14*14*64
        else:
            # Dynamic image with coordinates for all tokens
            if coords.ndim != 3:
                raise ValueError(f"Expected {coords.shape=} to be (B, L, 2)")

            t_x = coords[..., 0] / self.coord_high
            t_y = coords[..., 1] / self.coord_high

            freqs_x = torch.einsum("..., f -> ... f", t_x, self.freqs)
            freqs_x = repeat(freqs_x, "... n -> ... (n r)", r=2)
            freqs_y = torch.einsum("..., f -> ... f", t_y, self.freqs)
            freqs_y = repeat(freqs_y, "... n -> ... (n r)", r=2)
            freqs = broadcat((freqs_x[:, None, :], freqs_y[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, 1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, 1, freqs.shape[-1])
        res = x * freqs_cos + rotate_half(x) * freqs_sin

        # for reference, these are the shapes for 62936 tokens of 8x160=1280 dim:
        #    x.shape=torch.Size([1, 62936, 8, 160]) coords.shape=torch.Size([1, 62936, 2])
        #    t_x.shape=torch.Size([1, 62936]) freqs_x.shape=torch.Size([1, 62936, 80])
        #    freqs.shape=torch.Size([1, 1, 62936, 160]) freqs_cos.shape=torch.Size([62936, 1, 160])
        #    res.shape=torch.Size([1, 62936, 8, 160])
        return res


def rotate_freqs(freqs, angle_deg):
    if not torch.compiler.is_compiling():
        assert freqs.ndim == 4 and freqs.shape[0] == freqs.shape[1], (
            "Input must have shape (n, n, d1, d2)"
        )
    n, _, d1, d2 = freqs.shape
    freq_type = freqs.dtype
    angle_rad = math.radians(angle_deg)

    # Reshape from (n, n, d1, d2) → (n, n, d1 * d2)
    freqs = freqs.reshape(n, n, -1)

    # Permute to (1, C, H, W) where C = d1 * d2
    freqs = freqs.permute(2, 0, 1).unsqueeze(0)

    # Rotation matrix (2x3)
    theta = torch.tensor(
        [
            [math.cos(angle_rad), -math.sin(angle_rad), 0.0],
            [math.sin(angle_rad), math.cos(angle_rad), 0.0],
        ],
        dtype=torch.float32,
        device=freqs.device,
    ).unsqueeze(0)

    freqs = freqs.to(torch.float32)

    # Build sampling grid
    grid = F.affine_grid(theta, freqs.size(), align_corners=True)

    # Rotate using bilinear interpolation, with border padding
    rotated = F.grid_sample(
        freqs, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    # Convert back: (1, C, H, W) → (H, W, C)
    rotated = rotated.squeeze(0).permute(1, 2, 0).to(freq_type)

    # Reshape back to (n, n, d1, d2)
    return rotated.reshape(n, n, d1, d2)
