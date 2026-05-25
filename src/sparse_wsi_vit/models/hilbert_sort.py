import torch


def hilbert_sort(coords: torch.Tensor) -> torch.Tensor:
    """Sort patches by 2-D Hilbert curve so spatially nearby patches are contiguous.

    Args:
        coords: ``(B, N, 2)`` pixel coordinates.

    Returns:
        ``(B, N)`` indices that reorder patches along the Hilbert curve.
    """
    batch_size, num_patches, _ = coords.shape
    bits = 16
    grid_size = 1 << bits

    cmin = coords.amin(dim=1, keepdim=True)
    cmax = coords.amax(dim=1, keepdim=True)
    span = (cmax - cmin).clamp(min=1).float()
    norm = ((coords - cmin).float() / span * (grid_size - 1)).long()

    x = norm[..., 0].clone()
    y = norm[..., 1].clone()
    d = torch.zeros(batch_size, num_patches, dtype=torch.long, device=coords.device)

    s = grid_size >> 1
    while s > 0:
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        d += s * s * ((3 * rx) ^ ry)
        no_ry = ry == 0
        flip = no_ry & (rx == 1)
        x = torch.where(flip, s - 1 - x, x)
        y = torch.where(flip, s - 1 - y, y)
        x_new = torch.where(no_ry, y, x)
        y_new = torch.where(no_ry, x, y)
        x, y = x_new, y_new
        s >>= 1

    return d.argsort(dim=1)
