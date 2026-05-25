import torch
import torch.nn.functional as F


def varlen_collate_nsa(batch):
    """Pad WSI bags to a multiple of 64, needed for blocks input to CUTLASS kernel"""
    BLOCK_SIZE = 64

    assert len(batch) == 1, "Only B=1 for now"

    item = batch[0]
    x, coords = item["input"], item["coords"]
    seqlen = x.shape[0]

    pad_len = (-seqlen) % BLOCK_SIZE

    if pad_len > 0:
        x = F.pad(
            x, (0, 0, 0, pad_len), value=0.0
        )  # 4 coords in pad, last is right side
        coords = F.pad(
            coords, (0, 0, 0, pad_len), value=-1.0
        )  # should be masked, dummy val

    return {
        "input": x.unsqueeze(0),  # (1, Padded_N, D)
        "coords": coords.unsqueeze(0),  # (1, Padded_N, 2)
        "label": torch.tensor(item["target"]),
        "true_lengths": torch.tensor([seqlen], dtype=torch.long),  # original size
    }
