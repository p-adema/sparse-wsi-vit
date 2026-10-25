
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan_x import make_data_x


fig, axes = plt.subplots(1, 3, figsize=(13, 5))

for ax, num_tiles in zip(axes, [1, 4, 16]):
    data = make_data_x(8, seed=42, num_tiles=num_tiles)

    # Reconstruct the image from 4x4 RGB tokens.
    tokens = data["x"][0].cpu().numpy()
    side = int(np.sqrt(len(tokens)))

    patches = tokens.reshape(side, side, 4, 4, 3)
    image = patches.transpose(0, 2, 1, 3, 4)
    image = image.reshape(side * 4, side * 4, 3)

    # Background sentinel (-1) becomes black.
    image = np.clip(image, 0, 1)

    ax.imshow(image, interpolation="nearest")
    ax.set_title(f"{num_tiles} tiles | {len(tokens)} tokens")
    ax.axis("off")

plt.tight_layout()
plt.savefig("mondriaan_x_examples.png", dpi=200)
plt.show()
