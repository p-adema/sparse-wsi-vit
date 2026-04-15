"""
PyTorch Dataset for CAMELYON16 using MONAI and cuCIM for extremely fast WSI patch loading.

This dataset expects a pre-extracted list of patch coordinates to avoid costly
tissue-masking and background filtering during the training loop.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

try:
    from monai.data.wsi_reader import WSIReader
except ImportError:
    WSIReader = None


class Camelyon16PatchDataset(Dataset):
    """Dataset for fast CAMELYON16 WSI patch extraction.

    Expects `patch_info` to be a list of dictionaries, for example:
    [
        {"image_path": "/path/to/slide.tif", "x": 1000, "y": 2000, "label": 1},
        ...
    ]
    """

    def __init__(
        self,
        patch_info: List[Dict[str, Any]],
        patch_size: Tuple[int, int] = (256, 256),
        level: int = 0,
        transform: Optional[Callable] = None,
        backend: str = "cucim",
    ):
        """Initialize the dataset.

        Args:
            patch_info: List of dictionaries containing patch metadata.
            patch_size: (height, width) of the patch to extract.
            level: The WSI resolution level to extract from (0 is highest).
            transform: Optional torchvision or albumentations transforms to apply.
            backend: WSI backend (cucim is highly recommended on Linux/GPU).
        """
        if WSIReader is None:
            raise ImportError(
                "MONAI is required. Install with `pip install monai cucim`."
            )

        self.patch_info = patch_info
        self.patch_size = patch_size
        self.level = level
        self.transform = transform

        # Instantiate reader. Note: cuCIM is exceptionally fast for TIFF loading.
        self.reader = WSIReader(backend=backend)

        # Cache of read WSI objects to prevent reopening the same file continuously
        # if __getitem__ asks for patches from the same slide sequentially.
        self._current_slide_path = None
        self._current_wsi_obj = None

    def __len__(self) -> int:
        return len(self.patch_info)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.patch_info[idx]
        image_path = info["image_path"]
        x, y = info["x"], info["y"]
        label = info.get("label", 0)

        # Basic slide caching (avoids WSIReader.read() overhead if reading same slide)
        if image_path != self._current_slide_path:
            self._current_slide_path = image_path
            self._current_wsi_obj = self.reader.read(image_path)

        # Extraction via MONAI WSIReader.
        # WSIReader.get_data returns a tuple: (image_data, meta_dict)
        # Note: 'location' in MONAI WSIReader is typically (h, w) or (y, x) depending on backend.
        # usually cucim/OpenSlide expects (x, y) spatial coordinates at level 0.
        img_data, _ = self.reader.get_data(
            self._current_wsi_obj,
            location=(x, y),
            size=self.patch_size,
            level=self.level,
        )

        # img_data is usually shaped (C, H, W). Convert to standard Tensor.
        if not isinstance(img_data, torch.Tensor):
            img_data = torch.from_numpy(img_data)

        if self.transform:
            img_data = self.transform(img_data)

        return {
            "image": img_data,
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": image_path,
            "x": x,
            "y": y,
        }
