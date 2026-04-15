"""
Lightning DataModule for CAMELYON16 using MONAI/cuCIM.
"""

import json
from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sparse_wsi_vit.datasets.camelyon16_dataset import Camelyon16PatchDataset


class Camelyon16DataModule(pl.LightningDataModule):
    """
    LightningDataModule for CAMELYON16 WSI patches.

    Expects pre-computed JSON indices mapping out valid tissue patches.
    """

    def __init__(
        self,
        data_dir: str,
        train_index_file: str,
        val_index_file: str,
        test_index_file: Optional[str] = None,
        patch_size: tuple[int, int] = (256, 256),
        level: int = 0,
        batch_size: int = 32,
        num_workers: int = 8,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        backend: str = "cucim",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_index_file = train_index_file
        self.val_index_file = val_index_file
        self.test_index_file = test_index_file

        self.patch_size = patch_size
        self.level = level
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.backend = backend

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _load_index(self, index_path: str) -> list[dict]:
        path = Path(index_path)
        if not path.is_absolute():
            path = self.data_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(path, "r") as f:
            return json.load(f)

    def setup(self, stage: Optional[str] = None):
        """Load JSON indices and instantiate the MONAI-backed datasets."""

        if stage in ("fit", None):
            train_info = self._load_index(self.train_index_file)
            val_info = self._load_index(self.val_index_file)

            self.train_dataset = Camelyon16PatchDataset(
                patch_info=train_info,
                patch_size=self.patch_size,
                level=self.level,
                transform=self.train_transforms,
                backend=self.backend,
            )

            self.val_dataset = Camelyon16PatchDataset(
                patch_info=val_info,
                patch_size=self.patch_size,
                level=self.level,
                transform=self.val_transforms,
                backend=self.backend,
            )

        if stage in ("test", None) and self.test_index_file:
            test_info = self._load_index(self.test_index_file)
            self.test_dataset = Camelyon16PatchDataset(
                patch_info=test_info,
                patch_size=self.patch_size,
                level=self.level,
                transform=self.val_transforms,
                backend=self.backend,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
