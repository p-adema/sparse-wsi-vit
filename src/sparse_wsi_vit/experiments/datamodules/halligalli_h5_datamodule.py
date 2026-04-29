import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from sparse_wsi_vit.datasets.h5_slidedataset.h5_dataset import H5FeatureBagDataset


def _mil_collate_fn(batch: list[dict], corners_only: bool = False) -> dict:
    """Add a batch dimension to a single MIL bag."""
    assert len(batch) == 1, "HalliGalliH5DataModule expects batch_size=1"
    coords = batch[0]["coords"]
    inputs = batch[0]["input"]

    if corners_only:
        min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
        min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
        mask = (
            ((coords[:, 0] == min_x) | (coords[:, 0] == max_x)) &
            ((coords[:, 1] == min_y) | (coords[:, 1] == max_y))
        )
        inputs = inputs[mask]
        coords = coords[mask]

    return batch[0] | {
        "input": inputs[None],
        "label": batch[0]["label"][None],
        "coords": coords[None],
    }


class HalliGalliH5DataModule(pl.LightningDataModule):
    """Lightning DataModule for pre-extracted HalliGalli Virchow2 features.

    Expects the directory layout produced by extract_halligalli.py:

        data_dir/
            train/features/*.h5  +  train/labels.csv
            val/features/*.h5    +  val/labels.csv
            test/features/*.h5   +  test/labels.csv

    Args:
        data_dir: Root directory of the extracted dataset.
        in_features: Feature dimension D (1280 for CLS-only, 2560 for concat).
        batch_size: Samples per batch. MIL models expect 1.
        num_workers: DataLoader worker processes.
    """

    def __init__(
        self,
        data_dir: str,
        in_features: int = 1280,
        batch_size: int = 1,
        num_workers: int = 4,
        corners_only: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.corners_only = corners_only

        self.input_channels = in_features
        self.output_channels = 2

    def _make_dataset(self, split: str) -> H5FeatureBagDataset:
        split_dir = os.path.join(self.data_dir, split)
        return H5FeatureBagDataset(
            csv_path=os.path.join(split_dir, "labels.csv"),
            features_dir=os.path.join(split_dir, "features"),
        )

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = self._make_dataset("train")
        self.val_dataset = self._make_dataset("val")
        self.test_dataset = self._make_dataset("test")

    def _collate_fn(self, batch):
        return _mil_collate_fn(batch, corners_only=self.corners_only)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
