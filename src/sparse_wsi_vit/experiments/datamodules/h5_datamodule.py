import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from sparse_wsi_vit.datasets.h5_slidedataset.h5_dataset import H5FeatureBagDataset


def mil_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-size MIL bags into a batch.

    Standard AB-MIL is trained with batch_size=1. When bags happen to have the
    same number of patches they are stacked into a (B, N, D) tensor; otherwise
    a list is returned so callers can handle padding themselves.

    Args:
        batch: List of sample dicts produced by :class:`H5FeatureBagDataset`.

    Returns:
        Dict with keys ``"input"``, ``"label"``, ``"slide_name"``, ``"coords"``.
    """
    assert len(batch) == 1, (
        "Some code assumes batch-size 1. We'd need a more elegant fix than tossing them in a list."
    )
    res = batch[0] | {
        "input": batch[0]["input"][None],
        "label": batch[0]["label"][None],
        "coords": batch[0]["coords"][None],
    }
    return res

    # This code doesn't actually do anything more useful than the above:

    # inputs = [b["input"] for b in batch]
    # labels = torch.stack([b["label"] for b in batch])
    # slide_names = [b["slide_name"] for b in batch]
    #
    # inputs = torch.stack(inputs, dim=0) if len(inputs) == 1 else inputs
    #
    # return {
    #     "input": inputs,
    #     "label": labels,
    #     "slide_name": slide_names,
    # }


class H5FeatureBagDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for HDF5 feature-bag MIL datasets.

    Args:
        train_csv: Path to CSV with training slide metadata.
        val_csv: Path to CSV with validation slide metadata.
        features_dir: Directory containing ``{slide_name}.h5`` feature files.
        label_col_name: Column name in the CSV files for the target label.
        batch_size: Number of slides per batch (typically 1 for AB-MIL).
        num_workers: DataLoader worker processes.
    """

    def __init__(
            self,
            train_csv: str,
            val_csv: str,
            features_dir: str = "",
            label_col_name: str = "label",
            batch_size: int = 1,
            num_workers: int = 4,
            class_weights: bool = False,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.features_dir = features_dir
        self.label_col_name = label_col_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_channels = 1280
        self.output_channels = 1
        self.class_weights = class_weights

    def setup(self, stage: str | None = None) -> None:
        """Instantiate train and validation datasets.

        Args:
            stage: Either ``"fit"`` or ``None``; only ``"fit"`` is supported.
        """
        if stage in ("fit", None):
            self.train_dataset = H5FeatureBagDataset(
                csv_path=self.train_csv,
                features_dir=self.features_dir,
                label_col_name=self.label_col_name,
                class_weights=self.class_weights,
            )
            self.val_dataset = H5FeatureBagDataset(
                csv_path=self.val_csv,
                features_dir=self.features_dir,
                label_col_name=self.label_col_name,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=mil_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=mil_collate_fn,
            pin_memory=True,
        )
