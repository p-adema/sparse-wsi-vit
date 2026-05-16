import re
from pathlib import Path

import h5py
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class MondriaanFlatDataset(Dataset):
    """Dataset for the flat HDF5 layout produced by extract_mondriaan.py.

    All samples live in a single directory; split and label are encoded
    in the filename: ``{split}_{idx:06d}_label{label}.h5``.

    Args:
        data_dir: Directory containing the flat .h5 files.
        split: One of ``"train"``, ``"val"``, ``"test"``.
    """

    _LABEL_RE = re.compile(r"_label(\d+)$")

    def __init__(self, data_dir: str | Path, split: str):
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.glob(f"{split}_*.h5"))
        if not files:
            raise FileNotFoundError(
                f"No .h5 files matching '{split}_*.h5' in {self.data_dir}"
            )
        self.samples = []
        for f in files:
            m = self._LABEL_RE.search(f.stem)
            if m is None:
                raise ValueError(f"Cannot parse label from filename: {f.name}")
            self.samples.append((f, int(m.group(1))))

        print(f"[MondriaanFlatDataset] {split}: {len(self.samples)} samples "
              f"from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        with h5py.File(path, "r") as hf:
            features = torch.from_numpy(hf["features"][:]).float()  # (N, 3)
            coords   = torch.from_numpy(hf["coords"][:]).float()    # (N, 2)
        return {
            "input":      features,
            "label":      torch.tensor(label, dtype=torch.long),
            "coords":     coords,
            "slide_name": path.stem,
        }


def _collate(batch: list[dict]) -> dict:
    assert len(batch) == 1, "MondriaanH5DataModule expects batch_size=1"
    item = batch[0]
    return {
        "input":      item["input"][None],
        "label":      item["label"][None],
        "coords":     item["coords"][None],
        "slide_name": item["slide_name"],
    }


class MondriaanH5DataModule(pl.LightningDataModule):
    """Lightning DataModule for pre-extracted Mondriaan pixel features.

    Reads the flat layout produced by extract_mondriaan.py (all splits
    in one directory, split/label encoded in filename).

    Args:
        data_dir: Directory containing the flat .h5 files.
        in_features: Feature dimension (3 for raw RGB pixels).
        batch_size: Must be 1 for MIL bags.
        num_workers: DataLoader worker processes.
    """

    def __init__(
        self,
        data_dir: str,
        in_features: int = 3,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir    = data_dir
        self.in_features = in_features
        self.batch_size  = batch_size
        self.num_workers = num_workers

        self.input_channels  = in_features
        self.output_channels = 2

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = MondriaanFlatDataset(self.data_dir, "train")
        self.val_dataset   = MondriaanFlatDataset(self.data_dir, "val")
        self.test_dataset  = MondriaanFlatDataset(self.data_dir, "test")

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
