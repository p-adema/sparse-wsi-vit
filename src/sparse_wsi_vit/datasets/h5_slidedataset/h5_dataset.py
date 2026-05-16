from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


class H5FeatureBagDataset(Dataset):
    """
    A PyTorch Dataset to read slide-level feature bags (N x 1280) from individual
    HDF5 files produced by the FastPathology extraction pipeline.
    Intended for MIL (Multiple Instance Learning) classification.
    """

    def __init__(
        self,
        csv_path,
        features_dir,
        label_col_name="label",
        transform=None,
        class_weights=False,
        features_name: str = "features",
        coords_name: str = "coords",
        flatten_block: bool = True,
        downscale_block: int = 1,
        labels: tuple[str, ...] | None = None,
    ):
        """
        Args:
            csv_path (str): Path to CSV containing 'slidename' and label.
            features_dir (str): Directory containing the extracted {slide_name}.h5 files.
            label_col_name (str): Column name in the CSV for the target label.
            transform (callable, optional): Optional transform applied to the bag of features.
            features_name (str): Array key in H5 file for features
            coords_name (str): Array key in H5 file for coordinates
            flatten_block (bool): If features/coords have a block (64) dim, flatten it into L
        """
        super().__init__()
        self.features_dir = Path(features_dir)
        self.transform = transform
        self.label_col_name = label_col_name
        self.class_weights = class_weights
        self.features_name = features_name
        self.coords_name = coords_name
        self.flatten_block = flatten_block
        self.downscale_block = downscale_block

        # Load slide-level metadata
        df = pd.read_csv(csv_path)

        # Mapping for string to int labels
        if labels is None:
            self.label_map = {}
        else:
            self.label_map = {lbl: i for i, lbl in enumerate(labels)}
        label_counts = {}

        # Keep only slides for which the feature .h5 file actually exist
        valid_slides = []
        for idx, row in df.iterrows():
            slide_name = str(row.get("slidename", row.get("ID")))
            h5_path = self.features_dir / f"{slide_name}.h5"

            raw_label = row.get(label_col_name)
            if h5_path.exists() and pd.notna(raw_label):
                # Dynamically map strings to integers if required and no provided map
                if isinstance(raw_label, str):
                    if raw_label not in self.label_map:
                        if labels is not None:
                            raise ValueError(f"Unknown {raw_label=} given {labels=}")
                        self.label_map[raw_label] = len(self.label_map)
                    mapped_label = self.label_map[raw_label]
                else:
                    mapped_label = int(raw_label)

                label_counts[mapped_label] = label_counts.get(mapped_label, 0) + 1

                valid_slides.append(
                    {
                        "slide_name": slide_name,
                        "label": mapped_label,
                        "h5_path": h5_path,
                    }
                )
            else:
                print(f"Warning: invalid {h5_path.resolve()=}")
        assert len(valid_slides) > 0, f"Probably misconfigured {features_dir=} {df=}"
        per_cls = len(valid_slides) / len(label_counts)  # N / |C|
        if self.class_weights:
            for slide in valid_slides:
                # The average class weight with this formulation is still 1.0
                slide["class_weight"] = per_cls / label_counts[slide["label"]]

        self.slides = valid_slides
        print(f"Loaded {len(self.slides)} valid WSI feature bags from {features_dir}")

    def __len__(self) -> int:
        """Return the number of valid slides in the dataset."""
        return len(self.slides)

    def __getitem__(self, index: int) -> dict:
        """Load a single slide's feature bag from disk.

        Args:
            index: Index into the dataset.

        Returns:
            Dict with keys:
                - ``"input"`` (torch.Tensor): Feature bag of shape (N, D), float32.
                - ``"label"`` (torch.Tensor): Scalar class label, int64.
                - ``"slide_name"`` (str): Slide identifier.
                - ``"coords"`` (torch.Tensor): Patch coordinates, shape (N, 2), float32.
        """
        item = self.slides[index]
        h5_path = item["h5_path"]

        with h5py.File(h5_path, "r") as f:
            # shape: (N_patches, 1280) or (N_blocks, 64, 1280)
            features = f[self.features_name][:]
            # shape: (N_patches, 2) or (N_blocks, 64, 2)
            coords = f[self.coords_name][:]

        if self.downscale_block > 1:
            assert self.downscale_block in (2, 4, 8), "Can only downscale by power of 2"
            assert len(features.shape) == 3, "Can only downscale when given block info"
            n_blocks = features.shape[0]
            bcount, bsize = 8 // self.downscale_block, self.downscale_block
            features = features.reshape(n_blocks, bcount, bsize, bcount, bsize, 1280)
            features = features.mean((2, 4)).reshape(n_blocks, bcount * bcount, 1280)
            coords = coords.reshape(n_blocks, bcount, bsize, bcount, bsize, 2)
            coords = coords[:, :, 0, :, 0, :]

        if self.flatten_block and len(features.shape) == 3:
            features = features.reshape(-1, 1280)
            coords = coords.reshape(-1, 2)

        features_t = torch.from_numpy(features).float()
        coords_t = torch.from_numpy(coords).float()

        label = item["label"]

        if self.transform is not None:
            features_t = self.transform(features_t)

        # Note: MIL models generally expect shape (N, feature_dim).
        res = {
            "input": features_t,
            "label": torch.tensor(label, dtype=torch.long),
            "slide_name": item["slide_name"],
            "coords": coords_t,
        }
        if self.class_weights:
            res["class_weight"] = item["class_weight"]
        return res
