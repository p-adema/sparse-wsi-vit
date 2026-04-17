import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import h5py


class H5FeatureBagDataset(Dataset):
    """
    A PyTorch Dataset to read slide-level feature bags (N x 1280) from individual
    HDF5 files produced by the FastPathology extraction pipeline.
    Intended for MIL (Multiple Instance Learning) classification.
    """

    def __init__(self, csv_path, features_dir, label_col_name="label", transform=None):
        """
        Args:
            csv_path (str): Path to CSV containing 'slidename' and label.
            features_dir (str): Directory containing the extracted {slide_name}.h5 files.
            label_col_name (str): Column name in the CSV for the target label.
            transform (callable, optional): Optional transform applied to the bag of features.
        """
        super().__init__()
        self.features_dir = Path(features_dir)
        self.transform = transform
        self.label_col_name = label_col_name

        # Load slide-level metadata
        df = pd.read_csv(csv_path)

        # Mapping for string to int labels
        self.label_map = {}

        # Keep only slides for which the feature .h5 file actually exist
        valid_slides = []
        for idx, row in df.iterrows():
            slide_name = str(row.get("slidename", row.get("ID")))
            h5_path = self.features_dir / f"{slide_name}.h5"

            raw_label = row.get(label_col_name)
            if h5_path.exists() and pd.notna(raw_label):
                # Dynamically map strings to integers if required
                if isinstance(raw_label, str):
                    if raw_label not in self.label_map:
                        self.label_map[raw_label] = len(self.label_map)
                    mapped_label = self.label_map[raw_label]
                else:
                    mapped_label = int(raw_label)

                valid_slides.append(
                    {
                        "slide_name": slide_name,
                        "label": mapped_label,
                        "h5_path": h5_path,
                    }
                )

        self.slides = valid_slides
        print(f"Loaded {len(self.slides)} valid WSI feature bags from {features_dir}")

    def __len__(self) -> int:
        """Return the number of valid slides in the dataset."""
        return len(self.slides)

    def __getitem__(self, idx: int) -> dict:
        """Load a single slide's feature bag from disk.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict with keys:
                - ``"input"`` (torch.Tensor): Feature bag of shape (N, D), float32.
                - ``"label"`` (torch.Tensor): Scalar class label, int64.
                - ``"slide_name"`` (str): Slide identifier.
                - ``"coords"`` (torch.Tensor): Patch coordinates, shape (N, 2), float32.
        """
        item = self.slides[idx]
        h5_path = item["h5_path"]

        with h5py.File(h5_path, "r") as f:
            features = f["features"][:]  # shape: (N_patches, 1280)
            coords = f["coords"][:]  # shape: (N_patches, 2)

        features_t = torch.from_numpy(features).float()
        coords_t = torch.from_numpy(coords).float()

        label = item["label"]

        if self.transform is not None:
            features_t = self.transform(features_t)

        # Note: MIL models generally expect shape (N, feature_dim).
        return {
            "input": features_t,
            "label": torch.tensor(label, dtype=torch.long),
            "slide_name": item["slide_name"],
            "coords": coords_t,
        }
