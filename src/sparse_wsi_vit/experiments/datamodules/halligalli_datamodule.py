import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from sparse_wsi_vit.datasets.halligalli_dataset.halligalli import HalliGalliGenerator


class HalliGalliDataset(Dataset):
    """On-the-fly HalliGalli patch-bag dataset.

    Each __getitem__ call generates one HalliGalli sample and returns it
    as a flattened patch bag of shape (N, D), ready for MIL models.

    Args:
        length: Number of samples in the dataset (virtual size).
        image_size: Height and width of each generated image.
        patch_size: Side length of each square patch in pixels.
        **kwargs: Forwarded verbatim to :meth:`HalliGalliGenerator.generate_single`.
    """

    def __init__(
        self,
        length: int,
        image_size: int = 64,
        patch_size: int = 16,
        **kwargs,
    ):
        self.length = length
        self.image_size = image_size
        self.patch_size = patch_size
        self.kwargs = kwargs

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        img, label, _, _ = HalliGalliGenerator.generate_single(
            image_size=self.image_size,
            **self.kwargs,
        )
        # img: (H, W, 3) ndarray → Tensor (3, H, W)
        x = torch.from_numpy(img).permute(2, 0, 1)

        # Split into non-overlapping patches → (N, D)
        ps = self.patch_size
        x = x.unfold(1, ps, ps).unfold(2, ps, ps)  # (3, n_h, n_w, ps, ps)
        x = x.permute(1, 2, 0, 3, 4).contiguous()  # (n_h, n_w, 3, ps, ps)
        n_h, n_w, C, ph, pw = x.shape
        bag = x.view(n_h * n_w, C * ph * pw)        # (N, D)

        return {
            "input": bag,
            "label": torch.tensor(label, dtype=torch.int64),
        }


class HalliGalliDataModule(pl.LightningDataModule):
    """Lightning DataModule for the HalliGalli synthetic MIL benchmark.

    Images are generated on the fly; all bags have identical shape (N, D)
    so no custom collate function is needed.

    Attributes
    ----------
    input_channels : int
        Feature dimension D = 3 * patch_size².  Read by the runner to
        instantiate the network.
    output_channels : int
        Always 2 (binary task, CrossEntropy).  Read by the runner.

    Args:
        train_size: Virtual size of the training set.
        val_size: Virtual size of the validation set.
        test_size: Virtual size of the test set.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        image_size: Height = width of each generated image.
        patch_size: Patch side length in pixels (must divide image_size).
        **kwargs: Forwarded to :class:`HalliGalliDataset` (and on to
            :meth:`HalliGalliGenerator.generate_single`).
    """

    def __init__(
        self,
        train_size: int = 2000,
        val_size: int = 400,
        test_size: int = 400,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 64,
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}."
            )
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.patch_size = patch_size
        self.kwargs = kwargs

        self.input_channels = 3 * patch_size ** 2
        self.output_channels = 2

    def setup(self, stage: str | None = None) -> None:
        ds_kw = dict(image_size=self.image_size, patch_size=self.patch_size, **self.kwargs)
        self.train_dataset = HalliGalliDataset(self.train_size, **ds_kw)
        self.val_dataset = HalliGalliDataset(self.val_size, **ds_kw)
        self.test_dataset = HalliGalliDataset(self.test_size, **ds_kw)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
