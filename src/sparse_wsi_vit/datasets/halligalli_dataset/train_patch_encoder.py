"""Train the ShapePatchCNN patch encoder on synthetic HalliGalli shape patches.

Generates a synthetic patch dataset on the fly using the same rendering
pipeline as the HalliGalli slides, trains a small CNN to classify shape
type, then saves the checkpoint and runs post-training sanity checks.

Classes:
    0 – circle
    1 – triangle
    2 – square
    3 – cross
    4 – star
    5 – background (clutter only, no shape)

Usage
-----
    uv run src/sparse_wsi_vit/datasets/halligalli_dataset/train_patch_encoder.py \\
        --n_per_class 20000 \\
        --output_dir checkpoints/patch_cnn \\
        --device cuda:0
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from sparse_wsi_vit.datasets.halligalli_dataset.halligalli import (
    ALL_SHAPES,
    _draw_clutter,
    _random_color,
    _stamp_shape,
)
from sparse_wsi_vit.datasets.halligalli_dataset.patch_cnn import ShapePatchCNN

# class index for background patches (no shape)
BACKGROUND_CLASS = len(ALL_SHAPES)  # 5
NUM_CLASSES = len(ALL_SHAPES) + 1   # 6


def _gen_shape_patch(shape_name: str, patch_size: int, shape_radius: int,
                     clutter_density: float) -> np.ndarray:
    """Generate one 64×64 patch containing a single shape at center."""
    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
    if clutter_density > 0:
        _draw_clutter(canvas, density=clutter_density)
    cy = cx = patch_size // 2
    r = max(2, int(shape_radius * np.random.uniform(0.7, 1.3)))
    angle = np.random.uniform(0, 2 * np.pi)
    deform = np.random.uniform(0, 0.25)
    _stamp_shape(canvas, shape_name, cy, cx, r, angle_rad=angle,
                 deform_strength=deform, color=_random_color())
    return np.clip(canvas, 0.0, 1.0)


def _gen_background_patch(patch_size: int, clutter_density: float) -> np.ndarray:
    """Generate one 64×64 clutter-only background patch."""
    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
    if clutter_density > 0:
        _draw_clutter(canvas, density=clutter_density)
    return np.clip(canvas, 0.0, 1.0)


class ShapePatchDataset(Dataset):
    """On-the-fly synthetic patch dataset.

    Each __getitem__ call generates a fresh random patch, so every epoch
    sees different samples. For the validation split, patches are
    pre-generated once in __init__ and cached.
    """

    def __init__(
        self,
        n_per_class: int,
        patch_size: int = 64,
        shape_radius: int = 20,
        clutter_density: float = 4.0,
        augment: bool = True,
        pregenerate: bool = False,
        seed: int = 0,
    ):
        self.n_per_class = n_per_class
        self.patch_size = patch_size
        self.shape_radius = shape_radius
        self.clutter_density = clutter_density
        self.augment = augment

        self.aug_transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(45),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            T.ToTensor(),
        ])
        self.plain_transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
        ])

        if pregenerate:
            np_state = np.random.get_state()
            py_state = random.getstate()
            np.random.seed(seed)
            random.seed(seed)
            self._cache: list[tuple[np.ndarray, int]] = []
            for cls_idx in range(NUM_CLASSES):
                for _ in range(n_per_class):
                    self._cache.append((self._gen(cls_idx), cls_idx))
            np.random.set_state(np_state)
            random.setstate(py_state)
        else:
            self._cache = None

    def _gen(self, cls_idx: int) -> np.ndarray:
        if cls_idx < len(ALL_SHAPES):
            return _gen_shape_patch(ALL_SHAPES[cls_idx], self.patch_size,
                                    self.shape_radius, self.clutter_density)
        return _gen_background_patch(self.patch_size, self.clutter_density)

    def __len__(self) -> int:
        return NUM_CLASSES * self.n_per_class

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        cls_idx = idx // self.n_per_class
        if self._cache is not None:
            patch_np, label = self._cache[idx]
        else:
            patch_np = self._gen(cls_idx)
            label = cls_idx

        patch_uint8 = (patch_np * 255).astype(np.uint8)
        transform = self.aug_transform if self.augment else self.plain_transform
        return transform(patch_uint8), label


def _run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = total_correct = total = 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, total_correct / total


def _sanity_check(model: ShapePatchCNN, val_dataset: ShapePatchDataset,
                  device: torch.device, n_per_class: int = 500) -> float:
    """Compute intra/inter class cosine similarity gap on the validation set."""
    model.eval()
    loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
    embeddings: dict[int, list[torch.Tensor]] = defaultdict(list)

    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = F.normalize(model.forward_features(imgs), dim=-1).cpu()
            for feat, label in zip(feats, labels.tolist()):
                if len(embeddings[label]) < n_per_class:
                    embeddings[label].append(feat)
            if (len(embeddings) == NUM_CLASSES
                    and all(len(v) >= n_per_class for v in embeddings.values())):
                break

    classes = sorted(embeddings.keys())
    stacked = [torch.stack(embeddings[c][:n_per_class]) for c in classes]

    intra, inter = [], []
    for i, fi in enumerate(stacked):
        sims = (fi @ fi.T)
        mask = ~torch.eye(len(fi), dtype=torch.bool)
        intra.append(sims[mask].mean().item())
        for j, fj in enumerate(stacked):
            if j > i:
                inter.append((fi @ fj.T).mean().item())

    mean_intra = float(np.mean(intra))
    mean_inter = float(np.mean(inter))
    gap = mean_intra - mean_inter
    print(f"  Intra-class cosine similarity : {mean_intra:.4f}")
    print(f"  Inter-class cosine similarity : {mean_inter:.4f}")
    print(f"  Gap                           : {gap:.4f}  (target > 0.3)")
    return gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=20_000,
                        help="Training patches per class (6 classes total)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of n_per_class used for validation")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--shape_radius", type=int, default=20,
                        help="Radius of key shapes in training patches. "
                             "Should match --shape_radius used in extract_halligalli.py.")
    parser.add_argument("--clutter_density", type=float, default=4.0)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(args.n_per_class * args.val_fraction))
    train_dataset = ShapePatchDataset(
        n_per_class=args.n_per_class,
        patch_size=args.patch_size,
        shape_radius=args.shape_radius,
        clutter_density=args.clutter_density,
        augment=True,
        pregenerate=False,
    )
    val_dataset = ShapePatchDataset(
        n_per_class=n_val,
        patch_size=args.patch_size,
        shape_radius=args.shape_radius,
        clutter_density=args.clutter_density,
        augment=False,
        pregenerate=True,
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = ShapePatchCNN(embed_dim=args.embed_dim, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = 0.0
    best_ckpt = args.output_dir / "patch_cnn.pt"

    print(f"Training ShapePatchCNN on {len(train_dataset):,} patches "
          f"({NUM_CLASSES} classes, {args.n_per_class:,} per class)")
    print(f"Val set: {len(val_dataset):,} patches (pre-generated)\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, None, device, train=False)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "embed_dim": args.embed_dim,
                "num_classes": NUM_CLASSES,
                "val_acc": val_acc,
                "epoch": epoch,
                "patch_size": args.patch_size,
                "shape_radius": args.shape_radius,
            }, best_ckpt)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
              + ("  *" if val_acc == best_val_acc else ""))

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {best_ckpt}")

    # reload best checkpoint for sanity checks
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("\n── Sanity checks ──────────────────────────────────────")
    _, final_val_acc = _run_epoch(model, val_loader, None, device, train=False)
    print(f"  5-way patch accuracy (val)    : {final_val_acc:.4f}  (target >= 0.99)")

    gap = _sanity_check(model, val_dataset, device)

    results = {
        "val_acc": final_val_acc,
        "cosine_gap": gap,
        "embed_dim": args.embed_dim,
        "num_classes": NUM_CLASSES,
        "patch_size": args.patch_size,
        "shape_radius": args.shape_radius,
    }
    (args.output_dir / "sanity_check.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output_dir / 'sanity_check.json'}")

    if final_val_acc < 0.99:
        print("WARNING: val_acc below 0.99 — consider training longer or adjusting shape_radius.")
    if gap < 0.3:
        print("WARNING: cosine gap below 0.3 — features may not be discriminative enough.")
    else:
        print("All sanity checks passed.")


if __name__ == "__main__":
    main()
