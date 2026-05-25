import gc
import time

import torch
import torchmetrics

from sparse_wsi_vit.experiments.default_cfg import ExperimentConfig
from sparse_wsi_vit.experiments.lightning_wrappers.base_lightning_wrapper import (
    LightningWrapperBase,
)


class MILWrapper(LightningWrapperBase):
    """Lightning wrapper for Multiple Instance Learning (MIL) classification tasks."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        use_bce_loss: bool = True,
    ):
        """Initialize the MILWrapper.

        Args:
            network: MIL network to wrap. Must expose an ``out_features`` attribute
                when used for multiclass classification.
            cfg: Experiment configuration.
            use_bce_loss: Use BCEWithLogitsLoss for binary classification.
                When *network.out_features* > 1 and this is False, CrossEntropyLoss
                is used instead.
        """
        super().__init__(network=network, cfg=cfg)

        self.multiclass = hasattr(network, "out_features") and network.out_features > 1

        if self.multiclass:
            acc_kwargs = {"task": "multiclass", "num_classes": network.out_features}
            prf_kwargs = {
                "task": "multiclass",
                "num_classes": network.out_features,
                "average": "macro",
            }
        else:
            acc_kwargs = {"task": "binary"}
            prf_kwargs = {"task": "binary"}

        self.train_acc = torchmetrics.Accuracy(**acc_kwargs)
        self.val_acc = torchmetrics.Accuracy(**acc_kwargs)
        self.val_precision = torchmetrics.Precision(**prf_kwargs)
        self.val_recall = torchmetrics.Recall(**prf_kwargs)
        self.val_f1 = torchmetrics.F1Score(**prf_kwargs)
        self.val_auc = torchmetrics.AUROC(**prf_kwargs)

        self.use_bce_loss = use_bce_loss
        if self.multiclass and not self.use_bce_loss:
            self.loss_metric = torch.nn.CrossEntropyLoss()
        else:
            self.loss_metric = torch.nn.BCEWithLogitsLoss()

    def _step(
        self, batch: dict[str, torch.Tensor], accuracy_calculator: torchmetrics.Metric
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Shared forward + loss computation for train and validation.

        Args:
            batch: Dict with keys ``"input"`` (B, N, D) and ``"label"`` (B,).
            accuracy_calculator: Metric accumulator to update with this step's predictions.

        Returns:
            A 3-tuple of ``(loss, predictions, output_dict)``.
        """
        inputs = batch["input"]
        labels = batch["label"]

        output_dict = self.network(inputs)
        logits = output_dict["logits"].squeeze(1)

        if not self.multiclass and self.use_bce_loss:
            loss = self.loss_metric(logits, labels.float())
            preds = (torch.sigmoid(logits) >= 0.5).int()
        else:
            loss = self.loss_metric(logits, labels)
            preds = torch.argmax(logits, dim=-1)

        accuracy_calculator.update(preds, labels)
        return loss, preds, {"logits": logits}

    def _update_prf_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Update precision, recall, F1 and AUC with probability scores."""
        if self.use_bce_loss:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)
        preds = (
            (probs >= 0.5).int() if not self.multiclass else torch.argmax(probs, dim=-1)
        )
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_auc.update(probs, labels)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform one training step and log loss.

        Args:
            batch: Dict with ``"input"`` and ``"label"`` tensors.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar training loss.
        """
        gc.collect()
        torch.cuda.empty_cache()

        loss, _, _ = self._step(batch, self.train_acc)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["input"].size(0),
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Log epoch-level training accuracy and reset the accumulator."""
        acc = self.train_acc.compute()
        self.log("train/acc", acc, sync_dist=True)
        self.train_acc.reset()

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._val_or_test_step(batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._val_or_test_step(batch, kind="test")

    def _val_or_test_step(
        self, batch: dict[str, torch.Tensor], kind: str = "val"
    ) -> torch.Tensor:
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        with torch.inference_mode():
            loss, _, output_dict = self._step(batch, self.val_acc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000

        logits = output_dict["logits"]
        labels = batch["label"]
        batch_size = batch["input"].size(0)
        num_patches = batch["input"].size(1)

        self._update_prf_metrics(logits, labels)

        self.log(
            f"{kind}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{kind}/num_patches",
            float(num_patches),
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(f"{kind}/step_ms", step_ms, on_epoch=True, batch_size=batch_size)
        self.log(
            f"{kind}/logits_mean",
            logits.mean(),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        if logits.numel() > 1:
            self.log(
                f"{kind}/logits_std",
                logits.std(),
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch-level validation metrics and reset accumulators."""
        self.log("val/acc", self.val_acc.compute(), sync_dist=True)
        self.val_acc.reset()
        self.log("val/precision", self.val_precision.compute(), sync_dist=True)
        self.val_precision.reset()
        self.log("val/recall", self.val_recall.compute(), sync_dist=True)
        self.val_recall.reset()
        self.log("val/f1", self.val_f1.compute(), sync_dist=True)
        self.val_f1.reset()
        self.log("val/auc", self.val_auc.compute(), sync_dist=True)
        self.val_auc.reset()

    def on_test_epoch_end(self) -> None:
        """Log epoch-level test metrics and reset accumulators."""
        self.log("test/acc", self.val_acc.compute(), sync_dist=True)
        self.val_acc.reset()
        self.log("test/precision", self.val_precision.compute(), sync_dist=True)
        self.val_precision.reset()
        self.log("test/recall", self.val_recall.compute(), sync_dist=True)
        self.val_recall.reset()
        self.log("test/f1", self.val_f1.compute(), sync_dist=True)
        self.val_f1.reset()
        self.log("test/auc", self.val_auc.compute(), sync_dist=True)
        self.val_auc.reset()
