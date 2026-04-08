import torch
import torchmetrics

from sparse_wsi_vit.experiments.default_cfg import ExperimentConfig
from sparse_wsi_vit.experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


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
        else:
            acc_kwargs = {"task": "binary"}

        self.train_acc = torchmetrics.Accuracy(**acc_kwargs)
        self.val_acc = torchmetrics.Accuracy(**acc_kwargs)

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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one training step and log loss.

        Args:
            batch: Dict with ``"input"`` and ``"label"`` tensors.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar training loss.
        """
        loss, _, _ = self._step(batch, self.train_acc)
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, sync_dist=True,
            batch_size=batch["input"].size(0),
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Log epoch-level training accuracy and reset the accumulator."""
        acc = self.train_acc.compute()
        self.log("train/acc", acc, sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one validation step and log loss.

        Args:
            batch: Dict with ``"input"`` and ``"label"`` tensors.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar validation loss.
        """
        loss, _, _ = self._step(batch, self.val_acc)
        self.log(
            "val/loss", loss,
            on_step=False, on_epoch=True, sync_dist=True,
            batch_size=batch["input"].size(0),
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch-level validation accuracy and reset the accumulator."""
        acc = self.val_acc.compute()
        self.log("val/acc", acc, sync_dist=True)
        self.val_acc.reset()
