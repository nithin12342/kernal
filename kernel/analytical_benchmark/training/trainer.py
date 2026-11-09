"""Trainer: end-to-end training loop with early stopping and metric collection.

Applies identical loss, optimizer, and learning-rate schedule to DNN, GNN, and
SNN so comparative results isolate architecture effects.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from .scheduler import TrainConfig, build_optimizer, build_scheduler, build_criterion, train_epoch, evaluate
from .metrics import MetricsCollector, f1_scores


class Trainer:
    """Trains a node-classification model with shared hyperparameters.

    Attributes:
        model: The model to train.
        data: The dataset (CoraDataset with masks).
        config: Shared training configuration.
        device: Torch device for compute.
        metrics: Collected per-epoch telemetry.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data,
        config: Optional[TrainConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or TrainConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.data.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.criterion = build_criterion()
        self.metrics = MetricsCollector()

    def fit(self, verbose: bool = True) -> MetricsCollector:
        """Run the full training loop with early stopping on validation loss."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.epochs):
            start = time.perf_counter()
            train_loss = train_epoch(self.model, self.criterion, self.optimizer, self.data, self.device)
            train_time = time.perf_counter() - start

            eval_res = evaluate(self.model, self.criterion, self.data, self.device)

            # F1 on train mask
            self.model.eval()
            with torch.no_grad():
                if getattr(self.model, "uses_graph", False):
                    logits = self.model(self.data.x, self.data.edge_index)
                else:
                    logits = self.model(self.data.x)
                pred = logits[self.data.train_mask].argmax(dim=1)
                f1_macro, f1_micro = f1_scores(pred, self.data.y[self.data.train_mask], self.data.num_classes)

            memory_mb = self.metrics.latest_memory_mb()
            self.metrics.record_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=eval_res["train_mask_acc"],
                val_loss=eval_res["val_mask_loss"],
                val_acc=eval_res["val_mask_acc"],
                test_loss=eval_res["test_mask_loss"],
                test_acc=eval_res["test_mask_acc"],
                f1_macro=f1_macro,
                f1_micro=f1_micro,
                train_time_s=train_time,
                memory_mb=memory_mb,
            )

            self.scheduler.step()

            if verbose and (epoch % 20 == 0 or epoch == self.config.epochs - 1):
                print(
                    f"[Epoch {epoch:>3}] train_loss={train_loss:.4f} "
                    f"train_acc={eval_res['train_mask_acc']:.4f} "
                    f"val_loss={eval_res['val_mask_loss']:.4f} "
                    f"val_acc={eval_res['val_mask_acc']:.4f} "
                    f"test_acc={eval_res['test_mask_acc']:.4f}"
                )

            # Early stopping on validation loss
            if eval_res["val_mask_loss"] < best_val_loss:
                best_val_loss = eval_res["val_mask_loss"]
                patience_counter = 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"[EarlyStopping] triggered at epoch {epoch}")
                    break

        # Restore best state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.metrics
