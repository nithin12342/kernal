"""Training entry point.

Trains a single model (DNN / GCN / SheafNN) on the configured dataset and
prints the final validation / test accuracy.

Usage:
    python train.py --model dnn        # or gcn / sheafnn
    python train.py --model sheafnn --kernel rust
"""

from __future__ import annotations

import argparse
import torch

from .config import (
    DATASET, HIDDEN_DIM, NUM_LAYERS, DROPOUT, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, PATIENCE, SHEAF_STALK_DIM, SHEAF_TAU, KERNEL_BACKEND, RANDOM_SEED,
)
from .datasets import load_dataset
from .models import DNN, GCN, SheafNN
from .training import Trainer
from .training.scheduler import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DNN / GNN / SheafNN on Cora.")
    parser.add_argument("--model", choices=["dnn", "gcn", "sheafnn"], default="sheafnn")
    parser.add_argument("--kernel", choices=["pytorch", "rust", "cuda", "mojo"], default=None)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")

    data = load_dataset(name=args.dataset)
    num_features = data.num_features
    num_classes = data.num_classes

    kernel = args.kernel or KERNEL_BACKEND

    if args.model == "dnn":
        model = DNN(num_features, num_classes, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    elif args.model == "gcn":
        model = GCN(num_features, num_classes, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    else:
        model = SheafNN(
            num_features, num_classes, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
            SHEAF_STALK_DIM, SHEAF_TAU, kernel_backend=kernel,
        )

    cfg = TrainConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=args.epochs,
        patience=PATIENCE,
    )
    trainer = Trainer(model, data, config=cfg, device=device)
    metrics = trainer.fit(verbose=True)

    best = max(metrics.epochs, key=lambda e: e.val_acc) if metrics.epochs else None
    if best:
        print(f"\nBest val_acc={best.val_acc:.4f} test_acc={best.test_acc:.4f} "
              f"f1_macro={best.f1_macro:.4f} f1_micro={best.f1_micro:.4f}")


if __name__ == "__main__":
    main()
