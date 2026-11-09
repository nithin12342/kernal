"""Shared training configuration: loss, optimizer, and learning-rate schedule.

Identical settings are applied across DNN, GNN, and SNN so that comparative
results reflect model architecture rather than optimization differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


@dataclass
class TrainConfig:
    """Training hyperparameters shared by all model types."""
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 30
    step_size: int = 50
    gamma: float = 0.5


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> Adam:
    """Build an Adam optimizer with the shared configuration."""
    return Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: Adam, cfg: TrainConfig) -> StepLR:
    """Build a StepLR learning-rate scheduler with the shared configuration."""
    return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)


def build_criterion() -> nn.CrossEntropyLoss:
    """Build the shared cross-entropy loss."""
    return nn.CrossEntropyLoss()


def train_epoch(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: Adam,
    data,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    optimizer.zero_grad()

    if getattr(model, "uses_graph", False):
        # Graph-based models (GCN, SheafNN) consume edge_index.
        logits = model(data.x, data.edge_index)
    else:
        logits = model(data.x)

    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    data,
    device: torch.device,
) -> dict:
    """Evaluate loss and accuracy on train / val / test masks."""
    model.eval()
    if getattr(model, "uses_graph", False):
        logits = model(data.x, data.edge_index)
    else:
        logits = model(data.x)

    results = {}
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, mask_name)
        loss = criterion(logits[mask], data.y[mask]).item()
        pred = logits[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).float().mean().item()
        results[f"{mask_name}_loss"] = loss
        results[f"{mask_name}_acc"] = acc
    return results
