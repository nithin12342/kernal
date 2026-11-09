"""Training metrics: accuracy, F1, memory footprint, and execution latency.

Collects per-epoch and per-backend telemetry for the comparative benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class EpochMetrics:
    """Per-epoch training telemetry."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    test_loss: float
    test_acc: float
    f1_macro: float
    f1_micro: float
    train_time_s: float
    memory_mb: float


@dataclass
class LatencyRecord:
    """Single forward/backward latency measurement for a kernel backend."""
    backend: str
    forward_ms: float
    backward_ms: float
    total_ms: float


@dataclass
class MetricsCollector:
    """Aggregates epoch metrics and latency records across models/backends."""
    epochs: List[EpochMetrics] = field(default_factory=list)
    latencies: List[LatencyRecord] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_test_acc: float = 0.0

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        test_loss: float,
        test_acc: float,
        f1_macro: float,
        f1_micro: float,
        train_time_s: float,
        memory_mb: float,
    ) -> None:
        self.epochs.append(EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            f1_macro=f1_macro,
            f1_micro=f1_micro,
            train_time_s=train_time_s,
            memory_mb=memory_mb,
        ))
        self.best_val_acc = max(self.best_val_acc, val_acc)
        self.best_test_acc = max(self.best_test_acc, test_acc)

    def record_latency(self, backend: str, forward_ms: float, backward_ms: float) -> None:
        self.latencies.append(LatencyRecord(
            backend=backend,
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            total_ms=forward_ms + backward_ms,
        ))

    def latest_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0


def f1_scores(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> tuple:
    """Compute macro and micro F1 scores."""
    f1_macro = 0.0
    for c in range(num_classes):
        tp = int(((pred == c) & (target == c)).sum().item())
        fp = int(((pred == c) & (target != c)).sum().item())
        fn = int(((pred != c) & (target == c)).sum().item())
        denom = 2 * tp + fp + fn
        f1_macro += (2 * tp / denom) if denom > 0 else 0.0
    f1_macro /= num_classes

    tp = int((pred == target).sum().item())
    fp = int((pred != target).sum().item())
    fn = fp  # micro: fp == fn
    denom = 2 * tp + fp + fn
    f1_micro = (2 * tp / denom) if denom > 0 else 0.0
    return f1_macro, f1_micro
