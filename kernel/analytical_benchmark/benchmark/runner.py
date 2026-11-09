"""Benchmark runner: executes all model x kernel combinations and records results.

Trains DNN, GNN, and SheafNN under each available sheaf kernel backend and
logs accuracy, F1, latency, and memory metrics for the comparative report.
"""

from __future__ import annotations

import time
import torch

from ..config import (
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, PATIENCE, SHEAF_STALK_DIM, SHEAF_TAU, KERNEL_BACKEND, RANDOM_SEED,
)
from ..datasets import load_dataset
from ..models import DNN, GCN, SheafNN
from ..kernels import available_backends
from ..training import Trainer
from ..training.scheduler import TrainConfig
from .logger import BenchmarkLogger


def _measure_latency(model, data, device, kernel_name: str, num_warmup: int = 3, num_runs: int = 10) -> tuple:
    """Measure forward + backward latency for a model on a given kernel backend."""
    model.eval()
    x = data.x
    edge_index = data.edge_index

    # Warmup
    for _ in range(num_warmup):
        if getattr(model, "uses_graph", False):
            logits = model(x, edge_index)
        else:
            logits = model(x)
        loss = logits[data.train_mask].sum()
        loss.backward()
        model.zero_grad(set_to_none=True)

    # Timed runs
    fwd_times, bwd_times = [], []
    for _ in range(num_runs):
        torch.cpu.synchronize() if device.type == "cpu" else torch.cuda.synchronize()
        t0 = time.perf_counter()
        if getattr(model, "uses_graph", False):
            logits = model(x, edge_index)
        else:
            logits = model(x)
        torch.cpu.synchronize() if device.type == "cpu" else torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = logits[data.train_mask].sum()
        loss.backward()
        torch.cpu.synchronize() if device.type == "cpu" else torch.cuda.synchronize()
        t2 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        fwd_times.append((t1 - t0) * 1000.0)
        bwd_times.append((t2 - t1) * 1000.0)

    return sum(fwd_times) / len(fwd_times), sum(bwd_times) / len(bwd_times)


def run_benchmark(
    dataset_name: str = "cora",
    backends: list | None = None,
    output_dir: str = "benchmark_results",
) -> BenchmarkLogger:
    """Run the comparative benchmark across models and kernel backends.

    Args:
        dataset_name: Dataset identifier ("cora").
        backends: List of kernel backend names to test. Defaults to
            ``available_backends()``.
        output_dir: Directory for CSV and table output.

    Returns:
        The populated :class:`BenchmarkLogger`.
    """
    if backends is None:
        backends = available_backends()

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")

    data = load_dataset(name=dataset_name)
    num_features = data.num_features
    num_classes = data.num_classes

    cfg = TrainConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        patience=PATIENCE,
    )

    logger = BenchmarkLogger(output_dir=output_dir)

    model_specs = {
        "DNN": lambda: DNN(num_features, num_classes, HIDDEN_DIM, NUM_LAYERS, DROPOUT),
        "GCN": lambda: GCN(num_features, num_classes, HIDDEN_DIM, NUM_LAYERS, DROPOUT),
        "SheafNN": lambda: SheafNN(
            num_features, num_classes, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
            SHEAF_STALK_DIM, SHEAF_TAU, kernel_backend=KERNEL_BACKEND,
        ),
    }

    for backend in backends:
        for model_name, model_fn in model_specs.items():
            model = model_fn()
            trainer = Trainer(model, data, config=cfg, device=device)
            metrics = trainer.fit(verbose=False)

            best_epoch = max(metrics.epochs, key=lambda e: e.val_acc) if metrics.epochs else None
            if best_epoch is None:
                continue

            # Latency measurement
            fwd_ms, bwd_ms = _measure_latency(model, data, device, backend)

            logger.log(
                model=model_name,
                kernel=backend,
                dataset=dataset_name,
                epochs=len(metrics.epochs),
                val_acc=best_epoch.val_acc,
                test_acc=best_epoch.test_acc,
                f1_macro=best_epoch.f1_macro,
                f1_micro=best_epoch.f1_micro,
                forward_ms=fwd_ms,
                backward_ms=bwd_ms,
                memory_mb=best_epoch.memory_mb,
                train_time_s=sum(e.train_time_s for e in metrics.epochs),
            )

    logger.write_csv()
    logger.print_summary()
    return logger
