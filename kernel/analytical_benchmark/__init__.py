"""Analytical Benchmark Package Router.

Public API boundary for the analytical_benchmark package. External consumers
must import from this entry point only; deep private sub-paths are not part of
the public contract.
"""

from .config import (
    DATASET,
    DATA_DIR,
    DOWNLOAD_URLS,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    SHEAF_STALK_DIM,
    SHEAF_TAU,
    SHEAF_NUM_DIFFUSION_LAYERS,
    KERNEL_BACKEND,
    RUST_BENCH_BINARY,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)

from .datasets import load_dataset, CoraDataset
from .models import DNN, GCN, SheafNN
from .kernels import SheafKernel, PyTorchSheafKernel, available_backends
from .training import Trainer
from .benchmark import run_benchmark, BenchmarkLogger

__all__ = [
    # config
    "DATASET",
    "DATA_DIR",
    "DOWNLOAD_URLS",
    "HIDDEN_DIM",
    "NUM_LAYERS",
    "DROPOUT",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "EPOCHS",
    "PATIENCE",
    "SHEAF_STALK_DIM",
    "SHEAF_TAU",
    "SHEAF_NUM_DIFFUSION_LAYERS",
    "KERNEL_BACKEND",
    "RUST_BENCH_BINARY",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",
    "RANDOM_SEED",
    # datasets
    "load_dataset",
    "CoraDataset",
    # models
    "DNN",
    "GCN",
    "SheafNN",
    # kernels
    "SheafKernel",
    "PyTorchSheafKernel",
    "available_backends",
    # training
    "Trainer",
    # benchmark
    "run_benchmark",
    "BenchmarkLogger",
]
