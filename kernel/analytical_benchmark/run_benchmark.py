"""Benchmark entry point.

Runs the comparative benchmark across all model x kernel combinations and
writes tabular results to ``benchmark_results/``.

Usage:
    python run_benchmark.py --dataset cora
    python run_benchmark.py --backend rust --backend pytorch
"""

from __future__ import annotations

import argparse

from .benchmark import run_benchmark
from .config import DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DNN / GNN / SheafNN comparative benchmark.")
    parser.add_argument("--dataset", default=DATASET, choices=["cora", "texas", "chameleon", "squirrel"])
    parser.add_argument("--backend", action="append", default=None, choices=["pytorch", "rust", "cuda", "mojo"])
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    run_benchmark(
        dataset_name=args.dataset,
        backends=args.backend,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
