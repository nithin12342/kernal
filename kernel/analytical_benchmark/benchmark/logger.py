"""Benchmark logger: tabular runtime results comparing baseline vs custom kernels.

Writes human-readable and machine-parseable output (CSV + pretty table) for
forward/backward latency and accuracy across all model/kernel combinations.
"""

from __future__ import annotations

import csv
import os
from typing import List, Dict, Any


class BenchmarkLogger:
    """Logs benchmark results to a directory as CSV files and a summary table."""

    def __init__(self, output_dir: str = "benchmark_results") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []

    def log(
        self,
        model: str,
        kernel: str,
        dataset: str,
        epochs: int,
        val_acc: float,
        test_acc: float,
        f1_macro: float,
        f1_micro: float,
        forward_ms: float,
        backward_ms: float,
        memory_mb: float,
        train_time_s: float,
    ) -> None:
        """Append a single benchmark result row."""
        self.rows.append({
            "model": model,
            "kernel": kernel,
            "dataset": dataset,
            "epochs": epochs,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "total_ms": forward_ms + backward_ms,
            "memory_mb": memory_mb,
            "train_time_s": train_time_s,
        })

    def write_csv(self, filename: str = "results.csv") -> str:
        """Write all rows to CSV and return the path."""
        path = os.path.join(self.output_dir, filename)
        if not self.rows:
            return path
        fieldnames = list(self.rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def print_summary(self) -> None:
        """Print a pretty-printed summary table to stdout."""
        if not self.rows:
            print("No benchmark results to display.")
            return

        header = (
            f"{'Model':<10} | {'Kernel':<10} | {'Dataset':<8} | "
            f"{'ValAcc':<7} | {'TestAcc':<7} | {'F1-macro':<8} | "
            f"{'FwdMs':<7} | {'BwdMs':<7} | {'MemMB':<7} | {'TimeS':<7}"
        )
        print("\n" + "=" * len(header))
        print("COMPARATIVE BENCHMARK SUMMARY")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for r in self.rows:
            print(
                f"{r['model']:<10} | {r['kernel']:<10} | {r['dataset']:<8} | "
                f"{r['val_acc']:<7.4f} | {r['test_acc']:<7.4f} | "
                f"{r['f1_macro']:<8.4f} | {r['forward_ms']:<7.2f} | "
                f"{r['backward_ms']:<7.2f} | {r['memory_mb']:<7.1f} | "
                f"{r['train_time_s']:<7.2f}"
            )
        print("=" * len(header) + "\n")
