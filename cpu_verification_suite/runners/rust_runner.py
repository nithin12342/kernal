"""Rust Kernel Test & Benchmark Runner.

Executes cargo test and optimized release benchmarks for:
1. Rust Standard Kernel (BSR, forward, backward, telemetry)
2. Rust Metaprogrammed Kernel (ConstSheafEngine, compile-time unrolling)
"""

import subprocess
from pathlib import Path
from typing import Dict, Tuple


class RustKernelRunner:
    """Executes and reports on the Rust test suite and benchmark binary."""

    @staticmethod
    def run_tests(crate_dir: Path) -> Tuple[bool, str]:
        """Runs `cargo test` and returns (passed, stdout)."""
        try:
            res = subprocess.run(
                ["cargo", "test", "--", "--nocapture"],
                cwd=crate_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            return (res.returncode == 0), res.stdout + "\n" + res.stderr
        except Exception as exc:
            return False, f"Failed to execute cargo test: {exc}"

    @staticmethod
    def run_benchmark(crate_dir: Path) -> Tuple[bool, str]:
        """Runs `cargo run --release --bin bench` and returns (passed, stdout)."""
        try:
            res = subprocess.run(
                ["cargo", "run", "--release", "--bin", "bench"],
                cwd=crate_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            return (res.returncode == 0), res.stdout + "\n" + res.stderr
        except Exception as exc:
            return False, f"Failed to execute cargo benchmark: {exc}"
