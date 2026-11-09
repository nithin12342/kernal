"""Benchmark package router.

Exposes the public benchmarking and logging contract. External consumers
import from this entry point only.
"""

from .runner import run_benchmark
from .logger import BenchmarkLogger

__all__ = ["run_benchmark", "BenchmarkLogger"]
