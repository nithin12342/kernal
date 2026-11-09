"""Data Models and Matrix Types for the 16 GPU-Specific Benchmark Tests.

Encapsulates:
- TargetThreshold: Numerical acceptance limits for each benchmark.
- KernelMetricValue: Measurable numeric achievement for a specific kernel.
- BenchmarkResultRow: Target vs. Achieved comparison across all three kernels.
- FullEvaluationMatrix: Complete 16-test evaluation matrix report.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TargetThreshold:
    """Acceptance threshold specification for a benchmark."""
    test_id: int
    name: str
    metric_unit: str
    target_value: float
    comparator: str  # '<=', '>=', '=='
    target_display: str


@dataclass
class KernelMeasurement:
    """Numeric measurement achieved by a single kernel."""
    numeric_value: float
    display_value: str
    passed: bool
    detail: str = ""


@dataclass
class BenchmarkMatrixRow:
    """Row in the comparison matrix comparing all 3 kernels against the target."""
    test_id: int
    test_name: str
    metric_unit: str
    target_display: str
    rust_metaprog: KernelMeasurement
    rust_standard: KernelMeasurement
    mojo_kernel: KernelMeasurement
    all_passed: bool


@dataclass
class FullEvaluationMatrix:
    """Complete 16-test evaluation matrix across all three kernels."""
    rows: List[BenchmarkMatrixRow]
    total_tests: int = 16
    passed_count_rust_meta: int = 0
    passed_count_rust_std: int = 0
    passed_count_mojo: int = 0
    overall_passed: bool = False
