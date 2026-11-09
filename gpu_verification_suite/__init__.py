"""GPU Verification Suite Package.

Evaluates all 16 GPU-specific benchmarks across the three kernel targets:
1. Rust with Metaprogramming (ConstSheafEngine)
2. Rust Standard Kernel (BSR ping-pong)
3. Mojo Kernel (Hardware SIMD & parameter unrolling)
"""

from .models import (
    BenchmarkMatrixRow,
    FullEvaluationMatrix,
    KernelMeasurement,
    TargetThreshold,
)
from .benchmarks import (
    evaluate_determinism_test,
    evaluate_hardware_profiler_tests,
    evaluate_numerical_tests,
    evaluate_runtime_systems_tests,
    evaluate_sanitizer_tests,
)

__all__ = [
    "BenchmarkMatrixRow",
    "FullEvaluationMatrix",
    "KernelMeasurement",
    "TargetThreshold",
    "evaluate_determinism_test",
    "evaluate_hardware_profiler_tests",
    "evaluate_numerical_tests",
    "evaluate_runtime_systems_tests",
    "evaluate_sanitizer_tests",
]
