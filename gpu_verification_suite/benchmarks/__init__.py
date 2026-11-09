"""Benchmark Evaluation Subpackage for the 16 GPU Tests.

Organized into modular evaluators:
- numerical_equivalence: Tests 1 & 2 (FP equivalence & gradcheck)
- sanitizer_verifier: Tests 3-6 (memcheck, racecheck, synccheck, initcheck)
- hardware_profiler: Tests 7-11 (register spills, branch divergence, bank conflicts, latency hiding, SOL roofline)
- runtime_systems: Tests 12-15 (CUDA graph, multi-stream, write-combined PCIe, VRAM locality)
- determinism_stress: Test 16 (10,000-run bit-exact hashing)
"""

from .numerical_equivalence import evaluate_numerical_tests
from .sanitizer_verifier import evaluate_sanitizer_tests
from .hardware_profiler import evaluate_hardware_profiler_tests
from .runtime_systems import evaluate_runtime_systems_tests
from .determinism_stress import evaluate_determinism_test

__all__ = [
    "evaluate_numerical_tests",
    "evaluate_sanitizer_tests",
    "evaluate_hardware_profiler_tests",
    "evaluate_runtime_systems_tests",
    "evaluate_determinism_test",
]
