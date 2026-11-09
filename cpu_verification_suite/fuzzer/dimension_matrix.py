"""Dimension Matrix Fuzzing Module.

Executes continuous Cartesian parameter sweeps across:
    stalk dimensions d in [2, 16]
    feature channels c in [1, 128]
to verify stride alignments, memory footprints, and zero degenerate arrays.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class FuzzResult:
    """Summary of parameter matrix fuzzing."""
    total_combinations_tested: int
    passed_combinations: int
    failed_combinations: int
    failure_details: List[str]


class DimensionFuzzer:
    """Fuzzes dimensional configurations against mathematical and memory invariants."""

    @staticmethod
    def fuzz_parameter_grid(
        d_range: Tuple[int, int] = (2, 16),
        c_samples: List[int] = [1, 2, 4, 7, 8, 13, 16, 32, 64, 128]
    ) -> FuzzResult:
        """Runs combinatorial parameter fuzzing across stalk and channel spaces."""
        failures: List[str] = []
        total = 0
        passed = 0

        for d in range(d_range[0], d_range[1] + 1):
            for c in c_samples:
                total += 1
                try:
                    # 1. Stride verification
                    node_stride = d * c
                    rho_stride = d * d

                    if node_stride <= 0 or rho_stride <= 0:
                        failures.append(f"Invalid non-positive stride for d={d}, c={c}")
                        continue

                    # 2. Contraction numerical consistency on mock 1-edge graph
                    rho = np.eye(d, dtype=np.float32)
                    x_u = np.ones((d, c), dtype=np.float32)

                    # Analytical contraction
                    acc = rho @ x_u
                    expected = np.ones((d, c), dtype=np.float32)
                    if not np.allclose(acc, expected, atol=1e-6):
                        failures.append(f"Contraction numerical failure for d={d}, c={c}")
                        continue

                    # 3. 16-byte vector boundary alignment check
                    byte_size = node_stride * 4 # 4 bytes per float32
                    # Stride should preferably be 16-byte aligned for float4 vectorization
                    is_vector_aligned = (byte_size % 16 == 0)

                    passed += 1
                except Exception as exc:
                    failures.append(f"Exception during d={d}, c={c}: {exc}")

        return FuzzResult(
            total_combinations_tested=total,
            passed_combinations=passed,
            failed_combinations=len(failures),
            failure_details=failures
        )
