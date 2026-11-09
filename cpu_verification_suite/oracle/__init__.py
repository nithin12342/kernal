"""Oracle Subpackage for CPU-Only Reference Validation.

Provides 64-bit floating point mathematical references for Sheaf Laplacian
construction, diffusion dynamics, and algebraic invariant proofs.
"""

from .sheaf_laplacian import SheafLaplacianOracle
from .invariants import AlgebraicInvariantProver

__all__ = ["SheafLaplacianOracle", "AlgebraicInvariantProver"]
