"""CPU Verification Suite Package.

Top-level routing node for CPU-only, static CI, and pre-silicon validation
of the Fused Block-Sparse Sheaf Diffusion Engine (FBS-SDE).
"""

from .oracle import SheafLaplacianOracle, AlgebraicInvariantProver
from .graph import BSRPackager, TopologyGenerator
from .memory import MockWriteCombinedRingBuffer, AlignmentChecker
from .static_analysis import KernelSyntaxAuditor, RustClippyRunner
from .fuzzer import DimensionFuzzer
from .runners import RustKernelRunner, MojoSemanticValidator

__all__ = [
    "SheafLaplacianOracle",
    "AlgebraicInvariantProver",
    "BSRPackager",
    "TopologyGenerator",
    "MockWriteCombinedRingBuffer",
    "AlignmentChecker",
    "KernelSyntaxAuditor",
    "RustClippyRunner",
    "DimensionFuzzer",
    "RustKernelRunner",
    "MojoSemanticValidator",
]
