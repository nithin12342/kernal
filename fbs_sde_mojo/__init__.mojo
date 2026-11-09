# Fused Block-Sparse Sheaf Diffusion Engine (FBS-SDE) for Mojo.
# Root package entry point.

from fbs_sde_mojo.types import BlockSparseSheafGraph, SheafTelemetryRecord
from fbs_sde_mojo.contraction import contract_sheaf_edge, contract_sheaf_edge_simd
from fbs_sde_mojo.forward import FusedSheafEngine
from fbs_sde_mojo.backward import SheafBackwardAdjointEngine
from fbs_sde_mojo.telemetry import compute_dirichlet_energy_edge, compute_orthogonality_drift
from fbs_sde_mojo.metaprogramming import generate_mojo_kernel_spec
