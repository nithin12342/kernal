"""Graph Topology & Block-Sparse Row Packaging Subpackage.

Converts edge lists and restriction maps into BSR tensors and validates
structural invariants against isolated nodes, pendant nodes, and dense clusters.
"""

from .bsr_packager import BSRSerializationResult, BSRPackager
from .topology_cases import TopologyGenerator

__all__ = ["BSRSerializationResult", "BSRPackager", "TopologyGenerator"]
