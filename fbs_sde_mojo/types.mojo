# Block-Sparse Sheaf Graph & Buffer Types for Mojo.
# Maps non-Euclidean Sheaf Laplacians into contiguous memory layouts:
# - row_ptr: |V| + 1 array indexing edge ranges.
# - col_ind: |E| array of neighbor node indices.
# - rho_data: |E| * D * D continuous restriction maps in O(d).

from memory import UnsafePointer


@value
struct BlockSparseSheafGraph:
    """Represents a validated Block-Sparse Row (BSR) graph structure."""
    var num_nodes: Int
    var num_edges: Int
    var stalk_dim: Int
    var row_ptr: UnsafePointer[Int]
    var col_ind: UnsafePointer[Int]
    var rho_data: UnsafePointer[Float32]

    fn __init__(
        inout self,
        num_nodes: Int,
        num_edges: Int,
        stalk_dim: Int,
        row_ptr: UnsafePointer[Int],
        col_ind: UnsafePointer[Int],
        rho_data: UnsafePointer[Float32]
    ):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.stalk_dim = stalk_dim
        self.row_ptr = row_ptr
        self.col_ind = col_ind
        self.rho_data = rho_data

    @always_inline
    fn get_edge_rho_offset(self, edge: Int) -> Int:
        """Computes memory offset for edge restriction map of size D x D."""
        return edge * (self.stalk_dim * self.stalk_dim)


@value
struct SheafTelemetryRecord:
    """In-kernel telemetry metrics collected across diffusion steps."""
    var dirichlet_energy: Float32
    var is_unstable: Bool
    var is_collapsed: Bool

    fn __init__(inout self, energy: Float32 = 0.0):
        self.dirichlet_energy = energy
        self.is_unstable = False
        self.is_collapsed = energy < 1e-7
