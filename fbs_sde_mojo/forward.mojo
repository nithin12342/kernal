# Fused Forward Sheaf Diffusion Engine in Mojo.
# Evaluates:
# x_v^(l+1) = (1 - tau * deg(v)) * x_v^(l) + tau * sum_{e=(u, v)} rho_e * x_u
# using parallelization and circular double buffering.

from algorithm import parallelize
from memory import UnsafePointer
from fbs_sde_mojo.contraction import contract_sheaf_edge_simd
from fbs_sde_mojo.telemetry import compute_dirichlet_energy_edge
from fbs_sde_mojo.types import BlockSparseSheafGraph, SheafTelemetryRecord


struct FusedSheafEngine[D: Int, CHANNELS: Int, SIMD_W: Int = 8]:
    """Parallelized, SIMD-accelerated cellular sheaf diffusion engine."""
    var graph: BlockSparseSheafGraph

    fn __init__(inout self, graph: BlockSparseSheafGraph):
        self.graph = graph

    fn forward_layer(
        self,
        x_in: UnsafePointer[Float32],
        x_out: UnsafePointer[Float32],
        tau: Float32
    ) -> SheafTelemetryRecord:
        """Executes a single fused forward diffusion layer over all nodes."""
        let node_stride = D * CHANNELS

        @parameter
        fn process_node(v: Int):
            let start_edge = self.graph.row_ptr[v]
            let end_edge = self.graph.row_ptr[v + 1]
            let degree = Float32(end_edge - start_edge)
            let decay = 1.0 - tau * degree

            let v_offset = v * node_stride
            let x_v = x_in + v_offset

            # Allocate in-register / local accumulator
            let acc = UnsafePointer[Float32].alloc(node_stride)
            for idx in range(node_stride):
                acc[idx] = 0.0

            for e in range(start_edge, end_edge):
                let u = self.graph.col_ind[e]
                let u_offset = u * node_stride
                let x_u = x_in + u_offset
                let rho = self.graph.rho_data + self.graph.get_edge_rho_offset(e)

                contract_sheaf_edge_simd[D, CHANNELS, SIMD_W](rho, x_u, acc)

            # Store updated node stalk
            let dst_v = x_out + v_offset
            for idx in range(node_stride):
                dst_v[idx] = decay * x_v[idx] + tau * acc[idx]

            acc.free()

        parallelize[process_node](self.graph.num_nodes)
        return SheafTelemetryRecord(0.0)
