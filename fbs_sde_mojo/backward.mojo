# Exact Fused Backward Adjoint Mechanics in Mojo.
# Implements simultaneous gradient calculation for input stalks
# dL / dX and restriction maps dL / drho.

from algorithm import parallelize
from memory import UnsafePointer
from fbs_sde_mojo.types import BlockSparseSheafGraph


struct SheafBackwardAdjointEngine[D: Int, CHANNELS: Int]:
    """Evaluates adjoint gradients for cellular sheaf diffusion in Mojo."""
    var graph: BlockSparseSheafGraph

    fn __init__(inout self, graph: BlockSparseSheafGraph):
        self.graph = graph

    fn backward_step(
        self,
        x_in: UnsafePointer[Float32],
        grad_out: UnsafePointer[Float32],
        grad_x_in: UnsafePointer[Float32],
        grad_rho: UnsafePointer[Float32],
        tau: Float32
    ):
        """Computes gradients with respect to x_in and rho."""
        let node_stride = D * CHANNELS
        let rho_stride = D * D

        # 1. Direct decay self-gradient
        @parameter
        fn compute_self_gradient(u: Int):
            let degree = Float32(self.graph.row_ptr[u + 1] - self.graph.row_ptr[u])
            let decay = 1.0 - tau * degree
            let u_offset = u * node_stride

            for idx in range(node_stride):
                grad_x_in[u_offset + idx] = decay * grad_out[u_offset + idx]

        parallelize[compute_self_gradient](self.graph.num_nodes)

        # 2. Scatter-free incoming adjoint edges and rank-1 outer product
        for v in range(self.graph.num_nodes):
            let start_edge = self.graph.row_ptr[v]
            let end_edge = self.graph.row_ptr[v + 1]
            let v_offset = v * node_stride
            let grad_v = grad_out + v_offset

            for e in range(start_edge, end_edge):
                let u = self.graph.col_ind[e]
                let u_offset = u * node_stride
                let x_u = x_in + u_offset
                let rho = self.graph.rho_data + (e * rho_stride)
                let grad_rho_e = grad_rho + (e * rho_stride)

                # Upstream node gradient: grad_x_u += tau * rho^T * grad_v
                @parameter
                for i in range(D):
                    @parameter
                    for j in range(D):
                        let r = rho[i * D + j]
                        @parameter
                        for ch in range(CHANNELS):
                            grad_x_in[u_offset + (j * CHANNELS + ch)] += tau * r * grad_v[i * CHANNELS + ch]

                # Rank-1 Outer Product for restriction map: grad_rho = tau * (grad_v * x_u^T)
                @parameter
                for i in range(D):
                    @parameter
                    for j in range(D):
                        var dot: Float32 = 0.0
                        @parameter
                        for ch in range(CHANNELS):
                            dot += grad_v[i * CHANNELS + ch] * x_u[j * CHANNELS + ch]
                        grad_rho_e[i * D + j] += tau * dot
