# ==============================================================================
# LAYER 4: MOJO (MODULAR MAX) - FUSED BLOCK-SPARSE SHEAF GPU OPERATOR
# Target: Hardware Accelerator / MLIR SIMD Pipeline
# ==============================================================================

from memory import UnsafePointer
from sys.info import simdwidthof

alias D = 4
alias C = 32
alias STRIDE = D * C
alias RHO_STRIDE = D * D
alias SIMD_WIDTH = 8

struct SheafGraphMojo:
    var num_nodes: Int
    var num_edges: Int
    var row_ptr: UnsafePointer[Int32]
    var col_ind: UnsafePointer[Int32]
    var rho_data: UnsafePointer[Float32]

    fn __init__(
        out self,
        num_nodes: Int,
        num_edges: Int,
        row_ptr: UnsafePointer[Int32],
        col_ind: UnsafePointer[Int32],
        rho_data: UnsafePointer[Float32],
    ):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.row_ptr = row_ptr
        self.col_ind = col_ind
        self.rho_data = rho_data

@always_inline
fn fused_sheaf_diffusion_step_mojo(
    graph: SheafGraphMojo,
    x_src: UnsafePointer[Float32],
    x_dst: UnsafePointer[Float32],
    tau: Float32,
    num_nodes: Int,
):
    """
    Fused block-sparse sheaf diffusion step.
    Tiles node stalks and restriction maps in registers/SRAM,
    evaluates coboundary contraction, and updates node states in a single pass.
    """
    for node in range(num_nodes):
        var start_edge = int(graph.row_ptr[node])
        var end_edge = int(graph.row_ptr[node + 1])
        var degree = Float32(end_edge - start_edge)
        var decay = 1.0 - tau * degree

        var node_off = node * STRIDE
        var acc = InlineArray[Float32, STRIDE](0.0)

        for e in range(start_edge, end_edge):
            var neighbor = int(graph.col_ind[e])
            var neighbor_off = neighbor * STRIDE
            var rho_off = e * RHO_STRIDE

            # Register-tiled 4x4 matrix contraction with 32-channel stalk
            for i in range(D):
                var r0 = graph.rho_data[rho_off + i * 4 + 0]
                var r1 = graph.rho_data[rho_off + i * 4 + 1]
                var r2 = graph.rho_data[rho_off + i * 4 + 2]
                var r3 = graph.rho_data[rho_off + i * 4 + 3]

                for ch in range(C):
                    var u0 = x_src[neighbor_off + 0 * C + ch]
                    var u1 = x_src[neighbor_off + 1 * C + ch]
                    var u2 = x_src[neighbor_off + 2 * C + ch]
                    var u3 = x_src[neighbor_off + 3 * C + ch]

                    acc[i * C + ch] += r0 * u0 + r1 * u1 + r2 * u2 + r3 * u3

        for idx in range(STRIDE):
            var self_val = x_src[node_off + idx]
            x_dst[node_off + idx] = decay * self_val + tau * acc[idx]
