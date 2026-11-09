# Staged Metaprogramming and Comptime Dynamic Kernel Emission in Mojo.
# Uses compile-time parameters and unrolling to synthesize loopless
# matrix-vector sheaf diffusion pipelines.

from fbs_sde_mojo.types import BlockSparseSheafGraph


@always_inline
fn execute_unrolled_sheaf_contraction[
    D: Int,
    CHANNELS: Int
](
    rho: Pointer[Float32],
    x_neighbor: Pointer[Float32],
    acc: Pointer[Float32]
):
    """Zero-overhead compile-time unrolled matrix contraction."""
    @parameter
    for i in range(D):
        let row_offset = i * D
        let out_offset = i * CHANNELS
        @parameter
        for j in range(D):
            let r = rho[row_offset + j]
            let in_offset = j * CHANNELS
            @parameter
            for ch in range(CHANNELS):
                acc[out_offset + ch] += r * x_neighbor[in_offset + ch]


fn generate_mojo_kernel_spec(d: Int, channels: Int) -> String:
    """Emits high-level specialization summary for Mojo JIT compilation."""
    var spec = String("FBS-SDE Mojo Kernel Specialization:\n")
    spec += "  Stalk Dimension D: " + String(d) + "\n"
    spec += "  Feature Channels C: " + String(channels) + "\n"
    spec += "  Contraction Complexity: " + String(d * d * channels) + " FMAs per edge\n"
    spec += "  Register Allocation: " + String(d * channels) + " floats per node stalk\n"
    return spec
