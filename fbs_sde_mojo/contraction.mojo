# In-Register Dense Matrix-Vector Contraction in Mojo.
# Evaluates local sheaf edge tensor contractions:
# acc_i += sum_j (rho_{i, j} * x_{u, j})
# using compile-time @parameter unrolling and hardware SIMD vectors.

from memory import UnsafePointer


@always_inline
fn contract_sheaf_edge[D: Int, CHANNELS: Int](
    rho: UnsafePointer[Float32],
    x_u: UnsafePointer[Float32],
    inout acc: SIMD[DType.float32, D * CHANNELS]
):
    """Contracts D x D restriction map with D x CHANNELS neighbor stalk."""
    @parameter
    for i in range(D):
        let row_offset = i * D
        @parameter
        for j in range(D):
            let r = rho[row_offset + j]
            let in_offset = j * CHANNELS
            @parameter
            for ch in range(CHANNELS):
                let u_val = x_u[in_offset + ch]
                acc[i * CHANNELS + ch] += r * u_val


@always_inline
fn contract_sheaf_edge_simd[D: Int, CHANNELS: Int, SIMD_W: Int](
    rho: UnsafePointer[Float32],
    x_u: UnsafePointer[Float32],
    inout acc: UnsafePointer[Float32]
):
    """Vectorized block contraction using SIMD slices of width SIMD_W."""
    @parameter
    for i in range(D):
        let row_offset = i * D
        let out_offset = i * CHANNELS
        @parameter
        for j in range(D):
            let r_scalar = rho[row_offset + j]
            let r_vec = SIMD[DType.float32, SIMD_W](r_scalar)
            let in_offset = j * CHANNELS

            var ch_idx = 0
            while ch_idx + SIMD_W <= CHANNELS:
                let x_vec = x_u.load[width=SIMD_W](in_offset + ch_idx)
                let current_acc = acc.load[width=SIMD_W](out_offset + ch_idx)
                acc.store[width=SIMD_W](out_offset + ch_idx, current_acc + r_vec * x_vec)
                ch_idx += SIMD_W

            # Handle channel remainder scalar loop
            while ch_idx < CHANNELS:
                acc[out_offset + ch_idx] += r_scalar * x_u[in_offset + ch_idx]
                ch_idx += 1
