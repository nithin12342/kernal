//! In-Register Dense Matrix-Vector Contraction Primitives.
//!
//! Evaluates local sheaf edge tensor contractions:
//! $acc_i += \sum_j \rho_{i, j} \cdot x_{u, j}$
//! with compile-time loop unrolling and zero dynamic branch overhead.

/// Contracts a dense $D \times D$ orthogonal restriction map with a neighbor stalk $x_u$ of shape $D \times C$.
///
/// Accumulates directly in-place: `acc[i][c] += rho[i * D + j] * x_u[j][c]`.
#[inline(always)]
pub fn contract_edge_stalk<const D: usize, const C: usize>(
    rho: &[f32],
    x_u: &[[f32; C]; D],
    acc: &mut [[f32; C]; D],
) {
    debug_assert!(rho.len() >= D * D);
    for (i, acc_row) in acc.iter_mut().enumerate().take(D) {
        let row_offset = i * D;
        for j in 0..D {
            let r = rho[row_offset + j];
            for ch in 0..C {
                acc_row[ch] += r * x_u[j][ch];
            }
        }
    }
}

/// Dynamic fallback for arbitrary runtime stalk dimension and channel sizes.
#[inline(always)]
pub fn contract_edge_dynamic(
    rho: &[f32],
    x_u: &[f32],
    acc: &mut [f32],
    d: usize,
    c: usize,
) {
    for i in 0..d {
        let row_offset = i * d;
        let out_offset = i * c;
        for j in 0..d {
            let r = rho[row_offset + j];
            let in_offset = j * c;
            for ch in 0..c {
                acc[out_offset + ch] += r * x_u[in_offset + ch];
            }
        }
    }
}
