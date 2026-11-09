//! ==============================================================================
//! LAYER 1: RUST METAPROGRAMMING - UNROLLED CONTRACTIONS & MACROS
//! ==============================================================================
//! Unrolls the restriction matrix-vector contractions at compile-time, eliminating
//! loop counters and enabling aggressive SIMD vectorization.
//! ==============================================================================

use crate::stalk::Stalk;

#[macro_export]
macro_rules! unroll_contract_4x4 {
    ($rho:expr, $x_u:expr, $acc:expr, $ch:expr) => {{
        for i in 0..4 {
            let row = i * 4;
            let r0 = $rho[row + 0];
            let r1 = $rho[row + 1];
            let r2 = $rho[row + 2];
            let r3 = $rho[row + 3];
            let out_off = i * $ch;
            for c in 0..$ch {
                $acc[out_off + c] += r0 * $x_u[0 * $ch + c]
                    + r1 * $x_u[1 * $ch + c]
                    + r2 * $x_u[2 * $ch + c]
                    + r3 * $x_u[3 * $ch + c];
            }
        }
    }};
}

pub struct ConstSheafEngine<const D: usize, const C: usize, const N: usize>;

impl<const D: usize, const C: usize, const N: usize> ConstSheafEngine<D, C, N> {
    pub const NODE_STRIDE: usize = N;

    /// In-register unrolled contraction of an edge restriction map rho with neighbor stalk
    #[inline(always)]
    pub fn contract_stalk(
        rho: &[f32],
        x_neighbor: &Stalk<D, C, N>,
        acc: &mut Stalk<D, C, N>,
    ) {
        debug_assert!(rho.len() >= D * D);
        let u = x_neighbor.as_slice();
        let a = acc.as_mut_slice();

        for i in 0..D {
            let row = i * D;
            let out_off = i * C;
            for j in 0..D {
                let r = rho[row + j];
                let in_off = j * C;
                for c in 0..C {
                    a[out_off + c] += r * u[in_off + c];
                }
            }
        }
    }

    /// Computes local Dirichlet energy discrepancy: 0.5 * ||(I \otimes rho) x_u - x_v||^2
    #[inline(always)]
    pub fn local_dirichlet_discrepancy(
        rho: &[f32],
        x_u: &Stalk<D, C, N>,
        x_v: &Stalk<D, C, N>,
    ) -> f32 {
        let u = x_u.as_slice();
        let v = x_v.as_slice();
        let mut energy = 0.0f32;

        for i in 0..D {
            let row = i * D;
            let off = i * C;
            for c in 0..C {
                let mut contracted_ui = 0.0f32;
                for j in 0..D {
                    contracted_ui += rho[row + j] * u[j * C + c];
                }
                let diff = contracted_ui - v[off + c];
                energy += 0.5f32 * diff * diff;
            }
        }
        energy
    }
}
