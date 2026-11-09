//! Pure Const-Generic Compile-Time Metaprogramming for Sheaf Diffusion.
//!
//! Synthesizes unrolled, zero-branch in-register contractions directly
//! within Rust's type system via const generics and compile-time evaluation.

use crate::domain::bsr::BlockSparseSheafGraph;
use crate::domain::telemetry::SheafTelemetry;
use crate::kernel::backward::SheafGradients;

/// Compile-time metaprogrammed diffusion engine specialized for (D, C).
pub struct ConstSheafEngine<const D: usize, const C: usize>;

impl<const D: usize, const C: usize> ConstSheafEngine<D, C> {
    pub const NODE_STRIDE: usize = D * C;
    pub const RHO_STRIDE: usize = D * D;

    /// Evaluates in-register unrolled contraction: acc[i][ch] += rho[i * D + j] * x_u[j * C + ch].
    #[inline(always)]
    pub fn contract_in_register(
        rho: &[f32],
        x_u: &[f32],
        acc: &mut [f32],
    ) {
        debug_assert_eq!(acc.len(), Self::NODE_STRIDE);
        for i in 0..D {
            let row = i * D;
            let out_off = i * C;
            for j in 0..D {
                let r = rho[row + j];
                let in_off = j * C;
                for ch in 0..C {
                    acc[out_off + ch] += r * x_u[in_off + ch];
                }
            }
        }
    }

    /// Evaluates forward diffusion step using compile-time constants.
    pub fn forward_step(
        graph: &BlockSparseSheafGraph,
        x_in: &[f32],
        x_out: &mut [f32],
        tau: f32,
    ) -> SheafTelemetry {
        let mut total_dirichlet_energy = 0.0f32;

        for node in 0..graph.num_nodes {
            let start = graph.row_ptr[node];
            let end = graph.row_ptr[node + 1];
            let degree = graph.degrees[node] as f32;
            let decay = 1.0f32 - tau * degree;

            let node_offset = node * Self::NODE_STRIDE;
            let x_self = &x_in[node_offset..node_offset + Self::NODE_STRIDE];

            let mut acc = vec![0.0f32; Self::NODE_STRIDE];

            for e in start..end {
                let neighbor = graph.col_ind[e];
                let neighbor_offset = neighbor * Self::NODE_STRIDE;
                let x_u = &x_in[neighbor_offset..neighbor_offset + Self::NODE_STRIDE];
                let rho = graph.get_rho_slice(e);

                Self::contract_in_register(rho, x_u, &mut acc);

                // Live Dirichlet energy calculation: 0.5 * ||rho * x_u - x_self||^2
                for i in 0..D {
                    let r_row = i * D;
                    let out_off = i * C;
                    for ch in 0..C {
                        let mut contracted_ui = 0.0f32;
                        for j in 0..D {
                            contracted_ui += rho[r_row + j] * x_u[j * C + ch];
                        }
                        let diff = contracted_ui - x_self[out_off + ch];
                        total_dirichlet_energy += 0.5f32 * diff * diff;
                    }
                }
            }

            let dst = &mut x_out[node_offset..node_offset + Self::NODE_STRIDE];
            for idx in 0..Self::NODE_STRIDE {
                dst[idx] = decay * x_self[idx] + tau * acc[idx];
            }
        }

        SheafTelemetry {
            dirichlet_energy: total_dirichlet_energy,
            max_orthogonality_drift: 0.0,
            is_unstable: total_dirichlet_energy.is_nan() || total_dirichlet_energy.is_infinite(),
            is_collapsed: total_dirichlet_energy < 1e-7,
        }
    }

    /// Evaluates exact backward adjoint step using compile-time constants.
    pub fn backward_step(
        graph: &BlockSparseSheafGraph,
        x_in: &[f32],
        grad_out: &[f32],
        tau: f32,
    ) -> SheafGradients {
        let num_nodes = graph.num_nodes;
        let mut grad_x_in = vec![0.0f32; num_nodes * Self::NODE_STRIDE];
        let mut grad_rho = vec![0.0f32; graph.num_edges * Self::RHO_STRIDE];

        // 1. Direct decay self-gradient
        for u in 0..num_nodes {
            let degree = graph.degrees[u] as f32;
            let decay = 1.0f32 - tau * degree;
            let u_offset = u * Self::NODE_STRIDE;

            for idx in 0..Self::NODE_STRIDE {
                grad_x_in[u_offset + idx] += decay * grad_out[u_offset + idx];
            }
        }

        // 2. Adjoint edges and rank-1 outer product
        for v in 0..num_nodes {
            let start_edge = graph.row_ptr[v];
            let end_edge = graph.row_ptr[v + 1];
            let v_offset = v * Self::NODE_STRIDE;
            let grad_v = &grad_out[v_offset..v_offset + Self::NODE_STRIDE];

            for e in start_edge..end_edge {
                let u = graph.col_ind[e];
                let u_offset = u * Self::NODE_STRIDE;
                let x_u = &x_in[u_offset..u_offset + Self::NODE_STRIDE];
                let rho = graph.get_rho_slice(e);

                let e_rho_offset = e * Self::RHO_STRIDE;
                let grad_rho_slice = &mut grad_rho[e_rho_offset..e_rho_offset + Self::RHO_STRIDE];

                // Node gradient: grad_x_u += tau * rho^T * grad_v
                for i in 0..D {
                    for j in 0..D {
                        let r = rho[i * D + j];
                        for ch in 0..C {
                            grad_x_in[u_offset + (j * C + ch)] += tau * r * grad_v[i * C + ch];
                        }
                    }
                }

                // Restriction map gradient via Rank-1 Outer Product
                for i in 0..D {
                    for j in 0..D {
                        let mut dot = 0.0f32;
                        for ch in 0..C {
                            dot += grad_v[i * C + ch] * x_u[j * C + ch];
                        }
                        grad_rho_slice[i * D + j] += tau * dot;
                    }
                }
            }
        }

        SheafGradients { grad_x_in, grad_rho }
    }
}
