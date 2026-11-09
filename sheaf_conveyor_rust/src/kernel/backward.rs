//! Fused Backward Adjoint Mechanics & Autograd Bridge.
//!
//! Implements Section 6 of the specification:
//! Computes exact gradients with respect to input stalk embeddings $\partial \mathcal{L} / \partial X$
//! and restriction maps $\partial \mathcal{L} / \partial \rho$ simultaneously.

use crate::domain::bsr::BlockSparseSheafGraph;

/// Output gradients for a single backward sheaf diffusion layer.
pub struct SheafGradients {
    /// Gradient with respect to input stalks: shape `|V| * D * C`.
    pub grad_x_in: Vec<f32>,
    /// Gradient with respect to restriction maps: shape `|E| * D * D`.
    pub grad_rho: Vec<f32>,
}

/// Evaluates a single backward adjoint step for cellular sheaf diffusion.
///
/// # Arguments
/// * `graph` - Validated BlockSparseSheafGraph.
/// * `x_in` - Forward activations $X^{(l)}$ of size `|V| * D * C`.
/// * `grad_x_out` - Upstream gradient $\partial \mathcal{L} / \partial X^{(l+1)}$ of size `|V| * D * C`.
/// * `channels` - Channel count per stalk $C$.
/// * `tau` - Diffusion step size $\tau$.
pub fn fused_sheaf_diffusion_backward_step(
    graph: &BlockSparseSheafGraph,
    x_in: &[f32],
    grad_x_out: &[f32],
    channels: usize,
    tau: f32,
) -> SheafGradients {
    let d = graph.stalk_dim;
    let num_nodes = graph.num_nodes;
    let node_stride = d * channels;
    let rho_stride = d * d;

    let mut grad_x_in = vec![0.0f32; num_nodes * node_stride];
    let mut grad_rho = vec![0.0f32; graph.num_edges * rho_stride];

    // 1. Compute direct decay self-gradient: (1 - tau * deg(u)) * grad_x_out[u]
    for u in 0..num_nodes {
        let degree = graph.degrees[u] as f32;
        let decay = 1.0f32 - tau * degree;
        let u_offset = u * node_stride;

        for idx in 0..node_stride {
            grad_x_in[u_offset + idx] += decay * grad_x_out[u_offset + idx];
        }
    }

    // 2. Accumulate incoming adjoint edges and compute rank-1 outer product for grad_rho
    for v in 0..num_nodes {
        let start_edge = graph.row_ptr[v];
        let end_edge = graph.row_ptr[v + 1];
        let v_offset = v * node_stride;
        let grad_v = &grad_x_out[v_offset..v_offset + node_stride];

        for e in start_edge..end_edge {
            let u = graph.col_ind[e];
            let u_offset = u * node_stride;
            let x_u = &x_in[u_offset..u_offset + node_stride];
            let rho = graph.get_rho_slice(e);

            let e_rho_offset = e * rho_stride;
            let grad_rho_slice = &mut grad_rho[e_rho_offset..e_rho_offset + rho_stride];

            // Edge gradient: \grad_{x_u} += tau * rho^T * grad_v
            for i in 0..d {
                for j in 0..d {
                    let r = rho[i * d + j];
                    for ch in 0..channels {
                        // rho^T[j, i] = rho[i, j]
                        grad_x_in[u_offset + (j * channels + ch)] += tau * r * grad_v[i * channels + ch];
                    }
                }
            }

            // Restriction map gradient via Rank-1 Outer Product:
            // \grad_rho[i, j] = tau * sum_ch (grad_v[i, ch] * x_u[j, ch])
            for i in 0..d {
                for j in 0..d {
                    let mut dot = 0.0f32;
                    for ch in 0..channels {
                        dot += grad_v[i * channels + ch] * x_u[j * channels + ch];
                    }
                    grad_rho_slice[i * d + j] += tau * dot;
                }
            }
        }
    }

    SheafGradients { grad_x_in, grad_rho }
}
