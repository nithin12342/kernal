//! Fused Forward Sheaf Diffusion Engine.
//!
//! Evaluates the discretized cellular sheaf diffusion equation:
//! $X^{(l+1)} = (1 - \tau \deg(v)) X^{(l)} + \tau \sum_{e=(u,v)} \rho_{v \trianglelefteq e}^T \rho_{u \trianglelefteq e} X_u^{(l)}$
//! with multi-hop in-memory ping-pong double buffering and live telemetry.

use crate::domain::bsr::BlockSparseSheafGraph;
use crate::domain::telemetry::SheafTelemetry;
use crate::kernel::contraction::contract_edge_dynamic;

/// Executes multi-hop fused sheaf diffusion over `num_layers` iterations.
///
/// # Arguments
/// * `graph` - Validated Block-Sparse Sheaf Graph.
/// * `x_initial` - Input node stalk tensor of size `|V| * D * C`.
/// * `channels` - Feature channels per stalk node (e.g. 32).
/// * `tau` - Diffusion step size $\tau > 0$.
/// * `num_layers` - Number of diffusion layers $L$.
///
/// # Returns
/// A tuple containing:
/// * Final diffused stalk tensor of size `|V| * D * C`.
/// * Vector of per-layer `SheafTelemetry` metrics.
pub fn fused_sheaf_diffusion_forward(
    graph: &BlockSparseSheafGraph,
    x_initial: &[f32],
    channels: usize,
    tau: f32,
    num_layers: usize,
) -> (Vec<f32>, Vec<SheafTelemetry>) {
    let d = graph.stalk_dim;
    let num_nodes = graph.num_nodes;
    let node_stride = d * channels;
    let total_elements = num_nodes * node_stride;

    assert_eq!(
        x_initial.len(),
        total_elements,
        "x_initial length must equal num_nodes * stalk_dim * channels"
    );

    // Ping-pong double buffers to avoid memory reallocation
    let mut buffer_a = x_initial.to_vec();
    let mut buffer_b = vec![0.0f32; total_elements];

    let mut telemetry_history = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let (src, dst) = if layer % 2 == 0 {
            (&buffer_a, &mut buffer_b)
        } else {
            (&buffer_b, &mut buffer_a)
        };

        let mut layer_dirichlet_energy = 0.0f32;

        // Fused Node Update: Compute x_v^(l+1) in a single pass
        for node in 0..num_nodes {
            let start_edge = graph.row_ptr[node];
            let end_edge = graph.row_ptr[node + 1];
            let degree = graph.degrees[node] as f32;

            let node_offset = node * node_stride;
            let self_stalk = &src[node_offset..node_offset + node_stride];

            // Local accumulator for edge tension: \sum \rho_{v \le e}^T \rho_{u \le e} x_u
            let mut acc = vec![0.0f32; node_stride];

            for e in start_edge..end_edge {
                let neighbor = graph.col_ind[e];
                let neighbor_offset = neighbor * node_stride;
                let neighbor_stalk = &src[neighbor_offset..neighbor_offset + node_stride];
                let rho = graph.get_rho_slice(e);

                contract_edge_dynamic(rho, neighbor_stalk, &mut acc, d, channels);

                // Live Dirichlet Energy Accumulation: 0.5 * ||rho * x_u - x_v||^2
                // When rho in O(d), Delta x_e = rho * x_u - x_v
                for i in 0..d {
                    for ch in 0..channels {
                        let mut contracted_ui = 0.0f32;
                        for j in 0..d {
                            contracted_ui += rho[i * d + j] * neighbor_stalk[j * channels + ch];
                        }
                        let diff = contracted_ui - self_stalk[i * channels + ch];
                        layer_dirichlet_energy += 0.5f32 * diff * diff;
                    }
                }
            }

            // Fused update: x_v^(l+1) = (1 - tau * deg(v)) * x_v^(l) + tau * acc
            let self_decay = 1.0f32 - tau * degree;
            let dst_slice = &mut dst[node_offset..node_offset + node_stride];

            for idx in 0..node_stride {
                dst_slice[idx] = self_decay * self_stalk[idx] + tau * acc[idx];
            }
        }

        let is_unstable = layer_dirichlet_energy.is_nan() || layer_dirichlet_energy.is_infinite();
        let is_collapsed = !is_unstable && layer_dirichlet_energy < 1e-7;

        telemetry_history.push(SheafTelemetry {
            dirichlet_energy: layer_dirichlet_energy,
            max_orthogonality_drift: 0.0,
            is_unstable,
            is_collapsed,
        });
    }

    let final_result = if num_layers.is_multiple_of(2) {
        buffer_a
    } else {
        buffer_b
    };

    (final_result, telemetry_history)
}
