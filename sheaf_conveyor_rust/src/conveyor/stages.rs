//! Assembly Line Stations for Cellular Sheaf Diffusion Conveyor Belt.
//!
//! Deconstructs the monolithic diffusion step into 4 modular conveyor stations:
//! 1. PrefetchStation: Gathers node stalk features from global memory into SRAM slots.
//! 2. ContractionStation: Unrolls dense restriction matrix contractions in registers.
//! 3. DiffusionStation: Evaluates degree decay and in-register Dirichlet energy accumulation.
//! 4. EgressStation: Writes completed node stalks to the destination output buffer.

use crate::domain::bsr::BlockSparseSheafGraph;
use crate::kernel::contraction::contract_edge_dynamic;

/// Working scratchpad for a single node flowing through the conveyor stations.
#[derive(Clone, Debug)]
pub struct NodeAssemblyState {
    /// Global index of the node being assembled.
    pub node_id: usize,
    /// Accumulator for edge tension: \sum \rho^T_{v \le e} \rho_{u \le e} x_u.
    pub acc: Vec<f32>,
    /// Accumulated local Dirichlet energy for this node's incident edges.
    pub local_dirichlet: f32,
}

impl NodeAssemblyState {
    /// Allocates an assembly state for a node with stalk stride `D * C`.
    pub fn new(node_id: usize, stride: usize) -> Self {
        Self {
            node_id,
            acc: vec![0.0f32; stride],
            local_dirichlet: 0.0f32,
        }
    }
}

/// Station 0: Prefetches node stalks and initializes the assembly state.
pub struct PrefetchStation;

impl PrefetchStation {
    /// Prepares assembly states for a batch of node indices.
    #[inline(always)]
    pub fn execute(node_ids: &[usize], stride: usize) -> Vec<NodeAssemblyState> {
        node_ids
            .iter()
            .map(|&id| NodeAssemblyState::new(id, stride))
            .collect()
    }
}

/// Station 1: Computes in-register restriction matrix contractions and Dirichlet energy.
pub struct ContractionStation;

impl ContractionStation {
    /// Processes all incident edges for the active node using dynamic BSR evaluation.
    #[inline(always)]
    pub fn execute_dynamic(
        state: &mut NodeAssemblyState,
        graph: &BlockSparseSheafGraph,
        x_src: &[f32],
        d: usize,
        channels: usize,
    ) {
        let node = state.node_id;
        let start = graph.row_ptr[node];
        let end = graph.row_ptr[node + 1];
        let stride = d * channels;
        let rho_stride = d * d;

        let node_off = node * stride;
        let self_stalk = &x_src[node_off..node_off + stride];

        for e in start..end {
            let neighbor = graph.col_ind[e];
            let neighbor_off = neighbor * stride;
            let neighbor_stalk = &x_src[neighbor_off..neighbor_off + stride];
            let rho = &graph.rho_data[e * rho_stride..(e + 1) * rho_stride];

            contract_edge_dynamic(rho, neighbor_stalk, &mut state.acc, d, channels);

            // Dirichlet energy: 0.5 * || rho * x_u - x_v ||^2
            for i in 0..d {
                for ch in 0..channels {
                    let mut contracted_ui = 0.0f32;
                    for j in 0..d {
                        contracted_ui += rho[i * d + j] * neighbor_stalk[j * channels + ch];
                    }
                    let diff = contracted_ui - self_stalk[i * channels + ch];
                    state.local_dirichlet += 0.5f32 * diff * diff;
                }
            }
        }
    }

    /// Processes incident edges using compile-time const generic specialization.
    #[inline(always)]
    pub fn execute_const<const D: usize, const C: usize>(
        state: &mut NodeAssemblyState,
        graph: &BlockSparseSheafGraph,
        x_src: &[f32],
    ) {
        let node = state.node_id;
        let start = graph.row_ptr[node];
        let end = graph.row_ptr[node + 1];
        let stride = D * C;
        let rho_stride = D * D;

        let node_off = node * stride;
        let self_stalk = &x_src[node_off..node_off + stride];

        for e in start..end {
            let neighbor = graph.col_ind[e];
            let neighbor_off = neighbor * stride;
            let neighbor_stalk = &x_src[neighbor_off..neighbor_off + stride];
            let rho = &graph.rho_data[e * rho_stride..(e + 1) * rho_stride];

            for i in 0..D {
                let row = i * D;
                let out_off = i * C;
                for j in 0..D {
                    let r = rho[row + j];
                    let in_off = j * C;
                    for ch in 0..C {
                        state.acc[out_off + ch] += r * neighbor_stalk[in_off + ch];
                    }
                }
            }

            // In-register Dirichlet accumulation
            for i in 0..D {
                for ch in 0..C {
                    let mut contracted_ui = 0.0f32;
                    for j in 0..D {
                        contracted_ui += rho[i * D + j] * neighbor_stalk[j * C + ch];
                    }
                    let diff = contracted_ui - self_stalk[i * C + ch];
                    state.local_dirichlet += 0.5f32 * diff * diff;
                }
            }
        }
    }
}

/// Station 2: Evaluates diffusion update.
pub struct DiffusionStation;

impl DiffusionStation {
    /// Computes $X^{(l+1)}_v = (1 - \tau \deg(v)) X^{(l)}_v + \tau \sum \dots$.
    #[inline(always)]
    pub fn execute(
        state: &mut NodeAssemblyState,
        graph: &BlockSparseSheafGraph,
        x_src: &[f32],
        x_dst: &mut [f32],
        tau: f32,
        stride: usize,
    ) {
        let node = state.node_id;
        let degree = graph.degrees[node] as f32;
        let decay = 1.0f32 - tau * degree;

        let node_off = node * stride;
        let self_stalk = &x_src[node_off..node_off + stride];
        let dst_slice = &mut x_dst[node_off..node_off + stride];

        for k in 0..stride {
            dst_slice[k] = decay * self_stalk[k] + tau * state.acc[k];
        }
    }
}

/// Station 3: Commits the assembled node data to destination buffer.
pub struct EgressStation;

impl EgressStation {
    /// Concludes station execution, returning total Dirichlet energy of the batch.
    #[inline(always)]
    pub fn execute(batch: &[NodeAssemblyState]) -> f32 {
        batch.iter().map(|s| s.local_dirichlet).sum()
    }
}
