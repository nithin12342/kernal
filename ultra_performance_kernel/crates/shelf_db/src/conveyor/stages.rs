//! ==============================================================================
//! LAYER 2: SHELFDB CONVEYOR - ASSEMBLY LINE STATIONS
//! ==============================================================================

use crate::storage::BlockSparseSheafGraph;

#[derive(Clone, Debug)]
pub struct NodeAssemblyState {
    pub node_id: usize,
    pub acc: Vec<f32>,
    pub local_dirichlet: f32,
}

impl NodeAssemblyState {
    pub fn new(node_id: usize, stride: usize) -> Self {
        Self {
            node_id,
            acc: vec![0.0f32; stride],
            local_dirichlet: 0.0f32,
        }
    }
}

/// Stage 1: Prefetch Station initializes node assembly workspaces
pub struct PrefetchStation;

impl PrefetchStation {
    #[inline(always)]
    pub fn execute(node_ids: &[usize], stride: usize) -> Vec<NodeAssemblyState> {
        node_ids.iter().map(|&id| NodeAssemblyState::new(id, stride)).collect()
    }
}

/// Stage 2: Contraction Station executes in-register restriction matrix multiplication
pub struct ContractionStation;

impl ContractionStation {
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

            for i in 0..d {
                let row = i * d;
                let out_off = i * channels;
                for j in 0..d {
                    let r = rho[row + j];
                    let in_off = j * channels;
                    for ch in 0..channels {
                        state.acc[out_off + ch] += r * neighbor_stalk[in_off + ch];
                    }
                }
            }

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
}

/// Stage 3: Diffusion Station computes decay and adds accumulated neighboring messages
pub struct DiffusionStation;

impl DiffusionStation {
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

/// Stage 4: Egress Station aggregates Dirichlet energy across chunks
pub struct EgressStation;

impl EgressStation {
    #[inline(always)]
    pub fn execute(batch: &[NodeAssemblyState]) -> f32 {
        batch.iter().map(|s| s.local_dirichlet).sum()
    }
}
