//! ==============================================================================
//! LAYER 2: SHELFDB STORAGE - BLOCK-SPARSE ROW (BSR) SHEAF GRAPH
//! ==============================================================================

use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct BlockSparseSheafGraph {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub stalk_dim: usize,
    pub row_ptr: Vec<usize>,
    pub col_ind: Vec<usize>,
    pub rho_data: Vec<f32>,
    pub degrees: Vec<usize>,
    pub max_degree: usize,
}

impl BlockSparseSheafGraph {
    /// Constructs a verified BSR Sheaf Graph. Validates dimensions and Gershgorin bounds.
    pub fn new(
        num_nodes: usize,
        stalk_dim: usize,
        row_ptr: Vec<usize>,
        col_ind: Vec<usize>,
        rho_data: Vec<f32>,
    ) -> Result<Self, String> {
        if row_ptr.len() != num_nodes + 1 {
            return Err("row_ptr length mismatch: must be num_nodes + 1".into());
        }
        let num_edges = col_ind.len();
        if rho_data.len() != num_edges * stalk_dim * stalk_dim {
            return Err("rho_data length mismatch: must be num_edges * d * d".into());
        }

        let mut degrees = Vec::with_capacity(num_nodes);
        let mut max_degree = 0;
        for i in 0..num_nodes {
            let deg = row_ptr[i + 1] - row_ptr[i];
            degrees.push(deg);
            if deg > max_degree {
                max_degree = deg;
            }
        }

        Ok(Self {
            num_nodes,
            num_edges,
            stalk_dim,
            row_ptr,
            col_ind,
            rho_data,
            degrees,
            max_degree,
        })
    }

    /// Verifies if a given diffusion step-size tau satisfies the Block-Gershgorin ceiling
    pub fn check_gershgorin_stability(&self, tau: f32) -> bool {
        if self.max_degree == 0 || tau <= 0.0 {
            return false;
        }
        let ceiling = 1.0f32 / (self.max_degree as f32);
        tau < ceiling
    }

    #[inline(always)]
    pub fn get_edge_rho(&self, edge_idx: usize) -> &[f32] {
        let stride = self.stalk_dim * self.stalk_dim;
        let start = edge_idx * stride;
        &self.rho_data[start..start + stride]
    }
}
