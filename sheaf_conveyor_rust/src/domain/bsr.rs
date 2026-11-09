//! Block-Sparse Row (BSR) Graph Representation for Sheaf Diffusion.
//!
//! Maps non-Euclidean Sheaf Laplacians into contiguous memory layouts:
//! - `row_ptr`: `|V| + 1` array indexing edge ranges for each target node.
//! - `col_ind`: `|E|` array containing source neighbor indices.
//! - `rho_data`: Continuous tensor of size `|E| * D * D` containing orthogonal
//!   restriction maps $\rho_{v \trianglelefteq e}^T \rho_{u \trianglelefteq e} \in \mathrm{O}(d)$.

use std::fmt::Debug;

/// Represents a validated Block-Sparse Row (BSR) graph with cellular sheaf restriction maps.
#[derive(Clone, Debug)]
pub struct BlockSparseSheafGraph {
    /// Number of vertices |V| in the base graph.
    pub num_nodes: usize,
    /// Number of directed edges |E| in the base graph.
    pub num_edges: usize,
    /// Stalk dimension d (e.g. 4 for so(4) or 8 for transcriptomic Stiefel manifolds).
    pub stalk_dim: usize,
    /// Neighborhood start pointers of size `num_nodes + 1`.
    pub row_ptr: Vec<usize>,
    /// Source neighbor column indices of size `num_edges`.
    pub col_ind: Vec<usize>,
    /// Contiguous restriction map data: `num_edges * stalk_dim * stalk_dim` elements.
    pub rho_data: Vec<f32>,
    /// Degrees of each vertex: `deg(v) = row_ptr[v+1] - row_ptr[v]`.
    pub degrees: Vec<usize>,
}

impl BlockSparseSheafGraph {
    /// Creates and validates a new Block-Sparse Sheaf Graph.
    ///
    /// # Arguments
    /// * `num_nodes` - Total number of nodes in graph |V|.
    /// * `stalk_dim` - Dimension of each stalk space d.
    /// * `row_ptr` - BSR row offset slice of length `num_nodes + 1`.
    /// * `col_ind` - BSR neighbor index slice of length `num_edges`.
    /// * `rho_data` - Flattened restriction map data of length `num_edges * d * d`.
    pub fn new(
        num_nodes: usize,
        stalk_dim: usize,
        row_ptr: Vec<usize>,
        col_ind: Vec<usize>,
        rho_data: Vec<f32>,
    ) -> Result<Self, String> {
        if row_ptr.len() != num_nodes + 1 {
            return Err(format!(
                "row_ptr length {} does not match num_nodes + 1 ({})",
                row_ptr.len(),
                num_nodes + 1
            ));
        }

        let num_edges = col_ind.len();
        if row_ptr[num_nodes] != num_edges {
            return Err(format!(
                "row_ptr[last] {} does not match num_edges {}",
                row_ptr[num_nodes], num_edges
            ));
        }

        let expected_rho_len = num_edges * stalk_dim * stalk_dim;
        if rho_data.len() != expected_rho_len {
            return Err(format!(
                "rho_data length {} does not match expected {}",
                rho_data.len(),
                expected_rho_len
            ));
        }

        let mut degrees = Vec::with_capacity(num_nodes);
        for v in 0..num_nodes {
            let start = row_ptr[v];
            let end = row_ptr[v + 1];
            if start > end || end > num_edges {
                return Err(format!("Invalid edge bounds for node {}: [{}, {}]", v, start, end));
            }
            degrees.push(end - start);
        }

        Ok(Self {
            num_nodes,
            num_edges,
            stalk_dim,
            row_ptr,
            col_ind,
            rho_data,
            degrees,
        })
    }

    /// Retrieves the restriction map slice for edge `e` with shape `d x d`.
    #[inline(always)]
    pub fn get_rho_slice(&self, edge_idx: usize) -> &[f32] {
        let stride = self.stalk_dim * self.stalk_dim;
        let start = edge_idx * stride;
        &self.rho_data[start..start + stride]
    }
}
