//! ==============================================================================
//! LAYER 2: SHELFDB STORAGE ROUTER
//! ==============================================================================

pub mod bsr_graph;
pub mod telemetry;

pub use bsr_graph::BlockSparseSheafGraph;
pub use telemetry::{cayley_projection_4x4, compute_orthogonality_residual, SheafTelemetry};
