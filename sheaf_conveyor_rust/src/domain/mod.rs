//! Domain Layer for Sheaf Diffusion.
//!
//! Exposes Block-Sparse Row graph representations and in-kernel telemetry monitors.

pub mod bsr;
pub mod telemetry;

pub use bsr::BlockSparseSheafGraph;
pub use telemetry::{cayley_projection_4x4, compute_orthogonality_residual, SheafTelemetry};
