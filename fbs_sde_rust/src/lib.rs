//! Fused Block-Sparse Sheaf Diffusion Engine (FBS-SDE).
//!
//! A high-performance implementation of cellular sheaf diffusion on graphs,
//! featuring Block-Sparse Row (BSR) layouts, in-register dense matrix contractions,
//! in-kernel Dirichlet energy telemetry, exact backward adjoint gradients,
//! and compile-time const generic metaprogramming.

pub mod domain;
pub mod kernel;
pub mod metaprogramming;

// Re-export primary domain types and kernel operations
pub use domain::{
    cayley_projection_4x4, compute_orthogonality_residual, BlockSparseSheafGraph, SheafTelemetry,
};
pub use kernel::{
    contract_edge_dynamic, contract_edge_stalk, fused_sheaf_diffusion_backward_step,
    fused_sheaf_diffusion_forward, SheafGradients,
};
pub use metaprogramming::ConstSheafEngine;
