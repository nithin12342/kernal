//! Kernel Layer for Fused Block-Sparse Sheaf Diffusion.
//!
//! Exposes in-register contraction primitives, forward multi-hop diffusion,
//! and exact adjoint backward gradient solvers.

pub mod backward;
pub mod contraction;
pub mod forward;

pub use backward::{fused_sheaf_diffusion_backward_step, SheafGradients};
pub use contraction::{contract_edge_dynamic, contract_edge_stalk};
pub use forward::fused_sheaf_diffusion_forward;
