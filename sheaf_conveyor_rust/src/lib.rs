//! Sheaf Conveyor Engine (SCE).
//!
//! A high-performance cellular sheaf diffusion engine unifying:
//! 1. Block-Sparse Row (BSR) graph representations and orthogonal restriction maps.
//! 2. Compile-time Rust const-generic metaprogramming for unrolled in-register contractions.
//! 3. The Conveyor Belt Architecture featuring circular multi-stage ring buffering
//!    and decoupled producer-consumer streaming.
//! 4. Tiered Specialization Dispatching (Tier 1 AOT hot-grid + Tier 0 dynamic fallback).

pub mod conveyor;
pub mod domain;
pub mod kernel;
pub mod metaprogramming;

// Re-export primary domain types and baseline kernel operations
pub use domain::{
    cayley_projection_4x4, compute_orthogonality_residual, BlockSparseSheafGraph, SheafTelemetry,
};
pub use kernel::{
    contract_edge_dynamic, contract_edge_stalk, fused_sheaf_diffusion_backward_step,
    fused_sheaf_diffusion_forward, SheafGradients,
};
pub use metaprogramming::ConstSheafEngine;

// Re-export conveyor belt & TAAS components
pub use conveyor::{
    AsyncJitWorker, BlockGeometry, ConveyorConfig, ConveyorDispatcher, ConveyorPipeline,
    ConveyorRingBuffer, ConveyorSlot, ConveyorTier, GpuSheafSession, JitTelemetry, KernelAst,
    KernelSlot, ShapeEmaEstimator, SharedMemDecl, SlotState, SpeculationConfig, TensorShapeKey,
    TkType, TkValidator,
};
