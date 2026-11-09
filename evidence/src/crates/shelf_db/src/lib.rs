//! ==============================================================================
//! LAYER 2: SHELFDB ROOT LIBRARY ROUTER
//! ==============================================================================

pub mod conveyor;
pub mod gpu;
pub mod storage;

pub use conveyor::{
    ContractionStation, ConveyorConfig, ConveyorPipeline, ConveyorRingBuffer, ConveyorSlot,
    DiffusionStation, EgressStation, KernelSlot, NodeAssemblyState, PrefetchStation,
    ShapeEmaEstimator, SlotState, SpecializedContractFn, SpeculationConfig, TensorShapeKey,
};
pub use gpu::GpuSheafSession;
pub use storage::{
    cayley_projection_4x4, compute_orthogonality_residual, BlockSparseSheafGraph, SheafTelemetry,
};
