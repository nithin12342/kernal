//! Conveyor Belt Pipelining Module for Cellular Sheaf Diffusion.
//!
//! Exposes:
//! - Circular multi-stage ring buffer (`ConveyorRingBuffer`)
//! - Assembly line stations (`PrefetchStation`, `ContractionStation`, `DiffusionStation`, `EgressStation`)
//! - Decoupled execution pipeline (`ConveyorPipeline`)
//! - Tiered Specialization Dispatcher (`ConveyorDispatcher`, `ConveyorTier`)
//! - Lock-free atomic hot-swapping (`KernelSlot`, `SpecializedContractFn`)
//! - Online speculative frequency tracking (`ShapeEmaEstimator`, `SpeculationConfig`, `TensorShapeKey`)
//! - Typed Kernel Intermediate Representation AST (`KernelAst`, `TkValidator`, `TkType`, `BlockGeometry`)
//! - Asynchronous speculative JIT compilation worker (`AsyncJitWorker`, `JitTelemetry`)

pub mod dispatcher;
pub mod gpu_backend;
pub mod hot_swap;
pub mod jit_worker;
pub mod pipeline;
pub mod ring_buffer;
pub mod speculation;
pub mod stages;
pub mod tkir;

pub use dispatcher::{ConveyorDispatcher, ConveyorTier};
pub use gpu_backend::GpuSheafSession;
pub use hot_swap::{KernelSlot, SpecializedContractFn};
pub use jit_worker::{AsyncJitWorker, JitTelemetry};
pub use pipeline::{ConveyorConfig, ConveyorPipeline};
pub use ring_buffer::{ConveyorRingBuffer, ConveyorSlot, SlotState};
pub use speculation::{ShapeEmaEstimator, SpeculationConfig, TensorShapeKey};
pub use stages::{
    ContractionStation, DiffusionStation, EgressStation, NodeAssemblyState, PrefetchStation,
};
pub use tkir::{BlockGeometry, KernelAst, SharedMemDecl, TkType, TkValidator};
