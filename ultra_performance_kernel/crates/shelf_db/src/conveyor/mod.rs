//! ==============================================================================
//! LAYER 2: SHELFDB CONVEYOR ROUTER
//! ==============================================================================

pub mod hot_swap;
pub mod pipeline;
pub mod ring_buffer;
pub mod speculation;
pub mod stages;

pub use hot_swap::{KernelSlot, SpecializedContractFn};
pub use pipeline::{ConveyorConfig, ConveyorPipeline};
pub use ring_buffer::{ConveyorRingBuffer, ConveyorSlot, SlotState};
pub use speculation::{ShapeEmaEstimator, SpeculationConfig, TensorShapeKey};
pub use stages::{
    ContractionStation, DiffusionStation, EgressStation, NodeAssemblyState, PrefetchStation,
};
