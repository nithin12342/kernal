//! Tiered Specialization Dispatcher for Conveyor Belt Sheaf Diffusion.
//!
//! Full implementation of the TAAS (Tiered Asynchronous Adaptive Specialization) architecture:
//! - Tier 1: Canonical Const-Generic AOT specialized kernel.
//! - Tier 2: Speculatively JIT-compiled specialized kernel (atomically hot-swapped).
//! - Tier 0: Generalized dynamic conveyor fallback (zero stall).
//!
//! Features online EMA frequency tracking and non-blocking background JIT worker orchestration.

use crate::domain::bsr::BlockSparseSheafGraph;
use crate::domain::telemetry::SheafTelemetry;
use crate::conveyor::hot_swap::KernelSlot;
use crate::conveyor::jit_worker::AsyncJitWorker;
use crate::conveyor::pipeline::{ConveyorConfig, ConveyorPipeline};
use crate::conveyor::speculation::{ShapeEmaEstimator, SpeculationConfig, TensorShapeKey};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Execution tier selected by the conveyor dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConveyorTier {
    /// Tier 1: Canonical Const-Generic AOT specialized kernel.
    Tier1Aot { stalk_dim: usize, channels: usize },
    /// Tier 2: Speculatively compiled JIT kernel (hot-swapped into conveyor slot).
    Tier2Jit { stalk_dim: usize, channels: usize },
    /// Tier 0: Generalized dynamic conveyor fallback.
    Tier0Dynamic { stalk_dim: usize, channels: usize },
}

/// The Conveyor Dispatcher managing tiered specialization routing and speculative JIT.
pub struct ConveyorDispatcher {
    pipeline: ConveyorPipeline,
    speculation_guard: ShapeEmaEstimator,
    jit_worker: Arc<AsyncJitWorker>,
    slots: Mutex<HashMap<TensorShapeKey, Arc<KernelSlot>>>,
}

impl Default for ConveyorDispatcher {
    fn default() -> Self {
        Self::new(ConveyorConfig::default(), SpeculationConfig::default())
    }
}

impl ConveyorDispatcher {
    /// Creates a new dispatcher with conveyor and speculation configuration.
    pub fn new(config: ConveyorConfig, speculation_config: SpeculationConfig) -> Self {
        Self {
            pipeline: ConveyorPipeline::new(config),
            speculation_guard: ShapeEmaEstimator::new(speculation_config),
            jit_worker: Arc::new(AsyncJitWorker::new()),
            slots: Mutex::new(HashMap::new()),
        }
    }

    /// Dispatches a sheaf diffusion request to the optimal conveyor specialization tier.
    pub fn dispatch(
        &self,
        graph: &BlockSparseSheafGraph,
        x_initial: &[f32],
        channels: usize,
        tau: f32,
        num_layers: usize,
    ) -> (Vec<f32>, Vec<SheafTelemetry>, ConveyorTier) {
        let d = graph.stalk_dim;
        let shape = TensorShapeKey::new(d, channels);

        // Tier 1 Fast-Path: Monomorphized const-generic hot-grid
        match (d, channels) {
            (4, 32) => {
                let (res, tel) = self.pipeline.execute_const::<4, 32>(
                    graph,
                    x_initial,
                    tau,
                    num_layers,
                );
                return (res, tel, ConveyorTier::Tier1Aot { stalk_dim: 4, channels: 32 });
            }
            (2, 16) => {
                let (res, tel) = self.pipeline.execute_const::<2, 16>(
                    graph,
                    x_initial,
                    tau,
                    num_layers,
                );
                return (res, tel, ConveyorTier::Tier1Aot { stalk_dim: 2, channels: 16 });
            }
            (8, 32) => {
                let (res, tel) = self.pipeline.execute_const::<8, 32>(
                    graph,
                    x_initial,
                    tau,
                    num_layers,
                );
                return (res, tel, ConveyorTier::Tier1Aot { stalk_dim: 8, channels: 32 });
            }
            _ => {}
        }

        // Get or initialize the execution slot for this non-canonical shape
        let slot = {
            let mut lock = self.slots.lock().unwrap();
            Arc::clone(lock.entry(shape).or_insert_with(|| Arc::new(KernelSlot::default())))
        };

        // Tier 2: Check if slot has been hot-swapped by background JIT worker
        if slot.active_tier() == 2 {
            let (res, tel) = self.pipeline.execute_slot(
                &slot,
                graph,
                x_initial,
                channels,
                tau,
                num_layers,
            );
            return (res, tel, ConveyorTier::Tier2Jit { stalk_dim: d, channels });
        }

        // Tier 0: Immediate dynamic conveyor execution (Zero stall!)
        let (res, tel) = self.pipeline.execute_dynamic(
            graph,
            x_initial,
            channels,
            tau,
            num_layers,
        );

        // Online Speculation Guard: Check if recurring shape should trigger async JIT
        if self.speculation_guard.record_and_evaluate(shape) {
            self.jit_worker.enqueue_compilation(shape, Arc::clone(&slot));
        }

        (res, tel, ConveyorTier::Tier0Dynamic { stalk_dim: d, channels })
    }

    /// Accessor for the background JIT worker.
    pub fn jit_worker(&self) -> &AsyncJitWorker {
        &self.jit_worker
    }

    /// Accessor for the online speculative frequency estimator.
    pub fn speculation_guard(&self) -> &ShapeEmaEstimator {
        &self.speculation_guard
    }
}
