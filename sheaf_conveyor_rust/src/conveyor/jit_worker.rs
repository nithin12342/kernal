//! Asynchronous Speculative JIT Compilation Worker & Content-Addressed Cache.
//!
//! Implements Section 2 and Section 4 of the TAAS specification:
//! Offloads runtime specialization to an asynchronous background worker thread.
//! Compiles validated TKIR ASTs and atomically hot-swaps them into active conveyor slots.

use crate::conveyor::hot_swap::{KernelSlot, SpecializedContractFn};
use crate::conveyor::speculation::TensorShapeKey;
use crate::conveyor::stages::{ContractionStation, NodeAssemblyState};
use crate::conveyor::tkir::{BlockGeometry, KernelAst, TkType, TkValidator};
use crate::domain::bsr::BlockSparseSheafGraph;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Task dispatched to the background speculative compilation worker.
pub struct CompileTask {
    pub shape: TensorShapeKey,
    pub target_slot: Arc<KernelSlot>,
}

/// Telemetry metrics for runtime speculative compilation.
#[derive(Default, Debug)]
pub struct JitTelemetry {
    pub tasks_enqueued: AtomicU32,
    pub compilations_completed: AtomicU32,
    pub compilations_failed: AtomicU32,
    pub cache_hits: AtomicU32,
    pub total_compile_time_us: AtomicU64,
}

/// Speculative JIT Worker Engine.
pub struct AsyncJitWorker {
    sender: Sender<CompileTask>,
    _worker_thread: Option<JoinHandle<()>>,
    telemetry: Arc<JitTelemetry>,
    cache: Arc<Mutex<HashMap<String, SpecializedContractFn>>>,
}

impl AsyncJitWorker {
    /// Spawns the background compilation worker thread.
    pub fn new() -> Self {
        let (sender, receiver) = channel::<CompileTask>();
        let telemetry = Arc::new(JitTelemetry::default());
        let cache = Arc::new(Mutex::new(HashMap::<String, SpecializedContractFn>::new()));

        let tel_clone = Arc::clone(&telemetry);
        let cache_clone = Arc::clone(&cache);

        let worker_thread = thread::spawn(move || {
            while let Ok(task) = receiver.recv() {
                let start = std::time::Instant::now();
                let shape = task.shape;

                // 1. Construct candidate TKIR AST
                let mut ast = KernelAst {
                    stalk_dim: shape.stalk_dim,
                    channels: shape.channels,
                    precision: TkType::F32,
                    block_geometry: BlockGeometry::auto_tune(shape.stalk_dim),
                    shared_memory_tiles: vec![],
                    unroll_factor: shape.stalk_dim,
                    predicted_registers_per_thread: 32,
                };

                // 2. Perform static formal verification checks
                if let Err(_) = TkValidator::validate(&mut ast) {
                    tel_clone.compilations_failed.fetch_add(1, Ordering::Relaxed);
                    continue; // Invariant check failed: leave Tier 0 in place
                }

                let digest = ast.content_digest();

                // 3. Check Content-Addressed Cache
                let mut cached_fn: Option<SpecializedContractFn> = None;
                {
                    let lock = cache_clone.lock().unwrap();
                    if let Some(&func) = lock.get(&digest) {
                        tel_clone.cache_hits.fetch_add(1, Ordering::Relaxed);
                        cached_fn = Some(func);
                    }
                }

                let specialized_fn = if let Some(func) = cached_fn {
                    func
                } else {
                    // 4. Synthesize specialized kernel implementation
                    let compiled_fn = synthesize_specialized_kernel(shape.stalk_dim, shape.channels);
                    let mut lock = cache_clone.lock().unwrap();
                    lock.insert(digest, compiled_fn);
                    compiled_fn
                };

                // 5. Atomic Hot-Swap on the moving conveyor belt (Tier 2 JIT)
                task.target_slot.swap_specialized(specialized_fn, 2);

                let duration = start.elapsed().as_micros() as u64;
                tel_clone.compilations_completed.fetch_add(1, Ordering::Relaxed);
                tel_clone.total_compile_time_us.fetch_add(duration, Ordering::Relaxed);
            }
        });

        Self {
            sender,
            _worker_thread: Some(worker_thread),
            telemetry,
            cache,
        }
    }

    /// Enqueues an asynchronous compilation task for a recurring shape.
    pub fn enqueue_compilation(&self, shape: TensorShapeKey, target_slot: Arc<KernelSlot>) {
        self.telemetry.tasks_enqueued.fetch_add(1, Ordering::Relaxed);
        let _ = self.sender.send(CompileTask { shape, target_slot });
    }

    /// Telemetry accessor.
    pub fn telemetry(&self) -> &JitTelemetry {
        &self.telemetry
    }

    /// Number of compiled kernels currently held in the content-addressed cache.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

/// Specialized function for (D=4, C=32) const-generic unrolling.
fn const_contract_4x32(
    state: &mut NodeAssemblyState,
    graph: &BlockSparseSheafGraph,
    x_src: &[f32],
) {
    ContractionStation::execute_const::<4, 32>(state, graph, x_src);
}

/// Specialized function for (D=2, C=16) const-generic unrolling.
fn const_contract_2x16(
    state: &mut NodeAssemblyState,
    graph: &BlockSparseSheafGraph,
    x_src: &[f32],
) {
    ContractionStation::execute_const::<2, 16>(state, graph, x_src);
}

/// Specialized function for (D=8, C=32) const-generic unrolling.
fn const_contract_8x32(
    state: &mut NodeAssemblyState,
    graph: &BlockSparseSheafGraph,
    x_src: &[f32],
) {
    ContractionStation::execute_const::<8, 32>(state, graph, x_src);
}

/// Dynamic contraction wrapper for arbitrary dimensions without capturing state.
fn dynamic_contract_wrapper(
    state: &mut NodeAssemblyState,
    graph: &BlockSparseSheafGraph,
    x_src: &[f32],
) {
    let d = graph.stalk_dim;
    let channels = state.acc.len() / d;
    ContractionStation::execute_dynamic(state, graph, x_src, d, channels);
}

/// Synthesizes a specialized kernel function pointer tailored to (D, C).
fn synthesize_specialized_kernel(d: usize, channels: usize) -> SpecializedContractFn {
    match (d, channels) {
        (4, 32) => const_contract_4x32,
        (2, 16) => const_contract_2x16,
        (8, 32) => const_contract_8x32,
        _ => dynamic_contract_wrapper,
    }
}
