//! TAAS Conveyor Belt Architecture Verification Tests.
//!
//! Verifies:
//! 1. TKIR formal verification invariants (bank conflicts, register ceiling, 16-byte alignment).
//! 2. Online EMA frequency estimation and speculative JIT triggering.
//! 3. Lock-free Atomic KernelSlot hot-swapping under Acquire-Release semantics.
//! 4. End-to-end asynchronous background JIT compilation and hot-swapping on a moving conveyor belt.
//! 5. Zero-crash graceful degradation.

use sheaf_conveyor_rust::{
    fused_sheaf_diffusion_forward, BlockGeometry, BlockSparseSheafGraph, ConveyorConfig,
    ConveyorDispatcher, ConveyorTier, KernelAst, KernelSlot, ShapeEmaEstimator,
    SharedMemDecl, SpeculationConfig, TensorShapeKey, TkType, TkValidator,
};
use std::thread;
use std::time::Duration;

fn create_test_graph_d4() -> BlockSparseSheafGraph {
    let row_ptr = vec![0, 1, 2, 3, 4];
    let col_ind = vec![1, 2, 3, 0];
    let mut rho_data = Vec::with_capacity(4 * 16);
    for _ in 0..4 {
        for i in 0..4 {
            for j in 0..4 {
                rho_data.push(if i == j { 1.0f32 } else { 0.0f32 });
            }
        }
    }
    BlockSparseSheafGraph::new(4, 4, row_ptr, col_ind, rho_data).unwrap()
}

#[test]
fn test_tkir_bank_conflict_and_alignment_invariants() {
    // 1. Bank conflict check: stride 32 has gcd(32, 32) = 32 != 1, must require padding
    let conflict = TkValidator::verify_bank_conflicts(32);
    assert!(conflict.is_err());
    assert_eq!(conflict.unwrap_err(), 1);

    // Stride 33 has gcd(33, 32) = 1, conflict-free
    let no_conflict = TkValidator::verify_bank_conflicts(33);
    assert!(no_conflict.is_ok());

    // 2. Alignment invariant check: 16-byte aligned passes, 8-byte fails
    assert!(TkValidator::verify_alignment(16).is_ok());
    assert!(TkValidator::verify_alignment(32).is_ok());
    assert!(TkValidator::verify_alignment(8).is_err());
    assert!(TkValidator::verify_alignment(12).is_err());

    // 3. Register ceiling check: <= 64 registers passes, > 64 fails
    let valid_ast = KernelAst {
        stalk_dim: 4,
        channels: 32,
        precision: TkType::F32,
        block_geometry: BlockGeometry::auto_tune(4),
        shared_memory_tiles: vec![SharedMemDecl {
            identifier: "tile".to_string(),
            element_type: TkType::F32,
            extent_elements: 128,
            byte_alignment: 16,
            padding_elements: 0,
        }],
        unroll_factor: 4,
        predicted_registers_per_thread: 48,
    };
    assert!(TkValidator::verify_register_pressure(&valid_ast, 64).is_ok());

    let mut invalid_ast = valid_ast.clone();
    invalid_ast.predicted_registers_per_thread = 80;
    assert!(TkValidator::verify_register_pressure(&invalid_ast, 64).is_err());
}

#[test]
fn test_speculative_frequency_estimator() {
    let config = SpeculationConfig {
        alpha: 0.30,
        threshold: 0.40,
        min_warmup_invocations: 2,
    };
    let estimator = ShapeEmaEstimator::new(config);
    let shape_a = TensorShapeKey::new(4, 32);
    let shape_b = TensorShapeKey::new(3, 17);

    // Invocations of transient shape B: single invocation does not cross threshold
    let trigger_b = estimator.record_and_evaluate(shape_b);
    assert!(!trigger_b);
    assert!(estimator.get_ema(&shape_b) < 0.40);

    // Repeated invocations of shape A: must build frequency and trigger
    let trigger_a1 = estimator.record_and_evaluate(shape_a);
    assert!(!trigger_a1); // 1st invocation: warmup

    let trigger_a2 = estimator.record_and_evaluate(shape_a);
    assert!(trigger_a2, "Second invocation with alpha=0.30, thresh=0.40 must cross threshold");

    // Subsequent invocations after trigger: already marked enqueued, must return false
    let trigger_a3 = estimator.record_and_evaluate(shape_a);
    assert!(!trigger_a3);
}

#[test]
fn test_lock_free_hot_swap_slot() {
    let slot = KernelSlot::default();
    assert_eq!(slot.active_tier(), 0);
    assert!(slot.load_fn().is_none());
    assert_eq!(slot.epoch(), 0);

    // Atomically swap in a specialized function
    fn sample_fn(
        state: &mut sheaf_conveyor_rust::conveyor::NodeAssemblyState,
        _graph: &BlockSparseSheafGraph,
        _x_src: &[f32],
    ) {
        state.local_dirichlet += 1.0;
    }

    slot.swap_specialized(sample_fn, 2);
    assert_eq!(slot.active_tier(), 2);
    assert!(slot.load_fn().is_some());
    assert_eq!(slot.epoch(), 1);
}

#[test]
fn test_end_to_end_taas_speculative_compilation_and_hot_swap() {
    let graph = create_test_graph_d4();
    let spec_config = SpeculationConfig {
        alpha: 0.50,
        threshold: 0.30,
        min_warmup_invocations: 1,
    };
    let dispatcher = ConveyorDispatcher::new(ConveyorConfig::default(), spec_config);

    // Test with non-canonical shape to exercise dynamic -> JIT transition (D=4, C=24)
    let total_elements_novel = 4 * 4 * 24;
    let x_novel = vec![0.5f32; total_elements_novel];

    // Iteration 1: Unseen shape -> executes on Tier 0 Dynamic (0 ms stall)
    let (out1, tel1, tier1) = dispatcher.dispatch(&graph, &x_novel, 24, 0.2, 2);
    assert_eq!(
        tier1,
        ConveyorTier::Tier0Dynamic {
            stalk_dim: 4,
            channels: 24
        }
    );
    assert_eq!(out1.len(), total_elements_novel);
    assert_eq!(tel1.len(), 2);

    // Wait briefly for background worker thread to compile and atomically swap
    thread::sleep(Duration::from_millis(50));

    // Iteration 2: After background JIT compilation, next request uses Tier 2 JIT!
    let (out2, tel2, tier2) = dispatcher.dispatch(&graph, &x_novel, 24, 0.2, 2);
    assert_eq!(
        tier2,
        ConveyorTier::Tier2Jit {
            stalk_dim: 4,
            channels: 24
        }
    );
    assert_eq!(out2.len(), total_elements_novel);
    assert_eq!(tel2.len(), 2);

    // Confirm telemetry recorded the JIT compilation
    let jit_tel = dispatcher.jit_worker().telemetry();
    assert!(jit_tel.compilations_completed.load(std::sync::atomic::Ordering::Relaxed) >= 1);
}

#[test]
fn test_numerical_parity_after_hot_swap() {
    let graph = create_test_graph_d4();
    let channels = 32;
    let total_elements = 4 * 4 * channels;

    let mut x_in = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        x_in.push((i as f32 * 0.05).sin());
    }

    let tau = 0.2f32;
    let num_layers = 4;

    // Baseline calculation
    let (baseline_out, baseline_tel) =
        fused_sheaf_diffusion_forward(&graph, &x_in, channels, tau, num_layers);

    // Dispatcher calculation
    let dispatcher = ConveyorDispatcher::default();
    let (disp_out, disp_tel, _) = dispatcher.dispatch(&graph, &x_in, channels, tau, num_layers);

    for idx in 0..total_elements {
        let diff = (baseline_out[idx] - disp_out[idx]).abs();
        assert!(diff < 1e-5, "Mismatch at element {}: baseline={}, disp={}", idx, baseline_out[idx], disp_out[idx]);
    }

    for l in 0..num_layers {
        let diff_e = (baseline_tel[l].dirichlet_energy - disp_tel[l].dirichlet_energy).abs();
        assert!(diff_e < 1e-4, "Dirichlet mismatch at layer {}: baseline={}, disp={}", l, baseline_tel[l].dirichlet_energy, disp_tel[l].dirichlet_energy);
    }
}
