//! Integration & Invariant Tests for Sheaf Conveyor Engine.
//!
//! Verifies:
//! 1. ConveyorRingBuffer invariants (monotonic counters, wrapping, no-overflow).
//! 2. Numerical parity between ConveyorPipeline (dynamic & const) and fused_sheaf_diffusion_forward.
//! 3. Tiered ConveyorDispatcher routing logic.

use sheaf_conveyor_rust::{
    fused_sheaf_diffusion_forward, BlockSparseSheafGraph, ConveyorConfig, ConveyorDispatcher,
    ConveyorPipeline, ConveyorRingBuffer, ConveyorTier, SlotState,
};

fn create_4node_sheaf_graph_d4() -> BlockSparseSheafGraph {
    // 4-node directed cycle: 0 -> 1 -> 2 -> 3 -> 0
    let row_ptr = vec![0, 1, 2, 3, 4];
    let col_ind = vec![1, 2, 3, 0];

    // Orthogonal identity 4x4 matrices for each edge
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

fn create_4node_sheaf_graph_d2() -> BlockSparseSheafGraph {
    let row_ptr = vec![0, 1, 2, 3, 4];
    let col_ind = vec![1, 2, 3, 0];

    // Orthogonal 2x2 90-degree rotation matrices: [[0, -1], [1, 0]]
    let mut rho_data = Vec::with_capacity(4 * 4);
    for _ in 0..4 {
        rho_data.extend_from_slice(&[0.0, -1.0, 1.0, 0.0]);
    }

    BlockSparseSheafGraph::new(4, 2, row_ptr, col_ind, rho_data).unwrap()
}

#[test]
fn test_ring_buffer_invariants() {
    let mut buffer = ConveyorRingBuffer::<f32>::new(4, 32);
    assert_eq!(buffer.stages(), 4);
    assert_eq!(buffer.produced_count(), 0);
    assert_eq!(buffer.consumed_count(), 0);
    assert!(buffer.can_produce());
    assert!(!buffer.can_consume());

    // Fill all 4 stages
    for i in 0..4 {
        assert!(buffer.can_produce());
        let (slot_idx, slot) = buffer.acquire_producer_slot().unwrap();
        assert_eq!(slot_idx, i);
        slot.node_indices.push(i);
        buffer.commit_produced(slot_idx);
    }

    // Now buffer is full: can_produce must be false
    assert!(!buffer.can_produce());
    assert!(buffer.acquire_producer_slot().is_none());
    assert!(buffer.can_consume());
    assert_eq!(buffer.produced_count(), 4);

    // Consume 2 stages
    for i in 0..2 {
        assert!(buffer.can_consume());
        let (slot_idx, slot) = buffer.acquire_consumer_slot().unwrap();
        assert_eq!(slot_idx, i);
        assert_eq!(slot.state, SlotState::InProcess);
        buffer.commit_consumed(slot_idx);
    }
    assert_eq!(buffer.consumed_count(), 2);

    // Now buffer has 2 empty slots
    assert!(buffer.can_produce());
    let (slot_idx, _) = buffer.acquire_producer_slot().unwrap();
    assert_eq!(slot_idx, 0); // Wraps around to 0
    buffer.commit_produced(slot_idx);
    assert_eq!(buffer.produced_count(), 5);
}

#[test]
fn test_conveyor_dynamic_numerical_parity() {
    let graph = create_4node_sheaf_graph_d4();
    let channels = 32;
    let total_elements = 4 * 4 * channels;

    // Pseudo-random deterministic initial stalk features
    let mut x_initial = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        x_initial.push((i as f32 * 0.137).sin());
    }

    let tau = 0.25f32;
    let num_layers = 5;

    // Baseline execution
    let (baseline_out, baseline_tel) =
        fused_sheaf_diffusion_forward(&graph, &x_initial, channels, tau, num_layers);

    // Conveyor Belt execution (chunk_size = 2, stages = 4)
    let pipeline = ConveyorPipeline::new(ConveyorConfig {
        num_stages: 4,
        chunk_size: 2,
    });
    let (conveyor_out, conveyor_tel) =
        pipeline.execute_dynamic(&graph, &x_initial, channels, tau, num_layers);

    assert_eq!(baseline_out.len(), conveyor_out.len());
    for idx in 0..total_elements {
        let diff = (baseline_out[idx] - conveyor_out[idx]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at element {}: baseline={}, conveyor={}",
            idx,
            baseline_out[idx],
            conveyor_out[idx]
        );
    }

    assert_eq!(baseline_tel.len(), conveyor_tel.len());
    for l in 0..num_layers {
        let e_diff = (baseline_tel[l].dirichlet_energy - conveyor_tel[l].dirichlet_energy).abs();
        assert!(
            e_diff < 1e-4,
            "Dirichlet energy mismatch at layer {}: baseline={}, conveyor={}",
            l,
            baseline_tel[l].dirichlet_energy,
            conveyor_tel[l].dirichlet_energy
        );
    }
}

#[test]
fn test_conveyor_const_generic_numerical_parity_d4_c32() {
    let graph = create_4node_sheaf_graph_d4();
    let total_elements = 4 * 4 * 32;

    let mut x_initial = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        x_initial.push((i as f32 * 0.071).cos());
    }

    let tau = 0.2f32;
    let num_layers = 4;

    let (baseline_out, _) = fused_sheaf_diffusion_forward(&graph, &x_initial, 32, tau, num_layers);

    let pipeline = ConveyorPipeline::new(ConveyorConfig {
        num_stages: 4,
        chunk_size: 2,
    });
    let (const_out, _) = pipeline.execute_const::<4, 32>(&graph, &x_initial, tau, num_layers);

    for idx in 0..total_elements {
        let diff = (baseline_out[idx] - const_out[idx]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at element {}: baseline={}, const={}",
            idx,
            baseline_out[idx],
            const_out[idx]
        );
    }
}

#[test]
fn test_conveyor_const_generic_numerical_parity_d2_c16() {
    let graph = create_4node_sheaf_graph_d2();
    let total_elements = 4 * 2 * 16;

    let mut x_initial = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        x_initial.push((i as f32 * 0.033).sin());
    }

    let tau = 0.3f32;
    let num_layers = 3;

    let (baseline_out, _) = fused_sheaf_diffusion_forward(&graph, &x_initial, 16, tau, num_layers);

    let pipeline = ConveyorPipeline::new(ConveyorConfig {
        num_stages: 3,
        chunk_size: 1,
    });
    let (const_out, _) = pipeline.execute_const::<2, 16>(&graph, &x_initial, tau, num_layers);

    for idx in 0..total_elements {
        let diff = (baseline_out[idx] - const_out[idx]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at element {}: baseline={}, const={}",
            idx,
            baseline_out[idx],
            const_out[idx]
        );
    }
}

#[test]
fn test_conveyor_dispatcher_routing() {
    let dispatcher = ConveyorDispatcher::default();

    // 1. D=4, C=32 -> should route to Tier 1 AOT
    let graph_d4 = create_4node_sheaf_graph_d4();
    let x_d4 = vec![1.0f32; 4 * 4 * 32];
    let (_, _, tier_d4) = dispatcher.dispatch(&graph_d4, &x_d4, 32, 0.2, 2);
    assert_eq!(
        tier_d4,
        ConveyorTier::Tier1Aot {
            stalk_dim: 4,
            channels: 32
        }
    );

    // 2. D=2, C=16 -> should route to Tier 1 AOT
    let graph_d2 = create_4node_sheaf_graph_d2();
    let x_d2 = vec![1.0f32; 4 * 2 * 16];
    let (_, _, tier_d2) = dispatcher.dispatch(&graph_d2, &x_d2, 16, 0.2, 2);
    assert_eq!(
        tier_d2,
        ConveyorTier::Tier1Aot {
            stalk_dim: 2,
            channels: 16
        }
    );

    // 3. D=4, C=17 (arbitrary/unseen channels) -> should route to Tier 0 Dynamic
    let x_d4_c17 = vec![1.0f32; 4 * 4 * 17];
    let (_, _, tier_dyn) = dispatcher.dispatch(&graph_d4, &x_d4_c17, 17, 0.2, 2);
    assert_eq!(
        tier_dyn,
        ConveyorTier::Tier0Dynamic {
            stalk_dim: 4,
            channels: 17
        }
    );
}
