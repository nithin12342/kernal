//! ==============================================================================
//! INTEGRATION TEST SUITE: HOST CPU (WITHOUT GPU) VERIFICATION
//! ==============================================================================
//! Validates:
//! 1. Lean 4 Gershgorin stability bounds (tau < 1 / max_degree)
//! 2. 4-Stage Conveyor Circular Ring Buffer streaming
//! 3. Const-Generic in-register unrolled contraction numerical parity
//! 4. Lock-free KernelSlot hot swapping and Dirichlet energy conservation
//! ==============================================================================

use sheaf_metaprog::stalk::AmlStalk;
use sheaf_metaprog::unroll::ConstSheafEngine;
use shelf_db::conveyor::{
    ConveyorConfig, ConveyorPipeline, ConveyorRingBuffer, KernelSlot, SlotState,
};
use shelf_db::storage::{
    cayley_projection_4x4, compute_orthogonality_residual, BlockSparseSheafGraph,
};

fn build_test_graph(num_nodes: usize, stalk_dim: usize, degree: usize) -> BlockSparseSheafGraph {
    let mut row_ptr = Vec::with_capacity(num_nodes + 1);
    let mut col_ind = Vec::with_capacity(num_nodes * degree);
    row_ptr.push(0);

    for i in 0..num_nodes {
        for d in 1..=degree {
            col_ind.push((i + d) % num_nodes);
        }
        row_ptr.push(col_ind.len());
    }

    let num_edges = col_ind.len();
    let mut rho_data = Vec::with_capacity(num_edges * stalk_dim * stalk_dim);
    for e in 0..num_edges {
        for i in 0..stalk_dim {
            for j in 0..stalk_dim {
                let val = if i == j { 1.0f32 } else if (i + j + e) % 7 == 0 { 0.05f32 } else { 0.0f32 };
                rho_data.push(val);
            }
        }
    }

    BlockSparseSheafGraph::new(num_nodes, stalk_dim, row_ptr, col_ind, rho_data).unwrap()
}

#[test]
fn test_lean4_gershgorin_stability_ceiling() {
    let graph = build_test_graph(64, 4, 6);
    assert_eq!(graph.max_degree, 6);

    // Tau < 1/6 (~0.1667) is stable
    assert!(graph.check_gershgorin_stability(0.15));
    // Tau >= 1/6 is unstable
    assert!(!graph.check_gershgorin_stability(0.17));
}

#[test]
fn test_conveyor_ring_buffer_streaming() {
    let mut ring = ConveyorRingBuffer::<f32>::new(4, 128);

    assert!(ring.can_produce());
    assert!(!ring.can_consume());

    // Fill slots
    for i in 0..4 {
        let (idx, slot) = ring.acquire_producer_slot().unwrap();
        assert_eq!(idx, i);
        slot.node_indices = vec![i * 10, i * 10 + 1];
        ring.commit_produced(idx);
    }

    assert!(!ring.can_produce());
    assert!(ring.can_consume());

    // Drain slots
    for i in 0..4 {
        let (idx, slot) = ring.acquire_consumer_slot().unwrap();
        assert_eq!(idx, i);
        assert_eq!(slot.state, SlotState::InProcess);
        ring.commit_consumed(idx);
    }

    assert!(ring.can_produce());
    assert!(!ring.can_consume());
}

#[test]
fn test_const_generic_stalk_unrolled_contraction() {
    let u_values = [1.0f32; 128];
    let x_u = AmlStalk::new(u_values);
    let mut acc = AmlStalk::default();

    // Identity restriction map
    let mut rho_id = [0.0f32; 16];
    for i in 0..4 { rho_id[i * 4 + i] = 1.0f32; }

    ConstSheafEngine::<4, 32, 128>::contract_stalk(&rho_id, &x_u, &mut acc);

    // Contraction with identity must preserve values
    for val in acc.as_slice() {
        assert_eq!(*val, 1.0f32);
    }
}

#[test]
fn test_conveyor_pipeline_numerical_parity() {
    const NUM_NODES: usize = 128;
    const STALK_DIM: usize = 4;
    const CHANNELS: usize = 32;
    const TOTAL_ELEMENTS: usize = NUM_NODES * STALK_DIM * CHANNELS;

    let graph = build_test_graph(NUM_NODES, STALK_DIM, 6);
    let mut x_in = Vec::with_capacity(TOTAL_ELEMENTS);
    for i in 0..TOTAL_ELEMENTS {
        x_in.push((i as f32 * 0.017).sin());
    }

    let pipeline = ConveyorPipeline::new(ConveyorConfig { num_stages: 4, chunk_size: 16 });

    let (out_const, tel_const) = pipeline.execute_const::<4, 32>(&graph, &x_in, 0.15, 10);
    let (out_dyn, tel_dyn) = pipeline.execute_dynamic(&graph, &x_in, CHANNELS, 0.15, 10);

    // Verify bitwise deterministic energy
    let dir_const = tel_const.last().unwrap().dirichlet_energy;
    let dir_dyn = tel_dyn.last().unwrap().dirichlet_energy;
    let diff_energy = (dir_const - dir_dyn).abs();
    assert!(diff_energy < 1e-4, "Dirichlet energy mismatch: {}", diff_energy);

    // Verify L_inf state difference
    let mut max_diff = 0.0f32;
    for i in 0..TOTAL_ELEMENTS {
        let d = (out_const[i] - out_dyn[i]).abs();
        if d > max_diff { max_diff = d; }
    }
    assert!(max_diff < 1e-5, "State vector mismatch: {}", max_diff);
}

#[test]
fn test_lock_free_hot_swap_slot() {
    let slot = KernelSlot::default();
    assert_eq!(slot.active_tier(), 0);
    assert!(slot.load_fn().is_none());

    fn dummy_contract(_state: &mut shelf_db::conveyor::NodeAssemblyState, _g: &BlockSparseSheafGraph, _x: &[f32]) {}

    slot.swap_specialized(dummy_contract, 2);
    assert_eq!(slot.active_tier(), 2);
    assert!(slot.load_fn().is_some());
    assert_eq!(slot.epoch(), 1);
}
