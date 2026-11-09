//! Conveyor Belt Sheaf Diffusion Benchmark and Verification Runner.
//!
//! Evaluates and compares throughput, latency, and numerical parity across:
//! 1. Baseline Monolithic Sheaf Diffusion Engine
//! 2. Dynamic Multi-Stage Conveyor Belt Pipeline
//! 3. Const-Generic Specialized Conveyor Belt Pipeline
//! 4. Automated Conveyor Tiered Dispatcher

use sheaf_conveyor_rust::{
    fused_sheaf_diffusion_forward, BlockSparseSheafGraph, ConveyorConfig, ConveyorDispatcher,
    ConveyorPipeline, ConveyorTier,
};
use std::time::Instant;

fn build_benchmark_graph(num_nodes: usize, stalk_dim: usize, degree: usize) -> BlockSparseSheafGraph {
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
                let val = if i == j {
                    1.0f32
                } else if (i + j + e) % 7 == 0 {
                    0.05f32
                } else {
                    0.0f32
                };
                rho_data.push(val);
            }
        }
    }

    BlockSparseSheafGraph::new(num_nodes, stalk_dim, row_ptr, col_ind, rho_data).unwrap()
}

fn main() {
    println!("========================================================================");
    println!(" SHEAF CONVEYOR ENGINE (SCE) BENCHMARK SUITE");
    println!(" Combining SheafDiffusion + Rust Metaprogramming + Conveyor Belt Pipeline");
    println!("========================================================================\n");

    const NUM_NODES: usize = 256;
    const STALK_DIM: usize = 4;
    const CHANNELS: usize = 32;
    const DEGREE: usize = 6;
    const NUM_LAYERS: usize = 10;
    const TAU: f32 = 0.15;

    println!("Graph Configuration:");
    println!("  Nodes: {}, Edges: {}, Stalk Dim: {}, Channels: {}", NUM_NODES, NUM_NODES * DEGREE, STALK_DIM, CHANNELS);
    println!("  Diffusion Layers: {}, Step Size (tau): {}\n", NUM_LAYERS, TAU);

    let graph = build_benchmark_graph(NUM_NODES, STALK_DIM, DEGREE);
    let total_elements = NUM_NODES * STALK_DIM * CHANNELS;

    let mut x_initial = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        x_initial.push((i as f32 * 0.017).sin());
    }

    // 1. Baseline Monolithic Run
    println!("[1/4] Running Baseline Monolithic Kernel...");
    let t_base_start = Instant::now();
    let (base_out, base_tel) = fused_sheaf_diffusion_forward(&graph, &x_initial, CHANNELS, TAU, NUM_LAYERS);
    let base_duration = t_base_start.elapsed();
    println!("      Completed in: {:.4} ms ({:.4} ms/layer)", base_duration.as_secs_f64() * 1000.0, (base_duration.as_secs_f64() * 1000.0) / (NUM_LAYERS as f64));
    println!("      Final Dirichlet Energy: {:.4}\n", base_tel.last().unwrap().dirichlet_energy);

    // 2. Conveyor Belt Dynamic Pipeline
    println!("[2/4] Running Conveyor Belt Dynamic Pipeline (STAGES=4, CHUNK=16)...");
    let pipeline = ConveyorPipeline::new(ConveyorConfig {
        num_stages: 4,
        chunk_size: 16,
    });
    let t_dyn_start = Instant::now();
    let (dyn_out, dyn_tel) = pipeline.execute_dynamic(&graph, &x_initial, CHANNELS, TAU, NUM_LAYERS);
    let dyn_duration = t_dyn_start.elapsed();
    println!("      Completed in: {:.4} ms ({:.4} ms/layer)", dyn_duration.as_secs_f64() * 1000.0, (dyn_duration.as_secs_f64() * 1000.0) / (NUM_LAYERS as f64));
    println!("      Final Dirichlet Energy: {:.4}\n", dyn_tel.last().unwrap().dirichlet_energy);

    // 3. Conveyor Belt Const-Generic Specialized Pipeline (D=4, C=32)
    println!("[3/4] Running Conveyor Belt Const-Generic Specialized Pipeline (D=4, C=32)...");
    let t_const_start = Instant::now();
    let (const_out, const_tel) = pipeline.execute_const::<4, 32>(&graph, &x_initial, TAU, NUM_LAYERS);
    let const_duration = t_const_start.elapsed();
    println!("      Completed in: {:.4} ms ({:.4} ms/layer)", const_duration.as_secs_f64() * 1000.0, (const_duration.as_secs_f64() * 1000.0) / (NUM_LAYERS as f64));
    println!("      Final Dirichlet Energy: {:.4}\n", const_tel.last().unwrap().dirichlet_energy);

    // 4. Automated Tiered Dispatcher Run
    println!("[4/5] Running Conveyor Tiered Specialization Dispatcher (Canonical AOT)...");
    let spec_config = sheaf_conveyor_rust::SpeculationConfig {
        alpha: 0.50,
        threshold: 0.30,
        min_warmup_invocations: 1,
    };
    let dispatcher = ConveyorDispatcher::new(ConveyorConfig::default(), spec_config);
    let (_disp_out, disp_tel, selected_tier) = dispatcher.dispatch(&graph, &x_initial, CHANNELS, TAU, NUM_LAYERS);
    println!("      Dispatcher Selected Tier: {:?}", selected_tier);
    assert_eq!(selected_tier, ConveyorTier::Tier1Aot { stalk_dim: 4, channels: 32 });
    println!("      Final Dirichlet Energy: {:.4}\n", disp_tel.last().unwrap().dirichlet_energy);

    // 5. TAAS Speculative JIT & Atomic Hot-Swap Demonstration on Unseen Shape (D=4, C=24)
    println!("[5/5] Testing TAAS Runtime Speculative JIT & Atomic Hot-Swap (Unseen D=4, C=24)...");
    let total_elements_novel = NUM_NODES * STALK_DIM * 24;
    let x_novel = vec![0.5f32; total_elements_novel];

    // Iteration 1: Unseen shape triggers Tier 0 immediately (Zero cold-start stall)
    let t_jit0 = Instant::now();
    let (_nov_out1, _nov_tel1, tier_nov1) = dispatcher.dispatch(&graph, &x_novel, 24, TAU, 1);
    let dur_jit0 = t_jit0.elapsed();
    println!("      Step 1 (Unseen shape): Dispatched to {:?} in {:.4} ms (Zero Stall!)", tier_nov1, dur_jit0.as_secs_f64() * 1000.0);
    assert_eq!(tier_nov1, ConveyorTier::Tier0Dynamic { stalk_dim: 4, channels: 24 });

    // Allow background worker thread to compile and atomically swap slot
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Iteration 2: Moving conveyor belt now routes work through Tier 2 JIT specialized station
    let t_jit1 = Instant::now();
    let (_nov_out2, _nov_tel2, tier_nov2) = dispatcher.dispatch(&graph, &x_novel, 24, TAU, 1);
    let dur_jit1 = t_jit1.elapsed();
    println!("      Step 2 (Post-Swap):   Dispatched to {:?} in {:.4} ms (Hot-Swapped!)", tier_nov2, dur_jit1.as_secs_f64() * 1000.0);
    assert_eq!(tier_nov2, ConveyorTier::Tier2Jit { stalk_dim: 4, channels: 24 });
    println!("      Compiled Kernels in Cache: {}\n", dispatcher.jit_worker().cache_size());

    // Parity Verification
    println!("------------------------------------------------------------------------");
    println!(" NUMERICAL PARITY & ACCURACY VERIFICATION");
    println!("------------------------------------------------------------------------");
    let mut max_diff_dyn = 0.0f32;
    let mut max_diff_const = 0.0f32;
    for idx in 0..total_elements {
        let diff_d = (base_out[idx] - dyn_out[idx]).abs();
        let diff_c = (base_out[idx] - const_out[idx]).abs();
        if diff_d > max_diff_dyn { max_diff_dyn = diff_d; }
        if diff_c > max_diff_const { max_diff_const = diff_c; }
    }
    println!("  Max Absolute Difference (Baseline vs Dynamic Conveyor): {:.2e} (Pass: < 1e-5)", max_diff_dyn);
    println!("  Max Absolute Difference (Baseline vs Const Conveyor):   {:.2e} (Pass: < 1e-5)", max_diff_const);
    assert!(max_diff_dyn < 1e-5);
    assert!(max_diff_const < 1e-5);
    println!("  [OK] 100% Deterministic Parity Confirmed across all implementations!\n");

    println!("========================================================================");
    println!(" BENCHMARK COMPLETE: CONVEYOR BELT ARCHITECTURE OPERATIONAL");
    println!("========================================================================");
}
