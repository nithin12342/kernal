//! ==============================================================================
//! BENCHMARK BINARY: 100-STEP CPU SIMD CONVEYOR BENCHMARK
//! ==============================================================================
//! Architecture & Invariants:
//! Runs 100 multi-hop diffusion steps using the Rust Conveyor Engine on CPU SIMD.
//! Formats execution metrics as key-value pairs for automated test harness parsing.
//! ==============================================================================

use sheaf_conveyor_rust::{BlockSparseSheafGraph, ConveyorConfig, ConveyorPipeline};
use std::time::Instant;

fn main() {
    const NUM_NODES: usize = 256;
    const DEGREE: usize = 6;
    const STALK_DIM: usize = 4;
    const CHANNELS: usize = 32;
    const NUM_LAYERS: usize = 100;
    const TAU: f32 = 0.15;

    let mut row_ptr = Vec::with_capacity(NUM_NODES + 1);
    let mut col_ind = Vec::with_capacity(NUM_NODES * DEGREE);
    row_ptr.push(0);
    for i in 0..NUM_NODES {
        for d in 1..=DEGREE {
            col_ind.push((i + d) % NUM_NODES);
        }
        row_ptr.push(col_ind.len());
    }

    let num_edges = col_ind.len();
    let mut rho_data = Vec::with_capacity(num_edges * STALK_DIM * STALK_DIM);
    for e in 0..num_edges {
        for i in 0..STALK_DIM {
            for j in 0..STALK_DIM {
                let val = if i == j { 1.0f32 } else if (i + j + e) % 7 == 0 { 0.05f32 } else { 0.0f32 };
                rho_data.push(val);
            }
        }
    }

    let graph = BlockSparseSheafGraph::new(NUM_NODES, STALK_DIM, row_ptr, col_ind, rho_data).unwrap();
    let total_elements = NUM_NODES * STALK_DIM * CHANNELS;
    let mut x_in = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        x_in.push((i as f32 * 0.017).sin());
    }

    let pipeline = ConveyorPipeline::new(ConveyorConfig { num_stages: 4, chunk_size: 16 });
    let t0 = Instant::now();
    let (out, tel) = pipeline.execute_const::<4, 32>(&graph, &x_in, TAU, NUM_LAYERS);
    let elapsed = t0.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    println!("RUST_TOTAL_MS={:.6}", total_ms);
    println!("RUST_PER_STEP_MS={:.6}", total_ms / (NUM_LAYERS as f64));
    println!("RUST_DIRICHLET={:.6}", tel.last().unwrap().dirichlet_energy);
    println!("RUST_OUT_0={:.6}", out[0]);
    println!("RUST_OUT_1={:.6}", out[1]);
}
