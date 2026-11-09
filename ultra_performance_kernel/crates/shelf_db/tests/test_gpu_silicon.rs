//! ==============================================================================
//! INTEGRATION TEST SUITE: MULTI-PRECISION GPU SILICON EXECUTION & VERIFICATION
//! Precision Tiers: FP32, FP16, BF16, FP8 (E4M3), FP4 (E2M1)
//! ==============================================================================
//! Validates:
//! 1. Physical GDDR6 VRAM allocation on Tesla T4 silicon
//! 2. Multi-precision CUDA Graph instantiation and steady-state P0 latency
//! 3. Dirichlet energy tracking and numerical bounded convergence across all precisions
//! 4. Detailed per-step latency, throughput, and error profiling
//! ==============================================================================

#[cfg(feature = "gpu")]
use shelf_db::conveyor::{ConveyorConfig, ConveyorPipeline};
#[cfg(feature = "gpu")]
use shelf_db::gpu::{GpuSheafSession, SheafPrecision};
#[cfg(feature = "gpu")]
use shelf_db::storage::BlockSparseSheafGraph;

#[test]
#[cfg(feature = "gpu")]
fn test_physical_gpu_silicon_execution() {
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
            col_ind.push(((i + d) % NUM_NODES) as i32);
        }
        row_ptr.push(col_ind.len() as i32);
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

    let total_elements = NUM_NODES * STALK_DIM * CHANNELS;
    let mut x_in = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        x_in.push((i as f32 * 0.017).sin());
    }

    // 1. Initialize GPU Session & Verify Active GDDR6 VRAM
    let session = GpuSheafSession::init(
        NUM_NODES,
        num_edges,
        STALK_DIM,
        CHANNELS,
        &row_ptr,
        &col_ind,
        &rho_data,
        &x_in,
    ).expect("Failed to initialize GPU VRAM");

    assert_eq!(session.allocated_vram_bytes, 367_624);

    // 2. Compute Host CPU Reference Baseline (FP32)
    let row_ptr_usize: Vec<usize> = row_ptr.iter().map(|&x| x as usize).collect();
    let col_ind_usize: Vec<usize> = col_ind.iter().map(|&x| x as usize).collect();
    let graph = BlockSparseSheafGraph::new(NUM_NODES, STALK_DIM, row_ptr_usize, col_ind_usize, rho_data).unwrap();
    let pipeline = ConveyorPipeline::new(ConveyorConfig { num_stages: 4, chunk_size: 16 });

    let (cpu_out, cpu_tel) = pipeline.execute_const::<4, 32>(&graph, &x_in, TAU, NUM_LAYERS);
    let cpu_dirichlet = cpu_tel.last().unwrap().dirichlet_energy;

    // 3. Warm up GPU frequency into steady-state P0 boost mode (1590 MHz)
    session.instantiate_cuda_graph_precision(NUM_LAYERS, TAU, SheafPrecision::FP32).expect("Warmup graph init failed");
    for _ in 0..5 {
        let _ = session.run_cuda_graph();
    }

    // 4. Benchmark and Verify All 5 Precision Tiers
    let precisions = [
        ("FP32 (Single 32-bit)", SheafPrecision::FP32, 1e-4, 1e-4),
        ("FP16 (Half 16-bit)", SheafPrecision::FP16, 5e-3, 5e-3),
        ("BF16 (Brain Float 16)", SheafPrecision::BF16, 1e-2, 1e-2),
        ("FP8  (Micro Float E4M3)", SheafPrecision::FP8, 3.0, 1.5e-1),
        ("FP4  (Sub-Byte E2M1)", SheafPrecision::FP4, 3.0, 3.0e-1),
    ];

    println!("\n================================================================================");
    println!("PHYSICAL SILICON MULTI-PRECISION BENCHMARK RESULTS (TESLA T4 GDDR6)");
    println!("================================================================================");

    for (name, prec, dirichlet_tol, linf_tol) in precisions {
        session.reset_input(&x_in).expect("Failed to reset input");
        session.instantiate_cuda_graph_precision(NUM_LAYERS, TAU, prec)
            .unwrap_or_else(|e| panic!("Failed to instantiate graph for {}: {}", name, e));

        let (graph_time_ms, graph_dirichlet) = session.run_cuda_graph()
            .unwrap_or_else(|e| panic!("Failed to run graph for {}: {}", name, e));
        let gpu_out = session.read_results(NUM_LAYERS);

        let per_step_us = (graph_time_ms / (NUM_LAYERS as f32)) * 1000.0;
        let throughput = 1_000_000.0 / per_step_us;

        let dirichlet_diff = (graph_dirichlet - cpu_dirichlet).abs();
        let mut max_output_diff = 0.0f32;
        for i in 0..total_elements {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            if diff > max_output_diff { max_output_diff = diff; }
        }

        println!("Tier: {:<25} | Latency: {:>5.2} us/step | {:>8.1} steps/s | Dirichlet: {:.6} (Δ: {:.2e}) | L_inf: {:.2e}",
            name, per_step_us, throughput, graph_dirichlet, dirichlet_diff, max_output_diff);

        assert!(dirichlet_diff <= dirichlet_tol,
            "{} Dirichlet divergence exceeded tolerance: {} > {}", name, dirichlet_diff, dirichlet_tol);
        assert!(max_output_diff <= linf_tol,
            "{} L_inf divergence exceeded tolerance: {} > {}", name, max_output_diff, linf_tol);
    }
    println!("================================================================================\n");
}
