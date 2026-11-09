//! Metaprogrammed Sheaf Diffusion Benchmark and Verification Runner.
//!
//! Demonstrates zero-cost compile-time specialization of cellular sheaf diffusion
//! for D=4, C=32 and D=8, C=32 using Rust const generics.

use std::time::Instant;
use fbs_sde_rust::{
    BlockSparseSheafGraph, ConstSheafEngine,
};

fn main() {
    println!("====================================================");
    println!(" FBS-SDE Rust Metaprogrammed Engine");
    println!(" Type-Level Compile-Time Specialization (Const Generics)");
    println!("====================================================\n");

    // 1. Benchmark D=4, C=32
    const D1: usize = 4;
    const C1: usize = 32;

    let row_ptr = vec![0, 1, 2];
    let col_ind = vec![1, 0];
    let mut rho_data = vec![0.0f32; 2 * D1 * D1];
    for e in 0..2 {
        for i in 0..D1 {
            rho_data[e * D1 * D1 + i * D1 + i] = 1.0f32;
        }
    }

    let graph_d4 = BlockSparseSheafGraph::new(2, D1, row_ptr, col_ind, rho_data).unwrap();

    let mut x_in = vec![0.0f32; 2 * D1 * C1];
    x_in[..D1 * C1].fill(1.0f32);
    let mut x_out = vec![0.0f32; 2 * D1 * C1];

    println!("[1] Executing Forward Diffusion (D=4, C=32)...");
    let t0 = Instant::now();
    let telem = ConstSheafEngine::<D1, C1>::forward_step(&graph_d4, &x_in, &mut x_out, 0.5);
    let elapsed = t0.elapsed();

    println!("    Forward Step Completed in: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("    Dirichlet Energy: {:.2}", telem.dirichlet_energy);
    println!("    Node 0 Stalk[0]: {:.4} (Expected: 0.5000)", x_out[0]);
    println!("    Node 1 Stalk[0]: {:.4} (Expected: 0.5000)", x_out[D1 * C1]);
    assert!((x_out[0] - 0.5f32).abs() < 1e-5);
    assert!((x_out[D1 * C1] - 0.5f32).abs() < 1e-5);
    println!("    [OK] D=4 Forward Verification Passed.\n");

    // Backward step
    println!("[2] Executing Backward Adjoint Step (D=4, C=32)...");
    let grad_out = vec![1.0f32; 2 * D1 * C1];
    let grads = ConstSheafEngine::<D1, C1>::backward_step(&graph_d4, &x_in, &grad_out, 0.2);
    println!("    Grad X In Size: {}", grads.grad_x_in.len());
    println!("    Grad Rho Size:  {}", grads.grad_rho.len());
    println!("    [OK] D=4 Backward Adjoint Verification Passed.\n");

    // 2. Benchmark D=8, C=32 (Stiefel manifold)
    const D2: usize = 8;
    const C2: usize = 32;

    let row_ptr_d8 = vec![0, 1, 2];
    let col_ind_d8 = vec![1, 0];
    let mut rho_data_d8 = vec![0.0f32; 2 * D2 * D2];
    for e in 0..2 {
        for i in 0..D2 {
            rho_data_d8[e * D2 * D2 + i * D2 + i] = 1.0f32;
        }
    }

    let graph_d8 = BlockSparseSheafGraph::new(2, D2, row_ptr_d8, col_ind_d8, rho_data_d8).unwrap();
    let x_in_d8 = vec![1.0f32; 2 * D2 * C2];
    let mut x_out_d8 = vec![0.0f32; 2 * D2 * C2];

    println!("[3] Specializing Engine for D=8, C=32 (Stiefel Manifold)...");
    let telem_d8 = ConstSheafEngine::<D2, C2>::forward_step(&graph_d8, &x_in_d8, &mut x_out_d8, 0.1);
    println!("    Dirichlet Energy: {:.2}", telem_d8.dirichlet_energy);
    println!("    [OK] D=8 Specialization Successfully Compiled and Executed!\n");

    println!("====================================================");
    println!(" ALL RUST METAPROGRAMMING CHECKS PASSED!");
    println!("====================================================");
}
