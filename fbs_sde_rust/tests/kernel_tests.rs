use fbs_sde_rust::{
    cayley_projection_4x4, compute_orthogonality_residual, fused_sheaf_diffusion_backward_step,
    fused_sheaf_diffusion_forward, BlockSparseSheafGraph, ConstSheafEngine,
};

#[test]
fn test_bsr_graph_construction_and_validation() {
    let row_ptr = vec![0, 1, 2];
    let col_ind = vec![1, 0];
    let rho_data = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0, // Edge 0 (Identity)
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0, // Edge 1 (Identity)
    ];

    let graph = BlockSparseSheafGraph::new(2, 4, row_ptr, col_ind, rho_data).unwrap();
    assert_eq!(graph.num_nodes, 2);
    assert_eq!(graph.num_edges, 2);
    assert_eq!(graph.degrees, vec![1, 1]);
}

#[test]
fn test_forward_diffusion_step_and_telemetry() {
    let row_ptr = vec![0, 1, 2];
    let col_ind = vec![1, 0];
    let rho_data = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let graph = BlockSparseSheafGraph::new(2, 4, row_ptr, col_ind, rho_data).unwrap();

    let channels = 2;
    let mut x_in = vec![0.0f32; 2 * 4 * channels];
    x_in[..4 * channels].fill(1.0f32);

    let tau = 0.5f32;
    let (x_out, telemetry) = fused_sheaf_diffusion_forward(&graph, &x_in, channels, tau, 1);

    assert_eq!(telemetry.len(), 1);
    assert!(telemetry[0].dirichlet_energy > 0.0);
    assert!(!telemetry[0].is_unstable);

    for val in &x_out {
        assert!((val - 0.5f32).abs() < 1e-5);
    }
}

#[test]
fn test_orthogonality_and_cayley_projection() {
    let id_4x4 = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert!(compute_orthogonality_residual(&id_4x4, 4) < 1e-6);

    let mut drifted = id_4x4.clone();
    drifted[0] = 1.05;
    let drift = compute_orthogonality_residual(&drifted, 4);
    assert!(drift > 0.01);

    cayley_projection_4x4(&mut drifted);
    let post_project_drift = compute_orthogonality_residual(&drifted, 4);
    assert!(post_project_drift < 1e-4);
}

#[test]
fn test_backward_adjoint_gradients() {
    let row_ptr = vec![0, 1, 2];
    let col_ind = vec![1, 0];
    let rho_data = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let graph = BlockSparseSheafGraph::new(2, 4, row_ptr, col_ind, rho_data).unwrap();

    let channels = 2;
    let x_in = vec![1.0f32; 2 * 4 * channels];
    let grad_out = vec![1.0f32; 2 * 4 * channels];

    let grads = fused_sheaf_diffusion_backward_step(&graph, &x_in, &grad_out, channels, 0.2);
    assert_eq!(grads.grad_x_in.len(), 2 * 4 * channels);
    assert_eq!(grads.grad_rho.len(), 2 * 4 * 4);
}

#[test]
fn test_const_generic_metaprogrammed_engine() {
    let row_ptr = vec![0, 1, 2];
    let col_ind = vec![1, 0];
    let rho_data = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let graph = BlockSparseSheafGraph::new(2, 4, row_ptr, col_ind, rho_data).unwrap();

    let mut x_in = vec![0.0f32; 2 * 4 * 2];
    x_in[..8].fill(1.0f32);
    let mut x_out = vec![0.0f32; 2 * 4 * 2];

    let telem = ConstSheafEngine::<4, 2>::forward_step(&graph, &x_in, &mut x_out, 0.5);
    assert!(telem.dirichlet_energy > 0.0);
    for val in &x_out {
        assert!((val - 0.5f32).abs() < 1e-5);
    }
}
