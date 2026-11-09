//! ==============================================================================
//! LAYER 2: SHELFDB STORAGE - TELEMETRY & CAYLEY RETRACTIONS
//! ==============================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SheafTelemetry {
    pub dirichlet_energy: f32,
    pub max_orthogonality_drift: f32,
    pub is_unstable: bool,
    pub is_collapsed: bool,
}

/// Computes the Frobenius orthogonality residual: ||rho^T * rho - I_d||_F
pub fn compute_orthogonality_residual(rho: &[f32], d: usize) -> f32 {
    let mut sum_sq = 0.0f32;
    for i in 0..d {
        for j in 0..d {
            let mut dot = 0.0f32;
            for k in 0..d {
                dot += rho[k * d + i] * rho[k * d + j];
            }
            let target = if i == j { 1.0f32 } else { 0.0f32 };
            let diff = dot - target;
            sum_sq += diff * diff;
        }
    }
    sum_sq.sqrt()
}

/// Fast in-place Cayley projection from skew-symmetric parameter matrix to SO(4)
pub fn cayley_projection_4x4(rho: &mut [f32]) {
    debug_assert_eq!(rho.len(), 16);
    let mut a = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            a[i * 4 + j] = 0.5 * (rho[i * 4 + j] - rho[j * 4 + i]);
        }
    }
    // Identity + A
    let mut i_plus_a = [0.0f32; 16];
    let mut i_minus_a = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let id = if i == j { 1.0f32 } else { 0.0f32 };
            i_plus_a[i * 4 + j] = id + a[i * 4 + j];
            i_minus_a[i * 4 + j] = id - a[i * 4 + j];
        }
    }
}
