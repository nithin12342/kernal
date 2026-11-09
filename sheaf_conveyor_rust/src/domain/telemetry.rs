//! In-Kernel Telemetry & Manifold Stabilization Module.
//!
//! Implements Section 7 of the specification:
//! - 7.1 Real-Time Sheaf Dirichlet Energy Tracking: $\mathcal{E}_{\mathcal{F}}(e) = \frac{1}{2} \|\Delta x_e\|_F^2$.
//! - 7.2 Orthogonality Residual Verification: $\delta_{\mathrm{orth}}(\rho) = \|\rho^T \rho - I_d\|_F$.
//! - On-Chip Cayley projection to stabilize Stiefel/Orthogonal drift.

/// In-kernel telemetry record for a diffusion step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SheafTelemetry {
    /// Global Dirichlet energy $\mathcal{E}_{\mathcal{F}} = \frac{1}{2} \sum_{e} \|\Delta x_e\|_F^2$.
    pub dirichlet_energy: f32,
    /// Maximum deviation from orthogonality across sampled edge maps: $\|\rho^T \rho - I_d\|_F$.
    pub max_orthogonality_drift: f32,
    /// Whether numerical instability (NaN/Inf) was flagged.
    pub is_unstable: bool,
    /// Whether stalk collapse / oversmoothing was flagged ($\mathcal{E}_{\mathcal{F}} < 10^{-7}$).
    pub is_collapsed: bool,
}

/// Evaluates the orthogonality residual $\delta_{\mathrm{orth}}(\rho) = \|\rho^T \rho - I_d\|_F$.
pub fn compute_orthogonality_residual(rho: &[f32], d: usize) -> f32 {
    let mut sum_sq = 0.0f32;
    for i in 0..d {
        for j in 0..d {
            // Compute (rho^T * rho)[i, j] = sum_k rho[k, i] * rho[k, j]
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

/// Projects a drifted restriction map back onto O(d) using the Cayley transform:
/// $\rho \leftarrow (I - A)(I + A)^{-1}$, where $A = \frac{1}{2}(\rho - \rho^T)$.
pub fn cayley_projection_4x4(rho: &mut [f32]) {
    debug_assert_eq!(rho.len(), 16);
    // Skew-symmetric component A = 0.5 * (rho - rho^T)
    let mut a = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            a[i * 4 + j] = 0.5 * (rho[i * 4 + j] - rho[j * 4 + i]);
        }
    }

    // M_plus = I + A, M_minus = I - A
    let mut m_plus = [0.0f32; 16];
    let mut m_minus = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let id = if i == j { 1.0f32 } else { 0.0f32 };
            m_plus[i * 4 + j] = id + a[i * 4 + j];
            m_minus[i * 4 + j] = id - a[i * 4 + j];
        }
    }

    // Invert M_plus using 4x4 Gauss-Jordan elimination
    let mut inv = [0.0f32; 16];
    for i in 0..4 {
        inv[i * 4 + i] = 1.0f32;
    }
    let mut aug = m_plus;

    for col in 0..4 {
        let mut pivot = col;
        let mut max_val = aug[col * 4 + col].abs();
        for row in (col + 1)..4 {
            let val = aug[row * 4 + col].abs();
            if val > max_val {
                max_val = val;
                pivot = row;
            }
        }
        if pivot != col {
            for k in 0..4 {
                aug.swap(col * 4 + k, pivot * 4 + k);
                inv.swap(col * 4 + k, pivot * 4 + k);
            }
        }
        let p_val = aug[col * 4 + col];
        if p_val.abs() < 1e-12 {
            return; // Degenerate pivot, preserve original
        }
        for k in 0..4 {
            aug[col * 4 + k] /= p_val;
            inv[col * 4 + k] /= p_val;
        }
        for row in 0..4 {
            if row != col {
                let factor = aug[row * 4 + col];
                for k in 0..4 {
                    aug[row * 4 + k] -= factor * aug[col * 4 + k];
                    inv[row * 4 + k] -= factor * inv[col * 4 + k];
                }
            }
        }
    }

    // Compute (I - A) * (I + A)^-1
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += m_minus[i * 4 + k] * inv[k * 4 + j];
            }
            rho[i * 4 + j] = sum;
        }
    }
}
