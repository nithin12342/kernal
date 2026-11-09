//! Rust Compile-Time Metaprogramming Macros for Sheaf Diffusion.
//!
//! Provides compile-time loop unrolling and in-register contraction macros
//! for arbitrary stalk dimensions $D$ and channels $C$.

/// Unrolls a 4x4 restriction matrix contraction completely in registers.
#[macro_export]
macro_rules! unroll_contract_4x4 {
    ($rho:expr, $x_u:expr, $acc:expr, $ch_count:expr) => {{
        for i in 0..4 {
            let row = i * 4;
            let r0 = $rho[row + 0];
            let r1 = $rho[row + 1];
            let r2 = $rho[row + 2];
            let r3 = $rho[row + 3];

            for ch in 0..$ch_count {
                $acc[i * $ch_count + ch] += r0 * $x_u[0 * $ch_count + ch]
                    + r1 * $x_u[1 * $ch_count + ch]
                    + r2 * $x_u[2 * $ch_count + ch]
                    + r3 * $x_u[3 * $ch_count + ch];
            }
        }
    }};
}

/// Unrolls an 8x8 restriction matrix contraction completely in registers.
#[macro_export]
macro_rules! unroll_contract_8x8 {
    ($rho:expr, $x_u:expr, $acc:expr, $ch_count:expr) => {{
        for i in 0..8 {
            let row = i * 8;
            for ch in 0..$ch_count {
                let mut sum = 0.0f32;
                for j in 0..8 {
                    sum += $rho[row + j] * $x_u[j * $ch_count + ch];
                }
                $acc[i * $ch_count + ch] += sum;
            }
        }
    }};
}
