//! ==============================================================================
//! LAYER 1: RUST METAPROGRAMMING - CONST GENERIC STALKS
//! ==============================================================================
//! Bakes stalk dimension D, channel count C, and total length N directly into the
//! Rust type system, ensuring zero heap allocation and 64-byte alignment in stable Rust.
//! ==============================================================================

use std::fmt::Debug;

/// Stalk representation baking stalk dimension D, channels C, and stride N into compile-time types.
/// Aligned to 64 bytes for cache line saturation.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Stalk<const D: usize, const C: usize, const N: usize> {
    pub values: [f32; N],
}

impl<const D: usize, const C: usize, const N: usize> Debug for Stalk<D, C, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stalk<D={}, C={}, N={}>(len={})", D, C, N, N)
    }
}

impl<const D: usize, const C: usize, const N: usize> Default for Stalk<D, C, N> {
    fn default() -> Self {
        Self { values: [0.0f32; N] }
    }
}

impl<const D: usize, const C: usize, const N: usize> Stalk<D, C, N> {
    pub const TOTAL_ELEMENTS: usize = N;
    pub const BYTE_SIZE: usize = N * std::mem::size_of::<f32>();

    #[inline(always)]
    pub fn new(values: [f32; N]) -> Self {
        debug_assert_eq!(D * C, N);
        Self { values }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    #[inline(always)]
    pub fn get_channel_stalk(&self, channel: usize) -> [f32; D] {
        debug_assert!(channel < C);
        let mut out = [0.0f32; D];
        for d in 0..D {
            out[d] = self.values[d * C + channel];
        }
        out
    }

    #[inline(always)]
    pub fn l2_norm_sq(&self) -> f32 {
        let mut sum = 0.0f32;
        for &v in &self.values {
            sum += v * v;
        }
        sum
    }
}

/// Pre-specialized type alias for 128-D latent financial anti-money laundering (AML: D=4, C=32, N=128)
pub type AmlStalk = Stalk<4, 32, 128>;

/// Pre-specialized type alias for 512-D latent precision oncology synergy (Oncology: D=8, C=64, N=512)
pub type OncologyStalk = Stalk<8, 64, 512>;
