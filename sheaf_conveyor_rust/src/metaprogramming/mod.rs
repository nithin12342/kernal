//! Metaprogramming Layer for Fused Sheaf Diffusion in Rust.
//!
//! Provides compile-time macros and const-generic metaprogrammed engines
//! for synthesizing zero-overhead loopless contractions.

pub mod const_metaprog;
pub mod macros;

pub use const_metaprog::ConstSheafEngine;
