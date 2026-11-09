//! Lock-Free Atomic Kernel Hot-Swapping & Epoch Tracking.
//!
//! Implements Section 3 of the TAAS specification:
//! Enables hot-swapping dynamically compiled kernel entry points on a moving
//! conveyor belt without mutex locks or GPU/CPU pipeline stalls.

use crate::conveyor::stages::NodeAssemblyState;
use crate::domain::bsr::BlockSparseSheafGraph;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU8, Ordering};

/// Function pointer type for specialized kernel contraction station.
pub type SpecializedContractFn = fn(
    state: &mut NodeAssemblyState,
    graph: &BlockSparseSheafGraph,
    x_src: &[f32],
);

/// A lock-free atomic execution slot on the conveyor belt.
pub struct KernelSlot {
    /// Atomic pointer to specialized function entry point (null if unspecialized).
    active_fn: AtomicPtr<SpecializedContractFn>,
    /// Active tier indicator: 0 = Dynamic, 1 = AOT, 2 = JIT.
    active_tier: AtomicU8,
    /// Monotonic epoch counter for in-flight execution retirement.
    epoch: AtomicU64,
}

impl Default for KernelSlot {
    fn default() -> Self {
        Self {
            active_fn: AtomicPtr::new(std::ptr::null_mut()),
            active_tier: AtomicU8::new(0),
            epoch: AtomicU64::new(0),
        }
    }
}

impl KernelSlot {
    /// Creates a new slot initialized with a specialized function and tier.
    pub fn new(specialized_fn: SpecializedContractFn, tier: u8) -> Self {
        let boxed = Box::new(specialized_fn);
        Self {
            active_fn: AtomicPtr::new(Box::into_raw(boxed)),
            active_tier: AtomicU8::new(tier),
            epoch: AtomicU64::new(1),
        }
    }

    /// Fast-path lock-free load: compiles to a single 64-bit register load with Acquire semantics.
    #[inline(always)]
    pub fn load_fn(&self) -> Option<SpecializedContractFn> {
        let ptr = self.active_fn.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(*ptr) }
        }
    }

    /// Queries the currently active tier.
    #[inline(always)]
    pub fn active_tier(&self) -> u8 {
        self.active_tier.load(Ordering::Acquire)
    }

    /// Atomically swaps in a newly compiled specialized kernel entry point.
    ///
    /// The Release ordering ensures all machine code generation and validation
    /// memory writes are visible before the pointer becomes observable to the conveyor belt.
    pub fn swap_specialized(&self, new_fn: SpecializedContractFn, tier: u8) {
        let boxed = Box::new(new_fn);
        let raw = Box::into_raw(boxed);
        let old_ptr = self.active_fn.swap(raw, Ordering::Release);
        self.active_tier.store(tier, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::Relaxed);

        // Deallocate previous specialized function if present
        if !old_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(old_ptr));
            }
        }
    }

    /// Current slot epoch count.
    #[inline(always)]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }
}

impl Drop for KernelSlot {
    fn drop(&mut self) {
        let ptr = self.active_fn.load(Ordering::Acquire);
        if !ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}
