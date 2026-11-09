//! ==============================================================================
//! LAYER 2: SHELFDB CONVEYOR - LOCK-FREE HOT-SWAPPABLE KERNEL SLOT
//! ==============================================================================

use crate::conveyor::stages::NodeAssemblyState;
use crate::storage::BlockSparseSheafGraph;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU8, Ordering};

pub type SpecializedContractFn = fn(
    state: &mut NodeAssemblyState,
    graph: &BlockSparseSheafGraph,
    x_src: &[f32],
);

pub struct KernelSlot {
    active_fn: AtomicPtr<SpecializedContractFn>,
    active_tier: AtomicU8,
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
    pub fn new(specialized_fn: SpecializedContractFn, tier: u8) -> Self {
        let boxed = Box::new(specialized_fn);
        Self {
            active_fn: AtomicPtr::new(Box::into_raw(boxed)),
            active_tier: AtomicU8::new(tier),
            epoch: AtomicU64::new(1),
        }
    }

    #[inline(always)]
    pub fn load_fn(&self) -> Option<SpecializedContractFn> {
        let ptr = self.active_fn.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(*ptr) }
        }
    }

    #[inline(always)]
    pub fn active_tier(&self) -> u8 {
        self.active_tier.load(Ordering::Acquire)
    }

    pub fn swap_specialized(&self, new_fn: SpecializedContractFn, tier: u8) {
        let boxed = Box::new(new_fn);
        let raw = Box::into_raw(boxed);
        let old_ptr = self.active_fn.swap(raw, Ordering::Release);
        self.active_tier.store(tier, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::Relaxed);
        if !old_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(old_ptr));
            }
        }
    }

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
