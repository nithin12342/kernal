//! Cache-Aligned Circular Multi-Stage Ring Buffer for GPU/CPU Conveyor Pipelines.
//!
//! Provides a lock-free, zero-allocation circular multi-stage buffer governed by
//! monotonic 64-bit sequence counters. Decouples producer data loading from
//! consumer arithmetic processing.

use std::sync::atomic::{AtomicU64, Ordering};

/// Slot status within the circular conveyor belt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is available for producer loading.
    Empty,
    /// Data has been loaded by producer; ready for consumer processing.
    Ready,
    /// Currently being processed by consumer pipeline stations.
    InProcess,
    /// Processing complete; awaiting egress writeback.
    Done,
}

/// A pre-allocated data slot on the conveyor belt holding a batch of node stalks.
#[derive(Clone, Debug)]
pub struct ConveyorSlot<T> {
    /// Tile payload buffer (pre-allocated to avoid runtime heap allocation).
    pub payload: Vec<T>,
    /// Global node indices assigned to this batch slot.
    pub node_indices: Vec<usize>,
    /// Current lifecycle state of this conveyor slot.
    pub state: SlotState,
}

impl<T: Default + Clone> ConveyorSlot<T> {
    /// Allocates an empty slot with pre-sized capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            payload: vec![T::default(); capacity],
            node_indices: Vec::with_capacity(capacity),
            state: SlotState::Empty,
        }
    }
}

/// Multi-stage circular ring buffer for the conveyor belt pipeline.
///
/// Parameterized by stage depth `STAGES` (typically 4 stages to achieve
/// 100% memory-compute overlap).
pub struct ConveyorRingBuffer<T> {
    /// Pre-allocated ring buffer slots.
    slots: Vec<ConveyorSlot<T>>,
    /// Number of stages in the circular belt.
    num_stages: usize,
    /// Monotonic sequence counter of produced batches.
    produced_seq: AtomicU64,
    /// Monotonic sequence counter of consumed batches.
    consumed_seq: AtomicU64,
}

impl<T: Default + Clone> ConveyorRingBuffer<T> {
    /// Creates a new circular ring buffer with `num_stages` slots of given tile capacity.
    pub fn new(num_stages: usize, slot_capacity: usize) -> Self {
        assert!(num_stages >= 2, "Conveyor belt requires at least 2 stages");
        let mut slots = Vec::with_capacity(num_stages);
        for _ in 0..num_stages {
            slots.push(ConveyorSlot::new(slot_capacity));
        }
        Self {
            slots,
            num_stages,
            produced_seq: AtomicU64::new(0),
            consumed_seq: AtomicU64::new(0),
        }
    }

    /// Number of conveyor stages.
    #[inline(always)]
    pub fn stages(&self) -> usize {
        self.num_stages
    }

    /// Checks if a slot is available for production without overflowing the belt.
    ///
    /// Invariant: P_head - C_tail < STAGES.
    #[inline(always)]
    pub fn can_produce(&self) -> bool {
        let p = self.produced_seq.load(Ordering::Acquire);
        let c = self.consumed_seq.load(Ordering::Acquire);
        p - c < (self.num_stages as u64)
    }

    /// Checks if a ready slot is available for consumption.
    ///
    /// Invariant: C_tail < P_head.
    #[inline(always)]
    pub fn can_consume(&self) -> bool {
        let p = self.produced_seq.load(Ordering::Acquire);
        let c = self.consumed_seq.load(Ordering::Acquire);
        c < p
    }

    /// Acquires mutable access to the current producer slot.
    /// Returns `None` if the belt is currently full (all stages occupied).
    pub fn acquire_producer_slot(&mut self) -> Option<(usize, &mut ConveyorSlot<T>)> {
        let p = self.produced_seq.load(Ordering::Acquire);
        let c = self.consumed_seq.load(Ordering::Acquire);
        if p - c >= (self.num_stages as u64) {
            return None;
        }
        let slot_idx = (p as usize) % self.num_stages;
        Some((slot_idx, &mut self.slots[slot_idx]))
    }

    /// Commits the current producer slot and advances the produced sequence counter.
    pub fn commit_produced(&mut self, slot_idx: usize) {
        assert_eq!(
            slot_idx,
            (self.produced_seq.load(Ordering::Relaxed) as usize) % self.num_stages
        );
        self.slots[slot_idx].state = SlotState::Ready;
        self.produced_seq.fetch_add(1, Ordering::Release);
    }

    /// Acquires mutable access to the current consumer slot.
    /// Returns `None` if the belt is empty (producers have not committed new tiles).
    pub fn acquire_consumer_slot(&mut self) -> Option<(usize, &mut ConveyorSlot<T>)> {
        let p = self.produced_seq.load(Ordering::Acquire);
        let c = self.consumed_seq.load(Ordering::Acquire);
        if c >= p {
            return None;
        }
        let slot_idx = (c as usize) % self.num_stages;
        self.slots[slot_idx].state = SlotState::InProcess;
        Some((slot_idx, &mut self.slots[slot_idx]))
    }

    /// Commits the consumer slot and advances the consumed sequence counter.
    pub fn commit_consumed(&mut self, slot_idx: usize) {
        assert_eq!(
            slot_idx,
            (self.consumed_seq.load(Ordering::Relaxed) as usize) % self.num_stages
        );
        self.slots[slot_idx].state = SlotState::Empty;
        self.slots[slot_idx].node_indices.clear();
        self.consumed_seq.fetch_add(1, Ordering::Release);
    }

    /// Current monotonic produced sequence count.
    #[inline(always)]
    pub fn produced_count(&self) -> u64 {
        self.produced_seq.load(Ordering::Acquire)
    }

    /// Current monotonic consumed sequence count.
    #[inline(always)]
    pub fn consumed_count(&self) -> u64 {
        self.consumed_seq.load(Ordering::Acquire)
    }

    /// Resets sequence counters and slot states for a new diffusion layer.
    pub fn reset(&mut self) {
        self.produced_seq.store(0, Ordering::Release);
        self.consumed_seq.store(0, Ordering::Release);
        for slot in &mut self.slots {
            slot.state = SlotState::Empty;
            slot.node_indices.clear();
        }
    }
}
