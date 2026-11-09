//! ==============================================================================
//! LAYER 2: SHELFDB CONVEYOR - CIRCULAR MULTI-STAGE RING BUFFER
//! ==============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    Empty,
    Ready,
    InProcess,
    Done,
}

#[derive(Clone, Debug)]
pub struct ConveyorSlot<T> {
    pub payload: Vec<T>,
    pub node_indices: Vec<usize>,
    pub state: SlotState,
}

impl<T: Default + Clone> ConveyorSlot<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            payload: vec![T::default(); capacity],
            node_indices: Vec::with_capacity(capacity),
            state: SlotState::Empty,
        }
    }
}

pub struct ConveyorRingBuffer<T> {
    slots: Vec<ConveyorSlot<T>>,
    num_stages: usize,
    produced_seq: AtomicU64,
    consumed_seq: AtomicU64,
}

impl<T: Default + Clone> ConveyorRingBuffer<T> {
    pub fn new(num_stages: usize, slot_capacity: usize) -> Self {
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

    #[inline(always)]
    pub fn can_produce(&self) -> bool {
        let p = self.produced_seq.load(Ordering::Acquire);
        let c = self.consumed_seq.load(Ordering::Acquire);
        p - c < (self.num_stages as u64)
    }

    #[inline(always)]
    pub fn can_consume(&self) -> bool {
        let p = self.produced_seq.load(Ordering::Acquire);
        let c = self.consumed_seq.load(Ordering::Acquire);
        c < p
    }

    pub fn acquire_producer_slot(&mut self) -> Option<(usize, &mut ConveyorSlot<T>)> {
        let p = self.produced_seq.load(Ordering::Acquire);
        let c = self.consumed_seq.load(Ordering::Acquire);
        if p - c >= (self.num_stages as u64) {
            return None;
        }
        let slot_idx = (p as usize) % self.num_stages;
        Some((slot_idx, &mut self.slots[slot_idx]))
    }

    pub fn commit_produced(&mut self, slot_idx: usize) {
        self.slots[slot_idx].state = SlotState::Ready;
        self.produced_seq.fetch_add(1, Ordering::Release);
    }

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

    pub fn commit_consumed(&mut self, slot_idx: usize) {
        self.slots[slot_idx].state = SlotState::Empty;
        self.slots[slot_idx].node_indices.clear();
        self.consumed_seq.fetch_add(1, Ordering::Release);
    }

    pub fn reset(&mut self) {
        self.produced_seq.store(0, Ordering::Release);
        self.consumed_seq.store(0, Ordering::Release);
        for slot in &mut self.slots {
            slot.state = SlotState::Empty;
            slot.node_indices.clear();
        }
    }
}
