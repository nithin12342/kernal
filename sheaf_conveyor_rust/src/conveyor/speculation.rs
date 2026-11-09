//! Online Speculative Frequency Estimator & JIT Triggering Guard.
//!
//! Implements Section 1.3 of the TAAS specification:
//! Tracks tensor dimensional configuration frequencies via an Exponential Moving Average (EMA)
//! to prevent compiling transient, one-off preprocessing shapes while triggering
//! speculative JIT compilation for recurring tensor configurations.

use std::collections::HashMap;
use std::sync::Mutex;

/// Canonical key identifying a tensor shape configuration (stalk_dim, channels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorShapeKey {
    pub stalk_dim: usize,
    pub channels: usize,
}

impl TensorShapeKey {
    pub fn new(stalk_dim: usize, channels: usize) -> Self {
        Self { stalk_dim, channels }
    }
}

/// Configuration settings for the speculative compilation guard.
#[derive(Clone, Debug)]
pub struct SpeculationConfig {
    /// Smoothing factor \alpha in EMA update: E_t(s) = \alpha * 1 + (1 - \alpha) * E_{t-1}(s).
    pub alpha: f32,
    /// Threshold \theta: trigger compilation if E_t(s) >= \theta.
    pub threshold: f32,
    /// Minimum consecutive invocations before compilation can be enqueued.
    pub min_warmup_invocations: usize,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.20,
            threshold: 0.35,
            min_warmup_invocations: 2,
        }
    }
}

/// Thread-safe online EMA frequency estimator.
pub struct ShapeEmaEstimator {
    config: SpeculationConfig,
    state: Mutex<EstimatorState>,
}

struct EstimatorState {
    frequencies: HashMap<TensorShapeKey, f32>,
    invocation_counts: HashMap<TensorShapeKey, usize>,
    enqueued_shapes: HashMap<TensorShapeKey, bool>,
}

impl ShapeEmaEstimator {
    /// Creates a new speculative estimator with given configuration.
    pub fn new(config: SpeculationConfig) -> Self {
        Self {
            config,
            state: Mutex::new(EstimatorState {
                frequencies: HashMap::new(),
                invocation_counts: HashMap::new(),
                enqueued_shapes: HashMap::new(),
            }),
        }
    }

    /// Records an invocation of the given shape and returns `true` if JIT compilation should be triggered.
    pub fn record_and_evaluate(&self, active_shape: TensorShapeKey) -> bool {
        let mut state = self.state.lock().unwrap();

        // 1. Update invocation count
        let count = state.invocation_counts.entry(active_shape).or_insert(0);
        *count += 1;
        let current_count = *count;

        // 2. Check if already enqueued or compiled
        if *state.enqueued_shapes.get(&active_shape).unwrap_or(&false) {
            return false;
        }

        // 3. Update EMA for all tracked shapes
        let alpha = self.config.alpha;
        let mut should_trigger = false;

        // Ensure active shape is present
        state.frequencies.entry(active_shape).or_insert(0.0);

        for (shape, ema) in state.frequencies.iter_mut() {
            let indicator = if *shape == active_shape { 1.0f32 } else { 0.0f32 };
            *ema = alpha * indicator + (1.0f32 - alpha) * (*ema);

            if *shape == active_shape
                && *ema >= self.config.threshold
                && current_count >= self.config.min_warmup_invocations
            {
                should_trigger = true;
            }
        }

        if should_trigger {
            state.enqueued_shapes.insert(active_shape, true);
        }

        should_trigger
    }

    /// Queries current EMA score of a given shape.
    pub fn get_ema(&self, shape: &TensorShapeKey) -> f32 {
        let state = self.state.lock().unwrap();
        *state.frequencies.get(shape).unwrap_or(&0.0)
    }

    /// Marks a shape as compiled/specialized.
    pub fn mark_compiled(&self, shape: TensorShapeKey) {
        let mut state = self.state.lock().unwrap();
        state.enqueued_shapes.insert(shape, true);
    }
}
