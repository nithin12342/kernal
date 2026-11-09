//! ==============================================================================
//! LAYER 2: SHELFDB CONVEYOR - SPECULATIVE SHAPE TRACKER
//! ==============================================================================

use std::collections::HashMap;
use std::sync::Mutex;

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

#[derive(Clone, Debug)]
pub struct SpeculationConfig {
    pub alpha: f32,
    pub threshold: f32,
    pub min_warmup_invocations: usize,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.50,
            threshold: 0.30,
            min_warmup_invocations: 1,
        }
    }
}

pub struct ShapeEmaEstimator {
    config: SpeculationConfig,
    frequencies: Mutex<HashMap<TensorShapeKey, f32>>,
    counts: Mutex<HashMap<TensorShapeKey, usize>>,
}

impl ShapeEmaEstimator {
    pub fn new(config: SpeculationConfig) -> Self {
        Self {
            config,
            frequencies: Mutex::new(HashMap::new()),
            counts: Mutex::new(HashMap::new()),
        }
    }

    pub fn record_and_evaluate(&self, shape: TensorShapeKey) -> bool {
        let mut counts = self.counts.lock().unwrap();
        let c = counts.entry(shape).or_insert(0);
        *c += 1;
        let mut freqs = self.frequencies.lock().unwrap();
        let f = freqs.entry(shape).or_insert(0.0);
        *f = self.config.alpha * 1.0 + (1.0 - self.config.alpha) * (*f);
        *f >= self.config.threshold && *c >= self.config.min_warmup_invocations
    }
}
