//! Pipelined Producer-Consumer Execution Engine for Sheaf Diffusion.
//!
//! Orchestrates the multi-stage conveyor belt across node chunks, overlapping
//! producer batch loading with consumer arithmetic transformation.

use crate::domain::bsr::BlockSparseSheafGraph;
use crate::domain::telemetry::SheafTelemetry;
use crate::conveyor::hot_swap::KernelSlot;
use crate::conveyor::ring_buffer::ConveyorRingBuffer;
use crate::conveyor::stages::{
    ContractionStation, DiffusionStation, EgressStation, PrefetchStation,
};

/// Configuration parameters for the conveyor belt pipeline.
#[derive(Clone, Debug)]
pub struct ConveyorConfig {
    /// Number of stages in the circular ring buffer (default: 4).
    pub num_stages: usize,
    /// Batch chunk size (number of nodes processed per conveyor slot).
    pub chunk_size: usize,
}

impl Default for ConveyorConfig {
    fn default() -> Self {
        Self {
            num_stages: 4,
            chunk_size: 16,
        }
    }
}

/// The Conveyor Belt Pipeline Engine for Cellular Sheaf Diffusion.
pub struct ConveyorPipeline {
    config: ConveyorConfig,
}

impl ConveyorPipeline {
    /// Creates a new conveyor pipeline with specified configuration.
    pub fn new(config: ConveyorConfig) -> Self {
        Self { config }
    }

    /// Executes multi-layer sheaf diffusion using the dynamic conveyor belt.
    pub fn execute_dynamic(
        &self,
        graph: &BlockSparseSheafGraph,
        x_initial: &[f32],
        channels: usize,
        tau: f32,
        num_layers: usize,
    ) -> (Vec<f32>, Vec<SheafTelemetry>) {
        let d = graph.stalk_dim;
        let num_nodes = graph.num_nodes;
        let stride = d * channels;
        let total_elements = num_nodes * stride;

        assert_eq!(x_initial.len(), total_elements);

        let mut buffer_a = x_initial.to_vec();
        let mut buffer_b = vec![0.0f32; total_elements];

        let mut telemetry_history = Vec::with_capacity(num_layers);

        // Pre-allocate circular ring buffer with capacity for chunk_size * stride elements
        let slot_capacity = self.config.chunk_size * stride;
        let mut ring_buffer = ConveyorRingBuffer::<f32>::new(self.config.num_stages, slot_capacity);

        for layer in 0..num_layers {
            let (src, dst) = if layer % 2 == 0 {
                (&buffer_a, &mut buffer_b)
            } else {
                (&buffer_b, &mut buffer_a)
            };

            let mut layer_dirichlet = 0.0f32;
            ring_buffer.reset();

            // Conveyor Belt Execution across node chunks
            let node_chunks: Vec<Vec<usize>> = (0..num_nodes)
                .collect::<Vec<_>>()
                .chunks(self.config.chunk_size)
                .map(|c| c.to_vec())
                .collect();

            let num_chunks = node_chunks.len();
            let mut chunk_idx = 0;

            while chunk_idx < num_chunks || ring_buffer.can_consume() {
                // Producer Stage: Feed conveyor belt until stages are full
                while chunk_idx < num_chunks && ring_buffer.can_produce() {
                    if let Some((slot_idx, slot)) = ring_buffer.acquire_producer_slot() {
                        slot.node_indices = node_chunks[chunk_idx].clone();
                        ring_buffer.commit_produced(slot_idx);
                        chunk_idx += 1;
                    } else {
                        break;
                    }
                }

                // Consumer Stage: Drain conveyor belt and process through stations
                while ring_buffer.can_consume() {
                    if let Some((slot_idx, slot)) = ring_buffer.acquire_consumer_slot() {
                        let active_nodes = slot.node_indices.clone();

                        // Station 0: Prefetch & Initialize Assembly States
                        let mut assembly_states = PrefetchStation::execute(&active_nodes, stride);

                        // Station 1: In-Register Contraction
                        for state in &mut assembly_states {
                            ContractionStation::execute_dynamic(state, graph, src, d, channels);
                        }

                        // Station 2: Diffusion Update
                        for state in &mut assembly_states {
                            DiffusionStation::execute(state, graph, src, dst, tau, stride);
                        }

                        // Station 3: Egress & Telemetry Summation
                        layer_dirichlet += EgressStation::execute(&assembly_states);

                        ring_buffer.commit_consumed(slot_idx);
                    } else {
                        break;
                    }
                }
            }

            let is_unstable = layer_dirichlet.is_nan() || layer_dirichlet.is_infinite();
            let is_collapsed = !is_unstable && layer_dirichlet < 1e-7;

            telemetry_history.push(SheafTelemetry {
                dirichlet_energy: layer_dirichlet,
                max_orthogonality_drift: 0.0,
                is_unstable,
                is_collapsed,
            });
        }

        let final_result = if num_layers % 2 == 0 {
            buffer_a
        } else {
            buffer_b
        };

        (final_result, telemetry_history)
    }

    /// Executes multi-layer sheaf diffusion using an atomically hot-swappable KernelSlot.
    pub fn execute_slot(
        &self,
        slot: &KernelSlot,
        graph: &BlockSparseSheafGraph,
        x_initial: &[f32],
        channels: usize,
        tau: f32,
        num_layers: usize,
    ) -> (Vec<f32>, Vec<SheafTelemetry>) {
        let d = graph.stalk_dim;
        let num_nodes = graph.num_nodes;
        let stride = d * channels;
        let total_elements = num_nodes * stride;

        assert_eq!(x_initial.len(), total_elements);

        let mut buffer_a = x_initial.to_vec();
        let mut buffer_b = vec![0.0f32; total_elements];

        let mut telemetry_history = Vec::with_capacity(num_layers);
        let slot_capacity = self.config.chunk_size * stride;
        let mut ring_buffer = ConveyorRingBuffer::<f32>::new(self.config.num_stages, slot_capacity);

        // Fast-path lock-free function pointer load
        let specialized_fn = slot.load_fn();

        for layer in 0..num_layers {
            let (src, dst) = if layer % 2 == 0 {
                (&buffer_a, &mut buffer_b)
            } else {
                (&buffer_b, &mut buffer_a)
            };

            let mut layer_dirichlet = 0.0f32;
            ring_buffer.reset();

            let node_chunks: Vec<Vec<usize>> = (0..num_nodes)
                .collect::<Vec<_>>()
                .chunks(self.config.chunk_size)
                .map(|c| c.to_vec())
                .collect();

            let num_chunks = node_chunks.len();
            let mut chunk_idx = 0;

            while chunk_idx < num_chunks || ring_buffer.can_consume() {
                while chunk_idx < num_chunks && ring_buffer.can_produce() {
                    if let Some((slot_idx, slot_item)) = ring_buffer.acquire_producer_slot() {
                        slot_item.node_indices = node_chunks[chunk_idx].clone();
                        ring_buffer.commit_produced(slot_idx);
                        chunk_idx += 1;
                    } else {
                        break;
                    }
                }

                while ring_buffer.can_consume() {
                    if let Some((slot_idx, slot_item)) = ring_buffer.acquire_consumer_slot() {
                        let active_nodes = slot_item.node_indices.clone();
                        let mut assembly_states = PrefetchStation::execute(&active_nodes, stride);

                        // Station 1: Specialized or Dynamic Contraction
                        if let Some(func) = specialized_fn {
                            for state in &mut assembly_states {
                                func(state, graph, src);
                            }
                        } else {
                            for state in &mut assembly_states {
                                ContractionStation::execute_dynamic(state, graph, src, d, channels);
                            }
                        }

                        // Station 2: Diffusion Update
                        for state in &mut assembly_states {
                            DiffusionStation::execute(state, graph, src, dst, tau, stride);
                        }

                        layer_dirichlet += EgressStation::execute(&assembly_states);
                        ring_buffer.commit_consumed(slot_idx);
                    } else {
                        break;
                    }
                }
            }

            let is_unstable = layer_dirichlet.is_nan() || layer_dirichlet.is_infinite();
            let is_collapsed = !is_unstable && layer_dirichlet < 1e-7;

            telemetry_history.push(SheafTelemetry {
                dirichlet_energy: layer_dirichlet,
                max_orthogonality_drift: 0.0,
                is_unstable,
                is_collapsed,
            });
        }

        let final_result = if num_layers % 2 == 0 {
            buffer_a
        } else {
            buffer_b
        };

        (final_result, telemetry_history)
    }

    /// Executes multi-layer sheaf diffusion using const-generic metaprogrammed conveyor stations.
    pub fn execute_const<const D: usize, const C: usize>(
        &self,
        graph: &BlockSparseSheafGraph,
        x_initial: &[f32],
        tau: f32,
        num_layers: usize,
    ) -> (Vec<f32>, Vec<SheafTelemetry>) {
        assert_eq!(graph.stalk_dim, D);
        let num_nodes = graph.num_nodes;
        let stride = D * C;
        let total_elements = num_nodes * stride;

        assert_eq!(x_initial.len(), total_elements);

        let mut buffer_a = x_initial.to_vec();
        let mut buffer_b = vec![0.0f32; total_elements];

        let mut telemetry_history = Vec::with_capacity(num_layers);

        let slot_capacity = self.config.chunk_size * stride;
        let mut ring_buffer = ConveyorRingBuffer::<f32>::new(self.config.num_stages, slot_capacity);

        for layer in 0..num_layers {
            let (src, dst) = if layer % 2 == 0 {
                (&buffer_a, &mut buffer_b)
            } else {
                (&buffer_b, &mut buffer_a)
            };

            let mut layer_dirichlet = 0.0f32;
            ring_buffer.reset();

            let node_chunks: Vec<Vec<usize>> = (0..num_nodes)
                .collect::<Vec<_>>()
                .chunks(self.config.chunk_size)
                .map(|c| c.to_vec())
                .collect();

            let num_chunks = node_chunks.len();
            let mut chunk_idx = 0;

            while chunk_idx < num_chunks || ring_buffer.can_consume() {
                // Producer Stage
                while chunk_idx < num_chunks && ring_buffer.can_produce() {
                    if let Some((slot_idx, slot)) = ring_buffer.acquire_producer_slot() {
                        slot.node_indices = node_chunks[chunk_idx].clone();
                        ring_buffer.commit_produced(slot_idx);
                        chunk_idx += 1;
                    } else {
                        break;
                    }
                }

                // Consumer Stage with Const-Generic Unrolled Stations
                while ring_buffer.can_consume() {
                    if let Some((slot_idx, slot)) = ring_buffer.acquire_consumer_slot() {
                        let active_nodes = slot.node_indices.clone();

                        let mut assembly_states = PrefetchStation::execute(&active_nodes, stride);

                        for state in &mut assembly_states {
                            ContractionStation::execute_const::<D, C>(state, graph, src);
                        }

                        for state in &mut assembly_states {
                            DiffusionStation::execute(state, graph, src, dst, tau, stride);
                        }

                        layer_dirichlet += EgressStation::execute(&assembly_states);

                        ring_buffer.commit_consumed(slot_idx);
                    } else {
                        break;
                    }
                }
            }

            let is_unstable = layer_dirichlet.is_nan() || layer_dirichlet.is_infinite();
            let is_collapsed = !is_unstable && layer_dirichlet < 1e-7;

            telemetry_history.push(SheafTelemetry {
                dirichlet_energy: layer_dirichlet,
                max_orthogonality_drift: 0.0,
                is_unstable,
                is_collapsed,
            });
        }

        let final_result = if num_layers % 2 == 0 {
            buffer_a
        } else {
            buffer_b
        };

        (final_result, telemetry_history)
    }
}
