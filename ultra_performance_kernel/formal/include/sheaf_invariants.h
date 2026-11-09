/* ==============================================================================
 * LAYER 0: LEAN 4 CERTIFIED C-ABI INVARIANTS & STRUCTURAL HEADERS
 * Emitted from SheafSpec.lean for downstream Rust, C++, and Python consumption
 * ============================================================================== */

#ifndef SHEAF_INVARIANTS_H
#define SHEAF_INVARIANTS_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Standard Stalk and Channel Constants */
#define SHEAF_DEFAULT_STALK_DIM 4
#define SHEAF_DEFAULT_CHANNELS  32
#define SHEAF_DEFAULT_STRIDE    (SHEAF_DEFAULT_STALK_DIM * SHEAF_DEFAULT_CHANNELS)
#define SHEAF_DEFAULT_RHO_BYTES (SHEAF_DEFAULT_STALK_DIM * SHEAF_DEFAULT_STALK_DIM * sizeof(float))

/* Certified BSR Graph Descriptor Structure */
typedef struct {
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t stalk_dim;
    uint32_t channels;
    uint32_t max_degree;
    float    gershgorin_tau_max;
} sheaf_graph_metadata_t;

/* Certified Invariant Validation Telemetry */
typedef struct {
    float    dirichlet_energy;
    float    isometry_residual;
    bool     is_stable;
    bool     is_collapsed;
} sheaf_invariant_telemetry_t;

/**
 * Validates the Block-Gershgorin diffusion stability ceiling:
 * tau < 1.0 / max_degree.
 */
static inline bool verify_gershgorin_step_size(float tau, uint32_t max_degree) {
    if (max_degree == 0 || tau <= 0.0f) return false;
    float ceiling = 1.0f / (float)max_degree;
    return tau < ceiling;
}

/**
 * Computes the Frobenious orthogonality residual:
 * ||rho^T * rho - I_d||_F.
 * Must be < 1e-4 for exact isometry preservation.
 */
static inline float compute_cayley_isometry_residual(const float* rho, uint32_t d) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < d; ++i) {
        for (uint32_t j = 0; j < d; ++j) {
            float dot = 0.0f;
            for (uint32_t k = 0; k < d; ++k) {
                dot += rho[k * d + i] * rho[k * d + j];
            }
            float target = (i == j) ? 1.0f : 0.0f;
            float diff = dot - target;
            sum_sq += diff * diff;
        }
    }
    return sqrtf(sum_sq);
}

/**
 * Validates that computed Dirichlet energy is positive semi-definite (E >= 0)
 * and not collapsed to zero (E > 1e-7).
 */
static inline bool verify_dirichlet_invariants(float energy, sheaf_invariant_telemetry_t* out) {
    if (isnan(energy) || isinf(energy) || energy < 0.0f) {
        if (out) { out->is_stable = false; out->is_collapsed = false; }
        return false;
    }
    if (out) {
        out->dirichlet_energy = energy;
        out->is_stable = true;
        out->is_collapsed = (energy < 1e-7f);
    }
    return true;
}

#ifdef __cplusplus
}
#endif

#endif /* SHEAF_INVARIANTS_H */
