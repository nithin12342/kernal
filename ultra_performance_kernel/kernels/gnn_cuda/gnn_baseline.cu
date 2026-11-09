// ==============================================================================
// LAYER 5: CUDA C++ - HOMOPHILIC GNN MESSAGE-PASSING BASELINE
// Silicon Target: NVIDIA Tesla T4 (sm_75)
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHANNELS 32

__global__ void gnn_baseline_forward_kernel(
    const int num_nodes,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ x_src,
    float* __restrict__ x_dst,
    const float self_weight,
    const float neighbor_weight
) {
    int node = blockIdx.x;
    if (node >= num_nodes) return;

    int tid = threadIdx.x; // 0..31 (one warp per node)
    if (tid >= CHANNELS) return;

    int start_edge = row_ptr[node];
    int end_edge = row_ptr[node + 1];
    int degree = end_edge - start_edge;
    float norm_factor = (degree > 0) ? (1.0f / sqrtf((float)degree)) : 1.0f;

    int node_offset = node * CHANNELS;
    float self_val = __ldg(&x_src[node_offset + tid]);

    float acc = 0.0f;
    for (int e = start_edge; e < end_edge; ++e) {
        int neighbor = __ldg(&col_ind[e]);
        int neighbor_offset = neighbor * CHANNELS;
        acc += __ldg(&x_src[neighbor_offset + tid]);
    }

    x_dst[node_offset + tid] = self_weight * self_val + neighbor_weight * norm_factor * acc;
}

extern "C" {

int run_gnn_baseline_step(
    int num_nodes,
    int num_edges,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_x_src,
    float* d_x_dst,
    float self_weight,
    float neighbor_weight,
    cudaStream_t stream
) {
    gnn_baseline_forward_kernel<<<num_nodes, CHANNELS, 0, stream>>>(
        num_nodes,
        d_row_ptr,
        d_col_ind,
        d_x_src,
        d_x_dst,
        self_weight,
        neighbor_weight
    );
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

}
