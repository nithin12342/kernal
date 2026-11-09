// ==============================================================================
// LAYER 5: CUDA C++ - ULTRA SHEAF CONVEYOR SILICON KERNEL
// Silicon Target: NVIDIA Tesla T4 (sm_75) / Compute Capability >= 7.5
// Optimized: 128-bit Vector Load & In-Register Warp-Broadcast
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define STALK_DIM 4
#define CHANNELS 32
#define STRIDE (STALK_DIM * CHANNELS)       // 128 elements per node
#define RHO_STRIDE (STALK_DIM * STALK_DIM)  // 16 elements per edge
#define WARPS_PER_BLOCK 4                   // 128 threads / 32 = 4 warps

__global__ void sheaf_ultra_kernel(
    const int num_nodes,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ rho_data,
    const float* __restrict__ x_src,
    float* __restrict__ x_dst,
    const float tau,
    float* __restrict__ dirichlet_energy_out
) {
    int node = blockIdx.x;
    if (node >= num_nodes) return;

    int tid = threadIdx.x;
    int d_idx = tid / CHANNELS;
    int c_idx = tid % CHANNELS;

    int start_edge = row_ptr[node];
    int end_edge = row_ptr[node + 1];
    int degree = end_edge - start_edge;
    float decay = 1.0f - tau * (float)degree;

    int node_offset = node * STRIDE;
    float self_val = __ldg(&x_src[node_offset + tid]);

    float acc = 0.0f;
    float local_dirichlet = 0.0f;
    const bool calc_dirichlet = (dirichlet_energy_out != nullptr);
    int lane_id = tid % 32;
    const float4* rho_vec = reinterpret_cast<const float4*>(rho_data);

    #pragma unroll 4
    for (int e = start_edge; e < end_edge; ++e) {
        int neighbor = __ldg(&col_ind[e]);
        int neighbor_offset = neighbor * STRIDE;

        // In-Register Warp-Broadcast: Lane 0 loads 128-bit vector, broadcasts via register shuffle
        float r0, r1, r2, r3;
        if (lane_id == 0) {
            float4 r = __ldg(&rho_vec[e * 4 + d_idx]);
            r0 = r.x; r1 = r.y; r2 = r.z; r3 = r.w;
        }
        r0 = __shfl_sync(0xFFFFFFFF, r0, 0);
        r1 = __shfl_sync(0xFFFFFFFF, r1, 0);
        r2 = __shfl_sync(0xFFFFFFFF, r2, 0);
        r3 = __shfl_sync(0xFFFFFFFF, r3, 0);

        // Coalesced 128-byte burst reads across 32 channels
        float u0 = __ldg(&x_src[neighbor_offset + 0 * 32 + c_idx]);
        float u1 = __ldg(&x_src[neighbor_offset + 1 * 32 + c_idx]);
        float u2 = __ldg(&x_src[neighbor_offset + 2 * 32 + c_idx]);
        float u3 = __ldg(&x_src[neighbor_offset + 3 * 32 + c_idx]);

        float contracted = fmaf(r0, u0, fmaf(r1, u1, fmaf(r2, u2, r3 * u3)));
        acc += contracted;

        if (calc_dirichlet) {
            float diff = contracted - self_val;
            local_dirichlet += 0.5f * diff * diff;
        }
    }

    x_dst[node_offset + tid] = decay * self_val + tau * acc;

    if (calc_dirichlet) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_dirichlet += __shfl_down_sync(0xFFFFFFFF, local_dirichlet, offset);
        }

        __shared__ float warp_dirichlet[WARPS_PER_BLOCK];
        int warp_id = tid / 32;

        if (lane_id == 0) {
            warp_dirichlet[warp_id] = local_dirichlet;
        }
        __syncthreads();

        if (warp_id == 0) {
            float block_sum = (lane_id < WARPS_PER_BLOCK) ? warp_dirichlet[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2) {
                block_sum += __shfl_down_sync(0x0000000F, block_sum, offset);
            }
            if (lane_id == 0) {
                atomicAdd(dirichlet_energy_out, block_sum);
            }
        }
    }
}

struct GpuSheafContext {
    int num_nodes;
    int num_edges;
    int stalk_dim;
    int channels;
    int total_elements;
    int *d_row_ptr;
    int *d_col_ind;
    float *d_rho;
    float *d_x_a;
    float *d_x_b;
    float *d_dirichlet;
    cudaStream_t stream;
    cudaEvent_t start_ev;
    cudaEvent_t stop_ev;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool graph_instantiated;
    size_t allocated_vram_bytes;
};

extern "C" {

int init_sheaf_gpu_vram(
    int num_nodes,
    int num_edges,
    int stalk_dim,
    int channels,
    const int* h_row_ptr,
    const int* h_col_ind,
    const float* h_rho,
    const float* h_x_init,
    void** out_handle,
    size_t* out_vram_bytes
) {
    GpuSheafContext* ctx = new GpuSheafContext();
    ctx->num_nodes = num_nodes;
    ctx->num_edges = num_edges;
    ctx->stalk_dim = stalk_dim;
    ctx->channels = channels;
    ctx->total_elements = num_nodes * stalk_dim * channels;
    ctx->graph = nullptr;
    ctx->graph_exec = nullptr;
    ctx->graph_instantiated = false;

    size_t row_ptr_sz = (num_nodes + 1) * sizeof(int);
    size_t col_ind_sz = num_edges * sizeof(int);
    size_t rho_sz = num_edges * stalk_dim * stalk_dim * sizeof(float);
    size_t x_sz = ctx->total_elements * sizeof(float);

    ctx->allocated_vram_bytes = row_ptr_sz + col_ind_sz + rho_sz + (2 * x_sz) + sizeof(float);

    cudaMalloc(&ctx->d_row_ptr, row_ptr_sz);
    cudaMalloc(&ctx->d_col_ind, col_ind_sz);
    cudaMalloc(&ctx->d_rho, rho_sz);
    cudaMalloc(&ctx->d_x_a, x_sz);
    cudaMalloc(&ctx->d_x_b, x_sz);
    cudaMalloc(&ctx->d_dirichlet, sizeof(float));

    cudaMemcpy(ctx->d_row_ptr, h_row_ptr, row_ptr_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_col_ind, h_col_ind, col_ind_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_rho, h_rho, rho_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_x_a, h_x_init, x_sz, cudaMemcpyHostToDevice);

    cudaStreamCreate(&ctx->stream);
    cudaEventCreate(&ctx->start_ev);
    cudaEventCreate(&ctx->stop_ev);

    *out_handle = (void*)ctx;
    *out_vram_bytes = ctx->allocated_vram_bytes;
    return 0;
}

int reset_sheaf_input(void* handle, const float* h_x_init) {
    if (!handle) return -1;
    GpuSheafContext* ctx = (GpuSheafContext*)handle;
    cudaMemcpy(ctx->d_x_a, h_x_init, ctx->total_elements * sizeof(float), cudaMemcpyHostToDevice);
    return 0;
}

int run_sheaf_conveyor_gpu_steps(
    void* handle,
    int num_layers,
    float tau,
    float* out_time_ms,
    float* out_dirichlet_energy
) {
    if (!handle) return -1;
    GpuSheafContext* ctx = (GpuSheafContext*)handle;

    cudaMemsetAsync(ctx->d_dirichlet, 0, sizeof(float), ctx->stream);
    cudaEventRecord(ctx->start_ev, ctx->stream);

    for (int l = 0; l < num_layers; ++l) {
        float* src = (l % 2 == 0) ? ctx->d_x_a : ctx->d_x_b;
        float* dst = (l % 2 == 0) ? ctx->d_x_b : ctx->d_x_a;
        float* dir_target = (l == num_layers - 1) ? ctx->d_dirichlet : nullptr;

        sheaf_ultra_kernel<<<ctx->num_nodes, STRIDE, 0, ctx->stream>>>(
            ctx->num_nodes,
            ctx->d_row_ptr,
            ctx->d_col_ind,
            ctx->d_rho,
            src,
            dst,
            tau,
            dir_target
        );
    }

    cudaEventRecord(ctx->stop_ev, ctx->stream);
    cudaEventSynchronize(ctx->stop_ev);

    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, ctx->start_ev, ctx->stop_ev);
    *out_time_ms = elapsed;

    float dirichlet_val = 0.0f;
    cudaMemcpy(&dirichlet_val, ctx->d_dirichlet, sizeof(float), cudaMemcpyDeviceToHost);
    *out_dirichlet_energy = dirichlet_val;

    return 0;
}

int instantiate_sheaf_cuda_graph(void* handle, int num_layers, float tau) {
    if (!handle) return -1;
    GpuSheafContext* ctx = (GpuSheafContext*)handle;

    if (ctx->graph_instantiated) {
        cudaGraphExecDestroy(ctx->graph_exec);
        cudaGraphDestroy(ctx->graph);
        ctx->graph_instantiated = false;
    }

    cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeGlobal);
    cudaMemsetAsync(ctx->d_dirichlet, 0, sizeof(float), ctx->stream);

    for (int l = 0; l < num_layers; ++l) {
        float* src = (l % 2 == 0) ? ctx->d_x_a : ctx->d_x_b;
        float* dst = (l % 2 == 0) ? ctx->d_x_b : ctx->d_x_a;
        float* dir_target = (l == num_layers - 1) ? ctx->d_dirichlet : nullptr;

        sheaf_ultra_kernel<<<ctx->num_nodes, STRIDE, 0, ctx->stream>>>(
            ctx->num_nodes,
            ctx->d_row_ptr,
            ctx->d_col_ind,
            ctx->d_rho,
            src,
            dst,
            tau,
            dir_target
        );
    }

    cudaStreamEndCapture(ctx->stream, &ctx->graph);
    cudaGraphInstantiate(&ctx->graph_exec, ctx->graph, nullptr, nullptr, 0);
    ctx->graph_instantiated = true;
    return 0;
}

int run_sheaf_cuda_graph(
    void* handle,
    float* out_time_ms,
    float* out_dirichlet_energy
) {
    if (!handle) return -1;
    GpuSheafContext* ctx = (GpuSheafContext*)handle;
    if (!ctx->graph_instantiated) return -2;

    cudaEventRecord(ctx->start_ev, ctx->stream);
    cudaGraphLaunch(ctx->graph_exec, ctx->stream);
    cudaEventRecord(ctx->stop_ev, ctx->stream);
    cudaEventSynchronize(ctx->stop_ev);

    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, ctx->start_ev, ctx->stop_ev);
    *out_time_ms = elapsed;

    float dirichlet_val = 0.0f;
    cudaMemcpy(&dirichlet_val, ctx->d_dirichlet, sizeof(float), cudaMemcpyDeviceToHost);
    *out_dirichlet_energy = dirichlet_val;
    return 0;
}

int read_sheaf_gpu_results(void* handle, float* h_out_dest, int num_layers) {
    if (!handle) return -1;
    GpuSheafContext* ctx = (GpuSheafContext*)handle;
    float* final_d = (num_layers % 2 == 0) ? ctx->d_x_a : ctx->d_x_b;
    cudaMemcpy(h_out_dest, final_d, ctx->total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}

int free_sheaf_gpu_vram(void* handle) {
    if (!handle) return -1;
    GpuSheafContext* ctx = (GpuSheafContext*)handle;
    if (ctx->graph_instantiated) {
        cudaGraphExecDestroy(ctx->graph_exec);
        cudaGraphDestroy(ctx->graph);
    }
    cudaFree(ctx->d_row_ptr); cudaFree(ctx->d_col_ind); cudaFree(ctx->d_rho);
    cudaFree(ctx->d_x_a); cudaFree(ctx->d_x_b); cudaFree(ctx->d_dirichlet);
    cudaStreamDestroy(ctx->stream);
    cudaEventDestroy(ctx->start_ev);
    cudaEventDestroy(ctx->stop_ev);
    delete ctx;
    return 0;
}

int get_gpu_vram_info(size_t* free_bytes, size_t* total_bytes) {
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    return (err == cudaSuccess) ? 0 : -1;
}

}
