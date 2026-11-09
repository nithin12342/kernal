// ==============================================================================
// LAYER 5: CUDA C++ - ULTRA MULTI-PRECISION SHEAF CONVEYOR SILICON KERNEL
// Silicon Target: NVIDIA Tesla T4 (sm_75) / Compute Capability >= 7.5
// Supports: FP32, FP16, BF16, FP8 (E4M3), FP4 (E2M1)
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "sheaf_precision.cuh"

#define STALK_DIM 4
#define CHANNELS 32
#define STRIDE (STALK_DIM * CHANNELS)       // 128 elements per node
#define RHO_STRIDE (STALK_DIM * STALK_DIM)  // 16 elements per edge
#define WARPS_PER_BLOCK 4                   // 128 threads / 32 = 4 warps

template<SheafPrecisionMode PRECISION, bool CALC_DIRICHLET>
__global__ __launch_bounds__(STRIDE, 8) void sheaf_ultra_kernel_prec(
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
    const float4* rho_vec = reinterpret_cast<const float4*>(rho_data);

    #pragma unroll 4
    for (int e = start_edge; e < end_edge; ++e) {
        int neighbor = __ldg(&col_ind[e]);
        int neighbor_offset = neighbor * STRIDE;

        // Uniform 128-bit vector load: broadcast by hardware uniform datapath in 1 cycle
        const float4 r = __ldg(&rho_vec[e * 4 + d_idx]);

        // Coalesced 128-byte burst reads across 32 channels
        float u0 = __ldg(&x_src[neighbor_offset + 0 * 32 + c_idx]);
        float u1 = __ldg(&x_src[neighbor_offset + 1 * 32 + c_idx]);
        float u2 = __ldg(&x_src[neighbor_offset + 2 * 32 + c_idx]);
        float u3 = __ldg(&x_src[neighbor_offset + 3 * 32 + c_idx]);

        // Compile-time precision specialization for tensor state reads
        if (PRECISION == PRECISION_FP16) {
            u0 = half_to_float(float_to_half(u0));
            u1 = half_to_float(float_to_half(u1));
            u2 = half_to_float(float_to_half(u2));
            u3 = half_to_float(float_to_half(u3));
        } else if (PRECISION == PRECISION_BF16) {
            u0 = bfloat16_to_float(float_to_bfloat16(u0));
            u1 = bfloat16_to_float(float_to_bfloat16(u1));
            u2 = bfloat16_to_float(float_to_bfloat16(u2));
            u3 = bfloat16_to_float(float_to_bfloat16(u3));
        } else if (PRECISION == PRECISION_FP8) {
            u0 = fp8_e4m3_to_float(float_to_fp8_e4m3(u0));
            u1 = fp8_e4m3_to_float(float_to_fp8_e4m3(u1));
            u2 = fp8_e4m3_to_float(float_to_fp8_e4m3(u2));
            u3 = fp8_e4m3_to_float(float_to_fp8_e4m3(u3));
        } else if (PRECISION == PRECISION_FP4) {
            u0 = fp4_nibble_to_float(float_to_fp4_nibble(u0));
            u1 = fp4_nibble_to_float(float_to_fp4_nibble(u1));
            u2 = fp4_nibble_to_float(float_to_fp4_nibble(u2));
            u3 = fp4_nibble_to_float(float_to_fp4_nibble(u3));
        }

        float contracted = fmaf(r.x, u0, fmaf(r.y, u1, fmaf(r.z, u2, r.w * u3)));
        acc += contracted;

        if (CALC_DIRICHLET) {
            float diff = contracted - self_val;
            local_dirichlet += 0.5f * diff * diff;
        }
    }

    float next_val = decay * self_val + tau * acc;
    if (PRECISION == PRECISION_FP16) {
        next_val = half_to_float(float_to_half(next_val));
    } else if (PRECISION == PRECISION_BF16) {
        next_val = bfloat16_to_float(float_to_bfloat16(next_val));
    } else if (PRECISION == PRECISION_FP8) {
        next_val = fp8_e4m3_to_float(float_to_fp8_e4m3(next_val));
    } else if (PRECISION == PRECISION_FP4) {
        next_val = fp4_nibble_to_float(float_to_fp4_nibble(next_val));
    }
    x_dst[node_offset + tid] = next_val;

    if (CALC_DIRICHLET) {
        int lane_id = tid % 32;
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
            for (int offset = 16; offset > 0; offset /= 2) {
                block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
            }
            if (lane_id == 0) {
                atomicAdd(dirichlet_energy_out, block_sum);
            }
        }
    }
}

template<bool CALC_DIRICHLET>
inline void launch_sheaf_dispatch(
    int prec, int nodes, int stride, cudaStream_t stream,
    int num_nodes, const int* row_ptr, const int* col_ind, const float* rho_data,
    const float* src, float* dst, float tau, float* dirichlet
) {
    if (prec == PRECISION_FP16) {
        sheaf_ultra_kernel_prec<PRECISION_FP16, CALC_DIRICHLET><<<nodes, stride, 0, stream>>>(
            num_nodes, row_ptr, col_ind, rho_data, src, dst, tau, dirichlet);
    } else if (prec == PRECISION_BF16) {
        sheaf_ultra_kernel_prec<PRECISION_BF16, CALC_DIRICHLET><<<nodes, stride, 0, stream>>>(
            num_nodes, row_ptr, col_ind, rho_data, src, dst, tau, dirichlet);
    } else if (prec == PRECISION_FP8) {
        sheaf_ultra_kernel_prec<PRECISION_FP8, CALC_DIRICHLET><<<nodes, stride, 0, stream>>>(
            num_nodes, row_ptr, col_ind, rho_data, src, dst, tau, dirichlet);
    } else if (prec == PRECISION_FP4) {
        sheaf_ultra_kernel_prec<PRECISION_FP4, CALC_DIRICHLET><<<nodes, stride, 0, stream>>>(
            num_nodes, row_ptr, col_ind, rho_data, src, dst, tau, dirichlet);
    } else {
        sheaf_ultra_kernel_prec<PRECISION_FP32, CALC_DIRICHLET><<<nodes, stride, 0, stream>>>(
            num_nodes, row_ptr, col_ind, rho_data, src, dst, tau, dirichlet);
    }
}

struct GpuSheafContext {
    int num_nodes;
    int num_edges;
    int stalk_dim;
    int channels;
    int total_elements;
    int precision_mode;
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
    ctx->precision_mode = PRECISION_FP32;
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

int run_sheaf_conveyor_gpu_steps_precision(
    void* handle,
    int num_layers,
    float tau,
    int precision_mode,
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

        if (l == num_layers - 1) {
            launch_sheaf_dispatch<true>(
                precision_mode, ctx->num_nodes, STRIDE, ctx->stream,
                ctx->num_nodes, ctx->d_row_ptr, ctx->d_col_ind, ctx->d_rho,
                src, dst, tau, ctx->d_dirichlet
            );
        } else {
            launch_sheaf_dispatch<false>(
                precision_mode, ctx->num_nodes, STRIDE, ctx->stream,
                ctx->num_nodes, ctx->d_row_ptr, ctx->d_col_ind, ctx->d_rho,
                src, dst, tau, nullptr
            );
        }
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

int run_sheaf_conveyor_gpu_steps(
    void* handle,
    int num_layers,
    float tau,
    float* out_time_ms,
    float* out_dirichlet_energy
) {
    return run_sheaf_conveyor_gpu_steps_precision(
        handle, num_layers, tau, PRECISION_FP32, out_time_ms, out_dirichlet_energy
    );
}

int instantiate_sheaf_cuda_graph_precision(
    void* handle,
    int num_layers,
    float tau,
    int precision_mode
) {
    if (!handle) return -1;
    GpuSheafContext* ctx = (GpuSheafContext*)handle;
    ctx->precision_mode = precision_mode;

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

        if (l == num_layers - 1) {
            launch_sheaf_dispatch<true>(
                precision_mode, ctx->num_nodes, STRIDE, ctx->stream,
                ctx->num_nodes, ctx->d_row_ptr, ctx->d_col_ind, ctx->d_rho,
                src, dst, tau, ctx->d_dirichlet
            );
        } else {
            launch_sheaf_dispatch<false>(
                precision_mode, ctx->num_nodes, STRIDE, ctx->stream,
                ctx->num_nodes, ctx->d_row_ptr, ctx->d_col_ind, ctx->d_rho,
                src, dst, tau, nullptr
            );
        }
    }

    cudaStreamEndCapture(ctx->stream, &ctx->graph);
    cudaGraphInstantiate(&ctx->graph_exec, ctx->graph, nullptr, nullptr, 0);
    ctx->graph_instantiated = true;
    return 0;
}

int instantiate_sheaf_cuda_graph(void* handle, int num_layers, float tau) {
    return instantiate_sheaf_cuda_graph_precision(handle, num_layers, tau, PRECISION_FP32);
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
