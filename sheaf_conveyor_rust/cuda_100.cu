// ==============================================================================
// BENCHMARK HARNESS: NATIVE CUDA + C++ 100-STEP SHEAF DIFFUSION BENCHMARK
// Target: NVIDIA Tesla T4 (sm_75) / Compute Capability >= 7.5
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define STALK_DIM 4
#define CHANNELS 32
#define STRIDE (STALK_DIM * CHANNELS)
#define RHO_STRIDE (STALK_DIM * STALK_DIM)

__global__ void sheaf_step_kernel(
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
    float self_val = x_src[node_offset + tid];

    float acc = 0.0f;
    float local_dirichlet = 0.0f;

    for (int e = start_edge; e < end_edge; ++e) {
        int neighbor = col_ind[e];
        int neighbor_offset = neighbor * STRIDE;
        const float* rho = &rho_data[e * RHO_STRIDE];

        float r0 = rho[d_idx * 4 + 0];
        float r1 = rho[d_idx * 4 + 1];
        float r2 = rho[d_idx * 4 + 2];
        float r3 = rho[d_idx * 4 + 3];

        float u0 = x_src[neighbor_offset + 0 * CHANNELS + c_idx];
        float u1 = x_src[neighbor_offset + 1 * CHANNELS + c_idx];
        float u2 = x_src[neighbor_offset + 2 * CHANNELS + c_idx];
        float u3 = x_src[neighbor_offset + 3 * CHANNELS + c_idx];

        float contracted = r0 * u0 + r1 * u1 + r2 * u2 + r3 * u3;
        acc += contracted;

        float diff = contracted - self_val;
        local_dirichlet += 0.5f * diff * diff;
    }

    x_dst[node_offset + tid] = decay * self_val + tau * acc;
    if (dirichlet_energy_out != nullptr) {
        atomicAdd(dirichlet_energy_out, local_dirichlet);
    }
}

int main() {
    const int num_nodes = 256;
    const int degree = 6;
    const int num_edges = num_nodes * degree;
    const int total_steps = 100;
    const float tau = 0.15f;

    std::vector<int> h_row_ptr(num_nodes + 1, 0);
    std::vector<int> h_col_ind(num_edges);
    for (int i = 0; i < num_nodes; ++i) {
        for (int d = 1; d <= degree; ++d) {
            h_col_ind[i * degree + (d - 1)] = (i + d) % num_nodes;
        }
        h_row_ptr[i + 1] = (i + 1) * degree;
    }

    std::vector<float> h_rho(num_edges * RHO_STRIDE, 0.0f);
    for (int e = 0; e < num_edges; ++e) {
        for (int i = 0; i < STALK_DIM; ++i) {
            for (int j = 0; j < STALK_DIM; ++j) {
                if (i == j) h_rho[e * RHO_STRIDE + i * STALK_DIM + j] = 1.0f;
                else if ((i + j + e) % 7 == 0) h_rho[e * RHO_STRIDE + i * STALK_DIM + j] = 0.05f;
            }
        }
    }

    const int total_elements = num_nodes * STRIDE;
    std::vector<float> h_x(total_elements);
    for (int i = 0; i < total_elements; ++i) {
        h_x[i] = std::sin((float)i * 0.017f);
    }

    int *d_row_ptr, *d_col_ind;
    float *d_rho, *d_x_a, *d_x_b, *d_dirichlet;
    cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, num_edges * sizeof(int));
    cudaMalloc(&d_rho, h_rho.size() * sizeof(float));
    cudaMalloc(&d_x_a, total_elements * sizeof(float));
    cudaMalloc(&d_x_b, total_elements * sizeof(float));
    cudaMalloc(&d_dirichlet, sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho.data(), h_rho.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_a, h_x.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_dirichlet, 0, sizeof(float));
    cudaEventRecord(start);

    for (int l = 0; l < total_steps; ++l) {
        float* src = (l % 2 == 0) ? d_x_a : d_x_b;
        float* dst = (l % 2 == 0) ? d_x_b : d_x_a;
        sheaf_step_kernel<<<num_nodes, STRIDE>>>(num_nodes, d_row_ptr, d_col_ind, d_rho, src, dst, tau, (l == total_steps - 1) ? d_dirichlet : nullptr);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    float final_energy = 0.0f;
    cudaMemcpy(&final_energy, d_dirichlet, sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> h_out(total_elements);
    float* final_d = (total_steps % 2 == 0) ? d_x_a : d_x_b;
    cudaMemcpy(h_out.data(), final_d, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUDA_TOTAL_MS=" << total_ms << "\n";
    std::cout << "CUDA_PER_STEP_MS=" << (total_ms / total_steps) << "\n";
    std::cout << "CUDA_DIRICHLET=" << final_energy << "\n";
    std::cout << "CUDA_OUT_0=" << h_out[0] << "\n";
    std::cout << "CUDA_OUT_1=" << h_out[1] << "\n";

    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_rho);
    cudaFree(d_x_a); cudaFree(d_x_b); cudaFree(d_dirichlet);
    return 0;
}
