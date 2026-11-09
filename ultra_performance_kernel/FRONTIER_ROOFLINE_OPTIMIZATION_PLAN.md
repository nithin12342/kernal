# Frontier Roofline Optimization Plan: In-Register Warp-Broadcast & Coalesced Vector Streaming

**Timestamp:** 2026-11-05T12:06:00+05:30  
**Scope:** Micro-architectural optimization of Layer 5 CUDA C++ and Layer 1 Metaprogramming to approach the theoretical hardware ceiling (~1.5–2.0 µs).

---

## 1. Architectural Strategy & Root Cause Analysis

In the golden baseline (7.61–8.14 µs/step), profiling revealed that while the 100-step CUDA Graph eliminated CPU launch overhead, the GPU inner loop suffered from **redundant memory transaction amplification**:
- In each 128-thread block (4 stalk dim $\times$ 32 channels), all 32 channels of stalk dimension $d$ read the identical restriction matrix row $(r_0, r_1, r_2, r_3)$ from global VRAM.
- This issued $128 \times 4 = 512$ memory instructions per edge.

### The In-Register Warp-Broadcast Optimization
1. **128-bit Vector Load (`LDG.E.128`):**
   Only lane 0 of each warp ($d_{\text{idx}} = \text{tid} / 32$) loads a `float4` from global memory.
2. **Zero-Overhead In-Register Broadcast (`__shfl_sync`):**
   Lane 0 broadcasts $(r_0, r_1, r_2, r_3)$ to all 31 worker lanes in 1 clock cycle using register shuffle.
3. **Reduction Ratio:**
   Reduces restriction memory transactions from 512 per edge down to **4 transactions per edge (128x transaction reduction)**.
4. **Zero Block Stalls:**
   Requires zero shared memory allocation and zero `__syncthreads()` barrier delays, operating at warp execution frequency.
5. **Fused Multiply-Add (`fmaf`):**
   Inner contraction $r_0 u_0 + r_1 u_1 + r_2 u_2 + r_3 u_3$ is emitted as 4 pipelined FMA instructions directly in registers.

---

## 2. Invariant & Safety Preservation

- **Public C-ABI Invariant:** `init_sheaf_gpu_vram`, `instantiate_sheaf_cuda_graph`, `run_sheaf_cuda_graph` signatures remain completely invariant.
- **Memory Safety:** Every index is bounded by compile-time static dimensions ($D=4, C=32$). Zero dynamic allocation or type decay.
- **Numerical Parity:** Bitwise equivalence with the CPU SIMD baseline and Lean 4 Dirichlet energy guarantees.
