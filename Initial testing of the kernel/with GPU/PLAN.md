# Verification Plan: Silicon Execution, Profiling & Sanitization (With GPU)

This test suite requires a dedicated physical GPU environment (NVIDIA Ampere `sm_80`, Hopper `sm_90`, or Blackwell `sm_100`). It validates runtime hardware execution, IEEE-754 numerical tolerance, memory sanitization, hardware pipeline profiling, and multi-stream concurrency.

---

## 1. Test Matrix & Profiling Overview

| Category | Verification Target | Tooling | Threshold / Acceptance Gate |
| :--- | :--- | :--- | :--- |
| **Numerical** | FP Equivalence vs CPU Oracle | PyTorch / Python Test Harness | $\text{rtol} \le 10^{-5}$, $\text{atol} \le 10^{-6}$ |
| **Autograd** | Backward Adjoint $\frac{\partial \mathcal{L}}{\partial X}, \frac{\partial \mathcal{L}}{\partial \rho}$ | `torch.autograd.gradcheck` | Analytical vs Finite-Difference match |
| **Sanitizer** | Out-of-Bounds Memory Safety | `compute-sanitizer --tool memcheck` | 0 errors, valid `LDG.E.128` alignment |
| **Sanitizer** | Shared Memory Race Hazards | `compute-sanitizer --tool racecheck` | 0 RAW, 0 WAR, 0 WAW hazard violations |
| **Sanitizer** | Warp Barrier Synchronization | `compute-sanitizer --tool synccheck` | 0 divergent barrier or shuffle mask errors |
| **Sanitizer** | Uninitialized Memory Read | `compute-sanitizer --tool initcheck` | 0 uninitialized global/SRAM reads |
| **Profiling** | Dynamic Register Spilling | `ncu --metrics ...spill_loads` | `SPILL_LOADS == 0`, `SPILL_STORES == 0` |
| **Profiling** | Warp Branch Divergence | `ncu --metrics ...divergent.sum` | 0 divergent branch targets across warp |
| **Profiling** | Shared Memory 32-Bank Conflicts | `ncu --metrics ...bank_conflicts` | `l1tex...bank_conflicts.sum == 0` |
| **Profiling** | `cp.async` Latency Hiding | `ncu --metrics ...wait_barrier_pct` | Stall time $< 5\%$ |
| **Roofline** | Speed-of-Light (SOL) Throughput | NVIDIA Nsight Compute | Active SM $\ge 75\%$, Memory Bandwidth $\ge 80\%$ |
| **Runtime** | CUDA Graph Capture & Replay | CUDA Driver API / PyTorch | Bit-exact output replay, 0 driver faults |
| **Concurrency**| Multi-Stream Batch Execution | Multi-Stream Harness | Zero cross-stream serialization |
| **Interconnect**| Write-Combined PCIe DMA | Nsight Systems (`nsys`) | Peak PCIe Gen 4/5 throughput burst |
| **Storage P2P**| GPUDirect Storage (cuFile) | Magnum IO cuFile Benchmark | Zero-copy NVMe to VRAM DMA |
| **Stability** | 10,000-Launch Determinism | Dedicated Endurance Script | Bit-exact hash across 10,000 runs |

---

## 2. Detailed Hardware Verification Procedures

### 2.1 Forward Pass Floating-Point Equivalence
* **Procedure:** Run the compiled GPU kernel across synthetic and real-world graph topologies (e.g. Cora, Elliptic++). Compare output device tensors against the FP64 CPU Oracle.
* **Gate:** Mean and max absolute/relative errors must satisfy:
  $$\max |X_{\text{gpu}} - X_{\text{cpu}}| \le 10^{-6}, \quad \frac{|X_{\text{gpu}} - X_{\text{cpu}}|}{|X_{\text{cpu}}| + 10^{-8}} \le 10^{-5}$$

### 2.2 Numerical Finite-Difference Gradient Check
* **Procedure:** Execute `torch.autograd.gradcheck` and `gradgradcheck` using double-precision inputs over random micro-graphs ($|V|=16, |E|=32, d=4, c=4$).
* **Gate:** Validate that analytical backwards adjoint formulations match two-sided central differences:
  $$\frac{\partial \mathcal{L}}{\partial \theta_{\text{num}}} = \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

### 2.3 Out-of-Bounds Memory Safety (`memcheck`)
* **Procedure:** Run under NVIDIA Compute Sanitizer:
  ```bash
  compute-sanitizer --tool memcheck --leak-check full ./fbs_sde_gpu_runner
  ```
* **Gate:** Zero memory access violations, zero unaligned memory transactions, and proper 128-bit alignment for vector loads (`LDG.E.128`).

### 2.4 Shared Memory Race Hazards (`racecheck`)
* **Procedure:** Instrument `cp.async.ca` double-buffering rings:
  ```bash
  compute-sanitizer --tool racecheck ./fbs_sde_gpu_runner
  ```
* **Gate:** Verify zero Read-After-Write (RAW), Write-After-Read (WAR), or Write-After-Write (WAW) hazards between write stage $k+1$ and compute stage $k$.

### 2.5 Hardware Synchronization Verification (`synccheck`)
* **Procedure:** Audit warp shuffle and barrier instructions:
  ```bash
  compute-sanitizer --tool synccheck ./fbs_sde_gpu_runner
  ```
* **Gate:** Ensure all threads in active mask `0xffffffff` enter `__shfl_down_sync` synchronously and no threads participate in divergent `__syncthreads()`.

### 2.6 Uninitialized Device Memory Auditing (`initcheck`)
* **Procedure:** Verify memory sanitization:
  ```bash
  compute-sanitizer --tool initcheck ./fbs_sde_gpu_runner
  ```
* **Gate:** Zero reads from uninitialized global memory or uninitialized Shared Memory accumulation buffers.

### 2.7 Physical Register Spill Verification
* **Procedure:** Profile register allocations in hardware with Nsight Compute:
  ```bash
  ncu --metrics smsp__sass_lmem_bytes_spilled.sum,smsp__sass_lmem_bytes_reloaded.sum ./fbs_sde_gpu_runner
  ```
* **Gate:** `SPILL_LOADS == 0` and `SPILL_STORES == 0`. Register files must hold 100% of the active contraction tile.

### 2.8 Warp Branch Divergence Measurement
* **Procedure:** Measure instruction counters across all warp execution units:
  ```bash
  ncu --metrics smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_branch_targets_threads_nodivergent.sum ./fbs_sde_gpu_runner
  ```
* **Gate:** Zero divergent branch instructions (`divergent.sum == 0`) inside the primary edge accumulation pipeline.

### 2.9 Shared Memory 32-Bank Conflict Detection
* **Procedure:** Profile shared memory Load/Store Units (LSU):
  ```bash
  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu.sum ./fbs_sde_gpu_runner
  ```
* **Gate:** Zero bank conflicts (`conflicts_pipe_lsu.sum == 0`) across 32 threads accessing staged restriction maps $\rho$.

### 2.10 Hardware Asynchronous Latency-Hiding Efficiency
* **Procedure:** Measure warp issue stall cycles during `cp.async` wait groups:
  ```bash
  ncu --metrics smsp__warp_issue_stalled_wait_barrier_pct,smsp__warp_issue_stalled_short_scoreboard_pct ./fbs_sde_gpu_runner
  ```
* **Gate:** Barrier wait stalls must be $< 5\%$, proving compute stage $k$ completely hides stage $k+1$ HBM load latency.

### 2.11 Speed-of-Light (SOL) Hardware Roofline Utilization
* **Procedure:** Evaluate overall silicon hardware utilization:
  ```bash
  ncu --section SpeedOfLight ./fbs_sde_gpu_runner
  ```
* **Gate:** Achieve $\ge 75\%$ SM Active Throughput and $\ge 80\%$ Memory Bus Bandwidth Saturation on target silicon.

### 2.12 CUDA Graph Capture & Replay
* **Procedure:** Wrap multi-hop diffusion loops ($L=16$) inside CUDA Graph capture streams:
  ```cpp
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  launch_sheaf_diffusion(..., stream);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaGraphLaunch(instance, stream);
  ```
* **Gate:** Successful instantiation and replay without CPU driver overhead or illegal address faults.

### 2.13 Concurrent Multi-Stream Execution
* **Procedure:** Launch multiple independent subgraphs concurrently on separate non-default CUDA streams (`cudaStreamNonBlocking`).
* **Gate:** Nsight Systems timeline must show overlapping kernel bars without false serialization on stream 0.

### 2.14 Write-Combined PCIe Burst Behavior
* **Procedure:** Profile host-to-device streaming from `cudaHostAllocWriteCombined` buffers using Nsight Systems (`nsys profile`).
* **Gate:** Confirm DMA line buffers flush in continuous 128-byte transactions, achieving maximum theoretical PCIe bandwidth.

### 2.15 GPUDirect Storage (GDS / `cuFile`) P2P DMA
* **Procedure:** Benchmark direct NVMe block transfers into GPU VRAM using NVIDIA Magnum IO `cuFileRead`.
* **Gate:** Verify direct DMA transfers bypassing host DRAM with zero CPU memory staging.

### 2.16 Bit-Exact Determinism Stress Testing
* **Procedure:** Execute the kernel 10,000 consecutive times on fixed input tensors and graph topologies.
* **Gate:** Compute SHA-256 hash of final output tensor after each launch. All 10,000 hashes must be strictly identical, confirming zero non-deterministic race conditions.
