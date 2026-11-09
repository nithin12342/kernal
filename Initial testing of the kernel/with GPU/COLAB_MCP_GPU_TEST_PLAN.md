# Detailed Colab MCP GPU Execution & Verification Plan

**Target Infrastructure:** Google Colab Cloud GPU (NVIDIA T4 / L4 / A100)  
**Control Interface:** Antigravity Agent via `colab-mcp` (WebSocket Session Proxy)  
**Target Kernels:** Rust Metaprogrammed (`ConstSheafEngine`), Rust Standard, and Mojo SIMD  
**Directory Path:** [`Initial testing of the kernel/with GPU/`](file:///d:/kernal/Initial%20testing%20of%20the%20kernel/with%20GPU/)

---

## 1. Architectural Overview & Colab MCP Bridge Mechanics

Because local testing is restricted to CPU execution, physical GPU verification is offloaded to a Google Colab cloud instance using the **Colab MCP server**. The MCP client programmatically constructs, uploads, executes, and parses test cells in a connected Colab notebook tab.

```text
+-----------------------+        WebSocket (127.0.0.1)        +--------------------------+
|  Antigravity Agent    | <=================================> |  Google Colab Frontend   |
|  (Local Client / MCP) |   open_colab_browser_connection     |  (Chrome / Browser Tab)  |
+-----------------------+                                     +-------------+------------+
           |                                                                |
           | MCP Tools: add_code_cell, run_code_cell, get_cells             | Cloud VM
           v                                                                v
+-----------------------+                                     +--------------------------+
| Test Harness Driver   |                                     | Google Cloud Compute VM  |
| colab_ctl.py / Tests  |                                     | (NVIDIA T4/L4/A100 GPU)  |
+-----------------------+                                     +--------------------------+
```

### 1.1 Colab MCP Protocol Operations
1. **`open_colab_browser_connection`**: Handshakes with the active browser session and unlocks cell operations.
2. **`add_code_cell(content)`**: Injects self-contained test scripts and benchmarking routines into the notebook.
3. **`run_code_cell(cellId)`**: Triggers remote execution on the attached GPU runtime.
4. **`get_cells(includeOutputs=True)`**: Retrieves raw stdout, stderr, execution counters, and exit codes.
5. **`update_cell(cellId, content)`**: Re-runs parameterized sweeps without cluttering the notebook canvas.

---

## 2. Phase 1: Cloud Environment Bootstrapping & Hardware Pre-Flight

Before executing kernel benchmarks, the agent injects a bootstrap cell to configure the headless Linux environment on the Colab VM.

### Cell 1.1: Hardware & CUDA Driver Verification
```bash
!nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
!nvcc --version
```
* **Acceptance Gate:** Confirms active NVIDIA GPU (T4 with 16GB VRAM, Compute Capability 7.5, or L4/A100 with CC 8.9/8.0).

### Cell 1.2: Host Toolchain & Compiler Installation
```bash
# Install Rust toolchain on the Colab VM
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
import os
os.environ["PATH"] += ":/root/.cargo/bin"
!cargo --version && rustc --version

# Verify PyTorch CUDA integration
import torch
assert torch.cuda.is_available(), "CUDA runtime not detected in PyTorch!"
print(f"PyTorch Device: {torch.cuda.get_device_name(0)}")
```
* **Acceptance Gate:** Rust 1.80+ and PyTorch CUDA bindings operational on the cloud VM.

### Cell 1.3: Injecting Kernel Source Code
The local agent base64-encodes or directly streams the repository files (`fbs_sde_rust` and test harnesses) into the Colab VM filesystem (`/content/fbs_sde/`).

---

## 3. Phase 2: Execution Matrix for the 16 GPU-Specific Tests

The agent executes the 16 tests in sequence, capturing execution metrics directly into local logs.

### Test 1: Forward Pass Floating-Point Equivalence
* **Action:** Launch the compiled GPU forward kernel against a 1,000-node random graph with $d=4, c=32, \tau=0.15$. Compare output against the FP64 CPU Oracle.
* **Metric:** Assert $\max |X_{\text{gpu}} - X_{\text{oracle}}| \le 10^{-6}$ and relative error $\le 10^{-5}$.

### Test 2: Numerical Finite-Difference Gradient Check
* **Action:** Run a PyTorch Autograd bridge script executing `torch.autograd.gradcheck`:
  ```python
  from torch.autograd import gradcheck
  test = gradcheck(fused_sheaf_diffusion, (x_in, rho_data), eps=1e-4, atol=1e-4)
  print(f"Gradcheck Passed: {test}")
  ```
* **Metric:** Two-sided perturbations must match analytical adjoint gradients $\frac{\partial \mathcal{L}}{\partial X}$ and $\frac{\partial \mathcal{L}}{\partial \rho}$.

### Test 3: Out-of-Bounds Memory Safety (`memcheck`)
* **Action:** Run NVIDIA Compute Sanitizer memory boundary checks:
  ```bash
  !compute-sanitizer --tool memcheck --leak-check full ./target/release/gpu_runner
  ```
* **Metric:** `0 errors reported`, 0 illegal memory accesses, and valid 128-bit vector loads (`LDG.E.128`).

### Test 4: Shared Memory Race Conditions (`racecheck`)
* **Action:** Check for Hazards on double-buffered circular memory:
  ```bash
  !compute-sanitizer --tool racecheck ./target/release/gpu_runner
  ```
* **Metric:** 0 Read-After-Write (RAW), Write-After-Read (WAR), or Write-After-Write (WAW) hazards between write stage $k+1$ and compute stage $k$.

### Test 5: Hardware Synchronization Verification (`synccheck`)
* **Action:** Check warp shuffle masks:
  ```bash
  !compute-sanitizer --tool synccheck ./target/release/gpu_runner
  ```
* **Metric:** 0 invalid lane masks in `__shfl_down_sync`, 0 divergent thread barriers.

### Test 6: Uninitialized Device Memory Auditing (`initcheck`)
* **Action:** Run uninitialized memory scan:
  ```bash
  !compute-sanitizer --tool initcheck ./target/release/gpu_runner
  ```
* **Metric:** 0 reads from uninitialized global memory or uninitialized Shared Memory accumulation buffers.

### Test 7: Physical Register Spill Verification
* **Action:** Profile local memory usage via Nsight Compute (`ncu`):
  ```bash
  !ncu --metrics smsp__sass_lmem_bytes_spilled.sum,smsp__sass_lmem_bytes_reloaded.sum ./target/release/gpu_runner
  ```
* **Metric:** `SPILL_LOADS == 0` and `SPILL_STORES == 0`. Register files hold 100% of the active tile.

### Test 8: Warp Branch Divergence Measurement
* **Action:** Profile SASS branch instructions across warp lanes:
  ```bash
  !ncu --metrics smsp__sass_branch_targets_threads_divergent.sum ./target/release/gpu_runner
  ```
* **Metric:** `smsp__sass_branch_targets_threads_divergent.sum == 0` (100% straight-line execution).

### Test 9: Shared Memory 32-Bank Conflict Detection
* **Action:** Profile Load/Store Units (LSU) across threads:
  ```bash
  !ncu --metrics l1tex__data_bank_conflicts_pipe_lsu.sum ./target/release/gpu_runner
  ```
* **Metric:** `l1tex__data_bank_conflicts_pipe_lsu.sum == 0` (zero 32-way shared memory bank conflicts).

### Test 10: Hardware Asynchronous Latency-Hiding Efficiency
* **Action:** Measure warp issue stall cycles during `cp.async` wait groups:
  ```bash
  !ncu --metrics smsp__warp_issue_stalled_wait_barrier_pct ./target/release/gpu_runner
  ```
* **Metric:** Barrier wait stalls $< 5\%$, proving compute stage $k$ completely hides stage $k+1$ HBM read latency.

### Test 11: Speed-of-Light (SOL) Hardware Utilization
* **Action:** Evaluate overall hardware efficiency against roofline limits:
  ```bash
  !ncu --section SpeedOfLight ./target/release/gpu_runner
  ```
* **Metric:** Achieve $\ge 75\%$ SM Active Throughput and $\ge 80\%$ Memory Bus Bandwidth Saturation.

### Test 12: CUDA Graph Capture & Replay
* **Action:** Capture a 16-hop diffusion iteration inside a CUDA stream graph and replay 500 times:
  ```python
  g = torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
      y = fused_sheaf_diffusion(x, rho)
  g.replay()
  ```
* **Metric:** Replay matches dynamic execution bit-for-bit with 0 host launch overhead.

### Test 13: Concurrent Multi-Stream Execution
* **Action:** Launch separate graph batches across multiple non-blocking CUDA streams (`torch.cuda.Stream()`).
* **Metric:** Nsight Systems timeline confirms overlapping kernel execution bars without stream serialization.

### Test 14: Write-Combined Host-to-Device Streaming Simulation
* **Action:** Benchmark pinned memory transfer speeds from write-combined host allocations into device HBM.
* **Metric:** Achieve $\ge 90\%$ of peak PCIe bandwidth (e.g. $\ge 14.5$ GB/s on PCIe Gen 3/4).

### Test 15: In-VRAM Multi-Hop Memory Locality (Traffic Profile)
* **Action:** Instrument total VRAM traffic across 8 diffusion layers using Nsight Compute:
  ```bash
  !ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./target/release/gpu_runner
  ```
* **Metric:** Verify that total multi-hop memory traffic does not exceed $\sim 1.64$ GB (an $11.1\times$ reduction over baseline streaming tapes).

### Test 16: 10,000-Launch Bit-Exact Determinism Stress Testing
* **Action:** Run 10,000 consecutive iterations on identical input tensors, calculating SHA-256 hash of device output:
  ```python
  import hashlib
  hashes = set()
  for _ in range(10000):
      out = kernel(x, rho)
      hashes.add(hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest())
  assert len(hashes) == 1, f"Non-deterministic execution detected: {len(hashes)} unique hashes!"
  print("Bit-Exact Determinism Confirmed across 10,000 runs.")
  ```
* **Metric:** Exactly 1 unique hash across all 10,000 executions, confirming zero race conditions.

---

## 4. Phase 3: Colab MCP Automation Driver Workflow

The local agent executes the entire plan via a dedicated Python automation script:

```bash
# 1. Initialize Colab MCP connection
python colab_ctl.py status

# 2. Open browser tab with proxy token (if not already connected)
python colab_ctl.py launch

# 3. Execute cloud GPU test runner
python Initial\ testing\ of\ the\ kernel/with\ GPU/run_colab_gpu_suite.py
```

### 4.1 Automated Result Collection & Artifact Archival
1. The driver pulls cell execution outputs via `get_cells(includeOutputs=True)`.
2. Raw profiler output files (`.ncu-rep`, `.nsys-rep`) are downloaded to `Initial testing of the kernel/with GPU/profiles/`.
3. The results are aggregated into `Initial testing of the kernel/with GPU/RESULT_REPORT.md` and committed to git file-by-file.

---

## 5. Contingency Protocols & Quota Management

1. **Compute Unit Exhaustion / Disconnect:**
   If the Colab session disconnects during long endurance runs (Test 16), the driver checkpoints progress every 1,000 iterations to Google Drive (`/content/drive/MyDrive/fbs_sde_checkpoints/`).
2. **Device Memory Overflows (OOM):**
   If testing large graphs ($|V| > 100\text{K}$), the batching parameter is dynamically partitioned using BSR graph sub-tiling.
3. **Timeout Mitigation:**
   Long compilation or `ncu` profiling steps are bounded by 180-second cell execution timeouts, issuing periodic progress heartbeats.
