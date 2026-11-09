# Initial Testing of the Kernel: Fused Block-Sparse Sheaf Diffusion Engine (FBS-SDE)

This directory organizes the verification and validation framework for the **Fused Block-Sparse Sheaf Diffusion Engine (FBS-SDE)** into two distinct execution tiers:

---

## Directory Organization

```text
Initial testing of the kernel/
├── without GPU/
│   └── PLAN.md    # Pre-Silicon, Static CI, Mathematical Oracle & Algebra Proofs
├── with GPU/
│   └── PLAN.md    # Dedicated GPU Silicon Execution, Sanitization & Hardware Profiling
└── README.md      # Navigation & Strategy Overview
```

---

## 1. [Without GPU Verification Plan](file:///d:/kernal/Initial%20testing%20of%20the%20kernel/without%20GPU/PLAN.md)
**Focus:** Runs in standard headless CI/CD, local CPU environments, and pre-commit hooks.
* **Metaprogramming & Syntax:** AST and unrolled code structure verification.
* **Compiler Analysis:** Offline compilation (`nvcc -ptx`, `nvcc -c`), static register pressure audits (`lmem = 0 bytes`).
* **Mathematical Invariants:** FP64 numerical oracle, harmonic section preservation ($L_{\mathcal{F}} X = 0$), non-negative Dirichlet energy, $O(d)$ orthogonal equivariance.
* **Topology & Host Memory:** BSR index bounds validation, write-combining ring buffers, 128-byte memory alignment.
* **Static Quality:** Linters (`clippy`, `clang-tidy`), ABI bindings, dimension fuzzing ($d \in [2, 16], c \in [1, 128]$).

---

## 2. [With GPU Verification Plan](file:///d:/kernal/Initial%20testing%20of%20the%20kernel/with%20GPU/PLAN.md)
**Focus:** Requires physical GPU hardware (Ampere `sm_80`, Hopper `sm_90`, Blackwell `sm_100`).
* **Silicon Equivalence:** FP comparison vs CPU Oracle ($\text{rtol} \le 10^{-5}, \text{atol} \le 10^{-6}$).
* **Autograd Bridge:** Two-sided numerical finite-difference gradient checks (`gradcheck`, `gradgradcheck`).
* **Hardware Sanitizers (`compute-sanitizer`):** `memcheck` (memory boundaries), `racecheck` (RAW/WAR/WAW hazards), `synccheck` (warp masks), `initcheck` (uninitialized memory).
* **Hardware Profiling (`ncu` / `nsys`):** 0 register spills, 0 branch divergence, 0 shared memory bank conflicts, async latency hiding ($<5\%$ barrier stalls), and Speed-of-Light utilization ($\ge 75\%$ SM, $\ge 80\%$ memory bus).
* **Systems Integration:** CUDA Graph capture/replay, multi-stream concurrency, write-combined PCIe bursts, GPUDirect Storage (cuFile), and 10,000-run bit-exact determinism.
