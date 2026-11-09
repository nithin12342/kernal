# Verification Plan: CPU-Only, Static CI & Pre-Silicon (Without GPU)

This test suite executes entirely in CPU-only, containerized, or headless CI/CD environments. It validates mathematical invariants, code generation mechanics, host memory architectures, and compiler constraints before executing on physical silicon.

---

## 1. Test Matrix & Scope Overview

| Test Module | Primary Tooling | Validation Target | Success Criteria |
| :--- | :--- | :--- | :--- |
| **1.1 JIT Codegen Testing** | Rust `cargo test`, Python `pytest` | AST & String Builders | Zero dynamic loops, valid unrolled FMA syntax |
| **1.2 Offline Compilation** | `nvcc -ptx`, `nvcc -c`, `clang++` | Syntax & Template Expansion | Zero syntax errors, clean template instantiations |
| **1.3 Register Spilling** | `ptxas -v` parser | Static Register Footprint | `lmem = 0 bytes`, $\le 64$ registers/thread |
| **1.4 FP64 CPU Oracle** | NumPy / SciPy / Rust FP64 | Discretized Heat Equation | Analytical sheaf diffusion reference |
| **1.5 Algebraic Invariants** | Property-based testing (`proptest`) | Harmonicity & Equivariance | $L_{\mathcal{F}} X = 0$, $\mathcal{E}_{\mathcal{F}} \ge 0$, $O(d)$ equivariance |
| **1.6 BSR Graph Serialization**| Rust / Python unit tests | CSR/BSR Layout Validation | Valid monotonic offsets, zero out-of-bounds indices |
| **1.7 Host Ring Buffer** | Mock C/Rust Memory Allocators | Write-Combined Layout | 128-byte alignment, zero CPU cache pollution |
| **1.8 Static Code Analysis** | `clang-tidy`, `cppcheck`, `clippy` | Pointer Safety & Concurrency | Zero dangling pointers, zero integer overflow risks |
| **1.9 Host C-ABI Bindings** | `pybind11`, PyO3, `nm`, `dumpbin` | Shared Library Symbols | Clean symbol export, zero unresolved externals |
| **1.10 Parameter Matrix Fuzz**| Hypo/Property Fuzzer | Dimension Coverage ($d, c$) | Valid syntax across $d \in [2, 16], c \in [1, 128]$ |

---

## 2. Detailed Verification Specifications

### 2.1 Metaprogrammed JIT Code-Generation Testing
* **Objective:** Ensure all metaprogrammed kernels synthesize straight-line code with fully unrolled loops.
* **Execution Plan:**
  1. Test AST generation across stalk dimensions $d \in \{2, 4, 8, 16\}$ and channel counts $c \in \{1, 4, 16, 32, 64\}$.
  2. Parse generated AST/source text using regex and lexers to assert zero runtime loop constructs (`for`, `while`) in the inner contraction blocks.
  3. Verify constant substitution: ensure stalk stride $d \cdot 32$ and restriction matrix indices $i \cdot d + j$ are hardcoded constants.

### 2.2 Offline Cross-Compilation & Syntax Parsing
* **Objective:** Ensure kernels compile cleanly on standard CPU build nodes without requiring an active GPU driver.
* **Execution Plan:**
  1. Invoke `nvcc -ptx -arch=compute_80` and `nvcc -c -arch=sm_80` on generated sources.
  2. Validate compile-time template parameters ($D$, $C$) and verify macros (`unroll_contract_4x4!`, `__launch_bounds__(128, 4)`).
  3. Ensure zero compilation warnings with `-Wall -Werror`.

### 2.3 PTX / SASS Register Pressure & Spilling Inspection
* **Objective:** Verify static register budget limits to prevent local memory spill latency.
* **Execution Plan:**
  1. Capture stdout from `ptxas -v -arch=sm_80`.
  2. Parse regex: `lmem\s*=\s*0\s*bytes`. Fail build immediately if `lmem > 0`.
  3. Verify static register count $\le 64$ registers/thread to guarantee a minimum theoretical occupancy of $50\%$.

### 2.4 CPU Double-Precision Mathematical Oracle
* **Objective:** Create a bit-accurate 64-bit floating-point golden reference for sheaf diffusion.
* **Execution Plan:**
  1. Implement reference Sheaf Laplacian $L_{\mathcal{F}} = D_{\mathcal{F}} - A_{\mathcal{F}}$ in 64-bit float (`f64`).
  2. Compute analytical forward step: $X^{(l+1)} = X^{(l)} - \tau L_{\mathcal{F}} X^{(l)}$.
  3. Store serialized golden input/output tensors (`.npy` / binary) for cross-platform equivalence testing.

### 2.5 Reference Algebraic Invariant Proofs
* **Objective:** Mathematically prove that the diffusion operator adheres to cellular sheaf theory axioms.
* **Execution Plan:**
  1. **Harmonic Trivial Sections:** Generate global section $X_0 \in \ker(L_{\mathcal{F}})$; verify that $L_{\mathcal{F}} X_0 = 0$ and $X^{(l+1)} = X^{(l)}$.
  2. **Non-negative Dirichlet Energy:** Verify that $\mathcal{E}_{\mathcal{F}}(X) = \frac{1}{2} X^T L_{\mathcal{F}} X \ge 0$ for arbitrary stalk tensors.
  3. **Orthogonal Equivariance:** For arbitrary rotation $Q \in O(d)$, verify that diffusing $(Q X)$ under conjugate restriction maps $(Q \rho Q^T)$ equals $Q X^{(l+1)}$.

### 2.6 Graph Topology & Block-Sparse Row (BSR) Packaging
* **Objective:** Validate host-side graph serialization and memory packing.
* **Execution Plan:**
  1. Test edge list conversion into `row_ptr`, `col_ind`, and continuous `rho_data`.
  2. Verify edge-case topologies: disconnected nodes ($\deg(v) = 0$), pendant nodes ($\deg(v) = 1$), and dense complete subgraphs.
  3. Enforce strict monotonic ordering: `row_ptr[v] <= row_ptr[v+1]` and column bounds $col\_ind[e] < |V|$.

### 2.7 Host Pinned Ring Buffer Allocation Logic
* **Objective:** Verify host-side write-combining memory layout and circular buffer abstractions.
* **Execution Plan:**
  1. Mock `cudaHostAllocWriteCombined` allocator with 128-byte aligned virtual memory buffers.
  2. Verify circular head/tail pointer progression under multi-hop double-buffering ($L = 1 \dots 16$).
  3. Confirm that CPU writes stream sequentially to avoid line-buffer splits.

### 2.8 Static Code Analysis & Linting
* **Objective:** Prevent memory vulnerabilities and undefined behavior.
* **Execution Plan:**
  1. Run `cargo clippy --all-targets -- -D warnings` across Rust implementations.
  2. Run `clang-tidy` on C++/CUDA sources with checks for `bugprone-*`, `cert-*`, and `performance-*`.
  3. Check integer arithmetic for potential overflow during index offsets: `node * d * channels`.

### 2.9 Host C-ABI & PyTorch Extension Binding
* **Objective:** Verify binary symbol linkage between the kernel engine and host frameworks.
* **Execution Plan:**
  1. Compile host shared library targets (`.so` / `.dll`).
  2. Inspect symbol tables with `llvm-nm` / `dumpbin` to verify un-mangled C-linkage export of core diffusion entry points.
  3. Verify PyO3 / `pybind11` module import without missing dynamic symbols.

### 2.10 Type-Check & Metaprogramming Parameter Matrix Generation
* **Objective:** Fuzz the metaprogramming synthesizer over broad configuration spaces.
* **Execution Plan:**
  1. Execute a Cartesian sweep over $d \in [2, 16]$ and $c \in [1, 128]$.
  2. Assert valid compilation and memory stride calculation for all non-power-of-two dimensions.
  3. Verify that zero degenerate (0-length) arrays or syntax errors are generated.

---

## 3. Execution Pipeline & Automation

Run the complete CPU test suite locally or in CI:

```bash
# 1. Rust Domain & Metaprogramming Tests
cd d:\kernal\fbs_sde_rust
cargo test --all-targets

# 2. Release Mode Invariant & Benchmark Checks
cargo run --release --bin bench

# 3. Static Analysis
cargo clippy -- -D warnings
```
