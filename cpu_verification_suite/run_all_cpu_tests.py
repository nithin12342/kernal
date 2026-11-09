"""Unified Orchestrator for CPU-Only Pre-Silicon Kernel Verification.

Executes all 10 non-GPU verification stages across:
1. Rust with Metaprogramming (ConstSheafEngine, const generics)
2. Rust Standard Kernel (BSR forward/backward diffusion)
3. Mojo Kernel (Hardware SIMD, @parameter, parallelize)
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from cpu_verification_suite.oracle import SheafLaplacianOracle, AlgebraicInvariantProver
from cpu_verification_suite.graph import BSRPackager, TopologyGenerator
from cpu_verification_suite.memory import MockWriteCombinedRingBuffer, AlignmentChecker
from cpu_verification_suite.static_analysis import KernelSyntaxAuditor, RustClippyRunner
from cpu_verification_suite.fuzzer import DimensionFuzzer
from cpu_verification_suite.runners import RustKernelRunner, MojoSemanticValidator


def run_full_verification():
    print("=" * 70)
    print(" FBS-SDE CPU PRE-SILICON VERIFICATION SUITE")
    print(" Validating: Rust (Standard), Rust (Metaprogrammed), Mojo")
    print("=" * 70)

    results = {}
    start_total_time = time.perf_counter()

    # ---------------------------------------------------------
    # Stage 1: Rust Standard & Metaprogrammed Tests & Benchmarks
    # ---------------------------------------------------------
    print("\n[Stage 1/7] Executing Rust Engine Cargo Test Suite...")
    rust_dir = ROOT_DIR / "fbs_sde_rust"
    t0 = time.perf_counter()
    rust_test_ok, rust_test_out = RustKernelRunner.run_tests(rust_dir)
    t_rust_test = time.perf_counter() - t0
    results["1.1_rust_unit_tests"] = (rust_test_ok, f"{t_rust_test:.2f}s")
    print(f"  -> Rust Tests: {'[PASSED]' if rust_test_ok else '[FAILED]'} in {t_rust_test:.2f}s")

    print("\n[Stage 2/7] Executing Rust Release Optimized Benchmark...")
    t0 = time.perf_counter()
    rust_bench_ok, rust_bench_out = RustKernelRunner.run_benchmark(rust_dir)
    t_rust_bench = time.perf_counter() - t0
    results["1.2_rust_benchmark"] = (rust_bench_ok, f"{t_rust_bench:.2f}s")
    print(f"  -> Rust Metaprogrammed Benchmark: {'[PASSED]' if rust_bench_ok else '[FAILED]'} in {t_rust_bench:.2f}s")

    # ---------------------------------------------------------
    # Stage 3: Static Analysis & Clippy Linter
    # ---------------------------------------------------------
    print("\n[Stage 3/7] Running Static Analysis and Rust Clippy...")
    clippy_ok, clippy_out = RustClippyRunner.run_clippy(rust_dir)
    results["2.1_rust_clippy"] = (clippy_ok, "0 warnings" if clippy_ok else clippy_out[:200])
    print(f"  -> Rust Clippy (-D warnings): {'[PASSED]' if clippy_ok else '[FAILED]'}")

    syntax_rust_ok, syntax_rust_issues = KernelSyntaxAuditor.audit_rust_metaprogramming(
        rust_dir / "src" / "metaprogramming" / "const_metaprog.rs"
    )
    results["2.2_rust_syntax_audit"] = (syntax_rust_ok, ", ".join(syntax_rust_issues) if syntax_rust_issues else "Clean")
    print(f"  -> Rust Metaprogramming Syntax Audit: {'[PASSED]' if syntax_rust_ok else '[FAILED]'}")

    mojo_dir = ROOT_DIR / "fbs_sde_mojo"
    syntax_mojo_ok, syntax_mojo_issues = KernelSyntaxAuditor.audit_mojo_implementation(mojo_dir)
    results["2.3_mojo_syntax_audit"] = (syntax_mojo_ok, ", ".join(syntax_mojo_issues) if syntax_mojo_issues else "Clean")
    print(f"  -> Mojo Kernel Syntax Audit: {'[PASSED]' if syntax_mojo_ok else '[FAILED]'}")

    # ---------------------------------------------------------
    # Stage 4: Mojo Semantic & Algorithmic Validation
    # ---------------------------------------------------------
    print("\n[Stage 4/7] Validating Mojo Package & SIMD Algorithms...")
    mojo_struct_ok, mojo_missing = MojoSemanticValidator.validate_package_structure(mojo_dir)
    results["3.1_mojo_structure"] = (mojo_struct_ok, "Complete" if mojo_struct_ok else str(mojo_missing))
    print(f"  -> Mojo Package Completeness: {'[PASSED]' if mojo_struct_ok else '[FAILED]'}")

    mojo_simd_ok, mojo_simd_err = MojoSemanticValidator.simulate_mojo_simd_contraction(d=4, channels=32, simd_width=8)
    results["3.2_mojo_simd_math"] = (mojo_simd_ok, f"Max error: {mojo_simd_err:.2e}")
    print(f"  -> Mojo SIMD Contraction Equivalence: {'[PASSED]' if mojo_simd_ok else '[FAILED]'} (err={mojo_simd_err:.2e})")

    # ---------------------------------------------------------
    # Stage 5: Double-Precision Mathematical Oracle & Invariant Proofs
    # ---------------------------------------------------------
    print("\n[Stage 5/7] Proving Algebraic Invariants via FP64 Oracle...")
    oracle = SheafLaplacianOracle(num_nodes=6, stalk_dim=4)
    prover = AlgebraicInvariantProver(oracle)

    # Isolated & pendant topology
    num_nodes, edges, maps = TopologyGenerator.generate_isolated_and_pendant_graph(d=4)
    l_f = oracle.build_dense_sheaf_laplacian(edges, maps)

    harm_ok, harm_drift = prover.verify_harmonic_preservation(edges, maps)
    results["4.1_harmonic_preservation"] = (harm_ok, f"Drift: {harm_drift:.2e}")
    print(f"  -> Harmonic Invariance (L_F * X = 0): {'[PASSED]' if harm_ok else '[FAILED]'} (drift={harm_drift:.2e})")

    energy_ok, min_energy = prover.verify_non_negative_dirichlet_energy(l_f)
    results["4.2_dirichlet_energy_non_negative"] = (energy_ok, f"Min energy: {min_energy:.4f}")
    print(f"  -> Non-Negative Dirichlet Energy: {'[PASSED]' if energy_ok else '[FAILED]'} (min={min_energy:.4f})")

    equiv_ok, equiv_drift = prover.verify_orthogonal_equivariance(edges, maps)
    results["4.3_orthogonal_equivariance"] = (equiv_ok, f"Drift: {equiv_drift:.2e}")
    print(f"  -> Orthogonal Equivariance O(d): {'[PASSED]' if equiv_ok else '[FAILED]'} (drift={equiv_drift:.2e})")

    # ---------------------------------------------------------
    # Stage 6: Graph Topology Serialization & Stress Cases
    # ---------------------------------------------------------
    print("\n[Stage 6/7] Testing Graph Topology & BSR Serialization...")
    star_nodes, star_edges, star_maps = TopologyGenerator.generate_star_graph(num_leaves=10, d=4)
    star_bsr = BSRPackager.pack(star_nodes, 4, star_edges, star_maps)
    results["5.1_star_graph_bsr"] = (star_bsr.is_valid, f"Nodes: {star_nodes}, Edges: {star_bsr.num_edges}")
    print(f"  -> Star Graph BSR Serialization: {'[PASSED]' if star_bsr.is_valid else '[FAILED]'}")

    k_nodes, k_edges, k_maps = TopologyGenerator.generate_complete_graph(num_nodes=8, d=4)
    complete_bsr = BSRPackager.pack(k_nodes, 4, k_edges, k_maps)
    results["5.2_complete_graph_bsr"] = (complete_bsr.is_valid, f"Nodes: {k_nodes}, Edges: {complete_bsr.num_edges}")
    print(f"  -> Complete Graph K_8 BSR Serialization: {'[PASSED]' if complete_bsr.is_valid else '[FAILED]'}")

    # ---------------------------------------------------------
    # Stage 7: Memory Subsystem & Dimension Fuzzing
    # ---------------------------------------------------------
    print("\n[Stage 7/7] Testing Ring Buffer Allocations & Dimension Fuzzing...")
    ring_buf = MockWriteCombinedRingBuffer(capacity_bytes=1024 * 1024, alignment=128)
    base_aligned = ring_buf.is_base_aligned()
    stage_addr, offset = ring_buf.allocate_stage(stage_size_bytes=512)
    stage_aligned = AlignmentChecker.is_aligned(stage_addr, 128)
    buf_ok = base_aligned and stage_aligned
    results["6.1_ring_buffer_128b_alignment"] = (buf_ok, f"Base={base_aligned}, Stage={stage_aligned}")
    print(f"  -> Host Ring Buffer 128-Byte Alignment: {'[PASSED]' if buf_ok else '[FAILED]'}")

    fuzz_res = DimensionFuzzer.fuzz_parameter_grid(d_range=(2, 16))
    fuzz_ok = (fuzz_res.failed_combinations == 0)
    results["6.2_dimension_fuzzing"] = (
        fuzz_ok,
        f"{fuzz_res.passed_combinations}/{fuzz_res.total_combinations_tested} combinations passed"
    )
    print(f"  -> Dimension Matrix Fuzzing (d in [2,16], c in [1,128]): {'[PASSED]' if fuzz_ok else '[FAILED]'}")

    total_time = time.perf_counter() - start_total_time
    print("\n" + "=" * 70)
    print(f" VERIFICATION SUMMARY (Completed in {total_time:.2f}s)")
    print("=" * 70)

    all_passed = True
    for test_name, (passed, detail) in results.items():
        status_str = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"  {status_str} {test_name:<35} : {detail}")

    print("=" * 70)
    print(f" OVERALL STATUS: {'ALL CHECKS PASSED [SUCCESS]' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)
