"""Kernel Syntax and Lexical AST Auditor.

Inspects source code for the three kernel versions:
1. Rust with Metaprogramming (const generics, loop unrolling macros)
2. Rust standard kernel (BSR layouts, in-register FMA)
3. Mojo kernel (@parameter unrolling, hardware SIMD vector types)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


class KernelSyntaxAuditor:
    """Performs static lexical and semantic checks across kernel source files."""

    @staticmethod
    def audit_rust_metaprogramming(source_path: Path) -> Tuple[bool, List[str]]:
        """Audits Rust metaprogrammed kernel for compile-time unrolling and const generics."""
        issues: List[str] = []
        if not source_path.exists():
            return False, [f"File not found: {source_path}"]

        content = source_path.read_text(encoding="utf-8")

        # 1. Check for const generic declaration
        if "ConstSheafEngine<const D: usize, const C: usize>" not in content:
            issues.append("Missing ConstSheafEngine<const D: usize, const C: usize> definition")

        # 2. Check for in-register contract attribute
        if "#[inline(always)]" not in content:
            issues.append("Missing #[inline(always)] attribute on critical contraction path")

        # 3. Check for dynamic allocation in inner contraction loop
        inner_contract_match = re.search(r"fn contract_in_register\s*\(.*?\)\s*\{(.*?)\}", content, re.DOTALL)
        if inner_contract_match:
            body = inner_contract_match.group(1)
            if "vec!" in body or "Vec::new" in body:
                issues.append("Dynamic memory allocation found inside inner contraction function")

        return len(issues) == 0, issues

    @staticmethod
    def audit_mojo_implementation(source_dir: Path) -> Tuple[bool, List[str]]:
        """Audits Mojo source files for @parameter unrolling and hardware SIMD usage."""
        issues: List[str] = []
        if not source_dir.exists():
            return False, [f"Mojo directory not found: {source_dir}"]

        # Check contraction.mojo
        contract_file = source_dir / "contraction.mojo"
        if not contract_file.exists():
            issues.append("Missing contraction.mojo")
        else:
            txt = contract_file.read_text(encoding="utf-8")
            if "@parameter" not in txt:
                issues.append("contraction.mojo does not use @parameter for compile-time unrolling")
            if "SIMD[DType.float32" not in txt:
                issues.append("contraction.mojo does not use SIMD vector types")

        # Check forward.mojo
        fwd_file = source_dir / "forward.mojo"
        if not fwd_file.exists():
            issues.append("Missing forward.mojo")
        else:
            txt = fwd_file.read_text(encoding="utf-8")
            if "parallelize" not in txt:
                issues.append("forward.mojo does not use parallelize for multi-threading")

        return len(issues) == 0, issues
