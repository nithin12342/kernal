"""Static Analysis and Syntax Validation Subpackage.

Runs AST and lexical audits on Mojo and Rust kernel implementations,
and executes strict Clippy compiler linter passes.
"""

from .syntax_validator import KernelSyntaxAuditor
from .clippy_runner import RustClippyRunner

__all__ = ["KernelSyntaxAuditor", "RustClippyRunner"]
