"""Kernels package router.

Exposes the public sheaf kernel integration contract. External consumers
import from this entry point only.
"""

from .rust_sheaf import RustSheafKernel
from .cuda_sheaf import CudaSheafKernel
from .mojo_sheaf import MojoSheafKernel
from .base import SheafKernel, PyTorchSheafKernel, available_backends

__all__ = ["SheafKernel", "PyTorchSheafKernel", "RustSheafKernel", "CudaSheafKernel", "MojoSheafKernel", "available_backends"]
