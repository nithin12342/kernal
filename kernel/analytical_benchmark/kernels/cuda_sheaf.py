"""CUDA sheaf kernel integration stub.

Provides a :class:`CudaSheafKernel` that would drive the compiled CUDA sheaf
diffusion kernel (sheaf_conveyor_gpu_bridge.cu / cuda_100.cu) via a shared
library or subprocess. Since nvcc and a CUDA-capable GPU are not present on
this machine, the backend reports itself as unavailable and falls back to the
PyTorch reference implementation.
"""

from __future__ import annotations

import shutil
from typing import Optional

import torch

from .base import SheafKernel


class CudaSheafKernel(SheafKernel):
    """CUDA-accelerated sheaf diffusion backend (stub; falls back to PyTorch)."""

    name = "cuda"

    def __init__(self, stalk_dim: int = 4, tau: float = 0.15) -> None:
        super().__init__(stalk_dim=stalk_dim, tau=tau)
        self._available = shutil.which("nvcc") is not None and torch.cuda.is_available()

    def available(self) -> bool:
        return self._available

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Run CUDA sheaf diffusion if available, else PyTorch reference."""
        if self._available and x.is_cuda:
            # Placeholder: actual CUDA launch would go here once nvcc/GPU present.
            raise NotImplementedError("CUDA kernel launch not wired without nvcc + GPU.")
        return _pytorch_sheaf_forward(x, edge_index, rho, self.tau)


def _pytorch_sheaf_forward(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    rho: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    num_nodes = x.shape[0]
    d = x.shape[1]
    c = x.shape[2]
    num_edges = edge_index.shape[1]

    degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    degree.index_add_(0, edge_index[1], torch.ones(num_edges, device=x.device, dtype=x.dtype))
    decay = 1.0 - tau * degree

    src = edge_index[0]
    dst = edge_index[1]
    x_u = x[src]
    contracted = torch.bmm(rho, x_u)

    acc = torch.zeros_like(x)
    acc.index_add_(0, dst, contracted)

    x_out = decay.unsqueeze(-1).unsqueeze(-1) * x + tau * acc
    return x_out
