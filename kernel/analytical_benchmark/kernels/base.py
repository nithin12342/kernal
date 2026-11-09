"""Kernel integration base contract and available backend discovery.

Defines the abstract :class:`SheafKernel` interface that every sheaf diffusion
backend must satisfy, plus a registry of available backends based on the
compiled toolchains present on this machine.
"""

from __future__ import annotations

import os
import shutil
from typing import List, Optional

import torch
import torch.nn as nn


class SheafKernel(nn.Module):
    """Abstract interface for cellular sheaf diffusion compute backends.

    Each backend implements a single sheaf diffusion layer:
        X^(l+1) = (1 - tau * deg(v)) * X^(l) + tau * sum_e rho_e^T rho_e X_u

    Substituting a backend is done by constructing the appropriate concrete
    subclass; the SNN model is agnostic to the underlying implementation.
    """

    name: str = "base"

    def __init__(self, stalk_dim: int = 4, tau: float = 0.15) -> None:
        super().__init__()
        self.stalk_dim = stalk_dim
        self.tau = tau

    def forward(
        self,
        x: "torch.Tensor",
        edge_index: "torch.Tensor",
        rho: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute one sheaf diffusion layer.

        Args:
            x: Node stalk tensor of shape |V| x D x C.
            edge_index: Graph connectivity of shape 2 x |E|.
            rho: Restriction maps of shape |E| x D x D.

        Returns:
            Updated node stalks of shape |V| x D x C.
        """
        raise NotImplementedError

    def available(self) -> bool:
        """Return True if this backend is usable on the current machine."""
        return False


def available_backends() -> List[str]:
    """Return the list of backend names that are usable on this machine."""
    backends: List[str] = []
    # PyTorch reference is always available
    backends.append("pytorch")
    # Rust bench binary present?
    if shutil.which("bench") or _rust_binary_exists():
        backends.append("rust")
    # CUDA requires nvcc and a GPU
    if shutil.which("nvcc"):
        backends.append("cuda")
    # Mojo requires the mojo CLI
    if shutil.which("mojo"):
        backends.append("mojo")
    return backends


def _rust_binary_exists() -> bool:
    """Check for the prebuilt Rust sheaf bench binary."""
    from ..config import RUST_BENCH_BINARY

    return os_path_exists(RUST_BENCH_BINARY)


def os_path_exists(path: str) -> bool:
    """Thin wrapper around os.path.exists to keep this module import-safe."""
    import os

    return os.path.exists(path)


class PyTorchSheafKernel(SheafKernel):
    """Pure PyTorch reference sheaf diffusion backend (universal fallback)."""

    name = "pytorch"

    def __init__(self, stalk_dim: int = 4, tau: float = 0.15) -> None:
        super().__init__(stalk_dim=stalk_dim, tau=tau)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        return _pytorch_sheaf_forward(x, edge_index, rho, self.tau)

    def available(self) -> bool:
        return True


def _pytorch_sheaf_forward(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    rho: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Pure PyTorch reference sheaf diffusion layer."""
    num_nodes = x.shape[0]
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
