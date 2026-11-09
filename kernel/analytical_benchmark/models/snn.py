"""Sheaf Neural Network (SNN) model.

Cellular sheaf Laplacian-based message-passing network utilizing restriction
maps (stalk spaces over nodes and edges). Each layer applies sheaf diffusion::

    X^(l+1) = (1 - tau * deg(v)) * X^(l) + tau * sum_{e=(u,v)} rho_e^T rho_e X_u

followed by a learnable pointwise mix and non-linearity. The sheaf diffusion
compute is delegated to a :class:`SheafKernel` backend (PyTorch reference,
Rust conveyor, CUDA, or Mojo).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import (
    SheafKernel,
    PyTorchSheafKernel,
    RustSheafKernel,
    CudaSheafKernel,
    MojoSheafKernel,
)


def _build_kernel(backend: str, stalk_dim: int, tau: float) -> SheafKernel:
    """Instantiate the requested sheaf kernel backend with fallback chain."""
    if backend == "rust":
        k = RustSheafKernel(stalk_dim=stalk_dim, tau=tau)
        if k.available():
            return k
    if backend == "cuda":
        k = CudaSheafKernel(stalk_dim=stalk_dim, tau=tau)
        if k.available():
            return k
    if backend == "mojo":
        k = MojoSheafKernel(stalk_dim=stalk_dim, tau=tau)
        if k.available():
            return k
    # PyTorch reference is the universal fallback.
    return PyTorchSheafKernel(stalk_dim=stalk_dim, tau=tau)


class SheafNN(nn.Module):
    """Sheaf Neural Network node classifier.

    Architecture per layer:
        1. Sheaf diffusion over restriction maps (plug-in kernel backend).
        2. Learnable pointwise linear mix over stalk channels.
        3. ReLU + dropout.
    Final layer maps stalk representation to class logits.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.5,
        stalk_dim: int = 4,
        tau: float = 0.15,
        kernel_backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.stalk_dim = stalk_dim
        self.tau = tau
        self.kernel_backend = kernel_backend

        # Input projection: features -> stalk_dim * hidden channels
        self.in_proj = nn.Linear(num_features, stalk_dim * hidden_dim)

        # Sheaf diffusion layers, each with its own kernel instance
        self.sheaf_layers = nn.ModuleList()
        self.mix_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.sheaf_layers.append(_build_kernel(kernel_backend, stalk_dim, tau))
            self.mix_layers.append(nn.Linear(stalk_dim * hidden_dim, stalk_dim * hidden_dim))

        # Output projection: stalk -> classes
        self.out_proj = nn.Linear(stalk_dim * hidden_dim, num_classes)
        self.dropout = dropout
        self.uses_graph = True

    def _build_rho(self, num_edges: int, device: torch.device) -> torch.Tensor:
        """Construct learnable orthogonal restriction maps per edge.

        Returns a parameter of shape |E| x D x D initialized near identity so
        early training behaves like standard diffusion.
        """
        rho = torch.eye(self.stalk_dim, device=device).unsqueeze(0).repeat(num_edges, 1, 1)
        # Small random perturbation to break symmetry
        rho = rho + 0.01 * torch.randn_like(rho)
        return nn.Parameter(rho)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through sheaf diffusion layers.

        Args:
            x: Node feature matrix of shape |V| x F.
            edge_index: Graph connectivity of shape 2 x |E|.

        Returns:
            Logits of shape |V| x num_classes.
        """
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        device = x.device

        # Lift node features into stalk space: |V| x D x C
        h = self.in_proj(x)  # |V| x (D*C)
        h = h.view(num_nodes, self.stalk_dim, -1)  # |V| x D x C

        # Build / retrieve restriction map parameter (shared across forward calls)
        if not hasattr(self, "_rho_param") or self._rho_param.shape[0] != num_edges:
            self._rho_param = self._build_rho(num_edges, device)
        rho = self._rho_param

        # Sheaf diffusion stack
        for i, (kernel, mix) in enumerate(zip(self.sheaf_layers, self.mix_layers)):
            h = kernel.forward(h, edge_index, rho)  # |V| x D x C
            h = h.view(num_nodes, -1)  # flatten stalk x channels
            h = mix(h)
            if i < len(self.sheaf_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = h.view(num_nodes, self.stalk_dim, -1)  # back to stalk layout

        h = h.view(num_nodes, -1)
        logits = self.out_proj(h)
        return logits
