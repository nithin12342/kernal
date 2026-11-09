"""Graph Neural Network (GCN) baseline model.

Standard message-passing graph convolutional network operating on node features
and graph adjacency. Serves as the relational baseline for comparison against
the Sheaf Neural Network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """Graph Convolutional Network node classifier.

    Stacks :class:`GCNConv` layers with ReLU activations and dropout. Uses the
    symmetric normalized adjacency propagation rule::

        X^(l+1) = sigma( D^{-1/2} A D^{-1/2} X^(l) W^(l) )
    """

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))
        self.dropout = dropout
        self.uses_graph = True

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass over the graph.

        Args:
            x: Node feature matrix of shape |V| x F.
            edge_index: Graph connectivity of shape 2 x |E|.

        Returns:
            Logits of shape |V| x num_classes.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNConv(nn.Module):
    """Single symmetric-normalized graph convolution layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype))
        deg_inv_sqrt = torch.pow(deg.clamp(min=1), -0.5)

        src, dst = edge_index[0], edge_index[1]
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        # Propagate: sum_{u in N(v)} norm(v,u) * x_u
        out = torch.zeros(num_nodes, self.weight.shape[1], device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, norm.unsqueeze(-1) * (x[src] @ self.weight))
        out = out + self.bias
        return out
