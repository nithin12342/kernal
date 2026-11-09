"""DNN / MLP baseline model.

A standard multi-layer perceptron operating purely on node features, with no
graph structure consumed. Serves as the non-relational baseline for the
comparative benchmark.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DNN(nn.Module):
    """Multi-layer perceptron node classifier.

    Architecture: Linear -> ReLU -> Dropout -> Linear -> ... -> Linear
    with identical hidden dimension and layer count across all model types.
    """

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over node features only.

        Args:
            x: Node feature matrix of shape |V| x F.

        Returns:
            Logits of shape |V| x num_classes.
        """
        return self.mlp(x)
