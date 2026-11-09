"""Double-Precision (FP64) Mathematical Reference Oracle for Cellular Sheaf Diffusion.

Implements exact analytical cellular sheaf Laplacian construction and discrete heat equation
diffusion on synthetic and real graph topologies:
    L_F = D_F - A_F
    X^(l+1) = X^(l) - tau * L_F * X^(l)
"""

import numpy as np
from typing import Dict, List, Tuple


class SheafLaplacianOracle:
    """Exact analytical 64-bit floating point reference implementation."""

    def __init__(self, num_nodes: int, stalk_dim: int):
        self.num_nodes = num_nodes
        self.stalk_dim = stalk_dim
        self.total_dim = num_nodes * stalk_dim

    def build_dense_sheaf_laplacian(
        self,
        edges: List[Tuple[int, int]],
        restriction_maps: Dict[Tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """Constructs the full |V|d x |V|d cellular sheaf Laplacian matrix in FP64.

        L_F[v, v] = sum_{e=(u,v) or (v,u)} rho_{v <= e}^T * rho_{v <= e}
        L_F[v, u] = - rho_{v <= e}^T * rho_{u <= e}
        When restriction maps are orthogonal (rho in O(d)), L_F[v, v] = deg(v) * I_d.
        """
        d = self.stalk_dim
        l_f = np.zeros((self.total_dim, self.total_dim), dtype=np.float64)

        for u, v in edges:
            rho_u_e = restriction_maps[(u, (u, v))].astype(np.float64)
            rho_v_e = restriction_maps[(v, (u, v))].astype(np.float64)

            # Degree block contributions
            v_idx = v * d
            u_idx = u * d

            l_f[v_idx:v_idx + d, v_idx:v_idx + d] += rho_v_e.T @ rho_v_e
            l_f[u_idx:u_idx + d, u_idx:u_idx + d] += rho_u_e.T @ rho_u_e

            # Cross adjacency block contributions
            l_f[v_idx:v_idx + d, u_idx:u_idx + d] -= rho_v_e.T @ rho_u_e
            l_f[u_idx:u_idx + d, v_idx:v_idx + d] -= rho_u_e.T @ rho_v_e

        return l_f

    def diffuse_analytical(
        self,
        x_init: np.ndarray,
        l_f: np.ndarray,
        tau: float,
        num_layers: int
    ) -> np.ndarray:
        """Executes exact multi-hop heat equation diffusion: X^(l+1) = (I - tau * L_F) * X^(l)."""
        x_current = x_init.astype(np.float64).copy()
        eye = np.eye(self.total_dim, dtype=np.float64)
        transition_matrix = eye - tau * l_f

        for _ in range(num_layers):
            x_current = transition_matrix @ x_current

        return x_current

    def compute_dirichlet_energy(self, x: np.ndarray, l_f: np.ndarray) -> float:
        """Evaluates global Sheaf Dirichlet energy: E_F(X) = 0.5 * Tr(X^T * L_F * X)."""
        x_64 = x.astype(np.float64)
        if x_64.ndim == 1:
            return float(0.5 * (x_64.T @ l_f @ x_64))
        return float(0.5 * np.trace(x_64.T @ l_f @ x_64))
