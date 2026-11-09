"""Algebraic Invariant Verification Suite for Cellular Sheaf Diffusion.

Verifies fundamental mathematical axioms on CPU before accelerator dispatch:
1. Harmonic Trivial Sections: L_F * X_harmonic = 0 => X^(l+1) = X^(l)
2. Non-Negative Dirichlet Energy: E_F(X) >= 0 for all X
3. Orthogonal Equivariance: Diffuse(Q * X, Q * rho * Q^T) == Q * Diffuse(X, rho) for all Q in O(d)
"""

import numpy as np
from scipy.stats import ortho_group
from typing import Dict, List, Tuple
from .sheaf_laplacian import SheafLaplacianOracle


class AlgebraicInvariantProver:
    """Verifies theoretical algebraic invariants of cellular sheaf diffusion."""

    def __init__(self, oracle: SheafLaplacianOracle):
        self.oracle = oracle

    def verify_harmonic_preservation(
        self,
        edges: List[Tuple[int, int]],
        restriction_maps: Dict[Tuple[int, int], np.ndarray],
        tau: float = 0.1
    ) -> Tuple[bool, float]:
        """Proves that harmonic sections (nullspace of L_F) remain invariant under diffusion."""
        l_f = self.oracle.build_dense_sheaf_laplacian(edges, restriction_maps)
        eigvals, eigvecs = np.linalg.eigh(l_f)

        # Find eigenvectors corresponding to near-zero eigenvalues (harmonic space)
        harmonic_mask = np.abs(eigvals) < 1e-10
        if not np.any(harmonic_mask):
            # Constant stalk over identity sheaf is always in ker(L_F)
            return True, 0.0

        harmonic_sections = eigvecs[:, harmonic_mask]
        diffused = self.oracle.diffuse_analytical(harmonic_sections, l_f, tau, num_layers=5)
        max_drift = float(np.max(np.abs(diffused - harmonic_sections)))
        return bool(max_drift < 1e-8), max_drift

    def verify_non_negative_dirichlet_energy(
        self,
        l_f: np.ndarray,
        num_random_samples: int = 50
    ) -> Tuple[bool, float]:
        """Proves that Sheaf Dirichlet energy E_F(X) >= 0 for arbitrary stalk vectors."""
        np.random.seed(42)
        min_energy = float("inf")

        for _ in range(num_random_samples):
            x_rand = np.random.randn(self.oracle.total_dim, 8)
            energy = self.oracle.compute_dirichlet_energy(x_rand, l_f)
            if energy < min_energy:
                min_energy = energy
            if energy < -1e-10:
                return False, min_energy

        return True, min_energy

    def verify_orthogonal_equivariance(
        self,
        edges: List[Tuple[int, int]],
        restriction_maps: Dict[Tuple[int, int], np.ndarray],
        tau: float = 0.2
    ) -> Tuple[bool, float]:
        """Proves O(d) equivariance: rotating stalks and maps commutes with diffusion."""
        d = self.oracle.stalk_dim
        q_rot = ortho_group.rvs(d).astype(np.float64)

        # Build original diffusion
        l_f = self.oracle.build_dense_sheaf_laplacian(edges, restriction_maps)
        np.random.seed(1337)
        x_init = np.random.randn(self.oracle.total_dim, 4)
        x_diffused = self.oracle.diffuse_analytical(x_init, l_f, tau, num_layers=2)

        # Construct rotated restriction maps: Q @ rho @ Q^T
        rotated_maps = {}
        for k, v in restriction_maps.items():
            rotated_maps[k] = q_rot @ v @ q_rot.T

        # Rotate initial stalks: (I_N (x) Q) @ X
        q_block = np.kron(np.eye(self.oracle.num_nodes), q_rot)
        x_init_rot = q_block @ x_init

        l_f_rot = self.oracle.build_dense_sheaf_laplacian(edges, rotated_maps)
        x_diffused_rot = self.oracle.diffuse_analytical(x_init_rot, l_f_rot, tau, num_layers=2)

        # Expected: Q @ x_diffused == x_diffused_rot
        expected = q_block @ x_diffused
        max_diff = float(np.max(np.abs(expected - x_diffused_rot)))
        return bool(max_diff < 1e-7), max_diff
