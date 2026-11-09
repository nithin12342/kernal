"""Graph Topology Test Case Generator.

Synthesizes stressful edge-case topologies:
1. Isolated Nodes (deg(v) = 0)
2. Pendant / Leaf Nodes (deg(v) = 1)
3. Star Graph (central hub with high degree divergence)
4. Dense Complete Graph (K_N)
5. Disconnected Components
"""

from typing import Dict, List, Tuple
import numpy as np


class TopologyGenerator:
    """Generates synthetic graph topologies and random O(d) restriction maps."""

    @staticmethod
    def random_orthogonal(d: int) -> np.ndarray:
        """Generates a random orthogonal matrix in O(d) via QR decomposition."""
        h = np.random.randn(d, d)
        q, r = np.linalg.qr(h)
        # Ensure proper reflection signs
        d_diag = np.diag(r)
        ph = d_diag / np.abs(d_diag)
        return (q * ph).astype(np.float32)

    @classmethod
    def generate_star_graph(cls, num_leaves: int, d: int) -> Tuple[int, List[Tuple[int, int]], Dict]:
        """Star graph with center node 0 and leaves 1..N."""
        num_nodes = num_leaves + 1
        edges = [(0, i) for i in range(1, num_nodes)]
        maps = {}
        for u, v in edges:
            maps[(u, (u, v))] = cls.random_orthogonal(d)
            maps[(v, (u, v))] = cls.random_orthogonal(d)
        return num_nodes, edges, maps

    @classmethod
    def generate_isolated_and_pendant_graph(cls, d: int) -> Tuple[int, List[Tuple[int, int]], Dict]:
        """Graph containing isolated nodes, a pair, and a pendant node.
        Total nodes: 6
        Node 0: isolated
        Node 1, 2: connected pair
        Node 3, 4, 5: path graph (4 is center, 3 and 5 are pendants)
        """
        num_nodes = 6
        edges = [(1, 2), (3, 4), (4, 5)]
        maps = {}
        for u, v in edges:
            maps[(u, (u, v))] = cls.random_orthogonal(d)
            maps[(v, (u, v))] = cls.random_orthogonal(d)
        return num_nodes, edges, maps

    @classmethod
    def generate_complete_graph(cls, num_nodes: int, d: int) -> Tuple[int, List[Tuple[int, int]], Dict]:
        """Complete graph K_N."""
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.append((i, j))
        maps = {}
        for u, v in edges:
            maps[(u, (u, v))] = cls.random_orthogonal(d)
            maps[(v, (u, v))] = cls.random_orthogonal(d)
        return num_nodes, edges, maps
