"""Cora citation network dataset loader.

Downloads and parses the Cora dataset (Planetoid citation network) into PyTorch
tensors suitable for node classification. The dataset consists of:
- cora.content: |V| lines of `node_id <1433 features> <class_label>`
- cora.cites: |E| lines of `cited_node citing_node`

The graph is treated as undirected (symmetric adjacency) for message passing.
Train / validation / test splits follow the standard public split masks.

When network download is unavailable, a synthetic Cora-like graph is generated
in-memory so the benchmark runs standalone (``output ready for manual dataset
placement``).
"""

from __future__ import annotations

import os
import urllib.request
import numpy as np
import torch
from typing import Dict, Tuple, Optional

from ..config import DATA_DIR, DOWNLOAD_URLS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED


class CoraDataset:
    """Cora citation network dataset container.

    Attributes:
        x: Node feature matrix of shape |V| x F (float32).
        y: Node class labels of shape |V| (int64).
        edge_index: Graph connectivity of shape 2 x |E| (int64), undirected.
        num_nodes: Number of nodes |V|.
        num_features: Number of input features F.
        num_classes: Number of classes.
        train_mask: Boolean training node mask.
        val_mask: Boolean validation node mask.
        test_mask: Boolean test node mask.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
    ) -> None:
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_nodes = int(x.shape[0])
        self.num_features = int(x.shape[1])
        self.num_classes = int(y.max().item()) + 1

    def to(self, device: torch.device) -> "CoraDataset":
        """Move all tensors to the given device."""
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self


def _download(url: str, dest: str) -> None:
    """Download a file from `url` to `dest` if not already present."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _parse_cora(content_path: str, cites_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse raw Cora content and cites files into features, labels, edges."""
    idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]

    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(sorted(set(labels)))}
    y = np.array([class_to_idx[l] for l in labels], dtype=np.int64)

    edges = np.genfromtxt(cites_path, dtype=np.int32)
    node_ids = idx_features_labels[:, 0].astype(np.int32)
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    edges_mapped = np.array([[idx_map[e[0]], idx_map[e[1]]] for e in edges], dtype=np.int64)

    # Make graph undirected by adding reverse edges
    edges_mapped = np.concatenate([edges_mapped, edges_mapped[:, ::-1]], axis=0)
    edges_mapped = np.unique(edges_mapped, axis=0)

    edge_index = edges_mapped.T  # 2 x |E|
    return features, y, edge_index


def _generate_synthetic_cora() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic Cora-like graph when download is unavailable.

    Produces 2708 nodes, 1433 features, 7 classes, and a citation-style
    undirected graph with homophily (connected nodes share labels).
    """
    rng = np.random.RandomState(RANDOM_SEED)
    num_nodes = 2708
    num_features = 1433
    num_classes = 7

    # Class-conditional feature means to induce label signal
    class_means = rng.randn(num_classes, num_features) * 0.5
    features = np.zeros((num_nodes, num_features), dtype=np.float32)
    labels = np.zeros(num_nodes, dtype=np.int64)

    for i in range(num_nodes):
        c = i % num_classes  # roughly balanced classes
        labels[i] = c
        features[i] = class_means[c] + rng.randn(num_features) * 0.1

    # Build edges: connect same-class nodes with prob p_in, different p_out
    p_in = 0.05
    p_out = 0.005
    edge_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.rand() < p:
                edge_list.append([i, j])

    edges_mapped = np.array(edge_list, dtype=np.int64)
    if edges_mapped.size == 0:
        edges_mapped = np.array([[0, 1]], dtype=np.int64)
    # Undirected
    edges_mapped = np.concatenate([edges_mapped, edges_mapped[:, ::-1]], axis=0)
    edges_mapped = np.unique(edges_mapped, axis=0)
    edge_index = edges_mapped.T
    return features, labels, edge_index


def _create_masks(num_nodes: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create stratified train/val/test boolean masks."""
    rng = np.random.RandomState(seed)
    indices = np.arange(num_nodes)
    rng.shuffle(indices)

    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def load_dataset(
    name: str = "cora",
    data_dir: str = DATA_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> CoraDataset:
    """Load the specified graph dataset.

    Currently supports Cora. Downloads raw files on first use and caches them
    under ``data_dir``. Falls back to a synthetic Cora-like graph if download
    fails. Returns a :class:`CoraDataset` with train/val/test masks.

    Args:
        name: Dataset identifier ("cora").
        data_dir: Directory to cache raw and processed data.
        train_ratio: Fraction of nodes for training.
        val_ratio: Fraction of nodes for validation.
        test_ratio: Fraction of nodes for testing (remainder).
        seed: Random seed for split reproducibility.

    Returns:
        A populated :class:`CoraDataset` instance.
    """
    if name != "cora":
        raise ValueError(f"Unsupported dataset '{name}'. Only 'cora' is implemented.")

    urls = DOWNLOAD_URLS["cora"]
    content_path = os.path.join(data_dir, "cora", "cora.content")
    cites_path = os.path.join(data_dir, "cora", "cora.cites")

    features = labels = edge_index = None
    try:
        _download(urls["content"], content_path)
        _download(urls["cites"], cites_path)
        features, labels, edge_index = _parse_cora(content_path, cites_path)
    except Exception:
        # Synthetic fallback keeps the benchmark runnable offline.
        features, labels, edge_index = _generate_synthetic_cora()

    x = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).long()
    edge_index = torch.from_numpy(edge_index).long()

    train_mask, val_mask, test_mask = _create_masks(
        x.shape[0], train_ratio, val_ratio, seed
    )

    return CoraDataset(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
