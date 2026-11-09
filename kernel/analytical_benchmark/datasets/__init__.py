"""Datasets package router.

Exposes the public dataset loading contract. External consumers import from
this entry point only.
"""

from .loader import CoraDataset, load_dataset

__all__ = ["CoraDataset", "load_dataset"]
