"""Models package router.

Exposes the public model architecture contract. External consumers import from
this entry point only.
"""

from .dnn import DNN
from .gnn import GCN
from .snn import SheafNN

__all__ = ["DNN", "GCN", "SheafNN"]
