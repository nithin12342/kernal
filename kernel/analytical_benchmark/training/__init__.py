"""Training package router.

Exposes the public training pipeline contract. External consumers import from
this entry point only.
"""

from .trainer import Trainer

__all__ = ["Trainer"]
