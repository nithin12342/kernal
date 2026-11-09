"""Dimension Matrix Fuzzing Subpackage.

Fuzzes parameter spaces over stalk dimensions d in [2, 16] and channels c in [1, 128]
to verify stride alignments and zero degenerate arrays.
"""

from .dimension_matrix import DimensionFuzzer

__all__ = ["DimensionFuzzer"]
