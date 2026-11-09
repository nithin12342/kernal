"""Host Memory Architecture & Ring Buffer Subpackage.

Verifies write-combined memory abstractions, 128-byte alignment algorithms,
and zero-stall circular buffer pointer arithmetic.
"""

from .ring_buffer import MockWriteCombinedRingBuffer, AlignmentChecker

__all__ = ["MockWriteCombinedRingBuffer", "AlignmentChecker"]
