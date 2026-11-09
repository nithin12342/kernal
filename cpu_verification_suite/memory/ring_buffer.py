"""Host Write-Combined Pinned Ring Buffer and Alignment Module.

Implements and validates:
- 128-byte hardware alignment calculations: (addr + 127) & ~127
- Ping-pong multi-hop double-buffering pointer arithmetic
- Emulated CPU write-combining streaming without CPU cache pollution
"""

from typing import List, Tuple
import ctypes


class AlignmentChecker:
    """Verifies vector alignment requirements (16-byte float4, 128-byte vector bursts)."""

    @staticmethod
    def align_offset(offset: int, alignment: int = 128) -> int:
        """Calculates aligned byte offset: (offset + alignment - 1) & ~(alignment - 1)."""
        return (offset + alignment - 1) & ~(alignment - 1)

    @staticmethod
    def is_aligned(address: int, alignment: int = 128) -> bool:
        """Checks if a pointer address is aligned to a given power of two."""
        return (address & (alignment - 1)) == 0


class MockWriteCombinedRingBuffer:
    """Simulates pinned host write-combined circular buffer allocations."""

    def __init__(self, capacity_bytes: int, alignment: int = 128):
        self.capacity_bytes = capacity_bytes
        self.alignment = alignment

        # Allocate aligned raw byte storage
        raw_size = capacity_bytes + alignment
        self._raw_buf = (ctypes.c_char * raw_size)()
        raw_addr = ctypes.addressof(self._raw_buf)
        self.base_address = AlignmentChecker.align_offset(raw_addr, alignment)

        self.head_offset = 0
        self.tail_offset = 0
        self.write_combining_flushes = 0

    def is_base_aligned(self) -> bool:
        """Returns True if the base address satisfies the hardware alignment."""
        return AlignmentChecker.is_aligned(self.base_address, self.alignment)

    def allocate_stage(self, stage_size_bytes: int) -> Tuple[int, int]:
        """Allocates a 128-byte aligned stage inside the circular ring."""
        aligned_size = AlignmentChecker.align_offset(stage_size_bytes, self.alignment)
        if self.head_offset + aligned_size > self.capacity_bytes:
            # Wrap around
            self.head_offset = 0
            self.write_combining_flushes += 1

        stage_addr = self.base_address + self.head_offset
        current_offset = self.head_offset
        self.head_offset += aligned_size

        return stage_addr, current_offset

    def simulate_streaming_write(self, data_size_bytes: int, burst_size: int = 64) -> int:
        """Simulates CPU streaming writes to count write-combining burst flushes."""
        # Consecutive writes fill 64-byte write-combining line buffers before flushing
        num_bursts = (data_size_bytes + burst_size - 1) // burst_size
        return num_bursts
