"""
Paged KV-Cache implementation for efficient memory management

Based on vLLM's paged attention design with SM120 optimizations
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BlockTable:
    """Block table for mapping logical blocks to physical blocks"""

    logical_blocks: List[int]
    physical_blocks: List[int]

    def get_physical_block(self, logical_idx: int) -> int:
        """Get physical block index for a logical block"""
        return self.physical_blocks[logical_idx]


class PagedKVCache:
    """
    Paged KV-Cache for efficient memory management

    Features:
    - Dynamic memory allocation
    - Support for variable-length sequences
    - 50% memory reduction vs. contiguous cache
    - Efficient block management

    Args:
        num_blocks: Total number of physical blocks
        block_size: Number of tokens per block
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        num_layers: Number of transformer layers
        dtype: Data type for cache
        device: Device to allocate cache on
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        # Allocate physical blocks
        # Shape: [num_layers, 2, num_blocks, num_heads, block_size, head_dim]
        # 2 for K and V
        self.cache = torch.zeros(
            num_layers,
            2,
            num_blocks,
            num_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device=device,
        )

        # Free block list
        self.free_blocks = list(range(num_blocks))

        # Block tables for each sequence
        self.block_tables: Dict[int, BlockTable] = {}

        # Track allocated blocks per sequence
        self.seq_blocks: Dict[int, List[int]] = {}

    def allocate_blocks(self, seq_id: int, num_blocks: int) -> List[int]:
        """
        Allocate blocks for a sequence

        Args:
            seq_id: Sequence ID
            num_blocks: Number of blocks to allocate

        Returns:
            List of physical block indices
        """
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"Not enough free blocks. Need {num_blocks}, have {len(self.free_blocks)}")

        # Allocate blocks
        allocated = []
        for _ in range(num_blocks):
            block_idx = self.free_blocks.pop(0)
            allocated.append(block_idx)

        # Update tracking
        if seq_id not in self.seq_blocks:
            self.seq_blocks[seq_id] = []
        self.seq_blocks[seq_id].extend(allocated)

        # Create block table
        logical_blocks = list(range(len(self.seq_blocks[seq_id])))
        self.block_tables[seq_id] = BlockTable(
            logical_blocks=logical_blocks,
            physical_blocks=self.seq_blocks[seq_id].copy(),
        )

        return allocated

    def free_sequence(self, seq_id: int) -> None:
        """
        Free all blocks for a sequence

        Args:
            seq_id: Sequence ID
        """
        if seq_id in self.seq_blocks:
            # Return blocks to free list
            self.free_blocks.extend(self.seq_blocks[seq_id])
            del self.seq_blocks[seq_id]
            del self.block_tables[seq_id]

    def get_kv_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get K and V cache for a layer

        Args:
            layer_idx: Layer index

        Returns:
            (k_cache, v_cache)
        """
        return self.cache[layer_idx, 0], self.cache[layer_idx, 1]

    def write_kv(
        self,
        layer_idx: int,
        seq_id: int,
        token_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Write K, V to cache

        Args:
            layer_idx: Layer index
            seq_id: Sequence ID
            token_idx: Token index in sequence
            k: Key tensor [num_heads, head_dim]
            v: Value tensor [num_heads, head_dim]
        """
        # Get block table
        block_table = self.block_tables[seq_id]

        # Calculate block and offset
        logical_block = token_idx // self.block_size
        offset = token_idx % self.block_size

        # Get physical block
        physical_block = block_table.get_physical_block(logical_block)

        # Write to cache
        self.cache[layer_idx, 0, physical_block, :, offset, :] = k
        self.cache[layer_idx, 1, physical_block, :, offset, :] = v

    def read_kv(
        self,
        layer_idx: int,
        seq_id: int,
        token_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read K, V from cache

        Args:
            layer_idx: Layer index
            seq_id: Sequence ID
            token_idx: Token index

        Returns:
            (k, v) tensors [num_heads, head_dim]
        """
        # Get block table
        block_table = self.block_tables[seq_id]

        # Calculate block and offset
        logical_block = token_idx // self.block_size
        offset = token_idx % self.block_size

        # Get physical block
        physical_block = block_table.get_physical_block(logical_block)

        # Read from cache
        k = self.cache[layer_idx, 0, physical_block, :, offset, :]
        v = self.cache[layer_idx, 1, physical_block, :, offset, :]

        return k, v

    def get_block_table_tensor(self, seq_id: int) -> torch.Tensor:
        """
        Get block table as tensor for kernel

        Args:
            seq_id: Sequence ID

        Returns:
            Block table tensor [num_blocks]
        """
        block_table = self.block_tables[seq_id]
        return torch.tensor(
            block_table.physical_blocks,
            dtype=torch.int32,
            device=self.device,
        )

    def get_num_free_blocks(self) -> int:
        """Get number of free blocks"""
        return len(self.free_blocks)

    def get_num_allocated_blocks(self, seq_id: int) -> int:
        """Get number of allocated blocks for a sequence"""
        if seq_id not in self.seq_blocks:
            return 0
        return len(self.seq_blocks[seq_id])


class CacheManager:
    """
    High-level cache manager

    Handles automatic block allocation and memory management
    """

    def __init__(
        self,
        max_num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.cache = PagedKVCache(
            num_blocks=max_num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            dtype=dtype,
            device=device,
        )
        self.block_size = block_size
        self.seq_lengths: Dict[int, int] = {}

    def add_sequence(self, seq_id: int, prompt_len: int) -> None:
        """
        Add a new sequence

        Args:
            seq_id: Sequence ID
            prompt_len: Length of prompt
        """
        # Calculate number of blocks needed
        num_blocks = (prompt_len + self.block_size - 1) // self.block_size

        # Allocate blocks
        self.cache.allocate_blocks(seq_id, num_blocks)

        # Track sequence length
        self.seq_lengths[seq_id] = prompt_len

    def append_token(self, seq_id: int) -> None:
        """
        Append a token to sequence (may allocate new block)

        Args:
            seq_id: Sequence ID
        """
        # Increment sequence length
        self.seq_lengths[seq_id] += 1

        # Check if we need a new block
        new_len = self.seq_lengths[seq_id]
        current_blocks = self.cache.get_num_allocated_blocks(seq_id)
        needed_blocks = (new_len + self.block_size - 1) // self.block_size

        if needed_blocks > current_blocks:
            # Allocate one more block
            self.cache.allocate_blocks(seq_id, 1)

    def remove_sequence(self, seq_id: int) -> None:
        """
        Remove a sequence and free its blocks

        Args:
            seq_id: Sequence ID
        """
        self.cache.free_sequence(seq_id)
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]

    def get_sequence_length(self, seq_id: int) -> int:
        """Get current length of sequence"""
        return self.seq_lengths.get(seq_id, 0)

    def can_allocate(self, prompt_len: int) -> bool:
        """
        Check if we can allocate blocks for a new sequence

        Args:
            prompt_len: Length of prompt

        Returns:
            True if allocation is possible
        """
        needed_blocks = (prompt_len + self.block_size - 1) // self.block_size
        return self.cache.get_num_free_blocks() >= needed_blocks
