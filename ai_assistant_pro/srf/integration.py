"""
Integration of Stone Retrieval Function with AI Assistant Pro components

Provides intelligent memory management and request scheduling using SRF.
"""

import torch
from typing import List, Dict, Optional, Tuple
import time

from ai_assistant_pro.srf.core import (
    StoneRetrievalFunction,
    SRFConfig,
    MemoryCandidate,
    RetrievalResult,
)
from ai_assistant_pro.engine.cache import PagedKVCache, CacheManager
from ai_assistant_pro.engine.scheduler import ContinuousBatchScheduler, GenerationRequest


class SRFPagedKVCache(PagedKVCache):
    """
    Paged KV-Cache with SRF-based intelligent memory management

    Extends PagedKVCache to use Stone Retrieval Function for:
    - Intelligent block eviction (which blocks to free when memory is full)
    - Priority-based block allocation
    - Context-aware caching

    Key Innovation:
    Instead of simple LRU or FIFO eviction, uses SRF to identify which
    memory blocks are least likely to be needed based on:
    - Semantic relevance to current queries
    - Emotional importance
    - Association with other blocks
    - Recency and access patterns
    - Decay over time
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        srf_config: Optional[SRFConfig] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Initialize SRF-enhanced paged KV-cache

        Args:
            num_blocks: Total number of physical blocks
            block_size: Number of tokens per block
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            num_layers: Number of transformer layers
            srf_config: SRF configuration (uses defaults if None)
            dtype: Data type for cache
            device: Device to allocate cache on
        """
        super().__init__(
            num_blocks, block_size, num_heads, head_dim, num_layers, dtype, device
        )

        # Initialize SRF
        self.srf = StoneRetrievalFunction(srf_config)

        # Track memory candidates for each block
        self.block_candidates: Dict[int, MemoryCandidate] = {}

        # Track embeddings for semantic similarity
        self.block_embeddings: Dict[int, torch.Tensor] = {}

    def allocate_blocks(
        self,
        seq_id: int,
        num_blocks: int,
        query_embedding: Optional[torch.Tensor] = None,
        emotional_score: float = 0.5,
    ) -> List[int]:
        """
        Allocate blocks with SRF-based eviction if needed

        Args:
            seq_id: Sequence ID
            num_blocks: Number of blocks to allocate
            query_embedding: Query embedding for semantic matching
            emotional_score: Emotional importance (0-1)

        Returns:
            List of allocated physical block indices
        """
        # Check if we have enough free blocks
        if len(self.free_blocks) < num_blocks:
            # Need to evict some blocks using SRF
            blocks_to_evict = num_blocks - len(self.free_blocks)
            self._evict_blocks_srf(blocks_to_evict, query_embedding)

        # Allocate blocks (same as base class)
        allocated = super().allocate_blocks(seq_id, num_blocks)

        # Register blocks with SRF
        for block_idx in allocated:
            # Create memory candidate for this block
            # Use average embedding if available, else zero
            if query_embedding is not None:
                embedding = query_embedding
            else:
                embedding = torch.zeros(768, device=self.device)  # Default dim

            candidate = MemoryCandidate(
                id=block_idx,
                content=embedding,
                emotional_score=emotional_score,
                associations=[],
                timestamp=time.time(),
                metadata={"seq_id": seq_id, "block_idx": block_idx},
            )

            self.block_candidates[block_idx] = candidate
            self.srf.add_candidate(candidate)

        return allocated

    def _evict_blocks_srf(
        self,
        num_to_evict: int,
        query_embedding: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Evict blocks using SRF scoring

        Selects blocks to evict based on LOWEST SRF scores (least useful).

        Args:
            num_to_evict: Number of blocks to evict
            query_embedding: Current query for relevance scoring

        Returns:
            List of evicted block indices
        """
        if not self.block_candidates:
            return []

        # If no query provided, use neutral query
        if query_embedding is None:
            embedding_dim = next(iter(self.block_candidates.values())).content.shape[0]
            query_embedding = torch.zeros(embedding_dim, device=self.device)

        # Retrieve candidates with LOWEST scores (for eviction)
        # We want to keep high-scoring blocks and evict low-scoring ones
        results = self.srf.retrieve(
            query_embedding,
            top_k=len(self.block_candidates),
        )

        # Evict from the bottom (lowest scores)
        evicted = []
        for result in reversed(results):  # Start from lowest scores
            if len(evicted) >= num_to_evict:
                break

            block_idx = result.candidate.id
            seq_id = result.candidate.metadata.get("seq_id")

            # Free the block
            if seq_id is not None and seq_id in self.seq_blocks:
                if block_idx in self.seq_blocks[seq_id]:
                    self.seq_blocks[seq_id].remove(block_idx)
                    self.free_blocks.append(block_idx)
                    evicted.append(block_idx)

                    # Remove from SRF
                    self.srf.remove_candidate(block_idx)
                    if block_idx in self.block_candidates:
                        del self.block_candidates[block_idx]

        return evicted

    def update_block_embedding(
        self,
        block_idx: int,
        embedding: torch.Tensor,
        emotional_score: Optional[float] = None,
    ) -> None:
        """
        Update block embedding (e.g., after processing content)

        Args:
            block_idx: Block index
            embedding: New embedding
            emotional_score: New emotional score (optional)
        """
        if block_idx in self.block_candidates:
            candidate = self.block_candidates[block_idx]
            candidate.content = embedding

            if emotional_score is not None:
                candidate.emotional_score = emotional_score

    def add_block_association(self, block_idx1: int, block_idx2: int) -> None:
        """
        Add association between blocks

        Args:
            block_idx1: First block
            block_idx2: Second block (associated with first)
        """
        if block_idx1 in self.block_candidates:
            candidate = self.block_candidates[block_idx1]
            if block_idx2 not in candidate.associations:
                candidate.associations.append(block_idx2)


class SRFCacheManager(CacheManager):
    """
    Cache manager with SRF integration

    Extends CacheManager to use SRF for intelligent memory management.
    """

    def __init__(
        self,
        max_num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        srf_config: Optional[SRFConfig] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """Initialize SRF-enhanced cache manager"""
        # Use SRF-enhanced cache
        self.cache = SRFPagedKVCache(
            num_blocks=max_num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            srf_config=srf_config,
            dtype=dtype,
            device=device,
        )
        self.block_size = block_size
        self.seq_lengths: Dict[int, int] = {}
        self.seq_embeddings: Dict[int, torch.Tensor] = {}

    def add_sequence(
        self,
        seq_id: int,
        prompt_len: int,
        embedding: Optional[torch.Tensor] = None,
        emotional_score: float = 0.5,
    ) -> None:
        """
        Add sequence with SRF metadata

        Args:
            seq_id: Sequence ID
            prompt_len: Prompt length
            embedding: Sequence embedding for SRF
            emotional_score: Emotional importance
        """
        num_blocks = (prompt_len + self.block_size - 1) // self.block_size

        # Allocate with SRF
        self.cache.allocate_blocks(
            seq_id, num_blocks, embedding, emotional_score
        )

        self.seq_lengths[seq_id] = prompt_len

        if embedding is not None:
            self.seq_embeddings[seq_id] = embedding


class SRFScheduler(ContinuousBatchScheduler):
    """
    Continuous batching scheduler with SRF-based prioritization

    Uses Stone Retrieval Function to prioritize requests based on:
    - Semantic relevance to recent queries
    - Emotional importance
    - Request associations
    - Time-based factors
    """

    def __init__(
        self,
        max_batch_size: int = 64,
        max_num_sequences: int = 256,
        block_size: int = 16,
        srf_config: Optional[SRFConfig] = None,
    ):
        """
        Initialize SRF-based scheduler

        Args:
            max_batch_size: Maximum batch size
            max_num_sequences: Maximum concurrent sequences
            block_size: KV-cache block size
            srf_config: SRF configuration
        """
        super().__init__(max_batch_size, max_num_sequences, block_size)

        # Initialize SRF
        self.srf = StoneRetrievalFunction(srf_config)

        # Track request embeddings
        self.request_embeddings: Dict[int, torch.Tensor] = {}
        self.request_emotional_scores: Dict[int, float] = {}

    def add_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        embedding: Optional[torch.Tensor] = None,
        emotional_score: float = 0.5,
        priority: float = 1.0,
    ) -> int:
        """
        Add request with SRF metadata

        Args:
            prompt_tokens: Tokenized prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            embedding: Request embedding for SRF
            emotional_score: Emotional importance (0-1)
            priority: Base priority

        Returns:
            Request ID
        """
        request_id = super().add_request(
            prompt_tokens, max_tokens, temperature, top_p
        )

        # Store SRF metadata
        if embedding is not None:
            self.request_embeddings[request_id] = embedding
        self.request_emotional_scores[request_id] = emotional_score

        # Add to SRF
        if embedding is not None:
            candidate = MemoryCandidate(
                id=request_id,
                content=embedding,
                emotional_score=emotional_score,
                timestamp=time.time(),
                metadata={"request_id": request_id, "priority": priority},
            )
            self.srf.add_candidate(candidate)

        return request_id

    def schedule(
        self,
        num_free_blocks: int,
        current_query: Optional[torch.Tensor] = None,
    ) -> List[GenerationRequest]:
        """
        Schedule requests using SRF prioritization

        Args:
            num_free_blocks: Available KV-cache blocks
            current_query: Current query embedding for relevance

        Returns:
            List of requests to process (SRF-prioritized)
        """
        # First, include all running requests
        batch = []
        for request in self.running_requests.values():
            if not request.is_finished:
                batch.append(request)

        # Use SRF to prioritize waiting requests
        if self.waiting_queue and current_query is not None:
            # Retrieve top waiting requests using SRF
            waiting_with_embeddings = [
                r for r in self.waiting_queue
                if r.request_id in self.request_embeddings
            ]

            if waiting_with_embeddings:
                # Get SRF scores for waiting requests
                results = self.srf.retrieve(
                    current_query,
                    top_k=min(
                        len(waiting_with_embeddings),
                        self.max_batch_size - len(batch),
                    ),
                    filter_ids=[r.request_id for r in waiting_with_embeddings],
                )

                # Add top-scoring requests to batch
                for result in results:
                    request_id = result.candidate.id
                    request = next(
                        (r for r in self.waiting_queue if r.request_id == request_id),
                        None,
                    )

                    if request:
                        # Check memory
                        blocks_needed = (
                            request.prompt_len + self.block_size - 1
                        ) // self.block_size

                        if blocks_needed <= num_free_blocks:
                            self.waiting_queue.remove(request)
                            request.status.value = "running"
                            request.start_time = time.time()
                            self.running_requests[request_id] = request
                            batch.append(request)
                            num_free_blocks -= blocks_needed

                            if len(batch) >= self.max_batch_size:
                                break
        else:
            # Fall back to standard scheduling
            batch = super().schedule(num_free_blocks)

        return batch

    def get_statistics(self) -> Dict:
        """Get scheduler statistics including SRF metrics"""
        stats = super().get_statistics()
        stats["srf"] = self.srf.get_statistics()
        return stats
