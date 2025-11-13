"""
Core implementation of the Stone Retrieval Function (SRF)

This module implements the patented biologically-inspired memory retrieval system.
"""

import torch
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import time

from ai_assistant_pro.srf.components import (
    SemanticSimilarity,
    EmotionalWeighting,
    AssociativeStrength,
    RecencyTracker,
    DecayModel,
)


@dataclass
class SRFConfig:
    """
    Configuration for Stone Retrieval Function

    Hyperparameters (α, β, γ, δ) control the balance between different
    components of the retrieval score.

    Args:
        alpha: Emotional weight coefficient (default: 0.3)
        beta: Associative strength coefficient (default: 0.2)
        gamma: Recency coefficient (default: 0.25)
        delta: Decay coefficient (default: 0.15)

        semantic_model: Model for semantic similarity (default: "cosine")
        emotional_enabled: Enable emotional weighting (default: True)
        associative_enabled: Enable associative context (default: True)
        recency_enabled: Enable recency tracking (default: True)
        decay_enabled: Enable decay modeling (default: True)

        decay_half_life: Half-life for exponential decay in seconds (default: 3600)
        top_k: Number of top candidates to retrieve (default: 10)

        use_gpu: Use GPU acceleration for computation (default: True)
        use_triton: Use custom Triton kernels (default: True)
    """

    # Hyperparameters
    alpha: float = 0.3  # Emotional weight
    beta: float = 0.2   # Associative strength
    gamma: float = 0.25 # Recency
    delta: float = 0.15 # Decay

    # Component configuration
    semantic_model: str = "cosine"
    emotional_enabled: bool = True
    associative_enabled: bool = True
    recency_enabled: bool = True
    decay_enabled: bool = True

    # Temporal settings
    decay_half_life: float = 3600.0  # 1 hour

    # Retrieval settings
    top_k: int = 10
    min_score: float = 0.0  # Minimum score threshold

    # Performance settings
    use_gpu: bool = True
    use_triton: bool = True

    def validate(self) -> None:
        """Validate configuration"""
        assert self.alpha >= 0, "alpha must be non-negative"
        assert self.beta >= 0, "beta must be non-negative"
        assert self.gamma >= 0, "gamma must be non-negative"
        assert self.delta >= 0, "delta must be non-negative"
        assert self.decay_half_life > 0, "decay_half_life must be positive"
        assert self.top_k > 0, "top_k must be positive"


@dataclass
class MemoryCandidate:
    """
    A memory candidate for retrieval

    Attributes:
        id: Unique identifier
        content: Content embedding (tensor)
        text: Original text content (optional)
        emotional_score: Pre-computed emotional score
        associations: List of associated candidate IDs
        timestamp: Creation timestamp
        access_count: Number of times accessed
        metadata: Additional metadata
    """

    id: int
    content: torch.Tensor  # Embedding vector
    text: Optional[str] = None
    emotional_score: float = 0.0
    associations: List[int] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure content is a tensor"""
        if not isinstance(self.content, torch.Tensor):
            self.content = torch.tensor(self.content)


@dataclass
class RetrievalResult:
    """
    Result of SRF retrieval

    Attributes:
        candidate: The retrieved memory candidate
        score: Final SRF score (R_bio)
        components: Breakdown of score components
    """

    candidate: MemoryCandidate
    score: float
    components: Dict[str, float]

    def __lt__(self, other):
        """Enable sorting by score"""
        return self.score < other.score


class StoneRetrievalFunction:
    """
    Stone Retrieval Function (SRF) - Biologically-inspired memory retrieval

    Implements the patented retrieval function:
    R_bio = S(q, c) + αE(c) + βA(c) + γR(c) − δD(c)

    This system provides intelligent, context-aware memory retrieval by combining:
    - Semantic similarity (relevance to query)
    - Emotional weighting (importance/significance)
    - Associative strength (connection to other memories)
    - Temporal recency (how recent the memory is)
    - Decay modeling (de-prioritization of stale memories)

    Example:
        >>> config = SRFConfig(alpha=0.3, beta=0.2, gamma=0.25, delta=0.15)
        >>> srf = StoneRetrievalFunction(config)
        >>>
        >>> # Add memory candidates
        >>> candidate = MemoryCandidate(
        ...     id=0,
        ...     content=torch.randn(768),
        ...     text="Important information",
        ...     emotional_score=0.8,
        ... )
        >>> srf.add_candidate(candidate)
        >>>
        >>> # Retrieve relevant memories
        >>> query = torch.randn(768)
        >>> results = srf.retrieve(query, top_k=5)
    """

    def __init__(self, config: Optional[SRFConfig] = None):
        """
        Initialize Stone Retrieval Function

        Args:
            config: SRF configuration (uses defaults if None)
        """
        self.config = config or SRFConfig()
        self.config.validate()

        # Initialize components
        self.semantic = SemanticSimilarity(
            model=self.config.semantic_model,
            device="cuda" if self.config.use_gpu else "cpu",
        )

        if self.config.emotional_enabled:
            self.emotional = EmotionalWeighting()

        if self.config.associative_enabled:
            self.associative = AssociativeStrength()

        if self.config.recency_enabled:
            self.recency = RecencyTracker()

        if self.config.decay_enabled:
            self.decay = DecayModel(half_life=self.config.decay_half_life)

        # Memory storage
        self.candidates: Dict[int, MemoryCandidate] = {}
        self.next_id = 0

        # Statistics
        self.total_retrievals = 0
        self.total_candidates_processed = 0

    def add_candidate(self, candidate: MemoryCandidate) -> int:
        """
        Add a memory candidate

        Args:
            candidate: Memory candidate to add

        Returns:
            Candidate ID
        """
        if candidate.id is None:
            candidate.id = self.next_id
            self.next_id += 1

        self.candidates[candidate.id] = candidate
        return candidate.id

    def remove_candidate(self, candidate_id: int) -> None:
        """
        Remove a memory candidate

        Args:
            candidate_id: ID of candidate to remove
        """
        if candidate_id in self.candidates:
            del self.candidates[candidate_id]

    def compute_score(
        self,
        query: torch.Tensor,
        candidate: MemoryCandidate,
        current_time: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute SRF retrieval score for a candidate

        Implements: R_bio = S(q, c) + αE(c) + βA(c) + γR(c) − δD(c)

        Args:
            query: Query embedding
            candidate: Memory candidate
            current_time: Current timestamp (uses time.time() if None)

        Returns:
            (final_score, component_scores)
        """
        if current_time is None:
            current_time = time.time()

        components = {}

        # S(q, c): Semantic similarity
        semantic_score = self.semantic.compute(query, candidate.content)
        components["semantic"] = semantic_score.item()

        # αE(c): Emotional weight
        if self.config.emotional_enabled:
            emotional_score = self.emotional.compute(candidate)
            components["emotional"] = emotional_score * self.config.alpha
        else:
            components["emotional"] = 0.0

        # βA(c): Associative strength
        if self.config.associative_enabled:
            associative_score = self.associative.compute(
                candidate, self.candidates
            )
            components["associative"] = associative_score * self.config.beta
        else:
            components["associative"] = 0.0

        # γR(c): Recency
        if self.config.recency_enabled:
            recency_score = self.recency.compute(candidate, current_time)
            components["recency"] = recency_score * self.config.gamma
        else:
            components["recency"] = 0.0

        # δD(c): Decay
        if self.config.decay_enabled:
            decay_score = self.decay.compute(candidate, current_time)
            components["decay"] = decay_score * self.config.delta
        else:
            components["decay"] = 0.0

        # Final score: R_bio = S + αE + βA + γR - δD
        final_score = (
            components["semantic"]
            + components["emotional"]
            + components["associative"]
            + components["recency"]
            - components["decay"]
        )

        return final_score, components

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_ids: Optional[List[int]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k memory candidates using SRF

        Args:
            query: Query embedding
            top_k: Number of results to return (uses config if None)
            min_score: Minimum score threshold (uses config if None)
            filter_ids: Only consider these candidate IDs (considers all if None)

        Returns:
            List of RetrievalResult, sorted by score (descending)

        Performance:
            - O(n) for n candidates with standard computation
            - O(n/k) with Triton kernels on GPU (parallel processing)
        """
        if top_k is None:
            top_k = self.config.top_k
        if min_score is None:
            min_score = self.config.min_score

        current_time = time.time()
        results = []

        # Filter candidates
        if filter_ids is not None:
            candidates_to_process = {
                cid: c for cid, c in self.candidates.items() if cid in filter_ids
            }
        else:
            candidates_to_process = self.candidates

        # Compute scores for all candidates
        if self.config.use_triton and len(candidates_to_process) > 100:
            # Use optimized Triton kernel for batch processing
            results = self._retrieve_batch_triton(
                query, candidates_to_process, current_time
            )
        else:
            # Standard computation
            for candidate in candidates_to_process.values():
                score, components = self.compute_score(query, candidate, current_time)

                if score >= min_score:
                    results.append(
                        RetrievalResult(
                            candidate=candidate,
                            score=score,
                            components=components,
                        )
                    )

        # Sort by score (descending) and take top-k
        results.sort(reverse=True)
        results = results[:top_k]

        # Update statistics
        self.total_retrievals += 1
        self.total_candidates_processed += len(candidates_to_process)

        # Update access tracking
        for result in results:
            result.candidate.access_count += 1
            result.candidate.last_access = current_time

        return results

    def _retrieve_batch_triton(
        self,
        query: torch.Tensor,
        candidates: Dict[int, MemoryCandidate],
        current_time: float,
    ) -> List[RetrievalResult]:
        """
        Batch retrieval using optimized Triton kernels

        This provides significant speedup on GPU for large candidate sets.
        """
        # Prepare batch tensors
        candidate_ids = list(candidates.keys())
        n_candidates = len(candidate_ids)

        # Stack embeddings
        embeddings = torch.stack([
            candidates[cid].content for cid in candidate_ids
        ])

        # Prepare metadata tensors
        emotional_scores = torch.tensor([
            candidates[cid].emotional_score for cid in candidate_ids
        ], device=embeddings.device)

        timestamps = torch.tensor([
            candidates[cid].timestamp for cid in candidate_ids
        ], device=embeddings.device)

        last_access = torch.tensor([
            candidates[cid].last_access for cid in candidate_ids
        ], device=embeddings.device)

        # Use Triton kernel for batch scoring
        from ai_assistant_pro.srf.kernels import srf_batch_score

        scores, component_dict = srf_batch_score(
            query=query,
            embeddings=embeddings,
            emotional_scores=emotional_scores,
            timestamps=timestamps,
            last_access=last_access,
            current_time=current_time,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            delta=self.config.delta,
            decay_half_life=self.config.decay_half_life,
        )

        # Build results
        results = []
        for i, cid in enumerate(candidate_ids):
            components = {
                "semantic": component_dict["semantic"][i].item(),
                "emotional": component_dict["emotional"][i].item(),
                "associative": component_dict["associative"][i].item(),
                "recency": component_dict["recency"][i].item(),
                "decay": component_dict["decay"][i].item(),
            }

            results.append(
                RetrievalResult(
                    candidate=candidates[cid],
                    score=scores[i].item(),
                    components=components,
                )
            )

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get SRF statistics

        Returns:
            Dictionary with statistics
        """
        return {
            "total_candidates": len(self.candidates),
            "total_retrievals": self.total_retrievals,
            "total_candidates_processed": self.total_candidates_processed,
            "avg_candidates_per_retrieval": (
                self.total_candidates_processed / self.total_retrievals
                if self.total_retrievals > 0
                else 0
            ),
            "config": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "delta": self.config.delta,
            },
        }

    def clear(self) -> None:
        """Clear all memory candidates"""
        self.candidates.clear()
        self.next_id = 0

    def __len__(self) -> int:
        """Get number of memory candidates"""
        return len(self.candidates)

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"StoneRetrievalFunction("
            f"candidates={len(self.candidates)}, "
            f"α={self.config.alpha}, β={self.config.beta}, "
            f"γ={self.config.gamma}, δ={self.config.delta})"
        )
