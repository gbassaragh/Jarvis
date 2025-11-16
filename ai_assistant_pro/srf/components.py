"""
Individual components of the Stone Retrieval Function

Each component implements one part of the R_bio equation:
- S(q, c): Semantic Similarity
- E(c): Emotional Weighting
- A(c): Associative Strength
- R(c): Recency Tracker
- D(c): Decay Model
"""

import math
import time
from typing import Dict, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ai_assistant_pro.srf.core import MemoryCandidate  # pragma: no cover


class SemanticSimilarity:
    """
    S(q, c): Semantic similarity between query and candidate

    Computes how semantically similar a memory candidate is to the query.
    Supports multiple similarity metrics.
    """

    def __init__(self, model: str = "cosine", device: str = "cuda") -> None:
        """
        Initialize semantic similarity module

        Args:
            model: Similarity metric ("cosine", "dot", "euclidean")
            device: Device for computation
        """
        self.model = model
        self.device = device

        assert model in ["cosine", "dot", "euclidean"], \
            f"Unknown model: {model}. Use 'cosine', 'dot', or 'euclidean'"

    def compute(self, query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic similarity

        Args:
            query: Query embedding [dim]
            candidate: Candidate embedding [dim]

        Returns:
            Similarity score (0-1 for cosine, unbounded for others)
        """
        # Ensure tensors are on correct device
        query = query.to(self.device)
        candidate = candidate.to(self.device)

        if self.model == "cosine":
            # Cosine similarity: (q · c) / (||q|| ||c||)
            # Output range: [-1, 1], normalized to [0, 1]
            similarity = F.cosine_similarity(
                query.unsqueeze(0),
                candidate.unsqueeze(0),
                dim=1
            )
            # Normalize to [0, 1]
            return (similarity + 1.0) / 2.0

        if self.model == "dot":
            # Dot product similarity: q · c
            return torch.dot(query, candidate)

        if self.model == "euclidean":
            # Euclidean distance, converted to similarity
            # similarity = 1 / (1 + distance)
            distance = torch.norm(query - candidate)
            return 1.0 / (1.0 + distance)

        raise ValueError(f"Unsupported model: {self.model}")

    def batch_compute(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity for multiple candidates

        Args:
            query: Query embedding [dim]
            candidates: Candidate embeddings [n_candidates, dim]

        Returns:
            Similarity scores [n_candidates]
        """
        query = query.to(self.device)
        candidates = candidates.to(self.device)

        if self.model == "cosine":
            similarity = F.cosine_similarity(
                query.unsqueeze(0),
                candidates,
                dim=1
            )
            return (similarity + 1.0) / 2.0

        if self.model == "dot":
            return torch.matmul(candidates, query)

        if self.model == "euclidean":
            distances = torch.norm(candidates - query.unsqueeze(0), dim=1)
            return 1.0 / (1.0 + distances)

        raise ValueError(f"Unsupported model: {self.model}")


class EmotionalWeighting:
    """
    E(c): Emotional weight of candidate

    Assigns importance based on emotional significance.
    Higher scores indicate more emotionally significant memories.
    """

    def __init__(self) -> None:
        """Initialize emotional weighting module"""
        pass

    def compute(self, candidate: "MemoryCandidate") -> float:
        """
        Compute emotional weight

        Args:
            candidate: MemoryCandidate object

        Returns:
            Emotional weight (0-1)
        """
        # Use pre-computed emotional score from candidate
        # This could be computed from sentiment analysis, importance markers, etc.
        return max(0.0, min(1.0, candidate.emotional_score))

    def batch_compute(self, emotional_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute emotional weights for batch

        Args:
            emotional_scores: Pre-computed scores [n_candidates]

        Returns:
            Normalized emotional weights [n_candidates]
        """
        return torch.clamp(emotional_scores, 0.0, 1.0)

    @staticmethod
    def from_sentiment(sentiment: float) -> float:
        """
        Convert sentiment score to emotional weight

        Args:
            sentiment: Sentiment score (typically -1 to 1)

        Returns:
            Emotional weight (0-1)
        """
        # Map sentiment intensity to emotional weight
        # Strong emotions (positive or negative) are more memorable
        return abs(sentiment)

    @staticmethod
    def from_keywords(text: str, emotional_keywords: Dict[str, float]) -> float:
        """
        Compute emotional weight from keyword matching

        Args:
            text: Text content
            emotional_keywords: Dict mapping keywords to emotional weights

        Returns:
            Emotional weight (0-1)
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        weights = []

        for keyword, weight in emotional_keywords.items():
            if keyword.lower() in text_lower:
                weights.append(weight)

        if weights:
            return max(0.0, min(1.0, max(weights)))
        return 0.0


class AssociativeStrength:
    """
    A(c): Associative strength of candidate

    Measures how strongly connected a memory is to other memories.
    Memories with more associations are considered more important.
    """

    def __init__(self, association_boost: float = 0.1) -> None:
        """
        Initialize associative strength module

        Args:
            association_boost: Score boost per association
        """
        self.association_boost = association_boost

    def compute(self, candidate: "MemoryCandidate", all_candidates: Dict[int, "MemoryCandidate"]) -> float:
        """
        Compute associative strength

        Args:
            candidate: MemoryCandidate object
            all_candidates: Dictionary of all memory candidates

        Returns:
            Associative strength score (0-1)
        """
        # Count associations
        num_associations = len(candidate.associations)

        # Also check how many times this candidate is referenced by others
        referenced_count = sum(
            1 for other in all_candidates.values()
            if candidate.id in other.associations
        )

        # Combine bidirectional associations
        total_associations = num_associations + referenced_count

        # Convert to score (with saturation)
        score = min(1.0, total_associations * self.association_boost)

        return score

    def batch_compute(
        self,
        association_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute associative strength for batch

        Args:
            association_counts: Number of associations [n_candidates]

        Returns:
            Associative strength scores [n_candidates]
        """
        scores = association_counts.float() * self.association_boost
        return torch.clamp(scores, 0.0, 1.0)

    def add_association(self, candidate: "MemoryCandidate", target_id: int) -> None:
        """
        Add an association between candidates

        Args:
            candidate: Source MemoryCandidate
            target_id: Target candidate ID
        """
        if target_id not in candidate.associations:
            candidate.associations.append(target_id)


class RecencyTracker:
    """
    R(c): Recency score of candidate

    Tracks how recently a memory was accessed or created.
    More recent memories receive higher scores.
    """

    def __init__(self, time_scale: float = 3600.0) -> None:
        """
        Initialize recency tracker

        Args:
            time_scale: Time scale for recency calculation (seconds)
                       Smaller values give more weight to recent memories
        """
        self.time_scale = time_scale

    def compute(self, candidate: "MemoryCandidate", current_time: float) -> float:
        """
        Compute recency score

        Uses exponential decay based on time since last access:
        R(c) = exp(-(current_time - last_access) / time_scale)

        Args:
            candidate: MemoryCandidate object
            current_time: Current timestamp

        Returns:
            Recency score (0-1)
        """
        # Use last access time (or creation time if never accessed)
        last_time = candidate.last_access

        # Time difference
        time_diff = current_time - last_time

        # Exponential decay
        recency = math.exp(-time_diff / self.time_scale)

        return recency

    def batch_compute(
        self,
        last_access_times: torch.Tensor,
        current_time: float
    ) -> torch.Tensor:
        """
        Compute recency for batch

        Args:
            last_access_times: Last access timestamps [n_candidates]
            current_time: Current timestamp

        Returns:
            Recency scores [n_candidates]
        """
        time_diffs = current_time - last_access_times
        recency = torch.exp(-time_diffs / self.time_scale)
        return recency

    def update_access(self, candidate, current_time: Optional[float] = None) -> None:
        """
        Update access time for candidate

        Args:
            candidate: MemoryCandidate object
            current_time: Current timestamp (uses time.time() if None)
        """
        if current_time is None:
            current_time = time.time()

        candidate.last_access = current_time
        candidate.access_count += 1


class DecayModel:
    """
    D(c): Decay score for de-prioritizing stale memories

    Implements forgetting curve based on time since creation.
    Older memories decay more, reducing their retrieval probability.
    """

    def __init__(self, half_life: float = 3600.0, curve: str = "exponential") -> None:
        """
        Initialize decay model

        Args:
            half_life: Time for score to decay to 50% (seconds)
            curve: Decay curve type ("exponential", "power", "logarithmic")
        """
        self.half_life = half_life
        self.curve = curve

        assert curve in ["exponential", "power", "logarithmic"], \
            f"Unknown curve: {curve}"

    def compute(self, candidate: "MemoryCandidate", current_time: float) -> float:
        """
        Compute decay score

        Args:
            candidate: MemoryCandidate object
            current_time: Current timestamp

        Returns:
            Decay score (0-1, higher means more decay)
        """
        # Time since creation
        age = current_time - candidate.timestamp

        if self.curve == "exponential":
            # Exponential decay: D(t) = 1 - exp(-ln(2) * t / half_life)
            decay = 1.0 - math.exp(-0.693147 * age / self.half_life)

        elif self.curve == "power":
            # Power law decay: D(t) = 1 - (1 + t/half_life)^(-1)
            decay = 1.0 - (1.0 + age / self.half_life) ** (-1.0)

        elif self.curve == "logarithmic":
            # Logarithmic decay: D(t) = log(1 + t) / log(1 + max_age)
            # Slower decay than exponential
            max_age = self.half_life * 10  # Arbitrary maximum
            decay = math.log(1.0 + age) / math.log(1.0 + max_age)

        return min(1.0, max(0.0, decay))

    def batch_compute(
        self,
        timestamps: torch.Tensor,
        current_time: float
    ) -> torch.Tensor:
        """
        Compute decay for batch

        Args:
            timestamps: Creation timestamps [n_candidates]
            current_time: Current timestamp

        Returns:
            Decay scores [n_candidates]
        """
        ages = current_time - timestamps

        if self.curve == "exponential":
            decay = 1.0 - torch.exp(-0.693147 * ages / self.half_life)

        elif self.curve == "power":
            decay = 1.0 - (1.0 + ages / self.half_life) ** (-1.0)

        elif self.curve == "logarithmic":
            max_age = self.half_life * 10
            decay = torch.log(1.0 + ages) / math.log(1.0 + max_age)

        return torch.clamp(decay, 0.0, 1.0)

    def get_half_life_score(self) -> float:
        """
        Get decay score at half-life

        Returns:
            Decay score at t = half_life
        """
        if self.curve == "exponential":
            return 0.5  # By definition
        elif self.curve == "power":
            return 0.5  # Also 0.5 by design
        else:
            # Approximate for logarithmic
            return 0.3


class ContextualModulator:
    """
    Optional: Contextual modulation of SRF scores

    Adjusts retrieval scores based on current context, user state, etc.
    This can be used to further customize the SRF for specific applications.
    """

    def __init__(self):
        """Initialize contextual modulator"""
        self.context_weights = {}

    def set_context(self, context_key: str, weight: float) -> None:
        """
        Set context weight

        Args:
            context_key: Context identifier
            weight: Modulation weight
        """
        self.context_weights[context_key] = weight

    def modulate(
        self,
        score: float,
        candidate,
        context_keys: Optional[list] = None
    ) -> float:
        """
        Modulate score based on context

        Args:
            score: Base SRF score
            candidate: MemoryCandidate object
            context_keys: Active context keys

        Returns:
            Modulated score
        """
        if not context_keys:
            return score

        # Apply context-based modulation
        modulation = 1.0
        for key in context_keys:
            if key in candidate.metadata.get("contexts", []):
                modulation *= self.context_weights.get(key, 1.0)

        return score * modulation
