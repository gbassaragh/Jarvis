"""
Stone Retrieval Function (SRF) - Patented biologically-inspired memory retrieval

Patent Information:
The Stone Retrieval Function implements a biologically-inspired approach to memory
retrieval in AI systems, combining semantic similarity, emotional weighting,
associative context, temporal recency, and decay modeling.

Formula:
R_bio = S(q, c) + αE(c) + βA(c) + γR(c) − δD(c)

Where:
- S(q, c): Semantic similarity between query q and candidate c
- E(c): Emotional weight of candidate c
- A(c): Associative strength of candidate c
- R(c): Recency score of candidate c
- D(c): Decay score of candidate c
- α, β, γ, δ: Tunable hyperparameters

Applications:
- Intelligent KV-cache management
- Context-aware memory retrieval
- Priority-based request scheduling
- Long-term conversation continuity
"""

from ai_assistant_pro.srf.core import (
    StoneRetrievalFunction,
    SRFConfig,
    MemoryCandidate,
    RetrievalResult,
)
from ai_assistant_pro.srf.components import (
    SemanticSimilarity,
    EmotionalWeighting,
    AssociativeStrength,
    RecencyTracker,
    DecayModel,
)
from ai_assistant_pro.srf.kernels import (
    srf_batch_score,
    srf_top_k_retrieval,
)

__all__ = [
    "StoneRetrievalFunction",
    "SRFConfig",
    "MemoryCandidate",
    "RetrievalResult",
    "SemanticSimilarity",
    "EmotionalWeighting",
    "AssociativeStrength",
    "RecencyTracker",
    "DecayModel",
    "srf_batch_score",
    "srf_top_k_retrieval",
]
