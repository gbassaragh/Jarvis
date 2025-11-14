"""
Unit tests for Stone Retrieval Function (SRF)
"""

import pytest
import torch
import time
import numpy as np

from ai_assistant_pro.srf import (
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


class TestSRFConfig:
    """Test SRF configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SRFConfig()
        assert config.alpha == 0.3
        assert config.beta == 0.2
        assert config.gamma == 0.25
        assert config.delta == 0.15
        assert config.top_k == 10

    def test_custom_config(self):
        """Test custom configuration"""
        config = SRFConfig(
            alpha=0.4,
            beta=0.3,
            gamma=0.2,
            delta=0.1,
            top_k=5,
        )
        assert config.alpha == 0.4
        assert config.beta == 0.3
        assert config.gamma == 0.2
        assert config.delta == 0.1
        assert config.top_k == 5

    def test_config_validation(self):
        """Test configuration validation"""
        config = SRFConfig()
        config.validate()  # Should not raise

        # Test invalid config
        invalid_config = SRFConfig(alpha=-0.1)
        with pytest.raises(AssertionError):
            invalid_config.validate()


class TestMemoryCandidate:
    """Test memory candidate"""

    def test_candidate_creation(self):
        """Test creating a memory candidate"""
        embedding = torch.randn(768)
        candidate = MemoryCandidate(
            id=0,
            content=embedding,
            text="Test memory",
            emotional_score=0.8,
        )

        assert candidate.id == 0
        assert candidate.text == "Test memory"
        assert candidate.emotional_score == 0.8
        assert torch.allclose(candidate.content, embedding)

    def test_candidate_defaults(self):
        """Test candidate default values"""
        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
        )

        assert candidate.emotional_score == 0.0
        assert candidate.associations == []
        assert candidate.access_count == 0


class TestSemanticSimilarity:
    """Test semantic similarity component"""

    def test_cosine_similarity(self):
        """Test cosine similarity"""
        semantic = SemanticSimilarity(model="cosine", device="cpu")

        # Identical vectors
        v = torch.randn(768)
        sim = semantic.compute(v, v)
        assert torch.allclose(sim, torch.tensor(1.0), atol=1e-6)

        # Orthogonal vectors (approximately)
        v1 = torch.randn(768)
        v2 = torch.randn(768)
        sim = semantic.compute(v1, v2)
        assert 0.0 <= sim.item() <= 1.0

    def test_dot_similarity(self):
        """Test dot product similarity"""
        semantic = SemanticSimilarity(model="dot", device="cpu")

        v1 = torch.ones(768)
        v2 = torch.ones(768)
        sim = semantic.compute(v1, v2)
        assert sim.item() == 768.0

    def test_batch_compute(self):
        """Test batch similarity computation"""
        semantic = SemanticSimilarity(model="cosine", device="cpu")

        query = torch.randn(768)
        candidates = torch.randn(10, 768)

        sims = semantic.batch_compute(query, candidates)
        assert sims.shape == (10,)
        assert torch.all((sims >= 0) & (sims <= 1))


class TestEmotionalWeighting:
    """Test emotional weighting component"""

    def test_emotional_compute(self):
        """Test emotional weight computation"""
        emotional = EmotionalWeighting()

        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            emotional_score=0.7,
        )

        weight = emotional.compute(candidate)
        assert weight == 0.7

    def test_emotional_clamping(self):
        """Test emotional score clamping"""
        emotional = EmotionalWeighting()

        # Test upper bound
        candidate_high = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            emotional_score=1.5,
        )
        assert emotional.compute(candidate_high) == 1.0

        # Test lower bound
        candidate_low = MemoryCandidate(
            id=1,
            content=torch.randn(768),
            emotional_score=-0.5,
        )
        assert emotional.compute(candidate_low) == 0.0

    def test_sentiment_to_emotion(self):
        """Test sentiment to emotional weight conversion"""
        # Strong positive sentiment
        weight = EmotionalWeighting.from_sentiment(0.9)
        assert weight == 0.9

        # Strong negative sentiment (high absolute value)
        weight = EmotionalWeighting.from_sentiment(-0.9)
        assert weight == 0.9


class TestAssociativeStrength:
    """Test associative strength component"""

    def test_no_associations(self):
        """Test candidate with no associations"""
        associative = AssociativeStrength(association_boost=0.1)

        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            associations=[],
        )

        strength = associative.compute(candidate, {})
        assert strength == 0.0

    def test_with_associations(self):
        """Test candidate with associations"""
        associative = AssociativeStrength(association_boost=0.1)

        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            associations=[1, 2, 3],
        )

        strength = associative.compute(candidate, {})
        assert strength == 0.3  # 3 * 0.1

    def test_saturation(self):
        """Test association saturation at 1.0"""
        associative = AssociativeStrength(association_boost=0.1)

        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            associations=list(range(20)),  # 20 associations
        )

        strength = associative.compute(candidate, {})
        assert strength == 1.0  # Saturated


class TestRecencyTracker:
    """Test recency tracking component"""

    def test_recent_access(self):
        """Test recency for recently accessed candidate"""
        recency = RecencyTracker(time_scale=3600.0)

        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            last_access=time.time(),
        )

        score = recency.compute(candidate, time.time())
        assert score > 0.99  # Should be ~1.0

    def test_old_access(self):
        """Test recency for old candidate"""
        recency = RecencyTracker(time_scale=3600.0)

        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            last_access=time.time() - 7200,  # 2 hours ago
        )

        score = recency.compute(candidate, time.time())
        assert score < 0.5  # Should have decayed

    def test_exponential_decay(self):
        """Test exponential decay property"""
        recency = RecencyTracker(time_scale=3600.0)

        current_time = time.time()

        # Half time scale should give ~exp(-1) ≈ 0.368
        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            last_access=current_time - 3600,
        )

        score = recency.compute(candidate, current_time)
        expected = np.exp(-1)
        assert abs(score - expected) < 0.01


class TestDecayModel:
    """Test decay model component"""

    def test_no_decay_at_creation(self):
        """Test that newly created candidates have no decay"""
        decay = DecayModel(half_life=3600.0)

        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            timestamp=time.time(),
        )

        score = decay.compute(candidate, time.time())
        assert score < 0.01  # Should be ~0

    def test_half_life_decay(self):
        """Test decay at half-life"""
        decay = DecayModel(half_life=3600.0, curve="exponential")

        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            timestamp=time.time() - 3600,  # One half-life ago
        )

        score = decay.compute(candidate, time.time())
        assert abs(score - 0.5) < 0.01  # Should be ~0.5

    def test_different_curves(self):
        """Test different decay curves"""
        current_time = time.time()
        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            timestamp=current_time - 3600,
        )

        decay_exp = DecayModel(half_life=3600.0, curve="exponential")
        decay_power = DecayModel(half_life=3600.0, curve="power")
        decay_log = DecayModel(half_life=3600.0, curve="logarithmic")

        score_exp = decay_exp.compute(candidate, current_time)
        score_power = decay_power.compute(candidate, current_time)
        score_log = decay_log.compute(candidate, current_time)

        # All should be in [0, 1]
        assert 0 <= score_exp <= 1
        assert 0 <= score_power <= 1
        assert 0 <= score_log <= 1

        # Log should decay slowest
        assert score_log < score_exp
        assert score_log < score_power


class TestStoneRetrievalFunction:
    """Test main SRF class"""

    @pytest.fixture
    def srf(self):
        """Create SRF instance for testing"""
        config = SRFConfig(
            alpha=0.3,
            beta=0.2,
            gamma=0.25,
            delta=0.15,
            use_triton=False,  # Use CPU for tests
            use_gpu=False,
        )
        return StoneRetrievalFunction(config)

    def test_add_candidate(self, srf):
        """Test adding candidates"""
        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
            text="Test",
        )

        candidate_id = srf.add_candidate(candidate)
        assert candidate_id == 0
        assert len(srf) == 1

    def test_remove_candidate(self, srf):
        """Test removing candidates"""
        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
        )

        srf.add_candidate(candidate)
        assert len(srf) == 1

        srf.remove_candidate(0)
        assert len(srf) == 0

    def test_compute_score(self, srf):
        """Test score computation"""
        embedding = torch.randn(768)
        candidate = MemoryCandidate(
            id=0,
            content=embedding,
            emotional_score=0.8,
            associations=[],
        )

        srf.add_candidate(candidate)

        # Query similar to candidate
        query = embedding + torch.randn(768) * 0.1

        score, components = srf.compute_score(query, candidate)

        # Check score is reasonable
        assert isinstance(score, float)

        # Check all components present
        assert "semantic" in components
        assert "emotional" in components
        assert "associative" in components
        assert "recency" in components
        assert "decay" in components

    def test_retrieve(self, srf):
        """Test retrieval"""
        # Add multiple candidates
        for i in range(10):
            candidate = MemoryCandidate(
                id=i,
                content=torch.randn(768),
                text=f"Candidate {i}",
                emotional_score=i / 10.0,
            )
            srf.add_candidate(candidate)

        # Retrieve top-3
        query = torch.randn(768)
        results = srf.retrieve(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

        # Check sorted by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_updates_access(self, srf):
        """Test that retrieval updates access tracking"""
        candidate = MemoryCandidate(
            id=0,
            content=torch.randn(768),
        )
        srf.add_candidate(candidate)

        initial_count = candidate.access_count

        query = torch.randn(768)
        results = srf.retrieve(query, top_k=1)

        # Access count should increase
        assert results[0].candidate.access_count == initial_count + 1

    def test_statistics(self, srf):
        """Test statistics tracking"""
        # Add candidates
        for i in range(5):
            candidate = MemoryCandidate(
                id=i,
                content=torch.randn(768),
            )
            srf.add_candidate(candidate)

        # Perform retrievals
        query = torch.randn(768)
        srf.retrieve(query, top_k=3)
        srf.retrieve(query, top_k=3)

        stats = srf.get_statistics()

        assert stats["total_candidates"] == 5
        assert stats["total_retrievals"] == 2
        assert stats["total_candidates_processed"] == 10  # 5 * 2

    def test_clear(self, srf):
        """Test clearing all candidates"""
        for i in range(5):
            candidate = MemoryCandidate(
                id=i,
                content=torch.randn(768),
            )
            srf.add_candidate(candidate)

        assert len(srf) == 5

        srf.clear()
        assert len(srf) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
