import math
import time

import torch

from ai_assistant_pro.srf.components import (
    SemanticSimilarity,
    EmotionalWeighting,
    AssociativeStrength,
    RecencyTracker,
    DecayModel,
)
from ai_assistant_pro.srf.core import MemoryCandidate, SRFConfig, StoneRetrievalFunction


def test_semantic_similarity_basic():
    comp = SemanticSimilarity(model="cosine", device="cpu")
    v = torch.tensor([1.0, 0.0])
    assert torch.allclose(comp.compute(v, v), torch.tensor(1.0))

    batch = comp.batch_compute(v, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.allclose(batch[0], torch.tensor(1.0))
    assert 0.0 <= batch[1].item() <= 1.0


def test_emotional_weighting_and_keywords():
    ew = EmotionalWeighting()
    candidate = MemoryCandidate(id=1, content=torch.tensor([0.0]), emotional_score=0.8)
    assert ew.compute(candidate) == 0.8

    weight = EmotionalWeighting.from_keywords("important urgent", {"urgent": 0.9})
    assert weight > 0.0


def test_associative_strength_saturation():
    assoc = AssociativeStrength(association_boost=0.2)
    candidate = MemoryCandidate(id=1, content=torch.tensor([0.0]), associations=list(range(10)))
    assert assoc.compute(candidate, {}) == 1.0  # saturates at 1.0


def test_recency_and_decay_models():
    now = time.time()
    recency = RecencyTracker(time_scale=10.0)
    decay = DecayModel(half_life=10.0)

    recent = MemoryCandidate(id=1, content=torch.tensor([0.0]), last_access=now)
    old = MemoryCandidate(id=2, content=torch.tensor([0.0]), last_access=now - 20)

    assert recency.compute(recent, now) > recency.compute(old, now)

    created = MemoryCandidate(id=3, content=torch.tensor([0.0]), timestamp=now - 10)
    score = decay.compute(created, now)
    assert 0.4 < score < 0.6  # around half-life


def test_srf_retrieve_deterministic():
    config = SRFConfig(use_triton=False, use_gpu=False, top_k=2)
    srf = StoneRetrievalFunction(config)

    # Two candidates; make one clearly more similar
    base = torch.tensor([1.0, 0.0])
    cand1 = MemoryCandidate(id=1, content=base, emotional_score=0.1)
    cand2 = MemoryCandidate(id=2, content=torch.tensor([0.0, 1.0]), emotional_score=0.9)

    srf.add_candidate(cand1)
    srf.add_candidate(cand2)

    results = srf.retrieve(base, top_k=2)
    top_ids = [r.candidate.id for r in results]
    assert top_ids[0] == 1  # semantic similarity wins
    assert len(results) == 2
