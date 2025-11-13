"""
Basic example: Stone Retrieval Function (SRF)

Demonstrates the patented biologically-inspired memory retrieval system.

Formula: R_bio = S(q, c) + αE(c) + βA(c) + γR(c) − δD(c)
"""

import torch
import time

from ai_assistant_pro.srf import (
    StoneRetrievalFunction,
    SRFConfig,
    MemoryCandidate,
)


def main():
    print("=" * 70)
    print("Stone Retrieval Function (SRF) - Basic Example")
    print("Patented biologically-inspired memory retrieval")
    print("=" * 70)

    # 1. Configure SRF
    print("\n1. Configuring SRF...")
    config = SRFConfig(
        alpha=0.3,   # Emotional weight
        beta=0.2,    # Associative strength
        gamma=0.25,  # Recency
        delta=0.15,  # Decay
        decay_half_life=3600.0,  # 1 hour
        use_triton=True,  # Use GPU acceleration
    )

    # Initialize SRF
    srf = StoneRetrievalFunction(config)
    print(f"✓ SRF initialized: {srf}")

    # 2. Add memory candidates
    print("\n2. Adding memory candidates...")

    # Candidate 1: Important recent memory
    candidate1 = MemoryCandidate(
        id=0,
        content=torch.randn(768, device="cuda"),  # Embedding
        text="Critical system update completed",
        emotional_score=0.9,  # High importance
        associations=[],
        timestamp=time.time(),
    )
    srf.add_candidate(candidate1)

    # Candidate 2: Older, less important
    candidate2 = MemoryCandidate(
        id=1,
        content=torch.randn(768, device="cuda"),
        text="Routine log message",
        emotional_score=0.3,  # Low importance
        associations=[],
        timestamp=time.time() - 7200,  # 2 hours ago
    )
    srf.add_candidate(candidate2)

    # Candidate 3: Moderately important, associated with candidate 1
    candidate3 = MemoryCandidate(
        id=2,
        content=torch.randn(768, device="cuda"),
        text="System update verification",
        emotional_score=0.7,
        associations=[0],  # Associated with candidate 1
        timestamp=time.time() - 1800,  # 30 minutes ago
    )
    srf.add_candidate(candidate3)

    # Candidate 4: Recent but low importance
    candidate4 = MemoryCandidate(
        id=3,
        content=torch.randn(768, device="cuda"),
        text="Debug message",
        emotional_score=0.2,
        associations=[],
        timestamp=time.time() - 300,  # 5 minutes ago
    )
    srf.add_candidate(candidate4)

    print(f"✓ Added {len(srf)} memory candidates")

    # 3. Retrieve relevant memories
    print("\n3. Retrieving relevant memories...")

    # Create query (similar to candidate 1)
    query = candidate1.content + torch.randn(768, device="cuda") * 0.1

    # Retrieve top-3 candidates
    results = srf.retrieve(query, top_k=3)

    print(f"\nTop-3 retrieved memories:")
    for i, result in enumerate(results, 1):
        print(f"\n  Rank {i}: {result.candidate.text}")
        print(f"    Final Score: {result.score:.4f}")
        print(f"    Components:")
        print(f"      - Semantic:     {result.components['semantic']:.4f}")
        print(f"      - Emotional:    {result.components['emotional']:.4f}")
        print(f"      - Associative:  {result.components['associative']:.4f}")
        print(f"      - Recency:      {result.components['recency']:.4f}")
        print(f"      - Decay:        {result.components['decay']:.4f}")
        print(f"    Access count: {result.candidate.access_count}")

    # 4. Demonstrate component effects
    print("\n4. Demonstrating component effects...")

    # Query semantically similar to candidate 2
    query2 = candidate2.content + torch.randn(768, device="cuda") * 0.05

    results2 = srf.retrieve(query2, top_k=1)
    print(f"\n  Query similar to low-importance memory:")
    print(f"    Retrieved: {results2[0].candidate.text}")
    print(f"    Score: {results2[0].score:.4f}")
    print(f"    Note: Lower score due to low emotional weight and older age")

    # 5. Statistics
    print("\n5. SRF Statistics:")
    stats = srf.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # 6. Demonstrate decay effect
    print("\n6. Demonstrating decay effect...")
    print("  Simulating time passage...")

    # Fast-forward candidate 4's timestamp to show decay
    candidate4.timestamp = time.time() - 7200  # Make it 2 hours old

    results3 = srf.retrieve(query, top_k=4)
    print(f"\n  After candidate 4 aged:")
    for i, result in enumerate(results3, 1):
        print(f"    {i}. {result.candidate.text}: {result.score:.4f} "
              f"(decay: {result.components['decay']:.4f})")

    print("\n" + "=" * 70)
    print("✓ SRF Basic Example Complete!")
    print("=" * 70)

    print("\nKey Insights:")
    print("  • Semantic similarity finds relevant memories")
    print("  • Emotional weight boosts important memories")
    print("  • Associations strengthen related memories")
    print("  • Recency favors recent memories")
    print("  • Decay de-prioritizes old memories")
    print("\nFormula: R_bio = S(q,c) + αE(c) + βA(c) + γR(c) − δD(c)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: This example requires CUDA")
    else:
        main()
