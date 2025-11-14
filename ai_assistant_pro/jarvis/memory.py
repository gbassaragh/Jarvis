"""
JARVIS Conversational Memory System

Long-term memory for conversations using the Stone Retrieval Function (SRF).
Remembers past interactions, user preferences, and context.
"""

import torch
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ai_assistant_pro.srf import (
    StoneRetrievalFunction,
    SRFConfig,
    MemoryCandidate,
    RetrievalResult,
)
from ai_assistant_pro.utils.logging import get_logger

logger = get_logger("jarvis.memory")


@dataclass
class ConversationTurn:
    """A single turn in a conversation"""

    turn_id: int
    timestamp: float
    user_message: str
    assistant_response: str
    embedding: Optional[torch.Tensor] = None
    emotional_score: float = 0.5
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    importance: float = 0.5  # User-rated or auto-computed
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile with preferences and information"""

    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)  # Learned facts about user
    created_at: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)


class ConversationalMemory:
    """
    Long-term conversational memory using SRF

    Features:
    - Stores all conversation turns with embeddings
    - Retrieves relevant past conversations
    - Learns user preferences and facts
    - Maintains context across sessions
    - Emotional significance tracking
    """

    def __init__(
        self,
        user_id: str,
        srf_config: Optional[SRFConfig] = None,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize conversational memory

        Args:
            user_id: Unique user identifier
            srf_config: SRF configuration (uses defaults if None)
            embedding_model: Model for text embeddings
            device: Device for computation
        """
        self.user_id = user_id
        self.device = device

        # Initialize SRF for memory retrieval
        if srf_config is None:
            srf_config = SRFConfig(
                alpha=0.4,   # High emotional weight for conversations
                beta=0.3,    # Strong associations
                gamma=0.2,   # Moderate recency
                delta=0.1,   # Low decay (preserve old memories)
            )

        self.srf = StoneRetrievalFunction(srf_config)

        # Initialize embedding model
        self._init_embedding_model(embedding_model)

        # User profile
        self.profile = UserProfile(user_id=user_id)

        # Conversation history
        self.turns: List[ConversationTurn] = []
        self.next_turn_id = 0

        # Session context (current conversation)
        self.session_context: List[ConversationTurn] = []

        logger.info(f"✓ Conversational memory initialized for user: {user_id}")

    def _init_embedding_model(self, model_name: str):
        """Initialize sentence embedding model"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            logger.info("✓ Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        emotional_score: Optional[float] = None,
        importance: Optional[float] = None,
        topics: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
    ) -> ConversationTurn:
        """
        Add a conversation turn to memory

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            emotional_score: Emotional significance (0-1, auto-computed if None)
            importance: Importance rating (0-1)
            topics: Detected topics
            entities: Detected entities

        Returns:
            Created ConversationTurn
        """
        # Create embedding
        combined_text = f"{user_message} {assistant_response}"
        embedding = self._embed_text(combined_text)

        # Auto-compute emotional score if not provided
        if emotional_score is None:
            emotional_score = self._compute_emotional_score(user_message)

        # Create turn
        turn = ConversationTurn(
            turn_id=self.next_turn_id,
            timestamp=time.time(),
            user_message=user_message,
            assistant_response=assistant_response,
            embedding=embedding,
            emotional_score=emotional_score,
            topics=topics or [],
            entities=entities or [],
            importance=importance or 0.5,
        )

        self.next_turn_id += 1
        self.turns.append(turn)

        # Add to session context
        self.session_context.append(turn)

        # Add to SRF
        memory_candidate = MemoryCandidate(
            id=turn.turn_id,
            content=embedding,
            text=combined_text,
            emotional_score=emotional_score,
            timestamp=turn.timestamp,
        )
        self.srf.add_candidate(memory_candidate)

        logger.info(f"Added turn {turn.turn_id} to memory")

        return turn

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[ConversationTurn]:
        """
        Retrieve relevant past conversations

        Args:
            query: Query text (current context)
            top_k: Number of results
            min_score: Minimum relevance score

        Returns:
            List of relevant conversation turns
        """
        # Embed query
        query_embedding = self._embed_text(query)

        # Retrieve using SRF
        results = self.srf.retrieve(
            query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        # Map to conversation turns
        relevant_turns = []
        for result in results:
            turn = self.turns[result.candidate.id]
            relevant_turns.append(turn)

        logger.info(f"Retrieved {len(relevant_turns)} relevant past conversations")

        return relevant_turns

    def get_session_context(self, max_turns: int = 10) -> List[ConversationTurn]:
        """
        Get recent session context

        Args:
            max_turns: Maximum number of turns

        Returns:
            Recent conversation turns
        """
        return self.session_context[-max_turns:]

    def clear_session(self):
        """Clear current session context"""
        self.session_context = []
        logger.info("Session context cleared")

    def update_user_preference(self, key: str, value: Any):
        """
        Update user preference

        Args:
            key: Preference key
            value: Preference value
        """
        self.profile.preferences[key] = value
        logger.info(f"Updated preference: {key} = {value}")

    def add_user_fact(self, fact_type: str, fact: Any):
        """
        Add a learned fact about the user

        Args:
            fact_type: Type of fact (e.g., "favorite_color", "birthday")
            fact: Fact value
        """
        self.profile.facts[fact_type] = fact
        logger.info(f"Learned fact: {fact_type} = {fact}")

    def get_context_summary(self, query: Optional[str] = None) -> str:
        """
        Generate a context summary for the LLM

        Args:
            query: Optional current query for relevance

        Returns:
            Formatted context string
        """
        summary_parts = []

        # User profile
        if self.profile.name:
            summary_parts.append(f"User: {self.profile.name}")

        if self.profile.preferences:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.profile.preferences.items())
            summary_parts.append(f"Preferences: {prefs}")

        # Recent context
        recent_turns = self.get_session_context(max_turns=5)
        if recent_turns:
            summary_parts.append("\nRecent conversation:")
            for turn in recent_turns[-3:]:  # Last 3 turns
                summary_parts.append(f"User: {turn.user_message}")
                summary_parts.append(f"Assistant: {turn.assistant_response}")

        # Relevant past context
        if query:
            relevant_turns = self.retrieve_relevant(query, top_k=3)
            if relevant_turns:
                summary_parts.append("\nRelevant past conversations:")
                for turn in relevant_turns:
                    date = datetime.fromtimestamp(turn.timestamp).strftime("%Y-%m-%d")
                    summary_parts.append(f"[{date}] User: {turn.user_message}")

        return "\n".join(summary_parts)

    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed text using sentence transformer"""
        if self.embedding_model is None:
            # Fallback to random embedding
            return torch.randn(768)

        embedding = self.embedding_model.encode(
            text,
            convert_to_tensor=True,
            device=self.device,
        )

        return embedding

    def _compute_emotional_score(self, text: str) -> float:
        """
        Compute emotional significance of text

        Uses simple heuristics - could be replaced with sentiment analysis
        """
        # Emotional keywords
        high_emotion_words = [
            "love", "hate", "amazing", "terrible", "wonderful",
            "horrible", "fantastic", "awful", "excited", "angry",
            "sad", "happy", "important", "critical", "urgent"
        ]

        text_lower = text.lower()
        emotion_count = sum(1 for word in high_emotion_words if word in text_lower)

        # Normalize to 0-1
        score = min(emotion_count / 3.0, 1.0)

        # Base score
        return max(0.3, score)

    def export_memory(self, filepath: str):
        """Export memory to file"""
        import json

        data = {
            "user_id": self.user_id,
            "profile": {
                "name": self.profile.name,
                "preferences": self.profile.preferences,
                "interests": self.profile.interests,
                "facts": self.profile.facts,
            },
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "timestamp": turn.timestamp,
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "emotional_score": turn.emotional_score,
                    "importance": turn.importance,
                    "topics": turn.topics,
                    "entities": turn.entities,
                }
                for turn in self.turns
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Memory exported to {filepath}")

    def import_memory(self, filepath: str):
        """Import memory from file"""
        import json

        with open(filepath) as f:
            data = json.load(f)

        # Restore profile
        self.profile.name = data["profile"].get("name")
        self.profile.preferences = data["profile"].get("preferences", {})
        self.profile.interests = data["profile"].get("interests", [])
        self.profile.facts = data["profile"].get("facts", {})

        # Restore turns
        for turn_data in data["turns"]:
            self.add_turn(
                user_message=turn_data["user_message"],
                assistant_response=turn_data["assistant_response"],
                emotional_score=turn_data.get("emotional_score", 0.5),
                importance=turn_data.get("importance", 0.5),
                topics=turn_data.get("topics", []),
                entities=turn_data.get("entities", []),
            )

        logger.info(f"Memory imported from {filepath} ({len(self.turns)} turns)")

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_turns": len(self.turns),
            "session_turns": len(self.session_context),
            "user_preferences": len(self.profile.preferences),
            "user_facts": len(self.profile.facts),
            "srf_stats": self.srf.get_statistics(),
        }


class MultiUserMemory:
    """
    Manage memory for multiple users

    Provides user-specific memory isolation and management.
    """

    def __init__(self):
        self.user_memories: Dict[str, ConversationalMemory] = {}
        logger.info("Multi-user memory manager initialized")

    def get_memory(
        self,
        user_id: str,
        create_if_missing: bool = True,
    ) -> Optional[ConversationalMemory]:
        """
        Get memory for a user

        Args:
            user_id: User identifier
            create_if_missing: Create new memory if user not found

        Returns:
            ConversationalMemory instance or None
        """
        if user_id not in self.user_memories:
            if create_if_missing:
                self.user_memories[user_id] = ConversationalMemory(user_id)
                logger.info(f"Created new memory for user: {user_id}")
            else:
                return None

        return self.user_memories[user_id]

    def remove_user(self, user_id: str):
        """Remove user memory"""
        if user_id in self.user_memories:
            del self.user_memories[user_id]
            logger.info(f"Removed memory for user: {user_id}")

    def export_all(self, directory: str):
        """Export all user memories"""
        from pathlib import Path

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        for user_id, memory in self.user_memories.items():
            filepath = dir_path / f"{user_id}_memory.json"
            memory.export_memory(str(filepath))

        logger.info(f"Exported {len(self.user_memories)} user memories to {directory}")
