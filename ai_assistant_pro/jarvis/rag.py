"""
JARVIS RAG (Retrieval-Augmented Generation) System

Combines document retrieval with SRF for intelligent knowledge-based responses.
"""

import torch
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib

from ai_assistant_pro.srf import StoneRetrievalFunction, SRFConfig, MemoryCandidate
from ai_assistant_pro.utils.logging import get_logger

logger = get_logger("jarvis.rag")


class Document:
    """Document for RAG system"""

    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        self.content = content
        self.metadata = metadata or {}

        # Generate ID if not provided
        if doc_id is None:
            self.doc_id = hashlib.md5(content.encode()).hexdigest()
        else:
            self.doc_id = doc_id

    def __repr__(self):
        return f"Document(id={self.doc_id}, length={len(self.content)})"


class DocumentChunker:
    """Chunk documents for better retrieval"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """
        Chunk text into overlapping segments

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())

            # Move start with overlap
            start = end - self.chunk_overlap

        return chunks


class RAGSystem:
    """
    RAG system using SRF for intelligent retrieval

    Features:
    - Document ingestion and chunking
    - SRF-powered retrieval (semantic + emotional + temporal)
    - Context-aware response generation
    - Citation tracking
    """

    def __init__(
        self,
        srf_config: Optional[SRFConfig] = None,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize RAG system

        Args:
            srf_config: SRF configuration
            embedding_model: Model for embeddings
            chunk_size: Size of document chunks
            device: Device for computation
        """
        self.device = device

        # Initialize SRF
        if srf_config is None:
            srf_config = SRFConfig(
                alpha=0.2,   # Lower emotional weight for factual docs
                beta=0.35,   # High associative strength (linked knowledge)
                gamma=0.15,  # Low recency (facts don't age)
                delta=0.05,  # Minimal decay
            )

        self.srf = StoneRetrievalFunction(srf_config)

        # Initialize embedding model
        self._init_embedding_model(embedding_model)

        # Document chunker
        self.chunker = DocumentChunker(chunk_size=chunk_size)

        # Document store
        self.documents: Dict[str, Document] = {}
        self.chunk_to_doc: Dict[int, str] = {}  # Map chunk ID to document ID

        logger.info("✓ RAG system initialized")

    def _init_embedding_model(self, model_name: str):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            logger.info("✓ Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def add_document(
        self,
        document: Document,
        importance: float = 0.5,
    ):
        """
        Add document to RAG system

        Args:
            document: Document to add
            importance: Importance score (0-1)
        """
        # Store document
        self.documents[document.doc_id] = document

        # Chunk document
        chunks = self.chunker.chunk(document.content)

        logger.info(f"Adding document {document.doc_id} ({len(chunks)} chunks)")

        # Add each chunk to SRF
        for i, chunk in enumerate(chunks):
            # Create embedding
            embedding = self._embed_text(chunk)

            # Create memory candidate
            chunk_id = len(self.chunk_to_doc)
            self.chunk_to_doc[chunk_id] = document.doc_id

            candidate = MemoryCandidate(
                id=chunk_id,
                content=embedding,
                text=chunk,
                emotional_score=importance,
                metadata={
                    "doc_id": document.doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **document.metadata,
                },
            )

            self.srf.add_candidate(candidate)

        logger.info(f"✓ Added document {document.doc_id}")

    def add_documents_from_directory(
        self,
        directory: str,
        pattern: str = "*.txt",
        importance: float = 0.5,
    ):
        """
        Add all documents from directory

        Args:
            directory: Directory path
            pattern: File pattern (glob)
            importance: Importance score for all documents
        """
        from pathlib import Path

        dir_path = Path(directory)
        files = list(dir_path.glob(pattern))

        logger.info(f"Loading {len(files)} files from {directory}")

        for filepath in files:
            with open(filepath) as f:
                content = f.read()

            doc = Document(
                content=content,
                metadata={"source": str(filepath), "filename": filepath.name},
            )

            self.add_document(doc, importance=importance)

        logger.info(f"✓ Loaded {len(files)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks

        Args:
            query: Query text
            top_k: Number of results
            min_score: Minimum relevance score

        Returns:
            List of retrieval results with context
        """
        # Embed query
        query_embedding = self._embed_text(query)

        # Retrieve using SRF
        results = self.srf.retrieve(
            query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        # Format results
        formatted_results = []
        for result in results:
            doc_id = self.chunk_to_doc[result.candidate.id]
            document = self.documents[doc_id]

            formatted_results.append({
                "text": result.candidate.text,
                "score": result.score,
                "doc_id": doc_id,
                "document": document,
                "metadata": result.candidate.metadata,
                "srf_components": result.components,
            })

        logger.info(f"Retrieved {len(formatted_results)} relevant chunks")

        return formatted_results

    def generate_response(
        self,
        query: str,
        engine,
        top_k: int = 3,
        max_tokens: int = 300,
    ) -> Dict[str, Any]:
        """
        Generate RAG response

        Args:
            query: User query
            engine: AssistantEngine for generation
            top_k: Number of context chunks
            max_tokens: Maximum response tokens

        Returns:
            Dictionary with response and sources
        """
        # Retrieve relevant context
        results = self.retrieve(query, top_k=top_k)

        # Build context
        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source {i}]\n{result['text']}\n")

            sources.append({
                "index": i,
                "score": result["score"],
                "metadata": result["metadata"],
            })

        context = "\n".join(context_parts)

        # Build prompt
        prompt = f"""Use the following context to answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        response = engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for factual responses
        )

        return {
            "response": response,
            "sources": sources,
            "context": context,
        }

    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed text"""
        if self.embedding_model is None:
            return torch.randn(768)

        embedding = self.embedding_model.encode(
            text,
            convert_to_tensor=True,
            device=self.device,
        )

        return embedding

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG statistics"""
        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunk_to_doc),
            "srf_stats": self.srf.get_statistics(),
        }


# Example usage
if __name__ == "__main__":
    from ai_assistant_pro import AssistantEngine

    # Create RAG system
    rag = RAGSystem()

    # Add document
    doc = Document(
        content="""
        Python is a high-level programming language known for its simplicity and readability.
        It was created by Guido van Rossum and first released in 1991.
        Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
        """,
        metadata={"topic": "programming", "language": "python"},
    )

    rag.add_document(doc, importance=0.8)

    # Query
    results = rag.retrieve("What is Python?", top_k=2)

    print("Retrieved context:")
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"Text: {result['text'][:100]}...")
        print()

    # Generate response
    engine = AssistantEngine(model_name="gpt2")
    response = rag.generate_response("What is Python?", engine=engine)

    print(f"Response: {response['response']}")
    print(f"Sources: {len(response['sources'])}")
