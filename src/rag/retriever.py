"""
Retrieval system using FAISS for efficient similarity search.
Handles indexing and retrieval of relevant document chunks.
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RAGConfig
from utils.logger import get_logger
from rag.document_processor import Document
from rag.embedder import Embedder

logger = get_logger(__name__)


class FAISSRetriever:
    """
    FAISS-based retrieval system for RAG.
    Indexes documents and retrieves relevant chunks based on queries.
    """

    def __init__(
        self,
        config: RAGConfig,
        embedder: Embedder,
    ):
        """
        Initialize retriever.

        Args:
            config: RAG configuration.
            embedder: Embedder instance for generating embeddings.
        """
        self.config = config
        self.embedder = embedder
        self.index = None
        self.documents: List[Document] = []
        self.index_type = config.vector_store.get("index_type", "IndexFlatL2")
        self.persist_dir = Path(config.vector_store.get("persist_directory", "./data/vector_store"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FAISSRetriever initialized with index_type: {self.index_type}")

    def _create_index(self, dimension: int) -> faiss.Index:
        """
        Create FAISS index.

        Args:
            dimension: Dimension of embeddings.

        Returns:
            FAISS index.
        """
        if self.index_type == "IndexFlatL2":
            # Exact L2 distance search (most accurate)
            index = faiss.IndexFlatL2(dimension)

        elif self.index_type == "IndexFlatIP":
            # Exact inner product search (for normalized vectors = cosine similarity)
            index = faiss.IndexFlatIP(dimension)

        elif self.index_type == "IndexIVFFlat":
            # Inverted file index (faster but approximate)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)

        elif self.index_type == "IndexHNSWFlat":
            # Hierarchical Navigable Small World graph (fast approximate search)
            index = faiss.IndexHNSWFlat(dimension, 32)

        else:
            logger.warning(f"Unknown index type: {self.index_type}, using IndexFlatL2")
            index = faiss.IndexFlatL2(dimension)

        logger.info(f"Created FAISS index: {type(index).__name__}")
        return index

    def build_index(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> None:
        """
        Build FAISS index from documents.

        Args:
            documents: List of Document objects to index.
            show_progress: Whether to show progress bar.
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list")

        logger.info(f"Building index from {len(documents)} documents")

        # Store documents
        self.documents = documents

        # Extract texts
        texts = [doc.text for doc in documents]

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.embed_batch(
            texts,
            show_progress=show_progress,
        )

        # Create index
        dimension = embeddings.shape[1]
        self.index = self._create_index(dimension)

        # For IVF indices, need to train
        if isinstance(self.index, faiss.IndexIVFFlat):
            logger.info("Training IVF index...")
            self.index.train(embeddings.astype('float32'))

        # Add embeddings to index
        logger.info("Adding embeddings to index...")
        self.index.add(embeddings.astype('float32'))

        logger.info(
            f"Index built successfully. Total vectors: {self.index.ntotal}, "
            f"Dimension: {dimension}"
        )

    def add_documents(
        self,
        documents: List[Document],
        show_progress: bool = False,
    ) -> None:
        """
        Add new documents to existing index.

        Args:
            documents: List of Document objects to add.
            show_progress: Whether to show progress bar.
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call build_index first.")

        logger.info(f"Adding {len(documents)} documents to index")

        # Extract texts
        texts = [doc.text for doc in documents]

        # Generate embeddings
        embeddings = self.embedder.embed_batch(
            texts,
            show_progress=show_progress,
        )

        # Add to index
        self.index.add(embeddings.astype('float32'))

        # Store documents
        self.documents.extend(documents)

        logger.info(f"Added documents. Total vectors: {self.index.ntotal}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string.
            top_k: Number of top results to return. Uses config default if None.
            score_threshold: Minimum similarity score. Uses config default if None.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        # Use config defaults if not specified
        if top_k is None:
            top_k = self.config.retrieval.get("top_k", 5)
        if score_threshold is None:
            score_threshold = self.config.retrieval.get("similarity_threshold", 0.0)

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search index
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert to list of (document, score) tuples
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for not found
                continue

            # Convert distance to similarity score
            # For L2 distance, lower is better, so we invert
            # For IP (cosine), higher is better
            if isinstance(self.index, faiss.IndexFlatIP):
                score = float(distance)
            else:
                # L2 distance: convert to similarity (inverse)
                score = 1.0 / (1.0 + float(distance))

            # Apply threshold
            if score < score_threshold:
                continue

            doc = self.documents[idx]
            results.append((doc, score))

        logger.debug(f"Retrieved {len(results)} documents for query")
        return results

    def retrieve_with_metadata_filter(
        self,
        query: str,
        metadata_filter: Dict[str, any],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with metadata filtering.

        Args:
            query: Query string.
            metadata_filter: Dictionary of metadata key-value pairs to filter by.
            top_k: Number of results.

        Returns:
            Filtered list of (Document, score) tuples.
        """
        # Get all results
        all_results = self.retrieve(query, top_k=top_k * 3)  # Get more to filter

        # Filter by metadata
        filtered_results = []
        for doc, score in all_results:
            match = all(
                doc.metadata.get(key) == value
                for key, value in metadata_filter.items()
            )
            if match:
                filtered_results.append((doc, score))

            if len(filtered_results) >= (top_k or 5):
                break

        logger.debug(
            f"Filtered {len(all_results)} results to {len(filtered_results)} "
            f"matching {metadata_filter}"
        )
        return filtered_results

    def format_retrieved_context(
        self,
        results: List[Tuple[Document, float]],
        max_length: Optional[int] = None,
        include_scores: bool = False,
    ) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            results: List of (Document, score) tuples.
            max_length: Maximum length of context. If None, no limit.
            include_scores: Whether to include relevance scores.

        Returns:
            Formatted context string.
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        current_length = 0

        for i, (doc, score) in enumerate(results, 1):
            # Format entry
            if include_scores:
                entry = f"[Source {i}] (Relevance: {score:.3f})\n{doc.text}\n"
            else:
                entry = f"[Source {i}]\n{doc.text}\n"

            # Check length limit
            if max_length and current_length + len(entry) > max_length:
                break

            context_parts.append(entry)
            current_length += len(entry)

        context = "\n".join(context_parts)
        logger.debug(f"Formatted context with {len(results)} sources, {len(context)} chars")

        return context

    def save_index(self, name: str = "index") -> None:
        """
        Save index and documents to disk.

        Args:
            name: Name for saved files.
        """
        if self.index is None:
            raise ValueError("No index to save")

        index_path = self.persist_dir / f"{name}.faiss"
        docs_path = self.persist_dir / f"{name}_docs.pkl"

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save documents
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)

        logger.info(f"Saved index to {index_path} and documents to {docs_path}")

    def load_index(self, name: str = "index") -> None:
        """
        Load index and documents from disk.

        Args:
            name: Name of saved files.
        """
        index_path = self.persist_dir / f"{name}.faiss"
        docs_path = self.persist_dir / f"{name}_docs.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents file not found: {docs_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load documents
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)

        logger.info(
            f"Loaded index from {index_path}. "
            f"Total vectors: {self.index.ntotal}, Documents: {len(self.documents)}"
        )

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index statistics.
        """
        stats = {
            "index_built": self.index is not None,
            "num_documents": len(self.documents),
        }

        if self.index is not None:
            stats["num_vectors"] = self.index.ntotal
            stats["dimension"] = self.index.d
            stats["index_type"] = type(self.index).__name__

        return stats

    def clear(self) -> None:
        """Clear index and documents."""
        self.index = None
        self.documents = []
        logger.info("Cleared index and documents")
