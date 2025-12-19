"""
Embedding generation for RAG system.
Handles encoding text into vector representations.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RAGConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Generates embeddings for text using sentence transformers."""

    def __init__(self, config: RAGConfig):
        """Initialize embedder."""
        self.config = config
        self.model_name = config.embedding_model
        self.model = None
        self.device = self._get_device()

        logger.info(f"Embedder initialized with model: {self.model_name}")

    def _get_device(self) -> str:
        """Determine appropriate device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self) -> None:
        """Load the embedding model."""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            logger.info(f"Model loaded on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise

    def embed_text(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text.
            normalize: Whether to normalize embedding.

        Returns:
            Embedding vector as numpy array.
        """
        if self.model is None:
            self.load_model()

        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.
            batch_size: Batch size for encoding.
            normalize: Whether to normalize embeddings.
            show_progress: Whether to show progress bar.

        Returns:
            2D numpy array of embeddings.
        """
        if self.model is None:
            self.load_model()

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )

            logger.info(f"Generated embeddings for {len(texts)} texts, shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}", exc_info=True)
            raise

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings.

        Returns:
            Embedding dimension.
        """
        if self.model is None:
            self.load_model()

        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine",
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            metric: Similarity metric ("cosine" or "dot").

        Returns:
            Similarity score.
        """
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        elif metric == "dot":
            # Dot product
            similarity = np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

        return float(similarity)

    def compute_similarity_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: Optional[np.ndarray] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute similarity matrix between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (N x D).
            embeddings2: Second set of embeddings (M x D). If None, uses embeddings1.
            metric: Similarity metric ("cosine" or "dot").

        Returns:
            Similarity matrix (N x M).
        """
        if embeddings2 is None:
            embeddings2 = embeddings1

        if metric == "cosine":
            # Normalize embeddings
            embeddings1_norm = embeddings1 / np.linalg.norm(
                embeddings1, axis=1, keepdims=True
            )
            embeddings2_norm = embeddings2 / np.linalg.norm(
                embeddings2, axis=1, keepdims=True
            )
            similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)

        elif metric == "dot":
            similarity_matrix = np.dot(embeddings1, embeddings2.T)

        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

        return similarity_matrix

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            logger.info("Unloading embedding model")
            del self.model
            self.model = None

            # Clear CUDA cache if applicable
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup on deletion."""
        self.unload_model()
