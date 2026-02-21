import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RAGConfig
from utils.logger import get_logger
from rag.document_processor import Document
from rag.embedder import Embedder

logger = get_logger(__name__)


class FAISSRetriever:
    def __init__(self, config: RAGConfig, embedder: Embedder):
        self.config = config
        self.embedder = embedder
        self.index = None
        self.documents: List[Document] = []
        self.index_type = config.vector_store.get("index_type", "IndexFlatL2")
        self.persist_dir = Path(config.vector_store.get("persist_directory", "./data/vector_store"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FAISSRetriever: index_type={self.index_type}")

    def _create_index(self, dimension: int) -> faiss.Index:
        if self.index_type == "IndexFlatIP":
            # Inner product = cosine similarity when vectors are normalized
            index = faiss.IndexFlatIP(dimension)
        elif self.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif self.index_type == "IndexHNSWFlat":
            index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            if self.index_type != "IndexFlatL2":
                logger.warning(f"Unknown index type {self.index_type}, using IndexFlatL2")
            index = faiss.IndexFlatL2(dimension)
        logger.info(f"Created FAISS index: {type(index).__name__}")
        return index

    def build_index(self, documents: List[Document], show_progress: bool = True) -> None:
        if not documents:
            raise ValueError("Cannot build index from empty document list")
        logger.info(f"Building index from {len(documents)} documents")
        self.documents = documents
        embeddings = self.embedder.embed_batch([d.text for d in documents], show_progress=show_progress)
        dimension = embeddings.shape[1]
        self.index = self._create_index(dimension)
        if isinstance(self.index, faiss.IndexIVFFlat):
            logger.info("Training IVF index...")
            self.index.train(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Index built: {self.index.ntotal} vectors, dim={dimension}")

    def add_documents(self, documents: List[Document], show_progress: bool = False) -> None:
        if self.index is None:
            raise ValueError("Index not initialized. Call build_index first.")
        embeddings = self.embedder.embed_batch([d.text for d in documents], show_progress=show_progress)
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total vectors: {self.index.ntotal}")

    def retrieve(self, query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> List[Tuple[Document, float]]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        top_k = top_k or self.config.retrieval.get("top_k", 5)
        score_threshold = score_threshold if score_threshold is not None else self.config.retrieval.get("similarity_threshold", 0.0)

        query_embedding = self.embedder.embed_text(query).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 when fewer results exist than top_k
                continue
            # L2 distance → similarity: lower distance = higher score
            score = float(distance) if isinstance(self.index, faiss.IndexFlatIP) else 1.0 / (1.0 + float(distance))
            if score >= score_threshold:
                results.append((self.documents[idx], score))

        logger.debug(f"Retrieved {len(results)} documents for query")
        return results

    def format_retrieved_context(self, results: List[Tuple[Document, float]], max_length: Optional[int] = None, include_scores: bool = False) -> str:
        if not results:
            return "No relevant context found."
        context_parts = []
        current_length = 0
        for i, (doc, score) in enumerate(results, 1):
            entry = f"[Source {i}] (Relevance: {score:.3f})\n{doc.text}\n" if include_scores else f"[Source {i}]\n{doc.text}\n"
            if max_length and current_length + len(entry) > max_length:
                break
            context_parts.append(entry)
            current_length += len(entry)
        return "\n".join(context_parts)

    def save_index(self, name: str = "index") -> None:
        if self.index is None:
            raise ValueError("No index to save")
        index_path = self.persist_dir / f"{name}.faiss"
        docs_path = self.persist_dir / f"{name}_docs.pkl"
        faiss.write_index(self.index, str(index_path))
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Saved index to {index_path}")

    def load_index(self, name: str = "index") -> None:
        index_path = self.persist_dir / f"{name}.faiss"
        docs_path = self.persist_dir / f"{name}_docs.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents not found: {docs_path}")
        self.index = faiss.read_index(str(index_path))
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        logger.info(f"Loaded index: {self.index.ntotal} vectors, {len(self.documents)} docs")

    def get_statistics(self) -> Dict:
        stats = {"index_built": self.index is not None, "num_documents": len(self.documents)}
        if self.index is not None:
            stats.update({"num_vectors": self.index.ntotal, "dimension": self.index.d, "index_type": type(self.index).__name__})
        return stats

    def clear(self) -> None:
        self.index = None
        self.documents = []
        logger.info("Cleared index")
