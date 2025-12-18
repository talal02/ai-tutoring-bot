"""
RAG (Retrieval-Augmented Generation) Layer for the Large Tutoring Models system.
Handles document processing, embedding, and retrieval.
"""

from .document_processor import DocumentProcessor
from .embedder import Embedder
from .retriever import FAISSRetriever

__all__ = ['DocumentProcessor', 'Embedder', 'FAISSRetriever']
