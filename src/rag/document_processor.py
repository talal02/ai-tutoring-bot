"""
Document processing utilities for RAG system.
Handles loading, chunking, and cleaning of curriculum materials.
"""

import json
import re
from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RAGConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Represents a single document or chunk."""
    text: str
    metadata: Dict[str, any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """
    Processes documents for RAG system.
    Handles loading, cleaning, and chunking.
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize document processor.

        Args:
            config: RAG configuration object.
        """
        self.config = config
        self.chunk_size = config.chunking["chunk_size"]
        self.chunk_overlap = config.chunking["chunk_overlap"]
        self.separator = config.chunking["separator"]

        logger.info(
            f"DocumentProcessor initialized with chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def load_json_dataset(self, file_path: str) -> List[Document]:
        """
        Load Q&A dataset from JSON file.

        Args:
            file_path: Path to JSON file.

        Returns:
            List of Document objects.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If JSON is malformed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading dataset from {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        for item in data:
            # Combine question and answer as context
            text = f"Question: {item.get('question', '')}\n\nAnswer: {item.get('answer', '')}"

            doc = Document(
                text=text,
                metadata={
                    "topic": item.get("topic", "unknown"),
                    "source": "dataset",
                    "question": item.get("question", ""),
                }
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from dataset")
        return documents

    def load_text_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None,
    ) -> Document:
        """
        Load a single text file.

        Args:
            file_path: Path to text file.
            metadata: Optional metadata dictionary.

        Returns:
            Document object.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        doc_metadata = metadata or {}
        doc_metadata["source"] = str(path)
        doc_metadata["filename"] = path.name

        return Document(text=text, metadata=doc_metadata)

    def load_pdf_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None,
    ) -> Document:
        """
        Load a single PDF file.

        Args:
            file_path: Path to PDF file.
            metadata: Optional metadata dictionary.

        Returns:
            Document object with extracted text.
        """        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading PDF: {file_path}")
        
        reader = PdfReader(str(path))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                text_parts.append(page_text)
        
        text = "\n\n".join(text_parts)
        
        doc_metadata = metadata or {}
        doc_metadata["source"] = str(path)
        doc_metadata["filename"] = path.name
        doc_metadata["num_pages"] = len(reader.pages)
        doc_metadata["file_type"] = "pdf"

        logger.info(f"Loaded PDF with {len(reader.pages)} pages, {len(text)} characters")
        return Document(text=text, metadata=doc_metadata)

    def load_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.txt",
        recursive: bool = False,
    ) -> List[Document]:
        """
        Load all matching files from a directory.

        Args:
            directory_path: Path to directory.
            file_pattern: Glob pattern for files (e.g., "*.txt", "*.md").
            recursive: Whether to search recursively.

        Returns:
            List of Document objects.
        """
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents = []

        if recursive:
            files = path.rglob(file_pattern)
        else:
            files = path.glob(file_pattern)

        for file_path in files:
            try:
                # Determine file type and use appropriate loader
                if file_path.suffix.lower() == '.pdf':
                    doc = self.load_pdf_file(str(file_path))
                else:
                    doc = self.load_text_file(str(file_path))
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text.

        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)\'\"]+', '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Split text into chunks with overlap.

        Args:
            text: Input text to chunk.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of Document chunks.
        """
        # Clean text first
        text = self.clean_text(text)

        # Split by separator first
        parts = text.split(self.separator)

        chunks = []
        current_chunk = ""
        current_length = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            part_length = len(part)

            # If part itself is larger than chunk size, split it
            if part_length > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunks.append(Document(text=current_chunk.strip(), metadata=chunk_metadata))
                    current_chunk = ""
                    current_length = 0

                # Split large part by sentences
                sentences = self._split_into_sentences(part)
                temp_chunk = ""
                temp_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence)

                    if temp_length + sentence_length > self.chunk_size:
                        if temp_chunk:
                            chunk_metadata = metadata.copy() if metadata else {}
                            chunk_metadata["chunk_index"] = len(chunks)
                            chunks.append(Document(text=temp_chunk.strip(), metadata=chunk_metadata))
                        temp_chunk = sentence + " "
                        temp_length = sentence_length
                    else:
                        temp_chunk += sentence + " "
                        temp_length += sentence_length

                if temp_chunk:
                    current_chunk = temp_chunk
                    current_length = temp_length

            elif current_length + part_length > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunks.append(Document(text=current_chunk.strip(), metadata=chunk_metadata))

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + part + " "
                    current_length = len(overlap_text) + part_length
                else:
                    current_chunk = part + " "
                    current_length = part_length
            else:
                current_chunk += part + " "
                current_length += part_length

        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = len(chunks)
            chunks.append(Document(text=current_chunk.strip(), metadata=chunk_metadata))

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentences.
        """
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def process_documents(
        self,
        documents: List[Document],
        chunk: bool = True,
    ) -> List[Document]:
        """
        Process multiple documents (clean and optionally chunk).

        Args:
            documents: List of Document objects.
            chunk: Whether to chunk documents.

        Returns:
            List of processed Document objects.
        """
        processed_docs = []

        for doc in documents:
            if chunk:
                chunks = self.chunk_text(doc.text, doc.metadata)
                processed_docs.extend(chunks)
            else:
                cleaned_text = self.clean_text(doc.text)
                processed_docs.append(
                    Document(text=cleaned_text, metadata=doc.metadata)
                )

        logger.info(
            f"Processed {len(documents)} documents into {len(processed_docs)} chunks"
        )
        return processed_docs

    def get_statistics(self, documents: List[Document]) -> Dict[str, any]:
        """
        Get statistics about document collection.

        Args:
            documents: List of documents.

        Returns:
            Dictionary with statistics.
        """
        total_chars = sum(len(doc.text) for doc in documents)
        avg_chars = total_chars / len(documents) if documents else 0

        topics = {}
        for doc in documents:
            topic = doc.metadata.get("topic", "unknown")
            topics[topic] = topics.get(topic, 0) + 1

        return {
            "num_documents": len(documents),
            "total_characters": total_chars,
            "avg_characters": avg_chars,
            "topics": topics,
        }
