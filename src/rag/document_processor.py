import json
import re
from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RAGConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    text: str
    metadata: Dict[str, object] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunk_size = config.chunking["chunk_size"]
        self.chunk_overlap = config.chunking["chunk_overlap"]
        self.separator = config.chunking["separator"]
        logger.info(f"DocumentProcessor: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def load_json_dataset(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        logger.info(f"Loading dataset from {file_path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = [
            Document(
                text=f"Question: {item.get('question', '')}\n\nAnswer: {item.get('answer', '')}",
                metadata={"topic": item.get("topic", "unknown"), "source": "dataset", "question": item.get("question", "")},
            )
            for item in data
        ]
        logger.info(f"Loaded {len(documents)} documents from dataset")
        return documents

    def load_text_file(self, file_path: str, metadata: Optional[Dict] = None) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        meta = metadata or {}
        meta["source"] = str(path)
        meta["filename"] = path.name
        return Document(text=text, metadata=meta)

    def load_pdf_file(self, file_path: str, metadata: Optional[Dict] = None) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Loading PDF: {file_path}")
        reader = PdfReader(str(path))
        text = "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text().strip())
        meta = metadata or {}
        meta.update({"source": str(path), "filename": path.name, "num_pages": len(reader.pages), "file_type": "pdf"})
        logger.info(f"Loaded PDF: {len(reader.pages)} pages, {len(text)} chars")
        return Document(text=text, metadata=meta)

    def load_directory(self, directory_path: str, file_pattern: str = "*.txt", recursive: bool = False) -> List[Document]:
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        files = path.rglob(file_pattern) if recursive else path.glob(file_pattern)
        documents = []
        for file_path in files:
            try:
                doc = self.load_pdf_file(str(file_path)) if file_path.suffix.lower() == '.pdf' else self.load_text_file(str(file_path))
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)\'\"]+', '', text)
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        return text.strip()

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        text = self.clean_text(text)
        parts = text.split(self.separator)
        chunks = []
        current_chunk = ""
        current_length = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue
            part_length = len(part)

            if part_length > self.chunk_size:
                if current_chunk:
                    meta = metadata.copy() if metadata else {}
                    meta["chunk_index"] = len(chunks)
                    chunks.append(Document(text=current_chunk.strip(), metadata=meta))
                    current_chunk = ""
                    current_length = 0
                # Part is too large — split by sentence
                sentences = self._split_into_sentences(part)
                temp_chunk = ""
                temp_length = 0
                for sentence in sentences:
                    s_len = len(sentence)
                    if temp_length + s_len > self.chunk_size:
                        if temp_chunk:
                            meta = metadata.copy() if metadata else {}
                            meta["chunk_index"] = len(chunks)
                            chunks.append(Document(text=temp_chunk.strip(), metadata=meta))
                        temp_chunk = sentence + " "
                        temp_length = s_len
                    else:
                        temp_chunk += sentence + " "
                        temp_length += s_len
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_length = temp_length

            elif current_length + part_length > self.chunk_size:
                if current_chunk:
                    meta = metadata.copy() if metadata else {}
                    meta["chunk_index"] = len(chunks)
                    chunks.append(Document(text=current_chunk.strip(), metadata=meta))
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

        if current_chunk.strip():
            meta = metadata.copy() if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Document(text=current_chunk.strip(), metadata=meta))

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def process_documents(self, documents: List[Document], chunk: bool = True) -> List[Document]:
        processed = []
        for doc in documents:
            if chunk:
                processed.extend(self.chunk_text(doc.text, doc.metadata))
            else:
                processed.append(Document(text=self.clean_text(doc.text), metadata=doc.metadata))
        logger.info(f"Processed {len(documents)} documents into {len(processed)} chunks")
        return processed

    def get_statistics(self, documents: List[Document]) -> Dict:
        total_chars = sum(len(doc.text) for doc in documents)
        topics = {}
        for doc in documents:
            t = doc.metadata.get("topic", "unknown")
            topics[t] = topics.get(t, 0) + 1
        return {
            "num_documents": len(documents),
            "total_characters": total_chars,
            "avg_characters": total_chars / len(documents) if documents else 0,
            "topics": topics,
        }
