import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from typing import List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RAGConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_local_embedding_model_path(model_name: str):
    hf_home = Path(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))
    snapshots_dir = hf_home / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if snapshots_dir.exists():
        dirs = list(snapshots_dir.iterdir())
        if dirs:
            logger.info(f"Found cached embedding model: {dirs[0]}")
            return str(dirs[0])
    logger.warning(f"Embedding model not in cache, will attempt download: {model_name}")
    return None


class Embedder:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model_name = config.embedding_model
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Embedder: model={self.model_name}, device={self.device}")

    def load_model(self) -> None:
        if self.model is not None:
            return
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            load_path = _get_local_embedding_model_path(self.model_name) or self.model_name
            self.model = SentenceTransformer(load_path, device=self.device)
            logger.info(f"Embedding model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        if self.model is None:
            self.load_model()
        return self.model.encode(text, normalize_embeddings=normalize, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        if self.model is None:
            self.load_model()
        embeddings = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=normalize,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        logger.info(f"Generated embeddings: {embeddings.shape}")
        return embeddings

    def get_embedding_dim(self) -> int:
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")
