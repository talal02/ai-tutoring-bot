import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import LLMConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def get_local_model_path(model_name: str) -> Optional[str]:
    if Path(model_name).exists():
        return model_name
    hf_home = Path(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))
    snapshots_dir = hf_home / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if snapshots_dir.exists():
        dirs = list(snapshots_dir.iterdir())
        if dirs:
            logger.info(f"Found cached model at: {dirs[0]}")
            return str(dirs[0])
    logger.warning(f"Model not in cache, will download: {model_name}")
    return None


class ModelLoader:
    MODEL_CONFIGS = {
        "phi-3-mini":    "microsoft/Phi-3-mini-4k-instruct",
        "llama-3.1-8b":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.2-1b":  "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-3b":  "meta-llama/Llama-3.2-3B-Instruct",
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        logger.info(f"ModelLoader: device={self.device}")

    def _get_device(self) -> str:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if self.config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        logger.warning("GPU not available, falling back to CPU")
        return "cpu"

    def _get_model_name(self) -> str:
        name = self.config.model_name
        if name in self.MODEL_CONFIGS:
            resolved = self.MODEL_CONFIGS[name]
            logger.info(f"Resolved '{name}' → '{resolved}'")
            return resolved
        return name

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if self.config.load_in_4bit:
            logger.info("Loading with 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if self.config.load_in_8bit:
            logger.info("Loading with 8-bit quantization")
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def load_model(self, adapter_path: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        try:
            model_name = self._get_model_name()
            logger.info(f"Loading model: {model_name}")
            local_path = get_local_model_path(model_name)
            load_path = local_path or model_name

            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                trust_remote_code=True,
                local_files_only=local_path is not None,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quant_config = self._get_quantization_config()
            model_kwargs = {
                "trust_remote_code": True,
                "dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "local_files_only": local_path is not None,
                "device_map": "auto",
            }
            if quant_config:
                model_kwargs["quantization_config"] = quant_config

            self.model = AutoModelForCausalLM.from_pretrained(load_path, **model_kwargs)

            if adapter_path:
                logger.info(f"Loading LoRA adapter: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

            self.model.eval()

            if self.device == "cuda":
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                if allocated_gb < 1.0:
                    raise RuntimeError(
                        f"Model claims to be on CUDA but only {allocated_gb:.2f} GB allocated. "
                        "The model likely loaded on CPU — check that no process is holding all GPU memory."
                    )
                logger.info(f"GPU memory allocated: {allocated_gb:.2f} GB")

            logger.info(f"Model loaded: {self.get_model_size():.2f}B parameters")
            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def get_model_size(self) -> float:
        """Return total parameter count in billions."""
        if self.model is None:
            return 0.0
        return sum(p.numel() for p in self.model.parameters()) / 1e9

    def get_memory_usage(self) -> dict:
        stats = {"device": self.device, "model_loaded": self.model is not None}
        if self.device == "cuda" and torch.cuda.is_available():
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return stats

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")
