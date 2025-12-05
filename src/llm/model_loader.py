import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig
from typing import Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import LLMConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    MODEL_CONFIGS = {
        "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        logger.info(f"ModelLoader initialized with device: {self.device}")

    def _get_device(self) -> str:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("GPU not available, falling back to CPU")
            return "cpu"

    def _get_model_name(self) -> str:
        model_name = self.config.model_name
        if model_name in self.MODEL_CONFIGS:
            resolved_name = self.MODEL_CONFIGS[model_name]
            logger.info(f"Resolved '{model_name}' to '{resolved_name}'")
            return resolved_name
        return model_name

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if self.config.load_in_4bit:
            logger.info("Loading model with 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            logger.info("Loading model with 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )

        return None

    def load_model(self, adapter_path: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        try:
            model_name = self._get_model_name()
            logger.info(f"Loading model: {model_name}")

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir if self.config.use_cache else None,
                revision="main",
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            quant_config = self._get_quantization_config()

            logger.info("Loading model weights...")
            model_kwargs = {
                "trust_remote_code": True,
                "cache_dir": self.config.cache_dir if self.config.use_cache else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "revision": "main",
            }

            if quant_config:
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = self.device

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            if adapter_path:
                logger.info(f"Loading LoRA adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

            self.model.eval()

            logger.info(f"Model loaded successfully. Parameters: {self.get_model_size():.2f}B")
            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params / 1e9

    def get_memory_usage(self) -> dict:
        stats = {
            "device": self.device,
            "model_loaded": self.model is not None,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

        return stats

    def unload_model(self) -> None:
        if self.model is not None:
            logger.info("Unloading model from memory")
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")

    def __del__(self):
        self.unload_model()
