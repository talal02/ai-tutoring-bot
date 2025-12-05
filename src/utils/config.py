import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    device: str = "cuda"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_cache: bool = True
    cache_dir: str = "./cache/models"
    generation: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True,
    })


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunking: Dict[str, Any] = field(default_factory=lambda: {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "separator": "\n\n",
    })
    retrieval: Dict[str, Any] = field(default_factory=lambda: {
        "top_k": 5,
        "similarity_threshold": 0.3,
        "use_reranking": False,
        "reranker_model": None,
    })
    vector_store: Dict[str, Any] = field(default_factory=lambda: {
        "type": "faiss",
        "index_type": "IndexFlatL2",
        "persist_directory": "./data/vector_store",
    })
    embedding_cache: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "cache_dir": "./cache/embeddings",
    })


@dataclass
class FineTuningConfig:
    lora: Dict[str, Any] = field(default_factory=lambda: {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    })
    training: Dict[str, Any] = field(default_factory=lambda: {
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2.0e-4,
        "warmup_steps": 100,
        "max_seq_length": 1024,
        "output_dir": "./models/finetuned",
    })


@dataclass
class PromptsConfig:
    system_prompt: str = (
        "You are a knowledgeable and patient history tutor for high school students. "
        "Your goal is to help students understand historical concepts through clear explanations, "
        "Socratic questioning, and step-by-step guidance. Always ground your responses in "
        "curriculum materials and avoid speculation."
    )
    rag_prompt_template: str = (
        "Context from curriculum materials:\n{context}\n\n"
        "Student question: {question}\n\n"
        "Based on the context provided, give a clear and pedagogically sound answer. "
        "If the context doesn't contain enough information, acknowledge this limitation."
    )
    hint_prompt_template: str = (
        "The student is working on: {question}\n"
        "Their current understanding: {student_response}\n\n"
        "Provide a {hint_level} hint to guide them toward the correct answer without giving it away directly.\n"
        "Hint levels: nudge (very subtle), partial (more direct), full (nearly complete answer)"
    )


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "./logs/tutor.log"
    console_output: bool = True


@dataclass
class AppConfig:
    save_conversations: bool = True
    conversation_dir: str = "./data/conversations"
    max_conversation_history: int = 10


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.llm = LLMConfig()
        self.rag = RAGConfig()
        self.fine_tuning = FineTuningConfig()
        self.prompts = PromptsConfig()
        self.logging = LoggingConfig()
        self.app = AppConfig()

        if config_path:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML is malformed.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Update LLM config
        if 'llm' in config_dict:
            self._update_dataclass(self.llm, config_dict['llm'])

        # Update RAG config
        if 'rag' in config_dict:
            self._update_dataclass(self.rag, config_dict['rag'])

        # Update fine-tuning config
        if 'fine_tuning' in config_dict:
            self._update_dataclass(self.fine_tuning, config_dict['fine_tuning'])

        # Update prompts config
        if 'prompts' in config_dict:
            self._update_dataclass(self.prompts, config_dict['prompts'])

        # Update logging config
        if 'logging' in config_dict:
            self._update_dataclass(self.logging, config_dict['logging'])

        # Update app config
        if 'app' in config_dict:
            self._update_dataclass(self.app, config_dict['app'])

    @staticmethod
    def _update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
        """
        Update dataclass fields from dictionary.

        Args:
            obj: Dataclass instance to update.
            updates: Dictionary with updates.
        """
        for key, value in updates.items():
            if hasattr(obj, key):
                current_value = getattr(obj, key)
                # If current value is a dict, update it
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                else:
                    setattr(obj, key, value)

    def save_to_yaml(self, config_path: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            config_path: Path where to save configuration.
        """
        config_dict = {
            'llm': self._dataclass_to_dict(self.llm),
            'rag': self._dataclass_to_dict(self.rag),
            'fine_tuning': self._dataclass_to_dict(self.fine_tuning),
            'prompts': self._dataclass_to_dict(self.prompts),
            'logging': self._dataclass_to_dict(self.logging),
            'app': self._dataclass_to_dict(self.app),
        }

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {k: v for k, v in obj.__dict__.items()}

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.llm.cache_dir,
            self.rag.vector_store['persist_directory'],
            self.rag.embedding_cache['cache_dir'],
            self.app.conversation_dir,
            self.fine_tuning.training['output_dir'],
            os.path.dirname(self.logging.log_file),
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern).

    Args:
        config_path: Path to config file. Only used on first call.

    Returns:
        Config instance.
    """
    global _config
    if _config is None:
        _config = Config(config_path)
        _config.ensure_directories()
    return _config


def reload_config(config_path: str) -> Config:
    """
    Reload configuration from file.

    Args:
        config_path: Path to config file.

    Returns:
        New Config instance.
    """
    global _config
    _config = Config(config_path)
    _config.ensure_directories()
    return _config
