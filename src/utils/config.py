import copy
import yaml
from pathlib import Path
from types import SimpleNamespace

_config = None

_DEFAULTS = {
    'llm': {
        'model_name': 'microsoft/Phi-3-mini-4k-instruct',
        'device': 'cuda',
        'load_in_4bit': False,
        'load_in_8bit': False,
        'use_cache': True,
        'cache_dir': './cache/models',
        'generation': {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'do_sample': True,
        },
    },
    'rag': {
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'chunking': {'chunk_size': 512, 'chunk_overlap': 50, 'separator': '\n\n'},
        'retrieval': {'top_k': 5, 'similarity_threshold': 0.3},
        'vector_store': {
            'type': 'faiss',
            'index_type': 'IndexFlatL2',
            'persist_directory': './data/vector_store',
        },
        'embedding_cache': {'enabled': True, 'cache_dir': './cache/embeddings'},
    },
    'prompts': {
        'system_prompt': (
            "You are a knowledgeable and patient history tutor for high school students. "
            "Your goal is to help students understand historical concepts through clear explanations, "
            "Socratic questioning, and step-by-step guidance. Always ground your responses in "
            "curriculum materials and avoid speculation."
        ),
        'rag_prompt_template': (
            "Context from curriculum materials:\n{context}\n\n"
            "Student question: {question}\n\n"
            "Based on the context provided, give a clear and pedagogically sound answer. "
            "If the context doesn't contain enough information, acknowledge this limitation."
        ),
        'hint_prompt_template': (
            "The student is working on: {question}\n"
            "Their current understanding: {student_response}\n\n"
            "Provide a {hint_level} hint to guide them toward the correct answer without giving it away directly.\n"
            "Hint levels: nudge (very subtle), partial (more direct), full (nearly complete answer)"
        ),
    },
    'logging': {
        'level': 'INFO',
        'log_file': './logs/tutor.log',
        'console_output': True,
    },
    'app': {
        'save_conversations': True,
        'conversation_dir': './data/conversations',
        'max_conversation_history': 10,
    },
}


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k].update(v)
        else:
            base[k] = v


def get_config(config_path=None):
    global _config
    if _config is None:
        data = copy.deepcopy(_DEFAULTS)
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                overrides = yaml.safe_load(f) or {}
            for section, vals in overrides.items():
                if section in data and isinstance(vals, dict):
                    _deep_merge(data[section], vals)
        _config = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in data.items()})
        _ensure_dirs(_config)
    return _config


def reload_config(config_path: str):
    global _config
    _config = None
    return get_config(config_path)


def _ensure_dirs(cfg) -> None:
    for d in [
        cfg.llm.cache_dir,
        cfg.rag.vector_store['persist_directory'],
        cfg.rag.embedding_cache['cache_dir'],
        cfg.app.conversation_dir,
        str(Path(cfg.logging.log_file).parent),
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


# Backward-compatible type aliases used as type hints in other modules
LLMConfig = SimpleNamespace
RAGConfig = SimpleNamespace
PromptsConfig = SimpleNamespace
LoggingConfig = SimpleNamespace
AppConfig = SimpleNamespace


class Config:
    """Backward-compatible stub — use get_config() instead."""
    pass
