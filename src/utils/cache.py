from pathlib import Path


def ensure_model_cache_dir(cache_dir: str = "./cache/models") -> Path:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
