"""Model caching utilities for the tutoring system."""

from pathlib import Path


def ensure_model_cache_dir(cache_dir: str = "./cache/models") -> Path:
    """
    Ensure model cache directory exists.

    Args:
        cache_dir: Directory path for model cache.

    Returns:
        Path object for the cache directory.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
