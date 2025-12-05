import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Callable
import functools
from datetime import datetime, timedelta


class Cache:
    def __init__(self, cache_dir: str, ttl_hours: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours) if ttl_hours else None

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        # Hash the key to create a valid filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False

        if not cache_path.exists():
            return True

        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - file_time > self.ttl

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from cache.

        Args:
            key: Cache key.
            default: Default value if key not found or expired.

        Returns:
            Cached value or default.
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists() or self._is_expired(cache_path):
            return default

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # If cache is corrupted, return default
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            # Silent failure - caching is optional
            pass

    def delete(self, key: str) -> None:
        """
        Delete cache entry.

        Args:
            key: Cache key.
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def size(self) -> int:
        """Get number of cached entries."""
        return len(list(self.cache_dir.glob("*.pkl")))


def cache_result(
    cache_instance: Cache,
    key_func: Optional[Callable] = None,
    ignore_args: bool = False
):
    """
    Decorator to cache function results.

    Args:
        cache_instance: Cache instance to use.
        key_func: Function to generate cache key from args/kwargs.
                 If None, uses str representation of args.
        ignore_args: If True, same cache entry for all calls.

    Returns:
        Decorator function.

    Example:
        ```python
        cache = Cache("./cache")

        @cache_result(cache)
        def expensive_function(x, y):
            return x + y

        # First call computes and caches
        result = expensive_function(1, 2)

        # Second call retrieves from cache
        result = expensive_function(1, 2)
        ```
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if ignore_args:
                cache_key = f"{func.__name__}"
            elif key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Create key from function name and arguments
                args_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{args_str}"

            # Try to get from cache
            cached_value = cache_instance.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Compute and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            return result

        return wrapper
    return decorator


class EmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache = Cache(cache_dir)

    def get_embedding(self, text: str, model_name: str) -> Optional[Any]:
        """
        Get cached embedding for text.

        Args:
            text: Input text.
            model_name: Name of embedding model.

        Returns:
            Cached embedding array or None.
        """
        key = self._make_key(text, model_name)
        return self.cache.get(key)

    def set_embedding(self, text: str, model_name: str, embedding: Any) -> None:
        """
        Cache embedding for text.

        Args:
            text: Input text.
            model_name: Name of embedding model.
            embedding: Embedding array to cache.
        """
        key = self._make_key(text, model_name)
        self.cache.set(key, embedding)

    def get_batch_embeddings(
        self, texts: list[str], model_name: str
    ) -> tuple[list[Any], list[str]]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of input texts.
            model_name: Name of embedding model.

        Returns:
            Tuple of (cached_embeddings, uncached_texts).
            Cached embeddings will have None for uncached items.
        """
        embeddings = []
        uncached_texts = []

        for text in texts:
            embedding = self.get_embedding(text, model_name)
            embeddings.append(embedding)
            if embedding is None:
                uncached_texts.append(text)

        return embeddings, uncached_texts

    def set_batch_embeddings(
        self, texts: list[str], model_name: str, embeddings: list[Any]
    ) -> None:
        """
        Cache embeddings for multiple texts.

        Args:
            texts: List of input texts.
            model_name: Name of embedding model.
            embeddings: List of embedding arrays.
        """
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, model_name, embedding)

    @staticmethod
    def _make_key(text: str, model_name: str) -> str:
        """Create cache key from text and model name."""
        # Hash text to handle long inputs
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_name}:{text_hash}"

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
