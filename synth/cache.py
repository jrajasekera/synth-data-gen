"""Cache utilities for LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from diskcache import Cache


@dataclass(slots=True)
class CacheConfig:
    """Configuration for the disk cache."""

    directory: Optional[Path] = None
    shards: int = 1
    timeout: float = 0.1


class ResponseCache:
    """Wrapper around diskcache for storing LLM responses."""

    def __init__(self, config: CacheConfig) -> None:
        directory = (config.directory or Path.home() / ".synth-data-cache").expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(str(directory), timeout=config.timeout)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached response."""

        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a response with optional TTL."""

        if ttl is None:
            self.cache.set(key, value)
        else:
            self.cache.set(key, value, expire=ttl)

    def close(self) -> None:
        """Close the underlying cache."""

        self.cache.close()
