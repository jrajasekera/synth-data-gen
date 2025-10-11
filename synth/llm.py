"""Interface to the local LLM served via llama.cpp."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

from .cache import CacheConfig, ResponseCache
from .utils import stable_hash


@dataclass(slots=True)
class LLMConfig:
    """Configuration for connecting to a llama.cpp server."""

    endpoint: str = "http://127.0.0.1:8080/v1"
    model: str = "glm-4.5-air"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout: float = 60.0


class LLMClient:
    """Thin HTTP client around the llama.cpp server."""

    def __init__(self, config: Optional[LLMConfig] = None, cache_dir: Optional[Path] = None) -> None:
        self.config = config or LLMConfig()
        self.cache_dir = cache_dir
        self._client = httpx.Client(timeout=self.config.timeout)
        self._cache = None
        if cache_dir is not None:
            self._cache = ResponseCache(CacheConfig(directory=cache_dir))

    def map_reduce(
        self,
        prompts: Iterable[dict[str, Any]],
        max_workers: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Execute a batch of prompts with map-reduce semantics."""
        prompt_list = list(prompts)
        if not prompt_list:
            return []

        if len(prompt_list) == 1 or max_workers == 1:
            return [self._invoke(prompt_list[0])]

        worker_count = max_workers or min(4, len(prompt_list))
        results: list[Optional[dict[str, Any]]] = [None] * len(prompt_list)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self._invoke, prompt): idx for idx, prompt in enumerate(prompt_list)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()

        return [result for result in results if result is not None]

    def generate_text(self, prompt: dict[str, Any]) -> dict[str, Any]:
        """Generate text for a constrained field."""

        return self._invoke(prompt)

    def close(self) -> None:
        """Close underlying HTTP and cache resources."""

        self._client.close()
        if self._cache is not None:
            self._cache.close()

    def _invoke(self, prompt: dict[str, Any]) -> dict[str, Any]:
        payload = self._build_payload(prompt)
        cache_key = stable_hash([payload])

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        response = self._client.post(
            f"{self.config.endpoint}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        if self._cache is not None:
            self._cache.set(cache_key, data)

        return data

    def _build_payload(self, prompt: dict[str, Any]) -> dict[str, Any]:
        messages = prompt.get("messages")
        if not messages:
            raise ValueError("prompt must include 'messages'")

        payload: dict[str, Any] = {
            "model": prompt.get("model", self.config.model),
            "messages": messages,
            "temperature": prompt.get("temperature", self.config.temperature),
            "top_p": prompt.get("top_p", self.config.top_p),
            "max_tokens": prompt.get("max_tokens", self.config.max_tokens),
        }

        if "seed" in prompt and prompt["seed"] is not None:
            payload["seed"] = prompt["seed"]

        if "stop" in prompt:
            payload["stop"] = prompt["stop"]

        return payload

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
