"""Interface to the local LLM served via llama.cpp."""

from __future__ import annotations

import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
from jsonschema import ValidationError, validate

from .cache import CacheConfig, ResponseCache
from .utils import stable_hash


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMConfig:
    """Configuration for connecting to a llama.cpp server."""

    endpoint: str = "http://127.0.0.1:8080/v1"
    model: str = "glm-4.5-air"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout: float = 60.0
    max_concurrent_requests: int = 2
    batch_size: int = 4
    global_seed: Optional[int] = 0


class LLMResponseError(RuntimeError):
    """Raised when an LLM response cannot be parsed or validated."""

    def __init__(self, message: str, *, response: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.response = response


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
        *,
        schema: Optional[dict[str, Any]] = None,
        parse_json: bool = False,
    ) -> list[Any]:
        """Execute a batch of prompts with map-reduce semantics."""

        prompt_list = [self._prepare_prompt(prompt, idx) for idx, prompt in enumerate(prompts)]
        if not prompt_list:
            return []

        if len(prompt_list) == 1 or (max_workers == 1 == len(prompt_list)):
            try:
                response = self._invoke(prompt_list[0], schema=schema, parse_json=parse_json)
                return [response] if response is not None else []
            except LLMResponseError as error:
                logger.warning("LLM response discarded: %s", error)
                return []

        worker_limit = max_workers or self.config.max_concurrent_requests or 1
        worker_limit = max(1, min(worker_limit, len(prompt_list)))
        batch_size = self.config.batch_size or worker_limit
        batch_size = max(1, batch_size)
        results: list[Optional[Any]] = [None] * len(prompt_list)

        for batch_start in range(0, len(prompt_list), batch_size):
            batch_indices = range(batch_start, min(batch_start + batch_size, len(prompt_list)))
            batch_prompts = [prompt_list[idx] for idx in batch_indices]
            batch_workers = min(worker_limit, len(batch_prompts))

            if batch_workers == 1:
                for offset, prompt in zip(batch_indices, batch_prompts):
                    try:
                        results[offset] = self._invoke(prompt, schema=schema, parse_json=parse_json)
                    except LLMResponseError as error:
                        logger.warning("LLM response discarded: %s", error)
                        results[offset] = None
                    except Exception:  # pragma: no cover - unexpected failures
                        logger.exception("LLM invocation failed")
                        results[offset] = None
                continue

            with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                future_map = {
                    executor.submit(self._invoke, prompt, schema, parse_json): idx
                    for idx, prompt in zip(batch_indices, batch_prompts)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        results[idx] = future.result()
                    except LLMResponseError as error:
                        logger.warning("LLM response discarded: %s", error)
                        results[idx] = None
                    except Exception:  # pragma: no cover - unexpected failures
                        logger.exception("LLM invocation failed")
                        results[idx] = None

        return [result for result in results if result is not None]

    def generate_text(
        self,
        prompt: dict[str, Any],
        *,
        schema: Optional[dict[str, Any]] = None,
        parse_json: bool = False,
    ) -> Any:
        """Generate text for a constrained field."""

        prepared = self._prepare_prompt(prompt, index=0)
        return self._invoke(prepared, schema=schema, parse_json=parse_json)

    def close(self) -> None:
        """Close underlying HTTP and cache resources."""

        self._client.close()
        if self._cache is not None:
            self._cache.close()

    def _invoke(
        self,
        prompt: dict[str, Any],
        schema: Optional[dict[str, Any]] = None,
        parse_json: bool = False,
    ) -> Any:
        payload = self._build_payload(prompt)
        cache_key = stable_hash([payload])

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._post_process(cached, schema=schema, parse_json=parse_json)

        response = self._client.post(
            f"{self.config.endpoint}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        if self._cache is not None:
            self._cache.set(cache_key, data)

        return self._post_process(data, schema=schema, parse_json=parse_json)

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
            payload["seed"] = int(prompt["seed"])

        if "stop" in prompt:
            payload["stop"] = prompt["stop"]

        return payload

    def _prepare_prompt(self, prompt: dict[str, Any], index: int) -> dict[str, Any]:
        prepared = copy.deepcopy(prompt)
        if prepared.get("seed") is None:
            derived = self._derive_seed(prepared, index)
            if derived is not None:
                prepared["seed"] = derived
        return prepared

    def _derive_seed(self, prompt: dict[str, Any], index: int) -> Optional[int]:
        if self.config.global_seed is None:
            return None

        fingerprint = stable_hash(
            [
                index,
                prompt.get("model", self.config.model),
                prompt.get("messages"),
                prompt.get("stop"),
            ]
        )
        derived = int(fingerprint[:16], 16)
        return (self.config.global_seed + derived) % 2_147_483_647

    def _post_process(
        self,
        response: dict[str, Any],
        *,
        schema: Optional[dict[str, Any]],
        parse_json: bool,
    ) -> Any:
        if schema is None:
            return response

        content = self._extract_content(response)
        try:
            parsed = json.loads(content)
        except JSONDecodeError as exc:
            raise LLMResponseError("LLM response is not valid JSON", response=response) from exc

        try:
            validate(instance=parsed, schema=schema)
        except ValidationError as exc:
            raise LLMResponseError(f"LLM response failed schema validation: {exc.message}", response=response) from exc

        return parsed if parse_json else response

    def _extract_content(self, response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not choices:
            raise LLMResponseError("LLM response missing choices", response=response)
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("LLM response missing message payload", response=response)
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("LLM response content is empty", response=response)
        return content

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
