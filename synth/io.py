"""I/O utilities for streaming JSON data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration for JSON streaming chunk size."""

    size: int = 1000
    format: str | None = None
    max_records: int | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("chunk size must be positive")
        if self.format is not None and self.format not in {"json_array", "jsonl", "json_object"}:
            raise ValueError("format must be 'json_array', 'json_object', 'jsonl', or None")


class JSONStream:
    """Stream records from a JSON or JSONL file in fixed-size chunks."""

    def __init__(self, path: Path, config: ChunkingConfig) -> None:
        self.path = path
        self.config = config

    def iter_chunks(self) -> Iterator[list[dict[str, Any]]]:
        """Yield successive chunks of JSON objects.

        Supports newline-delimited JSON (JSONL) as well as standard JSON arrays.
        The implementation performs light streaming for JSON arrays by parsing
        one element at a time, avoiding full materialisation when possible.
        """

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        detected_format = self._detect_format()
        record_count = 0

        if detected_format == "jsonl":
            with self.open() as handle:
                chunk: list[dict[str, Any]] = []
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError("JSONL record is not an object")
                    chunk.append(obj)
                    record_count += 1
                    if len(chunk) >= self.config.size:
                        yield chunk
                        chunk = []
                    if self._should_stop(record_count):
                        break
                if chunk:
                    yield chunk
            return

        if detected_format == "json_object":
            with self.open() as handle:
                data = json.load(handle)
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object at root")
                yield [data]
            return

        yield from self._iter_json_array(record_limit=self.config.max_records)

    def open(self) -> TextIO:
        """Open the underlying file."""

        return self.path.open("r", encoding="utf-8")

    def __iter__(self) -> Iterable[list[dict[str, Any]]]:
        return self.iter_chunks()

    def _detect_format(self) -> str:
        if self.config.format:
            return self.config.format

        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            return "jsonl"

        with self.open() as handle:
            while True:
                char = handle.read(1)
                if not char:
                    break
                if char.isspace():
                    continue
                if char == "[":
                    return "json_array"
                if char == "{":
                    return "json_object"
        raise ValueError("Unable to detect JSON format")

    def _iter_json_array(self, record_limit: int | None = None) -> Iterator[list[dict[str, Any]]]:
        with self.open() as handle:
            decoder = json.JSONDecoder()
            buffer = ""
            chunk: list[dict[str, Any]] = []
            record_count = 0
            in_array = False
            eof = False

            while not eof:
                data = handle.read(65536)
                if not data:
                    eof = True
                buffer += data
                idx = 0

                while True:
                    idx = _consume_whitespace(buffer, idx)

                    if not in_array:
                        if idx >= len(buffer):
                            break
                        if buffer[idx] == "[":
                            in_array = True
                            idx += 1
                            continue
                        raise ValueError("Expected JSON array start")

                    idx = _consume_whitespace(buffer, idx)
                    if idx >= len(buffer):
                        break

                    if buffer[idx] == "]":
                        eof = True
                        idx += 1
                        break

                    obj, end = decoder.raw_decode(buffer, idx)
                    if not isinstance(obj, dict):
                        raise ValueError("Array elements must be JSON objects")
                    chunk.append(obj)
                    record_count += 1
                    idx = _consume_whitespace(buffer, end)
                    if idx < len(buffer) and buffer[idx] == ",":
                        idx += 1

                    if len(chunk) >= self.config.size:
                        yield chunk
                        chunk = []

                    if self._should_stop(record_count):
                        if chunk:
                            yield chunk
                        return

                buffer = buffer[idx:]

            if chunk:
                yield chunk

    def _should_stop(self, record_count: int) -> bool:
        return self.config.max_records is not None and record_count >= self.config.max_records


def _consume_whitespace(buffer: str, idx: int) -> int:
    while idx < len(buffer) and buffer[idx].isspace():
        idx += 1
    return idx
