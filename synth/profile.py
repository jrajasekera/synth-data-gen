"""LLM-driven profiling orchestrator."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .io import ChunkingConfig, JSONStream
from .llm import LLMClient
from .utils import RNGConfig

ProgressCallback = Callable[[int], None]


_EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_REGEX = re.compile(r"^\+?\d[\d\s().-]{6,}\d$")
_NAME_REGEX = re.compile(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$")


@dataclass(slots=True)
class NumericAccumulator:
    count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    total: float = 0.0
    squares: float = 0.0
    samples: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if np.isnan(value):
            return
        self.count += 1
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.total += value
        self.squares += value * value
        if len(self.samples) < 512:
            self.samples.append(value)

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {}
        arr = np.array(self.samples, dtype=float)
        quantiles = (
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 75)),
            float(np.percentile(arr, 95)),
        )
        mean = self.total / self.count
        variance = max(self.squares / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "min": float(self.minimum),
            "p5": quantiles[0],
            "p25": quantiles[1],
            "p50": quantiles[2],
            "p75": quantiles[3],
            "p95": quantiles[4],
            "max": float(self.maximum),
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
        }


@dataclass(slots=True)
class DateTimeAccumulator:
    count: int = 0
    minimum: Optional[datetime] = None
    maximum: Optional[datetime] = None
    by_weekday: Counter[int] = field(default_factory=Counter)
    by_hour: Counter[int] = field(default_factory=Counter)

    def add(self, value: datetime) -> None:
        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.by_weekday[value.weekday()] += 1
        self.by_hour[value.hour] += 1

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {}
        weekday_counts = [self.by_weekday.get(i, 0) for i in range(7)]
        hour_counts = [self.by_hour.get(i, 0) for i in range(24)]
        return {
            "count": self.count,
            "min": self.minimum.isoformat() if self.minimum else None,
            "max": self.maximum.isoformat() if self.maximum else None,
            "by_weekday": weekday_counts,
            "by_hour": hour_counts,
        }


@dataclass(slots=True)
class FieldStats:
    path: str
    total: int = 0
    nulls: int = 0
    type_counts: Counter[str] = field(default_factory=Counter)
    numeric: NumericAccumulator = field(default_factory=NumericAccumulator)
    length: NumericAccumulator = field(default_factory=NumericAccumulator)
    datetime_info: DateTimeAccumulator = field(default_factory=DateTimeAccumulator)
    categorical: Counter[Any] = field(default_factory=Counter)
    samples: list[Any] = field(default_factory=list)
    uniqueness: set[str] = field(default_factory=set)
    uniqueness_truncated: bool = False
    regex_candidates: list[dict[str, Any]] = field(default_factory=list)
    semantic_type: Optional[str] = None
    array_lengths: NumericAccumulator = field(default_factory=NumericAccumulator)
    pii_likelihood: float = 0.0
    anomalies: list[dict[str, Any]] = field(default_factory=list)

    MAX_UNIQUE_TRACKED = 20000

    def add(self, value: Any) -> None:
        self.total += 1
        if value is None:
            self.nulls += 1
            self.type_counts["null"] += 1
            return

        inferred_type, normalized, datetime_value = _infer_type(value)
        self.type_counts[inferred_type] += 1

        if inferred_type == "number":
            self.numeric.add(float(normalized))
        if inferred_type in {"string", "datetime"}:
            self.length.add(len(str(normalized)))
        if inferred_type == "datetime" and datetime_value is not None:
            self.datetime_info.add(datetime_value)
        if inferred_type == "array" and isinstance(value, list):
            self.array_lengths.add(len(value))

        if inferred_type in {"string", "boolean", "number"}:
            if len(self.samples) < 10:
                self.samples.append(normalized)
            if isinstance(normalized, (str, int, float, bool)):
                self.categorical[normalized] += 1

        self._track_uniqueness(normalized)

    def required_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.total - self.nulls) / self.total

    def to_summary(self) -> dict[str, Any]:
        types_summary = [
            {"name": name, "confidence": count / self.total}
            for name, count in sorted(self.type_counts.items(), key=lambda item: item[1], reverse=True)
            if self.total
        ]

        enum_candidates = [
            {"value": value, "count": count}
            for value, count in self.categorical.most_common(10)
        ]

        summary = {
            "types": types_summary,
            "required_rate": self.required_rate(),
            "enum_candidates": enum_candidates,
            "regex_candidates": self.regex_candidates,
            "length": self.length.summary(),
            "numeric": self.numeric.summary(),
            "datetime": self.datetime_info.summary(),
            "pii_likelihood": self.pii_likelihood,
            "anomalies": self.anomalies,
            "semantic_type": self.semantic_type,
        }

        if self.array_lengths.count:
            summary["array"] = {
                "min_items": self.array_lengths.minimum,
                "max_items": self.array_lengths.maximum,
                "mean_items": self.array_lengths.total / self.array_lengths.count,
            }

        return summary

    def uniqueness_ratio(self) -> float:
        non_null = self.total - self.nulls
        if non_null == 0 or not self.uniqueness:
            return 0.0
        return min(len(self.uniqueness) / non_null, 1.0)

    def is_unique(self) -> bool:
        if self.uniqueness_truncated:
            return False
        non_null = self.total - self.nulls
        return non_null > 0 and len(self.uniqueness) == non_null

    def _track_uniqueness(self, value: Any) -> None:
        if self.uniqueness_truncated or value is None:
            return
        try:
            hashable = json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            self.uniqueness_truncated = True
            self.uniqueness.clear()
            return
        if len(self.uniqueness) >= self.MAX_UNIQUE_TRACKED:
            self.uniqueness_truncated = True
            self.uniqueness.clear()
            return
        self.uniqueness.add(hashable)


class ProfileAccumulator:
    """Accumulate statistics across records."""

    def __init__(self) -> None:
        self.fields: dict[str, FieldStats] = {}

    def record(self, record: dict[str, Any]) -> None:
        self._walk(record, path="$")

    def _walk(self, value: Any, path: str) -> None:
        stats = self.fields.setdefault(path, FieldStats(path=path))
        stats.add(value)

        if value is None:
            return

        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
                self._walk(child, child_path)
        elif isinstance(value, list):
            item_path = f"{path}[]"
            for item in value:
                self._walk(item, item_path)

    def build(self) -> dict[str, Any]:
        return {path: stats.to_summary() for path, stats in sorted(self.fields.items())}


def profile_dataset(
    stream: JSONStream,
    rng: RNGConfig,
    cache_dir: Optional[Path],
    progress_callback: Optional[ProgressCallback] = None,
    llm: Optional[LLMClient] = None,
) -> dict[str, Any]:
    """Profile a dataset and return structured metadata."""

    accumulator = ProfileAccumulator()
    total_records = 0

    chunk_index = 0
    for chunk in stream.iter_chunks():
        for record in chunk:
            if not isinstance(record, dict):
                raise ValueError("Top-level records must be JSON objects")
            accumulator.record(record)
            total_records += 1
        if progress_callback is not None:
            progress_callback(len(chunk))

        if llm is not None:
            try:
                llm_summary = _llm_profile_chunk(chunk_index, chunk, llm, rng)
                _merge_llm_summary(accumulator, llm_summary)
            except Exception:
                pass

        chunk_index += 1

    for stats in accumulator.fields.values():
        _apply_semantic_inference(stats, llm)

    field_summaries = accumulator.build()
    pk_candidates = []

    for path, stats in accumulator.fields.items():
        if stats.is_unique():
            pk_candidates.append(
                {
                    "paths": [path],
                    "uniqueness": stats.uniqueness_ratio(),
                    "confidence": 0.8 if stats.uniqueness_truncated else 1.0,
                }
            )

    return {
        "meta": {"records_total": total_records},
        "field_summaries": field_summaries,
        "keys": {
            "pk_candidates": pk_candidates,
            "fk_candidates": [],
        },
        "functional_dependencies": [],
        "temporal_patterns": [],
    }


def profile_path(path: Path, *, chunk_size: int = 1000, format_hint: str | None = None) -> dict[str, Any]:
    """Convenience wrapper to profile a dataset from disk."""

    stream = JSONStream(path, ChunkingConfig(size=chunk_size, format=format_hint))
    return profile_dataset(stream=stream, rng=RNGConfig(None), cache_dir=None)


def _apply_semantic_inference(stats: FieldStats, llm: Optional[LLMClient]) -> None:
    string_samples = [sample for sample in stats.samples if isinstance(sample, str)]
    if not string_samples:
        return

    semantic = _semantic_from_samples(string_samples)
    regex_candidates: list[dict[str, Any]] = []

    if llm is not None:
        regex_candidates = _llm_regex_inference(string_samples, llm)

    if not regex_candidates:
        regex_candidates = _heuristic_regex(string_samples)

    if semantic and not stats.semantic_type:
        stats.semantic_type = semantic

    if regex_candidates:
        stats.regex_candidates = regex_candidates


def _semantic_from_samples(samples: list[str]) -> Optional[str]:
    if not samples:
        return None
    if all(_EMAIL_REGEX.match(sample) for sample in samples):
        return "email"
    if all(_PHONE_REGEX.match(sample) for sample in samples):
        return "phone"
    if all(_NAME_REGEX.match(sample) for sample in samples):
        return "name"
    return None


def _heuristic_regex(samples: list[str]) -> list[dict[str, Any]]:
    if not samples:
        return []
    support = len(samples)

    if all(_EMAIL_REGEX.match(sample) for sample in samples):
        return [_make_regex_candidate(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", support, 0.6)]

    if all(sample.isdigit() for sample in samples):
        return [_make_regex_candidate(r"^\d+$", support, 0.4)]

    if all(sample.isalpha() for sample in samples):
        if all(sample.isupper() for sample in samples):
            pattern = r"^[A-Z]+$"
        elif all(sample.islower() for sample in samples):
            pattern = r"^[a-z]+$"
        else:
            pattern = r"^[A-Za-z]+$"
        return [_make_regex_candidate(pattern, support, 0.5)]

    if all(sample.isalnum() for sample in samples):
        return [_make_regex_candidate(r"^[A-Za-z0-9]+$", support, 0.5)]

    return []


def _make_regex_candidate(pattern: str, support: int, generality: float) -> dict[str, Any]:
    return {"pattern": pattern, "support": support, "generality": generality}


def _llm_regex_inference(samples: list[str], llm: LLMClient) -> list[dict[str, Any]]:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You infer regular expressions for data fields. Respond with JSON array of objects: {\"pattern\", \"support\", \"generality\"}.",
            },
            {
                "role": "user",
                "content": json.dumps({"examples": samples[:5]}),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }

    try:
        response = llm.generate_text(payload)
    except Exception:  # pragma: no cover - network/LLM failures fallback to heuristics
        return []

    try:
        choices = response.get("choices") or []
        if not choices:
            return []
        content = choices[0]["message"]["content"].strip()
        data = json.loads(content)
        if not isinstance(data, list):
            return []
        regexes: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern")
            if not pattern:
                continue
            support = int(item.get("support", len(samples)))
            generality = float(item.get("generality", 0.5))
            regexes.append(_make_regex_candidate(str(pattern), support, generality))
        return regexes
    except Exception:  # pragma: no cover - parsing fallback
        return []


def _llm_profile_chunk(
    chunk_id: int,
    records: list[dict[str, Any]],
    llm: LLMClient,
    rng: RNGConfig,
) -> dict[str, Any]:
    if not records:
        return {}

    rand = rng.random()
    sample_size = min(20, len(records))
    if len(records) > sample_size:
        indices = rand.sample(range(len(records)), k=sample_size)
        sample = [records[i] for i in sorted(indices)]
    else:
        sample = records

    batch_size = 5
    prompts: list[dict[str, Any]] = []
    for slice_index in range(0, len(sample), batch_size):
        subset = sample[slice_index : slice_index + batch_size]
        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON data profiler. Analyse the records and respond with JSON containing `field_summaries` keyed by JSONPath. "
                        "For each field include optional `semantic_type`, `regex_candidates` (with pattern/support/generality), and `pii_likelihood` (0-1)."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "chunk_id": chunk_id,
                            "slice": slice_index // batch_size,
                            "records": subset,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        if rng.seed is not None:
            prompt["seed"] = rng.seed + chunk_id + slice_index
        prompts.append(prompt)

    responses = llm.map_reduce(prompts)
    combined: dict[str, Any] = {"field_summaries": {}}
    for response in responses:
        summary = _parse_llm_summary(response)
        if summary:
            _merge_field_summary(combined["field_summaries"], summary)

    return combined


def _merge_llm_summary(accumulator: "ProfileAccumulator", summary: dict[str, Any]) -> None:
    field_summaries = summary.get("field_summaries")
    if not isinstance(field_summaries, dict):
        return

    for path, details in field_summaries.items():
        stats = accumulator.fields.get(path)
        if stats is None or not isinstance(details, dict):
            continue

        semantic = details.get("semantic_type")
        if isinstance(semantic, str) and not stats.semantic_type:
            stats.semantic_type = semantic

        regex_candidates = details.get("regex_candidates")
        if isinstance(regex_candidates, list):
            for candidate in regex_candidates:
                if not isinstance(candidate, dict):
                    continue
                pattern = candidate.get("pattern")
                if not pattern or any(existing.get("pattern") == pattern for existing in stats.regex_candidates):
                    continue
                stats.regex_candidates.append(
                    {
                        "pattern": pattern,
                        "support": candidate.get("support", 0),
                        "generality": candidate.get("generality", 0.5),
                    }
                )

        enum_candidates = details.get("enum_candidates")
        if isinstance(enum_candidates, list):
            for candidate in enum_candidates:
                value = candidate.get("value") if isinstance(candidate, dict) else None
                if value is None:
                    continue
                if isinstance(value, (str, int, float)):
                    stats.categorical[value] += candidate.get("count", 1)

        pii_likelihood = details.get("pii_likelihood")
        if isinstance(pii_likelihood, (int, float)):
            stats.pii_likelihood = max(stats.pii_likelihood, float(pii_likelihood))

        anomalies = details.get("anomalies")
        if isinstance(anomalies, list):
            for anomaly in anomalies:
                if not isinstance(anomaly, dict):
                    continue
                if anomaly not in stats.anomalies:
                    stats.anomalies.append(anomaly)


def _parse_llm_summary(response: dict[str, Any]) -> Optional[dict[str, Any]]:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    summaries = data.get("field_summaries")
    if not isinstance(summaries, dict):
        return None
    return data


def _merge_field_summary(
    existing: dict[str, Any],
    new_summary: dict[str, Any],
) -> None:
    for path, details in new_summary.get("field_summaries", {}).items():
        if not isinstance(details, dict):
            continue
        target = existing.setdefault(path, {})
        semantic = details.get("semantic_type")
        if semantic and "semantic_type" not in target:
            target["semantic_type"] = semantic
        regex_candidates = details.get("regex_candidates")
        if isinstance(regex_candidates, list):
            target.setdefault("regex_candidates", [])
            for candidate in regex_candidates:
                if not isinstance(candidate, dict):
                    continue
                pattern = candidate.get("pattern")
                if not pattern or any(existing_candidate.get("pattern") == pattern for existing_candidate in target["regex_candidates"]):
                    continue
                target["regex_candidates"].append(candidate)
        enum_candidates = details.get("enum_candidates")
        if isinstance(enum_candidates, list):
            target.setdefault("enum_candidates", [])
            for candidate in enum_candidates:
                if candidate not in target["enum_candidates"]:
                    target["enum_candidates"].append(candidate)
        pii = details.get("pii_likelihood")
        if isinstance(pii, (int, float)):
            target["pii_likelihood"] = max(float(pii), float(target.get("pii_likelihood", 0.0)))
        anomalies = details.get("anomalies")
        if isinstance(anomalies, list):
            target.setdefault("anomalies", [])
            for anomaly in anomalies:
                if anomaly not in target["anomalies"]:
                    target["anomalies"].append(anomaly)



def _infer_type(value: Any) -> tuple[str, Any, Optional[datetime]]:
    if value is None:
        return "null", None, None
    if isinstance(value, bool):
        return "boolean", value, None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number", value, None
    if isinstance(value, dict):
        return "object", value, None
    if isinstance(value, list):
        return "array", value, None
    if isinstance(value, str):
        dt = _parse_datetime(value)
        if dt is not None:
            return "datetime", value, dt
        return "string", value, None
    return "string", str(value), None


def _parse_datetime(value: str) -> Optional[datetime]:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None
