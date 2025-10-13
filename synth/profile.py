"""LLM-driven profiling orchestrator."""

from __future__ import annotations

import json
import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .io import ChunkingConfig, JSONStream
from .llm import LLMClient, LLMResponseError
from .schemas import PROFILE_SUMMARY_SCHEMA, REGEX_CANDIDATE_ARRAY_SCHEMA
from .utils import RNGConfig, stable_hash

ProgressCallback = Callable[[int], None]


logger = logging.getLogger(__name__)


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
    llm_confidence: float = 0.0
    narratives: list[str] = field(default_factory=list)

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

        heuristic_confidence = self._heuristic_confidence(types_summary)
        llm_confidence = round(max(0.0, min(self.llm_confidence, 1.0)), 4)
        overall_confidence = round(max(heuristic_confidence, llm_confidence), 4)

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
            "confidence": {
                "heuristic": heuristic_confidence,
                "llm": llm_confidence,
                "overall": overall_confidence,
            },
            "narratives": list(self.narratives),
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

    def _heuristic_confidence(self, types_summary: list[dict[str, Any]]) -> float:
        if self.total == 0:
            return 0.0
        coverage = min(1.0, self.total / (self.total + 50))
        dominant_confidence = types_summary[0]["confidence"] if types_summary else 0.0
        anomaly_penalty = min(len(self.anomalies), 5) * 0.05
        base = 0.6 * coverage + 0.4 * dominant_confidence
        if self.semantic_type:
            base = min(1.0, base + 0.1)
        base = max(0.0, base * (1.0 - anomaly_penalty))
        return round(base, 4)


MAX_COMPOSITE_TRACKED = 20000
MAX_COMPOSITE_KEY_SIZE = 3


@dataclass(slots=True)
class CompositeTracker:
    values: set[str] = field(default_factory=set)
    truncated: bool = False
    observations: int = 0

    def add(self, fingerprint: str) -> None:
        self.observations += 1
        if self.truncated:
            return
        if len(self.values) >= MAX_COMPOSITE_TRACKED:
            self.truncated = True
            self.values.clear()
            return
        self.values.add(fingerprint)

    def uniqueness_ratio(self) -> float:
        if self.observations == 0 or self.truncated:
            return 0.0
        return min(len(self.values) / self.observations, 1.0)


def _serialize_scalar(value: Any) -> Optional[str]:
    if value is None or isinstance(value, (dict, list)):
        return None
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return None

class ProfileAccumulator:
    """Accumulate statistics across records."""

    def __init__(self) -> None:
        self.fields: dict[str, FieldStats] = {}
        self.composite_trackers: dict[tuple[str, ...], "CompositeTracker"] = {}

    def record(self, record: dict[str, Any]) -> None:
        self._track_composite_candidates(record)
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

    def _track_composite_candidates(self, record: dict[str, Any]) -> None:
        scalar_paths: dict[str, str] = {}
        for key, value in record.items():
            serialized = _serialize_scalar(value)
            if serialized is None:
                continue
            path = f"$.{key}"
            scalar_paths[path] = serialized

        if len(scalar_paths) < 2:
            return

        ordered_paths = sorted(scalar_paths)
        max_size = min(len(ordered_paths), MAX_COMPOSITE_KEY_SIZE)
        for size in range(2, max_size + 1):
            for combo in combinations(ordered_paths, size):
                tracker = self.composite_trackers.setdefault(combo, CompositeTracker())
                fingerprint = stable_hash([scalar_paths[path] for path in combo])
                tracker.add(fingerprint)

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
            except Exception as error:  # pragma: no cover - network faults logged
                logger.warning(
                    "LLM profiling skipped for chunk %s due to error: %s",
                    chunk_index,
                    error,
                    exc_info=True,
                )

        chunk_index += 1

    for stats in accumulator.fields.values():
        _apply_semantic_inference(stats, llm)

    field_summaries = accumulator.build()
    pk_candidates = []
    pk_path_sets: set[tuple[str, ...]] = set()

    for path, stats in accumulator.fields.items():
        if stats.is_unique():
            pk_candidates.append(
                {
                    "paths": [path],
                    "uniqueness": stats.uniqueness_ratio(),
                    "confidence": 0.8 if stats.uniqueness_truncated else 1.0,
                }
            )
            pk_path_sets.add((path,))

    composite_candidates = _composite_key_candidates(accumulator, total_records)
    for candidate in composite_candidates:
        path_key = tuple(candidate["paths"])
        if path_key in pk_path_sets:
            continue
        pk_candidates.append(candidate)
        pk_path_sets.add(path_key)

    fk_candidates = _foreign_key_candidates(accumulator)

    return {
        "meta": {"records_total": total_records},
        "field_summaries": field_summaries,
        "keys": {
            "pk_candidates": pk_candidates,
            "fk_candidates": fk_candidates,
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
        "max_tokens": 4096,
    }

    try:
        data = llm.generate_text(
            payload,
            schema=REGEX_CANDIDATE_ARRAY_SCHEMA,
            parse_json=True,
        )
    except (LLMResponseError, Exception) as error:  # pragma: no cover - network/LLM failures fallback
        logger.warning("LLM regex inference failed: %s", error, exc_info=True)
        return []

    try:
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
    sample_size = min(5, len(records))
    if len(records) > sample_size:
        indices = rand.sample(range(len(records)), k=sample_size)
        sample = [_compact_record_for_llm(records[i]) for i in sorted(indices)]
    else:
        sample = [_compact_record_for_llm(record) for record in records]

    batch_size = 3
    prompts: list[dict[str, Any]] = []
    for slice_index in range(0, len(sample), batch_size):
        subset = sample[slice_index : slice_index + batch_size]
        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON data profiler. Analyse the records and respond with JSON containing `field_summaries` keyed by JSONPath. "
                        "For each field include optional `semantic_type`, `regex_candidates` (with pattern/support/generality), and `pii_likelihood` (0-1). "
                        "Always emit a numeric `confidence` between 0 and 1 for each field, and when applicable include a `narratives` array of short anomaly explanations."
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
            "max_tokens": 4096,
        }
        if rng.seed is not None:
            prompt["seed"] = rng.seed + chunk_id + slice_index
        prompts.append(prompt)

    try:
        responses = llm.map_reduce(
            prompts,
            schema=PROFILE_SUMMARY_SCHEMA,
            parse_json=True,
        )
    except Exception as error:  # pragma: no cover - log and fall back
        logger.warning("LLM map_reduce failed for chunk %s: %s", chunk_id, error, exc_info=True)
        return {}
    combined: dict[str, Any] = {"field_summaries": {}}
    for response in responses:
        if isinstance(response, dict) and "field_summaries" in response:
            _merge_field_summary(combined["field_summaries"], response)

    return combined


def _compact_record_for_llm(
    record: dict[str, Any],
    *,
    max_depth: int = 4,
    max_list_items: int = 5,
    max_string_length: int = 256,
    max_keys: int = 20,
) -> dict[str, Any]:
    """Return a truncated copy of a record to keep LLM prompts lightweight."""

    def _compact(value: Any, depth: int) -> Any:
        if depth >= max_depth:
            return "<truncated>"

        if isinstance(value, dict):
            trimmed: dict[str, Any] = {}
            for idx, (key, child) in enumerate(value.items()):
                if idx >= max_keys:
                    trimmed["<truncated_keys>"] = len(value) - max_keys
                    break
                trimmed[key] = _compact(child, depth + 1)
            return trimmed

        if isinstance(value, list):
            if not value:
                return []
            items = [_compact(item, depth + 1) for item in value[:max_list_items]]
            if len(value) > max_list_items:
                items.append(f"<truncated_list:{len(value) - max_list_items}>")
            return items

        if isinstance(value, str):
            if len(value) > max_string_length:
                return value[:max_string_length] + "…"
            return value

        return value

    return _compact(record, 0)


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

        confidence = details.get("confidence")
        if isinstance(confidence, (int, float)):
            stats.llm_confidence = max(stats.llm_confidence, float(confidence))

        anomalies = details.get("anomalies")
        if isinstance(anomalies, list):
            for anomaly in anomalies:
                if not isinstance(anomaly, dict):
                    continue
                if anomaly not in stats.anomalies:
                    stats.anomalies.append(anomaly)

        narratives = details.get("narratives")
        if isinstance(narratives, list):
            for narrative in narratives:
                if isinstance(narrative, str) and narrative not in stats.narratives:
                    stats.narratives.append(narrative)


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



def _composite_key_candidates(accumulator: ProfileAccumulator, total_records: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if total_records == 0:
        return candidates

    for paths, tracker in accumulator.composite_trackers.items():
        if len(paths) < 2:
            continue
        if tracker.truncated or tracker.observations == 0:
            continue
        uniqueness = tracker.uniqueness_ratio()
        if uniqueness < 0.95:
            continue
        coverage = tracker.observations / total_records
        confidence = min(0.95, 0.5 + 0.3 * uniqueness + 0.2 * coverage)
        candidates.append(
            {
                "paths": list(paths),
                "uniqueness": round(uniqueness, 4),
                "coverage": round(coverage, 4),
                "confidence": round(confidence, 4),
            }
        )

    candidates.sort(key=lambda item: (-item["confidence"], len(item["paths"])))
    return candidates


def _foreign_key_candidates(accumulator: ProfileAccumulator) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    field_entries: list[tuple[str, FieldStats, set[str]]] = []

    for path, stats in accumulator.fields.items():
        if stats.uniqueness_truncated:
            continue
        non_null = stats.total - stats.nulls
        if non_null == 0:
            continue
        if not stats.uniqueness:
            continue
        field_entries.append((path, stats, set(stats.uniqueness)))

    for child_path, child_stats, child_values in field_entries:
        if len(child_values) == 0:
            continue
        child_required = child_stats.required_rate()
        for parent_path, parent_stats, parent_values in field_entries:
            if child_path == parent_path:
                continue
            if len(child_values) > len(parent_values):
                continue
            if not parent_stats.is_unique():
                continue
            if child_values.issubset(parent_values):
                support = len(child_values) / max(len(parent_values), 1)
                confidence = min(0.95, 0.4 + 0.4 * child_required + 0.15 * support + 0.05)
                candidates.append(
                    {
                        "parent": parent_path,
                        "child": child_path,
                        "coverage": round(child_required, 4),
                        "support": round(support, 4),
                        "confidence": round(confidence, 4),
                    }
                )

    unique_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["parent"], candidate["child"])
        existing = unique_pairs.get(key)
        if existing is None or candidate["confidence"] > existing["confidence"]:
            unique_pairs[key] = candidate

    return sorted(unique_pairs.values(), key=lambda item: -item["confidence"])



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
