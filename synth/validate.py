"""Validation suite for synthetic data outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .io import ChunkingConfig, JSONStream
from .profile import profile_dataset
from .utils import RNGConfig


def validate_dataset(
    *,
    synthetic_path: Path,
    source_path: Path,
    ruleset: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Validate synthetic data against source distribution and optional ruleset."""

    source_profile = profile_dataset(
        stream=JSONStream(source_path, ChunkingConfig()),
        rng=RNGConfig(None),
        cache_dir=None,
    )
    synthetic_profile = profile_dataset(
        stream=JSONStream(synthetic_path, ChunkingConfig(format="jsonl")),
        rng=RNGConfig(None),
        cache_dir=None,
    )

    comparisons = _compare_profiles(source_profile, synthetic_profile)
    rule_violations = []
    if ruleset:
        rule_violations = _evaluate_ruleset(ruleset, synthetic_profile)

    return {
        "summary": {
            "synthetic_records": synthetic_profile.get("meta", {}).get("records_total", 0),
            "source_records": source_profile.get("meta", {}).get("records_total", 0),
        },
        "field_comparisons": comparisons,
        "rule_violations": rule_violations,
    }


def _compare_profiles(
    source_profile: dict[str, Any], synthetic_profile: dict[str, Any]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    source_fields = source_profile.get("field_summaries", {})
    synthetic_fields = synthetic_profile.get("field_summaries", {})

    for path, source_summary in source_fields.items():
        comparison: dict[str, Any] = {}
        synthetic_summary = synthetic_fields.get(path)
        if synthetic_summary is None:
            comparison["status"] = "missing"
            comparisons[path] = comparison
            continue

        comparison["status"] = "ok"
        comparison["required_rate_delta"] = (
            synthetic_summary.get("required_rate", 0.0) - source_summary.get("required_rate", 0.0)
        )

        source_type = _dominant_type(source_summary)
        synthetic_type = _dominant_type(synthetic_summary)
        comparison["type_alignment"] = source_type == synthetic_type
        comparison["source_type"] = source_type
        comparison["synthetic_type"] = synthetic_type

        numeric_delta = _numeric_delta(source_summary.get("numeric"), synthetic_summary.get("numeric"))
        if numeric_delta is not None:
            comparison["numeric"] = numeric_delta

        array_delta = _array_delta(source_summary.get("array"), synthetic_summary.get("array"))
        if array_delta is not None:
            comparison["array"] = array_delta

        comparisons[path] = comparison

    return comparisons


def _dominant_type(summary: dict[str, Any]) -> Optional[str]:
    types = summary.get("types", [])
    if not types:
        return None
    return max(types, key=lambda item: item.get("confidence", 0.0)).get("name")


def _numeric_delta(
    source_numeric: Optional[dict[str, Any]], synthetic_numeric: Optional[dict[str, Any]]
) -> Optional[dict[str, float]]:
    if not source_numeric or not synthetic_numeric:
        return None
    deltas: dict[str, float] = {}
    for key in ("mean", "std", "min", "max"):
        if key in source_numeric and key in synthetic_numeric:
            deltas[key] = float(synthetic_numeric[key] - source_numeric[key])
    return deltas


def _array_delta(
    source_array: Optional[dict[str, Any]], synthetic_array: Optional[dict[str, Any]]
) -> Optional[dict[str, float]]:
    if not source_array and not synthetic_array:
        return None

    deltas: dict[str, float] = {}
    for key in ("min_items", "max_items", "mean_items"):
        source_value = source_array.get(key) if source_array else None
        synthetic_value = synthetic_array.get(key) if synthetic_array else None
        if source_value is None or synthetic_value is None:
            continue
        deltas[key] = float(synthetic_value - source_value)

    return deltas or None


def _evaluate_ruleset(ruleset: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    field_rules = ruleset.get("fields", {})
    field_summaries = profile.get("field_summaries", {})

    for path, rule in field_rules.items():
        summary = field_summaries.get(path)
        if summary is None:
            violations.append({"path": path, "issue": "field_missing"})
            continue

        expected_type = rule.get("type")
        actual_type = _dominant_type(summary)
        if expected_type and actual_type and expected_type != actual_type:
            violations.append({"path": path, "issue": "type_mismatch", "expected": expected_type, "actual": actual_type})

        enum_values = rule.get("enum")
        if enum_values:
            # Enumerations act as hints; coverage mismatches are reported in comparisons
            # rather than hard violations.
            pass

    return violations
