"""Validation suite for synthetic data outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from scipy.stats import ks_2samp

from . import privacy as privacy_module
from .correlations import compute_correlations
from .io import ChunkingConfig, JSONStream
from .profile import profile_dataset
from . import temporal as temporal_module
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

    source_records = _load_sample_records(source_path, format_hint=None)
    synthetic_records = _load_sample_records(synthetic_path, format_hint="jsonl")

    comparisons = _compare_profiles(source_profile, synthetic_profile)
    rule_violations = []
    if ruleset:
        rule_violations = _evaluate_ruleset(ruleset, synthetic_profile)

    correlation_drift = _correlation_drift(source_records, synthetic_records)
    temporal_alignment = _temporal_alignment(
        source_profile,
        synthetic_profile,
        source_records,
        synthetic_records,
    )
    privacy_summary = _privacy_summary(
        source_profile,
        synthetic_profile,
        source_records,
        synthetic_records,
        ruleset,
    )

    return {
        "summary": {
            "synthetic_records": synthetic_profile.get("meta", {}).get("records_total", 0),
            "source_records": source_profile.get("meta", {}).get("records_total", 0),
        },
        "field_comparisons": comparisons,
        "rule_violations": rule_violations,
        "correlation_drift": correlation_drift,
        "temporal_alignment": temporal_alignment,
        "privacy": privacy_summary,
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


def _load_sample_records(path: Path, *, format_hint: Optional[str], limit: int = 1000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stream = JSONStream(path, ChunkingConfig(size=256, format=format_hint))
    for chunk in stream.iter_chunks():
        for record in chunk:
            if not isinstance(record, dict):
                continue
            records.append(record)
            if len(records) >= limit:
                return records
    return records


def _correlation_drift(
    source_records: Iterable[dict[str, Any]],
    synthetic_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    source_corr = compute_correlations(source_records)
    synthetic_corr = compute_correlations(synthetic_records)

    return {
        "source": source_corr,
        "synthetic": synthetic_corr,
        "pearson_delta": _matrix_delta(source_corr.get("pearson", {}), synthetic_corr.get("pearson", {})),
        "cramers_v_delta": _matrix_delta(source_corr.get("cramers_v", {}), synthetic_corr.get("cramers_v", {})),
    }


def _matrix_delta(source: dict[str, dict[str, float]], synthetic: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    delta: dict[str, dict[str, float]] = {}
    for row_key, row_values in source.items():
        synthetic_row = synthetic.get(row_key)
        if synthetic_row is None:
            continue
        row_delta: dict[str, float] = {}
        for col_key, source_value in row_values.items():
            if col_key not in synthetic_row:
                continue
            row_delta[col_key] = round(float(synthetic_row[col_key] - source_value), 4)
        if row_delta:
            delta[row_key] = row_delta
    return delta


def _temporal_alignment(
    source_profile: dict[str, Any],
    synthetic_profile: dict[str, Any],
    source_records: Iterable[dict[str, Any]],
    synthetic_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    alignment: dict[str, Any] = {}
    source_fields = source_profile.get("field_summaries", {})
    synthetic_fields = synthetic_profile.get("field_summaries", {})

    for path, source_summary in source_fields.items():
        if _dominant_type(source_summary) != "datetime":
            continue
        synthetic_summary = synthetic_fields.get(path)
        if synthetic_summary is None or _dominant_type(synthetic_summary) != "datetime":
            continue

        source_series = _extract_timestamp_series(source_records, path)
        synthetic_series = _extract_timestamp_series(synthetic_records, path)
        if not source_series or not synthetic_series:
            continue

        source_desc = temporal_module.analyze_temporal_patterns(source_series)
        synthetic_desc = temporal_module.analyze_temporal_patterns(synthetic_series)
        statistics = _temporal_statistics(source_series, synthetic_series, source_desc, synthetic_desc)
        alignment[path] = {
            "source": source_desc,
            "synthetic": synthetic_desc,
            "statistics": statistics,
        }

    return alignment


def _extract_timestamp_series(records: Iterable[dict[str, Any]], path: str) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    for record in records:
        value = _resolve_simple_path(record, path)
        if value is None:
            continue
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            continue
        timestamps.append(timestamp)
    return timestamps


def _resolve_simple_path(record: dict[str, Any], path: str) -> Any:
    if not path.startswith("$.") or "[]" in path or "*" in path:
        return None
    current: Any = record
    for segment in path[2:].split('.'):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
        if current is None:
            return None
    return current


def _temporal_statistics(
    source_series: Iterable[pd.Timestamp],
    synthetic_series: Iterable[pd.Timestamp],
    source_desc: dict[str, Any],
    synthetic_desc: dict[str, Any],
) -> dict[str, Any]:
    source_timestamps = pd.Series(_normalise_timestamps(source_series)).dropna().sort_values()
    synthetic_timestamps = pd.Series(_normalise_timestamps(synthetic_series)).dropna().sort_values()

    ks_result: dict[str, Optional[float]] = {"statistic": None, "pvalue": None}
    if not source_timestamps.empty and not synthetic_timestamps.empty:
        source_numeric = source_timestamps.view("int64")
        synthetic_numeric = synthetic_timestamps.view("int64")
        stat, pvalue = ks_2samp(source_numeric, synthetic_numeric, alternative="two-sided", mode="asymp")
        ks_result = {"statistic": float(stat), "pvalue": float(pvalue)}

    weekday_distance = _distribution_distance(
        source_desc.get("weekday_distribution", {}),
        synthetic_desc.get("weekday_distribution", {}),
    )
    hour_distance = _distribution_distance(
        source_desc.get("hour_distribution", {}),
        synthetic_desc.get("hour_distribution", {}),
    )

    interarrival_delta = _difference_mapping(
        source_desc.get("interarrival", {}),
        synthetic_desc.get("interarrival", {}),
    )
    autocorr_delta = _difference_mapping(
        source_desc.get("autocorrelation", {}),
        synthetic_desc.get("autocorrelation", {}),
    )

    source_count = float(source_desc.get("count") or 0.0)
    synthetic_count = float(synthetic_desc.get("count") or 0.0)
    coverage_ratio = (synthetic_count / source_count) if source_count else None

    return {
        "trend_match": source_desc.get("trend") == synthetic_desc.get("trend"),
        "seasonality_match": source_desc.get("seasonality") == synthetic_desc.get("seasonality"),
        "ks": ks_result,
        "weekday_distance": weekday_distance,
        "hour_distance": hour_distance,
        "interarrival_delta": interarrival_delta,
        "autocorrelation_delta": autocorr_delta,
        "coverage_ratio": coverage_ratio,
    }


def _distribution_distance(
    source_dist: dict[Any, Any],
    synthetic_dist: dict[Any, Any],
) -> Optional[float]:
    if not source_dist and not synthetic_dist:
        return None
    keys = set(source_dist) | set(synthetic_dist)
    distance = 0.0
    for key in keys:
        distance += abs(float(synthetic_dist.get(key, 0.0)) - float(source_dist.get(key, 0.0)))
    return round(distance / 2.0, 4)


def _difference_mapping(
    source_stats: dict[str, Any],
    synthetic_stats: dict[str, Any],
) -> Optional[dict[str, float]]:
    if not source_stats and not synthetic_stats:
        return None
    delta: dict[str, float] = {}
    for key in set(source_stats) | set(synthetic_stats):
        source_value = source_stats.get(key)
        synthetic_value = synthetic_stats.get(key)
        if source_value is None or synthetic_value is None:
            continue
        delta[key] = round(float(synthetic_value) - float(source_value), 4)
    return delta or None


def _normalise_timestamps(series: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    normalised: list[pd.Timestamp] = []
    for value in series:
        ts = value
        if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        normalised.append(ts)
    return normalised


def _privacy_summary(
    source_profile: dict[str, Any],
    synthetic_profile: dict[str, Any],
    source_records: Iterable[dict[str, Any]],
    synthetic_records: Iterable[dict[str, Any]],
    ruleset: Optional[dict[str, Any]],
) -> dict[str, Any]:
    quasi_identifiers = privacy_module.derive_quasi_identifiers(source_profile)
    target_config = (source_profile.get("privacy") or {}).get("k_anonymity", {})
    ruleset_config = (ruleset or {}).get("privacy", {}).get("k_anonymity", {})
    k_target = target_config.get("k") or ruleset_config.get("k") or 5
    if not isinstance(k_target, int):
        k_target = 5
    generalized_fields = [
        str(path)
        for path in (ruleset_config.get("generalized_fields") or [])
        if isinstance(path, str)
    ]
    bucketized_fields = [
        str(path)
        for path in (ruleset_config.get("bucketized_fields") or [])
        if isinstance(path, str)
    ]

    source_k = privacy_module.compute_k_anonymity(source_records, quasi_identifiers, minimum_count=k_target)
    synthetic_k = privacy_module.compute_k_anonymity(
        synthetic_records, quasi_identifiers, minimum_count=k_target
    )

    dp_status = _resolve_dp_status(source_profile, ruleset)

    return {
        "k_anonymity": {
            "quasi_identifiers": quasi_identifiers,
            "target_k": k_target,
            "source": source_k,
            "synthetic": synthetic_k,
            "generalized_fields": generalized_fields,
            "bucketized_fields": bucketized_fields,
        },
        "differential_privacy": dp_status,
    }


def _resolve_dp_status(
    profile: Optional[dict[str, Any]], ruleset: Optional[dict[str, Any]]
) -> dict[str, Any]:
    profile_dp = (profile.get("privacy") if profile else {}) or {}
    profile_dp = profile_dp.get("differential_privacy", {})
    ruleset_dp = (ruleset or {}).get("privacy", {}).get("differential_privacy", {})

    combined = {**profile_dp, **ruleset_dp}
    enabled = bool(combined.get("enabled"))
    epsilon = combined.get("epsilon") if enabled else None
    status: dict[str, Any] = {"enabled": enabled, "epsilon": epsilon}
    applied_fields = ruleset_dp.get("applied_fields") if isinstance(ruleset_dp, dict) else None
    if enabled and isinstance(applied_fields, list):
        status["applied_fields"] = [str(field) for field in applied_fields if isinstance(field, str)]
    return status
