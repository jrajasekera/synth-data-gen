"""Temporal pattern detection utilities."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def analyze_temporal_patterns(series: Iterable[pd.Timestamp]) -> dict[str, Any]:
    """Detect coarse-grained temporal signals and summarise distributional statistics."""

    timestamps = _normalise_series(series)
    if timestamps.empty:
        return {
            "trend": "stable",
            "seasonality": "none",
            "count": 0,
            "weekday_distribution": {},
            "hour_distribution": {},
            "interarrival": {},
            "autocorrelation": {},
        }

    signature = _temporal_signature(timestamps)
    signature["trend"] = _detect_trend(timestamps)
    signature["seasonality"] = _detect_seasonality(timestamps)
    return signature


def _normalise_series(series: Iterable[pd.Timestamp]) -> pd.Series:
    cleaned = []
    for value in series:
        if isinstance(value, pd.Timestamp) and value.tzinfo is not None:
            cleaned.append(value.tz_localize(None))
        else:
            cleaned.append(value)
    values = pd.Series(cleaned, dtype="datetime64[ns]").dropna()
    if values.empty:
        return values
    return values.sort_values().reset_index(drop=True)


def _temporal_signature(timestamps: pd.Series) -> dict[str, Any]:
    """Return reusable descriptive statistics for downstream validation comparisons."""

    weekday_counts = timestamps.dt.dayofweek.value_counts(normalize=True).sort_index()
    hour_counts = timestamps.dt.hour.value_counts(normalize=True).sort_index()

    deltas = timestamps.diff().dropna()
    interarrival: dict[str, float] = {}
    autocorrelation: dict[str, float] = {}
    if not deltas.empty:
        seconds = deltas.dt.total_seconds()
        interarrival = {
            "mean": float(seconds.mean()),
            "median": float(seconds.median()),
            "std": float(seconds.std(ddof=0) if seconds.size > 1 else 0.0),
            "p95": float(np.percentile(seconds, 95)),
        }
        if seconds.size >= 2:
            lag_1 = seconds.autocorr(lag=1)
            if not pd.isna(lag_1):
                autocorrelation["lag_1"] = float(lag_1)

    return {
        "count": int(timestamps.size),
        "weekday_distribution": {int(idx): float(val) for idx, val in weekday_counts.items()},
        "hour_distribution": {int(idx): float(val) for idx, val in hour_counts.items()},
        "interarrival": interarrival,
        "autocorrelation": autocorrelation,
    }


def _detect_trend(timestamps: pd.Series) -> str:
    """Classify trend direction based on average successive deltas."""

    if timestamps.size < 2:
        return "stable"

    deltas = timestamps.diff().dropna()
    if deltas.empty:
        return "stable"

    seconds = deltas.dt.total_seconds()
    mean_delta = seconds.mean()
    threshold = np.std(seconds) if seconds.size > 1 else abs(mean_delta)
    if threshold == 0:
        threshold = 1.0

    if mean_delta > threshold * 0.1:
        return "increasing"
    if mean_delta < -threshold * 0.1:
        return "decreasing"
    return "stable"


def _detect_seasonality(timestamps: pd.Series) -> str:
    """Infer seasonality from weekday/hour distributions."""

    weekday_counts = timestamps.dt.dayofweek.value_counts(normalize=True)
    if not weekday_counts.empty:
        spread = weekday_counts.max() - weekday_counts.min()
        if spread > 0.2:
            return "weekly"

    hour_counts = timestamps.dt.hour.value_counts(normalize=True)
    if not hour_counts.empty:
        spread = hour_counts.max() - hour_counts.min()
        if spread > 0.2:
            return "daily"

    return "none"
