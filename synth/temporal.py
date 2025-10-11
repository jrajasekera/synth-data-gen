"""Temporal pattern detection utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def analyze_temporal_patterns(series: Iterable[pd.Timestamp]) -> dict[str, str]:
    """Detect trend and seasonality patterns.

    Placeholder returning neutral results until a full implementation is added.
    """

    return {"trend": "stable", "seasonality": "none"}
