"""Correlation modeling helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def compute_correlations(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute basic Pearson and Cramér's V correlations for a record sample."""

    frame = pd.DataFrame(list(records))
    if frame.empty:
        return {"pearson": {}, "cramers_v": {}}

    correlations: dict[str, Any] = {"pearson": {}, "cramers_v": {}}

    numeric_cols = frame.select_dtypes(include=["number"]).columns
    if len(numeric_cols) >= 2:
        pearson_matrix = frame[numeric_cols].corr().fillna(0.0)
        correlations["pearson"] = {
            column: pearson_matrix[column].round(4).to_dict()
            for column in pearson_matrix.columns
        }

    categorical_cols = frame.select_dtypes(include=["object", "category", "bool"]).columns
    if len(categorical_cols) >= 2:
        prepared = frame[categorical_cols].copy()
        prepared = prepared.fillna("<NA>").astype(str)
        for idx, col_a in enumerate(prepared.columns):
            for col_b in prepared.columns[idx + 1 :]:
                table = pd.crosstab(prepared[col_a], prepared[col_b])
                if table.size == 0 or table.values.sum() == 0:
                    continue
                chi2, _, _, _ = chi2_contingency(table, correction=False)
                n = table.values.sum()
                if n == 0:
                    continue
                r, c = table.shape
                if min(r, c) <= 1:
                    cramers_v = 0.0
                else:
                    cramers_v = float(np.sqrt(chi2 / (n * (min(r, c) - 1))))
                correlations["cramers_v"].setdefault(col_a, {})[col_b] = round(cramers_v, 4)
                correlations["cramers_v"].setdefault(col_b, {})[col_a] = round(cramers_v, 4)

    return correlations
