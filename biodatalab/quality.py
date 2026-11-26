
"""Data quality assessment utilities for BioDataLab."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Validate the input dataset."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")


def _safe_percentage(count: int, total: int) -> float:
    """Compute a percentage safely."""
    if total == 0:
        return 0.0
    return float(count / total * 100)


def assess_data_quality(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Assess dataset quality using descriptive quality indicators."""

    _validate_dataframe(dataframe)

    n_rows, n_columns = dataframe.shape
    total_cells = int(dataframe.size)

    missing_cells = int(dataframe.isna().sum().sum())
    duplicate_rows = int(dataframe.duplicated().sum())

    column_quality = {}

    for column in dataframe.columns:
        series = dataframe[column]

        missing_count = int(series.isna().sum())
        non_missing = series.dropna()

        unique_count = int(non_missing.nunique())
        unique_percentage = _safe_percentage(
            unique_count,
            len(non_missing),
        )

        column_quality[str(column)] = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percentage": _safe_percentage(
                missing_count,
                n_rows,
            ),
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
            "constant": unique_count <= 1,
            "empty": missing_count == n_rows,
        }

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    non_finite = {}

    for column in numeric_columns:
        values = dataframe[column]

        if pd.api.types.is_complex_dtype(values):
            finite_mask = np.isfinite(values)
        else:
            finite_mask = np.isfinite(values)

        non_finite_count = int(
            (~finite_mask & values.notna()).sum()
        )

        if non_finite_count > 0:
            non_finite[str(column)] = non_finite_count

    warnings = []

    if n_rows == 0:
        warnings.append("Dataset contains no rows.")

    if n_columns == 0:
        warnings.append("Dataset contains no columns.")

    if missing_cells > 0:
        warnings.append("Dataset contains missing values.")

    if duplicate_rows > 0:
        warnings.append("Dataset contains duplicate rows.")

    empty_columns = [
        column
        for column, metrics in column_quality.items()
        if metrics["empty"]
    ]

    if empty_columns:
        warnings.append("Dataset contains entirely empty columns.")

    constant_columns = [
        column
        for column, metrics in column_quality.items()
        if metrics["constant"] and not metrics["empty"]
    ]

    if constant_columns:
        warnings.append("Dataset contains constant columns.")

    if non_finite:
        warnings.append(
            "Dataset contains non-finite numeric values."
        )

    completeness = (
        100.0
        if total_cells == 0
        else _safe_percentage(
            total_cells - missing_cells,
            total_cells,
        )
    )

    return {
        "summary": {
            "rows": int(n_rows),
            "columns": int(n_columns),
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": _safe_percentage(
                missing_cells,
                total_cells,
            ),
            "completeness_percentage": completeness,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": _safe_percentage(
                duplicate_rows,
                n_rows,
            ),
        },
        "columns": column_quality,
        "non_finite_values": non_finite,
        "warnings": warnings,
    }
