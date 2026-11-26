
"""Dataset profiling utilities for BioDataLab."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a comprehensive overview of a DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names.")

    row_count, column_count = dataframe.shape
    columns: dict[str, dict[str, Any]] = {}

    for column in dataframe.columns:
        series = dataframe[column]
        non_missing = series.dropna()

        info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": (
                float(series.isna().mean() * 100)
                if row_count > 0
                else 0.0
            ),
            "unique_count": int(non_missing.nunique()),
        }

        if (
            pd.api.types.is_numeric_dtype(series)
            or non_missing.empty
        ):
            if len(non_missing) > 0 and pd.api.types.is_numeric_dtype(series):
                info.update(
                    {
                        "mean": float(non_missing.mean()),
                        "std": float(non_missing.std()),
                        "min": float(non_missing.min()),
                        "max": float(non_missing.max()),
                        "median": float(non_missing.median()),
                    }
                )
            else:
                info.update(
                    {
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "median": None,
                    }
                )

        else:
            mode = non_missing.mode()
            info["most_frequent"] = (
                mode.iloc[0] if not mode.empty else None
            )

        columns[str(column)] = info

    total_missing = int(dataframe.isna().sum().sum())

    return {
        "shape": (row_count, column_count),
        "row_count": row_count,
        "column_count": column_count,
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "total_missing_values": total_missing,
        "total_missing_percentage": (
            float(total_missing / dataframe.size * 100)
            if dataframe.size > 0
            else 0.0
        ),
        "columns": columns,
    }
