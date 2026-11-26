
"""Time series analysis utilities for BioDataLab."""

from __future__ import annotations

import pandas as pd


def analyze_time_series(
    dataframe: pd.DataFrame,
    time_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Analyze changes in a numeric variable over time."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names.")

    if time_column not in dataframe.columns:
        raise ValueError(f"Column '{time_column}' not found.")

    if value_column not in dataframe.columns:
        raise ValueError(f"Column '{value_column}' not found.")

    if time_column == value_column:
        raise ValueError("Time and value columns must be different.")

    if dataframe.empty:
        raise TypeError("Value column must be numeric.")

    if not pd.api.types.is_numeric_dtype(dataframe[value_column]):
        raise TypeError("Value column must be numeric.")

    result = dataframe[[time_column, value_column]].copy()

    result[time_column] = pd.to_datetime(
        result[time_column],
        errors="raise",
    )

    result = result.dropna(subset=[time_column, value_column])

    result = result.sort_values(time_column).reset_index(drop=True)

    if result.empty:
        raise ValueError("No valid observations found.")

    result["change"] = result[value_column].diff()

    result["percentage_change"] = (
        result[value_column]
        .pct_change(fill_method=None)
        .mul(100)
    )

    result["cumulative_change"] = (
        result[value_column] - result[value_column].iloc[0]
    )

    return result
