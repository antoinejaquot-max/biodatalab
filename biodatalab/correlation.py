
"""Correlation analysis utilities for BioDataLab."""

from __future__ import annotations

import pandas as pd


def correlation_matrix(
    dataframe: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate a correlation matrix for numeric columns."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names.")

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError(
            "Method must be 'pearson', 'spearman', or 'kendall'."
        )

    numeric = dataframe.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError("No numeric columns found.")

    numeric = numeric.loc[
        :, numeric.nunique(dropna=True) > 1
    ]

    if numeric.shape[1] < 2:
        raise ValueError(
            "At least two non-constant numeric columns are required."
        )

    return numeric.corr(method=method)


def correlation_pairs(
    dataframe: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Return unique variable pairs and their correlation coefficients."""

    matrix = correlation_matrix(dataframe, method=method)

    pairs = []

    for i, column1 in enumerate(matrix.columns):
        for column2 in matrix.columns[i + 1:]:
            value = matrix.loc[column1, column2]

            pairs.append(
                {
                    "variable_1": column1,
                    "variable_2": column2,
                    "correlation": (
                        float(value) if pd.notna(value) else None
                    ),
                }
            )

    result = pd.DataFrame(
        pairs,
        columns=["variable_1", "variable_2", "correlation"],
    )

    if not result.empty:
        result["absolute_correlation"] = (
            result["correlation"].abs()
        )
        result = result.sort_values(
            "absolute_correlation",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return result
