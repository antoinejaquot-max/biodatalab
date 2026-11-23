
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnStatistics:
    count: int
    missing: int
    mean: float | None
    median: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    q1: float | None
    q3: float | None


def analyze_column(
    series: pd.Series,
) -> ColumnStatistics:
    """Calculate descriptive statistics for a column."""

    numeric = pd.to_numeric(series, errors="coerce")

    valid = numeric.dropna()

    if valid.empty:
        return ColumnStatistics(
            count=0,
            missing=len(series),
            mean=None,
            median=None,
            std=None,
            minimum=None,
            maximum=None,
            q1=None,
            q3=None,
        )

    return ColumnStatistics(
        count=int(valid.count()),
        missing=int(series.isna().sum()),
        mean=float(valid.mean()),
        median=float(valid.median()),
        std=float(valid.std()),
        minimum=float(valid.min()),
        maximum=float(valid.max()),
        q1=float(valid.quantile(0.25)),
        q3=float(valid.quantile(0.75)),
    )


def analyze_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, ColumnStatistics]:
    """Analyze every numeric column in a dataset."""

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    results = {}

    for column in dataframe.columns:
        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            results[str(column)] = analyze_column(
                dataframe[column]
            )

    return results


def calculate_iqr_outliers(
    series: pd.Series,
    multiplier: float = 1.5,
) -> pd.Series:
    """Identify potential outliers using the IQR method."""

    if multiplier <= 0:
        raise ValueError("Multiplier must be positive.")

    values = pd.to_numeric(series, errors="coerce")

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return (
        (values < lower_bound)
        | (values > upper_bound)
    ).fillna(False)


def calculate_correlation(
    dataframe: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate correlations between numeric variables."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Unsupported correlation method.")

    numeric = dataframe.select_dtypes(
        include=[np.number]
    )

    if numeric.shape[1] < 2:
        raise ValueError(
            "At least two numeric columns are required."
        )

    return numeric.corr(method=method)
