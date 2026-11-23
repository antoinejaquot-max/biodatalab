
"""Outlier detection methods for BioDataLab."""

import pandas as pd


def detect_iqr(
    series: pd.Series,
    multiplier: float = 1.5,
) -> pd.Series:
    """Detect outliers using the interquartile range."""

    if multiplier <= 0:
        raise ValueError("Multiplier must be positive.")

    values = pd.to_numeric(series, errors="coerce")

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return (
        (values < lower) | (values > upper)
    ).fillna(False)


def detect_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """Detect potential outliers using standard scores."""

    if threshold <= 0:
        raise ValueError("Threshold must be positive.")

    values = pd.to_numeric(series, errors="coerce")

    mean = values.mean()
    std = values.std()

    if pd.isna(std) or std == 0:
        return pd.Series(
            False,
            index=series.index,
            dtype=bool,
        )

    z_scores = (values - mean) / std

    return (z_scores.abs() > threshold).fillna(False)


def summarize_outliers(
    dataframe: pd.DataFrame,
    method: str = "iqr",
) -> dict:
    """Summarize potential outliers in numeric columns."""

    if method not in {"iqr", "zscore"}:
        raise ValueError("Method must be iqr or zscore.")

    results = {}

    for column in dataframe.select_dtypes(
        include="number"
    ).columns:

        if method == "iqr":
            mask = detect_iqr(dataframe[column])
        else:
            mask = detect_zscore(dataframe[column])

        results[str(column)] = {
            "count": int(mask.sum()),
            "indices": dataframe.index[mask].tolist(),
        }

    return results
