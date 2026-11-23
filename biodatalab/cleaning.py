
"""Data cleaning utilities for BioDataLab."""

from __future__ import annotations

import pandas as pd


def remove_empty_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove columns containing only missing values."""

    return dataframe.dropna(axis=1, how="all").copy()


def remove_empty_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove rows containing only missing values."""

    return dataframe.dropna(axis=0, how="all").copy()


def fill_missing_numeric(
    dataframe: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """Fill missing numerical values using a statistical strategy."""

    if strategy not in {"mean", "median", "zero"}:
        raise ValueError(
            "Strategy must be mean, median or zero."
        )

    result = dataframe.copy()

    numeric_columns = result.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        if strategy == "mean":
            value = result[column].mean()
        elif strategy == "median":
            value = result[column].median()
        else:
            value = 0

        if pd.notna(value):
            result[column] = result[column].fillna(value)

    return result


def remove_duplicate_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove duplicated observations."""

    return dataframe.drop_duplicates().copy()


def clean_dataset(
    dataframe: pd.DataFrame,
    fill_strategy: str = "median",
) -> pd.DataFrame:
    """Apply a sequence of basic cleaning operations."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    result = remove_empty_columns(dataframe)
    result = remove_empty_rows(result)
    result = remove_duplicate_rows(result)
    result = fill_missing_numeric(
        result,
        strategy=fill_strategy,
    )

    return result
