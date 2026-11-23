
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


def remove_duplicate_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove duplicated observations."""
    return dataframe.drop_duplicates().copy()


def fill_missing_numeric(
    dataframe: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """Fill missing numerical values."""
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


def fill_missing_categorical(
    dataframe: pd.DataFrame,
    strategy: str = "mode",
    custom_value: str = "Unknown",
) -> pd.DataFrame:
    """Fill missing categorical values."""
    if strategy not in {"mode", "constant"}:
        raise ValueError(
            "Strategy must be 'mode' or 'constant'."
        )

    result = dataframe.copy()
    columns = result.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in columns:
        if strategy == "constant":
            result[column] = result[column].fillna(
                custom_value
            )
        else:
            modes = result[column].mode(dropna=True)
            if not modes.empty:
                result[column] = result[column].fillna(
                    modes.iloc[0]
                )

    return result


def remove_incomplete_rows(
    dataframe: pd.DataFrame,
    max_missing_fraction: float = 0.5,
) -> pd.DataFrame:
    """Remove rows exceeding a missing-value threshold."""
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if dataframe.shape[1] == 0:
        return dataframe.copy()

    missing_fraction = dataframe.isna().mean(axis=1)

    return dataframe.loc[
        missing_fraction <= max_missing_fraction
    ].copy()


def clean_dataset(
    dataframe: pd.DataFrame,
    fill_strategy: str = "median",
    categorical_strategy: str = "mode",
    max_missing_fraction: float | None = None,
) -> pd.DataFrame:
    """Apply a sequence of data cleaning operations."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if max_missing_fraction is not None:
        result = remove_incomplete_rows(
            dataframe,
            max_missing_fraction,
        )
    else:
        result = dataframe.copy()

    result = remove_empty_columns(result)
    result = remove_empty_rows(result)
    result = remove_duplicate_rows(result)

    result = fill_missing_numeric(
        result,
        strategy=fill_strategy,
    )

    result = fill_missing_categorical(
        result,
        strategy=categorical_strategy,
    )

    return result
