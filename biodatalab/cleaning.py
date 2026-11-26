
"""Data cleaning utilities for BioDataLab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Validate that the input is a DataFrame."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")


def remove_empty_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove columns containing only missing values."""
    _validate_dataframe(dataframe)
    return dataframe.dropna(axis=1, how="all").copy()


def remove_empty_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove rows containing only missing values."""
    _validate_dataframe(dataframe)
    return dataframe.dropna(axis=0, how="all").copy()


def remove_duplicate_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove duplicated observations."""
    _validate_dataframe(dataframe)
    return dataframe.drop_duplicates().copy()


def replace_non_finite(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Replace infinite numerical values with missing values."""
    _validate_dataframe(dataframe)

    result = dataframe.copy()
    numeric_columns = result.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        result[column] = result[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return result


def standardize_missing_values(
    dataframe: pd.DataFrame,
    tokens: tuple[str, ...] = (
        "",
        "NA",
        "N/A",
        "na",
        "n/a",
        "NULL",
        "null",
        "None",
        "none",
        "NaN",
        "nan",
    ),
) -> pd.DataFrame:
    """Convert common textual missing-value tokens to NaN."""
    _validate_dataframe(dataframe)

    result = dataframe.copy()

    for column in result.select_dtypes(
        include=["object", "string"]
    ).columns:
        result[column] = result[column].replace(
            list(tokens),
            np.nan,
        )

    return result


def strip_whitespace(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Strip leading and trailing whitespace from text values."""
    _validate_dataframe(dataframe)

    result = dataframe.copy()

    for column in result.select_dtypes(
        include=["object", "string"]
    ).columns:
        result[column] = result[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    return result


def remove_constant_columns(
    dataframe: pd.DataFrame,
    include_empty: bool = True,
) -> pd.DataFrame:
    """Remove columns containing only one unique non-missing value."""
    _validate_dataframe(dataframe)

    result = dataframe.copy()
    columns_to_remove = []

    for column in result.columns:
        unique_count = result[column].nunique(
            dropna=True
        )

        if unique_count <= 1:
            if include_empty or unique_count == 1:
                columns_to_remove.append(column)

    return result.drop(
        columns=columns_to_remove
    ).copy()


def remove_columns_by_missingness(
    dataframe: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Remove columns exceeding a missing-value fraction threshold."""
    _validate_dataframe(dataframe)

    if not isinstance(threshold, (int, float)):
        raise TypeError("Threshold must be numeric.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if dataframe.shape[0] == 0:
        return dataframe.copy()

    missing_fraction = dataframe.isna().mean(axis=0)

    columns_to_remove = missing_fraction[
        missing_fraction > threshold
    ].index

    return dataframe.drop(
        columns=columns_to_remove
    ).copy()


def fill_missing_numeric(
    dataframe: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """Fill missing numerical values using a selected strategy."""
    _validate_dataframe(dataframe)

    if strategy not in {"mean", "median", "zero"}:
        raise ValueError(
            "Strategy must be mean, median or zero."
        )

    result = dataframe.copy()
    numeric_columns = result.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        series = result[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if strategy == "mean":
            value = series.mean()
        elif strategy == "median":
            value = series.median()
        else:
            value = 0

        if pd.notna(value):
            result[column] = series.fillna(value)
        else:
            result[column] = series

    return result


def fill_missing_categorical(
    dataframe: pd.DataFrame,
    strategy: str = "mode",
    custom_value: str = "Unknown",
) -> pd.DataFrame:
    """Fill missing categorical and text values."""
    _validate_dataframe(dataframe)

    if strategy not in {"mode", "constant"}:
        raise ValueError(
            "Strategy must be 'mode' or 'constant'."
        )

    if strategy == "constant" and not isinstance(
        custom_value, str
    ):
        raise TypeError("custom_value must be a string.")

    result = dataframe.copy()

    columns = result.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in columns:
        if strategy == "constant":
            value = custom_value
        else:
            modes = result[column].mode(
                dropna=True
            )

            if modes.empty:
                continue

            value = modes.iloc[0]

        if isinstance(result[column].dtype, pd.CategoricalDtype):
            
            if value not in result[column].cat.categories:
                result[column] = result[column].cat.add_categories(
                    [value]
                )

        result[column] = result[column].fillna(value)

    return result


def remove_incomplete_rows(
    dataframe: pd.DataFrame,
    max_missing_fraction: float = 0.5,
) -> pd.DataFrame:
    """Remove rows exceeding a missing-value threshold."""
    _validate_dataframe(dataframe)

    if not isinstance(max_missing_fraction, (int, float)):
        raise TypeError("Threshold must be numeric.")

    if (
        not np.isfinite(max_missing_fraction)
        or not 0 <= max_missing_fraction <= 1
    ):
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
    standardize_missing: bool = True,
    strip_text: bool = True,
    replace_infinite: bool = True,
    remove_constant: bool = False,
) -> pd.DataFrame:
    """Apply a configurable cleaning pipeline to a dataset."""
    _validate_dataframe(dataframe)

    if max_missing_fraction is not None:
        if not isinstance(
            max_missing_fraction, (int, float)
        ):
            raise TypeError(
                "max_missing_fraction must be numeric."
            )

        if (
            not np.isfinite(max_missing_fraction)
            or not 0 <= max_missing_fraction <= 1
        ):
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

    result = dataframe.copy()

    if strip_text:
        result = strip_whitespace(result)

    if standardize_missing:
        result = standardize_missing_values(result)

    if replace_infinite:
        result = replace_non_finite(result)

    if max_missing_fraction is not None:
        result = remove_incomplete_rows(
            result,
            max_missing_fraction,
        )

    result = remove_empty_columns(result)
    result = remove_empty_rows(result)
    result = remove_duplicate_rows(result)

    if remove_constant:
        result = remove_constant_columns(
            result,
            include_empty=True,
        )

    result = fill_missing_numeric(
        result,
        strategy=fill_strategy,
    )

    result = fill_missing_categorical(
        result,
        strategy=categorical_strategy,
    )

    return result
