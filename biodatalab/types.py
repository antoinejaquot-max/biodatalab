
"""Column type detection utilities for BioDataLab."""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_column_types(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Detect statistical and inferred types for dataframe columns."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if dataframe.columns.duplicated().any():
        raise ValueError(
            "DataFrame contains duplicate column names."
        )

    result: dict[str, dict[str, Any]] = {}

    for column in dataframe.columns:
        series = dataframe[column]
        dtype = str(series.dtype)
        non_missing = series.dropna()
        unique_count = int(non_missing.nunique())
        missing_count = int(series.isna().sum())
        sample_count = len(non_missing)

        if pd.api.types.is_bool_dtype(series):
            inferred_type = "boolean"

        elif pd.api.types.is_numeric_dtype(series):
            if (
                pd.api.types.is_integer_dtype(series)
                and unique_count <= 20
                and sample_count > 0
            ):
                inferred_type = "categorical_numeric"
            else:
                inferred_type = "numeric"

        elif pd.api.types.is_datetime64_any_dtype(series):
            inferred_type = "datetime"

        elif isinstance(series.dtype, pd.CategoricalDtype):
            inferred_type = "categorical"

        else:
            inferred_type = "text"

            if sample_count > 0:
                try:
                    converted = pd.to_numeric(
                        non_missing,
                        errors="raise",
                    )

                    if (
                        converted.notna().all()
                        and unique_count > 20
                    ):
                        inferred_type = "numeric_candidate"

                except (ValueError, TypeError):
                    pass

                if inferred_type == "text":
                    try:
                        converted_dates = pd.to_datetime(
                            non_missing,
                            errors="raise",
                        )

                        if converted_dates.notna().all():
                            inferred_type = "datetime_candidate"

                    except (ValueError, TypeError):
                        pass

                if inferred_type == "text":
                    uniqueness_ratio = (
                        unique_count / sample_count
                    )

                    if (
                        unique_count <= 20
                        and uniqueness_ratio <= 0.5
                    ):
                        inferred_type = "categorical"

        result[str(column)] = {
            "dtype": dtype,
            "inferred_type": inferred_type,
            "missing_count": missing_count,
            "unique_count": unique_count,
            "non_missing_count": int(sample_count),
        }

    return result
