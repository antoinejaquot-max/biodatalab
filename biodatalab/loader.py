
"""Data loading and validation utilities for BioDataLab."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls"}


def _validate_headers(headers):
    """Validate and normalize column names."""

    normalized = [str(name).strip() for name in headers]

    if any(not name for name in normalized):
        raise ValueError("Dataset contains blank column names.")

    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate column names detected.")

    return normalized


def load_dataset(
    filepath: str | Path,
    separator: str = ",",
    encoding: str = "utf-8",
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Load and validate a CSV or Excel dataset."""

    if not isinstance(separator, str) or len(separator) != 1:
        raise ValueError("CSV separator must be a single character.")

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {suffix}")

    try:
        if suffix == ".csv":
            # Inspect raw headers first, before pandas can rename duplicates.
            with path.open(
                mode="r",
                encoding=encoding,
                newline="",
            ) as file:
                reader = csv.reader(file, delimiter=separator)
                headers = next(reader, None)

            if headers is None:
                raise ValueError("CSV file has no header row.")

            normalized_headers = _validate_headers(headers)

            dataframe = pd.read_csv(
                path,
                sep=separator,
                encoding=encoding,
            )

            dataframe.columns = normalized_headers

        else:
            dataframe = pd.read_excel(
                path,
                sheet_name=sheet_name,
            )

            dataframe.columns = _validate_headers(
                dataframe.columns
            )

    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Could not load dataset '{path}': {error}"
        ) from error

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    if dataframe.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    if dataframe.columns.isna().any():
        raise ValueError("Some column names are missing.")

    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> dict:
    """Return comprehensive structural diagnostics for a dataset."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    if dataframe.columns.isna().any():
        raise ValueError("Some column names are missing.")

    column_details = {}
    empty_columns = []
    constant_columns = []

    for column in dataframe.columns:
        series = dataframe[column]

        is_empty = series.isna().all()

        if is_empty:
            empty_columns.append(column)

        non_missing = series.dropna()
        is_constant = (
            len(non_missing) > 0
            and non_missing.nunique() <= 1
        )

        if is_constant:
            constant_columns.append(column)

        numeric_candidate = (
            pd.api.types.is_numeric_dtype(series)
            or (
                len(non_missing) > 0
                and pd.to_numeric(
                    non_missing,
                    errors="coerce",
                ).notna().all()
            )
        )

        non_finite_numeric_values = 0

        if numeric_candidate:
            numeric_values = pd.to_numeric(
                series,
                errors="coerce",
            )

            non_finite_numeric_values = int(
                np.isinf(numeric_values).sum()
            )

        column_details[column] = {
            "dtype": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "missing_percentage": (
                float(series.isna().mean() * 100)
                if len(series) > 0
                else 0.0
            ),
            "unique_values": int(series.nunique(dropna=True)),
            "numeric_candidate": bool(numeric_candidate),
            "empty": bool(is_empty),
            "constant": bool(is_constant),
            "non_finite_numeric_values": (
                non_finite_numeric_values
            ),
        }

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "column_names": list(dataframe.columns),
        "memory_usage_bytes": int(
            dataframe.memory_usage(deep=True).sum()
        ),
        "empty_columns": empty_columns,
        "constant_columns": constant_columns,
        "column_details": column_details,
    }
