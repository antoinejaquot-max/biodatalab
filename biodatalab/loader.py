
"""Data loading and validation utilities for BioDataLab."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import pandas as pd


SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls"}


def _validate_column_names(columns) -> None:
    names = list(columns)

    if any(pd.isna(name) for name in names):
        raise ValueError("Some column names are missing.")

    normalized = [str(name).strip() for name in names]

    if any(not name for name in normalized):
        raise ValueError("Some column names are empty.")

    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate column names detected.")

    if len(set(names)) != len(names):
        raise ValueError("Duplicate column names detected.")


def _read_csv(
    path: Path,
    separator: str,
    encoding: str,
) -> pd.DataFrame:
    if not isinstance(separator, str) or len(separator) != 1:
        raise ValueError("CSV separator must be a single character.")

    # Detect duplicate or blank headers before pandas can alter them.
    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.reader(file, delimiter=separator)
        header = next(reader, None)

    if not header:
        raise ValueError("The CSV file has no header.")

    _validate_column_names(header)

    dataframe = pd.read_csv(
        path,
        sep=separator,
        encoding=encoding,
    )

    return dataframe


def load_dataset(
    filepath: str | Path,
    separator: str = ",",
    encoding: str = "utf-8",
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Load a CSV or Excel dataset and validate its structure."""

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
            dataframe = _read_csv(path, separator, encoding)
        else:
            dataframe = pd.read_excel(
                path,
                sheet_name=sheet_name,
            )

    except (pd.errors.ParserError, UnicodeDecodeError,
            OSError, ValueError, csv.Error, StopIteration) as error:
        raise ValueError(
            f"Could not load dataset '{path}': {error}"
        ) from error

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Selected Excel sheet did not return a dataset.")

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    _validate_column_names(dataframe.columns)

    # Remove accidental surrounding whitespace in column names.
    dataframe.columns = [str(name).strip() for name in dataframe.columns]

    if dataframe.shape[1] == 0:
        raise ValueError("The dataset contains no columns.")

    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> dict:
    """Return structural, missing-value and column-level diagnostics."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    _validate_column_names(dataframe.columns)

    column_details = {}

    for column in dataframe.columns:
        series = dataframe[column]
        numeric = pd.to_numeric(series, errors="coerce")

        numeric_non_missing = numeric.dropna()
        nonfinite_count = int(
            np.isinf(numeric_non_missing.to_numpy(dtype=float)).sum()
        )

        column_details[str(column)] = {
            "dtype": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "missing_percentage": (
                float(series.isna().mean() * 100)
                if len(series)
                else 0.0
            ),
            "unique_values": int(series.nunique(dropna=True)),
            "numeric_candidate": bool(
                series.notna().any()
                and numeric.notna().sum() == series.notna().sum()
            ),
            "non_finite_numeric_values": nonfinite_count,
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
        "column_details": column_details,
        "empty_columns": [
            str(column)
            for column in dataframe.columns
            if dataframe[column].isna().all()
        ],
        "constant_columns": [
            str(column)
            for column in dataframe.columns
            if dataframe[column].nunique(dropna=True) <= 1
        ],
    }
