
"""Data loading utilities for BioDataLab."""

from pathlib import Path

import pandas as pd


SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls"}


def load_dataset(
    filepath: str | Path,
    separator: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load a CSV or Excel dataset."""

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: {path.suffix}"
        )

    if path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(
            path,
            sep=separator,
            encoding=encoding,
        )
    else:
        dataframe = pd.read_excel(path)

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    return dataframe


def validate_dataset(
    dataframe: pd.DataFrame,
) -> dict:
    """Return basic information about a dataset."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "column_names": list(dataframe.columns),
    }
