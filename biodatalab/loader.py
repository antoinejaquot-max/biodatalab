
"""Data loading utilities for BioDataLab."""

from pathlib import Path
import pandas as pd


SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls"}


def load_dataset(
    filepath: str | Path,
    separator: str = ",",
    encoding: str = "utf-8",
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Load and validate a CSV or Excel dataset."""

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
            dataframe = pd.read_csv(
                path,
                sep=separator,
                encoding=encoding,
            )
        else:
            dataframe = pd.read_excel(
                path,
                sheet_name=sheet_name,
            )
    except (pd.errors.ParserError, UnicodeDecodeError,
            OSError, ValueError) as error:
        raise ValueError(
            f"Could not load dataset '{path}': {error}"
        ) from error

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    if dataframe.columns.isna().any():
        raise ValueError("Some column names are missing.")

    if dataframe.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> dict:
    """Return basic information about a dataset."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "column_names": list(dataframe.columns),
        "memory_usage_bytes": int(
            dataframe.memory_usage(deep=True).sum()
        ),
    }
