
"""Reproducibility metadata utilities for BioDataLab."""

from __future__ import annotations

import hashlib
import platform
from typing import Any

import pandas as pd

from biodatalab import __version__


def generate_dataset_metadata(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Generate reproducibility metadata for a dataset."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    try:
        hashed_data = pd.util.hash_pandas_object(
            dataframe,
            index=True,
        ).values.tobytes()

        hashed_schema = repr(
            (
                tuple(dataframe.columns),
                tuple(str(dtype) for dtype in dataframe.dtypes),
                dataframe.shape,
            )
        ).encode("utf-8")

        fingerprint = hashlib.sha256(
            hashed_schema + hashed_data
        ).hexdigest()

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Unable to generate dataset fingerprint."
        ) from exc

    return {
        "dataset_fingerprint": fingerprint,
        "dataset_shape": tuple(dataframe.shape),
        "column_names": [str(column) for column in dataframe.columns],
        "column_types": {
            str(column): str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "biodatalab_version": __version__,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
    }
