
"""Export utilities for BioDataLab."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_json(value):
    """Convert common scientific Python objects into JSON-compatible values."""

    if isinstance(value, dict):
        return {
            str(key): _normalize_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Non-finite numbers cannot be exported.")
        return result

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, np.ndarray):
        return _normalize_json(value.tolist())

    if isinstance(value, pd.DataFrame):
        return _normalize_json(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return _normalize_json(value.to_dict())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Non-finite numbers cannot be exported.")
        return value

    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def export_report_json(report: dict, filepath: str | Path) -> Path:
    """Export an analysis report to JSON using an atomic file replacement."""

    if not isinstance(report, dict):
        raise TypeError("Report must be a dictionary.")

    path = Path(filepath)

    try:
        normalized = _normalize_json(report)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid report data: {error}") from error

    temporary_path = None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)

            json.dump(
                normalized,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")

        os.replace(temporary_path, path)

    except (OSError, TypeError, ValueError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

        raise ValueError(f"Could not export report: {error}") from error

    return path
