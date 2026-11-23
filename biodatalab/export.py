
"""Export utilities for BioDataLab."""

import json
from pathlib import Path


def export_report_json(report: dict, filepath: str | Path) -> Path:
    """Export an analysis report to a JSON file."""

    if not isinstance(report, dict):
        raise TypeError("Report must be a dictionary.")

    path = Path(filepath)

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Could not export report: {error}") from error

    return path
