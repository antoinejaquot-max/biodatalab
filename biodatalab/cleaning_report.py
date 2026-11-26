
"""Cleaning audit and reporting utilities for BioDataLab."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_dataframe(dataframe: pd.DataFrame, name: str) -> None:
    """Validate a dataframe used for cleaning comparisons."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate column names."
        )


def _safe_percentage(count: int, total: int) -> float:
    """Calculate a percentage safely."""
    if total == 0:
        return 0.0
    return float(count / total * 100)


def _missing_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Summarize missing values globally and by column."""
    missing_by_column = dataframe.isna().sum()

    return {
        "total": int(missing_by_column.sum()),
        "percentage": _safe_percentage(
            int(missing_by_column.sum()),
            int(dataframe.size),
        ),
        "by_column": {
            str(column): {
                "count": int(missing_by_column[column]),
                "percentage": _safe_percentage(
                    int(missing_by_column[column]),
                    len(dataframe),
                ),
            }
            for column in dataframe.columns
        },
    }


def _count_changed_cells(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> dict[str, Any]:
    """Count changed cells in comparable rows and columns."""

    common_columns = [
        column
        for column in original.columns
        if column in cleaned.columns
    ]

    if not common_columns:
        return {
            "count": None,
            "comparable_cells": 0,
            "comparison_method": "unavailable",
        }

    original_common = original[common_columns]
    cleaned_common = cleaned[common_columns]

    # Align by index if both indices are unique.
    if (
        original_common.index.is_unique
        and cleaned_common.index.is_unique
    ):
        common_index = original_common.index.intersection(
            cleaned_common.index
        )

        before = original_common.loc[common_index]
        after = cleaned_common.loc[common_index]

        method = "aligned_index"

    else:
        # Positional comparison is used when duplicate indices
        # prevent reliable label-based alignment.
        shared_rows = min(
            len(original_common),
            len(cleaned_common),
        )

        before = original_common.iloc[:shared_rows]
        after = cleaned_common.iloc[:shared_rows]

        method = "position"

    if before.shape != after.shape:
        return {
            "count": None,
            "comparable_cells": 0,
            "comparison_method": "unavailable",
        }

    comparable_cells = int(before.size)

    if comparable_cells == 0:
        return {
            "count": 0,
            "comparable_cells": 0,
            "comparison_method": method,
        }

    changed = 0

    for column in common_columns:
        left = before[column]
        right = after[column]

        equal = left.eq(right) | (
            left.isna() & right.isna()
        )

        changed += int((~equal).sum())

    return {
        "count": changed,
        "comparable_cells": comparable_cells,
        "comparison_method": method,
    }


def generate_cleaning_report(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a structured audit report comparing two datasets."""

    _validate_dataframe(original, "original")
    _validate_dataframe(cleaned, "cleaned")

    original_columns = list(original.columns)
    cleaned_columns = list(cleaned.columns)

    removed_columns = [
        column
        for column in original_columns
        if column not in cleaned_columns
    ]

    added_columns = [
        column
        for column in cleaned_columns
        if column not in original_columns
    ]

    if original.index.is_unique and cleaned.index.is_unique:
        removed_indices = original.index.difference(
            cleaned.index
        )
        added_indices = cleaned.index.difference(
            original.index
        )

        row_comparison_method = "aligned_index"

        removed_rows = len(removed_indices)
        added_rows = len(added_indices)

    else:
        removed_rows = max(
            0,
            len(original) - len(cleaned),
        )
        added_rows = max(
            0,
            len(cleaned) - len(original),
        )

        row_comparison_method = "row_count"

    original_missing = _missing_summary(original)
    cleaned_missing = _missing_summary(cleaned)

    changed_cells = _count_changed_cells(
        original,
        cleaned,
    )

    original_duplicates = int(
        original.duplicated().sum()
    )

    cleaned_duplicates = int(
        cleaned.duplicated().sum()
    )

    return {
        "summary": {
            "original_rows": int(len(original)),
            "cleaned_rows": int(len(cleaned)),
            "original_columns": int(original.shape[1]),
            "cleaned_columns": int(cleaned.shape[1]),
            "original_cells": int(original.size),
            "cleaned_cells": int(cleaned.size),
        },
        "changes": {
            "rows_removed": int(removed_rows),
            "rows_added": int(added_rows),
            "columns_removed": removed_columns,
            "columns_added": added_columns,
            "cells_changed": changed_cells["count"],
            "comparable_cells": (
                changed_cells["comparable_cells"]
            ),
            "cell_comparison_method": (
                changed_cells["comparison_method"]
            ),
            "row_comparison_method": (
                row_comparison_method
            ),
        },
        "missing_values": {
            "before": original_missing,
            "after": cleaned_missing,
            "reduction": (
                original_missing["total"]
                - cleaned_missing["total"]
            ),
        },
        "duplicates": {
            "before": original_duplicates,
            "after": cleaned_duplicates,
            "difference": (
                original_duplicates
                - cleaned_duplicates
            ),
        },
        "column_names": {
            "before": [str(c) for c in original_columns],
            "after": [str(c) for c in cleaned_columns],
        },
    }


def cleaning_report_dataframe(
    report: dict[str, Any],
) -> pd.DataFrame:
    """Convert the report summary into a compact dataframe."""

    summary = report["summary"]
    changes = report["changes"]
    missing = report["missing_values"]
    duplicates = report["duplicates"]

    return pd.DataFrame(
        [
            {
                "Metric": "Rows",
                "Before": summary["original_rows"],
                "After": summary["cleaned_rows"],
            },
            {
                "Metric": "Columns",
                "Before": summary["original_columns"],
                "After": summary["cleaned_columns"],
            },
            {
                "Metric": "Total cells",
                "Before": summary["original_cells"],
                "After": summary["cleaned_cells"],
            },
            {
                "Metric": "Missing values",
                "Before": missing["before"]["total"],
                "After": missing["after"]["total"],
            },
            {
                "Metric": "Duplicate rows",
                "Before": duplicates["before"],
                "After": duplicates["after"],
            },
            {
                "Metric": "Rows removed",
                "Before": 0,
                "After": changes["rows_removed"],
            },
            {
                "Metric": "Columns removed",
                "Before": 0,
                "After": len(changes["columns_removed"]),
            },
            {
                "Metric": "Changed cells",
                "Before": 0,
                "After": (
                    changes["cells_changed"]
                    if changes["cells_changed"] is not None
                    else "Unavailable"
                ),
            },
        ]
    )
