
"""Dataset validation utilities for BioDataLab."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_dataframe(
    dataframe: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a dataframe against optional validation rules."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")

    rules = rules or {}
    column_rules = rules.get("columns", {})
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def add_issue(
        target: list[dict[str, Any]],
        code: str,
        message: str,
        column: str | None = None,
        rows: list[int] | None = None,
    ) -> None:
        issue: dict[str, Any] = {
            "code": code,
            "message": message,
        }

        if column is not None:
            issue["column"] = column

        if rows is not None:
            issue["rows"] = rows

        target.append(issue)

    # Required columns
    required_columns = rules.get("required_columns", [])

    for column in required_columns:
        if column not in dataframe.columns:
            add_issue(
                errors,
                "missing_column",
                f"Required column '{column}' is missing.",
                column=column,
            )

    # Column-level validation
    for column, config in column_rules.items():
        if column not in dataframe.columns:
            if column not in required_columns:
                add_issue(
                    errors,
                    "missing_column",
                    f"Validation column '{column}' is missing.",
                    column=column,
                )
            continue

        series = dataframe[column]

        # Nullability
        if config.get("nullable", True) is False:
            missing_rows = [
                i
                for i, value in enumerate(series)
                if pd.isna(value)
            ]

            if missing_rows:
                add_issue(
                    errors,
                    "null_values",
                    f"Column '{column}' contains missing values.",
                    column=column,
                    rows=missing_rows,
                )

        # Type checks
        expected_type = config.get("type")

        if expected_type == "numeric":
            invalid_rows = [
                i
                for i, value in enumerate(series)
                if not pd.isna(value)
                and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                )
            ]

        elif expected_type == "integer":
            invalid_rows = [
                i
                for i, value in enumerate(series)
                if not pd.isna(value)
                and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not float(value).is_integer()
                )
            ]

        elif expected_type == "string":
            invalid_rows = [
                i
                for i, value in enumerate(series)
                if not pd.isna(value)
                and not isinstance(value, str)
            ]

        elif expected_type == "boolean":
            invalid_rows = [
                i
                for i, value in enumerate(series)
                if not pd.isna(value)
                and not isinstance(value, bool)
            ]

        else:
            invalid_rows = []

        if expected_type in (
            "numeric",
            "integer",
            "string",
            "boolean",
        ) and invalid_rows:
            add_issue(
                errors,
                "invalid_type",
                f"Column '{column}' must contain {expected_type} values.",
                column=column,
                rows=invalid_rows,
            )

        # Numeric ranges
        if "min" in config or "max" in config:
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            non_numeric = series.notna() & numeric.isna()

            if non_numeric.any():
                add_issue(
                    warnings,
                    "range_check_skipped",
                    f"Range validation for '{column}' skipped because non-numeric values are present.",
                    column=column,
                )
            else:
                below_min = (
                    numeric < config["min"]
                    if "min" in config
                    else pd.Series(False, index=series.index)
                )

                above_max = (
                    numeric > config["max"]
                    if "max" in config
                    else pd.Series(False, index=series.index)
                )

                out_of_range = below_min | above_max

                rows = [
                    i
                    for i, value in enumerate(out_of_range)
                    if value
                ]

                if rows:
                    add_issue(
                        errors,
                        "out_of_range",
                        f"Column '{column}' contains values outside the allowed range.",
                        column=column,
                        rows=rows,
                    )

        # Allowed values
        if "allowed" in config:
            allowed = config["allowed"]

            invalid_rows = [
                i
                for i, value in enumerate(series)
                if not pd.isna(value)
                and value not in allowed
            ]

            if invalid_rows:
                add_issue(
                    errors,
                    "invalid_value",
                    f"Column '{column}' contains values outside the allowed set.",
                    column=column,
                    rows=invalid_rows,
                )

        # Uniqueness
        if config.get("unique", False):
            duplicated = series.duplicated(keep=False)

            rows = [
                i
                for i, value in enumerate(duplicated)
                if value
            ]

            if rows:
                add_issue(
                    errors,
                    "duplicate_values",
                    f"Column '{column}' contains duplicate values.",
                    column=column,
                    rows=rows,
                )

    # Duplicate rows
    if rules.get("unique_rows", False):
        duplicated_rows = dataframe.duplicated(keep=False)

        rows = [
            i
            for i, value in enumerate(duplicated_rows)
            if value
        ]

        if rows:
            add_issue(
                errors,
                "duplicate_rows",
                "Dataset contains duplicate rows.",
                rows=rows,
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "total_rows": int(len(dataframe)),
            "total_columns": int(dataframe.shape[1]),
            "errors": int(len(errors)),
            "warnings": int(len(warnings)),
        },
    }
