
"""Dataset validation utilities for BioDataLab."""

from __future__ import annotations

from typing import Any

import numpy as np
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

    if rules is None:
        rules = {}

    if not isinstance(rules, dict):
        raise TypeError("rules must be a dictionary.")

    column_rules = rules.get("columns", {})
    required_columns = rules.get("required_columns", [])

    if not isinstance(column_rules, dict):
        raise TypeError("rules['columns'] must be a dictionary.")

    if not isinstance(required_columns, (list, tuple)):
        raise TypeError("required_columns must be a list or tuple.")

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

        if not isinstance(config, dict):
            raise TypeError(
                f"Rules for column '{column}' must be a dictionary."
            )

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
                i for i, value in enumerate(series)
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

        # Type validation
        expected_type = config.get("type")

        supported_types = {
            "numeric",
            "integer",
            "string",
            "boolean",
        }

        if expected_type is not None and expected_type not in supported_types:
            raise ValueError(
                f"Unsupported type '{expected_type}' for column '{column}'. "
                f"Supported types are: {sorted(supported_types)}."
            )

        invalid_rows: list[int] = []

        if expected_type == "numeric":
            for i, value in enumerate(series):
                if pd.isna(value):
                    continue

                if (
                    not pd.api.types.is_number(value)
                    or isinstance(value, (bool, np.bool_))
                    or not np.isfinite(value)
                ):
                    invalid_rows.append(i)

        elif expected_type == "integer":
            for i, value in enumerate(series):
                if pd.isna(value):
                    continue

                if (
                    not pd.api.types.is_number(value)
                    or isinstance(value, (bool, np.bool_))
                    or not np.isfinite(value)
                    or not float(value).is_integer()
                ):
                    invalid_rows.append(i)

        elif expected_type == "string":
            invalid_rows = [
                i for i, value in enumerate(series)
                if not pd.isna(value)
                and not isinstance(value, str)
            ]

        elif expected_type == "boolean":
            invalid_rows = [
                i for i, value in enumerate(series)
                if not pd.isna(value)
                and not isinstance(value, (bool, np.bool_))
            ]

        if invalid_rows:
            add_issue(
                errors,
                "invalid_type",
                f"Column '{column}' must contain {expected_type} values.",
                column=column,
                rows=invalid_rows,
            )

        # Numeric range validation
        if "min" in config or "max" in config:

            minimum = config.get("min")
            maximum = config.get("max")

            for bound_name, bound in (
                ("min", minimum),
                ("max", maximum),
            ):
                if bound is not None:
                    if (
                        not pd.api.types.is_number(bound)
                        or isinstance(bound, (bool, np.bool_))
                        or not np.isfinite(bound)
                    ):
                        raise ValueError(
                            f"The '{bound_name}' bound for column "
                            f"'{column}' must be a finite number."
                        )

            if (
                minimum is not None
                and maximum is not None
                and minimum > maximum
            ):
                raise ValueError(
                    f"Minimum value exceeds maximum for column '{column}'."
                )

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            non_numeric = series.notna() & numeric.isna()

            non_finite = numeric.notna() & ~np.isfinite(numeric)

            invalid_numeric = non_numeric | non_finite

            if invalid_numeric.any():
                rows = [
                    i for i, value in enumerate(invalid_numeric)
                    if value
                ]

                add_issue(
                    warnings,
                    "range_check_skipped",
                    f"Range validation for '{column}' skipped "
                    "for non-numeric or non-finite values.",
                    column=column,
                    rows=rows,
                )

            valid_numeric = numeric.notna() & np.isfinite(numeric)

            below_min = (
                numeric < minimum
                if minimum is not None
                else pd.Series(False, index=series.index)
            )

            above_max = (
                numeric > maximum
                if maximum is not None
                else pd.Series(False, index=series.index)
            )

            out_of_range = valid_numeric & (below_min | above_max)

            rows = [
                i for i, value in enumerate(out_of_range)
                if value
            ]

            if rows:
                add_issue(
                    errors,
                    "out_of_range",
                    f"Column '{column}' contains values outside "
                    "the allowed range.",
                    column=column,
                    rows=rows,
                )

        # Allowed values
        if "allowed" in config:
            allowed = config["allowed"]

            if not isinstance(allowed, (list, tuple, set)):
                raise TypeError(
                    f"'allowed' for column '{column}' "
                    "must be a list, tuple or set."
                )

            invalid_rows = [
                i for i, value in enumerate(series)
                if not pd.isna(value)
                and value not in allowed
            ]

            if invalid_rows:
                add_issue(
                    errors,
                    "invalid_value",
                    f"Column '{column}' contains values "
                    "outside the allowed set.",
                    column=column,
                    rows=invalid_rows,
                )

        # Uniqueness
        if config.get("unique", False):
            duplicated = series.duplicated(keep=False)

            rows = [
                i for i, value in enumerate(duplicated)
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
            i for i, value in enumerate(duplicated_rows)
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
