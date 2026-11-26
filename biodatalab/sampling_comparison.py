
"""Sampling comparison utilities for BioDataLab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare_samples(
    reference: pd.DataFrame,
    samples: dict[str, pd.DataFrame],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare sample statistics against a reference dataset."""

    if not isinstance(reference, pd.DataFrame):
        raise TypeError("Reference must be a pandas DataFrame.")

    if reference.empty:
        raise ValueError("Reference DataFrame must not be empty.")

    if not isinstance(samples, dict) or not samples:
        raise ValueError("Samples must be a non-empty dictionary.")

    if columns is None:
        columns = list(
            reference.select_dtypes(include="number").columns
        )

    if not columns:
        raise ValueError("At least one numeric column is required.")

    for column in columns:
        if column not in reference.columns:
            raise ValueError(f"Column '{column}' not found in reference.")

        if not pd.api.types.is_numeric_dtype(reference[column]):
            raise TypeError(f"Column '{column}' must be numeric.")

    reference_stats = {}

    for column in columns:
        values = reference[column].dropna()

        if values.empty:
            raise ValueError(
                f"Reference column '{column}' contains no valid values."
            )

        reference_stats[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }

    results = []

    for sample_name, sample in samples.items():
        if not isinstance(sample, pd.DataFrame):
            raise TypeError(
                f"Sample '{sample_name}' must be a DataFrame."
            )

        if sample.empty:
            raise ValueError(f"Sample '{sample_name}' must not be empty.")

        for column in columns:
            if column not in sample.columns:
                raise ValueError(
                    f"Column '{column}' missing from sample '{sample_name}'."
                )

            if not pd.api.types.is_numeric_dtype(sample[column]):
                raise TypeError(
                    f"Column '{column}' in sample '{sample_name}' "
                    "must be numeric."
                )

            values = sample[column].dropna()

            if values.empty:
                raise ValueError(
                    f"Sample '{sample_name}' column '{column}' "
                    "contains no valid values."
                )

            mean = float(values.mean())
            std = float(values.std(ddof=1))

            ref_mean = reference_stats[column]["mean"]
            ref_std = reference_stats[column]["std"]

            pooled_std = np.sqrt(
                (ref_std**2 + std**2) / 2
            )

            if pooled_std == 0:
                standardized_difference = (
                    0.0 if mean == ref_mean else np.nan
                )
            else:
                standardized_difference = (
                    mean - ref_mean
                ) / pooled_std

            results.append(
                {
                    "sample": sample_name,
                    "variable": column,
                    "reference_mean": ref_mean,
                    "sample_mean": mean,
                    "mean_difference": mean - ref_mean,
                    "reference_std": ref_std,
                    "sample_std": std,
                    "standardized_mean_difference": (
                        standardized_difference
                    ),
                    "sample_size": len(values),
                    "missing_fraction": (
                        sample[column].isna().mean()
                    ),
                }
            )

    return pd.DataFrame(results).set_index(
        ["sample", "variable"]
    )
