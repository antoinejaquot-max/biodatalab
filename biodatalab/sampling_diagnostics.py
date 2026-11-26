
"""Sampling diagnostic utilities for BioDataLab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sampling_diagnostics(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate sample representativeness against a population."""

    if not isinstance(population, pd.DataFrame):
        raise TypeError("Population must be a pandas DataFrame.")

    if not isinstance(sample, pd.DataFrame):
        raise TypeError("Sample must be a pandas DataFrame.")

    if population.empty or sample.empty:
        raise ValueError("Population and sample must not be empty.")

    if columns is None:
        columns = list(
            population.select_dtypes(include="number").columns
        )

    if not columns:
        raise ValueError("At least one numeric column is required.")

    records = []

    for column in columns:
        if column not in population.columns:
            raise ValueError(
                f"Column '{column}' not found in population."
            )

        if column not in sample.columns:
            raise ValueError(
                f"Column '{column}' not found in sample."
            )

        if not pd.api.types.is_numeric_dtype(population[column]):
            raise TypeError(
                f"Population column '{column}' must be numeric."
            )

        if not pd.api.types.is_numeric_dtype(sample[column]):
            raise TypeError(
                f"Sample column '{column}' must be numeric."
            )

        pop_values = population[column].dropna().to_numpy(dtype=float)
        sample_values = sample[column].dropna().to_numpy(dtype=float)

        if len(pop_values) == 0 or len(sample_values) == 0:
            raise ValueError(
                f"Column '{column}' has no valid observations."
            )

        if not np.isfinite(pop_values).all():
            raise ValueError(
                f"Population column '{column}' contains infinite values."
            )

        if not np.isfinite(sample_values).all():
            raise ValueError(
                f"Sample column '{column}' contains infinite values."
            )

        pop_mean = float(np.mean(pop_values))
        sample_mean = float(np.mean(sample_values))

        pop_std = float(np.std(pop_values, ddof=1)) if len(pop_values) > 1 else 0.0
        sample_std = float(np.std(sample_values, ddof=1)) if len(sample_values) > 1 else 0.0

        mean_difference = sample_mean - pop_mean

        if pop_std == 0:
            standardized_mean_difference = (
                0.0 if mean_difference == 0 else np.nan
            )
        else:
            standardized_mean_difference = mean_difference / pop_std

        variance_ratio = (
            sample_std**2 / pop_std**2
            if pop_std > 0
            else np.nan
        )

        pop_quantiles = np.quantile(
            pop_values, [0.25, 0.5, 0.75]
        )

        sample_quantiles = np.quantile(
            sample_values, [0.25, 0.5, 0.75]
        )

        records.append(
            {
                "variable": column,
                "population_size": len(pop_values),
                "sample_size": len(sample_values),
                "population_mean": pop_mean,
                "sample_mean": sample_mean,
                "mean_difference": mean_difference,
                "standardized_mean_difference": (
                    standardized_mean_difference
                ),
                "population_std": pop_std,
                "sample_std": sample_std,
                "variance_ratio": variance_ratio,
                "population_q25": pop_quantiles[0],
                "sample_q25": sample_quantiles[0],
                "q25_difference": (
                    sample_quantiles[0] - pop_quantiles[0]
                ),
                "population_median": pop_quantiles[1],
                "sample_median": sample_quantiles[1],
                "median_difference": (
                    sample_quantiles[1] - pop_quantiles[1]
                ),
                "population_q75": pop_quantiles[2],
                "sample_q75": sample_quantiles[2],
                "q75_difference": (
                    sample_quantiles[2] - pop_quantiles[2]
                ),
                "population_missing_fraction": (
                    population[column].isna().mean()
                ),
                "sample_missing_fraction": (
                    sample[column].isna().mean()
                ),
            }
        )

    return pd.DataFrame(records).set_index("variable")
