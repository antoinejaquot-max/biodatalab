
"""Bootstrap uncertainty estimation utilities for BioDataLab."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    data,
    statistic: Callable = np.mean,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, float]:
    """Estimate a confidence interval using bootstrap resampling."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    if not callable(statistic):
        raise TypeError("statistic must be callable.")

    values = np.asarray(data, dtype=float)
    values = values[~np.isnan(values)]

    if values.ndim != 1:
        raise ValueError("Data must be one-dimensional.")

    if len(values) == 0:
        raise ValueError("Data must contain valid observations.")

    rng = np.random.default_rng(random_state)

    estimate = float(statistic(values))

    bootstrap_statistics = np.empty(n_resamples)

    for i in range(n_resamples):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        bootstrap_statistics[i] = statistic(sample)

    alpha = 1 - confidence_level

    lower = float(np.quantile(bootstrap_statistics, alpha / 2))
    upper = float(np.quantile(bootstrap_statistics, 1 - alpha / 2))

    standard_error = float(np.std(bootstrap_statistics, ddof=1)) if n_resamples > 1 else 0.0

    return {
        "estimate": estimate,
        "lower_bound": lower,
        "upper_bound": upper,
        "standard_error": standard_error,
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
    }
