
"""Bootstrap uncertainty estimation utilities for BioDataLab."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    data,
    statistic: Callable = np.mean,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, float]:
    """Estimate a confidence interval using bootstrap resampling."""

    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool):
        raise TypeError("n_resamples must be an integer.")

    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")

    if not isinstance(confidence_level, (int, float)):
        raise TypeError("confidence_level must be numeric.")

    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    if not callable(statistic):
        raise TypeError("statistic must be callable.")

    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Data must contain numeric values.") from exc

    if values.ndim != 1:
        raise ValueError("Data must be one-dimensional.")

    values = values[~np.isnan(values)]

    if len(values) == 0:
        raise ValueError("Data must contain valid observations.")

    if not np.isfinite(values).all():
        raise ValueError("Data must not contain infinite values.")

    try:
        estimate = float(statistic(values))
    except Exception as exc:
        raise ValueError("Statistic could not be calculated.") from exc

    if not np.isfinite(estimate):
        raise ValueError("Statistic must return a finite value.")

    rng = np.random.default_rng(random_state)
    bootstrap_statistics = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        try:
            result = float(statistic(sample))
        except Exception as exc:
            raise ValueError(
                "Statistic failed during bootstrap resampling."
            ) from exc

        if not np.isfinite(result):
            raise ValueError(
                "Statistic returned a non-finite result."
            )

        bootstrap_statistics[i] = result

    alpha = 1 - confidence_level

    lower = float(
        np.quantile(bootstrap_statistics, alpha / 2)
    )
    upper = float(
        np.quantile(bootstrap_statistics, 1 - alpha / 2)
    )

    standard_error = (
        float(np.std(bootstrap_statistics, ddof=1))
        if n_resamples > 1
        else 0.0
    )

    return {
        "estimate": estimate,
        "lower_bound": lower,
        "upper_bound": upper,
        "standard_error": standard_error,
        "confidence_level": float(confidence_level),
        "n_resamples": n_resamples,
    }
