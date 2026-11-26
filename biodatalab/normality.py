"""Normality testing utilities for BioDataLab."""

from __future__ import annotations

import numpy as np
from scipy import stats


def test_normality(
    data,
    method: str = "shapiro",
    alpha: float = 0.05,
) -> dict[str, float | str | bool]:
    """Test whether data are consistent with a normal distribution."""

    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    methods = {
        "shapiro": stats.shapiro,
        "dagostino": stats.normaltest,
        "jarque_bera": stats.jarque_bera,
    }

    if method not in methods:
        raise ValueError(
            "method must be 'shapiro', 'dagostino', or 'jarque_bera'."
        )

    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Data must contain numeric values.") from exc

    if values.ndim != 1:
        raise ValueError("Data must be one-dimensional.")

    values = values[~np.isnan(values)]

    if not np.isfinite(values).all():
        raise ValueError("Data must not contain infinite values.")

    if len(values) < 3:
        raise ValueError("At least 3 observations are required.")

    if np.ptp(values) == 0:
        raise ValueError("Data must not be constant.")

    if method == "dagostino" and len(values) < 8:
        raise ValueError(
            "D'Agostino's test requires at least 8 observations."
        )

    result = methods[method](values)

    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    return {
        "method": method,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": float(alpha),
        "is_normal": bool(p_value >= alpha),
        "sample_size": int(len(values)),
    }
