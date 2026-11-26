"""Effect size calculation utilities for BioDataLab."""

from __future__ import annotations

import numpy as np


def cohens_d(
    group1,
    group2,
) -> float:
    """Calculate Cohen's d using pooled standard deviation."""

    x = _validate_data(group1, "group1")
    y = _validate_data(group2, "group2")

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each group must contain at least 2 observations.")

    variance_x = np.var(x, ddof=1)
    variance_y = np.var(y, ddof=1)

    pooled_variance = (
        (len(x) - 1) * variance_x
        + (len(y) - 1) * variance_y
    ) / (len(x) + len(y) - 2)

    if pooled_variance == 0:
        raise ValueError("Pooled standard deviation must be non-zero.")

    pooled_std = np.sqrt(pooled_variance)

    return float((np.mean(x) - np.mean(y)) / pooled_std)


def hedges_g(
    group1,
    group2,
) -> float:
    """Calculate Hedges' g with small-sample correction."""

    x = _validate_data(group1, "group1")
    y = _validate_data(group2, "group2")

    d = cohens_d(x, y)

    degrees_of_freedom = len(x) + len(y) - 2

    correction = 1 - (3 / (4 * degrees_of_freedom - 1))

    return float(d * correction)


def glass_delta(
    group1,
    group2,
    reference: str = "group2",
) -> float:
    """Calculate Glass' delta using a reference group's standard deviation."""

    x = _validate_data(group1, "group1")
    y = _validate_data(group2, "group2")

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each group must contain at least 2 observations.")

    if reference not in {"group1", "group2"}:
        raise ValueError("reference must be 'group1' or 'group2'.")

    reference_group = x if reference == "group1" else y

    reference_std = np.std(reference_group, ddof=1)

    if reference_std == 0:
        raise ValueError("Reference standard deviation must be non-zero.")

    return float((np.mean(x) - np.mean(y)) / reference_std)


def _validate_data(data, name: str) -> np.ndarray:
    """Validate and clean a numeric group."""

    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    values = values[~np.isnan(values)]

    if not np.isfinite(values).all():
        raise ValueError(f"{name} must not contain infinite values.")

    if len(values) < 2:
        raise ValueError(f"{name} must contain at least 2 observations.")

    return values
