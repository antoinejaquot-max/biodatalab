"""Permutation test utilities for BioDataLab."""

from __future__ import annotations

import numpy as np


def permutation_test(
    group1,
    group2,
    statistic=None,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    random_state: int | None = None,
) -> dict[str, float]:
    """Perform a two-sample permutation test."""

    if statistic is None:
        statistic = lambda x, y: np.mean(x) - np.mean(y)

    if not callable(statistic):
        raise TypeError("statistic must be callable.")

    if not isinstance(n_permutations, int) or isinstance(
        n_permutations, bool
    ):
        raise TypeError("n_permutations must be an integer.")

    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'."
        )

    x = _validate_group(group1, "group1")
    y = _validate_group(group2, "group2")

    observed = float(statistic(x, y))

    if not np.isfinite(observed):
        raise ValueError("Observed statistic must be finite.")

    combined = np.concatenate([x, y])
    n_x = len(x)

    rng = np.random.default_rng(random_state)
    permutation_statistics = np.empty(n_permutations)

    for i in range(n_permutations):
        shuffled = rng.permutation(combined)

        perm_x = shuffled[:n_x]
        perm_y = shuffled[n_x:]

        value = float(statistic(perm_x, perm_y))

        if not np.isfinite(value):
            raise ValueError("Statistic returned a non-finite value.")

        permutation_statistics[i] = value

    if alternative == "two-sided":
        extreme = np.sum(
            np.abs(permutation_statistics) >= abs(observed)
        )
    elif alternative == "greater":
        extreme = np.sum(permutation_statistics >= observed)
    else:
        extreme = np.sum(permutation_statistics <= observed)

    p_value = (extreme + 1) / (n_permutations + 1)

    return {
        "observed_statistic": observed,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "alternative": alternative,
    }


def _validate_group(data, name: str) -> np.ndarray:
    """Validate a group of numeric observations."""

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
