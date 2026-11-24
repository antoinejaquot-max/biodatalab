
"""Non-parametric statistical methods for biological datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _numeric_values(values, name: str) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    result = series.dropna().to_numpy(dtype=float)

    if len(result) == 0:
        raise ValueError(f"{name} contains no valid numeric values.")

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains infinite values.")

    return result


def mann_whitney(
    group1,
    group2,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> dict:
    """Perform a Mann-Whitney U test on independent samples."""

    _validate_alpha(alpha)

    x = _numeric_values(group1, "group1")
    y = _numeric_values(group2, "group2")

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("Invalid alternative hypothesis.")

    result = stats.mannwhitneyu(
        x,
        y,
        alternative=alternative,
        method="auto",
    )

    return {
        "test": "Mann-Whitney U",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "alpha": alpha,
        "sample_sizes": [len(x), len(y)],
        "medians": [float(np.median(x)), float(np.median(y))],
        "alternative": alternative,
    }


def wilcoxon_signed_rank(
    before,
    after,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> dict:
    """Perform a Wilcoxon signed-rank test on paired samples."""

    _validate_alpha(alpha)

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("Invalid alternative hypothesis.")

    x_series = pd.to_numeric(pd.Series(before), errors="coerce")
    y_series = pd.to_numeric(pd.Series(after), errors="coerce")

    if len(x_series) != len(y_series):
        raise ValueError("Paired samples must have equal lengths.")

    paired = pd.DataFrame({
        "before": x_series,
        "after": y_series,
    }).dropna()

    x = paired["before"].to_numpy(dtype=float)
    y = paired["after"].to_numpy(dtype=float)

    if len(x) == 0:
        raise ValueError("No valid paired observations.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Samples must contain finite values.")

    differences = y - x

    if np.all(differences == 0):
        return {
            "test": "Wilcoxon signed-rank",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "alpha": alpha,
            "sample_size": len(x),
            "median_difference": 0.0,
            "alternative": alternative,
        }

    result = stats.wilcoxon(
        x,
        y,
        alternative=alternative,
        method="auto",
    )

    return {
        "test": "Wilcoxon signed-rank",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "alpha": alpha,
        "sample_size": len(x),
        "median_difference": float(np.median(differences)),
        "alternative": alternative,
    }


def kruskal_wallis(
    groups: dict,
    alpha: float = 0.05,
) -> dict:
    """Perform a Kruskal-Wallis test across independent groups."""

    _validate_alpha(alpha)

    if not isinstance(groups, dict) or len(groups) < 2:
        raise ValueError("At least two independent groups are required.")

    samples = {
        str(name): _numeric_values(values, str(name))
        for name, values in groups.items()
    }

    result = stats.kruskal(*samples.values())

    return {
        "test": "Kruskal-Wallis",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "alpha": alpha,
        "groups": {
            name: {
                "n": len(values),
                "median": float(np.median(values)),
            }
            for name, values in samples.items()
        },
    }


def spearman_correlation(
    x,
    y,
    alpha: float = 0.05,
) -> dict:
    """Calculate Spearman's rank correlation."""

    _validate_alpha(alpha)

    x_values = pd.to_numeric(pd.Series(x), errors="coerce")
    y_values = pd.to_numeric(pd.Series(y), errors="coerce")

    if len(x_values) != len(y_values):
        raise ValueError("Variables must have equal lengths.")

    paired = pd.DataFrame({
        "x": x_values,
        "y": y_values,
    }).dropna()

    x_clean = paired["x"].to_numpy(dtype=float)
    y_clean = paired["y"].to_numpy(dtype=float)

    if len(x_clean) < 3:
        raise ValueError("At least three valid pairs are required.")

    if not np.all(np.isfinite(x_clean)) or not np.all(np.isfinite(y_clean)):
        raise ValueError("Variables must contain finite values.")

    if np.all(x_clean == x_clean[0]) or np.all(y_clean == y_clean[0]):
        raise ValueError("Correlation is undefined for constant variables.")

    result = stats.spearmanr(x_clean, y_clean)

    return {
        "test": "Spearman correlation",
        "correlation": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "alpha": alpha,
        "sample_size": len(x_clean),
    }


def holm_correction(
    p_values,
    alpha: float = 0.05,
) -> dict:
    """Apply Holm-Bonferroni correction to multiple p-values."""

    _validate_alpha(alpha)

    values = np.asarray(p_values, dtype=float)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "p_values must be a non-empty one-dimensional sequence."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("p_values must be finite.")

    if np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be between 0 and 1.")

    n = len(values)
    order = np.argsort(values)
    adjusted = np.empty(n, dtype=float)

    running_max = 0.0

    for rank, index in enumerate(order):
        corrected = (n - rank) * values[index]
        running_max = max(running_max, corrected)
        adjusted[index] = min(running_max, 1.0)

    return {
        "method": "Holm-Bonferroni",
        "alpha": alpha,
        "raw_p_values": values.tolist(),
        "adjusted_p_values": adjusted.tolist(),
        "significant": (adjusted < alpha).tolist(),
    }
