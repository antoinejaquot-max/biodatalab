
"""ANOVA methods and post-hoc comparisons for biological datasets."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _prepare_groups(groups: dict) -> dict:
    if not isinstance(groups, dict) or len(groups) < 2:
        raise ValueError("At least two independent groups are required.")

    prepared = {}

    for name, values in groups.items():
        series = pd.to_numeric(pd.Series(values), errors="coerce")
        sample = series.dropna().to_numpy(dtype=float)

        if len(sample) < 2:
            raise ValueError(
                f"Group '{name}' must contain at least two valid observations."
            )

        if not np.all(np.isfinite(sample)):
            raise ValueError(f"Group '{name}' contains infinite values.")

        prepared[str(name)] = sample

    if len(prepared) < 2:
        raise ValueError("At least two independent groups are required.")

    return prepared


def _eta_squared(samples: list[np.ndarray]) -> float:
    all_values = np.concatenate(samples)
    grand_mean = np.mean(all_values)

    between = sum(
        len(sample) * (np.mean(sample) - grand_mean) ** 2
        for sample in samples
    )

    total = np.sum((all_values - grand_mean) ** 2)

    return float(between / total) if total > 0 else 0.0


def _check_total_variation(samples: list[np.ndarray]) -> None:
    all_values = np.concatenate(samples)

    if np.all(all_values == all_values[0]):
        raise ValueError(
            "ANOVA is undefined because all observations have identical values."
        )


def _welch_anova(samples: list[np.ndarray]) -> tuple[float, float]:
    """Perform Welch's one-way ANOVA."""

    k = len(samples)

    means = np.array([np.mean(x) for x in samples])
    variances = np.array([np.var(x, ddof=1) for x in samples])
    sizes = np.array([len(x) for x in samples])

    if np.any(variances == 0):
        raise ValueError(
            "Welch ANOVA requires non-zero variance in every group."
        )

    weights = sizes / variances
    weighted_mean = np.sum(weights * means) / np.sum(weights)

    numerator = (
        np.sum(weights * (means - weighted_mean) ** 2)
        / (k - 1)
    )

    term = np.sum(
        (1 - weights / np.sum(weights)) ** 2 / (sizes - 1)
    )

    if term <= 0 or not np.isfinite(term):
        raise ValueError(
            "Welch ANOVA could not calculate valid degrees of freedom."
        )

    correction = (
        1
        + (2 * (k - 2) / (k**2 - 1)) * term
    )

    statistic = numerator / correction
    denominator = (k**2 - 1) / (3 * term)

    p_value = stats.f.sf(
        statistic,
        k - 1,
        denominator,
    )

    return float(statistic), float(p_value)


def one_way_anova(
    groups: dict,
    alpha: float = 0.05,
    equal_variances: bool = True,
) -> dict:
    """Perform classical or Welch one-way ANOVA.

    Groups must be independent.
    """

    _validate_alpha(alpha)

    prepared = _prepare_groups(groups)
    samples = list(prepared.values())

    _check_total_variation(samples)

    if equal_variances:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = stats.f_oneway(*samples)

        test_name = "One-way ANOVA"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    else:
        statistic, p_value = _welch_anova(samples)
        test_name = "Welch ANOVA"

    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError(
            "ANOVA produced an undefined result. "
            "Check the group variances and sample sizes."
        )

    levene = stats.levene(*samples, center="median")

    residuals = np.concatenate([
        sample - np.mean(sample)
        for sample in samples
    ])

    shapiro_result = (
        stats.shapiro(residuals)
        if 3 <= len(residuals) <= 5000
        and np.var(residuals) > 0
        else None
    )

    return {
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "effect_size_eta_squared": _eta_squared(samples),
        "assumption_checks": {
            "levene": {
                "statistic": float(levene.statistic),
                "p_value": float(levene.pvalue),
                "equal_variances_not_rejected": bool(
                    levene.pvalue >= alpha
                ),
            },
            "shapiro_residuals": (
                {
                    "statistic": float(shapiro_result.statistic),
                    "p_value": float(shapiro_result.pvalue),
                    "normality_not_rejected": bool(
                        shapiro_result.pvalue >= alpha
                    ),
                }
                if shapiro_result is not None
                else None
            ),
        },
        "groups": {
            name: {
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for name, values in prepared.items()
        },
    }


def tukey_hsd(
    groups: dict,
    alpha: float = 0.05,
) -> dict:
    """Perform Tukey HSD pairwise comparisons."""

    _validate_alpha(alpha)
    prepared = _prepare_groups(groups)

    samples = list(prepared.values())
    names = list(prepared.keys())

    _check_total_variation(samples)

    result = stats.tukey_hsd(*samples)
    confidence = result.confidence_interval(1 - alpha)

    comparisons = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            p_value = float(result.pvalue[i, j])
            difference = float(result.statistic[i, j])

            comparisons.append({
                "group1": names[i],
                "group2": names[j],
                "mean_difference": difference,
                "p_value": p_value,
                "significant": bool(p_value < alpha),
                "confidence_interval": [
                    float(confidence.low[i, j]),
                    float(confidence.high[i, j]),
                ],
            })

    return {
        "test": "Tukey HSD",
        "alpha": alpha,
        "comparisons": comparisons,
    }


def games_howell(
    groups: dict,
    alpha: float = 0.05,
) -> dict:
    """Perform Games-Howell pairwise comparisons."""

    _validate_alpha(alpha)
    prepared = _prepare_groups(groups)

    names = list(prepared.keys())
    comparisons = []
    k = len(prepared)

    for i in range(k):
        for j in range(i + 1, k):

            x = prepared[names[i]]
            y = prepared[names[j]]

            n1, n2 = len(x), len(y)

            mean_difference = float(
                np.mean(x) - np.mean(y)
            )

            v1 = np.var(x, ddof=1) / n1
            v2 = np.var(y, ddof=1) / n2

            standard_error = np.sqrt(v1 + v2)

            if standard_error == 0:
                raise ValueError(
                    "Games-Howell cannot compare groups "
                    "with zero combined variance."
                )

            denominator = (
                v1**2 / (n1 - 1)
                + v2**2 / (n2 - 1)
            )

            if denominator <= 0 or not np.isfinite(denominator):
                raise ValueError(
                    "Games-Howell could not calculate "
                    "valid degrees of freedom."
                )

            degrees_freedom = (
                (v1 + v2) ** 2 / denominator
            )

            # Studentized-range statistic:
            # q = sqrt(2) * |mean difference| / SE
            q_statistic = (
                abs(mean_difference)
                * np.sqrt(2)
                / standard_error
            )

            p_value = studentized_range.sf(
                q_statistic,
                k,
                degrees_freedom,
            )

            critical_value = studentized_range.ppf(
                1 - alpha,
                k,
                degrees_freedom,
            )

            margin = (
                critical_value
                * standard_error
                / np.sqrt(2)
            )

            comparisons.append({
                "group1": names[i],
                "group2": names[j],
                "mean_difference": mean_difference,
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
                "confidence_interval": [
                    float(mean_difference - margin),
                    float(mean_difference + margin),
                ],
                "degrees_of_freedom": float(degrees_freedom),
            })

    return {
        "test": "Games-Howell",
        "alpha": alpha,
        "comparisons": comparisons,
    }
