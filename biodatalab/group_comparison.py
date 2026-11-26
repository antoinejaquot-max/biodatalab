
"""Statistical comparison of independent experimental groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _welch_anova(samples: list[np.ndarray]) -> tuple[float, float]:
    """Calculate Welch's one-way ANOVA statistic and p-value."""

    k = len(samples)
    means = np.array([np.mean(x) for x in samples], dtype=float)
    variances = np.array([np.var(x, ddof=1) for x in samples], dtype=float)
    sizes = np.array([len(x) for x in samples], dtype=float)

    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        raise ValueError(
            "Welch's ANOVA requires positive finite variances "
            "in every group."
        )

    weights = sizes / variances
    weight_sum = np.sum(weights)
    weighted_mean = np.sum(weights * means) / weight_sum

    numerator = (
        np.sum(weights * (means - weighted_mean) ** 2)
        / (k - 1)
    )

    correction_term = np.sum(
        (1 - weights / weight_sum) ** 2
        / (sizes - 1)
    )

    correction = (
        1
        + (2 * (k - 2) / (k**2 - 1))
        * correction_term
    )

    denominator_df = (
        (k**2 - 1) / (3 * correction_term)
        if correction_term > 0
        else np.inf
    )

    statistic = numerator / correction

    p_value = stats.f.sf(
        statistic,
        k - 1,
        denominator_df,
    )

    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError(
            "Welch's ANOVA produced an undefined result."
        )

    return float(statistic), float(p_value)


def compare_groups(
    dataframe: pd.DataFrame,
    value_column: str,
    group_column: str,
    confidence_level: float = 0.95,
) -> dict:
    """Compare a numerical variable across independent groups.

    Two groups:
        Welch's independent t-test.

    Three or more groups:
        Welch's one-way ANOVA.

    Returns descriptive summaries, group-level confidence intervals,
    and a global statistical test.

    This function assumes independent observations.
    """

    if value_column not in dataframe.columns:
        raise ValueError(f"Unknown value column: {value_column}")

    if group_column not in dataframe.columns:
        raise ValueError(f"Unknown group column: {group_column}")

    if value_column == group_column:
        raise ValueError("Value and group columns must be different.")

    if (
        not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be between 0 and 1.")

    data = dataframe[[value_column, group_column]].copy()

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    )

    data = data.dropna()

    data = data[
        np.isfinite(data[value_column])
    ]

    groups = {
        str(name): group[value_column].to_numpy(dtype=float)
        for name, group in data.groupby(group_column)
    }

    if len(groups) < 2:
        raise ValueError(
            "At least two non-empty groups are required."
        )

    for name, values in groups.items():

        if len(values) < 2:
            raise ValueError(
                f"Group '{name}' must contain at least "
                "two valid observations."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Group '{name}' contains non-finite values."
            )

        if np.var(values, ddof=1) <= 0:
            raise ValueError(
                f"Group '{name}' has zero variance. "
                "Welch's tests cannot be applied reliably."
            )

    alpha = 1 - confidence_level
    summary = {}

    for name, values in groups.items():

        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))

        standard_error = stats.sem(values)

        margin = (
            stats.t.ppf(1 - alpha / 2, n - 1)
            * standard_error
        )

        summary[name] = {
            "n": n,
            "mean": mean,
            "std": std,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "confidence_interval": [
                float(mean - margin),
                float(mean + margin),
            ],
        }

    samples = list(groups.values())

    if len(samples) == 2:

        test_name = "Welch's t-test"

        test = stats.ttest_ind(
            samples[0],
            samples[1],
            equal_var=False,
        )

        statistic = float(test.statistic)
        p_value = float(test.pvalue)

    else:

        test_name = "Welch's ANOVA"

        statistic, p_value = _welch_anova(samples)

    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError(
            "The statistical test produced an undefined result."
        )

    return {
        "value_column": value_column,
        "group_column": group_column,
        "confidence_level": confidence_level,
        "groups": summary,
        "test": {
            "name": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "significant_at_0_05": bool(p_value < 0.05),
        },
    }
