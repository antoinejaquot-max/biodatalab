
"""Statistical comparison of experimental groups."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def compare_groups(
    dataframe: pd.DataFrame,
    value_column: str,
    group_column: str,
    confidence_level: float = 0.95,
) -> dict:
    """Compare a numerical variable across two or more groups.

    Returns descriptive statistics, confidence intervals,
    and an appropriate global statistical test.
    """

    if value_column not in dataframe.columns:
        raise ValueError(f"Unknown value column: {value_column}")

    if group_column not in dataframe.columns:
        raise ValueError(f"Unknown group column: {group_column}")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    data = dataframe[[value_column, group_column]].copy()
    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    )
    data = data.dropna()

    groups = {
        str(name): group[value_column].to_numpy(dtype=float)
        for name, group in data.groupby(group_column)
    }

    groups = {
        name: values
        for name, values in groups.items()
        if len(values) > 0
    }

    if len(groups) < 2:
        raise ValueError("At least two non-empty groups are required.")

    alpha = 1 - confidence_level
    summary = {}

    for name, values in groups.items():
        n = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else None

        if n > 1:
            standard_error = stats.sem(values)
            margin = stats.t.ppf(1 - alpha / 2, n - 1) * standard_error
            ci = [float(mean - margin), float(mean + margin)]
        else:
            ci = None

        summary[name] = {
            "n": n,
            "mean": mean,
            "std": std,
            "median": float(pd.Series(values).median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "confidence_interval": ci,
        }

    samples = list(groups.values())

    if len(samples) == 2:
        test_name = "Welch's t-test"
        test = stats.ttest_ind(
            samples[0],
            samples[1],
            equal_var=False,
        )
    else:
        test_name = "Welch's ANOVA"
        test = stats.f_oneway(*samples)

    statistic = float(test.statistic)
    p_value = float(test.pvalue)

    return {
        "value_column": value_column,
        "group_column": group_column,
        "confidence_level": confidence_level,
        "groups": summary,
        "test": {
            "name": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "significant_at_0_05": p_value < 0.05,
        },
    }
