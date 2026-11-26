"""Statistical power analysis utilities for BioDataLab."""

from __future__ import annotations

from scipy.stats import nct, t


def calculate_power(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Calculate power for a one-sample t-test."""

    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise TypeError("sample_size must be an integer.")

    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")

    if not isinstance(effect_size, (int, float)):
        raise TypeError("effect_size must be numeric.")

    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'."
        )

    if not float("-inf") < effect_size < float("inf"):
        raise ValueError("effect_size must be finite.")

    df = sample_size - 1
    noncentrality = effect_size * (sample_size**0.5)

    if alternative == "two-sided":
        critical = t.ppf(1 - alpha / 2, df)

        power = (
            1 - nct.cdf(critical, df, noncentrality)
            + nct.cdf(-critical, df, noncentrality)
        )

    elif alternative == "greater":
        critical = t.ppf(1 - alpha, df)
        power = 1 - nct.cdf(critical, df, noncentrality)

    else:
        critical = t.ppf(alpha, df)
        power = nct.cdf(critical, df, noncentrality)

    return float(power)


def estimate_sample_size(
    effect_size: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    max_sample_size: int = 10000,
) -> int:
    """Estimate sample size required to reach a target power."""

    if not isinstance(effect_size, (int, float)):
        raise TypeError("effect_size must be numeric.")

    if effect_size == 0:
        raise ValueError("effect_size must not be zero.")

    if not float("-inf") < effect_size < float("inf"):
        raise ValueError("effect_size must be finite.")

    if not isinstance(target_power, (int, float)):
        raise TypeError("target_power must be numeric.")

    if not 0 < target_power < 1:
        raise ValueError("target_power must be between 0 and 1.")

    if not isinstance(max_sample_size, int) or isinstance(
        max_sample_size, bool
    ):
        raise TypeError("max_sample_size must be an integer.")

    if max_sample_size < 2:
        raise ValueError("max_sample_size must be at least 2.")

    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'."
        )

    for n in range(2, max_sample_size + 1):
        power = calculate_power(
            effect_size=effect_size,
            sample_size=n,
            alpha=alpha,
            alternative=alternative,
        )

        if power >= target_power:
            return n

    raise ValueError(
        "Required sample size exceeds max_sample_size."
    )
