
"""Tests for ANOVA and post-hoc statistical methods."""

import numpy as np
import pytest
from scipy.stats import studentized_range

from biodatalab.anova import (
    one_way_anova,
    tukey_hsd,
    games_howell,
)


@pytest.fixture
def experimental_groups():
    return {
        "Control": [10, 11, 9, 10, 12, 11],
        "Treatment_A": [15, 16, 14, 15, 17, 16],
        "Treatment_B": [20, 21, 19, 20, 22, 21],
    }


# --------------------------------------------------
# Classical one-way ANOVA
# --------------------------------------------------

def test_one_way_anova_detects_differences(experimental_groups):
    result = one_way_anova(experimental_groups)

    assert result["test"] == "One-way ANOVA"
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_one_way_anova_returns_effect_size(experimental_groups):
    result = one_way_anova(experimental_groups)

    assert 0 <= result["effect_size_eta_squared"] <= 1


def test_one_way_anova_returns_group_statistics(experimental_groups):
    result = one_way_anova(experimental_groups)

    assert result["groups"]["Control"]["n"] == 6
    assert result["groups"]["Treatment_A"]["mean"] == pytest.approx(15.5)


def test_one_way_anova_returns_assumption_checks(experimental_groups):
    result = one_way_anova(experimental_groups)

    assert "levene" in result["assumption_checks"]
    assert "shapiro_residuals" in result["assumption_checks"]


def test_one_way_anova_invalid_alpha(experimental_groups):
    with pytest.raises(ValueError):
        one_way_anova(experimental_groups, alpha=1.5)


def test_one_way_anova_requires_two_groups():
    with pytest.raises(ValueError):
        one_way_anova({"Control": [1, 2, 3]})


# --------------------------------------------------
# Welch ANOVA
# --------------------------------------------------

def test_welch_anova_detects_differences(experimental_groups):
    result = one_way_anova(
        experimental_groups,
        equal_variances=False,
    )

    assert result["test"] == "Welch ANOVA"
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_welch_anova_returns_valid_p_value(experimental_groups):
    result = one_way_anova(
        experimental_groups,
        equal_variances=False,
    )

    assert 0 <= result["p_value"] <= 1


def test_welch_anova_rejects_zero_variance_group():
    groups = {
        "A": [1, 1, 1],
        "B": [2, 3, 4],
    }

    with pytest.raises(ValueError):
        one_way_anova(groups, equal_variances=False)


# --------------------------------------------------
# Tukey HSD
# --------------------------------------------------

def test_tukey_hsd_returns_all_comparisons(experimental_groups):
    result = tukey_hsd(experimental_groups)

    assert result["test"] == "Tukey HSD"
    assert len(result["comparisons"]) == 3


def test_tukey_hsd_detects_differences(experimental_groups):
    result = tukey_hsd(experimental_groups)

    for comparison in result["comparisons"]:
        assert comparison["p_value"] < 0.05
        assert comparison["significant"] is True


def test_tukey_hsd_confidence_intervals(experimental_groups):
    result = tukey_hsd(experimental_groups)

    for comparison in result["comparisons"]:
        lower, upper = comparison["confidence_interval"]

        assert lower < upper


def test_tukey_requires_two_groups():
    with pytest.raises(ValueError):
        tukey_hsd({"A": [1, 2, 3]})


# --------------------------------------------------
# Games-Howell
# --------------------------------------------------

def test_games_howell_returns_all_comparisons(experimental_groups):
    result = games_howell(experimental_groups)

    assert result["test"] == "Games-Howell"
    assert len(result["comparisons"]) == 3


def test_games_howell_detects_differences(experimental_groups):
    result = games_howell(experimental_groups)

    for comparison in result["comparisons"]:
        assert comparison["p_value"] < 0.05
        assert comparison["significant"] is True


def test_games_howell_returns_degrees_of_freedom(experimental_groups):
    result = games_howell(experimental_groups)

    for comparison in result["comparisons"]:
        assert comparison["degrees_of_freedom"] > 0


def test_games_howell_statistic_and_p_value_formula():
    groups = {
        "A": [1, 2, 4, 5, 7],
        "B": [4, 8, 9, 10, 12, 15, 17],
        "C": [3, 3, 6, 7, 8, 11],
    }

    alpha = 0.05
    result = games_howell(groups, alpha=alpha)

    comparison = result["comparisons"][0]

    x = np.asarray(groups["A"], dtype=float)
    y = np.asarray(groups["B"], dtype=float)

    n1, n2 = len(x), len(y)

    v1 = np.var(x, ddof=1) / n1
    v2 = np.var(y, ddof=1) / n2

    se = np.sqrt(v1 + v2)

    df = (
        (v1 + v2) ** 2
        / (
            v1**2 / (n1 - 1)
            + v2**2 / (n2 - 1)
        )
    )

    difference = np.mean(x) - np.mean(y)

    q = abs(difference) * np.sqrt(2) / se

    expected_p = studentized_range.sf(
        q,
        len(groups),
        df,
    )

    critical = studentized_range.ppf(
        1 - alpha,
        len(groups),
        df,
    )

    margin = critical * se / np.sqrt(2)

    assert comparison["mean_difference"] == pytest.approx(
        difference
    )

    assert comparison["p_value"] == pytest.approx(
        expected_p,
        rel=1e-10,
    )

    assert comparison["degrees_of_freedom"] == pytest.approx(
        df,
        rel=1e-10,
    )

    assert comparison["confidence_interval"] == pytest.approx(
        [
            difference - margin,
            difference + margin,
        ],
        rel=1e-10,
    )


def test_games_howell_rejects_zero_combined_variance():
    groups = {
        "A": [1, 1, 1],
        "B": [1, 1, 1],
    }

    with pytest.raises(ValueError):
        games_howell(groups)


def test_games_howell_requires_two_groups():
    with pytest.raises(ValueError):
        games_howell({"A": [1, 2, 3]})


# --------------------------------------------------
# Missing values and invalid inputs
# --------------------------------------------------

def test_anova_ignores_missing_values():
    groups = {
        "A": [1, 2, np.nan, 3],
        "B": [5, 6, 7, np.nan],
    }

    result = one_way_anova(groups)

    assert result["groups"]["A"]["n"] == 3
    assert result["groups"]["B"]["n"] == 3


def test_anova_rejects_groups_with_one_observation():
    with pytest.raises(ValueError):
        one_way_anova({
            "A": [1],
            "B": [2, 3],
        })


def test_anova_rejects_infinite_values():
    with pytest.raises(ValueError):
        one_way_anova({
            "A": [1, 2, np.inf],
            "B": [4, 5, 6],
        })


def test_anova_rejects_identical_values():
    with pytest.raises(ValueError):
        one_way_anova({
            "A": [5, 5, 5],
            "B": [5, 5, 5],
        })


def test_anova_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        one_way_anova(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            },
            alpha=0,
        )


def test_anova_rejects_nonfinite_alpha():
    with pytest.raises(ValueError):
        one_way_anova(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            },
            alpha=np.nan,
        )
