
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from biodatalab.group_comparison import compare_groups


@pytest.fixture
def experimental_data():
    return pd.DataFrame({
        "treatment": [
            "Control", "Control", "Control", "Control",
            "Fertilizer", "Fertilizer", "Fertilizer", "Fertilizer",
        ],
        "height": [
            10.0, 11.0, 9.0, 10.5,
            15.0, 16.0, 14.0, 15.5,
        ],
    })


def test_compare_two_groups(experimental_data):
    result = compare_groups(
        experimental_data,
        value_column="height",
        group_column="treatment",
    )

    assert set(result["groups"]) == {"Control", "Fertilizer"}
    assert result["groups"]["Control"]["n"] == 4
    assert result["groups"]["Fertilizer"]["n"] == 4


def test_group_means(experimental_data):
    result = compare_groups(
        experimental_data,
        value_column="height",
        group_column="treatment",
    )

    assert result["groups"]["Control"]["mean"] == pytest.approx(10.125)
    assert result["groups"]["Fertilizer"]["mean"] == pytest.approx(15.125)


def test_confidence_intervals(experimental_data):
    result = compare_groups(
        experimental_data,
        value_column="height",
        group_column="treatment",
    )

    for group in result["groups"].values():
        ci = group["confidence_interval"]
        assert ci[0] < group["mean"] < ci[1]


def test_two_group_test_is_welch(experimental_data):
    result = compare_groups(
        experimental_data,
        value_column="height",
        group_column="treatment",
    )

    assert result["test"]["name"] == "Welch's t-test"
    assert 0 <= result["test"]["p_value"] <= 1
    assert result["test"]["significant_at_0_05"] is True


def test_two_group_result_matches_scipy(experimental_data):
    result = compare_groups(
        experimental_data,
        value_column="height",
        group_column="treatment",
    )

    control = experimental_data.loc[
        experimental_data["treatment"] == "Control",
        "height",
    ].to_numpy()

    fertilizer = experimental_data.loc[
        experimental_data["treatment"] == "Fertilizer",
        "height",
    ].to_numpy()

    reference = stats.ttest_ind(
        control,
        fertilizer,
        equal_var=False,
    )

    assert result["test"]["statistic"] == pytest.approx(
        reference.statistic
    )

    assert result["test"]["p_value"] == pytest.approx(
        reference.pvalue
    )


def test_three_groups_are_really_welch_anova():

    data = pd.DataFrame({
        "group": (
            ["A"] * 5
            + ["B"] * 7
            + ["C"] * 6
        ),
        "value": (
            [1, 2, 3, 4, 5]
            + [5, 7, 8, 9, 10, 11, 12]
            + [2, 3, 4, 8, 10, 12]
        ),
    })

    result = compare_groups(
        data,
        value_column="value",
        group_column="group",
    )

    assert result["test"]["name"] == "Welch's ANOVA"
    assert 0 <= result["test"]["p_value"] <= 1
    assert np.isfinite(result["test"]["statistic"])


def test_three_group_welch_reference_calculation():

    data = pd.DataFrame({
        "group": (
            ["A"] * 5
            + ["B"] * 7
            + ["C"] * 6
        ),
        "value": (
            [1, 2, 3, 4, 5]
            + [5, 7, 8, 9, 10, 11, 12]
            + [2, 3, 4, 8, 10, 12]
        ),
    })

    result = compare_groups(
        data,
        value_column="value",
        group_column="group",
    )

    samples = [
        data.loc[data["group"] == name, "value"].to_numpy()
        for name in ["A", "B", "C"]
    ]

    k = len(samples)
    means = np.array([np.mean(x) for x in samples])
    variances = np.array([np.var(x, ddof=1) for x in samples])
    sizes = np.array([len(x) for x in samples])

    weights = sizes / variances
    weighted_mean = np.sum(weights * means) / np.sum(weights)

    numerator = (
        np.sum(weights * (means - weighted_mean) ** 2)
        / (k - 1)
    )

    term = np.sum(
        (1 - weights / np.sum(weights)) ** 2
        / (sizes - 1)
    )

    correction = (
        1 + (2 * (k - 2) / (k**2 - 1)) * term
    )

    expected_f = numerator / correction
    expected_df = (k**2 - 1) / (3 * term)

    expected_p = stats.f.sf(
        expected_f,
        k - 1,
        expected_df,
    )

    assert result["test"]["statistic"] == pytest.approx(
        expected_f,
        rel=1e-10,
    )

    assert result["test"]["p_value"] == pytest.approx(
        expected_p,
        rel=1e-10,
    )


def test_missing_column_raises_error(experimental_data):
    with pytest.raises(ValueError):
        compare_groups(
            experimental_data,
            value_column="unknown",
            group_column="treatment",
        )


def test_invalid_confidence_level(experimental_data):
    with pytest.raises(ValueError):
        compare_groups(
            experimental_data,
            value_column="height",
            group_column="treatment",
            confidence_level=1.5,
        )


def test_only_one_group_raises_error():
    dataframe = pd.DataFrame({
        "treatment": ["Control"] * 3,
        "height": [10.0, 11.0, 12.0],
    })

    with pytest.raises(ValueError):
        compare_groups(
            dataframe,
            value_column="height",
            group_column="treatment",
        )


def test_missing_values_are_ignored():
    dataframe = pd.DataFrame({
        "treatment": [
            "Control", "Control",
            "Fertilizer", "Fertilizer",
        ],
        "height": [10.0, np.nan, 15.0, 16.0],
    })

    with pytest.raises(ValueError):
        compare_groups(
            dataframe,
            value_column="height",
            group_column="treatment",
        )


def test_single_observation_group_rejected():
    dataframe = pd.DataFrame({
        "treatment": ["A", "A", "B"],
        "height": [1, 2, 3],
    })

    with pytest.raises(ValueError):
        compare_groups(
            dataframe,
            value_column="height",
            group_column="treatment",
        )


def test_infinite_values_are_excluded():
    dataframe = pd.DataFrame({
        "treatment": ["A"] * 4 + ["B"] * 4,
        "height": [
            1, 2, 3, np.inf,
            4, 5, 6, 7,
        ],
    })

    result = compare_groups(
        dataframe,
        value_column="height",
        group_column="treatment",
    )

    assert result["groups"]["A"]["n"] == 3


def test_zero_variance_group_rejected():
    dataframe = pd.DataFrame({
        "treatment": ["A"] * 3 + ["B"] * 3,
        "height": [1, 1, 1, 2, 3, 4],
    })

    with pytest.raises(ValueError):
        compare_groups(
            dataframe,
            value_column="height",
            group_column="treatment",
        )


def test_same_column_rejected(experimental_data):
    with pytest.raises(ValueError):
        compare_groups(
            experimental_data,
            value_column="height",
            group_column="height",
        )
