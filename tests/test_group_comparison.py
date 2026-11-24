
import numpy as np
import pandas as pd
import pytest

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

        assert ci is not None
        assert ci[0] < group["mean"] < ci[1]


def test_statistical_test(experimental_data):
    result = compare_groups(
        experimental_data,
        value_column="height",
        group_column="treatment",
    )

    assert result["test"]["name"] == "Welch's t-test"
    assert 0 <= result["test"]["p_value"] <= 1
    assert result["test"]["significant_at_0_05"] is True


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
        "treatment": ["Control", "Control", "Control"],
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
        "treatment": ["Control", "Control", "Fertilizer", "Fertilizer"],
        "height": [10.0, np.nan, 15.0, 16.0],
    })

    result = compare_groups(
        dataframe,
        value_column="height",
        group_column="treatment",
    )

    assert result["groups"]["Control"]["n"] == 1
    assert result["groups"]["Fertilizer"]["n"] == 2
    assert result["groups"]["Control"]["std"] is None
