
"""Tests for sampling comparison utilities."""

import pandas as pd
import pytest

from biodatalab.sampling_comparison import compare_samples


@pytest.fixture
def reference():
    return pd.DataFrame(
        {
            "height": [10, 20, 30, 40, 50],
            "weight": [2, 4, 6, 8, 10],
        }
    )


@pytest.fixture
def samples():
    return {
        "sample_a": pd.DataFrame(
            {
                "height": [10, 20, 30],
                "weight": [2, 4, 6],
            }
        ),
        "sample_b": pd.DataFrame(
            {
                "height": [30, 40, 50],
                "weight": [6, 8, 10],
            }
        ),
    }


def test_compare_samples_structure(reference, samples):
    result = compare_samples(reference, samples)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert "mean_difference" in result.columns
    assert "standardized_mean_difference" in result.columns


def test_compare_samples_mean_difference(reference, samples):
    result = compare_samples(reference, samples)

    assert result.loc[
        ("sample_a", "height"), "mean_difference"
    ] == -10

    assert result.loc[
        ("sample_b", "height"), "mean_difference"
    ] == 10


def test_compare_samples_reproducible(reference, samples):
    first = compare_samples(reference, samples)
    second = compare_samples(reference, samples)

    pd.testing.assert_frame_equal(first, second)


def test_compare_samples_selected_columns(reference, samples):
    result = compare_samples(
        reference,
        samples,
        columns=["height"],
    )

    assert len(result) == 2
    assert set(result.index.get_level_values("variable")) == {"height"}


def test_compare_samples_missing_column(reference, samples):
    with pytest.raises(ValueError):
        compare_samples(reference, samples, columns=["missing"])


def test_compare_samples_empty_samples(reference):
    with pytest.raises(ValueError):
        compare_samples(reference, {})


def test_compare_samples_empty_reference():
    with pytest.raises(ValueError):
        compare_samples(
            pd.DataFrame(),
            {"sample": pd.DataFrame({"x": [1, 2]})},
        )


def test_compare_samples_invalid_sample_type(reference):
    with pytest.raises(TypeError):
        compare_samples(reference, {"bad": "not a dataframe"})


def test_compare_samples_missing_values(reference):
    sample = pd.DataFrame(
        {"height": [10, None, 30, 40]}
    )

    result = compare_samples(
        reference,
        {"sample": sample},
        columns=["height"],
    )

    assert result.loc[
        ("sample", "height"), "missing_fraction"
    ] == 0.25


def test_compare_samples_missing_sample_column(reference):
    with pytest.raises(ValueError):
        compare_samples(
            reference,
            {"sample": pd.DataFrame({"other": [1, 2]})},
            columns=["height"],
        )


def test_compare_samples_non_numeric(reference):
    sample = pd.DataFrame(
        {"height": ["a", "b"]}
    )

    with pytest.raises(TypeError):
        compare_samples(
            reference,
            {"sample": sample},
            columns=["height"],
        )


def test_compare_samples_empty_numeric_column(reference):
    sample = pd.DataFrame(
        {"height": [None, None]}
    )

    with pytest.raises(ValueError):
        compare_samples(
            reference,
            {"sample": sample},
            columns=["height"],
        )
