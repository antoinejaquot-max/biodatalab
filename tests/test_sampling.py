
"""Tests for BioDataLab sampling utilities."""

import pandas as pd
import pytest

from biodatalab.sampling import sample_dataframe, stratified_sample


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "value": range(100),
            "group": ["A"] * 70 + ["B"] * 30,
        }
    )


def test_sample_dataframe_size(dataframe):
    result = sample_dataframe(dataframe, n=20, random_state=42)
    assert len(result) == 20


def test_sample_dataframe_reproducibility(dataframe):
    first = sample_dataframe(dataframe, n=20, random_state=42)
    second = sample_dataframe(dataframe, n=20, random_state=42)
    pd.testing.assert_frame_equal(first, second)


def test_sample_dataframe_fraction(dataframe):
    result = sample_dataframe(dataframe, frac=0.2, random_state=42)
    assert len(result) == 20


def test_sample_dataframe_with_replacement(dataframe):
    result = sample_dataframe(
        dataframe,
        n=150,
        replace=True,
        random_state=42,
    )
    assert len(result) == 150


def test_sample_dataframe_invalid_size(dataframe):
    with pytest.raises(ValueError):
        sample_dataframe(dataframe, n=200)


def test_sample_dataframe_missing_parameters(dataframe):
    with pytest.raises(ValueError):
        sample_dataframe(dataframe)


def test_sample_dataframe_both_parameters(dataframe):
    with pytest.raises(ValueError):
        sample_dataframe(dataframe, n=10, frac=0.1)


def test_stratified_sample_size(dataframe):
    result = stratified_sample(
        dataframe,
        stratify_by="group",
        n=20,
        random_state=42,
    )
    assert len(result) == 20


def test_stratified_sample_proportions(dataframe):
    result = stratified_sample(
        dataframe,
        stratify_by="group",
        n=50,
        random_state=42,
    )

    counts = result["group"].value_counts()

    assert counts["A"] == 35
    assert counts["B"] == 15


def test_stratified_sample_reproducibility(dataframe):
    first = stratified_sample(
        dataframe, "group", n=20, random_state=42
    )
    second = stratified_sample(
        dataframe, "group", n=20, random_state=42
    )

    pd.testing.assert_frame_equal(first, second)


def test_stratified_sample_fraction(dataframe):
    result = stratified_sample(
        dataframe, "group", frac=0.2, random_state=42
    )
    assert len(result) == 20


def test_stratified_sample_missing_column(dataframe):
    with pytest.raises(ValueError):
        stratified_sample(dataframe, "missing", n=10)


def test_stratified_sample_missing_values(dataframe):
    dataframe.loc[0, "group"] = None

    with pytest.raises(ValueError):
        stratified_sample(dataframe, "group", n=10)


def test_stratified_sample_invalid_size(dataframe):
    with pytest.raises(ValueError):
        stratified_sample(dataframe, "group", n=200)


def test_stratified_sample_full_dataset(dataframe):
    result = stratified_sample(
        dataframe, "group", n=100, random_state=42
    )

    assert len(result) == len(dataframe)
    assert set(result.index) == set(dataframe.index)
