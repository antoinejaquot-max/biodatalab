
"""Tests for sampling diagnostics."""

import numpy as np
import pandas as pd
import pytest

from biodatalab.sampling_diagnostics import sampling_diagnostics


@pytest.fixture
def population():
    return pd.DataFrame(
        {
            "height": [10, 20, 30, 40, 50],
            "weight": [2, 4, 6, 8, 10],
        }
    )


@pytest.fixture
def sample():
    return pd.DataFrame(
        {
            "height": [20, 30, 40],
            "weight": [4, 6, 8],
        }
    )


def test_diagnostics_structure(population, sample):
    result = sampling_diagnostics(population, sample)

    assert isinstance(result, pd.DataFrame)
    assert "mean_difference" in result.columns
    assert "variance_ratio" in result.columns
    assert "median_difference" in result.columns


def test_diagnostics_mean_difference(population, sample):
    result = sampling_diagnostics(population, sample)

    assert result.loc["height", "mean_difference"] == 0


def test_diagnostics_sample_size(population, sample):
    result = sampling_diagnostics(population, sample)

    assert result.loc["height", "sample_size"] == 3
    assert result.loc["height", "population_size"] == 5


def test_diagnostics_selected_columns(population, sample):
    result = sampling_diagnostics(
        population, sample, columns=["height"]
    )

    assert list(result.index) == ["height"]


def test_diagnostics_missing_column(population, sample):
    with pytest.raises(ValueError):
        sampling_diagnostics(
            population, sample, columns=["missing"]
        )


def test_diagnostics_empty_population():
    with pytest.raises(ValueError):
        sampling_diagnostics(
            pd.DataFrame(),
            pd.DataFrame({"x": [1]}),
        )


def test_diagnostics_empty_sample(population):
    with pytest.raises(ValueError):
        sampling_diagnostics(
            population,
            pd.DataFrame(),
        )


def test_diagnostics_non_numeric(population):
    sample = pd.DataFrame({"height": ["a", "b"]})

    with pytest.raises(TypeError):
        sampling_diagnostics(
            population,
            sample,
            columns=["height"],
        )


def test_diagnostics_missing_values(population):
    sample = pd.DataFrame(
        {"height": [10, None, 30]}
    )

    result = sampling_diagnostics(
        population,
        sample,
        columns=["height"],
    )

    assert result.loc["height", "sample_missing_fraction"] == 1 / 3


def test_diagnostics_variance_ratio(population, sample):
    result = sampling_diagnostics(population, sample)

    assert np.isfinite(result.loc["height", "variance_ratio"])


def test_diagnostics_infinite_values(population):
    sample = pd.DataFrame(
        {"height": [10, np.inf]}
    )

    with pytest.raises(ValueError):
        sampling_diagnostics(
            population,
            sample,
            columns=["height"],
        )


def test_diagnostics_quantiles(population, sample):
    result = sampling_diagnostics(population, sample)

    assert "q25_difference" in result.columns
    assert "q75_difference" in result.columns
