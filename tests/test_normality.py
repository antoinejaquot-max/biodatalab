import numpy as np
import pytest

from biodatalab.normality import test_normality


def test_normality_basic():
    data = np.random.default_rng(42).normal(size=100)

    result = test_normality(data)

    assert result["method"] == "shapiro"
    assert 0 <= result["p_value"] <= 1
    assert isinstance(result["is_normal"], bool)


def test_normality_dagostino():
    data = np.random.default_rng(42).normal(size=100)

    result = test_normality(data, method="dagostino")

    assert result["method"] == "dagostino"


def test_normality_jarque_bera():
    data = np.random.default_rng(42).normal(size=100)

    result = test_normality(data, method="jarque_bera")

    assert result["method"] == "jarque_bera"


def test_non_normal_data():
    data = np.random.default_rng(42).exponential(size=500)

    result = test_normality(data)

    assert result["is_normal"] is False


def test_nan_values_are_removed():
    data = np.array([1, 2, 3, 4, 5, np.nan])

    result = test_normality(data)

    assert result["sample_size"] == 5


def test_too_few_observations():
    with pytest.raises(ValueError):
        test_normality([1, 2])


def test_constant_data():
    with pytest.raises(ValueError):
        test_normality([5, 5, 5, 5])


def test_invalid_method():
    with pytest.raises(ValueError):
        test_normality([1, 2, 3, 4], method="invalid")


def test_invalid_alpha():
    with pytest.raises(ValueError):
        test_normality([1, 2, 3, 4], alpha=1)


def test_dagostino_small_sample():
    with pytest.raises(ValueError):
        test_normality(
            [1, 2, 3, 4, 5, 6, 7],
            method="dagostino",
        )


def test_infinite_values():
    with pytest.raises(ValueError):
        test_normality([1, 2, 3, np.inf])


def test_multidimensional_data():
    with pytest.raises(ValueError):
        test_normality([[1, 2], [3, 4]])
