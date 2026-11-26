import numpy as np
import pytest

from biodatalab.permutation import permutation_test


def test_permutation_basic():
    result = permutation_test(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        n_permutations=500,
        random_state=42,
    )

    assert 0 <= result["p_value"] <= 1
    assert result["observed_statistic"] < 0


def test_permutation_reproducibility():
    first = permutation_test(
        [1, 2, 3],
        [4, 5, 6],
        n_permutations=100,
        random_state=42,
    )

    second = permutation_test(
        [1, 2, 3],
        [4, 5, 6],
        n_permutations=100,
        random_state=42,
    )

    assert first == second


def test_permutation_greater():
    result = permutation_test(
        [5, 6, 7],
        [1, 2, 3],
        n_permutations=100,
        alternative="greater",
        random_state=42,
    )

    assert 0 <= result["p_value"] <= 1


def test_permutation_less():
    result = permutation_test(
        [1, 2, 3],
        [5, 6, 7],
        n_permutations=100,
        alternative="less",
        random_state=42,
    )

    assert 0 <= result["p_value"] <= 1


def test_permutation_two_sided():
    result = permutation_test(
        [1, 2, 3],
        [5, 6, 7],
        n_permutations=100,
        alternative="two-sided",
        random_state=42,
    )

    assert 0 < result["p_value"] <= 1


def test_permutation_custom_statistic():
    result = permutation_test(
        [1, 2, 3],
        [4, 5, 6],
        statistic=lambda x, y: np.median(x) - np.median(y),
        n_permutations=100,
        random_state=42,
    )

    assert result["observed_statistic"] == -3


def test_permutation_invalid_count():
    with pytest.raises(ValueError):
        permutation_test(
            [1, 2],
            [3, 4],
            n_permutations=0,
        )


def test_permutation_invalid_alternative():
    with pytest.raises(ValueError):
        permutation_test(
            [1, 2],
            [3, 4],
            alternative="invalid",
        )


def test_permutation_too_small_group():
    with pytest.raises(ValueError):
        permutation_test([1], [2, 3])


def test_permutation_nan_values():
    result = permutation_test(
        [1, 2, np.nan, 3],
        [4, 5, np.nan, 6],
        n_permutations=100,
        random_state=42,
    )

    assert 0 <= result["p_value"] <= 1


def test_permutation_infinite_values():
    with pytest.raises(ValueError):
        permutation_test([1, 2, np.inf], [3, 4, 5])


def test_permutation_multidimensional():
    with pytest.raises(ValueError):
        permutation_test([[1, 2], [3, 4]], [5, 6, 7])


def test_permutation_invalid_statistic():
    with pytest.raises(TypeError):
        permutation_test(
            [1, 2, 3],
            [4, 5, 6],
            statistic="invalid",
        )
