import numpy as np
import pytest

from biodatalab.effect_size import cohens_d, hedges_g, glass_delta


def test_cohens_d_basic():
    result = cohens_d([1, 2, 3, 4], [5, 6, 7, 8])

    assert result < 0


def test_cohens_d_equal_groups():
    result = cohens_d([1, 2, 3], [1, 2, 3])

    assert result == 0


def test_hedges_g_basic():
    result = hedges_g([1, 2, 3, 4], [5, 6, 7, 8])

    assert result < 0


def test_hedges_g_correction():
    d = cohens_d([1, 2, 3, 4], [5, 6, 7, 8])
    g = hedges_g([1, 2, 3, 4], [5, 6, 7, 8])

    assert abs(g) < abs(d)


def test_glass_delta_basic():
    result = glass_delta([1, 2, 3, 4], [5, 6, 7, 8])

    assert result < 0


def test_glass_delta_reference_group1():
    result = glass_delta(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        reference="group1",
    )

    assert result < 0


def test_glass_delta_invalid_reference():
    with pytest.raises(ValueError):
        glass_delta(
            [1, 2, 3],
            [4, 5, 6],
            reference="invalid",
        )


def test_cohens_d_small_groups():
    with pytest.raises(ValueError):
        cohens_d([1], [2])


def test_cohens_d_zero_variance():
    with pytest.raises(ValueError):
        cohens_d([5, 5, 5], [5, 5, 5])


def test_glass_delta_zero_reference_variance():
    with pytest.raises(ValueError):
        glass_delta([1, 2, 3], [5, 5, 5])


def test_nan_values():
    result = cohens_d(
        [1, 2, 3, np.nan],
        [4, 5, 6, np.nan],
    )

    assert np.isfinite(result)


def test_infinite_values():
    with pytest.raises(ValueError):
        cohens_d([1, 2, np.inf], [3, 4, 5])


def test_multidimensional_data():
    with pytest.raises(ValueError):
        cohens_d([[1, 2], [3, 4]], [5, 6, 7])
