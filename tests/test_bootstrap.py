
import numpy as np
import pytest

from biodatalab.bootstrap import bootstrap_ci


def test_bootstrap_mean():
    data = [1, 2, 3, 4, 5]

    result = bootstrap_ci(data, random_state=42)

    assert result["estimate"] == pytest.approx(3.0)
    assert result["lower_bound"] <= result["estimate"]
    assert result["upper_bound"] >= result["estimate"]


def test_bootstrap_reproducibility():
    data = [1, 2, 3, 4, 5]

    result1 = bootstrap_ci(data, random_state=42)
    result2 = bootstrap_ci(data, random_state=42)

    assert result1 == result2


def test_bootstrap_median():
    data = [1, 2, 3, 4, 5]

    result = bootstrap_ci(
        data,
        statistic=np.median,
        random_state=42,
    )

    assert result["estimate"] == pytest.approx(3.0)


def test_bootstrap_missing_values():
    data = [1, 2, np.nan, 4, 5]

    result = bootstrap_ci(data, random_state=42)

    assert result["estimate"] == pytest.approx(3.0)


def test_bootstrap_empty_data():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_invalid_resamples():
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2, 3], n_resamples=0)


def test_bootstrap_invalid_confidence():
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2, 3], confidence_level=1)


def test_bootstrap_invalid_statistic():
    with pytest.raises(TypeError):
        bootstrap_ci([1, 2, 3], statistic="mean")


def test_bootstrap_single_resample():
    result = bootstrap_ci(
        [1, 2, 3],
        n_resamples=1,
        random_state=42,
    )

    assert result["n_resamples"] == 1
    assert result["standard_error"] == 0.0


def test_bootstrap_result_structure():
    result = bootstrap_ci([1, 2, 3], random_state=42)

    expected_keys = {
        "estimate",
        "lower_bound",
        "upper_bound",
        "standard_error",
        "confidence_level",
        "n_resamples",
    }

    assert set(result.keys()) == expected_keys
