
"""Tests for linear regression methods."""

import numpy as np
import pandas as pd
import pytest

from biodatalab.regression import (
    fit_linear_regression,
    predict_linear_regression,
)


@pytest.fixture
def simple_data():
    return pd.DataFrame({
        "temperature": [10, 12, 14, 16, 18, 20, 22, 24],
        "growth": [5, 7, 9, 11, 13, 15, 17, 19],
    })


@pytest.fixture
def multiple_data():
    return pd.DataFrame({
        "temperature": [10, 12, 14, 16, 18, 20, 22, 24],
        "water": [1, 2, 1, 3, 2, 4, 3, 5],
        "growth": [8, 11, 10, 15, 16, 21, 22, 27],
    })


def test_simple_regression_coefficients(simple_data):
    result = fit_linear_regression(simple_data, "growth", ["temperature"])
    assert result["coefficients"]["temperature"]["coefficient"] == pytest.approx(1)
    assert result["coefficients"]["intercept"]["coefficient"] == pytest.approx(-5)


def test_simple_regression_r_squared(simple_data):
    result = fit_linear_regression(simple_data, "growth", ["temperature"])
    assert result["model"]["r_squared"] == pytest.approx(1)
    assert result["model"]["adjusted_r_squared"] == pytest.approx(1)


def test_regression_sample_size(simple_data):
    result = fit_linear_regression(simple_data, "growth", ["temperature"])
    assert result["n_observations"] == 8
    assert result["model"]["residual_degrees_of_freedom"] == 6


def test_coefficient_statistics(simple_data):
    result = fit_linear_regression(simple_data, "growth", ["temperature"])
    coefficient = result["coefficients"]["temperature"]
    assert "standard_error" in coefficient
    assert "t_statistic" in coefficient
    assert "p_value" in coefficient
    assert "confidence_interval" in coefficient


def test_multiple_regression(multiple_data):
    result = fit_linear_regression(
        multiple_data, "growth", ["temperature", "water"]
    )
    assert "temperature" in result["coefficients"]
    assert "water" in result["coefficients"]
    assert 0 <= result["model"]["r_squared"] <= 1


def test_prediction_simple(simple_data):
    result = fit_linear_regression(simple_data, "growth", ["temperature"])
    predictions = predict_linear_regression(
        pd.DataFrame({"temperature": [10, 20, 30]}),
        ["temperature"],
        result["coefficients"],
    )
    np.testing.assert_allclose(predictions, [5, 15, 25])


def test_prediction_multiple(multiple_data):
    result = fit_linear_regression(
        multiple_data, "growth", ["temperature", "water"]
    )
    predictions = predict_linear_regression(
        multiple_data[["temperature", "water"]],
        ["temperature", "water"],
        result["coefficients"],
    )
    assert len(predictions) == len(multiple_data)
    assert np.isfinite(predictions).all()


def test_missing_values_are_dropped():
    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [2, 4, np.nan, 8, 10, 12],
    })
    result = fit_linear_regression(data, "y", ["x"])
    assert result["n_observations"] == 5
    assert result["dropped_rows"] == 1


def test_constant_outcome_rejected():
    data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [5, 5, 5, 5]})
    with pytest.raises(ValueError):
        fit_linear_regression(data, "y", ["x"])


def test_missing_column_rejected(simple_data):
    with pytest.raises(ValueError):
        fit_linear_regression(simple_data, "unknown", ["temperature"])


def test_duplicate_predictors_rejected(simple_data):
    with pytest.raises(ValueError):
        fit_linear_regression(
            simple_data, "growth", ["temperature", "temperature"]
        )


def test_constant_predictor_rejected():
    data = pd.DataFrame({"x": [1, 1, 1, 1, 1], "y": [2, 3, 4, 5, 6]})
    with pytest.raises(ValueError):
        fit_linear_regression(data, "y", ["x"])


def test_invalid_alpha_rejected(simple_data):
    for alpha in [0, 1, -0.1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            fit_linear_regression(simple_data, "growth", ["temperature"], alpha)


def test_perfect_fit_has_undefined_coefficient_statistics():
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    result = fit_linear_regression(data, "y", ["x"])

    assert result["model"]["r_squared"] == pytest.approx(1)
    assert result["model"]["p_value"] == 0
    assert result["coefficients"]["x"]["standard_error"] is None
    assert result["coefficients"]["x"]["p_value"] is None
    assert result["diagnostics"]["shapiro_residuals"] is None


def test_prediction_rejects_infinite_values():
    data = pd.DataFrame({"x": [1, np.inf]})
    coefficients = {
        "intercept": {"coefficient": 0},
        "x": {"coefficient": 2},
    }
    with pytest.raises(ValueError):
        predict_linear_regression(data, ["x"], coefficients)


def test_prediction_rejects_invalid_coefficient():
    data = pd.DataFrame({"x": [1, 2]})
    coefficients = {
        "intercept": {"coefficient": 0},
        "x": {"coefficient": np.inf},
    }
    with pytest.raises(ValueError):
        predict_linear_regression(data, ["x"], coefficients)
