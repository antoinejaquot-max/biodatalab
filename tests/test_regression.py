
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


# --------------------------------------------------
# Simple linear regression
# --------------------------------------------------

def test_simple_regression_coefficients(simple_data):
    result = fit_linear_regression(
        simple_data,
        outcome="growth",
        predictors=["temperature"],
    )

    assert result["coefficients"]["temperature"]["coefficient"] == pytest.approx(1)
    assert result["coefficients"]["intercept"]["coefficient"] == pytest.approx(-5)


def test_simple_regression_r_squared(simple_data):
    result = fit_linear_regression(
        simple_data,
        outcome="growth",
        predictors=["temperature"],
    )

    assert result["model"]["r_squared"] == pytest.approx(1)
    assert result["model"]["adjusted_r_squared"] == pytest.approx(1)


def test_simple_regression_sample_size(simple_data):
    result = fit_linear_regression(
        simple_data,
        outcome="growth",
        predictors=["temperature"],
    )

    assert result["n_observations"] == 8
    assert result["model"]["residual_degrees_of_freedom"] == 6


def test_regression_returns_coefficient_statistics(simple_data):
    result = fit_linear_regression(
        simple_data,
        outcome="growth",
        predictors=["temperature"],
    )

    coefficient = result["coefficients"]["temperature"]

    assert "standard_error" in coefficient
    assert "t_statistic" in coefficient
    assert "p_value" in coefficient
    assert "confidence_interval" in coefficient


# --------------------------------------------------
# Multiple linear regression
# --------------------------------------------------

def test_multiple_regression_returns_both_predictors(multiple_data):
    result = fit_linear_regression(
        multiple_data,
        outcome="growth",
        predictors=["temperature", "water"],
    )

    assert "temperature" in result["coefficients"]
    assert "water" in result["coefficients"]


def test_multiple_regression_has_valid_r_squared(multiple_data):
    result = fit_linear_regression(
        multiple_data,
        outcome="growth",
        predictors=["temperature", "water"],
    )

    assert 0 <= result["model"]["r_squared"] <= 1


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def test_prediction_matches_known_equation(simple_data):
    result = fit_linear_regression(
        simple_data,
        outcome="growth",
        predictors=["temperature"],
    )

    predictions = predict_linear_regression(
        pd.DataFrame({"temperature": [10, 20, 30]}),
        ["temperature"],
        result["coefficients"],
    )

    np.testing.assert_allclose(
        predictions,
        [5, 15, 25],
        atol=1e-8,
    )


def test_prediction_multiple_regression(multiple_data):
    result = fit_linear_regression(
        multiple_data,
        outcome="growth",
        predictors=["temperature", "water"],
    )

    predictions = predict_linear_regression(
        multiple_data[["temperature", "water"]],
        ["temperature", "water"],
        result["coefficients"],
    )

    assert len(predictions) == len(multiple_data)
    assert np.isfinite(predictions).all()


# --------------------------------------------------
# Missing and invalid data
# --------------------------------------------------

def test_regression_drops_missing_values():
    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [2, 4, np.nan, 8, 10, 12],
    })

    result = fit_linear_regression(
        data,
        outcome="y",
        predictors=["x"],
    )

    assert result["n_observations"] == 5
    assert result["dropped_rows"] == 1


def test_regression_rejects_constant_outcome():
    data = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [5, 5, 5, 5],
    })

    with pytest.raises(ValueError):
        fit_linear_regression(data, "y", ["x"])


def test_regression_rejects_missing_column(simple_data):
    with pytest.raises(ValueError):
        fit_linear_regression(
            simple_data,
            outcome="unknown",
            predictors=["temperature"],
        )


def test_regression_rejects_duplicate_predictors(simple_data):
    with pytest.raises(ValueError):
        fit_linear_regression(
            simple_data,
            outcome="growth",
            predictors=["temperature", "temperature"],
        )


def test_regression_rejects_constant_predictor():
    data = pd.DataFrame({
        "x": [1, 1, 1, 1, 1],
        "y": [2, 3, 4, 5, 6],
    })

    with pytest.raises(ValueError):
        fit_linear_regression(data, "y", ["x"])


def test_regression_rejects_invalid_alpha(simple_data):
    with pytest.raises(ValueError):
        fit_linear_regression(
            simple_data,
            "growth",
            ["temperature"],
            alpha=1,
        )


# --------------------------------------------------
# Prediction validation
# --------------------------------------------------

def test_prediction_rejects_missing_coefficients():
    data = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError):
        predict_linear_regression(
            data,
            ["x"],
            {"intercept": {"coefficient": 0}},
        )


def test_prediction_rejects_missing_values():
    data = pd.DataFrame({"x": [1, np.nan, 3]})

    coefficients = {
        "intercept": {"coefficient": 0},
        "x": {"coefficient": 2},
    }

    with pytest.raises(ValueError):
        predict_linear_regression(data, ["x"], coefficients)


def test_prediction_rejects_missing_columns():
    data = pd.DataFrame({"temperature": [10, 20]})

    coefficients = {
        "intercept": {"coefficient": 0},
        "water": {"coefficient": 2},
    }

    with pytest.raises(ValueError):
        predict_linear_regression(
            data,
            ["water"],
            coefficients,
        )


# --------------------------------------------------
# Model diagnostics
# --------------------------------------------------

def test_regression_returns_diagnostics(simple_data):
    result = fit_linear_regression(
        simple_data,
        "growth",
        ["temperature"],
    )

    diagnostics = result["diagnostics"]

    assert "durbin_watson" in diagnostics
    assert "max_cooks_distance" in diagnostics
    assert "shapiro_residuals" in diagnostics
