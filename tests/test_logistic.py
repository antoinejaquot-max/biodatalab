import numpy as np
import pandas as pd
import pytest

from biodatalab.logistic import logistic_regression


def create_dataframe():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [2, 1, 4, 3, 6, 5, 8, 7],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_logistic_basic():
    result = logistic_regression(create_dataframe(), "target")

    assert "coefficients" in result
    assert "predictions" in result
    assert "accuracy" in result


def test_logistic_coefficient_shape():
    result = logistic_regression(create_dataframe(), "target")

    assert len(result["coefficients"]) == 3


def test_logistic_prediction_shape():
    result = logistic_regression(create_dataframe(), "target")

    assert len(result["predictions"]) == 8


def test_logistic_probabilities_range():
    result = logistic_regression(create_dataframe(), "target")

    probabilities = result["predictions"]["probability"]

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_logistic_accuracy_range():
    result = logistic_regression(create_dataframe(), "target")

    assert 0 <= result["accuracy"] <= 1


def test_logistic_odds_ratios():
    result = logistic_regression(create_dataframe(), "target")

    odds_ratios = result["coefficients"]["odds_ratio"]

    assert np.all(odds_ratios > 0)


def test_logistic_missing_target():
    with pytest.raises(ValueError):
        logistic_regression(create_dataframe(), "missing")


def test_logistic_non_numeric_predictor():
    dataframe = create_dataframe()
    dataframe["category"] = ["a", "b"] * 4

    with pytest.raises(TypeError):
        logistic_regression(dataframe, "target")


def test_logistic_missing_values():
    dataframe = create_dataframe()
    dataframe.loc[0, "feature1"] = np.nan

    with pytest.raises(ValueError):
        logistic_regression(dataframe, "target")


def test_logistic_invalid_target_classes():
    dataframe = create_dataframe()
    dataframe["target"] = [0, 0, 0, 0, 2, 2, 2, 2]

    with pytest.raises(ValueError):
        logistic_regression(dataframe, "target")


def test_logistic_constant_predictor():
    dataframe = create_dataframe()
    dataframe["constant"] = 1

    with pytest.raises(ValueError):
        logistic_regression(dataframe, "target")


def test_logistic_invalid_iterations():
    with pytest.raises(ValueError):
        logistic_regression(
            create_dataframe(),
            "target",
            max_iter=0,
        )


def test_logistic_invalid_tolerance():
    with pytest.raises(ValueError):
        logistic_regression(
            create_dataframe(),
            "target",
            tolerance=0,
        )


def test_logistic_empty_dataframe():
    with pytest.raises(ValueError):
        logistic_regression(pd.DataFrame(), "target")


def test_logistic_duplicate_columns():
    dataframe = pd.DataFrame(
        [[1, 2, 0], [3, 4, 1], [5, 6, 0]],
        columns=["A", "A", "target"],
    )

    with pytest.raises(ValueError):
        logistic_regression(dataframe, "target")
