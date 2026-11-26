
"""Logistic regression utilities for BioDataLab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute sigmoid values while avoiding numerical overflow."""
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def logistic_regression(
    dataframe: pd.DataFrame,
    target: str,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    """Fit a binary logistic regression using a stabilized Newton method."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("DataFrame must not be empty.")

    if dataframe.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names.")

    if target not in dataframe.columns:
        raise ValueError(f"Target column '{target}' not found.")

    if len(dataframe.columns) < 2:
        raise ValueError("At least one predictor is required.")

    if not isinstance(max_iter, int) or isinstance(max_iter, bool):
        raise TypeError("max_iter must be an integer.")

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    if not isinstance(tolerance, (int, float)):
        raise TypeError("tolerance must be numeric.")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    predictors = dataframe.drop(columns=[target])
    y = dataframe[target]

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in predictors.dtypes
    ):
        raise TypeError("Predictor columns must be numeric.")

    if not pd.api.types.is_numeric_dtype(y):
        raise TypeError("Target must be numeric.")

    if dataframe.isna().any().any():
        raise ValueError("DataFrame must not contain missing values.")

    try:
        x_values = predictors.to_numpy(dtype=float)
        y_values = y.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("DataFrame must contain numeric values.") from exc

    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("DataFrame must not contain infinite values.")

    unique_classes = np.unique(y_values)

    if len(unique_classes) != 2 or not np.isin(
        unique_classes, [0, 1]
    ).all():
        raise ValueError("Target must contain exactly the classes 0 and 1.")

    if len(y_values) < 3:
        raise ValueError("At least 3 observations are required.")

    if np.any(x_values.std(axis=0) == 0):
        raise ValueError("Predictors must not have zero variance.")

    means = x_values.mean(axis=0)
    stds = x_values.std(axis=0)
    x_scaled = (x_values - means) / stds

    x = np.column_stack([np.ones(len(x_scaled)), x_scaled])

    coefficients = np.zeros(x.shape[1], dtype=float)
    converged = False

    regularization = 1e-6
    penalty = np.eye(x.shape[1]) * regularization
    penalty[0, 0] = 0.0

    for iteration in range(max_iter):
        linear_predictor = x @ coefficients
        probabilities = _sigmoid(linear_predictor)

        gradient = x.T @ (y_values - probabilities)
        weights = np.clip(
            probabilities * (1.0 - probabilities),
            1e-10,
            None,
        )

        hessian = x.T @ (x * weights[:, None]) + penalty

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient

        if not np.isfinite(step).all():
            break

        coefficients += step

        if np.linalg.norm(step) < tolerance:
            converged = True
            break

    final_probabilities = _sigmoid(x @ coefficients)
    predictions = (final_probabilities >= 0.5).astype(int)

    accuracy = float(np.mean(predictions == y_values))

    odds_ratios = np.exp(np.clip(coefficients, -500, 500))

    coefficient_table = pd.DataFrame(
        {
            "coefficient": coefficients,
            "odds_ratio": odds_ratios,
        },
        index=["intercept"] + list(predictors.columns),
    )

    prediction_table = pd.DataFrame(
        {
            "prediction": predictions,
            "probability": final_probabilities,
        },
        index=dataframe.index,
    )

    return {
        "coefficients": coefficient_table,
        "predictions": prediction_table,
        "probabilities": pd.Series(
            final_probabilities,
            index=dataframe.index,
            name="probability",
        ),
        "accuracy": accuracy,
        "odds_ratios": pd.Series(
            odds_ratios[1:],
            index=predictors.columns,
        ),
        "converged": converged,
        "iterations": iteration + 1,
    }
