
"""Linear regression methods for biological and agricultural datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _prepare_data(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if not predictors:
        raise ValueError("At least one predictor is required.")

    if len(set(predictors)) != len(predictors):
        raise ValueError("Predictor names must be unique.")

    if outcome in predictors:
        raise ValueError("The outcome cannot also be a predictor.")

    required = [outcome] + predictors
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric = data[required].apply(
        pd.to_numeric,
        errors="coerce",
    )

    original_size = len(numeric)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    if len(numeric) < len(predictors) + 2:
        raise ValueError(
            "Not enough complete observations for the regression."
        )

    y = numeric[outcome].to_numpy(dtype=float)
    x = numeric[predictors].to_numpy(dtype=float)

    if np.var(y) == 0:
        raise ValueError("The outcome must not be constant.")

    design = np.column_stack([np.ones(len(x)), x])

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "Predictors are linearly dependent or constant."
        )

    return y, design, predictors, original_size - len(numeric)


def fit_linear_regression(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    alpha: float = 0.05,
) -> dict:
    """Fit an ordinary least-squares linear regression.

    Supports simple and multiple linear regression.

    Observations with missing or non-finite values are excluded.
    Predictors must be numeric and observations must be independent
    for the usual inferential statistics to be valid.
    """

    _validate_alpha(alpha)

    y, design, predictor_names, dropped_rows = _prepare_data(
        data,
        outcome,
        predictors,
    )

    n, parameter_count = design.shape
    predictor_count = parameter_count - 1
    residual_df = n - parameter_count

    if residual_df <= 0:
        raise ValueError(
            "The model requires more observations than parameters."
        )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    fitted = design @ coefficients
    residuals = y - fitted

    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y))**2))

    r_squared = 1 - sse / sst
    adjusted_r_squared = (
        1 - (1 - r_squared) * (n - 1) / residual_df
    )

    residual_variance = sse / residual_df
    covariance = residual_variance * np.linalg.inv(
        design.T @ design
    )

    standard_errors = np.sqrt(np.diag(covariance))
    t_statistics = coefficients / standard_errors

    p_values = 2 * stats.t.sf(
        np.abs(t_statistics),
        df=residual_df,
    )

    critical_t = stats.t.ppf(
        1 - alpha / 2,
        df=residual_df,
    )

    confidence_intervals = np.column_stack([
        coefficients - critical_t * standard_errors,
        coefficients + critical_t * standard_errors,
    ])

    if predictor_count > 0:
        explained_sum_squares = max(sst - sse, 0.0)
        model_f = (
            explained_sum_squares / predictor_count
        ) / residual_variance

        model_p = float(
            stats.f.sf(
                model_f,
                predictor_count,
                residual_df,
            )
        )
    else:
        model_f = float("nan")
        model_p = float("nan")

    # Diagnostic statistics
    leverage = np.sum(
        (design @ np.linalg.inv(design.T @ design)) * design,
        axis=1,
    )

    if residual_variance > 0:
        standardized_residuals = (
            residuals
            / np.sqrt(
                residual_variance * (1 - leverage)
            )
        )

        cooks_distance = (
            standardized_residuals**2
            * leverage
            / (
                parameter_count
                * (1 - leverage)
            )
        )
    else:
        standardized_residuals = np.zeros(n)
        cooks_distance = np.zeros(n)

    shapiro = (
        stats.shapiro(residuals)
        if 3 <= n <= 5000
        else None
    )

    durbin_watson = (
        float(
            np.sum(np.diff(residuals) ** 2) / sse
        )
        if sse > 0
        else float("nan")
    )

    names = ["intercept"] + predictor_names

    coefficient_results = {}

    for index, name in enumerate(names):
        coefficient_results[name] = {
            "coefficient": float(coefficients[index]),
            "standard_error": float(standard_errors[index]),
            "t_statistic": float(t_statistics[index]),
            "p_value": float(p_values[index]),
            "significant": bool(p_values[index] < alpha),
            "confidence_interval": [
                float(confidence_intervals[index, 0]),
                float(confidence_intervals[index, 1]),
            ],
        }

    return {
        "method": "Ordinary least squares",
        "outcome": outcome,
        "predictors": predictor_names,
        "n_observations": n,
        "dropped_rows": dropped_rows,
        "alpha": alpha,
        "coefficients": coefficient_results,
        "model": {
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(adjusted_r_squared),
            "f_statistic": float(model_f),
            "p_value": model_p,
            "significant": bool(model_p < alpha),
            "residual_standard_error": float(
                np.sqrt(residual_variance)
            ),
            "residual_degrees_of_freedom": residual_df,
        },
        "diagnostics": {
            "shapiro_residuals": (
                {
                    "statistic": float(shapiro.statistic),
                    "p_value": float(shapiro.pvalue),
                    "normality_not_rejected": bool(
                        shapiro.pvalue >= alpha
                    ),
                }
                if shapiro is not None
                else None
            ),
            "durbin_watson": durbin_watson,
            "max_absolute_standardized_residual": float(
                np.max(np.abs(standardized_residuals))
            ),
            "max_cooks_distance": float(
                np.max(cooks_distance)
            ),
        },
    }


def predict_linear_regression(
    data: pd.DataFrame,
    predictors: list[str],
    coefficients: dict,
) -> np.ndarray:
    """Generate predictions using fitted regression coefficients."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    required = ["intercept"] + predictors

    missing_coefficients = [
        name for name in required
        if name not in coefficients
    ]

    if missing_coefficients:
        raise ValueError(
            f"Missing coefficients: {missing_coefficients}"
        )

    missing_columns = [
        name for name in predictors
        if name not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing predictor columns: {missing_columns}"
        )

    numeric = data[predictors].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().any().any():
        raise ValueError(
            "Prediction data contains missing or non-numeric values."
        )

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(
            "Prediction data contains infinite values."
        )

    result = np.full(
        len(data),
        float(coefficients["intercept"]["coefficient"]),
    )

    for predictor in predictors:
        result += (
            numeric[predictor].to_numpy(dtype=float)
            * float(coefficients[predictor]["coefficient"])
        )

    return result
