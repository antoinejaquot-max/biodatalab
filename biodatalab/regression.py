
"""Linear regression methods for biological and agricultural datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _prepare_data(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if not isinstance(predictors, list) or not predictors:
        raise ValueError("At least one predictor is required.")

    if not all(isinstance(name, str) for name in predictors):
        raise ValueError("Predictor names must be strings.")

    if not isinstance(outcome, str):
        raise ValueError("Outcome name must be a string.")

    if len(set(predictors)) != len(predictors):
        raise ValueError("Predictor names must be unique.")

    if outcome in predictors:
        raise ValueError("The outcome cannot also be a predictor.")

    required = [outcome] + predictors
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric = data[required].apply(pd.to_numeric, errors="coerce")
    original_size = len(numeric)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    if len(numeric) < len(predictors) + 2:
        raise ValueError("Not enough complete observations.")

    y = numeric[outcome].to_numpy(dtype=float)
    x = numeric[predictors].to_numpy(dtype=float)

    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Data must contain finite values.")

    if np.var(y) == 0:
        raise ValueError("The outcome must not be constant.")

    design = np.column_stack([np.ones(len(x)), x])

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Predictors are linearly dependent or constant.")

    return y, design, predictors, original_size - len(numeric)


def fit_linear_regression(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    alpha: float = 0.05,
) -> dict:
    """Fit an ordinary least-squares linear regression."""

    _validate_alpha(alpha)

    y, design, predictor_names, dropped_rows = _prepare_data(
        data, outcome, predictors
    )

    n, parameter_count = design.shape
    predictor_count = parameter_count - 1
    residual_df = n - parameter_count

    if residual_df <= 0:
        raise ValueError("The model requires more observations than parameters.")

    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    fitted = design @ coefficients
    residuals = y - fitted

    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y))**2))

    r_squared = float(1 - sse / sst)
    adjusted_r_squared = float(
        1 - (1 - r_squared) * (n - 1) / residual_df
    )

    residual_variance = sse / residual_df
    residual_se = float(np.sqrt(residual_variance))

    matrix_inverse = np.linalg.inv(design.T @ design)
    covariance = residual_variance * matrix_inverse

    exact_fit = np.isclose(sse, 0.0, atol=1e-14)

    if exact_fit:
        standard_errors = np.full(parameter_count, np.nan)
        t_statistics = np.full(parameter_count, np.nan)
        p_values = np.full(parameter_count, np.nan)
        confidence_intervals = np.full((parameter_count, 2), np.nan)
    else:
        standard_errors = np.sqrt(np.diag(covariance))
        t_statistics = coefficients / standard_errors
        p_values = 2 * stats.t.sf(np.abs(t_statistics), residual_df)

        critical_t = stats.t.ppf(1 - alpha / 2, residual_df)
        confidence_intervals = np.column_stack([
            coefficients - critical_t * standard_errors,
            coefficients + critical_t * standard_errors,
        ])

    if predictor_count > 0 and not exact_fit:
        explained_sum_squares = max(sst - sse, 0.0)
        model_f = float(
            (explained_sum_squares / predictor_count)
            / residual_variance
        )
        model_p = float(
            stats.f.sf(model_f, predictor_count, residual_df)
        )
    elif predictor_count > 0 and exact_fit:
        model_f = float("inf")
        model_p = 0.0
    else:
        model_f = None
        model_p = None

    # Diagnostic statistics
    hat_matrix_diagonal = np.sum(
        (design @ matrix_inverse) * design, axis=1
    )

    if exact_fit:
        standardized_residuals = np.full(n, np.nan)
        cooks_distance = np.full(n, np.nan)
    else:
        remaining_leverage = np.maximum(1 - hat_matrix_diagonal, 0)

        denominator = np.sqrt(
            residual_variance * remaining_leverage
        )

        standardized_residuals = np.divide(
            residuals,
            denominator,
            out=np.full(n, np.nan),
            where=denominator > 0,
        )

        cooks_denominator = (
            parameter_count * remaining_leverage
        )

        cooks_distance = np.divide(
            standardized_residuals**2 * hat_matrix_diagonal,
            cooks_denominator,
            out=np.full(n, np.nan),
            where=cooks_denominator > 0,
        )

    shapiro = (
        stats.shapiro(residuals)
        if 3 <= n <= 5000 and not exact_fit
        else None
    )

    durbin_watson = (
        float(np.sum(np.diff(residuals)**2) / sse)
        if sse > 0
        else None
    )

    names = ["intercept"] + predictor_names
    coefficient_results = {}

    for index, name in enumerate(names):
        coefficient_results[name] = {
            "coefficient": float(coefficients[index]),
            "standard_error": (
                float(standard_errors[index])
                if np.isfinite(standard_errors[index])
                else None
            ),
            "t_statistic": (
                float(t_statistics[index])
                if np.isfinite(t_statistics[index])
                else None
            ),
            "p_value": (
                float(p_values[index])
                if np.isfinite(p_values[index])
                else None
            ),
            "significant": (
                bool(p_values[index] < alpha)
                if np.isfinite(p_values[index])
                else None
            ),
            "confidence_interval": (
                [
                    float(confidence_intervals[index, 0]),
                    float(confidence_intervals[index, 1]),
                ]
                if np.isfinite(confidence_intervals[index]).all()
                else None
            ),
        }

    valid_standardized = standardized_residuals[
        np.isfinite(standardized_residuals)
    ]
    valid_cooks = cooks_distance[np.isfinite(cooks_distance)]

    return {
        "method": "Ordinary least squares",
        "outcome": outcome,
        "predictors": predictor_names,
        "n_observations": n,
        "dropped_rows": dropped_rows,
        "alpha": alpha,
        "coefficients": coefficient_results,
        "model": {
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "f_statistic": model_f,
            "p_value": model_p,
            "significant": (
                bool(model_p < alpha)
                if model_p is not None
                else None
            ),
            "residual_standard_error": residual_se,
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
            "max_absolute_standardized_residual": (
                float(np.max(np.abs(valid_standardized)))
                if len(valid_standardized)
                else None
            ),
            "max_cooks_distance": (
                float(np.max(valid_cooks))
                if len(valid_cooks)
                else None
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

    if not predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Predictors must be non-empty and unique.")

    required = ["intercept"] + predictors

    missing_coefficients = [
        name for name in required if name not in coefficients
    ]

    if missing_coefficients:
        raise ValueError(f"Missing coefficients: {missing_coefficients}")

    missing_columns = [
        name for name in predictors if name not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing predictor columns: {missing_columns}")

    numeric = data[predictors].apply(pd.to_numeric, errors="coerce")

    if numeric.isna().any().any():
        raise ValueError("Prediction data contains missing or non-numeric values.")

    values = numeric.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Prediction data contains infinite values.")

    coefficient_values = {}

    for name in required:
        try:
            value = float(coefficients[name]["coefficient"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid coefficient for {name}.") from None

        if not np.isfinite(value):
            raise ValueError(f"Coefficient for {name} must be finite.")

        coefficient_values[name] = value

    result = np.full(len(data), coefficient_values["intercept"])

    for index, predictor in enumerate(predictors):
        result += values[:, index] * coefficient_values[predictor]

    return result
