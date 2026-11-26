"""Cross-validation utilities for BioDataLab."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def cross_validate(
    dataframe: pd.DataFrame,
    target: str,
    model_factory: Callable,
    metric: Callable,
    n_splits: int = 5,
    random_state: int | None = 42,
) -> dict[str, object]:
    """Evaluate a model using shuffled K-fold cross-validation."""

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

    if not isinstance(n_splits, int) or isinstance(n_splits, bool):
        raise TypeError("n_splits must be an integer.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if n_splits > len(dataframe):
        raise ValueError("n_splits cannot exceed the number of observations.")

    if not callable(model_factory):
        raise TypeError("model_factory must be callable.")

    if not callable(metric):
        raise TypeError("metric must be callable.")

    if dataframe.isna().any().any():
        raise ValueError("DataFrame must not contain missing values.")

    features = dataframe.drop(columns=[target])
    target_values = dataframe[target]

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in features.dtypes
    ):
        raise TypeError("Predictor columns must be numeric.")

    try:
        x = features.to_numpy(dtype=float)
        y = target_values.to_numpy()
    except (TypeError, ValueError) as exc:
        raise TypeError("Predictors must be numeric.") from exc

    if not np.isfinite(x).all():
        raise ValueError("Predictors must not contain infinite values.")

    if not np.isfinite(y.astype(float)).all():
        raise ValueError("Target must not contain infinite values.")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(dataframe))

    folds = np.array_split(indices, n_splits)
    scores = []

    for fold_index, validation_indices in enumerate(folds):
        training_indices = np.concatenate(
            [
                fold
                for i, fold in enumerate(folds)
                if i != fold_index
            ]
        )

        x_train = x[training_indices]
        y_train = y[training_indices]

        x_validation = x[validation_indices]
        y_validation = y[validation_indices]

        model = model_factory()

        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError(
                "Model must implement fit() and predict()."
            )

        model.fit(x_train, y_train)

        predictions = np.asarray(model.predict(x_validation))

        if predictions.shape != y_validation.shape:
            raise ValueError(
                "Model predictions must have the same shape as validation targets."
            )

        score = float(metric(y_validation, predictions))

        if not np.isfinite(score):
            raise ValueError("Metric returned a non-finite score.")

        scores.append(score)

    scores_array = np.asarray(scores, dtype=float)

    return {
        "fold_scores": scores_array,
        "mean_score": float(scores_array.mean()),
        "std_score": float(scores_array.std(ddof=1)),
        "n_splits": n_splits,
    }
