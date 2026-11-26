import numpy as np
import pandas as pd
import pytest

from biodatalab.cross_validation import cross_validate


class DummyModel:
    def fit(self, x, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def create_dataframe():
    return pd.DataFrame(
        {
            "feature1": np.arange(20, dtype=float),
            "feature2": np.arange(20, dtype=float) * 2,
            "target": np.arange(20, dtype=float),
        }
    )


def test_cross_validation_basic():
    result = cross_validate(
        create_dataframe(),
        target="target",
        model_factory=DummyModel,
        metric=mean_absolute_error,
    )

    assert len(result["fold_scores"]) == 5
    assert result["n_splits"] == 5


def test_cross_validation_mean_score():
    result = cross_validate(
        create_dataframe(),
        target="target",
        model_factory=DummyModel,
        metric=mean_absolute_error,
    )

    assert result["mean_score"] >= 0


def test_cross_validation_std_score():
    result = cross_validate(
        create_dataframe(),
        target="target",
        model_factory=DummyModel,
        metric=mean_absolute_error,
    )

    assert result["std_score"] >= 0


def test_cross_validation_custom_splits():
    result = cross_validate(
        create_dataframe(),
        target="target",
        model_factory=DummyModel,
        metric=mean_absolute_error,
        n_splits=4,
    )

    assert len(result["fold_scores"]) == 4


def test_cross_validation_reproducibility():
    first = cross_validate(
        create_dataframe(),
        target="target",
        model_factory=DummyModel,
        metric=mean_absolute_error,
        random_state=42,
    )

    second = cross_validate(
        create_dataframe(),
        target="target",
        model_factory=DummyModel,
        metric=mean_absolute_error,
        random_state=42,
    )

    assert np.array_equal(
        first["fold_scores"],
        second["fold_scores"],
    )


def test_cross_validation_missing_target():
    with pytest.raises(ValueError):
        cross_validate(
            create_dataframe(),
            target="missing",
            model_factory=DummyModel,
            metric=mean_absolute_error,
        )


def test_cross_validation_invalid_splits():
    with pytest.raises(ValueError):
        cross_validate(
            create_dataframe(),
            target="target",
            model_factory=DummyModel,
            metric=mean_absolute_error,
            n_splits=1,
        )


def test_cross_validation_too_many_splits():
    with pytest.raises(ValueError):
        cross_validate(
            create_dataframe(),
            target="target",
            model_factory=DummyModel,
            metric=mean_absolute_error,
            n_splits=25,
        )


def test_cross_validation_missing_values():
    dataframe = create_dataframe()
    dataframe.loc[0, "feature1"] = np.nan

    with pytest.raises(ValueError):
        cross_validate(
            dataframe,
            target="target",
            model_factory=DummyModel,
            metric=mean_absolute_error,
        )


def test_cross_validation_non_numeric_features():
    dataframe = create_dataframe()
    dataframe["category"] = ["a"] * 20

    with pytest.raises(TypeError):
        cross_validate(
            dataframe,
            target="target",
            model_factory=DummyModel,
            metric=mean_absolute_error,
        )


def test_cross_validation_invalid_model():
    with pytest.raises(TypeError):
        cross_validate(
            create_dataframe(),
            target="target",
            model_factory=lambda: object(),
            metric=mean_absolute_error,
        )


def test_cross_validation_invalid_metric():
    with pytest.raises(TypeError):
        cross_validate(
            create_dataframe(),
            target="target",
            model_factory=DummyModel,
            metric="invalid",
        )


def test_cross_validation_duplicate_columns():
    dataframe = pd.DataFrame(
        [[1, 2, 0], [3, 4, 1], [5, 6, 0]],
        columns=["A", "A", "target"],
    )

    with pytest.raises(ValueError):
        cross_validate(
            dataframe,
            target="target",
            model_factory=DummyModel,
            metric=mean_absolute_error,
        )
