
"""Tests for BioDataLab custom exceptions."""

import pytest

from biodatalab.exceptions import (
    BioDataLabError,
    DataLoadingError,
    DataProcessingError,
    DataValidationError,
    StatisticalAnalysisError,
)


def test_base_exception():
    with pytest.raises(BioDataLabError):
        raise BioDataLabError("Test error")


def test_validation_error():
    with pytest.raises(ValueError):
        raise DataValidationError("Invalid data")


def test_processing_error():
    with pytest.raises(BioDataLabError):
        raise DataProcessingError("Processing failed")


def test_loading_error():
    with pytest.raises(BioDataLabError):
        raise DataLoadingError("Loading failed")


def test_statistical_error():
    with pytest.raises(BioDataLabError):
        raise StatisticalAnalysisError("Analysis failed")


def test_custom_exceptions_inherit_from_base():
    exceptions = [
        DataValidationError,
        DataProcessingError,
        DataLoadingError,
        StatisticalAnalysisError,
    ]

    for exception in exceptions:
        assert issubclass(exception, BioDataLabError)


def test_exceptions_preserve_messages():
    message = "Dataset contains invalid values."

    error = DataValidationError(message)

    assert str(error) == message
