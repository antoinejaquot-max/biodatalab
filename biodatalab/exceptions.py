
"""Custom exceptions for BioDataLab."""


class BioDataLabError(Exception):
    """Base exception for all BioDataLab errors."""


class DataValidationError(BioDataLabError, ValueError):
    """Raised when a dataset fails validation."""


class DataProcessingError(BioDataLabError):
    """Raised when a data processing operation fails."""


class DataLoadingError(BioDataLabError):
    """Raised when loading a dataset fails."""


class StatisticalAnalysisError(BioDataLabError):
    """Raised when a statistical analysis cannot be performed."""
