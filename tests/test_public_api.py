
"""Tests for BioDataLab public API."""

import pandas as pd

from biodatalab import validate_dataframe


def test_validate_dataframe_public_import():
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    result = validate_dataframe(dataframe)

    assert result["valid"] is True


def test_public_api_validation_rules():
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    result = validate_dataframe(
        dataframe,
        {
            "columns": {
                "value": {
                    "type": "numeric",
                    "min": 0,
                    "max": 10,
                }
            }
        },
    )

    assert result["valid"] is True
