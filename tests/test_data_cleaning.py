"""Tests for data_cleaning.py — validates the clean() pipeline."""

import pandas as pd
import pytest

from config import CAT_COLS_TO_STRING
from data_cleaning import clean


class TestClean:
    """Verify that clean() produces correct transformations."""

    def test_clean_is_non_destructive(self, raw_df):
        """clean() must not mutate the original DataFrame."""
        original = raw_df.copy()
        _ = clean(raw_df)
        pd.testing.assert_frame_equal(raw_df, original)

    @pytest.mark.parametrize("col", ["Partner", "Dependents", "Churn"])
    def test_binary_columns_mapped_correctly(self, cleaned_df, col):
        """Yes/No columns must be mapped to 1/0 with no other values."""
        assert set(cleaned_df[col].dropna().unique()).issubset({0, 1})

    def test_string_columns_converted(self, cleaned_df):
        """Columns in CAT_COLS_TO_STRING must have 'string' dtype."""
        for col in CAT_COLS_TO_STRING:
            if col in cleaned_df.columns:
                assert cleaned_df[col].dtype == "string", f"{col} should be 'string' dtype"

    def test_whitespace_handling(self):
        """Whitespace-padded values should still map correctly."""
        df = pd.DataFrame(
            {
                "customerID": ["X1"],
                "gender": ["Male"],
                "SeniorCitizen": [0],
                "Partner": [" Yes "],
                "Dependents": [" No "],
                "tenure": [10],
                "PhoneService": ["Yes"],
                "MultipleLines": ["No"],
                "InternetService": ["DSL"],
                "OnlineSecurity": ["Yes"],
                "OnlineBackup": ["No"],
                "DeviceProtection": ["No"],
                "TechSupport": ["Yes"],
                "StreamingTV": ["No"],
                "StreamingMovies": ["Yes"],
                "Contract": ["One year"],
                "PaperlessBilling": ["Yes"],
                "PaymentMethod": ["Electronic check"],
                "MonthlyCharges": [50.0],
                "TotalCharges": ["500"],
                "Churn": [" Yes "],
            }
        )
        result = clean(df)
        assert result["Partner"].iloc[0] == 1
        assert result["Dependents"].iloc[0] == 0
        assert result["Churn"].iloc[0] == 1
