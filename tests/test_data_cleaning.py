"""Tests for data_cleaning.py — validates the clean() pipeline."""

import pandas as pd
import numpy as np
import pytest


class TestClean:
    """Verify that clean() produces the expected transformations."""

    def test_returns_dataframe(self, cleaned_df):
        assert isinstance(cleaned_df, pd.DataFrame)

    def test_does_not_modify_row_count(self, raw_df, cleaned_df):
        assert len(cleaned_df) == len(raw_df)

    def test_total_charges_numeric(self, cleaned_df):
        assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])

    def test_total_charges_no_nans(self, cleaned_df):
        assert cleaned_df["TotalCharges"].isna().sum() == 0

    def test_churn_is_binary(self, cleaned_df):
        assert set(cleaned_df["Churn"].dropna().unique()).issubset({0, 1})

    def test_partner_is_binary(self, cleaned_df):
        assert set(cleaned_df["Partner"].dropna().unique()).issubset({0, 1})

    def test_dependents_is_binary(self, cleaned_df):
        assert set(cleaned_df["Dependents"].dropna().unique()).issubset({0, 1})

    def test_clean_is_non_destructive(self, raw_df):
        """clean() must not mutate the original DataFrame."""
        from data_cleaning import clean

        original_copy = raw_df.copy()
        _ = clean(raw_df)
        pd.testing.assert_frame_equal(raw_df, original_copy)

    def test_string_columns_converted(self, cleaned_df):
        """Columns listed in CAT_COLS_TO_STRING must be string dtype."""
        from config import CAT_COLS_TO_STRING

        for col in CAT_COLS_TO_STRING:
            if col in cleaned_df.columns:
                assert cleaned_df[col].dtype == "string", (
                    f"{col} should be 'string' dtype"
                )
