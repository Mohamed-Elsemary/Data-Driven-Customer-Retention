"""Tests for feature_engineering.py — encoding and engineered features."""

import pandas as pd
import pytest


class TestEncode:
    """Validate the encode() transformation."""

    def test_customer_id_dropped(self, cleaned_df):
        from feature_engineering import encode

        df = encode(cleaned_df)
        assert "customerID" not in df.columns

    def test_gender_is_numeric(self, cleaned_df):
        from feature_engineering import encode

        df = encode(cleaned_df)
        assert pd.api.types.is_numeric_dtype(df["gender"])

    def test_phone_service_is_numeric(self, cleaned_df):
        from feature_engineering import encode

        df = encode(cleaned_df)
        assert pd.api.types.is_numeric_dtype(df["PhoneService"])

    def test_no_nulls_after_encode(self, cleaned_df):
        from feature_engineering import encode

        df = encode(cleaned_df)
        assert df.isnull().sum().sum() == 0

    def test_ternary_cols_are_binary(self, cleaned_df):
        from config import TERNARY_COLS
        from feature_engineering import encode

        df = encode(cleaned_df)
        for col in TERNARY_COLS:
            assert set(df[col].unique()).issubset({0, 1}), f"{col} should only have values 0 or 1"

    def test_one_hot_columns_created(self, cleaned_df):
        from feature_engineering import encode

        df = encode(cleaned_df)
        # At least one InternetService one-hot column should exist
        internet_cols = [c for c in df.columns if c.startswith("InternetService_")]
        assert len(internet_cols) > 0, "InternetService one-hot columns missing"


class TestEngineeredFeatures:
    """Validate add_engineered_features()."""

    @pytest.fixture
    def encoded_df(self, cleaned_df):
        from feature_engineering import encode

        return encode(cleaned_df)

    def test_total_addons_created(self, encoded_df):
        from feature_engineering import add_engineered_features

        df = add_engineered_features(encoded_df)
        assert "_Total_AddOns-Services" in df.columns

    def test_cost_per_service_created(self, encoded_df):
        from feature_engineering import add_engineered_features

        df = add_engineered_features(encoded_df)
        assert "_Cost_Per_Service" in df.columns

    def test_is_autopay_created(self, encoded_df):
        from feature_engineering import add_engineered_features

        df = add_engineered_features(encoded_df)
        assert "_Is_AutoPay" in df.columns
        assert set(df["_Is_AutoPay"].unique()).issubset({0, 1})

    def test_loyalty_score_binary(self, encoded_df):
        from feature_engineering import add_engineered_features

        df = add_engineered_features(encoded_df)
        assert "_LoyaltyScore" in df.columns
        assert set(df["_LoyaltyScore"].unique()).issubset({0, 1})

    def test_high_friction_payment_created(self, encoded_df):
        from feature_engineering import add_engineered_features

        df = add_engineered_features(encoded_df)
        assert "_HighFriction_Payment" in df.columns

    def test_household_stability_range(self, encoded_df):
        from feature_engineering import add_engineered_features

        df = add_engineered_features(encoded_df)
        assert "_Household_Stability" in df.columns
        assert df["_Household_Stability"].min() >= 0
        assert df["_Household_Stability"].max() <= 2

    def test_helper_columns_dropped(self, encoded_df):
        from feature_engineering import add_engineered_features

        df = add_engineered_features(encoded_df)
        assert "CoreServices_Count" not in df.columns
        assert "True_Total_Services" not in df.columns


class TestSplitAndEncode:
    """Validate train/test split and frequency encoding."""

    @pytest.fixture
    def full_encoded_df(self, cleaned_df):
        from feature_engineering import add_engineered_features, encode

        df = encode(cleaned_df)
        df = add_engineered_features(df)
        return df

    def test_split_shapes(self, full_encoded_df, monkeypatch):
        import feature_engineering

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.5)

        X_train, X_test, y_train, y_test = feature_engineering.split_and_encode(full_encoded_df)
        assert len(X_train) + len(X_test) == len(full_encoded_df)
        assert len(y_train) + len(y_test) == len(full_encoded_df)

    def test_churn_not_in_features(self, full_encoded_df, monkeypatch):
        import feature_engineering

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.5)

        X_train, X_test, _, _ = feature_engineering.split_and_encode(full_encoded_df)
        assert "Churn" not in X_train.columns
        assert "Churn" not in X_test.columns
