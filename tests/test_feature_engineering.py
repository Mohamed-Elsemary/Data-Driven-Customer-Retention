"""Tests for feature_engineering.py — encoding, derived features, and split contracts."""

import numpy as np
import pandas as pd

from config import ADDON_COLS, TERNARY_COLS
from feature_engineering import encode, split_and_encode
import feature_engineering

# ═══════════════════════════════════════════════════════════════
#  ENCODING CORRECTNESS
# ═══════════════════════════════════════════════════════════════


class TestEncode:
    """Verify encode() produces correct values, not just correct types."""

    def test_gender_mapped_correctly(self, cleaned_df):
        """Male → 1, Female → 0."""
        df = encode(cleaned_df)
        raw_male_mask = cleaned_df["gender"].astype(str) == "Male"
        assert (df.loc[raw_male_mask, "gender"] == 1).all()
        assert (df.loc[~raw_male_mask, "gender"] == 0).all()

    def test_ternary_cols_collapse_to_binary(self, cleaned_df):
        """'No internet/phone service' must collapse to 0."""
        df = encode(cleaned_df)
        for col in TERNARY_COLS:
            assert set(df[col].unique()).issubset({0, 1}), f"{col} has values beyond 0/1"

    def test_customer_id_dropped(self, encoded_df):
        assert "customerID" not in encoded_df.columns

    def test_no_nulls_after_encode(self, encoded_df):
        assert encoded_df.isnull().sum().sum() == 0


# ═══════════════════════════════════════════════════════════════
#  DERIVED FEATURE FORMULAS
# ═══════════════════════════════════════════════════════════════


class TestDerivedFeatures:
    """Verify the math behind each engineered feature."""

    def test_total_addons_sum(self, encoded_df, featured_df):
        """_Total_AddOns-Services = row-wise sum of ADDON_COLS."""
        expected = encoded_df[ADDON_COLS].sum(axis=1).values
        np.testing.assert_array_equal(featured_df["_Total_AddOns-Services"].values, expected)

    def test_cost_per_service_formula(self, encoded_df, featured_df):
        """_Cost_Per_Service = MonthlyCharges / (core + addons)."""
        core = encoded_df["PhoneService"] + (1 - encoded_df["InternetService_No"])
        addons = encoded_df[ADDON_COLS].sum(axis=1)
        expected = encoded_df["MonthlyCharges"] / (core + addons)
        pd.testing.assert_series_equal(
            featured_df["_Cost_Per_Service"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_household_stability_formula(self, encoded_df, featured_df):
        """_Household_Stability = Partner + Dependents."""
        expected = encoded_df["Partner"] + encoded_df["Dependents"]
        pd.testing.assert_series_equal(
            featured_df["_Household_Stability"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_high_friction_formula(self, encoded_df, featured_df):
        """_HighFriction_Payment = PaperlessBilling=1 AND ElectronicCheck=1."""
        expected = (
            (encoded_df["PaperlessBilling"] == 1)
            & (encoded_df["PaymentMethod_Electronic check"] == 1)
        ).astype(int)
        pd.testing.assert_series_equal(
            featured_df["_HighFriction_Payment"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_helper_columns_dropped(self, featured_df):
        """Temporary columns must not survive into the output."""
        assert "CoreServices_Count" not in featured_df.columns
        assert "True_Total_Services" not in featured_df.columns


# ═══════════════════════════════════════════════════════════════
#  SPLIT & LEAKAGE
# ═══════════════════════════════════════════════════════════════


class TestSplitContracts:
    """Verify split_and_encode() preserves data integrity."""

    def test_target_not_in_features(self, featured_df, monkeypatch):
        """Churn must never appear in X_train or X_test."""
        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)
        X_train, X_test, _, _ = split_and_encode(featured_df)
        assert "Churn" not in X_train.columns
        assert "Churn" not in X_test.columns

    def test_train_test_columns_aligned(self, featured_df, monkeypatch):
        """X_train and X_test must have identical column sets."""
        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)
        X_train, X_test, _, _ = split_and_encode(featured_df)
        assert list(X_train.columns) == list(X_test.columns)

    def test_no_index_overlap(self, featured_df, monkeypatch):
        """Train and test indices must not overlap."""
        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)
        X_train, X_test, _, _ = split_and_encode(featured_df)
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_frequency_encoding_uses_train_only(self, featured_df, monkeypatch):
        """Contract freq values in X_test must come from X_train distribution."""
        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)
        X_train, X_test, _, _ = split_and_encode(featured_df)
        train_vals = set(X_train["Contract"].unique())
        for val in X_test["Contract"].unique():
            assert val in train_vals, f"X_test Contract value {val} not in X_train freq map"
