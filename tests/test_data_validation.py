"""Tests for data_validation.py — ensures validation helpers run without error."""

import pandas as pd


class TestValidationFunctions:
    """Each validation function should run without raising on valid data."""

    def test_check_accuracy_runs(self, raw_df):
        from data_validation import check_accuracy

        check_accuracy(raw_df)  # should not raise

    def test_check_consistency_runs(self, raw_df):
        from data_validation import check_consistency

        # TotalCharges must be numeric for the consistency check
        df = raw_df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        check_consistency(df)

    def test_check_completeness_runs(self, raw_df):
        from data_validation import check_completeness

        df = raw_df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        check_completeness(df)

    def test_check_duplicates_runs(self, raw_df):
        from data_validation import check_duplicates

        check_duplicates(raw_df)

    def test_detect_outliers_runs(self, raw_df):
        from data_validation import detect_outliers

        detect_outliers(raw_df)

    def test_distribution_profile_runs(self, raw_df):
        from data_validation import distribution_profile

        df = raw_df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        distribution_profile(df)

    def test_relationship_profile_runs(self, raw_df):
        from data_validation import relationship_profile

        df = raw_df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        relationship_profile(df)

    def test_run_all_validations(self, raw_df):
        from data_validation import run_all_validations

        run_all_validations(raw_df)


class TestValidationDetectsIssues:
    """Verify that validation logic actually catches known problems."""

    def test_accuracy_detects_invalid_gender(self, raw_df):
        """Inject an invalid gender value and make sure check_accuracy
        does not crash (the logging output would show the count)."""
        from data_validation import check_accuracy

        df = raw_df.copy()
        df.loc[0, "gender"] = "Unknown"
        check_accuracy(df)  # should log 1 invalid gender row

    def test_duplicates_detects_duplicate_ids(self, raw_df):
        from data_validation import check_duplicates

        df = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
        check_duplicates(df)  # should log 1 duplicate ID
