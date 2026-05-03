"""Tests for modeling.py — business metrics and utility helpers."""

import numpy as np
import pandas as pd
import pytest


class TestSanitizeParams:
    """Verify _sanitize_params produces MLflow-safe strings."""

    def test_converts_values_to_strings(self):
        from modeling import _sanitize_params

        result = _sanitize_params({"C": 0.1, "penalty": "l2"})
        assert all(isinstance(v, str) for v in result.values())

    def test_truncates_long_values(self):
        from modeling import _sanitize_params

        long_val = "x" * 500
        result = _sanitize_params({"key": long_val})
        assert len(result["key"]) <= 250


class TestComputeBusinessMetrics:
    """Verify business-metric calculations with known inputs."""

    @pytest.fixture
    def known_case(self):
        """
        y_true:  [1, 1, 0, 0, 1]
        y_pred:  [1, 0, 0, 1, 1]
        charges: [80, 60, 40, 50, 100]

        Confusion: TN=1, FP=1, FN=1, TP=2
        TP mask rows: indices 0, 4  → avg charge = (80+100)/2 = 90
        FN mask row:  index 1       → avg charge = 60
        """
        return {
            "y_true": np.array([1, 1, 0, 0, 1]),
            "y_pred": np.array([1, 0, 0, 1, 1]),
            "monthly_charges": np.array([80.0, 60.0, 40.0, 50.0, 100.0]),
        }

    def test_returns_all_keys(self, known_case):
        from modeling import _compute_business_metrics

        result = _compute_business_metrics(**known_case)
        expected_keys = {
            "Revenue_Saved",
            "Net_Retention_Value",
            "Cost_Per_Detection",
            "Revenue_Leakage",
        }
        assert set(result.keys()) == expected_keys

    def test_revenue_saved(self, known_case):
        from config import AVG_CUSTOMER_LIFETIME_MONTHS
        from modeling import _compute_business_metrics

        result = _compute_business_metrics(**known_case)
        # TP=2, avg_charge_tp=90
        expected = 2 * 90.0 * AVG_CUSTOMER_LIFETIME_MONTHS
        assert result["Revenue_Saved"] == round(expected, 2)

    def test_revenue_leakage(self, known_case):
        from config import AVG_CUSTOMER_LIFETIME_MONTHS
        from modeling import _compute_business_metrics

        result = _compute_business_metrics(**known_case)
        # FN=1, avg_charge_fn=60
        expected = 1 * 60.0 * AVG_CUSTOMER_LIFETIME_MONTHS
        assert result["Revenue_Leakage"] == round(expected, 2)

    def test_cost_per_detection(self, known_case):
        from config import RETENTION_CAMPAIGN_COST
        from modeling import _compute_business_metrics

        result = _compute_business_metrics(**known_case)
        # campaign_cost = (TP + FP) * cost = (2+1)*50 = 150
        # cost_per_detected = 150 / 2 = 75
        expected = (2 + 1) * RETENTION_CAMPAIGN_COST / 2
        assert result["Cost_Per_Detection"] == round(expected, 2)


class TestMakeSkf:
    """Verify the StratifiedKFold factory."""

    def test_returns_stratified_kfold(self):
        from sklearn.model_selection import StratifiedKFold

        from modeling import _make_skf

        skf = _make_skf()
        assert isinstance(skf, StratifiedKFold)
        assert skf.n_splits == 5
