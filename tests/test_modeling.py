"""Tests for modeling.py — business metrics correctness with known inputs."""

import numpy as np
import pytest

from config import AVG_CUSTOMER_LIFETIME_MONTHS
from modeling import _compute_business_metrics


class TestBusinessMetrics:
    """Verify business-metric calculations with hand-computed expected values.

    Known case:
        y_true:  [1, 1, 0, 0, 1]
        y_pred:  [1, 0, 0, 1, 1]
        charges: [80, 60, 40, 50, 100]

        Confusion: TN=1, FP=1, FN=1, TP=2
        TP rows (idx 0, 4): avg_charge = (80+100)/2 = 90
        FN row  (idx 1):    avg_charge = 60
    """

    @pytest.fixture
    def known_case(self):
        return {
            "y_true": np.array([1, 1, 0, 0, 1]),
            "y_pred": np.array([1, 0, 0, 1, 1]),
            "monthly_charges": np.array([80.0, 60.0, 40.0, 50.0, 100.0]),
        }

    def test_revenue_saved(self, known_case):
        result = _compute_business_metrics(**known_case)
        # TP=2, avg_charge_tp=90
        expected = 2 * 90.0 * AVG_CUSTOMER_LIFETIME_MONTHS
        assert result["Revenue_Saved"] == round(expected, 2)

    def test_revenue_leakage(self, known_case):
        result = _compute_business_metrics(**known_case)
        # FN=1, avg_charge_fn=60
        expected = 1 * 60.0 * AVG_CUSTOMER_LIFETIME_MONTHS
        assert result["Revenue_Leakage"] == round(expected, 2)

    def test_zero_true_positives(self):
        """When TP=0, Revenue_Saved should be 0."""
        result = _compute_business_metrics(
            y_true=np.array([0, 0, 1]),
            y_pred=np.array([0, 0, 0]),
            monthly_charges=np.array([50.0, 60.0, 70.0]),
        )
        assert result["Revenue_Saved"] == 0.0
