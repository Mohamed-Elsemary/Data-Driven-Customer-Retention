"""Tests for modeling.py — business metrics correctness with known inputs."""

import numpy as np
import pytest

from config import AVG_CUSTOMER_LIFETIME_MONTHS, RETENTION_CAMPAIGN_COST
from modeling import _compute_business_metrics, _sanitize_params


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

    def test_cost_per_detection(self, known_case):
        result = _compute_business_metrics(**known_case)
        # campaign_cost = (TP+FP) * cost = 3 * 50 = 150; per TP = 150/2
        expected = (2 + 1) * RETENTION_CAMPAIGN_COST / 2
        assert result["Cost_Per_Detection"] == round(expected, 2)

    def test_net_retention_value(self, known_case):
        """Net = Revenue_Saved - campaign_cost."""
        result = _compute_business_metrics(**known_case)
        revenue_saved = 2 * 90.0 * AVG_CUSTOMER_LIFETIME_MONTHS
        campaign_cost = (2 + 1) * RETENTION_CAMPAIGN_COST
        expected = revenue_saved - campaign_cost
        assert result["Net_Retention_Value"] == round(expected, 2)

    def test_zero_true_positives(self):
        """When TP=0, cost_per_detection should be 0 (guarded division)."""
        result = _compute_business_metrics(
            y_true=np.array([0, 0, 1]),
            y_pred=np.array([0, 0, 0]),
            monthly_charges=np.array([50.0, 60.0, 70.0]),
        )
        assert result["Cost_Per_Detection"] == 0.0
        assert result["Revenue_Saved"] == 0.0
