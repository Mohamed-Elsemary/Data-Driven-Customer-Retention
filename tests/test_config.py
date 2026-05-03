"""Tests for config.py — verifies that shared constants are sane."""

import pytest
from config import (
    RANDOM_STATE, TEST_SIZE, THRESHOLD,
    NUMERIC_COLS, BINARY_MAP, TERNARY_MAP,
    CORR_DROP_THRESHOLD, PERM_STRONG_THRESHOLD,
    LR_PARAM_GRID, DT_PARAM_GRID, RF_PARAM_GRID,
    XGB_PARAM_GRID, LGB_PARAM_GRID,
    RETENTION_CAMPAIGN_COST, AVG_CUSTOMER_LIFETIME_MONTHS,
)


class TestConstants:
    """Ensure config values stay within reasonable bounds."""

    def test_random_state_is_int(self):
        assert isinstance(RANDOM_STATE, int)

    def test_test_size_in_range(self):
        assert 0 < TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"

    def test_threshold_in_range(self):
        assert 0 < THRESHOLD < 1, "THRESHOLD must be between 0 and 1"

    def test_numeric_cols_non_empty(self):
        assert len(NUMERIC_COLS) > 0

    def test_binary_map_keys(self):
        assert set(BINARY_MAP.keys()) == {"Yes", "No"}

    def test_ternary_map_covers_service_values(self):
        assert "No internet service" in TERNARY_MAP
        assert "No phone service" in TERNARY_MAP

    def test_corr_drop_threshold_positive(self):
        assert 0 < CORR_DROP_THRESHOLD <= 1

    def test_perm_threshold_positive(self):
        assert PERM_STRONG_THRESHOLD > 0


class TestParamGrids:
    """Each grid must be a non-empty dict with list values."""

    @pytest.mark.parametrize("grid", [
        LR_PARAM_GRID, DT_PARAM_GRID, RF_PARAM_GRID,
        XGB_PARAM_GRID, LGB_PARAM_GRID,
    ])
    def test_grid_is_dict_of_lists(self, grid):
        assert isinstance(grid, dict)
        assert len(grid) > 0
        for key, val in grid.items():
            assert isinstance(val, list), f"{key} should be a list"


class TestBusinessConstants:
    def test_retention_cost_positive(self):
        assert RETENTION_CAMPAIGN_COST > 0

    def test_lifetime_months_positive(self):
        assert AVG_CUSTOMER_LIFETIME_MONTHS > 0
