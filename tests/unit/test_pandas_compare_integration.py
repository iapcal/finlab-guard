"""Test the pandas.compare() integration in DataValidator."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from finlab_guard.cache.manager import CacheManager
from finlab_guard.cache.validator import DataValidator


class TestPandasCompareIntegration:
    """Test the new pandas.compare() based implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for cache manager."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_dir):
        """Create cache manager instance."""
        return CacheManager(temp_dir, {"compression": "snappy"})

    @pytest.fixture
    def validator(self):
        """Create DataValidator with default tolerance."""
        return DataValidator(tolerance=1e-12)

    def test_bool_dtype_handling(self, validator, cache_manager):
        """Test that bool dtypes don't cause subtraction errors."""
        # Create bool data that would previously fail
        original_data = pd.DataFrame(
            {
                "is_active": [True, False, True, False],
                "has_data": [False, True, False, True],
            },
            index=pd.date_range("2023-01-01", periods=4),
        )

        # Save original data
        cache_manager.save_data("test_bool", original_data, pd.Timestamp.now())

        # Create identical data (should not trigger changes)
        identical_data = original_data.copy()

        modifications, additions = validator.detect_changes_detailed(
            "test_bool", identical_data, cache_manager
        )

        # Should detect no changes
        assert len(modifications) == 0
        assert len(additions) == 0

    def test_bool_dtype_actual_changes(self, validator, cache_manager):
        """Test that actual bool changes are detected."""
        # Create bool data
        original_data = pd.DataFrame(
            {
                "is_active": [True, False, True, False],
            },
            index=pd.date_range("2023-01-01", periods=4),
        )

        # Save original data
        cache_manager.save_data("test_bool_change", original_data, pd.Timestamp.now())

        # Create modified data
        modified_data = original_data.copy()
        modified_data.iloc[1, 0] = True  # Change False to True

        modifications, additions = validator.detect_changes_detailed(
            "test_bool_change", modified_data, cache_manager
        )

        # Should detect exactly one change
        assert len(modifications) == 1
        assert len(additions) == 0
        assert not modifications[0].old_value
        assert modifications[0].new_value

    def test_mixed_dtype_tolerance(self, validator, cache_manager):
        """Test tolerance handling with mixed data types."""
        # Create mixed data
        original_data = pd.DataFrame(
            {
                "price": [100.1234567890123, 200.9876543210987],  # float64
                "count": [10, 20],  # int64
                "active": [True, False],  # bool
                "symbol": ["AAPL", "GOOGL"],  # object
            },
            index=["A", "B"],
        )

        cache_manager.save_data("test_mixed", original_data, pd.Timestamp.now())

        # Create data with tiny float precision differences
        modified_data = original_data.copy()
        modified_data.iloc[0, 0] += 1e-14  # Tiny float change (within tolerance)
        modified_data.iloc[1, 1] = 21  # Integer change (should be detected)

        modifications, additions = validator.detect_changes_detailed(
            "test_mixed", modified_data, cache_manager
        )

        # Should only detect the integer change, not the float precision difference
        assert len(modifications) == 1
        assert len(additions) == 0
        assert modifications[0].old_value == 20
        assert modifications[0].new_value == 21

    def test_adj_close_simulation(self, validator, cache_manager):
        """Simulate the etl:adj_close scenario that caused original issues."""
        # Simulate realistic adjusted close data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        stocks = ["2330", "2317", "2454"]

        # Create realistic price data
        adj_close_data = {}
        for stock in stocks:
            base_price = np.random.uniform(100, 500)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            adj_close_data[stock] = prices

        original_df = pd.DataFrame(adj_close_data, index=dates, dtype="float64")

        # Save original
        cache_manager.save_data("etl:adj_close", original_df, pd.Timestamp.now())

        # Simulate precision loss from parquet save/load
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save and reload to introduce precision artifacts
            original_df.to_parquet(tmp_path)
            reloaded_df = pd.read_parquet(tmp_path)

            modifications, additions = validator.detect_changes_detailed(
                "etl:adj_close", reloaded_df, cache_manager
            )

            # Should detect very few or no changes due to tolerance
            assert (
                len(modifications) <= 2
            )  # Allow for occasional genuine precision loss
            assert len(additions) == 0

            print(
                f"Detected {len(modifications)} modifications out of {original_df.size} values"
            )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_fallback_mechanism(self, cache_manager):
        """Test that fallback works when pandas.compare() fails."""
        # Create a validator that we can manipulate
        validator = DataValidator(tolerance=1e-12)

        # Create test data
        original_data = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=["A", "B", "C"])

        cache_manager.save_data("test_fallback", original_data, pd.Timestamp.now())

        # Create modified data
        modified_data = original_data.copy()
        modified_data.iloc[1, 0] = 2.5

        # Patch pandas.DataFrame.compare to raise an exception
        original_compare = pd.DataFrame.compare

        def failing_compare(*args, **kwargs):
            raise ValueError("Simulated compare() failure")

        try:
            pd.DataFrame.compare = failing_compare

            # Should still work via fallback
            modifications, additions = validator.detect_changes_detailed(
                "test_fallback", modified_data, cache_manager
            )

            assert len(modifications) == 1
            assert modifications[0].old_value == 2.0
            assert modifications[0].new_value == 2.5

        finally:
            # Restore original method
            pd.DataFrame.compare = original_compare

    def test_nan_handling_with_compare(self, validator, cache_manager):
        """Test NaN handling in the new compare-based implementation."""
        # Create data with NaN values
        original_data = pd.DataFrame(
            {"value": [1.0, np.nan, 3.0, np.nan]}, index=["A", "B", "C", "D"]
        )

        cache_manager.save_data("test_nan", original_data, pd.Timestamp.now())

        # Test various NaN scenarios
        test_cases = [
            # Case 1: No changes (identical NaN pattern)
            (original_data.copy(), 0),
            # Case 2: Change NaN to value
            (original_data.copy().fillna(2.0), 2),
            # Case 3: Change value to NaN
            (
                pd.DataFrame(
                    {"value": [np.nan, np.nan, np.nan, np.nan]},
                    index=["A", "B", "C", "D"],
                ),
                2,
            ),
        ]

        for i, (test_data, expected_changes) in enumerate(test_cases):
            # Use unique key for each test case to avoid cache interference
            test_key = f"test_nan_case_{i}"
            cache_manager.save_data(test_key, original_data, pd.Timestamp.now())

            modifications, additions = validator.detect_changes_detailed(
                test_key, test_data, cache_manager
            )

            assert len(modifications) == expected_changes, (
                f"Case {i + 1} failed: expected {expected_changes}, got {len(modifications)}"
            )
            assert len(additions) == 0
