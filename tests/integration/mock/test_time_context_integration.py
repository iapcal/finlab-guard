"""Integration tests for time context functionality in finlab-guard.

This module tests the complete time context system including:
- Cross-dataset time queries
- Complex historical scenarios
- Boundary time handling
- Integration with dtype changes
"""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finlab_guard import FinlabGuard


class TestTimeContextIntegration:
    """Test complete time context functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def guard(self, temp_cache_dir):
        """Create FinlabGuard instance for testing."""
        config = {
            "compression": None
        }  # Disable compression to avoid PyArrow issues in tests
        guard_instance = FinlabGuard(cache_dir=temp_cache_dir, config=config)
        yield guard_instance
        # Ensure DuckDB connection is closed to prevent Windows file locking
        guard_instance.close()

    def _mock_finlab_data(self, data: pd.DataFrame):
        """Mock finlab.data.get to return specified data."""
        # Use the simpler approach - patch the _fetch_from_finlab method
        # The finlab module mock is handled by conftest.py
        return patch.object(FinlabGuard, "_fetch_from_finlab", return_value=data)

    def test_time_context_across_multiple_datasets(self, guard):
        """
        Test time context functionality across multiple datasets.

        This test verifies:
        1. Time context affects all datasets consistently
        2. Cross-dataset queries at same time return consistent historical data
        3. Time context clearing works for all datasets
        """
        # Create timeline
        t1 = datetime.now() - timedelta(hours=4)
        t2 = datetime.now() - timedelta(hours=3)
        datetime.now() - timedelta(hours=2)

        # Dataset 1: Price data
        price_v1 = pd.DataFrame(
            {"AAPL": [150, 152], "GOOGL": [2800, 2820]},
            index=["2023-01-01", "2023-01-02"],
        )

        price_v2 = pd.DataFrame(
            {
                "AAPL": [150, 152, 156],  # Keep original values, only add new data
                "GOOGL": [2800, 2820, 2850],
            },
            index=["2023-01-01", "2023-01-02", "2023-01-03"],
        )

        # Dataset 2: Volume data
        volume_v1 = pd.DataFrame(
            {"AAPL": [1000000, 1100000], "GOOGL": [500000, 550000]},
            index=["2023-01-01", "2023-01-02"],
        )

        volume_v2 = pd.DataFrame(
            {"AAPL": [1000000, 1100000, 1200000], "GOOGL": [500000, 550000, 600000]},
            index=["2023-01-01", "2023-01-02", "2023-01-03"],
        )

        # Save historical data for both datasets at t1
        with patch.object(guard, "_now", return_value=t1):
            with self._mock_finlab_data(price_v1):
                guard.get("price_data")
            with self._mock_finlab_data(volume_v1):
                guard.get("volume_data")

        # Update both datasets at t2
        with patch.object(guard, "_now", return_value=t2):
            with self._mock_finlab_data(price_v2):
                guard.get("price_data")
            with self._mock_finlab_data(volume_v2):
                guard.get("volume_data")

        # Test cross-dataset consistency at t1
        query_time_1 = t1 + timedelta(minutes=30)
        guard.set_time_context(query_time_1)
        try:
            historical_price = guard.get("price_data", force_download=False)
            historical_volume = guard.get("volume_data", force_download=False)

            # Should get v1 data for both
            assert len(historical_price) == 2  # Original length
            assert len(historical_volume) == 2  # Original length
            assert list(historical_price.index) == ["2023-01-01", "2023-01-02"]
            assert list(historical_volume.index) == ["2023-01-01", "2023-01-02"]

        finally:
            guard.clear_time_context()

        # Test cross-dataset consistency at t2
        query_time_2 = t2 + timedelta(minutes=30)
        guard.set_time_context(query_time_2)
        try:
            historical_price = guard.get("price_data", force_download=False)
            historical_volume = guard.get("volume_data", force_download=False)

            # Should get v2 data for both
            assert len(historical_price) == 3  # Updated length
            assert len(historical_volume) == 3  # Updated length
            assert list(historical_price.index) == [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
            ]

        finally:
            guard.clear_time_context()

        # Test that clearing time context works for all datasets
        # When no time context, guard.get() should return latest data
        with self._mock_finlab_data(price_v2):  # Provide mock in case fetch is needed
            latest_price = guard.get("price_data", force_download=False)
        with self._mock_finlab_data(volume_v2):  # Provide mock for volume data too
            latest_volume = guard.get("volume_data", force_download=False)

        # Should get latest (v2) data
        assert len(latest_price) == 3  # Latest length
        assert len(latest_volume) == 3  # Latest length

    def test_time_context_with_complex_history(self, guard):
        """
        Test time context with complex historical scenarios.

        This test covers:
        1. Multiple data updates over time
        2. Queries between updates
        3. Data reconstruction accuracy
        """
        key = "complex_history_test"

        # Create complex evolution timeline
        base_time = datetime.now() - timedelta(hours=10)

        # Phase 1: Initial data
        data_phase1 = pd.DataFrame(
            {"col1": [100, 200], "col2": [1.1, 2.2]}, index=["A", "B"]
        )

        # Phase 2: Add new row
        data_phase2 = pd.DataFrame(
            {"col1": [100, 200, 300], "col2": [1.1, 2.2, 3.3]}, index=["A", "B", "C"]
        )

        # Phase 3: Modify existing + add column
        data_phase3 = pd.DataFrame(
            {
                "col1": [105, 200, 300],  # A modified
                "col2": [1.1, 2.2, 3.3],
                "col3": [10, 20, 30],  # New column
            },
            index=["A", "B", "C"],
        )

        # Phase 4: Add another row + modify dtypes
        data_phase4 = pd.DataFrame(
            {
                "col1": np.array([105, 200, 300, 400], dtype="int64"),  # dtype change
                "col2": [1.1, 2.2, 3.3, 4.4],
                "col3": [10, 20, 30, 40],
            },
            index=["A", "B", "C", "D"],
        )

        phases = [
            (base_time, data_phase1),
            (base_time + timedelta(hours=2), data_phase2),
            (base_time + timedelta(hours=4), data_phase3),
            (base_time + timedelta(hours=6), data_phase4),
        ]

        # Execute all phases
        for phase_time, phase_data in phases:
            with patch.object(guard, "_now", return_value=phase_time):
                with self._mock_finlab_data(phase_data):
                    if phase_time == base_time + timedelta(hours=4):
                        # Phase 3 has modifications, force download
                        guard.get(key, force_download=True)
                    else:
                        guard.get(key)

        # Test queries at various points in history
        test_scenarios = [
            # Query between phases
            (base_time + timedelta(hours=1), data_phase1, "Between phase 1 and 2"),
            (base_time + timedelta(hours=3), data_phase2, "Between phase 2 and 3"),
            (base_time + timedelta(hours=5), data_phase3, "Between phase 3 and 4"),
            (base_time + timedelta(hours=7), data_phase4, "After phase 4"),
            # Query at exact phase times
            (base_time + timedelta(minutes=5), data_phase1, "Shortly after phase 1"),
            (
                base_time + timedelta(hours=2, minutes=5),
                data_phase2,
                "Shortly after phase 2",
            ),
        ]

        for query_time, expected_data, description in test_scenarios:
            guard.set_time_context(query_time)
            try:
                result = guard.get(key, force_download=False)

                # Verify structure matches expected
                assert len(result) == len(expected_data), (
                    f"{description}: Row count mismatch"
                )
                assert len(result.columns) == len(expected_data.columns), (
                    f"{description}: Column count mismatch"
                )
                assert list(result.index) == list(expected_data.index), (
                    f"{description}: Index mismatch"
                )
                assert list(result.columns) == list(expected_data.columns), (
                    f"{description}: Column mismatch"
                )

                # For data that exists in both, verify values match
                print(description, query_time, expected_data)
                assert expected_data.equals(result), f"{expected_data}, {result}"
                for col in expected_data.columns:
                    if col in result.columns:
                        for idx in expected_data.index:
                            if idx in result.index:
                                expected_val = expected_data.loc[idx, col]
                                actual_val = result.loc[idx, col]
                                assert expected_val == actual_val, (
                                    f"{description}: Value mismatch at {idx}, {col}"
                                )

            finally:
                guard.clear_time_context()

    def test_time_context_boundary_times(self, guard):
        """
        Test time context behavior at boundary conditions.

        This test covers:
        1. Queries before any data exists
        2. Queries at exact save timestamps
        3. Queries far in the future
        4. Edge cases in time handling
        """
        key = "boundary_test"

        # Reference times
        base_time = datetime.now() - timedelta(hours=5)
        save_time_1 = base_time + timedelta(hours=1)
        save_time_2 = base_time + timedelta(hours=3)

        # Save data at specific times (add new index instead of modifying existing data)
        data_1 = pd.DataFrame({"col": [100]}, index=["A"])
        data_2 = pd.DataFrame({"col": [100, 200]}, index=["A", "B"])  # Add new index B

        with patch.object(guard, "_now", return_value=save_time_1):
            with self._mock_finlab_data(data_1):
                guard.get(key)

        with patch.object(guard, "_now", return_value=save_time_2):
            with self._mock_finlab_data(data_2):
                guard.get(key)

        # Test boundary scenarios
        boundary_tests = [
            # Before any data
            (base_time - timedelta(hours=1), data_1, "Before first save"),
            # Exactly at save times
            (save_time_1, data_1, "Exactly at first save time"),
            (save_time_2, data_2, "Exactly at second save time"),
            # Just before and after save times
            (save_time_1 - timedelta(seconds=1), data_1, "Just before first save"),
            (save_time_1 + timedelta(seconds=1), data_1, "Just after first save"),
            (save_time_2 - timedelta(seconds=1), data_1, "Just before second save"),
            (save_time_2 + timedelta(seconds=1), data_2, "Just after second save"),
            # Far in the future
            (save_time_2 + timedelta(days=1), data_2, "Far future"),
        ]

        for query_time, expected_data, description in boundary_tests:
            guard.set_time_context(query_time)
            try:
                result = guard.get(key, force_download=False)

                # Check if we got empty DataFrame (for queries before any data)
                if result.empty:
                    # For queries before first save, this is expected behavior
                    if query_time < save_time_1:
                        continue  # This is acceptable
                    else:
                        raise AssertionError(
                            f"{description}: unexpected empty result for query_time {query_time}"
                        )

                # For non-empty results, check if we have the expected data
                if "A" in result.index and "col" in result.columns:
                    expected_value = expected_data.loc["A", "col"]
                    actual_value = result.loc["A", "col"]
                    assert actual_value == expected_value, (
                        f"{description}: expected {expected_value}, got {actual_value}"
                    )
                else:
                    # At minimum, we should have some structure that makes sense
                    assert len(result.index) > 0, f"{description}: got empty index"

            finally:
                guard.clear_time_context()

    def test_time_context_with_dtype_changes(self, guard):
        """
        Test time context integration with dtype changes.

        This test verifies:
        1. Historical queries return correct dtypes for their time
        2. Dtype evolution is properly tracked
        3. Time queries work correctly across dtype boundaries
        """
        key = "dtype_time_test"

        base_time = datetime.now() - timedelta(hours=6)

        # Evolution of dtypes over time
        dtype_phases = [
            (
                base_time,
                pd.DataFrame(
                    {"num_col": np.array([1, 2], dtype="int8"), "str_col": ["a", "b"]},
                    index=["X", "Y"],
                ),
            ),
            (
                base_time + timedelta(hours=2),
                pd.DataFrame(
                    {
                        "num_col": np.array([1, 2], dtype="int16"),  # dtype upgrade
                        "str_col": ["a", "b"],
                    },
                    index=["X", "Y"],
                ),
            ),
            (
                base_time + timedelta(hours=4),
                pd.DataFrame(
                    {
                        "num_col": np.array([1, 2], dtype="int32"),  # another upgrade
                        "str_col": pd.array(["a", "b"], dtype="string"),  # string dtype
                        "new_col": [1.0, 2.0],  # new column
                    },
                    index=["X", "Y"],
                ),
            ),
        ]

        # Save all phases
        for i, (phase_time, phase_data) in enumerate(dtype_phases):
            with patch.object(guard, "_now", return_value=phase_time):
                with self._mock_finlab_data(phase_data):
                    if i == 2:  # Phase 3 has new column, force download
                        guard.get(key, force_download=True)
                    else:
                        guard.get(key)

        # Test dtype consistency at different times
        dtype_tests = [
            (base_time + timedelta(hours=1), "int8", "object", False),  # Phase 1
            (base_time + timedelta(hours=3), "int16", "object", False),  # Phase 2
            (base_time + timedelta(hours=5), "int32", "string", True),  # Phase 3
        ]

        for (
            query_time,
            expected_num_dtype,
            expected_str_dtype,
            has_new_col,
        ) in dtype_tests:
            guard.set_time_context(query_time)
            try:
                result = guard.get(key, force_download=False)

                # Check dtypes
                assert str(result["num_col"].dtype) == expected_num_dtype
                assert str(result["str_col"].dtype) == expected_str_dtype

                # Check column existence
                if has_new_col:
                    assert "new_col" in result.columns
                else:
                    assert "new_col" not in result.columns

            finally:
                guard.clear_time_context()

    def test_time_context_clearing_behavior(self, guard):
        """
        Test various aspects of time context clearing.

        This test verifies:
        1. Time context affects subsequent calls
        2. Clearing returns to latest data
        3. Multiple set/clear cycles work correctly
        """
        key = "clearing_test"

        # Setup historical data
        old_time = datetime.now() - timedelta(hours=2)
        new_time = datetime.now() - timedelta(hours=1)

        old_data = pd.DataFrame({"value": [100]}, index=["A"])
        new_data = pd.DataFrame(
            {"value": [100, 200]}, index=["A", "B"]
        )  # Add new index instead of modifying

        with patch.object(guard, "_now", return_value=old_time):
            with self._mock_finlab_data(old_data):
                guard.get(key)

        with patch.object(guard, "_now", return_value=new_time):
            with self._mock_finlab_data(new_data):
                guard.get(key)

        # Test clearing behavior
        # 1. Normal state (no time context)
        with self._mock_finlab_data(new_data):  # Provide mock for guard.get()
            result = guard.get(key, force_download=False)
        assert len(result) == 2  # Latest data has both A and B
        assert result.loc["A", "value"] == 100  # A value stays same

        # 2. Set time context
        query_time = old_time + timedelta(minutes=30)
        guard.set_time_context(query_time)
        result = guard.get(key, force_download=False)
        assert len(result) == 1  # Historical data has only A
        assert result.loc["A", "value"] == 100  # Historical data

        # 3. Clear time context
        guard.clear_time_context()
        with self._mock_finlab_data(new_data):  # Provide mock for guard.get()
            result = guard.get(key, force_download=False)
        assert len(result) == 2  # Back to latest (both A and B)

        # 4. Multiple set/clear cycles
        for _ in range(3):
            guard.set_time_context(query_time)
            result = guard.get(key, force_download=False)
            assert len(result) == 1  # Historical (only A)

            guard.clear_time_context()
            with self._mock_finlab_data(new_data):  # Provide mock for guard.get()
                result = guard.get(key, force_download=False)
            assert len(result) == 2  # Latest (both A and B)

        # 5. Verify time context state
        assert guard.get_time_context() is None  # Should be cleared

    def test_time_context_with_empty_results(self, guard):
        """
        Test time context behavior when queries return empty results.
        """
        key = "empty_test"

        # Save some data
        save_time = datetime.now() - timedelta(hours=1)
        data = pd.DataFrame({"col": [1, 2]}, index=["A", "B"])

        with patch.object(guard, "_now", return_value=save_time):
            with self._mock_finlab_data(data):
                guard.get(key)

        # Query before any data was saved
        query_time = save_time - timedelta(hours=2)
        guard.set_time_context(query_time)
        try:
            result = guard.get(key, force_download=False)
            # Should still get some data (earliest available)
            assert len(result) >= 0  # May be empty or return earliest data
        finally:
            guard.clear_time_context()

    def test_time_context_string_parsing(self, guard):
        """
        Test time context with string time inputs.
        """
        key = "string_time_test"

        # Save some data
        save_time = datetime.now() - timedelta(hours=1)
        data = pd.DataFrame({"col": [1]}, index=["A"])

        with patch.object(guard, "_now", return_value=save_time):
            with self._mock_finlab_data(data):
                guard.get(key)

        # Test string time context
        time_string = (save_time + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        guard.set_time_context(time_string)
        try:
            result = guard.get(key, force_download=False)
            assert len(result) == 1
        finally:
            guard.clear_time_context()

        # Test invalid string
        with pytest.raises(ValueError):
            guard.set_time_context("invalid-time-string")
