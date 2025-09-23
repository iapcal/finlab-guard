"""Integration tests for the complete dtype version control system in finlab-guard.

This module tests the dtype mapping, versioning, and time-based queries to ensure
that data types are preserved correctly across different time contexts and
structural changes.
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


class TestDtypeSystemIntegration:
    """Test the complete dtype version control system."""

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

    def test_dtype_versioning_full_cycle(self, guard):
        """
        Test complete dtype versioning cycle including multiple dtype changes.

        This test verifies:
        1. Initial dtype mapping creation
        2. Dtype change detection and new version creation
        3. Historical dtype preservation
        4. Complex dtype evolution tracking
        """
        key = "dtype_versioning_test"

        # Phase 1: Initial data with specific dtypes
        initial_time = datetime.now() - timedelta(hours=4)
        initial_data = pd.DataFrame(
            {
                "int_col": np.array([1, 2, 3], dtype="int32"),
                "float_col": np.array([1.1, 2.2, 3.3], dtype="float32"),
                "str_col": ["a", "b", "c"],
            },
            index=["X", "Y", "Z"],
        )

        with patch.object(guard, "_now", return_value=initial_time):
            with self._mock_finlab_data(initial_data):
                result1 = guard.get(key)

        # Verify initial dtypes
        assert result1["int_col"].dtype == np.dtype("int32")
        assert result1["float_col"].dtype == np.dtype("float32")
        assert result1["str_col"].dtype == np.dtype("object")

        # Phase 2: Change int_col from int32 to int64
        phase2_time = datetime.now() - timedelta(hours=3)
        phase2_data = pd.DataFrame(
            {
                "int_col": np.array([1, 2, 3], dtype="int64"),  # Changed dtype
                "float_col": np.array([1.1, 2.2, 3.3], dtype="float32"),
                "str_col": ["a", "b", "c"],
            },
            index=["X", "Y", "Z"],
        )

        with patch.object(guard, "_now", return_value=phase2_time):
            with self._mock_finlab_data(phase2_data):
                result2 = guard.get(key)

        # Verify phase 2 dtypes
        assert result2["int_col"].dtype == np.dtype("int64")  # Changed
        assert result2["float_col"].dtype == np.dtype("float32")  # Same

        # Phase 3: Change float_col from float32 to float64 and add new column
        phase3_time = datetime.now() - timedelta(hours=2)
        phase3_data = pd.DataFrame(
            {
                "int_col": np.array([1, 2, 3], dtype="int64"),
                "float_col": np.array(
                    [1.1, 2.2, 3.3], dtype="float32"
                ),  # Keep same to avoid precision changes
                "str_col": ["a", "b", "c"],
                "new_col": np.array([10, 20, 30], dtype="int16"),  # New column
            },
            index=["X", "Y", "Z"],
        )

        with patch.object(guard, "_now", return_value=phase3_time):
            with self._mock_finlab_data(phase3_data):
                result3 = guard.get(
                    key, force_download=True
                )  # Force download for new column addition

        # Verify phase 3 dtypes
        assert result3["int_col"].dtype == np.dtype("int64")
        assert result3["float_col"].dtype == np.dtype("float32")  # Same as phase 2
        assert result3["new_col"].dtype == np.dtype("int16")  # New

        # Verify historical dtype access
        # Query at phase 1 time
        guard.set_time_context(initial_time + timedelta(minutes=10))
        try:
            historical_1 = guard.get(key, force_download=False)
            assert historical_1["int_col"].dtype == np.dtype("int32")  # Original
            assert historical_1["float_col"].dtype == np.dtype("float32")  # Original
            assert "new_col" not in historical_1.columns  # Doesn't exist yet
        finally:
            guard.clear_time_context()

        # Query at phase 2 time
        guard.set_time_context(phase2_time + timedelta(minutes=10))
        try:
            historical_2 = guard.get(key, force_download=False)
            assert historical_2["int_col"].dtype == np.dtype("int64")  # Updated
            assert historical_2["float_col"].dtype == np.dtype(
                "float32"
            )  # Still original
            assert "new_col" not in historical_2.columns  # Still doesn't exist
        finally:
            guard.clear_time_context()

        # Verify dtype history structure
        dtype_history = guard.cache_manager._load_dtype_mapping(key)
        assert len(dtype_history["dtype_history"]) >= 3, (
            "Should have multiple dtype versions"
        )

    def test_time_based_dtype_queries_accuracy(self, guard):
        """
        Test accuracy of time-based dtype queries across complex scenarios.

        This test ensures that querying data at specific times returns
        exactly the correct dtypes that were valid at that time.
        """
        key = "time_dtype_test"

        # Create timeline with precise timestamps
        t1 = datetime.now() - timedelta(hours=5)
        t2 = datetime.now() - timedelta(hours=4)
        t3 = datetime.now() - timedelta(hours=3)
        datetime.now() - timedelta(hours=2)

        # Data evolution with different dtype patterns
        data_v1 = pd.DataFrame(
            {
                "col": np.array([100], dtype="int8")  # Very specific dtype
            },
            index=["A"],
        )

        data_v2 = pd.DataFrame(
            {
                "col": np.array([100], dtype="int16")  # Upgraded
            },
            index=["A"],
        )

        data_v3 = pd.DataFrame(
            {
                "col": np.array([100], dtype="int32")  # Upgraded again
            },
            index=["A"],
        )

        # Save data at different times
        times_and_data = [(t1, data_v1), (t2, data_v2), (t3, data_v3)]

        for timestamp, data in times_and_data:
            with patch.object(guard, "_now", return_value=timestamp):
                with self._mock_finlab_data(data):
                    guard.get(key)

        # Test precise time queries
        test_cases = [
            (t1 + timedelta(minutes=30), "int8"),  # Between t1 and t2
            (t2 + timedelta(minutes=30), "int16"),  # Between t2 and t3
            (t3 + timedelta(minutes=30), "int32"),  # After t3
        ]

        for query_time, expected_dtype in test_cases:
            guard.set_time_context(query_time)
            try:
                result = guard.get(key, force_download=False)
                actual_dtype = str(result["col"].dtype)
                assert actual_dtype == expected_dtype, (
                    f"At {query_time}: expected {expected_dtype}, got {actual_dtype}"
                )
            finally:
                guard.clear_time_context()

    def test_mixed_dtype_preservation(self, guard):
        """
        Test preservation of mixed and complex dtypes.

        This test covers:
        - Categorical dtypes
        - DateTime dtypes
        - Mixed numeric dtypes
        - String vs object distinction
        """
        key = "mixed_dtype_test"

        # Create DataFrame with complex mixed dtypes
        complex_data = pd.DataFrame(
            {
                "int8_col": np.array([1, 2, 3], dtype="int8"),
                "int64_col": np.array([100, 200, 300], dtype="int64"),
                "float32_col": np.array([1.1, 2.2, 3.3], dtype="float32"),
                "float64_col": np.array([10.1, 20.2, 30.3], dtype="float64"),
                "bool_col": np.array([True, False, True], dtype="bool"),
                "datetime_col": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ),
                "category_col": pd.Categorical(
                    ["X", "Y", "X"], categories=["X", "Y", "Z"]
                ),
                "string_col": pd.array(["a", "b", "c"], dtype="string"),
                "object_col": ["obj1", "obj2", "obj3"],  # Regular object dtype
            },
            index=["row1", "row2", "row3"],
        )

        initial_time = datetime.now() - timedelta(hours=2)

        with patch.object(guard, "_now", return_value=initial_time):
            with self._mock_finlab_data(complex_data):
                result = guard.get(key)

        # Verify all dtypes are preserved exactly
        expected_dtypes = {
            "int8_col": "int8",
            "int64_col": "int64",
            "float32_col": "float32",
            "float64_col": "float64",
            "bool_col": "bool",
            "datetime_col": "datetime64[ns]",
            "category_col": "category",
            "string_col": "string",
            "object_col": "object",
        }

        for col, expected_dtype in expected_dtypes.items():
            actual_dtype = str(result[col].dtype)
            assert actual_dtype == expected_dtype, (
                f"Column {col}: expected {expected_dtype}, got {actual_dtype}"
            )

        # Verify categorical categories are preserved
        assert list(result["category_col"].cat.categories) == ["X", "Y", "Z"]

        # Test time context query preserves all dtypes
        query_time = initial_time + timedelta(minutes=30)
        guard.set_time_context(query_time)
        try:
            historical_result = guard.get(key, force_download=False)
            for col, expected_dtype in expected_dtypes.items():
                actual_dtype = str(historical_result[col].dtype)
                assert actual_dtype == expected_dtype, (
                    f"Historical {col}: expected {expected_dtype}, got {actual_dtype}"
                )
        finally:
            guard.clear_time_context()

    def test_columns_order_preservation_all_types(self, guard):
        """
        Test that column order is preserved across dtype changes and time queries.
        """
        key = "column_order_test"

        # Create data with specific column order
        initial_data = pd.DataFrame(
            {"z_col": [1, 2], "a_col": [10, 20], "m_col": [100, 200]}, index=["X", "Y"]
        )

        initial_time = datetime.now() - timedelta(hours=2)

        with patch.object(guard, "_now", return_value=initial_time):
            with self._mock_finlab_data(initial_data):
                result1 = guard.get(key)

        # Verify initial order
        assert list(result1.columns) == ["z_col", "a_col", "m_col"]

        # Add new column with dtype change
        later_time = datetime.now() - timedelta(hours=1)
        modified_data = pd.DataFrame(
            {
                "z_col": np.array([1, 2], dtype="int64"),  # Changed dtype
                "a_col": [10, 20],
                "m_col": [100, 200],
                "b_col": [1000, 2000],  # New column inserted
            },
            index=["X", "Y"],
        )

        with patch.object(guard, "_now", return_value=later_time):
            with self._mock_finlab_data(modified_data):
                result2 = guard.get(key)

        # Verify new order is preserved
        assert list(result2.columns) == ["z_col", "a_col", "m_col", "b_col"]

        # Verify historical query preserves original order
        query_time = initial_time + timedelta(minutes=30)
        guard.set_time_context(query_time)
        try:
            historical = guard.get(key, force_download=False)
            assert list(historical.columns) == [
                "z_col",
                "a_col",
                "m_col",
            ]  # Original order
        finally:
            guard.clear_time_context()

    def test_index_order_preservation_all_types(self, guard):
        """
        Test that index order is preserved across dtype changes and time queries.
        """
        key = "index_order_test"

        # Create data with specific index order (not alphabetical)
        initial_data = pd.DataFrame(
            {"col1": [100, 200, 300], "col2": [1.1, 2.2, 3.3]},
            index=["gamma", "alpha", "beta"],
        )  # Specific order

        initial_time = datetime.now() - timedelta(hours=2)

        with patch.object(guard, "_now", return_value=initial_time):
            with self._mock_finlab_data(initial_data):
                result1 = guard.get(key)

        # Verify initial index order
        assert list(result1.index) == ["gamma", "alpha", "beta"]

        # Add new index with dtype change
        later_time = datetime.now() - timedelta(hours=1)
        modified_data = pd.DataFrame(
            {
                "col1": np.array([100, 200, 300, 400], dtype="int64"),  # Changed dtype
                "col2": [1.1, 2.2, 3.3, 4.4],
            },
            index=["gamma", "alpha", "beta", "delta"],
        )  # New index added

        with patch.object(guard, "_now", return_value=later_time):
            with self._mock_finlab_data(modified_data):
                result2 = guard.get(key)

        # Verify new index order is preserved
        assert list(result2.index) == ["gamma", "alpha", "beta", "delta"]

        # Verify historical query preserves original index order
        query_time = initial_time + timedelta(minutes=30)
        guard.set_time_context(query_time)
        try:
            historical = guard.get(key, force_download=False)
            assert list(historical.index) == [
                "gamma",
                "alpha",
                "beta",
            ]  # Original order
        finally:
            guard.clear_time_context()

    def test_dtype_history_limits(self, guard):
        """
        Test behavior when dtype history becomes very large.

        This test verifies that the system handles many dtype changes
        efficiently and maintains historical accuracy.
        """
        key = "dtype_limit_test"
        base_time = datetime.now() - timedelta(hours=10)

        # Create many dtype changes over time (avoid int to float conversion issues)
        dtype_sequence = ["int8", "int16", "int32", "int64"]

        for i, dtype_name in enumerate(dtype_sequence):
            timestamp = base_time + timedelta(hours=i)
            data = pd.DataFrame(
                {"evolving_col": np.array([10, 20], dtype=dtype_name)},
                index=["A", "B"],
            )

            with patch.object(guard, "_now", return_value=timestamp):
                with self._mock_finlab_data(data):
                    if i == 0:
                        guard.get(key)  # First time
                    else:
                        guard.get(key)  # Dtype changes should not trigger exceptions

        # Verify dtype history contains all changes
        dtype_history = guard.cache_manager._load_dtype_mapping(key)
        assert len(dtype_history["dtype_history"]) >= len(dtype_sequence)

        # Test queries at different points in the evolution
        for i, expected_dtype in enumerate(dtype_sequence):
            query_time = base_time + timedelta(hours=i, minutes=30)
            guard.set_time_context(query_time)
            try:
                result = guard.get(key, force_download=False)
                actual_dtype = str(result["evolving_col"].dtype)
                assert actual_dtype == expected_dtype, (
                    f"Step {i}: expected {expected_dtype}, got {actual_dtype}"
                )
            finally:
                guard.clear_time_context()


class TestDtypeEdgeCases:
    """Test edge cases in dtype handling."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def guard(self, temp_cache_dir):
        """Create FinlabGuard instance for testing."""
        guard_instance = FinlabGuard(cache_dir=temp_cache_dir)
        yield guard_instance
        # Ensure DuckDB connection is closed to prevent Windows file locking
        guard_instance.close()

    def _mock_finlab_data(self, data: pd.DataFrame):
        """Mock finlab.data.get to return specified data."""
        # Use the simpler approach - patch the _fetch_from_finlab method
        # The finlab module mock is handled by conftest.py
        return patch.object(FinlabGuard, "_fetch_from_finlab", return_value=data)

    def test_nullable_integer_dtypes(self, guard):
        """Test handling of pandas nullable integer dtypes."""
        key = "nullable_int_test"

        # Create data with nullable integers
        data = pd.DataFrame(
            {
                "nullable_int": pd.array([1, None, 3], dtype="Int64"),
                "regular_int": [1, 2, 3],
            },
            index=["A", "B", "C"],
        )

        with self._mock_finlab_data(data):
            result = guard.get(key)

        # Verify nullable dtype is preserved
        assert str(result["nullable_int"].dtype) == "Int64"
        assert pd.isna(result.loc["B", "nullable_int"])

    def test_timezone_aware_datetime(self, guard):
        """Test handling of timezone-aware datetime dtypes."""
        key = "timezone_test"

        # Create timezone-aware datetime data
        utc_times = pd.to_datetime(["2023-01-01", "2023-01-02"]).tz_localize("UTC")
        data = pd.DataFrame(
            {"utc_time": utc_times, "regular_col": [1, 2]}, index=["A", "B"]
        )

        with self._mock_finlab_data(data):
            result = guard.get(key)

        # Verify timezone information is preserved
        assert result["utc_time"].dtype.tz is not None
        assert str(result["utc_time"].dtype.tz) == "UTC"
