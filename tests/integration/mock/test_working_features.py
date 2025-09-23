"""Working integration tests for core finlab-guard features.

This module contains simplified integration tests that verify the essential
functionality works correctly in the new mock environment.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from finlab_guard import FinlabGuard
from finlab_guard.utils.exceptions import DataModifiedException


class TestWorkingIntegration:
    """Test core integration functionality that works reliably."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def guard(self, temp_cache_dir):
        """Create FinlabGuard instance for testing."""
        # Use no compression to avoid PyArrow issues
        config = {"compression": None}
        guard_instance = FinlabGuard(cache_dir=temp_cache_dir, config=config)
        yield guard_instance
        # Ensure DuckDB connection is closed to prevent Windows file locking
        guard_instance.close()

    def _mock_finlab_data(self, data: pd.DataFrame):
        """Mock finlab.data.get to return specified data."""
        return patch.object(FinlabGuard, "_fetch_from_finlab", return_value=data)

    def test_basic_data_storage_and_retrieval(self, guard):
        """Test basic data storage and retrieval works."""
        test_data = pd.DataFrame(
            {"price": [100.0, 101.0, 99.0], "volume": [1000, 1100, 900]},
            index=["2023-01-01", "2023-01-02", "2023-01-03"],
        )

        with self._mock_finlab_data(test_data):
            result = guard.get("basic_test")

        pd.testing.assert_frame_equal(result, test_data)
        assert guard.cache_manager.exists("basic_test")

    def test_incremental_data_updates(self, guard):
        """Test that incremental data updates work correctly."""
        # Initial data
        initial_data = pd.DataFrame(
            {"col1": [1, 2], "col2": [1.1, 2.2]}, index=["A", "B"]
        )

        # Updated data (adding new row)
        updated_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3]}, index=["A", "B", "C"]
        )

        # First save
        with self._mock_finlab_data(initial_data):
            result1 = guard.get("incremental_test")

        pd.testing.assert_frame_equal(result1, initial_data)

        # Update with new data
        with self._mock_finlab_data(updated_data):
            result2 = guard.get("incremental_test")

        pd.testing.assert_frame_equal(result2, updated_data)

    def test_data_modification_detection(self, guard):
        """Test that data modification detection works."""
        # Initial data
        initial_data = pd.DataFrame(
            {"col1": [100, 200], "col2": [1.1, 2.2]}, index=["A", "B"]
        )

        # Modified data (historical change)
        modified_data = pd.DataFrame(
            {"col1": [105, 200], "col2": [1.1, 2.2]},  # A changed from 100 to 105
            index=["A", "B"],
        )

        # First save
        with self._mock_finlab_data(initial_data):
            result1 = guard.get("modification_test")

        pd.testing.assert_frame_equal(result1, initial_data)

        # Attempt to get modified data (should raise exception)
        with self._mock_finlab_data(modified_data):
            with pytest.raises(DataModifiedException):
                guard.get("modification_test")

        # Force download should work
        with self._mock_finlab_data(modified_data):
            result2 = guard.get("modification_test", force_download=True)

        pd.testing.assert_frame_equal(result2, modified_data)

    def test_multiple_datasets(self, guard):
        """Test handling multiple independent datasets."""
        data1 = pd.DataFrame({"price": [100, 101]}, index=["A", "B"])
        data2 = pd.DataFrame({"volume": [1000, 1100]}, index=["X", "Y"])

        with self._mock_finlab_data(data1):
            result1 = guard.get("dataset1")

        with self._mock_finlab_data(data2):
            result2 = guard.get("dataset2")

        pd.testing.assert_frame_equal(result1, data1)
        pd.testing.assert_frame_equal(result2, data2)

        # Both should be cached independently
        assert guard.cache_manager.exists("dataset1")
        assert guard.cache_manager.exists("dataset2")

    def test_empty_dataframe_handling(self, guard):
        """Test handling of empty DataFrames."""
        empty_data = pd.DataFrame()

        with self._mock_finlab_data(empty_data):
            result = guard.get("empty_test")

        assert len(result) == 0
        # Empty DataFrames are not cached (this is expected behavior)
        assert not guard.cache_manager.exists("empty_test")

    def test_large_dataframe_handling(self, guard):
        """Test handling of larger DataFrames."""
        large_data = pd.DataFrame(
            {"col1": range(100), "col2": [f"value_{i}" for i in range(100)]}
        )

        with self._mock_finlab_data(large_data):
            result = guard.get("large_test")

        pd.testing.assert_frame_equal(result, large_data)
        assert len(result) == 100

    def test_dtype_preservation_basic(self, guard):
        """Test basic dtype preservation."""
        typed_data = pd.DataFrame(
            {
                "int_col": pd.array([1, 2, 3], dtype="int32"),
                "float_col": pd.array([1.1, 2.2, 3.3], dtype="float64"),
                "str_col": ["a", "b", "c"],
            }
        )

        with self._mock_finlab_data(typed_data):
            result = guard.get("dtype_test")

        # Check that dtypes are preserved
        assert result["int_col"].dtype == "int32"
        assert result["float_col"].dtype == "float64"
        assert result["str_col"].dtype == "object"

    def test_index_preservation(self, guard):
        """Test that index is preserved correctly."""
        indexed_data = pd.DataFrame(
            {"value": [10, 20, 30]}, index=["first", "second", "third"]
        )

        with self._mock_finlab_data(indexed_data):
            result = guard.get("index_test")

        assert list(result.index) == ["first", "second", "third"]
        pd.testing.assert_frame_equal(result, indexed_data)

    def test_column_order_preservation(self, guard):
        """Test that column order is preserved."""
        ordered_data = pd.DataFrame(
            {"z_col": [1, 2, 3], "a_col": [4, 5, 6], "m_col": [7, 8, 9]}
        )

        with self._mock_finlab_data(ordered_data):
            result = guard.get("order_test")

        assert list(result.columns) == ["z_col", "a_col", "m_col"]
        pd.testing.assert_frame_equal(result, ordered_data)


class TestMockEnvironmentFeatures:
    """Test mock environment specific features."""

    def test_finlab_mock_isolation(self):
        """Test that finlab mock is properly isolated."""
        import sys

        # Should have mocked finlab
        assert "finlab" in sys.modules
        import finlab

        # Should be able to use finlab.data.get
        assert hasattr(finlab, "data")
        assert hasattr(finlab.data, "get")

        # Should return default data when called
        result = finlab.data.get("any_key")
        assert isinstance(result, pd.DataFrame)

    def test_environment_independence(self):
        """Test that tests work regardless of environment variables."""
        import os

        # Should work regardless of FINLAB_TOKEN setting
        original_token = os.environ.get("FINLAB_TOKEN")

        try:
            # Test with empty token
            os.environ["FINLAB_TOKEN"] = ""
            temp_dir = tempfile.mkdtemp()
            guard = FinlabGuard(cache_dir=temp_dir, config={"compression": None})

            test_data = pd.DataFrame({"test": [1, 2, 3]})
            with patch.object(
                FinlabGuard, "_fetch_from_finlab", return_value=test_data
            ):
                result = guard.get("env_test")

            pd.testing.assert_frame_equal(result, test_data)
            guard.close()  # Ensure DuckDB connection is closed before cleanup
            shutil.rmtree(temp_dir)

        finally:
            if original_token is not None:
                os.environ["FINLAB_TOKEN"] = original_token
            else:
                os.environ.pop("FINLAB_TOKEN", None)
