"""Error scenarios and exception handling tests for finlab-guard."""

import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import duckdb
import pandas as pd
import pytest

from finlab_guard.cache.manager import CacheManager
from finlab_guard.cache.validator import DataValidator
from finlab_guard.core.guard import FinlabGuard
from finlab_guard.utils.exceptions import (
    DataModifiedException,
    FinlabConnectionException,
    InvalidDataTypeException,
    UnsupportedDataFormatException,
)


def safe_rmtree(path, ignore_errors=False):
    """Windows-compatible rmtree with retry logic for DuckDB file locking."""
    for attempt in range(5):
        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            break
        except (PermissionError, OSError):
            if attempt < 4:
                time.sleep(0.1)  # Wait 100ms before retry
                continue
            if not ignore_errors:
                raise


class TestFinlabGuardErrorScenarios:
    """Test error scenarios for FinlabGuard class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.guard = FinlabGuard(cache_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        # Close DuckDB connections to prevent Windows file locking
        if hasattr(self, "guard") and self.guard:
            self.guard.close()
        safe_rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_invalid_cache_dir_permissions(self):
        """Test initialization with invalid cache directory permissions."""
        # Skip this test on Windows as it handles permissions differently
        if sys.platform.startswith("win"):
            pytest.skip("Windows permission handling differs from Unix")

        # Create a read-only directory
        readonly_dir = tempfile.mkdtemp()
        guard = None
        try:
            os.chmod(readonly_dir, 0o444)  # Read-only
            # DuckDB architecture requires write permissions to create database file
            with pytest.raises(
                (OSError, RuntimeError, duckdb.IOException)
            ):  # DuckDB IOException or similar
                guard = FinlabGuard(cache_dir=readonly_dir)
        finally:
            if guard:
                guard.close()
            os.chmod(readonly_dir, 0o755)  # Restore permissions
            safe_rmtree(readonly_dir, ignore_errors=True)

    def test_set_time_context_with_invalid_string(self):
        """Test setting time context with invalid string."""
        with pytest.raises(
            (ValueError, TypeError)
        ):  # pandas will raise exception for invalid date string
            self.guard.set_time_context("invalid-date-string")

    def test_get_with_finlab_import_error(self):
        """Test get() when finlab package is not available."""
        with patch.object(
            self.guard,
            "_fetch_from_finlab",
            side_effect=FinlabConnectionException("finlab package not found"),
        ):
            with pytest.raises(
                FinlabConnectionException, match="finlab package not found"
            ):
                self.guard.get("test_key")

    def test_get_with_finlab_connection_error(self):
        """Test get() when finlab raises an exception."""
        with patch.object(
            self.guard,
            "_fetch_from_finlab",
            side_effect=ConnectionError("Network error"),
        ):
            with pytest.raises(
                FinlabConnectionException, match="Cannot fetch data from finlab"
            ):
                self.guard.get("test_key")

    def test_get_with_data_modification_no_force(self):
        """Test get() when data is modified and force_download=False."""
        # Create initial cache
        df1 = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )
        with patch.object(self.guard, "_fetch_from_finlab", return_value=df1):
            self.guard.get("test_key")

        # Simulate modified data
        df2 = pd.DataFrame(
            {"A": [99, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )
        with patch.object(self.guard, "_fetch_from_finlab", return_value=df2):
            with pytest.raises(DataModifiedException, match="Historical data modified"):
                self.guard.get("test_key", force_download=False)

    def test_get_with_data_modification_force_download(self):
        """Test get() when data is modified and force_download=True."""
        # Create initial cache
        df1 = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )
        with patch.object(self.guard, "_fetch_from_finlab", return_value=df1):
            self.guard.get("test_key")

        # Simulate modified data with force download
        df2 = pd.DataFrame(
            {"A": [99, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )
        with patch.object(self.guard, "_fetch_from_finlab", return_value=df2):
            result = self.guard.get("test_key", force_download=True)
            pd.testing.assert_frame_equal(result, df2)

    def test_install_patch_when_already_installed(self):
        """Test installing patch when already installed."""
        mock_finlab = Mock()
        mock_finlab.data._original_get = Mock()  # Simulate already installed

        with patch.dict(
            "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
        ):
            with pytest.raises(RuntimeError, match="finlab-guard already installed"):
                self.guard.install_patch()

    def test_install_patch_when_finlab_not_available(self):
        """Test installing patch when finlab is not available."""
        # Simulate finlab.data import failure
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "finlab.data":
                raise ImportError("No module named 'finlab.data'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="finlab package not found"):
                self.guard.install_patch()

    def test_install_patch_multiple_instances(self):
        """Test installing patch with multiple FinlabGuard instances."""
        guard2 = FinlabGuard(cache_dir=self.temp_dir)

        mock_finlab = Mock()
        delattr(mock_finlab.data, "_original_get") if hasattr(
            mock_finlab.data, "_original_get"
        ) else None

        with patch.dict(
            "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
        ):
            # First installation should work
            self.guard.install_patch()

            # Second installation should fail
            with pytest.raises(RuntimeError, match="finlab-guard already installed"):
                guard2.install_patch()

    def test_remove_patch_when_not_installed(self):
        """Test removing patch when not installed."""
        mock_finlab = Mock()
        # Ensure no _original_get attribute
        if hasattr(mock_finlab.data, "_original_get"):
            delattr(mock_finlab.data, "_original_get")

        with patch.dict(
            "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
        ):
            # Should not raise exception, just log warning
            self.guard.remove_patch()

    def test_remove_patch_when_finlab_not_available(self):
        """Test removing patch when finlab is not available."""
        # Simulate finlab.data import failure
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "finlab.data":
                raise ImportError("No module named 'finlab.data'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Should not raise exception, just log warning
            self.guard.remove_patch()

    def test_get_historical_data_with_nonexistent_key(self):
        """Test getting historical data for non-existent key."""
        self.guard.set_time_context(datetime(2023, 1, 1))

        # Should return empty DataFrame for non-existent key
        result = self.guard.get("nonexistent_key")
        assert result.empty

    def test_get_change_history_nonexistent_key(self):
        """Test getting change history for non-existent key."""
        result = self.guard.get_change_history("nonexistent_key")
        assert result.empty

    def test_get_storage_info_nonexistent_key(self):
        """Test getting storage info for non-existent key."""
        result = self.guard.get_storage_info("nonexistent_key")
        assert result == {}


class TestCacheManagerErrorScenarios:
    """Test error scenarios for CacheManager class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(
            Path(self.temp_dir), {"compression": "snappy"}
        )

    def teardown_method(self):
        """Clean up test environment."""
        # Close DuckDB connections to prevent Windows file locking
        if hasattr(self, "cache_manager") and self.cache_manager:
            self.cache_manager.close()
        safe_rmtree(self.temp_dir, ignore_errors=True)

    def test_load_data_with_corrupted_parquet(self):
        """Test loading data when DuckDB file is corrupted."""
        # Skip this test on Windows due to DuckDB file locking issues
        if sys.platform.startswith("win"):
            pytest.skip("Windows DuckDB file locking prevents direct file corruption")

        # Create a corrupted DuckDB file
        cache_path = self.cache_manager._get_cache_path("test_key")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Close connection before attempting to write to file
        self.cache_manager.close()

        with open(cache_path, "w") as f:
            f.write("corrupted data")

        # DuckDB should raise IOException when trying to connect to corrupted file
        # This tests that we handle database corruption gracefully
        with pytest.raises(
            duckdb.IOException, match="not a valid DuckDB database file"
        ):
            self.cache_manager = CacheManager(
                Path(self.temp_dir), {"compression": "snappy"}
            )

    def test_load_dtype_mapping_with_corrupted_json(self):
        """Test loading dtype mapping when JSON file is corrupted."""
        # Create a corrupted JSON file
        dtype_path = self.cache_manager._get_dtype_path("test_key")
        dtype_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dtype_path, "w") as f:
            f.write("corrupted json data")

        # Should handle corruption gracefully
        result = self.cache_manager._load_dtype_mapping("test_key")
        assert result is None

    def test_save_data_with_disk_full(self):
        """Test saving data when disk is full (simulated)."""
        df = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )

        # Mock save_snapshot to simulate disk full error
        with patch.object(
            self.cache_manager,
            "save_snapshot",
            side_effect=OSError("No space left on device"),
        ):
            with pytest.raises(OSError):
                self.cache_manager.save_data("test_key", df, datetime.now())

    def test_save_data_with_permission_denied(self):
        """Test saving data with permission denied."""
        df = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )

        # Mock save_snapshot to simulate permission denied
        with patch.object(
            self.cache_manager,
            "save_snapshot",
            side_effect=PermissionError("Permission denied"),
        ):
            with pytest.raises(PermissionError):
                self.cache_manager.save_data("test_key", df, datetime.now())

    def test_clear_key_with_permission_denied(self):
        """Test clearing key when file removal is denied."""
        # Create a cache file first
        df = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )
        self.cache_manager.save_data("test_key", df, datetime.now())

        # Mock to simulate permission denied
        with patch(
            "pathlib.Path.unlink", side_effect=PermissionError("Permission denied")
        ):
            # clear_key handles exceptions gracefully and logs them
            # instead of raising, so we just verify it doesn't crash
            self.cache_manager.clear_key("test_key")

    def test_get_cache_path_with_special_characters(self):
        """Test getting cache path with various special characters."""
        special_keys = [
            "key:with:colons",
            "key/with/slashes",
            "key\\with\\backslashes",
            "key<>with|special*chars?",
            "key\"with'quotes",
            "key with spaces",
        ]

        for key in special_keys:
            path = self.cache_manager._get_cache_path(key)
            # Should not contain problematic characters in filename
            assert (
                ":" not in path.name
            )  # Check only filename, not full path (Windows has C:)
            assert "/" not in path.name  # Only check filename, not full path
            assert "\\" not in path.name


class TestDataValidatorErrorScenarios:
    """Test error scenarios for DataValidator class."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = DataValidator()

    def test_validate_non_dataframe(self):
        """Test validating non-DataFrame objects."""
        invalid_inputs = [
            "string",
            123,
            [1, 2, 3],
            {"key": "value"},
            None,
            pd.Series([1, 2, 3]),
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(InvalidDataTypeException):
                self.validator.validate_dataframe_format(invalid_input)

    def test_validate_multiindex_columns(self):
        """Test validating DataFrame with MultiIndex columns."""
        arrays = [["A", "A", "B", "B"], ["one", "two", "one", "two"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

        df = pd.DataFrame([[1, 2, 3, 4]], columns=index)

        with pytest.raises(
            UnsupportedDataFormatException, match="MultiIndex columns are not supported"
        ):
            self.validator.validate_dataframe_format(df)

    def test_validate_multiindex_index(self):
        """Test validating DataFrame with MultiIndex index."""
        arrays = [["A", "A", "B", "B"], ["one", "two", "one", "two"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

        df = pd.DataFrame([[1], [2], [3], [4]], index=index, columns=["value"])

        with pytest.raises(
            UnsupportedDataFormatException, match="MultiIndex index is not supported"
        ):
            self.validator.validate_dataframe_format(df)

    def test_detect_changes_with_corrupted_cache(self):
        """Test change detection when cache is corrupted."""
        # Skip this test on Windows due to DuckDB file locking issues
        if sys.platform.startswith("win"):
            pytest.skip("Windows DuckDB file locking prevents direct file corruption")

        temp_dir = tempfile.mkdtemp()
        cache_manager = None
        try:
            cache_manager = CacheManager(Path(temp_dir), {"compression": "snappy"})

            # Create corrupted cache
            cache_path = cache_manager._get_cache_path("test_key")
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Close connection before attempting to write to file
            cache_manager.close()

            with open(cache_path, "w") as f:
                f.write("corrupted")

            # DuckDB should raise IOException when trying to connect to corrupted file
            # This tests that we detect database corruption appropriately
            with pytest.raises(
                duckdb.IOException, match="not a valid DuckDB database file"
            ):
                cache_manager = CacheManager(Path(temp_dir), {"compression": "snappy"})

        finally:
            if cache_manager:
                cache_manager.close()
            safe_rmtree(temp_dir, ignore_errors=True)

    def test_detect_changes_with_empty_new_data(self):
        """Test change detection with empty new DataFrame."""
        temp_dir = tempfile.mkdtemp()
        try:
            cache_manager = CacheManager(Path(temp_dir), {"compression": "snappy"})

            # Create some cached data first
            initial_data = pd.DataFrame(
                {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
            )
            cache_manager.save_data("test_key", initial_data, datetime.now())

            # Test with empty new data
            empty_data = pd.DataFrame(columns=["A"])

            modifications, additions = self.validator.detect_changes_detailed(
                "test_key", empty_data, cache_manager
            )
            # Empty data should result in no modifications or additions
            assert len(modifications) == 0
            assert len(additions) == 0

        finally:
            safe_rmtree(temp_dir, ignore_errors=True)

    def test_detect_changes_with_mismatched_columns(self):
        """Test change detection with completely different column structure."""
        temp_dir = tempfile.mkdtemp()
        try:
            cache_manager = CacheManager(Path(temp_dir), {"compression": "snappy"})

            # Create cached data with columns A, B
            initial_data = pd.DataFrame(
                {"A": [1, 2, 3], "B": [4, 5, 6]},
                index=pd.date_range("2023-01-01", periods=3),
            )
            cache_manager.save_data("test_key", initial_data, datetime.now())

            # New data with completely different columns
            new_data = pd.DataFrame(
                {"X": [7, 8, 9], "Y": [10, 11, 12]},
                index=pd.date_range("2023-01-01", periods=3),
            )

            modifications, additions = self.validator.detect_changes_detailed(
                "test_key", new_data, cache_manager
            )
            # Should handle gracefully - new columns treated as additions
            assert len(additions) > 0

        finally:
            safe_rmtree(temp_dir, ignore_errors=True)


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_change_class(self):
        """Test Change class functionality."""
        from finlab_guard.utils.exceptions import Change

        change = Change(
            coord=(0, "A"), old_value=1, new_value=2, timestamp=datetime(2023, 1, 1)
        )

        assert change.coord == (0, "A")
        assert change.old_value == 1
        assert change.new_value == 2
        assert change.timestamp == datetime(2023, 1, 1)

        # Test __repr__
        repr_str = repr(change)
        assert "Change(coord=(0, 'A'), 1 -> 2)" == repr_str

    def test_data_modified_exception(self):
        """Test DataModifiedException functionality."""
        from finlab_guard.utils.exceptions import Change, DataModifiedException

        changes = [
            Change((0, "A"), 1, 2, datetime(2023, 1, 1)),
            Change((1, "B"), 3, 4, datetime(2023, 1, 1)),
        ]

        exc = DataModifiedException("Test message", changes)
        assert exc.changes == changes

        # Test __str__ with few changes
        str_repr = str(exc)
        assert "Test message" in str_repr
        assert "Change(coord=(0, 'A'), 1 -> 2)" in str_repr

    def test_data_modified_exception_many_changes(self):
        """Test DataModifiedException with many changes (truncation)."""
        from finlab_guard.utils.exceptions import Change, DataModifiedException

        # Create more than 5 changes
        changes = [Change((i, "A"), i, i + 1, datetime(2023, 1, 1)) for i in range(10)]

        exc = DataModifiedException("Test message", changes)
        str_repr = str(exc)

        # Should show first 5 and indicate truncation
        assert "Test message" in str_repr
        assert "... and 5 more changes" in str_repr

    def test_other_exceptions(self):
        """Test other exception classes."""
        # Test FinlabConnectionException
        exc1 = FinlabConnectionException("Connection failed")
        assert str(exc1) == "Connection failed"

        # Test UnsupportedDataFormatException
        exc2 = UnsupportedDataFormatException("Format not supported")
        assert str(exc2) == "Format not supported"

        # Test InvalidDataTypeException
        exc3 = InvalidDataTypeException("Invalid type")
        assert str(exc3) == "Invalid type"
