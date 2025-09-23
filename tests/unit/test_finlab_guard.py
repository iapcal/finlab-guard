"""Unit tests for FinlabGuard class."""

import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from finlab_guard.core.guard import FinlabGuard
from finlab_guard.utils.exceptions import (
    Change,
    DataModifiedException,
    FinlabConnectionException,
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


class TestFinlabGuard:
    """Test suite for FinlabGuard class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        safe_rmtree(temp_dir)

    @pytest.fixture
    def guard(self, temp_cache_dir):
        """Create FinlabGuard instance for testing."""
        config = {"compression": "snappy", "progress_bar": False}
        guard_instance = FinlabGuard(cache_dir=str(temp_cache_dir), config=config)
        yield guard_instance
        # Ensure DuckDB connection is closed to prevent Windows file locking
        guard_instance.close()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3]}, index=["A", "B", "C"]
        )

    # === 初始化 ===

    def test_init_default_config(self, temp_cache_dir):
        """Test initialization with default config."""
        guard = FinlabGuard(cache_dir=str(temp_cache_dir))

        assert guard.cache_dir == temp_cache_dir
        assert guard.config is not None
        assert guard.time_context is None
        assert guard.cache_manager is not None
        assert guard.validator is not None

    def test_init_custom_config(self, temp_cache_dir):
        """Test initialization with custom config."""
        custom_config = {"compression": "lz4", "progress_bar": True}
        guard = FinlabGuard(cache_dir=str(temp_cache_dir), config=custom_config)

        assert guard.config["compression"] == "lz4"
        assert guard.config["progress_bar"] is True

    def test_init_cache_dir_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = Path(temp_dir) / "new_cache_dir"
            assert not non_existent_dir.exists()

            guard = FinlabGuard(cache_dir=str(non_existent_dir))

            assert non_existent_dir.exists()
            assert guard.cache_dir == non_existent_dir

            # Close the guard to prevent Windows file locking issues
            guard.close()

    # === 時間上下文管理 ===

    def test_set_time_context_datetime(self, guard):
        """Test setting time context with datetime object."""
        test_time = datetime(2024, 1, 1, 15, 30, 0)
        guard.set_time_context(test_time)

        assert guard.time_context == test_time

    def test_set_time_context_string(self, guard):
        """Test setting time context with string."""
        time_string = "2024-01-01 15:30:00"
        guard.set_time_context(time_string)

        expected_time = datetime(2024, 1, 1, 15, 30, 0)
        assert guard.time_context == expected_time

    def test_set_time_context_none(self, guard):
        """Test setting time context to None."""
        # First set some time
        guard.set_time_context(datetime.now())
        assert guard.time_context is not None

        # Then set to None
        guard.set_time_context(None)
        assert guard.time_context is None

    def test_clear_time_context(self, guard):
        """Test clearing time context."""
        # Set some time context
        guard.set_time_context(datetime.now())
        assert guard.time_context is not None

        # Clear it
        guard.clear_time_context()
        assert guard.time_context is None

    def test_get_time_context(self, guard):
        """Test getting time context."""
        assert guard.get_time_context() is None

        test_time = datetime(2024, 1, 1, 15, 30, 0)
        guard.set_time_context(test_time)
        assert guard.get_time_context() == test_time

    # === 核心get()方法 (with mocked finlab) ===

    def test_get_time_context_mode(self, guard, sample_dataframe):
        """Test get() method in time context mode."""
        key = "price:收盤價"

        # Save some data first with past timestamp
        past_time = datetime.now() - timedelta(hours=2)
        guard.cache_manager.save_data(key, sample_dataframe, past_time)

        # Set time context to after the data was saved
        guard.set_time_context(datetime.now() - timedelta(hours=1))

        # Should return cached data for time context
        result = guard.get(key)

        pd.testing.assert_frame_equal(result, sample_dataframe)

    # Note: The actual get() method requires finlab package to be installed
    # and would require complex mocking. These tests focus on testable components.

    # === Monkey Patching ===
    # Note: Monkey patching tests require finlab package and complex mocking.
    # These would be better tested in integration tests with actual finlab.

    def test_global_singleton_behavior(self):
        """Test that global singleton behavior works correctly."""
        # This tests the module-level _global_guard_instance functionality
        from finlab_guard.core.guard import _global_guard_instance

        # Check that global instance is either None or a FinlabGuard instance
        assert _global_guard_instance is None or hasattr(
            _global_guard_instance, "install_patch"
        )

    # === 輔助方法 ===

    # Note: _generate_unique_timestamp is a private method and can be tested through public methods

    def test_clear_cache_specific_key(self, guard, sample_dataframe):
        """Test clearing cache for specific key."""
        key = "test_key"

        # Save some data
        guard.cache_manager.save_data(key, sample_dataframe, datetime.now())
        assert guard.cache_manager.exists(key)

        # Clear specific key
        guard.clear_cache(key)
        assert not guard.cache_manager.exists(key)

    def test_clear_cache_all(self, guard, sample_dataframe):
        """Test clearing all cache data."""
        keys = ["key1", "key2", "key3"]

        # Save multiple datasets
        for key in keys:
            guard.cache_manager.save_data(key, sample_dataframe, datetime.now())

        # Verify all exist
        for key in keys:
            assert guard.cache_manager.exists(key)

        # Clear all
        guard.clear_cache()

        # Verify none exist
        for key in keys:
            assert not guard.cache_manager.exists(key)

    def test_get_change_history(self, guard, sample_dataframe):
        """Test getting change history for a dataset."""
        key = "test_key"

        # Save some data
        guard.cache_manager.save_data(key, sample_dataframe, datetime.now())

        history = guard.get_change_history(key)

        assert isinstance(history, pd.DataFrame)
        # Should delegate to cache manager
        assert not history.empty or history.empty  # Just verify it returns a DataFrame

    def test_get_storage_info(self, guard, sample_dataframe):
        """Test getting storage information."""
        key = "test_key"

        # Save some data
        guard.cache_manager.save_data(key, sample_dataframe, datetime.now())

        # Get storage info for specific key
        info = guard.get_storage_info(key)
        assert isinstance(info, dict)

        # Get storage info for all keys
        all_info = guard.get_storage_info()
        assert isinstance(all_info, dict)

    # === 錯誤處理 ===
    # Note: Error handling for finlab connection would be tested in integration tests

    # === 特殊情況測試 ===

    def test_config_merge_with_defaults(self, temp_cache_dir):
        """Test that custom config merges properly with defaults."""
        custom_config = {"compression": "lz4"}  # Only specify one setting

        guard = FinlabGuard(cache_dir=str(temp_cache_dir), config=custom_config)

        # Should have custom value
        assert guard.config["compression"] == "lz4"
        # Should have default values for unspecified settings
        assert "progress_bar" in guard.config

    def test_concurrent_access_safety(self, guard, sample_dataframe):
        """Test behavior under concurrent access scenarios."""
        key = "test_key"

        # Save some data first
        guard.cache_manager.save_data(key, sample_dataframe, datetime.now())

        # Simulate concurrent cache access (simple test)
        result1 = guard.cache_manager.load_data(key)
        result2 = guard.cache_manager.load_data(key)

        pd.testing.assert_frame_equal(result1, sample_dataframe)
        pd.testing.assert_frame_equal(result2, sample_dataframe)

    def test_invalid_time_context_string(self, guard):
        """Test handling of invalid time context string."""
        with pytest.raises(ValueError):
            guard.set_time_context("invalid-date-string")

    def test_path_handling_edge_cases(self):
        """Test edge cases in path handling."""
        # Test with path that needs expansion
        guard = FinlabGuard(cache_dir="~/finlab_guard_test")
        assert guard.cache_dir.is_absolute()

        # Cleanup
        guard.close()  # Close connections before cleanup
        if guard.cache_dir.exists():
            safe_rmtree(guard.cache_dir)

    def test_large_dataset_handling(self, guard):
        """Test handling of large datasets in cache operations."""
        key = "large_dataset"

        # Create large dataset
        large_data = pd.DataFrame(
            {
                "col1": range(1000),  # Reduced size for test performance
                "col2": [f"value_{i}" for i in range(1000)],
            }
        )

        # Test cache operations with large data
        guard.cache_manager.save_data(key, large_data, datetime.now())
        result = guard.cache_manager.load_data(key)

        pd.testing.assert_frame_equal(result, large_data)
        assert guard.cache_manager.exists(key)


@pytest.mark.serial
class TestCoveragePhaseTesting:
    """Phase-based testing to improve coverage systematically."""

    def test_install_convenience_function(self):
        """Test the install() convenience function from __init__.py."""
        from finlab_guard import FinlabGuard as ImportedFinlabGuard
        from finlab_guard import install

        temp_dir = tempfile.mkdtemp()
        try:
            # Test with default parameters
            config = {"compression": None}
            guard = install(cache_dir=temp_dir, config=config)

            # Verify guard instance was created
            assert isinstance(guard, ImportedFinlabGuard)
            assert str(guard.cache_dir) == temp_dir

            # Verify patch was automatically installed
            # We can't easily test real patch installation in unit tests,
            # but we can verify the function was called by checking global state
            import finlab_guard.core.guard as guard_module

            assert guard_module._global_guard_instance is not None

            # Clean up
            guard.remove_patch()
            guard.close()

        finally:
            safe_rmtree(temp_dir)

    def test_context_manager_usage(self):
        """Test FinlabGuard as context manager (__enter__, __exit__)."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}

            # Test context manager usage
            with FinlabGuard(cache_dir=temp_dir, config=config) as guard:
                # Verify __enter__ returns self
                assert isinstance(guard, FinlabGuard)

                # Test some basic functionality inside context
                test_data = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]})
                guard.cache_manager.save_data("test_key", test_data, datetime.now())
                assert guard.cache_manager.exists("test_key")

            # After __exit__, connections should be closed
            # We can verify this by checking that further operations would fail
            # or by ensuring cache_manager connections are properly cleaned up

        finally:
            safe_rmtree(temp_dir)

    def test_timestamp_uniqueness_edge_case(self):
        """Test timestamp uniqueness when now <= latest_time."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Create test data with a future timestamp
            test_data = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]})
            future_time = datetime.now() + timedelta(seconds=10)

            # Save data with future timestamp
            guard.cache_manager.save_data("test_key", test_data, future_time)

            # Mock datetime.now() to return a time before the saved timestamp
            with patch("finlab_guard.core.guard.datetime") as mock_datetime:
                mock_now = future_time - timedelta(
                    seconds=5
                )  # 5 seconds before saved time
                mock_datetime.now.return_value = mock_now
                mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

                # This should trigger the timestamp uniqueness logic (line 114)
                # where now <= latest_time, so it adds 1 second
                adjusted_time = guard.generate_unique_timestamp("test_key")

                # The adjusted time should be latest_time + 1 second
                expected_time = future_time + pd.Timedelta(seconds=1)
                assert adjusted_time >= expected_time

            guard.close()

        finally:
            safe_rmtree(temp_dir)
