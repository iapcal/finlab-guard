"""Real finlab integration tests."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from finlab_guard.core.guard import FinlabGuard
from finlab_guard.utils.exceptions import FinlabConnectionException

pytestmark = pytest.mark.real_finlab


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield str(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def finlab_guard(temp_cache_dir):
    """Create FinlabGuard instance."""
    guard_instance = FinlabGuard(cache_dir=temp_cache_dir)
    yield guard_instance
    # Ensure DuckDB connection is closed to prevent Windows file locking
    guard_instance.close()


@pytest.mark.real_finlab
class TestRealFinlabIntegration:
    """Test with real finlab package."""

    def test_finlab_import_and_login(self, finlab_available):
        """Test that finlab can be imported and logged in."""
        assert finlab_available, "Finlab should be available for testing"

        import finlab

        # Login should already be done in finlab_available fixture
        print(f"Finlab version: {finlab.__version__}")

    def test_original_finlab_get_works(self, finlab_available):
        """Test that original finlab.data.get works."""
        if not finlab_available:
            pytest.skip("Finlab not available")

        import finlab.data

        # Test a simple data fetch
        try:
            data = finlab.data.get("price:收盤價")
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            print(f"Successfully fetched data shape: {data.shape}")
            print(f"Data columns: {list(data.columns[:5])}...")  # Show first 5 columns
            print(f"Data index: {list(data.index[:5])}...")  # Show first 5 rows
        except Exception as e:
            pytest.fail(f"Original finlab.data.get failed: {e}")

    def test_monkey_patch_installation(self, finlab_available, finlab_guard):
        """Test monkey patch installation."""
        if not finlab_available:
            pytest.skip("Finlab not available")

        import finlab.data

        # Ensure no previous patch
        if hasattr(finlab.data, "_original_get"):
            delattr(finlab.data, "_original_get")

        original_get = finlab.data.get

        # Install patch
        finlab_guard.install_patch()

        # Verify patch is installed
        assert hasattr(finlab.data, "_original_get")
        assert finlab.data._original_get == original_get
        assert finlab.data.get != original_get

        print("✓ Monkey patch installed successfully")

        # Clean up
        finlab_guard.remove_patch()

    def test_monkey_patch_removal(self, finlab_available, finlab_guard):
        """Test monkey patch removal."""
        if not finlab_available:
            pytest.skip("Finlab not available")

        import finlab.data

        original_get = finlab.data.get

        # Install and then remove patch
        finlab_guard.install_patch()
        finlab_guard.remove_patch()

        # Verify patch is removed
        assert not hasattr(finlab.data, "_original_get")
        assert finlab.data.get == original_get

        print("✓ Monkey patch removed successfully")

    def test_patched_data_fetch(self, finlab_available, finlab_guard):
        """Test data fetching through patched function."""
        if not finlab_available:
            pytest.skip("Finlab not available")

        import finlab.data

        # Install patch
        finlab_guard.install_patch()

        try:
            # First fetch should cache the data
            data1 = finlab.data.get("price:收盤價")
            assert isinstance(data1, pd.DataFrame)
            assert not data1.empty

            # Verify data is cached
            assert finlab_guard.cache_manager.exists("price:收盤價")

            # Second fetch should come from cache (same result)
            data2 = finlab.data.get("price:收盤價")
            pd.testing.assert_frame_equal(data1, data2)

            print(f"✓ Successfully fetched and cached data: {data1.shape}")

        finally:
            # Clean up
            finlab_guard.remove_patch()

    def test_change_detection_with_real_data(self, finlab_available, finlab_guard):
        """Test change detection with real finlab data."""
        if not finlab_available:
            pytest.skip("Finlab not available")

        import finlab.data

        # Install patch
        finlab_guard.install_patch()

        try:
            # Fetch data twice - should not detect changes
            data1 = finlab.data.get("price:收盤價")
            data2 = finlab.data.get("price:收盤價")

            # Should be identical
            pd.testing.assert_frame_equal(data1, data2)
            print("✓ No changes detected in consecutive fetches")

        finally:
            # Clean up
            finlab_guard.remove_patch()

    def test_time_context_with_real_data(self, finlab_available, finlab_guard):
        """Test time context functionality with real data."""
        if not finlab_available:
            pytest.skip("Finlab not available")

        from datetime import datetime

        import finlab.data

        # Install patch
        finlab_guard.install_patch()

        try:
            # Fetch and cache data
            data = finlab.data.get("price:收盤價")

            # Get timestamp of when data was cached
            raw_data = finlab_guard.cache_manager.load_raw_data("price:收盤價")
            if raw_data is not None and not raw_data.empty:
                cache_time = raw_data["save_time"].max()

                # Set time context to cache time
                finlab_guard.set_time_context(cache_time)

                # Fetch data in time context - should get cached version
                historical_data = finlab.data.get("price:收盤價")

                # Should be same as original
                pd.testing.assert_frame_equal(data, historical_data)
                print("✓ Time context working with real data")

                # Clear time context
                finlab_guard.clear_time_context()

        finally:
            # Clean up
            finlab_guard.remove_patch()

    def test_multiple_datasets(self, finlab_available, finlab_guard):
        """Test with multiple different datasets."""
        if not finlab_available:
            pytest.skip("Finlab not available")

        import finlab.data

        # Install patch
        finlab_guard.install_patch()

        try:
            # Test multiple datasets
            datasets = ["price:收盤價", "price:開盤價"]

            for dataset in datasets:
                try:
                    data = finlab.data.get(dataset)
                    assert isinstance(data, pd.DataFrame)

                    # Verify cached
                    assert finlab_guard.cache_manager.exists(dataset)

                    print(f"✓ Successfully cached {dataset}: {data.shape}")
                except Exception as e:
                    print(f"Note: {dataset} not available: {e}")

        finally:
            # Clean up
            finlab_guard.remove_patch()

    def test_error_handling_with_invalid_dataset(self, finlab_available, finlab_guard):
        """Test error handling with invalid dataset key."""
        if not finlab_available:
            pytest.skip("Finlab not available")

        import finlab.data

        # Install patch
        finlab_guard.install_patch()

        try:
            # Try to fetch invalid dataset - FinlabGuard wraps all finlab errors as FinlabConnectionException
            with pytest.raises(
                FinlabConnectionException, match="Cannot fetch data from finlab"
            ):
                finlab.data.get("invalid:dataset:key")

        finally:
            # Clean up
            finlab_guard.remove_patch()
