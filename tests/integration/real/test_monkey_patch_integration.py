"""Integration tests for monkey patch functionality in finlab-guard.

This module tests the complete monkey patching system including:
- Patch installation and removal
- Singleton enforcement
- Function interception
- Integration with the guard system
"""

import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from finlab_guard import FinlabGuard
from finlab_guard.utils.exceptions import DataModifiedException


@pytest.mark.serial
class TestMonkeyPatchIntegration:
    """Test complete monkey patch integration."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def guard(self, temp_cache_dir):
        """Create FinlabGuard instance for testing."""
        config = {"compression": "snappy"}
        guard_instance = FinlabGuard(cache_dir=temp_cache_dir, config=config)
        yield guard_instance
        # Ensure DuckDB connection is closed to prevent Windows file locking
        guard_instance.close()

    @pytest.fixture(autouse=True)
    def cleanup_patches(self):
        """Ensure patches are cleaned up after each test."""
        # Clean up before test starts
        try:
            FinlabGuard.remove_patch()
        except Exception:
            pass

        # Clear global guard instance
        import finlab_guard.core.guard as guard_module

        guard_module._global_guard_instance = None

        yield

        # Clean up after test ends
        try:
            FinlabGuard.remove_patch()
        except Exception:
            pass

        # Clear global guard instance again
        guard_module._global_guard_instance = None

    def test_patch_installation_and_interception(self, guard):
        """
        Test complete patch installation and function interception.

        This test verifies:
        1. Patch installation succeeds
        2. finlab.data.get is properly intercepted
        3. Guard functionality works through the patch
        4. Original function is preserved
        """
        # For this test, we'll test the patch mechanism indirectly
        # by checking that install_patch works and creates the expected state

        # Mock the import of finlab at the method level
        mock_finlab = MagicMock()
        mock_data_module = MagicMock()
        mock_finlab.data = mock_data_module

        test_data = pd.DataFrame(
            {"col1": [100, 200, 300], "col2": [1.1, 2.2, 3.3]}, index=["A", "B", "C"]
        )

        original_get_func = MagicMock(return_value=test_data)
        mock_data_module.get = original_get_func

        # Ensure the mock object doesn't have _original_get attribute initially
        if hasattr(mock_data_module, "_original_get"):
            delattr(mock_data_module, "_original_get")

        # Patch the import statement within the method
        with patch.dict(
            "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_data_module}
        ):
            # Test patch installation
            guard.install_patch()

            # Verify that the original function is preserved
            assert hasattr(mock_finlab.data, "_original_get")
            assert mock_finlab.data._original_get == original_get_func

            # Test that the patched function works
            key = "test_key"
            result = mock_finlab.data.get(key)

            # The result should come through our guard system
            pd.testing.assert_frame_equal(result, test_data)

            # Verify that our guard system was used (data should be cached)
            assert guard.cache_manager.exists(key)

    def test_patch_removal_restoration(self, guard):
        """
        Test patch removal and function restoration.

        This test verifies:
        1. Patch can be removed cleanly
        2. Original function is restored
        3. Guard functionality is disabled
        """
        mock_finlab = MagicMock()
        mock_data_module = MagicMock()
        mock_finlab.data = mock_data_module

        original_get = MagicMock(return_value=pd.DataFrame({"test": [1]}))
        mock_data_module.get = original_get

        # Ensure the mock object doesn't have _original_get attribute initially
        if hasattr(mock_data_module, "_original_get"):
            delattr(mock_data_module, "_original_get")

        with patch.dict(
            "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_data_module}
        ):
            # Install patch
            guard.install_patch()
            assert hasattr(mock_finlab.data, "_original_get")

            # Remove patch
            guard.remove_patch()

            # Verify restoration
            assert mock_finlab.data.get == original_get
            assert not hasattr(mock_finlab.data, "_original_get")

    def test_singleton_enforcement(self, guard):
        """
        Test that singleton pattern is enforced for patches.

        This test verifies:
        1. Only one guard can install patches at a time
        2. Attempting to install multiple patches fails appropriately
        3. Global state is managed correctly
        """
        # Create a second guard instance
        temp_dir2 = tempfile.mkdtemp()
        try:
            guard2 = FinlabGuard(cache_dir=temp_dir2)

            mock_finlab = MagicMock()
            mock_finlab.data = MagicMock()
            mock_finlab.data.get = MagicMock()

            with patch("finlab_guard.core.guard.finlab", mock_finlab, create=True):
                # First guard installs patch successfully
                guard.install_patch()

                # Second guard should fail to install patch
                with pytest.raises(
                    RuntimeError, match="finlab-guard already installed"
                ):
                    guard2.install_patch()

                # After removing the first patch, second guard should succeed
                guard.remove_patch()
                guard2.install_patch()  # Should not raise

                # Clean up
                guard2.remove_patch()
            guard2.close()  # Ensure DuckDB connection is closed

        finally:
            shutil.rmtree(temp_dir2)

    def test_concurrent_patch_attempts(self, guard):
        """
        Test concurrent patch installation attempts.

        This test verifies thread safety of the patch system.
        """
        mock_finlab = MagicMock()
        mock_finlab.data = MagicMock()
        mock_finlab.data.get = MagicMock()

        results = []
        errors = []

        def try_install_patch(guard_instance, result_list, error_list):
            try:
                guard_instance.install_patch()
                result_list.append(True)
            except Exception as e:
                error_list.append(str(e))

        with patch("finlab_guard.core.guard.finlab", mock_finlab, create=True):
            # Create multiple guard instances
            temp_dirs = []
            guards = []
            threads = []

            try:
                for _i in range(3):
                    temp_dir = tempfile.mkdtemp()
                    temp_dirs.append(temp_dir)
                    g = FinlabGuard(cache_dir=temp_dir)
                    guards.append(g)

                # Try to install patches concurrently
                for g in guards:
                    thread = threading.Thread(
                        target=try_install_patch, args=(g, results, errors)
                    )
                    threads.append(thread)
                    thread.start()

                # Wait for all threads
                for thread in threads:
                    thread.join()

                # Only one should succeed, others should fail
                assert len(results) == 1, "Only one patch installation should succeed"
                assert len(errors) == 2, "Two patch installations should fail"

                # Clean up the successful patch
                for g in guards:
                    try:
                        g.remove_patch()
                        break
                    except Exception:
                        continue

                # Close all guard connections
                for g in guards:
                    g.close()  # Ensure DuckDB connections are closed

            finally:
                for temp_dir in temp_dirs:
                    shutil.rmtree(temp_dir)

    def test_finlab_integration_real_calls(self, guard):
        """
        Test integration with realistic finlab call patterns.

        This test simulates real usage patterns and verifies:
        1. Multiple sequential calls work correctly
        2. Caching behavior is correct
        3. Time context works with patches
        """
        mock_finlab = MagicMock()
        mock_data_module = MagicMock()
        mock_finlab.data = mock_data_module

        # Simulate finlab data that changes over time
        datasets = {
            "price:收盤價": pd.DataFrame(
                {"AAPL": [150, 152, 148], "GOOGL": [2800, 2820, 2790]},
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "fundamental:營收": pd.DataFrame(
                {"AAPL": [100, 105, 103], "GOOGL": [250, 255, 252]},
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        def mock_get(key):
            return datasets.get(key, pd.DataFrame())

        mock_data_module.get = mock_get

        # Ensure the mock object doesn't have _original_get attribute initially
        if hasattr(mock_data_module, "_original_get"):
            delattr(mock_data_module, "_original_get")

        with patch.dict(
            "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_data_module}
        ):
            # Install patch
            guard.install_patch()

            # Test multiple calls to different datasets
            price_data = mock_finlab.data.get("price:收盤價")
            fundamental_data = mock_finlab.data.get("fundamental:營收")

            # Verify data is returned correctly
            assert len(price_data) == 3
            assert "AAPL" in price_data.columns
            assert len(fundamental_data) == 3

            # Verify caching occurred
            assert guard.cache_manager.exists("price:收盤價")
            assert guard.cache_manager.exists("fundamental:營收")

            # Test time context functionality with future time
            # Since we just cached the data, querying for current time should return it
            current_time = datetime.now()
            guard.set_time_context(current_time)

            try:
                # This should return cached data from the current cache
                historical_price = mock_finlab.data.get("price:收盤價")
                pd.testing.assert_frame_equal(historical_price, price_data)
            finally:
                guard.clear_time_context()

    def test_patch_error_handling(self, guard):
        """
        Test error handling in patch system.

        This test verifies:
        1. Graceful handling of missing finlab module
        2. Proper error messages
        3. System state remains consistent after errors
        """

        # Test with missing finlab module by patching the import statement directly
        def mock_import(name, *args, **kwargs):
            if name == "finlab.data" or name == "finlab":
                raise ImportError("No module named 'finlab'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="finlab package not found"):
                guard.install_patch()

        # Test removing patch when none is installed (should just log warning, not raise)
        # This should not raise an exception, just log a warning
        guard.remove_patch()  # Should complete without error

    def test_patch_with_data_modifications(self, guard):
        """
        Test patch behavior when data modifications are detected.

        This test verifies that the patch system correctly handles
        DataModifiedException scenarios.
        """
        mock_finlab = MagicMock()
        mock_data_module = MagicMock()
        mock_finlab.data = mock_data_module

        # Initial data
        initial_data = pd.DataFrame(
            {"col1": [100, 200], "col2": [1.1, 2.2]}, index=["A", "B"]
        )

        # Modified data (historical change)
        modified_data = pd.DataFrame(
            {
                "col1": [105, 200],  # A changed from 100 to 105
                "col2": [1.1, 2.2],
            },
            index=["A", "B"],
        )

        call_count = 0

        def mock_get(key, force_download=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return initial_data
            else:
                return modified_data

        mock_data_module.get = mock_get

        # Ensure the mock object doesn't have _original_get attribute initially
        if hasattr(mock_data_module, "_original_get"):
            delattr(mock_data_module, "_original_get")

        with patch.dict(
            "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_data_module}
        ):
            guard.install_patch()

            # First call - should succeed
            result1 = mock_finlab.data.get("test_key")
            pd.testing.assert_frame_equal(result1, initial_data)

            # Second call - should detect modification and raise exception
            with pytest.raises(DataModifiedException):
                mock_finlab.data.get("test_key")

            # Clean up
            guard.remove_patch()

            # Force download should work
            result3 = mock_finlab.data.get("test_key", force_download=True)
            pd.testing.assert_frame_equal(result3, modified_data)


@pytest.mark.serial
class TestPatchStatePersistence:
    """Test patch state persistence and recovery."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_patch_state_across_guard_instances(self, temp_cache_dir):
        """
        Test that patch state is properly managed across different guard instances.
        """
        mock_finlab = MagicMock()
        mock_finlab.data = MagicMock()
        mock_finlab.data.get = MagicMock()

        with patch("finlab_guard.core.guard.finlab", mock_finlab, create=True):
            # Create first guard and install patch
            guard1 = FinlabGuard(cache_dir=temp_cache_dir)
            guard1.install_patch()

            # Create second guard - should recognize existing patch
            guard2 = FinlabGuard(cache_dir=temp_cache_dir)
            with pytest.raises(RuntimeError, match="finlab-guard already installed"):
                guard2.install_patch()

            # First guard removes patch
            guard1.remove_patch()

            # Now second guard should be able to install
            guard2.install_patch()  # Should not raise

            # Clean up
            guard2.remove_patch()
            guard1.close()  # Ensure DuckDB connections are closed
            guard2.close()

    def test_global_singleton_behavior(self, temp_cache_dir):
        """
        Test that the global singleton behavior works correctly.
        """
        mock_finlab = MagicMock()
        mock_finlab.data = MagicMock()
        mock_finlab.data.get = MagicMock()

        # Ensure the mock object doesn't have _original_get attribute initially
        if hasattr(mock_finlab.data, "_original_get"):
            delattr(mock_finlab.data, "_original_get")

        with patch.dict(
            "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
        ):
            # Create first instance and install patch
            guard1 = FinlabGuard(cache_dir=temp_cache_dir)
            guard1.install_patch()

            # Second instance should recognize existing patch
            guard2 = FinlabGuard(cache_dir=temp_cache_dir)
            with pytest.raises(RuntimeError, match="finlab-guard already installed"):
                guard2.install_patch()

            # First instance can remove patch
            guard1.remove_patch()

            # Now second instance should be able to install
            guard2.install_patch()  # Should not raise

            # Clean up
            guard2.remove_patch()
            guard1.close()  # Ensure DuckDB connections are closed
            guard2.close()
