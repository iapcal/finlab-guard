"""Simplified integration tests for monkey patch functionality in finlab-guard.

This module tests the essential monkey patching functionality with a focus
on the core mechanisms rather than complex mocking scenarios.

These tests intentionally disable the automatic finlab mocking to test
error handling when finlab is not available.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from finlab_guard import FinlabGuard


class TestMonkeyPatchSimplified:
    """Simplified tests for monkey patch integration."""

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

    @pytest.fixture(autouse=True)
    def disable_finlab_mock(self):
        """Disable the automatic finlab mocking for these tests."""
        # Remove finlab modules to simulate missing finlab
        finlab_modules = [
            module for module in sys.modules.keys() if module.startswith("finlab")
        ]
        removed_modules = {}
        for module in finlab_modules:
            removed_modules[module] = sys.modules.pop(module, None)

        # Also ensure that any patch.dict from conftest is overridden
        with patch.dict("sys.modules", {}, clear=False):
            # Remove finlab modules from the patched dict as well
            if "finlab" in sys.modules:
                sys.modules.pop("finlab", None)
            if "finlab.data" in sys.modules:
                sys.modules.pop("finlab.data", None)
            yield

        # Restore modules after test
        for module, mod_obj in removed_modules.items():
            if mod_obj is not None:
                sys.modules[module] = mod_obj

    def test_patch_without_finlab_module(self, guard):
        """
        Test patch installation when finlab module is not available.

        This test verifies that the system handles missing finlab gracefully.
        """

        # Mock import to simulate missing finlab module
        def mock_import(name, *args, **kwargs):
            if name == "finlab.data" or name == "finlab":
                raise ImportError("No module named 'finlab'")
            return __import__(name, *args, **kwargs)

        # Test installing patch when finlab is not available
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="finlab package not found"):
                guard.install_patch()

    def test_patch_removal_when_not_installed(self, guard):
        """
        Test removing patch when no patch is installed.

        This test verifies proper error handling for patch removal.
        """
        # Test removing patch when none is installed (should just log warning, not raise)
        # This should complete without error, just log a warning
        guard.remove_patch()  # Should complete successfully

    def test_singleton_pattern_enforcement(self, guard):
        """
        Test that the global singleton pattern is enforced.

        This test verifies that only one patch can be active at a time.
        """
        # Create a second guard instance
        temp_dir2 = tempfile.mkdtemp()
        try:
            guard2 = FinlabGuard(cache_dir=temp_dir2)

            # In mock environment with available finlab, test actual singleton behavior
            # First guard installs patch successfully
            guard.install_patch()

            # Second guard should fail to install due to singleton constraint
            with pytest.raises(RuntimeError, match="finlab-guard already installed"):
                guard2.install_patch()

            # First guard can remove patch
            guard.remove_patch()

            # Now second guard should be able to install
            guard2.install_patch()

            # Clean up
            guard2.remove_patch()
            guard2.close()  # Ensure DuckDB connection is closed

        finally:
            shutil.rmtree(temp_dir2)

    def test_patch_state_consistency(self, guard):
        """
        Test that patch state is consistently managed.

        This test verifies that the patch state tracking works correctly.
        """
        # Test that remove_patch works without error when nothing is installed
        # (it should just log a warning, not raise an exception)
        guard.remove_patch()  # Should complete successfully

        # Test install and remove cycle
        guard.install_patch()

        # Verify second install fails due to singleton
        temp_dir2 = tempfile.mkdtemp()
        try:
            guard2 = FinlabGuard(cache_dir=temp_dir2)
            with pytest.raises(RuntimeError, match="finlab-guard already installed"):
                guard2.install_patch()
            guard2.close()  # Ensure DuckDB connection is closed
        finally:
            shutil.rmtree(temp_dir2)

        # Remove patch
        guard.remove_patch()

    def test_patch_error_messages(self, guard):
        """
        Test that appropriate error messages are provided.

        This test verifies that error messages are helpful and informative.
        """

        # Test missing finlab error message by mocking import to fail
        def mock_import(name, *args, **kwargs):
            if name == "finlab.data" or name == "finlab":
                raise ImportError("No module named 'finlab'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            try:
                guard.install_patch()
                raise AssertionError("Should have raised ImportError")
            except ImportError as e:
                assert "finlab package not found" in str(e)

        # Test patch not installed - should just log warning, not raise
        guard.remove_patch()  # Should complete successfully without exception

    def test_multiple_guard_instances_patch_state(self, guard):
        """
        Test patch state across multiple guard instances.

        This test verifies that the global singleton state is properly
        managed across different guard instances.
        """
        # Create multiple guard instances
        temp_dirs = []
        guards = []

        try:
            for _i in range(3):
                temp_dir = tempfile.mkdtemp()
                temp_dirs.append(temp_dir)
                g = FinlabGuard(cache_dir=temp_dir)
                guards.append(g)

            # All should fail to install patches without finlab
            def mock_import(name, *args, **kwargs):
                if name == "finlab.data" or name == "finlab":
                    raise ImportError("No module named 'finlab'")
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                for g in guards:
                    with pytest.raises(ImportError):
                        g.install_patch()

            # All should handle removing non-existent patches gracefully
            for g in guards:
                g.remove_patch()  # Should complete without exception

            # Close all guard connections
            for g in guards:
                g.close()  # Ensure DuckDB connections are closed

        finally:
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir)

    def test_guard_functionality_without_patches(self, guard):
        """
        Test that guard functionality works normally without patches installed.

        This test verifies that the guard system works independently
        of the monkey patching functionality.
        """
        # Test basic guard functionality
        test_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3]}, index=["A", "B", "C"]
        )

        # This would normally require a finlab connection, but we can test
        # the cache manager directly
        from datetime import datetime

        key = "test_key"
        timestamp = datetime.now()
        guard.cache_manager.save_data(key, test_data, timestamp)

        # Verify data was saved
        assert guard.cache_manager.exists(key)

        # Load the data back
        loaded_data = guard.cache_manager.load_data(key)
        pd.testing.assert_frame_equal(loaded_data, test_data)

        # Test time context functionality
        from datetime import datetime, timedelta

        past_time = datetime.now() - timedelta(hours=1)
        guard.set_time_context(past_time)
        assert guard.get_time_context() == past_time

        guard.clear_time_context()
        assert guard.get_time_context() is None


class TestPatchIntegrationConcepts:
    """Test the conceptual integration of patch functionality."""

    def test_patch_workflow_concept(self):
        """
        Test the conceptual workflow of patch installation.

        This test documents and verifies the expected workflow
        without requiring actual finlab module presence.
        """
        # The expected workflow is:
        # 1. Check if finlab is available
        # 2. Check if patch is already installed
        # 3. Save original function
        # 4. Install patched function
        # 5. Set global state

        # Since we can't test with real finlab, we verify the error paths
        temp_dir = tempfile.mkdtemp()
        try:
            guard = FinlabGuard(cache_dir=temp_dir)

            # Step 1: Check finlab availability (should fail)
            def mock_import(name, *args, **kwargs):
                if name == "finlab.data" or name == "finlab":
                    raise ImportError("No module named 'finlab'")
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError):
                    guard.install_patch()

            # Patch removal should handle gracefully when no patch installed
            guard.remove_patch()  # Should complete without exception
            guard.close()  # Ensure DuckDB connection is closed

        finally:
            shutil.rmtree(temp_dir)

    def test_global_state_management(self):
        """
        Test global state management concepts.

        This test verifies that the global state tracking
        mechanisms work as expected.
        """
        # Test that state management works correctly
        temp_dir = tempfile.mkdtemp()
        try:
            # Test instance behavior - should handle gracefully when no patch installed
            guard = FinlabGuard(cache_dir=temp_dir)
            guard.remove_patch()  # Should complete without exception
            guard.close()  # Ensure DuckDB connection is closed

        finally:
            shutil.rmtree(temp_dir)
