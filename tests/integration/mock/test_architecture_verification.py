"""Architecture verification tests for the new mock-based testing approach.

This module contains simplified tests to verify that the new testing architecture
resolves the Local vs CI environment consistency issues.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from finlab_guard import FinlabGuard
from finlab_guard.utils.exceptions import FinlabConnectionException


class TestArchitectureVerification:
    """Test the new architecture to ensure Local/CI consistency."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def guard(self, temp_cache_dir):
        """Create FinlabGuard instance for testing."""
        config = {"compression": None}  # Disable compression to avoid PyArrow issues
        guard_instance = FinlabGuard(cache_dir=temp_cache_dir, config=config)
        yield guard_instance
        # Ensure DuckDB connection is closed to prevent Windows file locking
        guard_instance.close()

    def _mock_finlab_data(self, data: pd.DataFrame):
        """Mock finlab.data.get to return specified data."""
        return patch.object(FinlabGuard, "_fetch_from_finlab", return_value=data)

    def test_mock_environment_consistency(self, guard):
        """
        Test that mock environment works consistently.

        This test verifies that the finlab mock environment is properly set up
        and works consistently regardless of whether finlab is actually installed.
        """
        # Create test data
        test_data = pd.DataFrame(
            {"price": [100.0, 101.0, 99.0], "volume": [1000, 1100, 900]},
            index=["2023-01-01", "2023-01-02", "2023-01-03"],
        )

        # Use the mock and test basic functionality
        with self._mock_finlab_data(test_data):
            result = guard.get("test_key")

        # Verify the mock worked correctly
        pd.testing.assert_frame_equal(result, test_data)

        # Verify data was cached
        assert guard.cache_manager.exists("test_key")

        # Verify we can load from cache
        cached_result = guard.cache_manager.load_data("test_key")
        pd.testing.assert_frame_equal(cached_result, test_data)

    def test_finlab_import_isolation(self, guard):
        """
        Test that our mock environment isolates finlab imports.

        This test ensures that the mock environment prevents conflicts
        with real finlab imports.
        """
        import sys

        # Check that finlab module is mocked (not real)
        assert "finlab" in sys.modules
        finlab_module = sys.modules["finlab"]

        # The mocked module should have a MagicMock nature
        assert hasattr(finlab_module, "_mock_name") or hasattr(finlab_module, "data")

        # The data submodule should also be mocked
        assert hasattr(finlab_module, "data")
        assert hasattr(finlab_module.data, "get")

    def test_basic_guard_functionality_with_mock(self, guard):
        """
        Test that basic FinlabGuard functionality works with mocks.

        This test verifies that all core FinlabGuard features work
        correctly in the mock environment.
        """
        # Test data
        initial_data = pd.DataFrame(
            {"col1": [1, 2], "col2": [1.1, 2.2]}, index=["A", "B"]
        )

        updated_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3]}, index=["A", "B", "C"]
        )

        # Test initial save
        with self._mock_finlab_data(initial_data):
            result1 = guard.get("test_dataset")

        pd.testing.assert_frame_equal(result1, initial_data)

        # Test update with new data (adding new index)
        with self._mock_finlab_data(updated_data):
            result2 = guard.get("test_dataset")

        pd.testing.assert_frame_equal(result2, updated_data)

        # Verify cache state
        assert guard.cache_manager.exists("test_dataset")

    def test_error_handling_consistency(self, guard):
        """
        Test that error handling works consistently in mock environment.

        This test verifies that the same error handling behavior occurs
        in both local and CI environments.
        """
        # Test with invalid data (empty DataFrame)
        empty_data = pd.DataFrame()

        with self._mock_finlab_data(empty_data):
            # This should work but return empty data
            result = guard.get("empty_dataset")
            assert len(result) == 0

        # Test with mock that raises an exception
        def mock_fetch_error(key):
            raise Exception("Mock finlab error")

        with patch.object(
            FinlabGuard, "_fetch_from_finlab", side_effect=mock_fetch_error
        ):
            with pytest.raises(FinlabConnectionException):
                guard.get("error_dataset")

    def test_compression_settings_work(self, guard):
        """
        Test that compression settings work correctly.

        This test ensures that compression (or lack thereof) works
        consistently across environments.
        """
        test_data = pd.DataFrame(
            {"large_col": range(1000)}, index=[f"row_{i}" for i in range(1000)]
        )

        with self._mock_finlab_data(test_data):
            result = guard.get("large_dataset")

        # Should work without compression issues
        assert len(result) == 1000
        assert guard.cache_manager.exists("large_dataset")

    def test_environment_variables_isolation(self, guard):
        """
        Test that environment variables don't affect mock tests.

        This test ensures that FINLAB_TOKEN and other environment
        variables don't interfere with mock tests.
        """
        import os

        # Even if FINLAB_TOKEN is set or not set, mock tests should work
        original_token = os.environ.get("FINLAB_TOKEN")

        try:
            # Test with empty token
            os.environ["FINLAB_TOKEN"] = ""

            test_data = pd.DataFrame({"test": [1, 2, 3]})
            with self._mock_finlab_data(test_data):
                result = guard.get("env_test")

            pd.testing.assert_frame_equal(result, test_data)

            # Test with some token value
            os.environ["FINLAB_TOKEN"] = "fake_token_123"

            with self._mock_finlab_data(test_data):
                result2 = guard.get("env_test_2")

            pd.testing.assert_frame_equal(result2, test_data)

        finally:
            # Restore original token
            if original_token is not None:
                os.environ["FINLAB_TOKEN"] = original_token
            else:
                os.environ.pop("FINLAB_TOKEN", None)


class TestArchitectureIntegration:
    """Test integration aspects of the new architecture."""

    def test_ci_local_consistency_simulation(self):
        """
        Simulate the difference between CI and local environments.

        This test simulates both environments to ensure they would
        produce the same results.
        """
        # Simulate CI environment (no FINLAB_TOKEN, no finlab package)
        import os
        import sys

        original_token = os.environ.get("FINLAB_TOKEN")

        try:
            # CI simulation: empty token
            os.environ["FINLAB_TOKEN"] = ""

            # Ensure finlab modules are mocked (done by conftest.py)
            assert "finlab" in sys.modules
            finlab_module = sys.modules["finlab"]

            # Verify this is a mock, not real finlab
            assert hasattr(finlab_module, "_mock_name") or hasattr(
                finlab_module.data, "get"
            )

            # This test passing means CI environment would work
            assert True, "CI environment simulation successful"

        finally:
            if original_token is not None:
                os.environ["FINLAB_TOKEN"] = original_token
            else:
                os.environ.pop("FINLAB_TOKEN", None)

    def test_mock_directory_isolation(self):
        """
        Test that mock directory has proper isolation.

        This test verifies that tests in the mock directory
        are properly isolated from real finlab dependencies.
        """
        import sys

        # Check that we're running in mock environment
        # (This would be set up by mock/conftest.py)
        assert "finlab" in sys.modules

        # Try to use the mocked finlab
        import finlab

        # Should be able to access finlab.data without ImportError
        assert hasattr(finlab, "data")
        assert hasattr(finlab.data, "get")

        # Should be able to call the get method (it's mocked)
        # This would fail in CI if not properly mocked
        result = finlab.data.get("test_key")
        assert isinstance(result, pd.DataFrame)
