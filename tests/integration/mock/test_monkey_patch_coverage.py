"""Tests for monkey patch implementation coverage."""

import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import pandas as pd

from finlab_guard.core.guard import FinlabGuard


def safe_rmtree(path, ignore_errors=False):
    """Windows-compatible rmtree with retry logic for DuckDB file locking."""
    for attempt in range(5):
        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            break
        except (PermissionError, OSError):
            if attempt < 4:
                time.sleep(0.1)  # Wait 100ms before retry
            else:
                raise


class TestMonkeyPatchCoverage:
    """Test monkey patch implementation details for coverage."""

    def setup_method(self):
        """Clean up global state before each test."""
        import finlab_guard.core.guard as guard_module

        guard_module._global_guard_instance = None

    def teardown_method(self):
        """Clean up global state after each test."""
        import finlab_guard.core.guard as guard_module

        guard_module._global_guard_instance = None

    def test_patched_get_function_call(self):
        """Test the actual patched_get function execution (line 252)."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Create test data
            test_data = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]})

            # Mock finlab module
            mock_finlab = MagicMock()
            mock_finlab.data = MagicMock()
            mock_finlab.data.get = MagicMock()

            # Ensure _original_get doesn't exist initially
            if hasattr(mock_finlab.data, "_original_get"):
                delattr(mock_finlab.data, "_original_get")

            with patch.dict(
                "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
            ):
                # Install the patch
                guard.install_patch()

                # Mock guard.get to return test data
                with patch.object(guard, "get", return_value=test_data) as mock_get:
                    # Call the patched function - this executes line 252
                    result = mock_finlab.data.get("test_key", force_download=True)

                    # Verify the patched function called guard.get
                    mock_get.assert_called_once_with("test_key", force_download=True)
                    pd.testing.assert_frame_equal(result, test_data)

                # Clean up
                guard.remove_patch()

            guard.close()

        finally:
            safe_rmtree(temp_dir)

    def test_remove_patch_import_error_handling(self):
        """Test remove_patch ImportError handling (lines 281-282)."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock import to fail for finlab during remove_patch
            original_import = __import__
            with patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "finlab.data" or name == "finlab":
                        raise ImportError("No module named 'finlab'")
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                # Test remove_patch with ImportError
                # Just verify that remove_patch completes without raising an exception
                guard.remove_patch()

            guard.close()

        finally:
            safe_rmtree(temp_dir)
