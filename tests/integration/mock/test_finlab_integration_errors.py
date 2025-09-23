"""Tests for finlab integration error handling in FinlabGuard."""

import shutil
import tempfile
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from finlab_guard.core.guard import FinlabGuard
from finlab_guard.utils.exceptions import FinlabConnectionException


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


class TestFinlabFetchErrorHandling:
    """Test error handling in _fetch_from_finlab method."""

    def setup_method(self):
        """Clean up global state before each test."""
        import finlab_guard.core.guard as guard_module

        guard_module._global_guard_instance = None

    def teardown_method(self):
        """Clean up global state after each test."""
        import finlab_guard.core.guard as guard_module

        guard_module._global_guard_instance = None

    def test_fetch_from_finlab_import_error(self):
        """Test _fetch_from_finlab when finlab module is not available."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock import to fail for finlab
            original_import = __import__
            with patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "finlab.data" or name == "finlab":
                        raise ImportError("No module named 'finlab'")
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                # This should trigger line 226 (ImportError handling)
                with pytest.raises(
                    FinlabConnectionException, match="finlab package not found"
                ):
                    guard._fetch_from_finlab("test_key")

            guard.close()

        finally:
            safe_rmtree(temp_dir)

    def test_fetch_from_finlab_wrong_type(self):
        """Test _fetch_from_finlab when finlab returns non-DataFrame."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock finlab.data to return non-DataFrame that will fail conversion
            mock_finlab = MagicMock()
            mock_finlab.data = MagicMock()
            # Use a string that will fail DataFrame conversion
            mock_finlab.data.get = MagicMock(return_value="not_a_dataframe")

            # Ensure _original_get doesn't exist to force use of get()
            if hasattr(mock_finlab.data, "_original_get"):
                delattr(mock_finlab.data, "_original_get")

            with patch.dict(
                "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
            ):
                # This should trigger lines 218-224 (type coercion failure)
                with pytest.raises(
                    FinlabConnectionException, match="finlab returned unexpected type"
                ):
                    guard._fetch_from_finlab("test_key")

            guard.close()

        finally:
            safe_rmtree(temp_dir)

    def test_fetch_from_finlab_conversion_failure(self):
        """Test _fetch_from_finlab when DataFrame conversion fails."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock finlab.data to return something that can't be converted to DataFrame
            mock_finlab = MagicMock()
            mock_finlab.data = MagicMock()

            # Create an object that will fail DataFrame conversion
            bad_data = object()  # This will definitely fail pd.DataFrame() conversion
            mock_finlab.data.get = MagicMock(return_value=bad_data)

            # Ensure _original_get doesn't exist to force use of get()
            if hasattr(mock_finlab.data, "_original_get"):
                delattr(mock_finlab.data, "_original_get")

            with patch.dict(
                "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
            ):
                # This should trigger lines 219-224 (DataFrame conversion exception)
                with pytest.raises(
                    FinlabConnectionException, match="finlab returned unexpected type"
                ):
                    guard._fetch_from_finlab("test_key")

            guard.close()

        finally:
            safe_rmtree(temp_dir)

    def test_fetch_from_finlab_with_original_get(self):
        """Test _fetch_from_finlab using _original_get when available."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock finlab with _original_get
            mock_finlab = MagicMock()
            mock_finlab.data = MagicMock()

            test_data = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]})
            mock_finlab.data._original_get = MagicMock(return_value=test_data)
            mock_finlab.data.get = MagicMock(return_value="should_not_be_called")

            with patch.dict(
                "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
            ):
                # This should trigger line 210 (calling _original_get)
                result = guard._fetch_from_finlab("test_key")

                # Verify _original_get was called, not regular get
                mock_finlab.data._original_get.assert_called_once_with("test_key")
                mock_finlab.data.get.assert_not_called()

                pd.testing.assert_frame_equal(result, test_data)

            guard.close()

        finally:
            safe_rmtree(temp_dir)


class TestMonkeyPatchErrorHandling:
    """Test error handling in monkey patch operations."""

    def setup_method(self):
        """Clean up global state before each test."""
        import finlab_guard.core.guard as guard_module

        guard_module._global_guard_instance = None

    def teardown_method(self):
        """Clean up global state after each test."""
        import finlab_guard.core.guard as guard_module

        guard_module._global_guard_instance = None

    def test_remove_patch_import_error(self):
        """Test remove_patch when finlab import fails."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock import to fail for finlab
            original_import = __import__
            with patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "finlab.data" or name == "finlab":
                        raise ImportError("No module named 'finlab'")
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                # This should trigger line 281-282 (ImportError in remove_patch)
                # Just verify that remove_patch completes without raising an exception
                # when finlab import fails
                guard.remove_patch()

            guard.close()

        finally:
            safe_rmtree(temp_dir)

    def test_remove_patch_general_exception(self):
        """Test remove_patch when general exception occurs."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock finlab but cause a general exception during access
            mock_finlab = MagicMock()
            mock_finlab.data = MagicMock()

            # Make accessing finlab.data raise a general exception
            type(mock_finlab).data = PropertyMock(
                side_effect=RuntimeError("Finlab initialization error")
            )

            with patch.dict("sys.modules", {"finlab": mock_finlab}):
                # This should trigger lines 283-286 (general exception handling)
                # Just verify that remove_patch completes without raising an exception
                # when finlab access fails
                guard.remove_patch()

            guard.close()

        finally:
            safe_rmtree(temp_dir)

    def test_patched_get_call_coverage(self):
        """Test the patched_get function call to cover line 252."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock finlab module
            mock_finlab = MagicMock()
            mock_finlab.data = MagicMock()
            mock_finlab.data.get = MagicMock()

            # Ensure _original_get doesn't exist initially
            if hasattr(mock_finlab.data, "_original_get"):
                delattr(mock_finlab.data, "_original_get")

            test_data = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]})

            with patch.dict(
                "sys.modules", {"finlab": mock_finlab, "finlab.data": mock_finlab.data}
            ):
                # Install patch first
                guard.install_patch()

                # Mock the guard.get method to return test data
                with patch.object(guard, "get", return_value=test_data) as mock_get:
                    # Call the patched function - this should trigger line 252
                    result = mock_finlab.data.get("test_key")

                    # Verify the patched function called guard.get
                    mock_get.assert_called_once_with("test_key")
                    pd.testing.assert_frame_equal(result, test_data)

                guard.remove_patch()

            guard.close()

        finally:
            safe_rmtree(temp_dir)
