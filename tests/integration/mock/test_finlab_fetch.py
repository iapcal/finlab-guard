"""Tests for finlab data fetching functionality."""

import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

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


class TestFinlabFetch:
    """Test finlab data fetching functionality."""

    def test_fetch_from_finlab_with_original_get(self):
        """Test _fetch_from_finlab using _original_get path (line 210)."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock finlab with _original_get attribute
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

    def test_fetch_from_finlab_import_error(self):
        """Test _fetch_from_finlab ImportError handling (line 226)."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}
            guard = FinlabGuard(cache_dir=temp_dir, config=config)

            # Mock import to fail for finlab
            with patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "finlab.data" or name == "finlab":
                        raise ImportError("No module named 'finlab'")
                    return __import__(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                # This should trigger line 226 (ImportError handling)
                with pytest.raises(
                    FinlabConnectionException, match="finlab package not found"
                ):
                    guard._fetch_from_finlab("test_key")

            guard.close()

        finally:
            safe_rmtree(temp_dir)
