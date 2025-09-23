"""Tests for FinlabGuard context manager functionality."""

import shutil
import tempfile
import time

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


class TestContextManager:
    """Test FinlabGuard context manager functionality."""

    def test_context_manager_normal_exit(self):
        """Test context manager normal exit behavior."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}

            # Test normal context manager usage
            with FinlabGuard(cache_dir=temp_dir, config=config) as guard:
                assert isinstance(guard, FinlabGuard)
                # The __exit__ method should be called with None parameters
                # This covers lines 338-339 (parameter cleanup)

        finally:
            safe_rmtree(temp_dir)

    def test_context_manager_exception_exit(self):
        """Test context manager exit with exception handling."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = {"compression": None}

            # Test context manager with exception
            try:
                with FinlabGuard(cache_dir=temp_dir, config=config) as guard:
                    assert isinstance(guard, FinlabGuard)
                    # Trigger an exception to test __exit__ with exception parameters
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected - this tests the __exit__ method parameter handling

        finally:
            safe_rmtree(temp_dir)
