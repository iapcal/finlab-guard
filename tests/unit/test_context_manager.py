"""Tests for FinlabGuard context manager functionality."""

import shutil
import tempfile
import time

from finlab_guard.core.guard import FinlabGuard


def safe_rmtree(path, ignore_errors=False):
    """Windows-compatible rmtree with retry logic for DuckDB file locking."""
    import gc
    import shutil
    import time

    # Force garbage collection to release any lingering file handles
    gc.collect()

    for attempt in range(10):  # Increase retry attempts to 10
        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            break
        except (PermissionError, OSError):
            if attempt < 9:
                # Exponential backoff: 100ms, 200ms, 400ms, etc.
                wait_time = 0.1 * (2 ** min(attempt, 4))
                time.sleep(wait_time)
                gc.collect()  # Try garbage collection again
                continue
            if not ignore_errors:
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
