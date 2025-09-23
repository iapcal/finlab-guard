"""Test configuration for finlab-guard.

This is the main test configuration file. The specific test directories
have their own conftest.py files with specialized configurations:

- tests/integration/mock/conftest.py: Mock finlab environment
- tests/integration/real/conftest.py: Real finlab integration
"""

import os
import sys
from pathlib import Path

import pytest

# Load environment variables from .env file if it exists
# Only load if the environment variable is not already set
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # Only set if not already in environment (allows override)
                if key not in os.environ:
                    os.environ[key] = value


@pytest.fixture(autouse=True, scope="function")
def force_cleanup_global_state():
    """Force cleanup of all global state before and after each test.

    This fixture addresses issues with parallel test execution where
    global state can leak between tests, causing 'already installed' errors.
    """

    def _cleanup():
        # Clean up finlab-guard global state
        try:
            import finlab_guard.core.guard as guard_module

            guard_module._global_guard_instance = None
        except ImportError:
            pass

        # Clean up finlab module state from sys.modules
        # This prevents state from leaking between tests
        finlab_modules = [key for key in sys.modules.keys() if key.startswith("finlab")]
        for module_name in finlab_modules:
            if module_name != "finlab_guard.core.guard":  # Keep our own module
                module = sys.modules.get(module_name)
                if (
                    module
                    and hasattr(module, "data")
                    and hasattr(module.data, "_original_get")
                ):
                    # Clean up any lingering patch state
                    try:
                        delattr(module.data, "_original_get")
                    except (AttributeError, TypeError):
                        pass

    # Clean up before test
    _cleanup()

    yield

    # Clean up after test
    _cleanup()


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "real_finlab: marks tests that require real finlab connection"
    )
    config.addinivalue_line(
        "markers",
        "serial: marks tests that should run serially to avoid state conflicts",
    )
    # Note: mock_only marker is deprecated in favor of directory structure


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle serial markers."""
    # Add serial marker to items that need it based on path or class name
    for item in items:
        # Add serial marker to monkey patch related tests
        if (
            "monkey_patch" in str(item.fspath)
            or "TestCoveragePhaseTesting" in item.name
            or "TestMonkeyPatchIntegration" in item.name
            or "TestPatchStatePersistence" in item.name
        ):
            item.add_marker(pytest.mark.serial)
