"""Configuration for real finlab integration tests.

This module provides configuration for tests that require actual finlab
package and API connectivity.
"""

import os

import pytest


def reset_pyarrow_state():
    """Reset PyArrow extension type registry to avoid conflicts."""
    try:
        import pyarrow

        # Method 1: Clear the extension types registry
        if hasattr(pyarrow, "_extension_types_registry"):
            registry = pyarrow._extension_types_registry
            # Store pandas-related types for removal
            pandas_types = [k for k in registry.keys() if k.startswith("pandas.")]
            for k in pandas_types:
                registry.pop(k, None)

        # Method 2: Clear the global extension type map if it exists
        if hasattr(pyarrow, "_extension_type_registry"):
            pyarrow._extension_type_registry.clear()

        # Method 3: Try to access the lib extension registry
        try:
            import pyarrow.lib as lib

            if hasattr(lib, "_extension_type_registry"):
                lib._extension_type_registry.clear()
        except (ImportError, AttributeError):
            pass

    except ImportError:
        # PyArrow not installed, nothing to reset
        pass
    except Exception:
        # If any reset method fails, just continue
        pass


@pytest.fixture
def finlab_token():
    """Get finlab token from environment."""
    return os.getenv("FINLAB_TOKEN")


@pytest.fixture
def finlab_available(finlab_token):
    """Check if finlab is available for real testing."""
    if not finlab_token:
        return False

    try:
        import finlab

        # Set the token
        finlab.login(finlab_token)
        return True
    except Exception as e:
        print(f"Finlab not available: {e}")
        return False


def pytest_configure(config):
    """Configure pytest markers for real finlab tests."""
    # Reset PyArrow state first
    reset_pyarrow_state()

    config.addinivalue_line(
        "markers", "real_finlab: marks tests that require real finlab connection"
    )

    # Suppress PyArrow warnings
    import warnings

    warnings.filterwarnings("ignore", message=".*pandas.period already defined.*")
    warnings.filterwarnings("ignore", message=".*extension.*already.*registered.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pyarrow")


def pytest_collection_modifyitems(config, items):
    """Skip tests if finlab is not available."""
    finlab_token = os.getenv("FINLAB_TOKEN")

    if not finlab_token:
        # Skip all tests in this directory if no token
        skip_real = pytest.mark.skip(reason="No FINLAB_TOKEN provided")
        for item in items:
            item.add_marker(skip_real)
    else:
        # Try to import and login to verify finlab is working
        try:
            import finlab

            finlab.login(finlab_token)
        except Exception:
            skip_real = pytest.mark.skip(reason="Finlab not available or login failed")
            for item in items:
                item.add_marker(skip_real)


def pytest_runtest_setup(item):
    """Ensure finlab is properly set up for each test."""
    # Reset PyArrow state before each test
    reset_pyarrow_state()

    finlab_token = os.getenv("FINLAB_TOKEN")
    if finlab_token:
        try:
            import finlab

            # Re-login for each test to ensure fresh session
            finlab.login(finlab_token)
        except Exception as e:
            pytest.skip(f"Failed to setup finlab: {e}")
