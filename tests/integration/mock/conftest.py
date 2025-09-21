"""Configuration for mock integration tests.

This module provides complete finlab mocking environment to ensure
that mock tests run consistently across all environments without
requiring actual finlab package installation.
"""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def mock_finlab_environment():
    """
    Automatically mock finlab environment for all tests in this directory.

    This fixture ensures that:
    1. No real finlab module is imported
    2. All finlab calls are intercepted with mock objects
    3. Tests run consistently regardless of finlab installation
    """
    # Remove any existing finlab modules from sys.modules
    finlab_modules = [
        module for module in sys.modules.keys() if module.startswith("finlab")
    ]
    for module in finlab_modules:
        sys.modules.pop(module, None)

    # Create comprehensive mock finlab environment
    mock_finlab = MagicMock()
    mock_data_module = MagicMock()

    # Set up the module hierarchy
    mock_finlab.data = mock_data_module
    mock_finlab.__version__ = "1.0.0"

    # Mock the data.get function with realistic behavior
    def mock_get(key, **kwargs):
        """Mock implementation of finlab.data.get"""
        # Return simple test data - this will be overridden by test-specific mocks
        return pd.DataFrame({"default_col": [1, 2, 3]}, index=["A", "B", "C"])

    mock_data_module.get = mock_get

    # Mock login function
    def mock_login(token):
        """Mock implementation of finlab.login"""
        return True

    mock_finlab.login = mock_login

    # Install mocks in sys.modules to intercept all imports
    with patch.dict(
        "sys.modules",
        {
            "finlab": mock_finlab,
            "finlab.data": mock_data_module,
        },
    ):
        yield mock_finlab, mock_data_module


@pytest.fixture
def mock_finlab_data():
    """
    Convenience fixture for creating test-specific finlab data mocks.

    Returns a function that can be used to mock specific datasets.
    """

    def _mock_data(key, data):
        """Mock finlab.data.get for a specific key"""
        return patch.object(
            sys.modules["finlab.data"],
            "get",
            side_effect=lambda k, **kwargs: data if k == key else pd.DataFrame(),
        )

    return _mock_data


@pytest.fixture
def sample_finlab_data():
    """Provide sample data that mimics real finlab datasets."""
    return {
        "price:收盤價": pd.DataFrame(
            {
                "AAPL": [150.0, 152.0, 148.0],
                "GOOGL": [2800.0, 2820.0, 2790.0],
                "TSLA": [800.0, 810.0, 795.0],
            },
            index=pd.date_range("2023-01-01", periods=3, name="date"),
        ),
        "fundamental:營收": pd.DataFrame(
            {
                "AAPL": [100000, 105000, 103000],
                "GOOGL": [250000, 255000, 252000],
                "TSLA": [50000, 52000, 51000],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="QS", name="date"),
        ),
        "technical:RSI": pd.DataFrame(
            {
                "AAPL": [65.5, 70.2, 45.8],
                "GOOGL": [55.1, 62.3, 58.9],
                "TSLA": [75.4, 80.1, 42.6],
            },
            index=pd.date_range("2023-01-01", periods=3, name="date"),
        ),
    }


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


def pytest_configure():
    """Configure pytest for mock integration tests."""
    # Reset PyArrow state first
    reset_pyarrow_state()

    # Ensure we start with a clean slate for finlab modules
    finlab_modules = [
        module for module in sys.modules.keys() if module.startswith("finlab")
    ]
    for module in finlab_modules:
        sys.modules.pop(module, None)

    # Suppress PyArrow warnings
    import warnings

    warnings.filterwarnings("ignore", message=".*pandas.period already defined.*")
    warnings.filterwarnings("ignore", message=".*extension.*already.*registered.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pyarrow")


def pytest_runtest_setup(item):
    """Setup for each test to ensure mock environment."""
    # Reset PyArrow state before each test
    reset_pyarrow_state()

    # Verify that we're in mock mode - no real finlab should be importable
    # without our mocks
    if "finlab" in sys.modules and not hasattr(sys.modules["finlab"], "_mock_name"):
        # This means a real finlab module somehow got imported
        # Remove it to ensure we use our mocks
        finlab_modules = [
            module for module in sys.modules.keys() if module.startswith("finlab")
        ]
        for module in finlab_modules:
            sys.modules.pop(module, None)
