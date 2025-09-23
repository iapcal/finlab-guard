"""Tests for DataValidator fallback mechanisms."""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


class TestValidatorFallbacks:
    """Test DataValidator fallback mechanisms for edge cases."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        safe_rmtree(temp_dir)

    @pytest.fixture
    def guard(self, temp_cache_dir):
        """Create FinlabGuard instance for testing."""
        config = {"compression": None}
        guard_instance = FinlabGuard(cache_dir=temp_cache_dir, config=config)
        yield guard_instance
        # Ensure DuckDB connection is closed to prevent Windows file locking
        guard_instance.close()

    def test_compare_complex_types_fallback(self, guard):
        """Test comparison fallback for complex types (lines 181-183)."""
        # Create data with complex types that might fail direct comparison

        # Use a custom object that might cause comparison issues
        class ComplexType:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                # This will raise an exception to trigger fallback
                raise ValueError("Complex comparison failed")

            def __str__(self):
                return f"ComplexType({self.value})"

            def __repr__(self):
                return self.__str__()

        # Create DataFrames with complex objects
        complex_obj1 = ComplexType("value1")
        complex_obj2 = ComplexType("value2")

        df1 = pd.DataFrame(
            {"col1": [1, 2], "complex_col": [complex_obj1, complex_obj1]}
        )

        df2 = pd.DataFrame(
            {
                "col1": [1, 2],
                "complex_col": [complex_obj1, complex_obj2],  # Different in second row
            }
        )

        # Save the first DataFrame to establish a baseline
        from datetime import datetime

        guard.cache_manager.save_data("test_key", df1, datetime.now())

        # This should trigger the fallback mechanism (lines 181-183)
        # when direct comparison fails and falls back to string comparison
        modifications, additions = guard.validator.detect_changes_detailed(
            "test_key", df2, guard.cache_manager
        )

        # Should detect the change in complex_col[1]
        assert len(modifications) > 0

    def test_get_cell_value_edge_cases(self, guard):
        """Test _get_value_at_coord with edge cases (lines 269-270)."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        validator = guard.validator

        # Test with invalid indices - should trigger lines 269-270
        # This should return None when indices are out of bounds
        result1 = validator._get_value_at_coord(df, "nonexistent_row", "col1")
        assert result1 is None

        result2 = validator._get_value_at_coord(df, 0, "nonexistent_col")
        assert result2 is None

        result3 = validator._get_value_at_coord(df, 999, "col1")  # Out of bounds row
        assert result3 is None

    def test_values_equal_complex_comparison_fallback(self, guard):
        """Test _values_equal fallback for complex value comparison (lines 295-297)."""
        validator = guard.validator

        # Create objects that will fail direct comparison but work with string comparison
        class ComplexValue:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                # This will raise an exception to trigger fallback
                raise TypeError("Cannot compare complex values directly")

            def __str__(self):
                return f"ComplexValue({self.value})"

        val1 = ComplexValue("same")
        val2 = ComplexValue("same")  # Same string representation
        val3 = ComplexValue("different")

        # These should trigger the fallback mechanism (lines 295-297)
        # When direct comparison fails, it should fall back to string comparison

        # Same string representation should be equal
        assert validator._values_equal(val1, val2)

        # Different string representation should not be equal
        assert not validator._values_equal(val1, val3)

    def test_values_equal_with_various_types(self, guard):
        """Test _values_equal with various data types."""
        validator = guard.validator

        # Test normal comparisons work
        assert validator._values_equal(1, 1)
        assert not validator._values_equal(1, 2)
        assert validator._values_equal("a", "a")
        assert not validator._values_equal("a", "b")

        # Test NaN handling
        assert validator._values_equal(np.nan, np.nan)
        assert not validator._values_equal(np.nan, 1)
        assert not validator._values_equal(1, np.nan)

        # Test with pandas NA
        assert validator._values_equal(pd.NA, pd.NA)
        assert not validator._values_equal(pd.NA, 1)
