"""Unit tests for DataValidator class."""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from finlab_guard.cache.manager import CacheManager
from finlab_guard.cache.validator import DataValidator
from finlab_guard.utils.exceptions import (
    Change,
    InvalidDataTypeException,
    UnsupportedDataFormatException,
)


class TestDataValidator:
    """Test suite for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance for testing."""
        return DataValidator()

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock CacheManager for testing."""
        return Mock(spec=CacheManager)

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3]}, index=["A", "B", "C"]
        )

    # === 格式驗證 ===

    def test_validate_dataframe_format_valid(self, validator, sample_dataframe):
        """Test validation passes for valid DataFrame."""
        # Should not raise any exceptions
        validator.validate_dataframe_format(sample_dataframe)

    def test_validate_dataframe_format_not_dataframe(self, validator):
        """Test validation fails for non-DataFrame input."""
        with pytest.raises(InvalidDataTypeException) as exc_info:
            validator.validate_dataframe_format("not a dataframe")

        assert "Expected DataFrame" in str(exc_info.value)

    def test_validate_dataframe_format_multiindex_columns(self, validator):
        """Test validation fails for MultiIndex columns."""
        # Create DataFrame with MultiIndex columns
        columns = pd.MultiIndex.from_tuples([("A", "X"), ("A", "Y"), ("B", "X")])
        df = pd.DataFrame(np.random.randn(3, 3), columns=columns)

        with pytest.raises(UnsupportedDataFormatException) as exc_info:
            validator.validate_dataframe_format(df)

        assert "MultiIndex columns are not supported" in str(exc_info.value)

    def test_validate_dataframe_format_multiindex_index(self, validator):
        """Test validation fails for MultiIndex index."""
        # Create DataFrame with MultiIndex index
        index = pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1)])
        df = pd.DataFrame(np.random.randn(3, 2), index=index, columns=["X", "Y"])

        with pytest.raises(UnsupportedDataFormatException) as exc_info:
            validator.validate_dataframe_format(df)

        assert "MultiIndex index is not supported" in str(exc_info.value)

    # === 變更檢測核心 ===

    def test_detect_changes_detailed_no_existing_data(
        self, validator, mock_cache_manager, sample_dataframe
    ):
        """Test change detection when no existing data exists."""
        # Mock empty existing data
        mock_cache_manager.get_latest_data.return_value = pd.DataFrame()

        modifications, additions = validator.detect_changes_detailed(
            "test_key", sample_dataframe, mock_cache_manager
        )

        assert len(modifications) == 0
        assert len(additions) == 6  # 3 rows × 2 columns

        # Verify all changes are additions
        for change in additions:
            assert change.old_value is None
            assert change.new_value is not None

    def test_detect_changes_detailed_no_changes(
        self, validator, mock_cache_manager, sample_dataframe
    ):
        """Test change detection when no changes occurred."""
        # Mock existing data identical to new data
        mock_cache_manager.get_latest_data.return_value = sample_dataframe.copy()

        modifications, additions = validator.detect_changes_detailed(
            "test_key", sample_dataframe, mock_cache_manager
        )

        assert len(modifications) == 0
        assert len(additions) == 0

    def test_detect_changes_detailed_modifications_only(
        self, validator, mock_cache_manager, sample_dataframe
    ):
        """Test change detection with only modifications."""
        # Create existing data
        existing_data = sample_dataframe.copy()
        mock_cache_manager.get_latest_data.return_value = existing_data

        # Create new data with modifications
        new_data = sample_dataframe.copy()
        new_data.loc["A", "col1"] = 99  # Modify one value
        new_data.loc["B", "col2"] = 88.8  # Modify another value

        modifications, additions = validator.detect_changes_detailed(
            "test_key", new_data, mock_cache_manager
        )

        assert len(modifications) == 2
        assert len(additions) == 0

        # Verify modification details
        mod_coords = [change.coord for change in modifications]
        assert ("A", "col1") in mod_coords
        assert ("B", "col2") in mod_coords

        # Verify old and new values
        for change in modifications:
            if change.coord == ("A", "col1"):
                assert change.old_value == 1
                assert change.new_value == 99
            elif change.coord == ("B", "col2"):
                assert change.old_value == 2.2
                assert change.new_value == 88.8

    def test_detect_changes_detailed_additions_only(
        self, validator, mock_cache_manager
    ):
        """Test change detection with only additions."""
        # Create existing data
        existing_data = pd.DataFrame(
            {"col1": [1, 2], "col2": [1.1, 2.2]}, index=["A", "B"]
        )
        mock_cache_manager.get_latest_data.return_value = existing_data

        # Create new data with additional rows and columns
        new_data = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [1.1, 2.2, 3.3],
                "col3": [10, 20, 30],  # New column
            },
            index=["A", "B", "C"],
        )  # New row

        modifications, additions = validator.detect_changes_detailed(
            "test_key", new_data, mock_cache_manager
        )

        assert len(modifications) == 0
        assert len(additions) == 5  # C,col1 + C,col2 + A,col3 + B,col3 + C,col3

        # Verify all are additions
        for change in additions:
            assert change.old_value is None

    def test_detect_changes_detailed_mixed_changes(
        self, validator, mock_cache_manager, sample_dataframe
    ):
        """Test change detection with both modifications and additions."""
        # Create existing data
        existing_data = sample_dataframe.copy()
        mock_cache_manager.get_latest_data.return_value = existing_data

        # Create new data with both modifications and additions
        new_data = pd.DataFrame(
            {
                "col1": [99, 2, 3, 4],  # Modified A, added D
                "col2": [1.1, 2.2, 3.3, 4.4],  # Added D
                "col3": [10, 20, 30, 40],  # New column
            },
            index=["A", "B", "C", "D"],
        )

        modifications, additions = validator.detect_changes_detailed(
            "test_key", new_data, mock_cache_manager
        )

        assert len(modifications) == 1  # A,col1 modified
        assert len(additions) > 0  # New row D and new column col3

        # Verify modification
        mod = modifications[0]
        assert mod.coord == ("A", "col1")
        assert mod.old_value == 1
        assert mod.new_value == 99

    def test_detect_changes_detailed_large_dataset(self, validator, mock_cache_manager):
        """Test change detection performance with larger dataset."""
        # Create large existing data
        size = 1000
        existing_data = pd.DataFrame(
            {"value": range(size), "squared": [i**2 for i in range(size)]},
            index=[f"row_{i}" for i in range(size)],
        )
        mock_cache_manager.get_latest_data.return_value = existing_data

        # Create new data with some modifications
        new_data = existing_data.copy()
        new_data.loc["row_500", "value"] = 999999  # Single modification

        with patch("time.time", side_effect=[0, 1]):  # Mock timing
            modifications, additions = validator.detect_changes_detailed(
                "test_key", new_data, mock_cache_manager
            )

        assert len(modifications) == 1
        assert len(additions) == 0
        assert modifications[0].coord == ("row_500", "value")

    # === 輔助方法 ===

    def test_get_coordinates(self, validator, sample_dataframe):
        """Test coordinate extraction from DataFrame."""
        coordinates = validator._get_coordinates(sample_dataframe)

        # Should have 6 coordinates (3 rows × 2 columns)
        assert len(coordinates) == 6

        # Verify specific coordinates exist
        assert ("A", "col1") in coordinates
        assert ("C", "col2") in coordinates

    def test_get_value_at_coord_exists(self, validator, sample_dataframe):
        """Test getting value at existing coordinate."""
        value = validator._get_value_at_coord(sample_dataframe, "A", "col1")
        assert value == 1

    def test_get_value_at_coord_not_exists(self, validator, sample_dataframe):
        """Test getting value at non-existent coordinate."""
        # Non-existent row
        value = validator._get_value_at_coord(sample_dataframe, "Z", "col1")
        assert value is None

        # Non-existent column
        value = validator._get_value_at_coord(sample_dataframe, "A", "nonexistent")
        assert value is None

    def test_values_equal_both_nan(self, validator):
        """Test value comparison when both values are NaN."""
        assert validator._values_equal(float("nan"), float("nan"))

    def test_values_equal_one_nan(self, validator):
        """Test value comparison when one value is NaN."""
        assert not validator._values_equal(float("nan"), 1)
        assert not validator._values_equal(1, float("nan"))

    def test_values_equal_normal(self, validator):
        """Test value comparison with normal values."""
        assert validator._values_equal(1, 1)
        assert validator._values_equal(1.5, 1.5)
        assert validator._values_equal("hello", "hello")
        assert not validator._values_equal(1, 2)
        assert not validator._values_equal("hello", "world")

    def test_values_equal_complex_types(self, validator):
        """Test value comparison with complex types."""
        # Test with objects that don't support direct comparison
        obj1 = {"key": "value"}
        obj2 = {"key": "value"}

        # Should fall back to string comparison
        assert validator._values_equal(obj1, obj2)

    def test_create_changes_from_dataframe(self, validator):
        """Test creating Change objects from DataFrame."""
        df = pd.DataFrame(
            {
                "col1": [1, 2],
                "col2": [1.1, np.nan],  # Include NaN value
            },
            index=["A", "B"],
        )
        timestamp = datetime.now()

        # Test additions
        changes = validator._create_changes_from_dataframe(
            df, timestamp, is_addition=True
        )

        # Should only include non-NaN values
        assert len(changes) == 3  # A,col1 + A,col2 + B,col1 (B,col2 is NaN, excluded)

        for change in changes:
            assert change.old_value is None  # All are additions
            assert change.new_value is not None
            assert change.timestamp == timestamp

    def test_create_changes_from_dataframe_modifications(self, validator):
        """Test creating Change objects for modifications."""
        df = pd.DataFrame({"col1": [1, 2]}, index=["A", "B"])
        timestamp = datetime.now()

        # Test modifications
        changes = validator._create_changes_from_dataframe(
            df, timestamp, is_addition=False
        )

        assert len(changes) == 2

        for change in changes:
            assert (
                change.old_value == change.new_value
            )  # For modifications, old_value = new_value initially
            assert change.timestamp == timestamp

    # === 特殊情況測試 ===

    def test_detect_changes_with_nan_values(self, validator, mock_cache_manager):
        """Test change detection with NaN values."""
        # Existing data with NaN
        existing_data = pd.DataFrame(
            {"col1": [1, np.nan, 3], "col2": [1.1, 2.2, np.nan]}, index=["A", "B", "C"]
        )
        mock_cache_manager.get_latest_data.return_value = existing_data

        # New data with different NaN pattern
        new_data = pd.DataFrame(
            {
                "col1": [1, 2, 3],  # B,col1 changed from NaN to 2
                "col2": [1.1, 2.2, 3.3],  # C,col2 changed from NaN to 3.3
            },
            index=["A", "B", "C"],
        )

        modifications, additions = validator.detect_changes_detailed(
            "test_key", new_data, mock_cache_manager
        )

        # Should detect additions where NaN became a value
        assert len(additions) == 0
        assert len(modifications) == 2  # B,col1 and C,col2

    def test_detect_changes_with_different_dtypes(self, validator, mock_cache_manager):
        """Test change detection with different data types."""
        # Existing data
        existing_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}, index=["A", "B", "C"]
        )
        mock_cache_manager.get_latest_data.return_value = existing_data

        # New data with same values but different dtypes
        new_data = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0],  # int -> float
                "col2": ["a", "b", "c"],
            },
            index=["A", "B", "C"],
        )

        modifications, additions = validator.detect_changes_detailed(
            "test_key", new_data, mock_cache_manager
        )

        # Values are same, so no changes should be detected
        assert len(modifications) == 0
        assert len(additions) == 0

    def test_detect_changes_empty_dataframes(self, validator, mock_cache_manager):
        """Test change detection with empty DataFrames."""
        # Test with empty existing data
        mock_cache_manager.get_latest_data.return_value = pd.DataFrame()
        empty_new_data = pd.DataFrame()

        modifications, additions = validator.detect_changes_detailed(
            "test_key", empty_new_data, mock_cache_manager
        )

        assert len(modifications) == 0
        assert len(additions) == 0

    def test_float64_precision_issue(self, validator, mock_cache_manager):
        """Test that DataValidator handles float64 precision issues correctly."""
        # Create original data with float64 precision
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        stocks = ["2330", "2317"]

        # Generate realistic adjusted close prices that might have precision issues
        data = {}
        for stock in stocks:
            base_price = np.random.uniform(100, 500)
            prices = base_price * (1 + np.cumsum(np.random.normal(0, 0.01, len(dates))))
            data[stock] = prices

        original_df = pd.DataFrame(data, index=dates, dtype="float64")

        # Simulate precision differences that can occur during parquet save/load
        # Add tiny differences at machine epsilon level
        loaded_df = original_df.copy()

        # Add differences at different precision levels
        # 1. Machine epsilon level differences (should not be detected as changes)
        eps_diff = np.finfo(np.float64).eps
        loaded_df.iloc[0, 0] += eps_diff * 0.5  # Very small difference

        # 2. Slightly larger but still insignificant differences
        loaded_df.iloc[1, 0] += 1e-14  # Still very small

        # 3. Actual meaningful difference (should be detected)
        loaded_df.iloc[2, 0] += 0.01  # Meaningful difference

        # Mock the cache manager to return original data
        mock_cache_manager.get_latest_data.return_value = original_df

        # Test with default tolerance (should detect the meaningful change only)
        modifications, additions = validator.detect_changes_detailed(
            "test_key", loaded_df, mock_cache_manager
        )

        # Should only detect the meaningful change, not the precision artifacts
        assert len(modifications) == 1
        assert modifications[0].coord[0] == dates[2]  # Row with meaningful change
        assert modifications[0].coord[1] == stocks[0]  # First stock
        # The actual difference should be close to 0.01 (allowing for floating point precision)
        actual_diff = abs(modifications[0].new_value - modifications[0].old_value)
        assert actual_diff > 1e-12  # Much larger than tolerance
        assert 0.009 < actual_diff < 0.011  # Close to the 0.01 we added

    def test_nan_safe_diff_precision_tolerance(self, validator, mock_cache_manager):
        """Test that nan_safe_diff correctly handles precision with tolerance."""
        # Create data that would trigger false positives with direct comparison
        original = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=["x", "y", "z"]
        )

        # Create "identical" data with tiny precision differences
        modified = original.copy()
        modified.iloc[0, 0] += 1e-15  # Machine epsilon level
        modified.iloc[1, 1] += 2e-15  # Another tiny difference

        # Direct comparison would show differences
        direct_diff = original != modified
        assert direct_diff.sum().sum() > 0  # Sanity check

        # Mock the cache manager to return original data
        mock_cache_manager.get_latest_data.return_value = original

        # But our validator should not detect changes with default tolerance
        modifications, additions = validator.detect_changes_detailed(
            "test_key", modified, mock_cache_manager
        )
        assert len(modifications) == 0

        # Test with zero tolerance - should detect the tiny differences
        strict_validator = DataValidator(tolerance=0.0)
        strict_modifications, strict_additions = (
            strict_validator.detect_changes_detailed(
                "test_key", modified, mock_cache_manager
            )
        )
        assert len(strict_modifications) == 2  # Should detect both tiny changes

    def test_adj_close_realistic_scenario(self, validator, mock_cache_manager):
        """Test realistic adj_close scenario that caused the original issue."""
        # Simulate typical adj_close data structure
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        stocks = ["2330", "2317", "2454", "0050"]

        # Create realistic adjusted close prices
        np.random.seed(123)
        adj_close_data = {}
        for stock in stocks:
            # Start with a base price and create realistic price movements
            base_price = np.random.uniform(50, 300)
            daily_returns = np.random.normal(0.001, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(daily_returns))
            adj_close_data[stock] = prices

        original_adj_close = pd.DataFrame(adj_close_data, index=dates, dtype="float64")

        # Simulate the save/load cycle that might introduce precision issues
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save and reload (common source of precision issues)
            original_adj_close.to_parquet(tmp_path)
            reloaded_adj_close = pd.read_parquet(tmp_path)

            # Mock the cache manager to return original data
            mock_cache_manager.get_latest_data.return_value = original_adj_close

            # The validator should not detect spurious changes from precision issues
            modifications, additions = validator.detect_changes_detailed(
                "etl:adj_close", reloaded_adj_close, mock_cache_manager
            )

            # With proper tolerance, should detect very few or no changes
            # (any changes should be genuine differences, not precision artifacts)
            assert (
                len(modifications) <= 2
            )  # Allow for occasional genuine precision loss

            # If there are changes, they should be small
            for change in modifications:
                rel_diff = abs(change.new_value - change.old_value) / abs(
                    change.old_value
                )
                assert rel_diff < 1e-10  # Very small relative difference

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
