"""Tests for column_additions functionality."""

from datetime import datetime

import pandas as pd
import pytest

from finlab_guard.cache.manager import CacheManager


class TestColumnAdditions:
    """Test column additions functionality in cache manager."""

    @pytest.fixture
    def cache_manager(self, tmp_path):
        """Create a cache manager for testing."""
        config = {"compression": "snappy"}
        manager = CacheManager(tmp_path, config)
        yield manager
        # Ensure DuckDB connection is closed
        manager.close()

    def test_column_addition_basic(self, cache_manager):
        """Test basic column addition functionality."""
        # Initial DataFrame
        df1 = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["row1", "row2", "row3"]
        )

        # DataFrame with new column
        df2 = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [7, 8, 9],  # New column
            },
            index=["row1", "row2", "row3"],
        )

        timestamp = datetime.now()
        changes = cache_manager.get_changes_extended(df1, df2, timestamp)

        # Should have column additions, no cell changes
        assert len(changes.column_additions) == 1
        assert changes.column_additions.iloc[0]["col_key"] == "C"
        assert (
            len(changes.cell_changes) == 0
        )  # No cell changes since we use column_additions

        # Verify column data is stored correctly
        col_data_json = changes.column_additions.iloc[0]["col_data_json"]
        import json

        col_data = json.loads(col_data_json)
        # Values are stored as their native types in JSON (integers as integers)
        expected_data = {"row1": 7, "row2": 8, "row3": 9}
        assert col_data == expected_data

    def test_multiple_column_additions(self, cache_manager):
        """Test adding multiple columns at once."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["row1", "row2"])

        df2 = pd.DataFrame(
            {"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]}, index=["row1", "row2"]
        )

        timestamp = datetime.now()
        changes = cache_manager.get_changes_extended(df1, df2, timestamp)

        assert len(changes.column_additions) == 2
        col_keys = set(changes.column_additions["col_key"].tolist())
        assert col_keys == {"C", "D"}
        assert len(changes.cell_changes) == 0

    def test_column_addition_with_cell_changes(self, cache_manager):
        """Test column addition combined with cell value changes."""
        df1 = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["row1", "row2", "row3"]
        )

        df2 = pd.DataFrame(
            {
                "A": [1, 2, 99],  # Changed value in existing column
                "B": [4, 5, 6],
                "C": [7, 8, 9],  # New column
            },
            index=["row1", "row2", "row3"],
        )

        timestamp = datetime.now()
        changes = cache_manager.get_changes_extended(df1, df2, timestamp)

        # Should have both column additions and cell changes
        assert len(changes.column_additions) == 1
        assert changes.column_additions.iloc[0]["col_key"] == "C"
        assert len(changes.cell_changes) == 1
        assert changes.cell_changes.iloc[0]["col_key"] == "A"
        assert changes.cell_changes.iloc[0]["row_key"] == "row3"

    def test_column_addition_with_new_rows(self, cache_manager):
        """Test column addition with new rows."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["row1", "row2"])

        df2 = pd.DataFrame(
            {
                "A": [1, 2, 5],
                "B": [3, 4, 6],
                "C": [7, 8, 9],  # New column
            },
            index=["row1", "row2", "row3"],
        )  # New row

        timestamp = datetime.now()
        changes = cache_manager.get_changes_extended(df1, df2, timestamp)

        # Should have column additions for the new column and row additions for new row
        assert len(changes.column_additions) == 1
        assert changes.column_additions.iloc[0]["col_key"] == "C"
        assert len(changes.row_additions) == 1
        assert changes.row_additions.iloc[0]["row_key"] == "row3"
        assert len(changes.cell_changes) == 0

    def test_save_and_reconstruct_with_column_additions(self, cache_manager):
        """Test saving and reconstructing data with column additions."""
        # Initial save
        df1 = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["row1", "row2", "row3"]
        )

        timestamp1 = datetime(2023, 1, 1, 10, 0, 0)
        cache_manager.save_data("test_table", df1, timestamp1)

        # Add a column
        df2 = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]},
            index=["row1", "row2", "row3"],
        )

        timestamp2 = datetime(2023, 1, 1, 11, 0, 0)
        cache_manager.save_version("test_table", df1, df2, timestamp2)

        # Reconstruct at timestamp2
        reconstructed = cache_manager.reconstruct_as_of("test_table", timestamp2)

        # Should match df2 in shape and values (dtype might differ due to JSON serialization)
        assert reconstructed.shape == df2.shape
        assert set(reconstructed.columns) == set(df2.columns)

        # Check values are the same (convert to same dtype for comparison)
        for col in df2.columns:
            if col in reconstructed.columns:
                pd.testing.assert_series_equal(
                    reconstructed[col].astype(str),
                    df2[col].astype(str),
                    check_names=False,
                )

        # Reconstruct at timestamp1 (before column addition)
        reconstructed_old = cache_manager.reconstruct_as_of("test_table", timestamp1)

        # Should match df1
        pd.testing.assert_frame_equal(reconstructed_old.sort_index(), df1.sort_index())

    def test_column_addition_with_null_values(self, cache_manager):
        """Test column addition with null values."""
        df1 = pd.DataFrame({"A": [1, 2, 3]}, index=["row1", "row2", "row3"])

        df2 = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, None, 6],  # New column with null value
            },
            index=["row1", "row2", "row3"],
        )

        timestamp = datetime.now()
        changes = cache_manager.get_changes_extended(df1, df2, timestamp)

        assert len(changes.column_additions) == 1
        col_data_json = changes.column_additions.iloc[0]["col_data_json"]
        import json

        col_data = json.loads(col_data_json)
        expected_data = {"row1": 4, "row2": None, "row3": 6}
        assert col_data == expected_data

    def test_column_deletion_and_addition_cycle(self, cache_manager):
        """Test column deletion followed by re-addition."""
        # Initial DataFrame
        df1 = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]},
            index=["row1", "row2", "row3"],
        )

        timestamp1 = datetime(2023, 1, 1, 10, 0, 0)
        cache_manager.save_data("test_table", df1, timestamp1)

        # Delete column C
        df2 = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["row1", "row2", "row3"]
        )

        timestamp2 = datetime(2023, 1, 1, 11, 0, 0)
        changes2 = cache_manager.save_version("test_table", df1, df2, timestamp2)
        assert len(changes2.column_deletions) == 1

        # Re-add column C with different values
        df3 = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [10, 11, 12],  # Different values
            },
            index=["row1", "row2", "row3"],
        )

        timestamp3 = datetime(2023, 1, 1, 12, 0, 0)
        changes3 = cache_manager.save_version("test_table", df2, df3, timestamp3)
        assert len(changes3.column_additions) == 1

        # Reconstruct at final timestamp
        reconstructed = cache_manager.reconstruct_as_of("test_table", timestamp3)
        assert reconstructed.shape == df3.shape
        assert set(reconstructed.columns) == set(df3.columns)

        # Check values are the same
        for col in df3.columns:
            if col in reconstructed.columns:
                pd.testing.assert_series_equal(
                    reconstructed[col].astype(str),
                    df3[col].astype(str),
                    check_names=False,
                )

        # Reconstruct at middle timestamp (column deleted)
        reconstructed_middle = cache_manager.reconstruct_as_of("test_table", timestamp2)
        assert reconstructed_middle.shape == df2.shape
        assert set(reconstructed_middle.columns) == set(df2.columns)

    def test_column_additions_in_change_result_str(self, cache_manager):
        """Test that column additions are included in ChangeResult string representation."""
        from finlab_guard.utils.exceptions import DataModifiedException

        df1 = pd.DataFrame({"A": [1, 2]}, index=["row1", "row2"])
        df2 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["row1", "row2"])

        changes = cache_manager.get_changes_extended(df1, df2, datetime.now())

        # Create a DataModifiedException to test string formatting
        exc = DataModifiedException("Test message", changes)
        exc_str = str(exc)

        assert "Column additions: 1" in exc_str
