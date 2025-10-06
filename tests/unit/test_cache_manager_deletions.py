"""Unit tests for CacheManager deletion functionality."""

import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from finlab_guard.cache.manager import CacheManager


def safe_rmtree(path, ignore_errors=False):
    """Windows-compatible rmtree with retry logic for DuckDB file locking."""
    for attempt in range(5):
        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            break
        except (PermissionError, OSError):
            if attempt < 4:
                time.sleep(0.1)  # Wait 100ms before retry
                continue
            if not ignore_errors:
                raise


class TestCacheManagerDeletions:
    """Test suite for CacheManager deletion functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        safe_rmtree(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager instance for testing."""
        config = {"compression": "snappy"}
        manager = CacheManager(temp_cache_dir, config)
        yield manager
        manager.close()

    def test_detect_row_deletions(self, cache_manager):
        """Test detection of deleted rows."""
        # Initial DataFrame with 3 rows
        base_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )

        # DataFrame with row2 deleted
        modified_df = pd.DataFrame(
            {"A": [1, 3], "B": ["x", "z"]}, index=["row1", "row3"]
        )

        timestamp = datetime.now()
        cell_df, row_df, row_del_df, col_add_df, col_del_df, meta = (
            cache_manager._get_changes_extended_polars(base_df, modified_df, timestamp)
        )

        # Should detect row2 as deleted
        assert not row_del_df.empty
        assert len(row_del_df) == 1
        assert row_del_df.iloc[0]["row_key"] == "row2"
        assert "deleted_rows" in meta
        assert meta["deleted_rows"] == ["row2"]

    def test_detect_column_deletions(self, cache_manager):
        """Test detection of deleted columns."""
        # Initial DataFrame with 3 columns
        base_df = pd.DataFrame(
            {"A": [1, 2], "B": ["x", "y"], "C": [10, 20]}, index=["row1", "row2"]
        )

        # DataFrame with column B deleted
        modified_df = pd.DataFrame({"A": [1, 2], "C": [10, 20]}, index=["row1", "row2"])

        timestamp = datetime.now()
        cell_df, row_df, row_del_df, col_add_df, col_del_df, meta = (
            cache_manager._get_changes_extended_polars(base_df, modified_df, timestamp)
        )

        # Should detect column B as deleted
        assert not col_del_df.empty
        assert len(col_del_df) == 1
        assert col_del_df.iloc[0]["col_key"] == "B"
        assert "deleted_cols" in meta
        assert meta["deleted_cols"] == ["B"]

    def test_mixed_operations(self, cache_manager):
        """Test mixed add/modify/delete operations."""
        # Initial DataFrame
        base_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [10, 20, 30]},
            index=["row1", "row2", "row3"],
        )

        # Modified DataFrame:
        # - row2 deleted
        # - column C deleted
        # - row4 added
        # - column D added
        # - row1.A modified
        modified_df = pd.DataFrame(
            {
                "A": [100, 3, 4],  # row1.A changed from 1 to 100
                "B": ["x", "z", "w"],
                "D": [40, 50, 60],  # new column
            },
            index=["row1", "row3", "row4"],
        )  # row4 added

        timestamp = datetime.now()
        cell_df, row_df, row_del_df, col_add_df, col_del_df, meta = (
            cache_manager._get_changes_extended_polars(base_df, modified_df, timestamp)
        )

        # Check deletions
        assert len(row_del_df) == 1
        assert row_del_df.iloc[0]["row_key"] == "row2"
        assert len(col_del_df) == 1
        assert col_del_df.iloc[0]["col_key"] == "C"

        # Check additions
        assert "row4" in meta["new_rows"]
        assert "D" in meta["new_cols"]

        # Check deletions in meta
        assert meta["deleted_rows"] == ["row2"]
        assert meta["deleted_cols"] == ["C"]

    def test_save_deletions(self, cache_manager):
        """Test saving deletion records to database."""
        # Create initial snapshot
        base_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_table", base_df, timestamp1)

        # Create modified version with deletions
        modified_df = pd.DataFrame(
            {"A": [1, 3]}, index=["row1", "row3"]
        )  # row2 deleted, column B deleted

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("test_table", base_df, modified_df, timestamp2)

        # Check if deletions were saved to database
        row_del_query = "SELECT * FROM row_deletions WHERE table_id = 'test_table'"
        row_del_result = cache_manager.conn.execute(row_del_query).fetchdf()

        assert not row_del_result.empty
        assert len(row_del_result) == 1
        assert row_del_result.iloc[0]["row_key"] == "row2"

        col_del_query = "SELECT * FROM column_deletions WHERE table_id = 'test_table'"
        col_del_result = cache_manager.conn.execute(col_del_query).fetchdf()

        assert not col_del_result.empty
        assert len(col_del_result) == 1
        assert col_del_result.iloc[0]["col_key"] == "B"

    def test_reconstruct_with_deletions(self, cache_manager):
        """Test reconstruction with deletion filtering."""
        # Create initial snapshot
        base_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [10, 20, 30]},
            index=["row1", "row2", "row3"],
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_table", base_df, timestamp1)

        # Delete row2 and column C
        modified_df = pd.DataFrame(
            {"A": [1, 3], "B": ["x", "z"]}, index=["row1", "row3"]
        )

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("test_table", base_df, modified_df, timestamp2)

        # Reconstruct at timestamp2
        result = cache_manager.reconstruct_as_of(
            "test_table", timestamp2 + timedelta(seconds=1)
        )

        # Should not contain deleted row or column
        assert len(result) == 2  # Only row1 and row3
        assert "row2" not in result.index
        assert "C" not in result.columns
        assert list(result.index) == ["row1", "row3"]
        assert list(result.columns) == ["A", "B"]

        # Values should be correct
        assert result.loc["row1", "A"] == 1
        assert result.loc["row3", "A"] == 3

    def test_time_based_deletion_filtering(self, cache_manager):
        """Test that deletions are filtered by time."""
        # Create initial snapshot
        base_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_table", base_df, timestamp1)

        # Delete row2 at timestamp2
        modified_df = pd.DataFrame(
            {"A": [1, 3], "B": ["x", "z"]}, index=["row1", "row3"]
        )

        timestamp2 = timestamp1 + timedelta(minutes=5)
        cache_manager.save_version("test_table", base_df, modified_df, timestamp2)

        # Reconstruct before deletion (should include row2)
        result_before = cache_manager.reconstruct_as_of(
            "test_table", timestamp1 + timedelta(minutes=1)
        )
        assert len(result_before) == 3
        assert "row2" in result_before.index

        # Reconstruct after deletion (should not include row2)
        result_after = cache_manager.reconstruct_as_of(
            "test_table", timestamp2 + timedelta(minutes=1)
        )
        assert len(result_after) == 2
        assert "row2" not in result_after.index

    def test_load_deletions_method(self, cache_manager):
        """Test the _load_deletions method directly."""
        # Create some deletion records manually
        timestamp = datetime.now()

        # Insert row deletion
        cache_manager.conn.execute(
            "INSERT INTO row_deletions (table_id, row_key, delete_time) VALUES (?, ?, ?)",
            ["test_table", "row2", timestamp],
        )

        # Insert column deletion
        cache_manager.conn.execute(
            "INSERT INTO column_deletions (table_id, col_key, delete_time) VALUES (?, ?, ?)",
            ["test_table", "col_B", timestamp],
        )

        # Load deletions
        snapshot_time = timestamp - timedelta(minutes=1)
        target_time = timestamp + timedelta(minutes=1)
        deleted_rows, deleted_cols = cache_manager._load_deletions(
            "test_table", snapshot_time, target_time
        )

        assert "row2" in deleted_rows
        assert "col_B" in deleted_cols
        assert len(deleted_rows) == 1
        assert len(deleted_cols) == 1

    def test_finalize_dataframe_with_deletions(self, cache_manager):
        """Test _finalize_dataframe with deletion filtering."""
        # Create test data
        test_pl = pl.DataFrame(
            {
                "row_key": ["row1", "row2", "row3"],
                "A": [1, 2, 3],
                "B": ["x", "y", "z"],
                "C": [10, 20, 30],
            }
        )

        # Test with row deletions
        deleted_rows = {"row2"}
        result = cache_manager._finalize_dataframe(test_pl, deleted_rows=deleted_rows)

        assert len(result) == 2
        assert "row2" not in result.index
        assert list(result.index) == ["row1", "row3"]

        # Test with column deletions
        deleted_cols = {"C"}
        result = cache_manager._finalize_dataframe(test_pl, deleted_cols=deleted_cols)

        assert "C" not in result.columns
        assert list(result.columns) == ["A", "B"]

        # Test with both row and column deletions
        result = cache_manager._finalize_dataframe(
            test_pl, deleted_rows={"row2"}, deleted_cols={"C"}
        )

        assert len(result) == 2
        assert "row2" not in result.index
        assert "C" not in result.columns
        assert list(result.index) == ["row1", "row3"]
        assert list(result.columns) == ["A", "B"]

    def test_delete_all_rows(self, cache_manager):
        """Test deleting all rows."""
        # Initial DataFrame
        base_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )

        # Empty DataFrame (all rows deleted)
        modified_df = pd.DataFrame({"A": [], "B": []}).astype({"A": int, "B": str})

        timestamp = datetime.now()
        cell_df, row_df, row_del_df, col_add_df, col_del_df, meta = (
            cache_manager._get_changes_extended_polars(base_df, modified_df, timestamp)
        )

        # Should detect all rows as deleted
        assert len(row_del_df) == 3
        deleted_row_keys = set(row_del_df["row_key"])
        assert deleted_row_keys == {"row1", "row2", "row3"}

    def test_delete_all_columns(self, cache_manager):
        """Test deleting all columns."""
        # Initial DataFrame
        base_df = pd.DataFrame(
            {"A": [1, 2], "B": ["x", "y"], "C": [10, 20]}, index=["row1", "row2"]
        )

        # DataFrame with no columns (all deleted, but rows remain)
        modified_df = pd.DataFrame(index=["row1", "row2"])

        timestamp = datetime.now()
        cell_df, row_df, row_del_df, col_add_df, col_del_df, meta = (
            cache_manager._get_changes_extended_polars(base_df, modified_df, timestamp)
        )

        # Should detect all columns as deleted
        assert len(col_del_df) == 3
        deleted_col_keys = set(col_del_df["col_key"])
        assert deleted_col_keys == {"A", "B", "C"}

    def test_delete_then_readd(self, cache_manager):
        """Test deleting then re-adding the same row/column."""
        # Step 1: Initial data
        base_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_table", base_df, timestamp1)

        # Step 2: Delete row2
        deleted_df = pd.DataFrame(
            {"A": [1, 3], "B": ["x", "z"]}, index=["row1", "row3"]
        )

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("test_table", base_df, deleted_df, timestamp2)

        # Step 3: Re-add row2 (with different data)
        readded_df = pd.DataFrame(
            {
                "A": [1, 3, 5],  # new value for row2
                "B": ["x", "z", "new"],  # new value for row2
            },
            index=["row1", "row3", "row2"],
        )

        timestamp3 = timestamp2 + timedelta(minutes=1)
        cache_manager.save_version("test_table", deleted_df, readded_df, timestamp3)

        # Test reconstruction at each point
        result1 = cache_manager.reconstruct_as_of(
            "test_table", timestamp1 + timedelta(seconds=30)
        )
        assert len(result1) == 3
        assert "row2" in result1.index

        result2 = cache_manager.reconstruct_as_of(
            "test_table", timestamp2 + timedelta(seconds=30)
        )
        assert len(result2) == 2
        assert "row2" not in result2.index

        result3 = cache_manager.reconstruct_as_of(
            "test_table", timestamp3 + timedelta(seconds=30)
        )
        assert len(result3) == 3
        assert "row2" in result3.index
        assert result3.loc["row2", "A"] == 5  # Should have new value
        assert result3.loc["row2", "B"] == "new"  # Should have new value

    def test_progressive_deletions(self, cache_manager):
        """Test multiple progressive deletions over time."""
        # Initial data
        base_df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": ["w", "x", "y", "z"],
                "C": [10, 20, 30, 40],
                "D": [100, 200, 300, 400],
            },
            index=["row1", "row2", "row3", "row4"],
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_table", base_df, timestamp1)

        # Step 1: Delete row2
        step1_df = pd.DataFrame(
            {
                "A": [1, 3, 4],
                "B": ["w", "y", "z"],
                "C": [10, 30, 40],
                "D": [100, 300, 400],
            },
            index=["row1", "row3", "row4"],
        )

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("test_table", base_df, step1_df, timestamp2)

        # Step 2: Delete column C
        step2_df = pd.DataFrame(
            {"A": [1, 3, 4], "B": ["w", "y", "z"], "D": [100, 300, 400]},
            index=["row1", "row3", "row4"],
        )

        timestamp3 = timestamp2 + timedelta(minutes=1)
        cache_manager.save_version("test_table", step1_df, step2_df, timestamp3)

        # Step 3: Delete row4
        step3_df = pd.DataFrame(
            {"A": [1, 3], "B": ["w", "y"], "D": [100, 300]}, index=["row1", "row3"]
        )

        timestamp4 = timestamp3 + timedelta(minutes=1)
        cache_manager.save_version("test_table", step2_df, step3_df, timestamp4)

        # Test reconstruction at each step
        result1 = cache_manager.reconstruct_as_of(
            "test_table", timestamp1 + timedelta(seconds=30)
        )
        assert len(result1) == 4
        assert len(result1.columns) == 4

        result2 = cache_manager.reconstruct_as_of(
            "test_table", timestamp2 + timedelta(seconds=30)
        )
        assert len(result2) == 3  # row2 deleted
        assert len(result2.columns) == 4
        assert "row2" not in result2.index

        result3 = cache_manager.reconstruct_as_of(
            "test_table", timestamp3 + timedelta(seconds=30)
        )
        assert len(result3) == 3
        assert len(result3.columns) == 3  # column C deleted
        assert "C" not in result3.columns

        result4 = cache_manager.reconstruct_as_of(
            "test_table", timestamp4 + timedelta(seconds=30)
        )
        assert len(result4) == 2  # row4 also deleted
        assert len(result4.columns) == 3
        assert "row4" not in result4.index

    def test_deletions_with_special_values(self, cache_manager):
        """Test deletion functionality with inf/nan/None values."""
        # Initial DataFrame with special values
        base_df = pd.DataFrame(
            {
                "A": [1.0, np.inf, 3.0, np.nan],
                "B": ["x", None, "z", "w"],
                "C": [10, 20, -np.inf, 40],
            },
            index=["row1", "row2", "row3", "row4"],
        )

        # Delete rows containing special values
        modified_df = pd.DataFrame(
            {"A": [1.0, 3.0], "B": ["x", "z"], "C": [10, -np.inf]},
            index=["row1", "row3"],
        )  # row2 (inf) and row4 (nan) deleted

        timestamp = datetime.now()
        cell_df, row_df, row_del_df, col_add_df, col_del_df, meta = (
            cache_manager._get_changes_extended_polars(base_df, modified_df, timestamp)
        )

        # Should detect deletions correctly even with special values
        assert len(row_del_df) == 2
        deleted_row_keys = set(row_del_df["row_key"])
        assert deleted_row_keys == {"row2", "row4"}

        # Save and reconstruct to verify handling
        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_special", base_df, timestamp1)

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("test_special", base_df, modified_df, timestamp2)

        result = cache_manager.reconstruct_as_of(
            "test_special", timestamp2 + timedelta(seconds=1)
        )

        # Verify special values are preserved in remaining rows
        assert len(result) == 2
        assert result.loc["row1", "A"] == 1.0
        assert result.loc["row3", "A"] == 3.0
        assert result.loc["row3", "C"] == -np.inf  # -inf should be preserved

    def test_delete_columns_with_special_values(self, cache_manager):
        """Test column deletion when columns contain special values."""
        # Initial DataFrame with special values in different columns
        base_df = pd.DataFrame(
            {
                "normal": [1, 2, 3],
                "with_inf": [np.inf, -np.inf, 1.0],
                "with_nan": [np.nan, 2.0, np.nan],
                "with_none": ["a", None, "c"],
            },
            index=["row1", "row2", "row3"],
        )

        # Delete columns containing special values
        modified_df = pd.DataFrame(
            {"normal": [1, 2, 3]}, index=["row1", "row2", "row3"]
        )

        timestamp = datetime.now()
        cell_df, row_df, row_del_df, col_add_df, col_del_df, meta = (
            cache_manager._get_changes_extended_polars(base_df, modified_df, timestamp)
        )

        # Should detect all special-value columns as deleted
        assert len(col_del_df) == 3
        deleted_col_keys = set(col_del_df["col_key"])
        assert deleted_col_keys == {"with_inf", "with_nan", "with_none"}

        # Save and reconstruct
        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_special_cols", base_df, timestamp1)

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version(
            "test_special_cols", base_df, modified_df, timestamp2
        )

        result = cache_manager.reconstruct_as_of(
            "test_special_cols", timestamp2 + timedelta(seconds=1)
        )

        # Should only have the normal column
        assert list(result.columns) == ["normal"]
        assert len(result) == 3

    def test_json_serialization_of_special_values(self, cache_manager):
        """Test that special values are properly serialized/deserialized in deletion records."""
        # Create data with special values that will be deleted (use consistent types)
        base_df = pd.DataFrame(
            {
                "numeric": [np.inf, -np.inf, np.nan, 42.0, 1.0],
                "string": ["inf_str", "neg_inf_str", "nan_str", "none_str", "normal"],
            },
            index=["inf", "neg_inf", "nan", "none", "normal"],
        )

        # Delete rows with special values
        modified_df = pd.DataFrame(
            {"numeric": [1.0], "string": ["normal"]}, index=["normal"]
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_json_special", base_df, timestamp1)

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version(
            "test_json_special", base_df, modified_df, timestamp2
        )

        # Check that deletion records were created (they should be stored as strings)
        row_del_query = (
            "SELECT * FROM row_deletions WHERE table_id = 'test_json_special'"
        )
        row_del_result = cache_manager.conn.execute(row_del_query).fetchdf()

        assert len(row_del_result) == 4
        deleted_keys = set(row_del_result["row_key"])
        assert deleted_keys == {"inf", "neg_inf", "nan", "none"}

        # Reconstruction should work correctly
        result = cache_manager.reconstruct_as_of(
            "test_json_special", timestamp2 + timedelta(seconds=1)
        )
        assert len(result) == 1
        assert result.index[0] == "normal"

    def test_complex_inf_nan_deletion_readd_cycles(self, cache_manager):
        """Test complex scenarios with inf/nan values being deleted and re-added multiple times."""
        # Step 1: Initial data with various special values
        base_df = pd.DataFrame(
            {
                "values": [1.0, np.inf, -np.inf, np.nan, 5.0],
                "status": ["normal", "inf", "neg_inf", "nan", "normal2"],
            },
            index=["A", "B", "C", "D", "E"],
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("complex_test", base_df, timestamp1)

        # Step 2: Delete all rows with special values
        step2_df = pd.DataFrame(
            {"values": [1.0, 5.0], "status": ["normal", "normal2"]}, index=["A", "E"]
        )

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("complex_test", base_df, step2_df, timestamp2)

        # Step 3: Re-add one of the special value rows with different data
        step3_df = pd.DataFrame(
            {
                "values": [1.0, 5.0, np.inf],  # Re-add inf but with different row
                "status": ["normal", "normal2", "new_inf"],
            },
            index=["A", "E", "F"],
        )  # Different index for inf value

        timestamp3 = timestamp2 + timedelta(minutes=1)
        cache_manager.save_version("complex_test", step2_df, step3_df, timestamp3)

        # Step 4: Delete the re-added inf row and add original nan row back
        step4_df = pd.DataFrame(
            {
                "values": [1.0, 5.0, np.nan],  # Remove new inf, add back nan
                "status": ["normal", "normal2", "restored_nan"],
            },
            index=["A", "E", "D"],
        )  # Original nan row index

        timestamp4 = timestamp3 + timedelta(minutes=1)
        cache_manager.save_version("complex_test", step3_df, step4_df, timestamp4)

        # Step 5: Add all special values back but with modifications
        step5_df = pd.DataFrame(
            {
                "values": [1.0, 5.0, np.nan, np.inf, -np.inf, 0.0],
                "status": [
                    "normal",
                    "normal2",
                    "restored_nan",
                    "new_inf2",
                    "new_neg_inf",
                    "zero",
                ],
            },
            index=["A", "E", "D", "G", "H", "I"],
        )

        timestamp5 = timestamp4 + timedelta(minutes=1)
        cache_manager.save_version("complex_test", step4_df, step5_df, timestamp5)

        # Test reconstruction at each step
        result1 = cache_manager.reconstruct_as_of(
            "complex_test", timestamp1 + timedelta(seconds=30)
        )
        assert len(result1) == 5
        val_b = float(result1.loc["B", "values"])
        val_c = float(result1.loc["C", "values"])
        val_d = float(result1.loc["D", "values"])
        assert np.isinf(val_b) and val_b > 0
        assert np.isinf(val_c) and val_c < 0
        assert np.isnan(val_d)

        result2 = cache_manager.reconstruct_as_of(
            "complex_test", timestamp2 + timedelta(seconds=30)
        )
        assert len(result2) == 2
        assert (
            "B" not in result2.index
            and "C" not in result2.index
            and "D" not in result2.index
        )

        result3 = cache_manager.reconstruct_as_of(
            "complex_test", timestamp3 + timedelta(seconds=30)
        )
        assert len(result3) == 3
        assert "F" in result3.index
        assert np.isinf(result3.loc["F", "values"]) and result3.loc["F", "values"] > 0
        assert result3.loc["F", "status"] == "new_inf"

        result4 = cache_manager.reconstruct_as_of(
            "complex_test", timestamp4 + timedelta(seconds=30)
        )
        assert len(result4) == 3
        assert "F" not in result4.index  # inf row deleted again
        assert "D" in result4.index  # nan row restored
        assert np.isnan(result4.loc["D", "values"])
        assert result4.loc["D", "status"] == "restored_nan"

        result5 = cache_manager.reconstruct_as_of(
            "complex_test", timestamp5 + timedelta(seconds=30)
        )
        assert len(result5) == 6
        # Check all special values are present with correct values
        assert np.isnan(result5.loc["D", "values"])
        assert np.isinf(result5.loc["G", "values"]) and result5.loc["G", "values"] > 0
        assert np.isinf(result5.loc["H", "values"]) and result5.loc["H", "values"] < 0

    def test_inf_nan_column_operations(self, cache_manager):
        """Test complex column operations with inf/nan values."""
        # Step 1: Initial data with columns containing special values
        base_df = pd.DataFrame(
            {
                "pure_inf": [np.inf, np.inf, np.inf],
                "mixed": [1.0, np.inf, np.nan],
                "pure_nan": [np.nan, np.nan, np.nan],
                "normal": [1, 2, 3],
            },
            index=["row1", "row2", "row3"],
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("col_test", base_df, timestamp1)

        # Step 2: Delete pure special value columns
        step2_df = pd.DataFrame(
            {"mixed": [1.0, np.inf, np.nan], "normal": [1, 2, 3]},
            index=["row1", "row2", "row3"],
        )

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("col_test", base_df, step2_df, timestamp2)

        # Step 3: Add new column with special values
        step3_df = pd.DataFrame(
            {
                "mixed": [1.0, np.inf, np.nan],
                "normal": [1, 2, 3],
                "new_special": [-np.inf, 0.0, np.inf],
            },
            index=["row1", "row2", "row3"],
        )

        timestamp3 = timestamp2 + timedelta(minutes=1)
        cache_manager.save_version("col_test", step2_df, step3_df, timestamp3)

        # Step 4: Delete mixed column, keep new special column
        step4_df = pd.DataFrame(
            {"normal": [1, 2, 3], "new_special": [-np.inf, 0.0, np.inf]},
            index=["row1", "row2", "row3"],
        )

        timestamp4 = timestamp3 + timedelta(minutes=1)
        cache_manager.save_version("col_test", step3_df, step4_df, timestamp4)

        # Step 5: Restore original pure_inf column but with different values
        step5_df = pd.DataFrame(
            {
                "normal": [1, 2, 3],
                "new_special": [-np.inf, 0.0, np.inf],
                "pure_inf": [
                    -np.inf,
                    -np.inf,
                    -np.inf,
                ],  # Different values than original
            },
            index=["row1", "row2", "row3"],
        )

        timestamp5 = timestamp4 + timedelta(minutes=1)

        # Debug: Check what changes are detected for step 5
        cell_df5, row_df5, row_del_df5, col_add_df5, col_del_df5, meta5 = (
            cache_manager._get_changes_extended_polars(step4_df, step5_df, timestamp5)
        )
        print(f"DEBUG Step 5 - Cell changes: {len(cell_df5)} rows")
        if not cell_df5.empty:
            print(f"DEBUG Step 5 - Cell columns: {list(cell_df5.columns)}")
            print(f"DEBUG Step 5 - Cell data:\n{cell_df5}")
        print(f"DEBUG Step 5 - Row additions: {len(row_df5)} rows")
        print(f"DEBUG Step 5 - Col deletions: {len(col_del_df5)} rows")

        cache_manager.save_version("col_test", step4_df, step5_df, timestamp5)

        # Test reconstructions
        result1 = cache_manager.reconstruct_as_of(
            "col_test", timestamp1 + timedelta(seconds=30)
        )
        assert len(result1.columns) == 4
        assert all(np.isinf(result1["pure_inf"]) & (result1["pure_inf"] > 0))
        assert all(np.isnan(result1["pure_nan"]))

        result2 = cache_manager.reconstruct_as_of(
            "col_test", timestamp2 + timedelta(seconds=30)
        )
        assert len(result2.columns) == 2
        assert "pure_inf" not in result2.columns
        assert "pure_nan" not in result2.columns
        assert "mixed" in result2.columns

        result3 = cache_manager.reconstruct_as_of(
            "col_test", timestamp3 + timedelta(seconds=30)
        )
        assert len(result3.columns) == 3
        assert "new_special" in result3.columns
        # Convert to float to ensure proper type before checking
        val = float(result3.loc["row1", "new_special"])
        assert np.isinf(val) and val < 0

        result4 = cache_manager.reconstruct_as_of(
            "col_test", timestamp4 + timedelta(seconds=30)
        )
        assert len(result4.columns) == 2
        assert "mixed" not in result4.columns
        assert "new_special" in result4.columns

        # Debug reconstruction step by step
        target_time = timestamp5 + timedelta(seconds=30)
        base5, snapshot_time = cache_manager._load_base_snapshot(
            "col_test", target_time
        )
        changes5 = cache_manager._load_and_process_cell_changes(
            "col_test", snapshot_time, target_time
        )
        additions5 = cache_manager._load_and_process_row_additions(
            "col_test", snapshot_time, target_time
        )
        deleted_rows5, deleted_cols5 = cache_manager._load_deletions(
            "col_test", snapshot_time, target_time
        )

        print(f"DEBUG Reconstruction - Base columns: {list(base5.columns)}")
        print(f"DEBUG Reconstruction - Changes columns: {list(changes5.columns)}")
        print(f"DEBUG Reconstruction - Changes data:\n{changes5}")
        print(f"DEBUG Reconstruction - Additions columns: {list(additions5.columns)}")
        print(f"DEBUG Reconstruction - Deleted rows: {deleted_rows5}")
        print(f"DEBUG Reconstruction - Deleted cols: {deleted_cols5}")

        merged5 = cache_manager._merge_data_layers(base5, changes5, additions5, {})
        print(f"DEBUG Reconstruction - Merged columns: {list(merged5.columns)}")
        print(f"DEBUG Reconstruction - Merged data:\n{merged5}")

        result5 = cache_manager.reconstruct_as_of(
            "col_test", timestamp5 + timedelta(seconds=30)
        )
        print(f"DEBUG: result5 columns: {list(result5.columns)}")
        print(f"DEBUG: result5 shape: {result5.shape}")
        print(f"DEBUG: result5:\n{result5}")
        assert len(result5.columns) == 3
        assert "pure_inf" in result5.columns
        # Check that restored column has new values (all negative inf)
        pure_inf_values = result5["pure_inf"].astype(float)
        assert all(np.isinf(pure_inf_values) & (pure_inf_values < 0))

    def test_special_values_edge_cases_with_deletions(self, cache_manager):
        """Test edge cases with special values during deletion operations."""
        # Test case 1: DataFrame with only special values, then delete all
        special_only_df = pd.DataFrame(
            {"col1": [np.inf, -np.inf, np.nan], "col2": [np.nan, np.inf, -np.inf]},
            index=["inf", "neg_inf", "nan"],
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("edge_test", special_only_df, timestamp1)

        # Delete all rows
        empty_df = pd.DataFrame({"col1": [], "col2": []}).astype(
            {"col1": float, "col2": float}
        )
        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("edge_test", special_only_df, empty_df, timestamp2)

        # Test case 2: Add back rows with swapped special values
        swapped_df = pd.DataFrame(
            {
                "col1": [np.nan, np.inf, -np.inf],  # Swapped pattern
                "col2": [-np.inf, np.nan, np.inf],  # Swapped pattern
            },
            index=["inf", "neg_inf", "nan"],
        )  # Same indices

        timestamp3 = timestamp2 + timedelta(minutes=1)
        cache_manager.save_version("edge_test", empty_df, swapped_df, timestamp3)

        # Test case 3: Delete columns, add different columns with special values
        new_cols_df = pd.DataFrame(
            {"new_col1": [np.inf, np.inf, np.inf], "new_col2": [1.0, 2.0, 3.0]},
            index=["inf", "neg_inf", "nan"],
        )

        timestamp4 = timestamp3 + timedelta(minutes=1)
        cache_manager.save_version("edge_test", swapped_df, new_cols_df, timestamp4)

        # Verify reconstructions
        result1 = cache_manager.reconstruct_as_of(
            "edge_test", timestamp1 + timedelta(seconds=30)
        )
        assert len(result1) == 3
        assert len(result1.columns) == 2
        # Original pattern
        assert np.isinf(result1.loc["inf", "col1"]) and result1.loc["inf", "col1"] > 0
        assert np.isnan(result1.loc["inf", "col2"])

        result2 = cache_manager.reconstruct_as_of(
            "edge_test", timestamp2 + timedelta(seconds=30)
        )
        assert len(result2) == 0  # All deleted

        result3 = cache_manager.reconstruct_as_of(
            "edge_test", timestamp3 + timedelta(seconds=30)
        )
        assert len(result3) == 3
        # Swapped pattern
        assert np.isnan(result3.loc["inf", "col1"])
        assert np.isinf(result3.loc["inf", "col2"]) and result3.loc["inf", "col2"] < 0

        result4 = cache_manager.reconstruct_as_of(
            "edge_test", timestamp4 + timedelta(seconds=30)
        )
        assert len(result4) == 3
        assert len(result4.columns) == 2
        assert "col1" not in result4.columns
        assert "col2" not in result4.columns
        assert "new_col1" in result4.columns
        assert "new_col2" in result4.columns
        # All inf in new_col1 - convert to float first to handle any type issues
        new_col1_values = result4["new_col1"].astype(float)
        assert all(np.isinf(new_col1_values) & (new_col1_values > 0))

    def test_multiple_deletion_addition_cycles(self, cache_manager):
        """Test column that goes through multiple deletion/addition cycles."""
        # t1: Initial state with column A
        df_t1 = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )

        timestamp_t1 = datetime.now()
        cache_manager.save_snapshot("cycle_test", df_t1, timestamp_t1)

        # t2: Delete column A
        df_t2 = pd.DataFrame({"B": ["x", "y", "z"]}, index=["row1", "row2", "row3"])

        timestamp_t2 = timestamp_t1 + timedelta(minutes=1)
        cache_manager.save_version("cycle_test", df_t1, df_t2, timestamp_t2)

        # t3: Re-add column A with new values
        df_t3 = pd.DataFrame(
            {
                "A": [10, 20, 30],  # Different values
                "B": ["x", "y", "z"],
            },
            index=["row1", "row2", "row3"],
        )

        timestamp_t3 = timestamp_t2 + timedelta(minutes=1)
        cache_manager.save_version("cycle_test", df_t2, df_t3, timestamp_t3)

        # t4: Delete column A again
        df_t4 = pd.DataFrame({"B": ["x", "y", "z"]}, index=["row1", "row2", "row3"])

        timestamp_t4 = timestamp_t3 + timedelta(minutes=1)
        cache_manager.save_version("cycle_test", df_t3, df_t4, timestamp_t4)

        # t5: Re-add column A again with different values
        df_t5 = pd.DataFrame(
            {
                "A": [100, 200, 300],  # Different values again
                "B": ["x", "y", "z"],
            },
            index=["row1", "row2", "row3"],
        )

        timestamp_t5 = timestamp_t4 + timedelta(minutes=1)
        cache_manager.save_version("cycle_test", df_t4, df_t5, timestamp_t5)

        # Test reconstruction at each time point
        result_t1_5 = cache_manager.reconstruct_as_of(
            "cycle_test", timestamp_t1 + timedelta(seconds=30)
        )
        print(f"t1.5 - columns: {list(result_t1_5.columns)}")
        assert "A" in result_t1_5.columns
        assert result_t1_5["A"].tolist() == [1, 2, 3]

        result_t2_5 = cache_manager.reconstruct_as_of(
            "cycle_test", timestamp_t2 + timedelta(seconds=30)
        )
        print(f"t2.5 - columns: {list(result_t2_5.columns)}")
        assert "A" not in result_t2_5.columns  # Column A should be deleted

        result_t3_5 = cache_manager.reconstruct_as_of(
            "cycle_test", timestamp_t3 + timedelta(seconds=30)
        )
        print(f"t3.5 - columns: {list(result_t3_5.columns)}")
        if "A" in result_t3_5.columns:
            print(f"t3.5 - A values: {result_t3_5['A'].tolist()}")
            assert result_t3_5["A"].astype(int).tolist() == [10, 20, 30]
        else:
            raise AssertionError("Column A should be restored at t3.5")

        result_t4_5 = cache_manager.reconstruct_as_of(
            "cycle_test", timestamp_t4 + timedelta(seconds=30)
        )
        print(f"t4.5 - columns: {list(result_t4_5.columns)}")
        assert "A" not in result_t4_5.columns  # Column A should be deleted again

        result_t5_5 = cache_manager.reconstruct_as_of(
            "cycle_test", timestamp_t5 + timedelta(seconds=30)
        )
        print(f"t5.5 - columns: {list(result_t5_5.columns)}")
        if "A" in result_t5_5.columns:
            print(f"t5.5 - A values: {result_t5_5['A'].tolist()}")
            assert result_t5_5["A"].astype(int).tolist() == [100, 200, 300]
        else:
            raise AssertionError("Column A should be restored again at t5.5")

    def test_nan_inf_json_round_trip(self, cache_manager):
        """Test that inf/nan values survive JSON serialization/deserialization in deletion context."""
        # Create data that will exercise the JSON serialization paths
        original_df = pd.DataFrame(
            {
                "float_vals": [1.0, np.inf, -np.inf, np.nan, 0.0],
                "str_vals": ["normal", "inf_str", "neg_inf_str", "nan_str", "zero_str"],
            },
            index=["normal", "pos_inf", "neg_inf", "nan_val", "zero"],
        )

        timestamp1 = datetime.now()
        cache_manager.save_snapshot("json_test", original_df, timestamp1)

        # Perform multiple deletion and addition cycles to stress JSON handling
        for i in range(3):
            # Delete some rows
            remaining_rows = (
                ["normal", "zero"] if i % 2 == 0 else ["pos_inf", "nan_val"]
            )
            subset_df = original_df.loc[remaining_rows].copy()

            timestamp_del = timestamp1 + timedelta(minutes=i * 2 + 1)
            cache_manager.save_version(
                "json_test", original_df, subset_df, timestamp_del
            )

            # Re-add all rows
            timestamp_add = timestamp_del + timedelta(minutes=1)
            cache_manager.save_version(
                "json_test", subset_df, original_df, timestamp_add
            )

        # Final reconstruction should match original
        final_result = cache_manager.reconstruct_as_of(
            "json_test", timestamp_add + timedelta(seconds=30)
        )

        assert len(final_result) == 5
        assert (
            np.isinf(final_result.loc["pos_inf", "float_vals"])
            and final_result.loc["pos_inf", "float_vals"] > 0
        )
        assert (
            np.isinf(final_result.loc["neg_inf", "float_vals"])
            and final_result.loc["neg_inf", "float_vals"] < 0
        )
        assert np.isnan(final_result.loc["nan_val", "float_vals"])
        assert final_result.loc["normal", "float_vals"] == 1.0
        assert final_result.loc["zero", "float_vals"] == 0.0

        # Check string values are preserved
        assert final_result.loc["pos_inf", "str_vals"] == "inf_str"
        assert final_result.loc["neg_inf", "str_vals"] == "neg_inf_str"
        assert final_result.loc["nan_val", "str_vals"] == "nan_str"

    def test_multiple_row_deletion_addition_cycles(self, cache_manager):
        """Test row that goes through multiple deletion/addition cycles (8 cycles total)."""
        # t1: Initial state with 3 rows
        df_t1 = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )

        timestamp_t1 = datetime.now()
        cache_manager.save_snapshot("row_cycle_test", df_t1, timestamp_t1)

        # Track the state progression
        current_df = df_t1

        # Create 8 cycles of deletion and re-addition
        for cycle in range(1, 9):
            # Delete row2 - use absolute time calculation
            delete_timestamp = timestamp_t1 + timedelta(minutes=cycle * 2 - 1)
            delete_df = pd.DataFrame(
                {"A": [1, 3], "B": ["x", "z"]}, index=["row1", "row3"]
            )

            cache_manager.save_version(
                "row_cycle_test", current_df, delete_df, delete_timestamp
            )

            # Re-add row2 with new values
            add_timestamp = delete_timestamp + timedelta(minutes=1)
            add_df = pd.DataFrame(
                {
                    "A": [1, 3, cycle * 100],  # Different value each cycle
                    "B": ["x", "z", f"cycle_{cycle}"],  # Different value each cycle
                },
                index=["row1", "row3", "row2"],
            )

            cache_manager.save_version(
                "row_cycle_test", delete_df, add_df, add_timestamp
            )

            current_df = add_df

        # Test reconstruction at various points
        # Original state
        result_original = cache_manager.reconstruct_as_of(
            "row_cycle_test", timestamp_t1 + timedelta(seconds=30)
        )
        assert "row2" in result_original.index
        assert result_original.loc["row2", "A"] == 2
        assert result_original.loc["row2", "B"] == "y"

        # Test each cycle
        for cycle in range(1, 9):
            delete_time = timestamp_t1 + timedelta(minutes=cycle * 2 - 1)
            add_time = delete_time + timedelta(minutes=1)

            # Test deletion phase
            result_delete = cache_manager.reconstruct_as_of(
                "row_cycle_test", delete_time + timedelta(seconds=30)
            )
            assert "row2" not in result_delete.index, (
                f"Cycle {cycle}: row2 should be deleted"
            )
            assert len(result_delete) == 2, (
                f"Cycle {cycle}: should have 2 rows after deletion"
            )

            # Test addition phase
            result_add = cache_manager.reconstruct_as_of(
                "row_cycle_test", add_time + timedelta(seconds=30)
            )
            assert "row2" in result_add.index, f"Cycle {cycle}: row2 should be restored"
            assert result_add.loc["row2", "A"] == cycle * 100, (
                f"Cycle {cycle}: wrong A value"
            )
            assert result_add.loc["row2", "B"] == f"cycle_{cycle}", (
                f"Cycle {cycle}: wrong B value"
            )
            assert len(result_add) == 3, (
                f"Cycle {cycle}: should have 3 rows after addition"
            )

        # Test final state (after all 8 cycles)
        final_timestamp = timestamp_t1 + timedelta(minutes=16)  # 8*2
        final_result = cache_manager.reconstruct_as_of(
            "row_cycle_test", final_timestamp + timedelta(seconds=30)
        )

        assert "row2" in final_result.index
        assert final_result.loc["row2", "A"] == 800  # 8 * 100
        assert final_result.loc["row2", "B"] == "cycle_8"

        # Test intermediate states to ensure time-based reconstruction works
        # Check state after 3rd cycle
        cycle_3_time = timestamp_t1 + timedelta(minutes=6)  # 3*2
        result_cycle_3 = cache_manager.reconstruct_as_of(
            "row_cycle_test", cycle_3_time + timedelta(seconds=30)
        )
        assert "row2" in result_cycle_3.index
        assert result_cycle_3.loc["row2", "A"] == 300
        assert result_cycle_3.loc["row2", "B"] == "cycle_3"

        # Check state after 6th cycle
        cycle_6_time = timestamp_t1 + timedelta(minutes=12)  # 6*2
        result_cycle_6 = cache_manager.reconstruct_as_of(
            "row_cycle_test", cycle_6_time + timedelta(seconds=30)
        )
        assert "row2" in result_cycle_6.index
        assert result_cycle_6.loc["row2", "A"] == 600
        assert result_cycle_6.loc["row2", "B"] == "cycle_6"

        print("Successfully completed 8 deletion-addition cycles for row2")
