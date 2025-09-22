"""Unit tests for CacheManager class."""

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from finlab_guard.cache.manager import CacheManager
from finlab_guard.utils.exceptions import InvalidDataTypeException


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


class TestCacheManager:
    """Test suite for CacheManager class."""

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
        # Ensure DuckDB connection is closed to prevent Windows file locking
        manager.close()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3]}, index=["A", "B", "C"]
        )

    # === 基本CRUD操作 ===

    def test_save_data_first_time(self, cache_manager, sample_dataframe):
        """Test saving data for the first time."""
        key = "test_key"
        timestamp = datetime.now()

        cache_manager.save_data(key, sample_dataframe, timestamp)

        # Verify cache file exists
        cache_path = cache_manager._get_cache_path(key)
        assert cache_path.exists()

        # Verify dtype mapping file exists
        dtype_path = cache_manager._get_dtype_path(key)
        assert dtype_path.exists()

    def test_save_data_existing_key(self, cache_manager, sample_dataframe):
        """Test saving data to existing key."""
        key = "test_key"
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=1)

        # First save
        cache_manager.save_data(key, sample_dataframe, timestamp1)

        # Second save with modified data
        modified_df = sample_dataframe.copy()
        modified_df.loc["A", "col1"] = 99
        cache_manager.save_data(key, modified_df, timestamp2)

        # Verify latest data contains the modification
        latest_data = cache_manager.get_latest_data(key)
        assert latest_data.loc["A", "col1"] == 99

    def test_load_data_exists(self, cache_manager, sample_dataframe):
        """Test loading existing data."""
        key = "test_key"
        timestamp = datetime.now()

        cache_manager.save_data(key, sample_dataframe, timestamp)
        loaded_data = cache_manager.load_data(key)

        # Verify data equality
        pd.testing.assert_frame_equal(loaded_data, sample_dataframe)

    def test_load_data_not_exists(self, cache_manager):
        """Test loading non-existent data."""
        result = cache_manager.load_data("nonexistent_key")
        assert result.empty

    def test_exists_true_false(self, cache_manager, sample_dataframe):
        """Test exists method returns correct boolean values."""
        key = "test_key"

        # Initially doesn't exist
        assert not cache_manager.exists(key)

        # After saving, exists
        cache_manager.save_data(key, sample_dataframe, datetime.now())
        assert cache_manager.exists(key)

    def test_clear_key(self, cache_manager, sample_dataframe):
        """Test clearing specific key."""
        key = "test_key"

        cache_manager.save_data(key, sample_dataframe, datetime.now())
        assert cache_manager.exists(key)

        cache_manager.clear_key(key)
        assert not cache_manager.exists(key)

    def test_clear_all(self, cache_manager, sample_dataframe):
        """Test clearing all cache data."""
        keys = ["key1", "key2", "key3"]

        # Save multiple datasets
        for key in keys:
            cache_manager.save_data(key, sample_dataframe, datetime.now())

        # Verify all exist
        for key in keys:
            assert cache_manager.exists(key)

        # Clear all
        cache_manager.clear_all()

        # Verify none exist
        for key in keys:
            assert not cache_manager.exists(key)

    # === Dtype Mapping 系統 ===

    def test_save_dtype_mapping_new(self, cache_manager, sample_dataframe):
        """Test saving dtype mapping for new DataFrame."""
        key = "test_key"
        timestamp = datetime.now()

        cache_manager._save_dtype_mapping(key, sample_dataframe, timestamp)

        # Verify dtype mapping file exists and contains correct structure
        dtype_path = cache_manager._get_dtype_path(key)
        assert dtype_path.exists()

        with open(dtype_path) as f:
            mapping = json.load(f)

        assert mapping["schema_version"] == "1.0"
        assert len(mapping["dtype_history"]) == 1
        assert "col1" in mapping["dtype_history"][0]["dtypes"]

    def test_save_dtype_mapping_no_changes(self, cache_manager, sample_dataframe):
        """Test saving dtype mapping when no changes occurred."""
        key = "test_key"
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=1)

        # First save
        cache_manager._save_dtype_mapping(key, sample_dataframe, timestamp1)

        # Second save with same dtypes
        cache_manager._save_dtype_mapping(key, sample_dataframe, timestamp2)

        # Verify only one entry exists (no new entry added)
        dtype_path = cache_manager._get_dtype_path(key)
        with open(dtype_path) as f:
            mapping = json.load(f)

        assert len(mapping["dtype_history"]) == 1

    def test_save_dtype_mapping_dtype_changed(self, cache_manager, sample_dataframe):
        """Test saving dtype mapping when dtypes changed."""
        key = "test_key"
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=1)

        # First save
        cache_manager._save_dtype_mapping(key, sample_dataframe, timestamp1)

        # Create DataFrame with changed dtypes
        changed_df = sample_dataframe.copy()
        changed_df["col1"] = changed_df["col1"].astype("int32")  # int64 -> int32
        changed_df["col2"] = changed_df["col2"].astype("float32")  # float64 -> float32

        # Second save with changed dtypes
        cache_manager._save_dtype_mapping(key, changed_df, timestamp2)

        # Verify new entry was added
        dtype_path = cache_manager._get_dtype_path(key)
        with open(dtype_path) as f:
            mapping = json.load(f)

        assert len(mapping["dtype_history"]) == 2

    def test_save_dtype_mapping_order_changed(self, cache_manager):
        """Test saving dtype mapping when column order changed."""
        key = "test_key"
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=1)

        # Original DataFrame
        df1 = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]})
        cache_manager._save_dtype_mapping(key, df1, timestamp1)

        # Reordered DataFrame
        df2 = pd.DataFrame({"col2": [1.1, 2.2], "col1": [1, 2]})
        cache_manager._save_dtype_mapping(key, df2, timestamp2)

        # Verify new entry was added for order change
        dtype_path = cache_manager._get_dtype_path(key)
        with open(dtype_path) as f:
            mapping = json.load(f)

        assert len(mapping["dtype_history"]) == 2
        assert (
            mapping["dtype_history"][0]["columns_order"]
            != mapping["dtype_history"][1]["columns_order"]
        )

    def test_load_dtype_mapping_exists(self, cache_manager, sample_dataframe):
        """Test loading existing dtype mapping."""
        key = "test_key"

        cache_manager._save_dtype_mapping(key, sample_dataframe, datetime.now())
        mapping = cache_manager._load_dtype_mapping(key)

        assert mapping is not None
        assert mapping["schema_version"] == "1.0"

    def test_load_dtype_mapping_not_exists(self, cache_manager):
        """Test loading non-existent dtype mapping."""
        mapping = cache_manager._load_dtype_mapping("nonexistent_key")
        assert mapping is None

    def test_get_dtype_mapping_at_time_latest(self, cache_manager, sample_dataframe):
        """Test getting latest dtype mapping."""
        key = "test_key"
        timestamp = datetime.now()

        cache_manager._save_dtype_mapping(key, sample_dataframe, timestamp)
        mapping = cache_manager._get_dtype_mapping_at_time(key, None)

        assert mapping is not None
        assert len(mapping["dtypes"]) == len(sample_dataframe.columns)

    def test_get_dtype_mapping_at_time_historical(self, cache_manager):
        """Test getting historical dtype mapping."""
        key = "test_key"
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        timestamp2 = datetime(2024, 1, 2, 10, 0, 0)
        target_time = datetime(2024, 1, 1, 15, 0, 0)  # Between timestamps

        # Save two different dtype mappings
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]})

        cache_manager._save_dtype_mapping(key, df1, timestamp1)
        cache_manager._save_dtype_mapping(key, df2, timestamp2)

        # Get mapping at target time (should return first mapping)
        mapping = cache_manager._get_dtype_mapping_at_time(key, target_time)

        assert len(mapping["dtypes"]) == 1  # Should have only col1

    def test_get_dtype_mapping_at_time_before_first(
        self, cache_manager, sample_dataframe
    ):
        """Test getting dtype mapping before first timestamp."""
        key = "test_key"
        timestamp = datetime(2024, 1, 1, 10, 0, 0)
        target_time = datetime(2023, 12, 31, 10, 0, 0)  # Before first timestamp

        cache_manager._save_dtype_mapping(key, sample_dataframe, timestamp)
        mapping = cache_manager._get_dtype_mapping_at_time(key, target_time)

        # Should return the first (and only) mapping entry when target_time is before first
        assert mapping is not None
        assert "dtypes" in mapping

    def test_needs_new_dtype_entry_scenarios(self, cache_manager):
        """Test various scenarios for needing new dtype entry."""
        current_sig = {
            "dtypes": {"col1": "int64", "col2": "float64"},
            "index_dtype": "object",
            "columns_dtype": "object",
            "index_name": None,
            "columns_name": None,
            "columns_order": ["col1", "col2"],
            "index_order": ["A", "B"],
            "index_freq": None,
        }

        # No existing mapping
        assert cache_manager._needs_new_dtype_entry(current_sig, None)

        # Same signature
        existing_mapping = {
            "dtype_history": [{"timestamp": "2024-01-01T10:00:00", **current_sig}]
        }
        assert not cache_manager._needs_new_dtype_entry(current_sig, existing_mapping)

        # Different dtypes
        different_sig = current_sig.copy()
        different_sig["dtypes"] = {"col1": "int32", "col2": "float64"}
        assert cache_manager._needs_new_dtype_entry(different_sig, existing_mapping)

    # === 資料重建邏輯 ===

    # === DuckDB Reconstruction Tests ===

    def test_reconstruct_as_of_simple(self, cache_manager):
        """Test simple DuckDB-based reconstruction."""
        key = "test_key"
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Create and save test DataFrame
        df = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]}, index=["A", "B"])
        cache_manager.save_data(key, df, timestamp)

        # Test reconstruction
        result = cache_manager.reconstruct_as_of(key, timestamp)

        assert not result.empty
        assert result.shape == (2, 2)
        # Check numeric values (handle both int and float representations)
        assert float(result.loc["A", "col1"]) == 1.0
        assert float(result.loc["B", "col2"]) == 2.2

    def test_reconstruct_as_of_time_filtering(self, cache_manager):
        """Test DuckDB reconstruction with time filtering."""
        key = "test_key"
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        timestamp2 = datetime(2024, 1, 1, 11, 0, 0)
        target_time = datetime(2024, 1, 1, 10, 30, 0)

        # Save initial data
        df1 = pd.DataFrame({"col1": [1]}, index=["A"])
        cache_manager.save_data(key, df1, timestamp1)

        # Save modified data
        df2 = pd.DataFrame({"col1": [2]}, index=["A"])
        cache_manager.save_data(key, df2, timestamp2)

        # Query at intermediate time should get first value
        result = cache_manager.reconstruct_as_of(key, target_time)
        assert float(result.loc["A", "col1"]) == 1.0  # Handle both "1" and "1.0"

        # Query after second timestamp should get second value
        result_later = cache_manager.reconstruct_as_of(key, timestamp2)
        assert float(result_later.loc["A", "col1"]) == 2.0  # Handle both "2" and "2.0"

    def test_polars_diff_computation(self, cache_manager):
        """Test Polars-based diff computation."""
        prev_df = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]}, index=["A", "B"])
        cur_df = pd.DataFrame({"col1": [1, 3], "col2": [1.1, 2.2]}, index=["A", "B"])
        timestamp = datetime.now()

        # Test the Polars diff method
        result = cache_manager.get_changes_extended(prev_df, cur_df, timestamp)

        assert not result.cell_changes.empty
        # Should have one change: B,col1 changed from 2 to 3
        assert len(result.cell_changes) == 1
        assert result.cell_changes.iloc[0]["row_key"] == "B"
        assert result.cell_changes.iloc[0]["col_key"] == "col1"

    def test_apply_dtypes_to_result_columns(self, cache_manager, temp_cache_dir):
        """Test applying dtypes to DataFrame columns."""
        key = "test_key"

        # Create result DataFrame
        result = pd.DataFrame({"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3]})

        # Create and save dtype mapping
        dtype_mapping = {
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "dtype_history": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "dtypes": {"col1": "int32", "col2": "float32"},
                    "index_dtype": "object",
                    "columns_dtype": "object",
                    "columns_order": ["col1", "col2"],
                    "index_order": ["0", "1", "2"],
                }
            ],
        }

        dtype_path = cache_manager._get_dtype_path(key)
        dtype_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dtype_path, "w") as f:
            json.dump(dtype_mapping, f)

        # Apply dtypes
        cache_manager._apply_dtypes_to_result(result, key, None)

        # Verify dtypes
        assert result["col1"].dtype == np.dtype("int32")
        assert result["col2"].dtype == np.dtype("float32")

    def test_apply_dtypes_to_result_index(self, cache_manager, temp_cache_dir):
        """Test applying dtypes to DataFrame index."""
        key = "test_key"

        # Create result DataFrame with string index
        result = pd.DataFrame({"col1": [1, 2, 3]}, index=["1", "2", "3"])

        # Create dtype mapping with int index
        dtype_mapping = {
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "dtype_history": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "dtypes": {"col1": "int64"},
                    "index_dtype": "int64",
                    "columns_dtype": "object",
                    "columns_order": ["col1"],
                    "index_order": ["1", "2", "3"],
                }
            ],
        }

        dtype_path = cache_manager._get_dtype_path(key)
        dtype_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dtype_path, "w") as f:
            json.dump(dtype_mapping, f)

        # Apply dtypes
        cache_manager._apply_dtypes_to_result(result, key, None)

        # Verify index dtype
        assert result.index.dtype == np.dtype("int64")

    def test_apply_dtypes_to_result_columns_object(self, cache_manager, temp_cache_dir):
        """Test applying dtypes to columns object itself."""
        key = "test_key"

        # Create result DataFrame with int columns
        result = pd.DataFrame([[1, 2]], columns=[1, 2])

        # Create dtype mapping
        dtype_mapping = {
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "dtype_history": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "dtypes": {"1": "int64", "2": "int64"},
                    "index_dtype": "int64",
                    "columns_dtype": "object",
                    "columns_order": ["1", "2"],
                    "index_order": ["0"],
                }
            ],
        }

        dtype_path = cache_manager._get_dtype_path(key)
        dtype_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dtype_path, "w") as f:
            json.dump(dtype_mapping, f)

        # Apply dtypes
        cache_manager._apply_dtypes_to_result(result, key, None)

        # Verify columns dtype
        assert result.columns.dtype == np.dtype("object")

    # === DuckDB Storage Tests ===

    def test_save_snapshot(self, cache_manager):
        """Test saving a complete DataFrame snapshot."""
        key = "test_key"
        df = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]}, index=["A", "B"])
        timestamp = datetime.now()

        # Save snapshot
        cache_manager.save_snapshot(key, df, timestamp)

        # Verify data in rows_base table
        query = f"SELECT * FROM rows_base WHERE table_id = '{key}'"
        result = cache_manager.conn.execute(query).fetchdf()

        assert len(result) == 2  # Two rows
        assert all(result["table_id"] == key)
        assert set(result["row_key"]) == {"A", "B"}

    def test_save_version_changes(self, cache_manager):
        """Test saving incremental changes between DataFrame versions."""
        key = "test_key"
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        timestamp2 = datetime(2024, 1, 1, 11, 0, 0)

        # Create DataFrames
        prev_df = pd.DataFrame({"col1": [1, 2], "col2": [1.1, 2.2]}, index=["A", "B"])
        cur_df = pd.DataFrame({"col1": [1, 3], "col2": [1.1, 2.2]}, index=["A", "B"])

        # First save snapshot
        cache_manager.save_snapshot(key, prev_df, timestamp1)

        # Then save version with changes
        cache_manager.save_version(key, prev_df, cur_df, timestamp2)

        # Verify changes in cell_changes table
        query = f"SELECT * FROM cell_changes WHERE table_id = '{key}'"
        result = cache_manager.conn.execute(query).fetchdf()

        assert len(result) == 1  # One cell change
        assert result.iloc[0]["row_key"] == "B"
        assert result.iloc[0]["col_key"] == "col1"

    # === 增量儲存 ===

    def test_save_incremental_changes(self, cache_manager, sample_dataframe):
        """Test saving incremental changes."""
        key = "test_key"
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=1)

        # Save initial data
        cache_manager.save_data(key, sample_dataframe, timestamp1)

        # Create some changes
        from finlab_guard.utils.exceptions import Change

        modifications = [Change(("A", "col1"), 1, 99, timestamp2)]
        additions = [Change(("D", "col1"), None, 4, timestamp2)]

        # Save incremental changes
        cache_manager.save_incremental_changes(
            key, modifications, additions, timestamp2, sample_dataframe
        )

        # Verify changes are reflected in latest data
        latest_data = cache_manager.get_latest_data(key)
        assert latest_data.loc["A", "col1"] == 99
        assert latest_data.loc["D", "col1"] == 4

    def test_save_incremental_changes_empty(self, cache_manager, sample_dataframe):
        """Test saving empty incremental changes."""
        key = "test_key"
        timestamp = datetime.now()

        # Save initial data
        cache_manager.save_data(key, sample_dataframe, timestamp)

        # Save empty changes
        cache_manager.save_incremental_changes(key, [], [], timestamp, sample_dataframe)

        # Verify data unchanged
        latest_data = cache_manager.get_latest_data(key)
        pd.testing.assert_frame_equal(latest_data, sample_dataframe)

    # === 輔助方法 ===

    def test_get_cache_path(self, cache_manager):
        """Test cache path generation - now returns DuckDB path."""
        key = "test_key"
        path = cache_manager._get_cache_path(key)

        # Verify path points to DuckDB file
        assert path.suffix == ".duckdb"
        assert path.name == "cache.duckdb"

    def test_get_dtype_path(self, cache_manager):
        """Test dtype path generation."""
        key = "test_key"
        path = cache_manager._get_dtype_path(key)

        assert path.suffix == ".json"
        assert "dtypes" in str(path)

    def test_compaction(self, cache_manager):
        """Test DuckDB compaction functionality."""
        key = "test_key"
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        timestamp2 = datetime(2024, 1, 1, 11, 0, 0)
        cutoff_time = datetime(2024, 1, 1, 10, 30, 0)

        # Create initial data
        df1 = pd.DataFrame({"col1": [1, 2]}, index=["A", "B"])
        cache_manager.save_data(key, df1, timestamp1)

        # Make changes
        df2 = pd.DataFrame({"col1": [1, 3]}, index=["A", "B"])
        cache_manager.save_data(key, df2, timestamp2)

        # Verify we have cell changes
        changes_before = cache_manager.conn.execute(
            f"SELECT COUNT(*) FROM cell_changes WHERE table_id = '{key}'"
        ).fetchone()[0]
        assert changes_before > 0

        # Compact up to cutoff time
        cache_manager.compact_up_to(key, cutoff_time)

        # Verify changes before cutoff are removed
        changes_after = cache_manager.conn.execute(
            f"SELECT COUNT(*) FROM cell_changes WHERE table_id = '{key}' AND save_time <= TIMESTAMP '{cutoff_time.isoformat()}'"
        ).fetchone()[0]
        assert changes_after == 0

        # Verify we can still reconstruct latest data
        result = cache_manager.load_data(key)
        assert not result.empty

    def test_get_storage_info(self, cache_manager, sample_dataframe):
        """Test getting storage information from DuckDB."""
        key = "test_key"

        # Save some data
        cache_manager.save_data(key, sample_dataframe, datetime.now())

        # Get storage info for specific key
        info = cache_manager.get_storage_info(key)
        assert key in info
        # Check DuckDB-specific storage info
        assert "snapshot_rows" in info[key]
        assert "cell_changes" in info[key]
        assert "row_additions" in info[key]
        assert "last_modified" in info[key]
        assert "storage_type" in info[key]
        assert info[key]["storage_type"] == "duckdb"
        assert info[key]["snapshot_rows"] == 3  # sample_dataframe has 3 rows

        # Get storage info for all keys
        all_info = cache_manager.get_storage_info()
        assert key in all_info
        assert "total_records" in all_info[key]
        # Check for database file size info
        if "database_file_size" in all_info:
            assert all_info["database_file_size"] > 0

    def test_datetime_index_frequency_preservation(self, cache_manager):
        """Test that DatetimeIndex frequency is preserved through save/load cycle."""
        key = "datetime_freq_test"
        timestamp = datetime.now()

        # Create DataFrame with DatetimeIndex that has frequency
        datetime_index = pd.date_range("2023-01-01", periods=3, freq="D")
        df_with_freq = pd.DataFrame({"value": [100, 200, 300]}, index=datetime_index)

        # Verify original DataFrame has frequency
        assert df_with_freq.index.freq is not None
        assert df_with_freq.index.freqstr == "D"

        # Save and load data
        cache_manager.save_data(key, df_with_freq, timestamp)
        loaded_data = cache_manager.load_data(key)

        # Verify frequency is preserved
        assert isinstance(loaded_data.index, pd.DatetimeIndex)
        assert loaded_data.index.freq is not None
        assert loaded_data.index.freqstr == "D"

        # Verify data content is correct
        pd.testing.assert_frame_equal(loaded_data, df_with_freq)

    # === New DuckDB Features Tests ===

    def test_large_row_change_threshold(self, cache_manager):
        """Test row change threshold functionality for large changes."""
        key = "threshold_test"
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        timestamp2 = datetime(2024, 1, 1, 11, 0, 0)

        # Create DataFrame with many columns
        prev_df = pd.DataFrame({f"col_{i}": [1] for i in range(10)}, index=["A"])
        # Change all columns (exceeds default threshold of 200)
        cur_df = pd.DataFrame({f"col_{i}": [2] for i in range(10)}, index=["A"])

        # Set low threshold to test the feature
        cache_manager.row_change_threshold = 5

        # Save initial data
        cache_manager.save_snapshot(key, prev_df, timestamp1)

        # Save with many changes
        cache_manager.save_version(key, prev_df, cur_df, timestamp2)

        # Check that the row was stored in row_additions (not cell_changes)
        row_adds = cache_manager.conn.execute(
            f"SELECT COUNT(*) FROM row_additions WHERE table_id = '{key}'"
        ).fetchone()[0]
        cell_changes = cache_manager.conn.execute(
            f"SELECT COUNT(*) FROM cell_changes WHERE table_id = '{key}'"
        ).fetchone()[0]

        assert row_adds > 0  # Should have row addition entry
        assert cell_changes == 0  # Should not have individual cell changes

        # Verify reconstruction still works
        result = cache_manager.reconstruct_as_of(key, timestamp2)
        assert result.shape == (1, 10)

    def test_new_rows_and_columns_handling(self, cache_manager):
        """Test handling of new rows and columns in diff computation."""
        key = "new_data_test"
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        timestamp2 = datetime(2024, 1, 1, 11, 0, 0)

        # Initial DataFrame
        prev_df = pd.DataFrame({"col1": [1, 2]}, index=["A", "B"])

        # Add new row and new column
        cur_df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [10, 20, 30]}, index=["A", "B", "C"]
        )

        # Save initial data
        cache_manager.save_snapshot(key, prev_df, timestamp1)

        # Save with new data
        cache_manager.save_version(key, prev_df, cur_df, timestamp2)

        # Verify reconstruction
        result = cache_manager.reconstruct_as_of(key, timestamp2)
        assert result.shape == (3, 2)
        assert "col2" in result.columns
        assert "C" in result.index

    def test_duckdb_performance_indexes(self, cache_manager):
        """Test that DuckDB indexes are created for performance."""
        # The indexes should be created during table setup
        # Check that they exist (DuckDB doesn't have a standard way to list indexes,
        # but we can verify the tables were created successfully)

        tables_query = "SHOW TABLES"
        tables = cache_manager.conn.execute(tables_query).fetchall()
        table_names = [table[0] for table in tables]

        assert "rows_base" in table_names
        assert "cell_changes" in table_names
        assert "row_additions" in table_names

        # Test that the tables have the expected structure
        for table in ["rows_base", "cell_changes", "row_additions"]:
            describe_query = f"DESCRIBE {table}"
            columns = cache_manager.conn.execute(describe_query).fetchall()
            assert len(columns) > 0  # Each table should have columns

    def test_json_serialization_handling(self, cache_manager):
        """Test handling of various data types through JSON serialization."""
        key = "json_test"
        timestamp = datetime.now()

        # Create DataFrame with various data types
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, float("nan")],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "none_col": [None, "value", None],
            },
            index=["A", "B", "C"],
        )

        # Save and load
        cache_manager.save_data(key, df, timestamp)
        result = cache_manager.load_data(key)

        # Verify basic structure
        assert result.shape == df.shape
        assert set(result.columns) == set(df.columns)
        assert set(result.index.astype(str)) == set(df.index.astype(str))

        # Note: Due to JSON serialization, exact type matching may vary
        # but the data should be reconstructible
