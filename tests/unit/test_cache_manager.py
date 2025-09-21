"""Unit tests for CacheManager class."""

import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from finlab_guard.cache.manager import CacheManager
from finlab_guard.utils.exceptions import InvalidDataTypeException


class TestCacheManager:
    """Test suite for CacheManager class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager instance for testing."""
        config = {"compression": "snappy"}
        return CacheManager(temp_cache_dir, config)

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

    def test_reconstruct_dataframe_simple(self, cache_manager):
        """Test simple DataFrame reconstruction."""
        key = "test_key"
        stacked_data = pd.DataFrame(
            {
                "index": ["A", "B"],
                "column": ["col1", "col1"],
                "value": [1, 2],
                "save_time": [datetime.now(), datetime.now()],
            }
        )

        result = cache_manager._reconstruct_dataframe(stacked_data, None, key)

        assert not result.empty
        assert result.loc["A", "col1"] == 1
        assert result.loc["B", "col1"] == 2

    def test_reconstruct_dataframe_time_filtering(self, cache_manager):
        """Test DataFrame reconstruction with time filtering."""
        key = "test_key"
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        timestamp2 = datetime(2024, 1, 1, 11, 0, 0)
        target_time = datetime(2024, 1, 1, 10, 30, 0)

        stacked_data = pd.DataFrame(
            {
                "index": ["A", "A"],
                "column": ["col1", "col1"],
                "value": [1, 2],  # Second value should be filtered out
                "save_time": [timestamp1, timestamp2],
            }
        )

        result = cache_manager._reconstruct_dataframe(stacked_data, target_time, key)

        assert result.loc["A", "col1"] == 1  # Should get first value

    def test_reconstruct_dataframe_order_preservation(
        self, cache_manager, temp_cache_dir
    ):
        """Test that column and index order is preserved during reconstruction."""
        key = "test_key"

        # Create dtype mapping with specific order
        dtype_mapping = {
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "dtype_history": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "dtypes": {"col2": "int64", "col1": "int64"},
                    "index_dtype": "object",
                    "columns_dtype": "object",
                    "columns_order": ["col2", "col1"],  # Specific order
                    "index_order": ["B", "A"],  # Specific order
                }
            ],
        }

        # Save dtype mapping
        dtype_path = cache_manager._get_dtype_path(key)
        dtype_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dtype_path, "w") as f:
            json.dump(dtype_mapping, f)

        # Create stacked data
        stacked_data = pd.DataFrame(
            {
                "index": ["A", "A", "B", "B"],
                "column": ["col1", "col2", "col1", "col2"],
                "value": [1, 2, 3, 4],
                "save_time": [datetime.now()] * 4,
            }
        )

        result = cache_manager._reconstruct_dataframe(stacked_data, None, key)

        # Verify column order
        assert list(result.columns) == ["col2", "col1"]
        # Verify index order
        assert list(result.index) == ["B", "A"]

    def test_reconstruct_dataframe_numeric_types(self, cache_manager):
        """Test reconstruction with various numeric types."""
        key = "test_key"

        stacked_data = pd.DataFrame(
            {
                "index": ["A", "A", "A"],
                "column": ["int_col", "float_col", "bool_col"],
                "value": [1, 1.5, True],
                "save_time": [datetime.now()] * 3,
            }
        )

        result = cache_manager._reconstruct_dataframe(stacked_data, None, key)

        assert not result.empty
        assert result.loc["A", "int_col"] == 1
        assert result.loc["A", "float_col"] == 1.5
        assert result.loc["A", "bool_col"]

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

    def test_stack_dataframe_normal(self, cache_manager):
        """Test stacking normal DataFrame."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=["A", "B"])
        timestamp = datetime.now()

        result = cache_manager._stack_dataframe(df, timestamp)

        assert len(result) == 4  # 2 rows × 2 columns
        # Check expected columns (actual column names depend on index/columns names)
        expected_cols = [
            "_finlab_index_none",
            "_finlab_columns_none",
            "value",
            "save_time",
        ]
        assert all(col in result.columns for col in expected_cols)

    def test_stack_dataframe_none_names(self, cache_manager):
        """Test stacking DataFrame with None names."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.index.name = None
        df.columns.name = None
        timestamp = datetime.now()

        result = cache_manager._stack_dataframe(df, timestamp)

        assert len(result) == 4
        assert not result.empty

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
        """Test cache path generation."""
        key = "test:key/with\\special*chars"
        path = cache_manager._get_cache_path(key)

        # Verify path is valid and safe
        assert path.suffix == ".parquet"
        # Check that some special chars are replaced in filename (: / \)
        assert (
            ":" not in path.name
        )  # Check only filename, not full path (Windows has C:)
        assert "/" not in path.name  # / should be replaced in filename
        assert "\\" not in path.name  # \ should be replaced in filename

    def test_get_dtype_path(self, cache_manager):
        """Test dtype path generation."""
        key = "test_key"
        path = cache_manager._get_dtype_path(key)

        assert path.suffix == ".json"
        assert "dtypes" in str(path)

    def test_atomic_write_parquet(self, cache_manager, sample_dataframe):
        """Test atomic parquet writing."""
        key = "test_key"

        # This should not raise any exceptions
        cache_manager._atomic_write_parquet(key, sample_dataframe)

        # Verify file exists
        cache_path = cache_manager._get_cache_path(key)
        assert cache_path.exists()

    def test_get_storage_info(self, cache_manager, sample_dataframe):
        """Test getting storage information."""
        key = "test_key"

        # Save some data
        cache_manager.save_data(key, sample_dataframe, datetime.now())

        # Get storage info for specific key
        info = cache_manager.get_storage_info(key)
        assert key in info
        assert "file_size" in info[key]
        assert "modified_time" in info[key]
        assert "record_count" in info[key]

        # Get storage info for all keys
        all_info = cache_manager.get_storage_info()
        assert "total_size" in all_info
        assert key in all_info

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
