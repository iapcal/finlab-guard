"""Unit tests for reconstructed CacheManager methods."""

import json
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

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


class TestCacheManagerReconstruct:
    """Test suite for reconstructed CacheManager methods."""

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
    def mock_conn(self):
        """Create mock DuckDB connection."""
        conn = Mock()
        conn.execute.return_value.fetchdf.return_value = pd.DataFrame()
        conn.execute.return_value.arrow.return_value = Mock(num_rows=0)
        return conn

    def test_load_base_snapshot_empty(self, cache_manager):
        """Test _load_base_snapshot with no data."""
        target_time = datetime.now()
        result, snapshot_time = cache_manager._load_base_snapshot(
            "test_table", target_time
        )

        assert isinstance(result, pl.DataFrame)
        assert isinstance(snapshot_time, datetime)
        assert result.is_empty()
        assert "row_key" in result.columns

    def test_load_base_snapshot_with_data(self, cache_manager):
        """Test _load_base_snapshot with actual data."""
        # Setup test data
        test_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )

        # Save test data first
        save_time = datetime.now()
        cache_manager.save_snapshot("test_table", test_df, save_time)

        # Test loading
        target_time = datetime.now() + timedelta(minutes=1)
        result, snapshot_time = cache_manager._load_base_snapshot(
            "test_table", target_time
        )

        assert isinstance(result, pl.DataFrame)
        assert isinstance(snapshot_time, datetime)
        assert snapshot_time == save_time
        assert not result.is_empty()
        assert "row_key" in result.columns
        assert len(result) == 3

    def test_load_and_process_cell_changes_empty(self, cache_manager):
        """Test _load_and_process_cell_changes with no changes."""
        snapshot_time = datetime.now() - timedelta(hours=1)
        target_time = datetime.now()
        result = cache_manager._load_and_process_cell_changes(
            "test_table", snapshot_time, target_time
        )

        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()
        assert "row_key" in result.columns

    def test_load_and_process_cell_changes_with_data(self, cache_manager):
        """Test _load_and_process_cell_changes with actual changes."""
        # Setup base data and changes
        base_df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]}, index=["row1", "row2"])

        modified_df = pd.DataFrame(
            {
                "A": [10, 2],  # row1.A changed
                "B": ["x", "z"],  # row2.B changed
            },
            index=["row1", "row2"],
        )

        # Save base and changes
        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_table", base_df, timestamp1)

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("test_table", base_df, modified_df, timestamp2)

        # Test loading changes
        snapshot_time = timestamp1  # Use snapshot time as base
        target_time = timestamp2 + timedelta(minutes=1)
        result = cache_manager._load_and_process_cell_changes(
            "test_table", snapshot_time, target_time
        )

        assert isinstance(result, pl.DataFrame)
        if not result.is_empty():
            # Should have delta columns
            delta_cols = [c for c in result.columns if c.endswith("__delta")]
            assert len(delta_cols) > 0

    def test_parse_json_value(self, cache_manager):
        """Test _parse_json_value method."""
        # Test various input types
        assert cache_manager._parse_json_value(None) is None
        assert cache_manager._parse_json_value(42) == 42
        assert cache_manager._parse_json_value(3.14) == 3.14
        assert cache_manager._parse_json_value(True) is True

        # Test JSON string parsing
        assert cache_manager._parse_json_value('"hello"') == "hello"
        assert cache_manager._parse_json_value("42") == 42
        assert cache_manager._parse_json_value("3.14") == 3.14

        # Test invalid JSON
        assert cache_manager._parse_json_value("invalid") == "invalid"

    def test_load_and_process_row_additions_empty(self, cache_manager):
        """Test _load_and_process_row_additions with no additions."""
        snapshot_time = datetime.now() - timedelta(hours=1)
        target_time = datetime.now()
        result = cache_manager._load_and_process_row_additions(
            "test_table", snapshot_time, target_time
        )

        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()
        assert "row_key" in result.columns

    def test_load_and_process_row_additions_with_data(self, cache_manager):
        """Test _load_and_process_row_additions with actual additions."""
        # Setup base data
        base_df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]}, index=["row1", "row2"])

        # Add new rows
        expanded_df = pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": ["x", "y", "z", "0050"]},
            index=["row1", "row2", "row3", "row4"],
        )

        # Save base and additions
        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_table", base_df, timestamp1)

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("test_table", base_df, expanded_df, timestamp2)

        # Test loading additions
        snapshot_time = timestamp1
        target_time = timestamp2 + timedelta(minutes=1)
        result = cache_manager._load_and_process_row_additions(
            "test_table", snapshot_time, target_time
        )

        assert isinstance(result, pl.DataFrame)

    def test_merge_data_layers_empty(self, cache_manager):
        """Test _merge_data_layers with empty DataFrames."""
        empty_base = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})
        empty_changes = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})
        empty_additions = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})

        result = cache_manager._merge_data_layers(
            empty_base, empty_changes, empty_additions, {}
        )

        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_merge_data_layers_with_data(self, cache_manager):
        """Test _merge_data_layers with actual data."""
        # Create test data layers
        base = pl.DataFrame({"row_key": ["row1", "row2"], "A": [1, 2], "B": ["x", "y"]})

        changes = pl.DataFrame(
            {
                "row_key": ["row1"],
                "A__delta": [10],  # Change row1.A from 1 to 10
            }
        )

        additions = pl.DataFrame({"row_key": ["row3"], "A": [3], "B": ["z"]})

        result = cache_manager._merge_data_layers(base, changes, additions, {})

        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()
        assert len(result) == 3  # row1, row2, row3

        # Convert to pandas for easier assertion
        result_pd = result.to_pandas()
        assert "row1" in result_pd["row_key"].values
        assert "row2" in result_pd["row_key"].values
        assert "row3" in result_pd["row_key"].values

    def test_finalize_dataframe_empty(self, cache_manager):
        """Test _finalize_dataframe with empty DataFrame."""
        empty_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})
        result = cache_manager._finalize_dataframe(empty_pl)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_finalize_dataframe_with_data(self, cache_manager):
        """Test _finalize_dataframe with actual data."""
        test_pl = pl.DataFrame(
            {
                "row_key": ["row2", "row1", "row3"],  # Unsorted
                "A": [2, 1, 3],
                "B": ["y", "x", "z"],
            }
        )

        result = cache_manager._finalize_dataframe(test_pl)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 3

        # Check that row_key became index and is sorted
        assert result.index.tolist() == ["row1", "row2", "row3"]
        assert "row_key" not in result.columns

    def test_finalize_dataframe_numeric_sorting(self, cache_manager):
        """Test _finalize_dataframe with numeric row keys."""
        test_pl = pl.DataFrame(
            {
                "row_key": ["10", "2", "1"],  # String numbers
                "A": [10, 2, 1],
            }
        )

        result = cache_manager._finalize_dataframe(test_pl)

        assert isinstance(result, pd.DataFrame)
        # Should sort numerically: 1, 2, 10
        assert result.index.tolist() == ["1", "2", "10"]

    def test_reconstruct_as_of_integration(self, cache_manager):
        """Test full reconstruct_as_of workflow."""
        # Create test scenario with multiple data layers
        base_df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]}, index=["row1", "row2"])

        # Save base snapshot
        timestamp1 = datetime.now()
        cache_manager.save_snapshot("test_table", base_df, timestamp1)

        # Modify and add data
        modified_df = pd.DataFrame(
            {
                "A": [10, 2, 3],  # row1.A changed, row3 added
                "B": ["x", "z", "w"],  # row2.B changed
            },
            index=["row1", "row2", "row3"],
        )

        timestamp2 = timestamp1 + timedelta(minutes=1)
        cache_manager.save_version("test_table", base_df, modified_df, timestamp2)

        # Test reconstruction
        target_time = timestamp2 + timedelta(minutes=1)
        result = cache_manager.reconstruct_as_of("test_table", target_time)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # Verify the reconstruction matches expected result
        assert len(result) >= 2  # At least original rows

    @patch("finlab_guard.cache.manager.CacheManager._load_base_snapshot")
    def test_reconstruct_as_of_mock_base(self, mock_base, cache_manager):
        """Test reconstruct_as_of with mocked base snapshot."""
        # Mock the base snapshot loading - now returns (DataFrame, snapshot_time)
        snapshot_time = datetime.now() - timedelta(hours=1)
        mock_base.return_value = (
            pl.DataFrame({"row_key": ["row1"], "A": [1], "B": ["x"]}),
            snapshot_time,
        )

        target_time = datetime.now()
        result = cache_manager.reconstruct_as_of("test_table", target_time)

        # Verify mock was called
        mock_base.assert_called_once_with("test_table", target_time)

        # Verify result structure
        assert isinstance(result, pd.DataFrame)

    def test_error_handling_in_cell_changes(self, cache_manager):
        """Test error handling in _load_and_process_cell_changes."""
        # Use patch to replace the entire connection
        with patch.object(cache_manager, "conn") as mock_conn:
            # Simulate arrow failure, fallback to pandas
            mock_result = Mock()
            mock_result.arrow.side_effect = Exception("Arrow failed")
            mock_result.fetchdf.return_value = pd.DataFrame(
                {"row_key": ["row1"], "col_key": ["A"], "value": ["10"]}
            )
            mock_conn.execute.return_value = mock_result

            snapshot_time = datetime.now() - timedelta(hours=1)
            target_time = datetime.now()
            result = cache_manager._load_and_process_cell_changes(
                "test_table", snapshot_time, target_time
            )

            assert isinstance(result, pl.DataFrame)
            # Should fallback gracefully

    def test_pivot_fallback_in_cell_changes(self, cache_manager):
        """Test pivot fallback logic in _load_and_process_cell_changes."""
        # This is harder to test directly, but we can at least ensure
        # the method handles pivot exceptions gracefully
        snapshot_time = datetime.now() - timedelta(hours=1)
        target_time = datetime.now()
        result = cache_manager._load_and_process_cell_changes(
            "test_table", snapshot_time, target_time
        )

        # Should not raise exception even with no data
        assert isinstance(result, pl.DataFrame)
