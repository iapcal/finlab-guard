"""Unit tests for snapshot-time-aware reconstruction.

This module tests that reconstruction correctly applies changes only AFTER
the snapshot time, not before.
"""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finlab_guard.cache.manager import CacheManager


class TestSnapshotTimeReconstruction:
    """Test that reconstruction respects snapshot boundaries."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager instance for testing."""
        config = {"compression": None}
        manager = CacheManager(temp_cache_dir, config)
        yield manager
        manager.close()

    def test_reconstruction_ignores_changes_before_snapshot(self, cache_manager):
        """
        Test that reconstruction only applies changes AFTER the snapshot time.

        Timeline:
        T0: snapshot_1 with A=100, B=200
        T1: cell_change A=100->105 (should be ignored when using snapshot_2)
        T2: snapshot_2 with A=100, B=200 (dtype changed, new snapshot)
        T3: cell_change A=100->110

        Query at T2.5 (between T2 and T3):
        - Should use snapshot_2 (created at T2)
        - Should NOT apply T1 change (T1 < T2, before snapshot_2)
        - Should NOT apply T3 change (T3 > T2.5, after query time)
        - Result: A=100, B=200 (exactly as snapshot_2)
        """
        T0 = datetime(2025, 1, 1, 10, 0, 0)
        T1 = datetime(2025, 1, 1, 11, 0, 0)
        T2 = datetime(2025, 1, 1, 12, 0, 0)
        T3 = datetime(2025, 1, 1, 13, 0, 0)
        T2_5 = datetime(2025, 1, 1, 12, 30, 0)

        # T0: Create initial snapshot
        df0 = pd.DataFrame(
            {
                "col1": np.array([100, 200], dtype="int32"),
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        cache_manager.save_snapshot("test", df0, T0)

        # T1: Modify A to 105 (cell change)
        df1 = pd.DataFrame(
            {
                "col1": np.array([105, 200], dtype="int32"),
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        prev_df = cache_manager.load_data("test")
        cache_manager.save_version("test", prev_df, df1, T1)

        # T2: Create new snapshot (e.g., due to dtype change)
        df2 = pd.DataFrame(
            {
                "col1": np.array([100, 200], dtype="int64"),  # dtype changed
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        cache_manager.save_snapshot("test", df2, T2)

        # T3: Modify A to 110 (cell change after snapshot_2)
        df3 = pd.DataFrame(
            {
                "col1": np.array([110, 200], dtype="int64"),
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        prev_df2 = cache_manager.load_data("test")
        cache_manager.save_version("test", prev_df2, df3, T3)

        # Query at T2.5 (between snapshot_2 and T3)
        result = cache_manager.load_data("test", T2_5)

        # Expected: A=100, B=200 (from snapshot_2, no changes applied)
        # The T1 change should NOT be applied because T1 < T2 (before snapshot_2)
        # The T3 change should NOT be applied because T3 > T2.5 (after query time)
        assert result.loc["A", "col1"] == 100, (
            f"Expected A=100 (from snapshot_2), got {result.loc['A', 'col1']}. "
            "T1 change should NOT be applied (before snapshot_2)"
        )
        assert result.loc["B", "col1"] == 200

    def test_reconstruction_applies_changes_after_snapshot(self, cache_manager):
        """
        Test that reconstruction correctly applies changes after snapshot time.

        Timeline:
        T0: snapshot_1 with A=100, B=200
        T1: cell_change A=100->105
        T2: snapshot_2 with A=100, B=200 (new snapshot)
        T3: cell_change A=100->110

        Query at T3.5 (after T3):
        - Should use snapshot_2 (created at T2)
        - Should apply T3 change (T2 < T3 < T3.5)
        - Result: A=110, B=200
        """
        T0 = datetime(2025, 1, 1, 10, 0, 0)
        T1 = datetime(2025, 1, 1, 11, 0, 0)
        T2 = datetime(2025, 1, 1, 12, 0, 0)
        T3 = datetime(2025, 1, 1, 13, 0, 0)
        T3_5 = datetime(2025, 1, 1, 13, 30, 0)

        # T0: Create initial snapshot
        df0 = pd.DataFrame(
            {
                "col1": np.array([100, 200], dtype="int32"),
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        cache_manager.save_data("test", df0, T0)

        # T1: Modify A to 105
        df1 = pd.DataFrame(
            {
                "col1": np.array([105, 200], dtype="int32"),
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        cache_manager.save_data("test", df1, T1)

        # T2: Create new snapshot
        df2 = pd.DataFrame(
            {
                "col1": np.array([100, 200], dtype="int64"),
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        cache_manager.save_data("test", df2, T2)

        # T3: Modify A to 110
        df3 = pd.DataFrame(
            {
                "col1": np.array([110, 200], dtype="int64"),
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        cache_manager.save_data("test", df3, T3)

        # Query at T3.5 (after T3)
        result = cache_manager.load_data("test", T3_5)

        # Expected: A=110 (T3 change applied)
        assert result.loc["A", "col1"] == 110, (
            f"Expected A=110 (T3 change applied), got {result.loc['A', 'col1']}"
        )
        assert result.loc["B", "col1"] == 200

    def test_row_additions_only_after_snapshot(self, cache_manager):
        """
        Test that row additions are only applied if added after snapshot.

        Timeline:
        T0: snapshot_1 with A, B
        T1: add row C
        T2: snapshot_2 with A, B (C should be gone)
        T3: add row D

        Query at T2.5:
        - Should have A, B (from snapshot_2)
        - Should NOT have C (added before snapshot_2)
        - Should NOT have D (added after query time)
        """
        T0 = datetime(2025, 1, 1, 10, 0, 0)
        T1 = datetime(2025, 1, 1, 11, 0, 0)
        T2 = datetime(2025, 1, 1, 12, 0, 0)
        T3 = datetime(2025, 1, 1, 13, 0, 0)
        T2_5 = datetime(2025, 1, 1, 12, 30, 0)

        # T0: Create initial snapshot with A, B
        df0 = pd.DataFrame(
            {"col1": np.array([100, 200], dtype="int32")},
            index=["A", "B"],
        )
        cache_manager.save_snapshot("test", df0, T0)

        # T1: Add row C
        df1 = pd.DataFrame(
            {"col1": np.array([100, 200, 300], dtype="int32")},
            index=["A", "B", "C"],
        )
        prev_df = cache_manager.load_data("test")
        cache_manager.save_version("test", prev_df, df1, T1)

        # T2: Create new snapshot with only A, B (C removed)
        df2 = pd.DataFrame(
            {"col1": np.array([100, 200], dtype="int32")},
            index=["A", "B"],
        )
        cache_manager.save_snapshot("test", df2, T2)

        # T3: Add row D
        df3 = pd.DataFrame(
            {"col1": np.array([100, 200, 400], dtype="int32")},
            index=["A", "B", "D"],
        )
        prev_df2 = cache_manager.load_data("test")
        cache_manager.save_version("test", prev_df2, df3, T3)

        # Query at T2.5
        result = cache_manager.load_data("test", T2_5)

        # Should have only A, B (from snapshot_2)
        assert list(result.index) == ["A", "B"], (
            f"Expected index ['A', 'B'], got {list(result.index)}. "
            "C should not be present (added before snapshot_2)"
        )

    def test_column_additions_only_after_snapshot(self, cache_manager):
        """
        Test that column additions are only applied if added after snapshot.

        Timeline:
        T0: snapshot_1 with col1
        T1: add col2
        T2: snapshot_2 with col1 only (col2 removed)
        T3: add col3

        Query at T2.5:
        - Should have only col1 (from snapshot_2)
        - Should NOT have col2 (added before snapshot_2)
        - Should NOT have col3 (added after query time)
        """
        T0 = datetime(2025, 1, 1, 10, 0, 0)
        T1 = datetime(2025, 1, 1, 11, 0, 0)
        T2 = datetime(2025, 1, 1, 12, 0, 0)
        T3 = datetime(2025, 1, 1, 13, 0, 0)
        T2_5 = datetime(2025, 1, 1, 12, 30, 0)

        # T0: Create initial snapshot with col1
        df0 = pd.DataFrame(
            {"col1": np.array([100, 200], dtype="int32")},
            index=["A", "B"],
        )
        cache_manager.save_snapshot("test", df0, T0)

        # T1: Add col2
        df1 = pd.DataFrame(
            {
                "col1": np.array([100, 200], dtype="int32"),
                "col2": np.array([1.1, 2.2], dtype="float32"),
            },
            index=["A", "B"],
        )
        prev_df = cache_manager.load_data("test")
        cache_manager.save_version("test", prev_df, df1, T1)

        # T2: Create new snapshot with only col1 (col2 removed)
        df2 = pd.DataFrame(
            {"col1": np.array([100, 200], dtype="int32")},
            index=["A", "B"],
        )
        cache_manager.save_snapshot("test", df2, T2)

        # T3: Add col3
        df3 = pd.DataFrame(
            {
                "col1": np.array([100, 200], dtype="int32"),
                "col3": np.array([3.3, 4.4], dtype="float32"),
            },
            index=["A", "B"],
        )
        prev_df2 = cache_manager.load_data("test")
        cache_manager.save_version("test", prev_df2, df3, T3)

        # Query at T2.5
        result = cache_manager.load_data("test", T2_5)

        # Should have only col1 (from snapshot_2)
        assert list(result.columns) == ["col1"], (
            f"Expected columns ['col1'], got {list(result.columns)}. "
            "col2 should not be present (added before snapshot_2)"
        )
