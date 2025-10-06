"""Tests for row/column re-addition with cell changes lifecycle filtering.

This test module verifies that cell_changes are properly filtered by row and column
lifecycles to prevent stale changes from affecting re-added rows/columns.

The core problem: When a row or column is deleted and then re-added with the same key,
old cell_changes should NOT affect the new version of that row/column.
"""

import shutil
import tempfile
import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from finlab_guard import FinlabGuard


def safe_rmtree(path, ignore_errors=False):
    """Windows-compatible rmtree with retry logic for DuckDB file locking."""
    import gc

    # Force garbage collection to release any lingering file handles
    gc.collect()

    for attempt in range(10):  # Increase retry attempts to 10
        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            break
        except (PermissionError, OSError):
            if attempt < 9:
                # Exponential backoff: 100ms, 200ms, 400ms, etc.
                wait_time = 0.1 * (2 ** min(attempt, 4))
                time.sleep(wait_time)
                gc.collect()  # Try garbage collection again
                continue
            if not ignore_errors:
                raise


class TestColumnReadditionCellChanges:
    """Test cell changes lifecycle filtering for column re-addition scenarios."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        safe_rmtree(temp_dir)

    def test_column_readd_with_cell_change_before_deletion(self, temp_cache_dir):
        """
        Test scenario from issue description (column re-addition):

        Step 1 (T1): Add column col_3, index 0 value = "random_str_52"
        Step 2 (T2): Cell change col_3[0] from "random_str_52" to "random_str_376"
        Step 3 (T3): Delete column col_3
        Step 4 (T4): Re-add column col_3, index 0 value = "random_str_310"

        Expected:
        - Query at Step 2: col_3[0] = "random_str_376" ✅
        - Query at Step 3: col_3 does not exist ✅
        - Query at Step 4: col_3[0] = "random_str_310" ✅ (NOT "random_str_376"!)
        """
        guard = FinlabGuard(cache_dir=temp_cache_dir)
        key = "test_column_readd"

        # Base snapshot (no col_3 yet)
        t1 = datetime(2024, 1, 1, 10, 0, 0)
        df1 = pd.DataFrame({"col_1": [1, 2], "col_2": [10, 20]}, index=["A", "B"])

        # Step 1: Add col_3
        t2 = t1 + timedelta(seconds=1)
        df2 = df1.copy()
        df2["col_3"] = ["random_str_52", "random_str_53"]

        # Step 2: Cell change col_3[A] to "random_str_376"
        t3 = t1 + timedelta(seconds=2)
        df3 = df2.copy()
        df3.loc["A", "col_3"] = "random_str_376"

        # Step 3: Delete col_3
        t4 = t1 + timedelta(seconds=3)
        df4 = df3.drop(columns=["col_3"])

        # Step 4: Re-add col_3 with new values
        t5 = t1 + timedelta(seconds=4)
        df5 = df4.copy()
        df5["col_3"] = ["random_str_310", "random_str_311"]

        # Save all versions
        guard.cache_manager.save_data(key, df1, t1)
        guard.cache_manager.save_data(key, df2, t2)
        guard.cache_manager.save_data(key, df3, t3)
        guard.cache_manager.save_data(key, df4, t4)
        guard.cache_manager.save_data(key, df5, t5)

        # Verify Step 2 (cell change should be applied)
        result_t3 = guard.cache_manager.load_data(key, t3)
        assert "col_3" in result_t3.columns
        assert result_t3.loc["A", "col_3"] == "random_str_376", (
            "Step 2: Cell change should be applied"
        )

        # Verify Step 3 (column deleted)
        result_t4 = guard.cache_manager.load_data(key, t4)
        assert "col_3" not in result_t4.columns, "Step 3: col_3 should be deleted"

        # Verify Step 4 (re-added column, old cell_change should NOT apply)
        result_t5 = guard.cache_manager.load_data(key, t5)
        assert "col_3" in result_t5.columns
        assert result_t5.loc["A", "col_3"] == "random_str_310", (
            "Step 4: Should use new column value, NOT old cell_change"
        )

        guard.close()

    def test_multiple_column_readd_cycles(self, temp_cache_dir):
        """Test multiple delete/re-add cycles for the same column."""
        guard = FinlabGuard(cache_dir=temp_cache_dir)
        key = "test_multi_column_cycles"

        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Initial: col_A exists
        df1 = pd.DataFrame({"col_A": [1, 2]}, index=["X", "Y"])
        guard.cache_manager.save_data(key, df1, base_time)

        # Cycle 1: modify -> delete -> readd
        t1 = base_time + timedelta(seconds=1)
        df2 = df1.copy()
        df2.loc["X", "col_A"] = 100  # Cell change
        guard.cache_manager.save_data(key, df2, t1)

        t2 = base_time + timedelta(seconds=2)
        df3 = df2.drop(columns=["col_A"])  # Delete
        guard.cache_manager.save_data(key, df3, t2)

        t3 = base_time + timedelta(seconds=3)
        df4 = df3.copy()
        df4["col_A"] = [200, 201]  # Re-add
        guard.cache_manager.save_data(key, df4, t3)

        # Cycle 2: modify -> delete -> readd again
        t4 = base_time + timedelta(seconds=4)
        df5 = df4.copy()
        df5.loc["X", "col_A"] = 300  # Cell change in cycle 2
        guard.cache_manager.save_data(key, df5, t4)

        t5 = base_time + timedelta(seconds=5)
        df6 = df5.drop(columns=["col_A"])  # Delete again
        guard.cache_manager.save_data(key, df6, t5)

        t6 = base_time + timedelta(seconds=6)
        df7 = df6.copy()
        df7["col_A"] = [400, 401]  # Re-add again
        guard.cache_manager.save_data(key, df7, t6)

        # Verify each stage
        result_t1 = guard.cache_manager.load_data(key, t1)
        assert result_t1.loc["X", "col_A"] == 100, "Cycle 1: Cell change applied"

        result_t3 = guard.cache_manager.load_data(key, t3)
        assert result_t3.loc["X", "col_A"] == 200, "Cycle 1: Re-add value (NOT 100)"

        result_t4 = guard.cache_manager.load_data(key, t4)
        assert result_t4.loc["X", "col_A"] == 300, "Cycle 2: Cell change applied"

        result_t6 = guard.cache_manager.load_data(key, t6)
        assert result_t6.loc["X", "col_A"] == 400, (
            "Cycle 2: Re-add value (NOT 100 or 300)"
        )

        guard.close()


class TestRowReadditionCellChanges:
    """Test cell changes lifecycle filtering for row re-addition scenarios."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        safe_rmtree(temp_dir)

    def test_row_readd_with_cell_change_before_deletion(self, temp_cache_dir):
        """
        Test row re-addition scenario (symmetric to column test):

        Step 1 (T1): Add row idx_A, col_1 value = "value_52"
        Step 2 (T2): Cell change [idx_A, col_1] from "value_52" to "value_376"
        Step 3 (T3): Delete row idx_A
        Step 4 (T4): Re-add row idx_A, col_1 value = "value_310"

        Expected:
        - Query at Step 2: [idx_A, col_1] = "value_376" ✅
        - Query at Step 3: row idx_A does not exist ✅
        - Query at Step 4: [idx_A, col_1] = "value_310" ✅ (NOT "value_376"!)
        """
        guard = FinlabGuard(cache_dir=temp_cache_dir)
        key = "test_row_readd"

        # Base snapshot (no row idx_A yet)
        t1 = datetime(2024, 1, 1, 10, 0, 0)
        df1 = pd.DataFrame({"col_1": [1, 2], "col_2": [10, 20]}, index=["B", "C"])

        # Step 1: Add row idx_A (use int to avoid dtype change which creates new snapshot)
        t2 = t1 + timedelta(seconds=1)
        df2 = pd.concat(
            [df1, pd.DataFrame({"col_1": [100], "col_2": [30]}, index=["A"])]
        )

        # Step 2: Cell change [A, col_1] to 200
        t3 = t1 + timedelta(seconds=2)
        df3 = df2.copy()
        df3.loc["A", "col_1"] = 200

        # Step 3: Delete row A
        t4 = t1 + timedelta(seconds=3)
        df4 = df3.drop(index=["A"])

        # Step 4: Re-add row A with new values
        t5 = t1 + timedelta(seconds=4)
        df5 = pd.concat(
            [df4, pd.DataFrame({"col_1": [300], "col_2": [40]}, index=["A"])]
        )

        # Save all versions
        guard.cache_manager.save_data(key, df1, t1)
        guard.cache_manager.save_data(key, df2, t2)
        guard.cache_manager.save_data(key, df3, t3)
        guard.cache_manager.save_data(key, df4, t4)
        guard.cache_manager.save_data(key, df5, t5)

        # Verify Step 2 (cell change should be applied)
        result_t3 = guard.cache_manager.load_data(key, t3)
        assert "A" in result_t3.index
        assert result_t3.loc["A", "col_1"] == 200, (
            "Step 2: Cell change should be applied"
        )

        # Verify Step 3 (row deleted)
        result_t4 = guard.cache_manager.load_data(key, t4)
        assert "A" not in result_t4.index, "Step 3: Row A should be deleted"

        # Verify Step 4 (re-added row, old cell_change should NOT apply)
        result_t5 = guard.cache_manager.load_data(key, t5)
        assert "A" in result_t5.index
        assert result_t5.loc["A", "col_1"] == 300, (
            "Step 4: Should use new row value, NOT old cell_change"
        )

        guard.close()

    def test_multiple_row_readd_cycles(self, temp_cache_dir):
        """Test multiple delete/re-add cycles for the same row."""
        guard = FinlabGuard(cache_dir=temp_cache_dir)
        key = "test_multi_row_cycles"

        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Initial: row X exists
        df1 = pd.DataFrame({"col_A": [1, 2]}, index=["X", "Y"])
        guard.cache_manager.save_data(key, df1, base_time)

        # Cycle 1: modify -> delete -> readd
        t1 = base_time + timedelta(seconds=1)
        df2 = df1.copy()
        df2.loc["X", "col_A"] = 100  # Cell change
        guard.cache_manager.save_data(key, df2, t1)

        t2 = base_time + timedelta(seconds=2)
        df3 = df2.drop(index=["X"])  # Delete
        guard.cache_manager.save_data(key, df3, t2)

        t3 = base_time + timedelta(seconds=3)
        df4 = pd.concat([df3, pd.DataFrame({"col_A": [200]}, index=["X"])])  # Re-add
        guard.cache_manager.save_data(key, df4, t3)

        # Cycle 2: modify -> delete -> readd again
        t4 = base_time + timedelta(seconds=4)
        df5 = df4.copy()
        df5.loc["X", "col_A"] = 300  # Cell change in cycle 2
        guard.cache_manager.save_data(key, df5, t4)

        t5 = base_time + timedelta(seconds=5)
        df6 = df5.drop(index=["X"])  # Delete again
        guard.cache_manager.save_data(key, df6, t5)

        t6 = base_time + timedelta(seconds=6)
        df7 = pd.concat(
            [df6, pd.DataFrame({"col_A": [400]}, index=["X"])]
        )  # Re-add again
        guard.cache_manager.save_data(key, df7, t6)

        # Verify each stage
        result_t1 = guard.cache_manager.load_data(key, t1)
        assert result_t1.loc["X", "col_A"] == 100, "Cycle 1: Cell change applied"

        result_t3 = guard.cache_manager.load_data(key, t3)
        assert result_t3.loc["X", "col_A"] == 200, "Cycle 1: Re-add value (NOT 100)"

        result_t4 = guard.cache_manager.load_data(key, t4)
        assert result_t4.loc["X", "col_A"] == 300, "Cycle 2: Cell change applied"

        result_t6 = guard.cache_manager.load_data(key, t6)
        assert result_t6.loc["X", "col_A"] == 400, (
            "Cycle 2: Re-add value (NOT 100 or 300)"
        )

        guard.close()


class TestMixedRowColumnReaddition:
    """Test cell changes with both row AND column re-additions."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        safe_rmtree(temp_dir)

    def test_cell_change_survives_only_in_valid_lifecycle(self, temp_cache_dir):
        """
        Test that a cell_change is only valid when BOTH its row and column are alive.

        Timeline:
        T1: Base with row A, col_1
        T2: Cell change [A, col_1] = 100
        T3: Delete col_1
        T4: Re-add col_1 (cell change from T2 should NOT apply)
        T5: Delete row A
        T6: Re-add row A (cell change from T2 should NOT apply)
        """
        guard = FinlabGuard(cache_dir=temp_cache_dir)
        key = "test_mixed_lifecycle"

        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # T1: Base
        df1 = pd.DataFrame({"col_1": [1, 2]}, index=["A", "B"])
        guard.cache_manager.save_data(key, df1, base_time)

        # T2: Cell change
        t2 = base_time + timedelta(seconds=1)
        df2 = df1.copy()
        df2.loc["A", "col_1"] = 100
        guard.cache_manager.save_data(key, df2, t2)

        # T3: Delete col_1
        t3 = base_time + timedelta(seconds=2)
        df3 = df2.drop(columns=["col_1"])
        guard.cache_manager.save_data(key, df3, t3)

        # T4: Re-add col_1
        t4 = base_time + timedelta(seconds=3)
        df4 = df3.copy()
        df4["col_1"] = [200, 201]
        guard.cache_manager.save_data(key, df4, t4)

        # T5: Delete row A
        t5 = base_time + timedelta(seconds=4)
        df5 = df4.drop(index=["A"])
        guard.cache_manager.save_data(key, df5, t5)

        # T6: Re-add row A
        t6 = base_time + timedelta(seconds=5)
        df6 = pd.concat([df5, pd.DataFrame({"col_1": [300]}, index=["A"])])
        guard.cache_manager.save_data(key, df6, t6)

        # Verify T2: cell change applied
        result_t2 = guard.cache_manager.load_data(key, t2)
        assert result_t2.loc["A", "col_1"] == 100

        # Verify T4: column re-added, cell change from T2 should NOT apply
        result_t4 = guard.cache_manager.load_data(key, t4)
        assert result_t4.loc["A", "col_1"] == 200, (
            "T4: Old cell_change blocked by column lifecycle"
        )

        # Verify T6: row re-added, cell change from T2 should NOT apply
        result_t6 = guard.cache_manager.load_data(key, t6)
        assert result_t6.loc["A", "col_1"] == 300, (
            "T6: Old cell_change blocked by row lifecycle"
        )

        guard.close()

    def test_simultaneous_row_col_delete_and_readd(self, temp_cache_dir):
        """Test edge case: both row and column deleted then re-added together."""
        guard = FinlabGuard(cache_dir=temp_cache_dir)
        key = "test_simultaneous"

        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # T1: Base with row A, col_X
        df1 = pd.DataFrame({"col_X": [1, 2], "col_Y": [10, 20]}, index=["A", "B"])
        guard.cache_manager.save_data(key, df1, base_time)

        # T2: Cell change [A, col_X] = 999
        t2 = base_time + timedelta(seconds=1)
        df2 = df1.copy()
        df2.loc["A", "col_X"] = 999
        guard.cache_manager.save_data(key, df2, t2)

        # T3: Delete BOTH row A and col_X
        t3 = base_time + timedelta(seconds=2)
        df3 = df2.drop(index=["A"]).drop(columns=["col_X"])
        guard.cache_manager.save_data(key, df3, t3)

        # T4: Re-add BOTH row A and col_X
        t4 = base_time + timedelta(seconds=3)
        df4 = pd.concat([df3, pd.DataFrame({"col_Y": [30]}, index=["A"])])
        # Explicitly set values by index to avoid confusion with row order
        df4.loc["B", "col_X"] = 101
        df4.loc["A", "col_X"] = 100
        guard.cache_manager.save_data(key, df4, t4)

        # Verify T2: cell change applied
        result_t2 = guard.cache_manager.load_data(key, t2)
        assert result_t2.loc["A", "col_X"] == 999

        # Verify T4: both re-added, old cell_change should NOT apply
        result_t4 = guard.cache_manager.load_data(key, t4)
        assert result_t4.loc["A", "col_X"] == 100, (
            "T4: Old cell_change blocked by both row and column lifecycle"
        )

        guard.close()
