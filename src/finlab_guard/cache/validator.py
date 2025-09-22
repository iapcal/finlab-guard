"""Optimized DataValidator for finlab-guard

This file contains a rewritten DataValidator.detect_changes_detailed that
focuses on performance:
 - subset to intersections (avoid constructing full product MultiIndex)
 - split numeric vs non-numeric columns
 - use NumPy vectorized ops for numeric columns
 - use pandas.compare / vectorized boolean masks for non-numeric
 - batch-create Change objects using iat / to_numpy to avoid repeated label lookups

Note: this module expects `Change` and exception classes to be available in the package.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from ..utils.exceptions import (
    Change,
    InvalidDataTypeException,
    UnsupportedDataFormatException,
)
from .manager import CacheManager

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates DataFrame format and detects data changes (optimized).

    Key optimizations:
      - operate on the intersection sub-DataFrames for modification detection
      - separate numeric and non-numeric columns and use efficient vector ops
      - use stack(dropna=True) for additions
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize DataValidator with specified tolerance for float comparisons.

        Args:
            tolerance: Absolute tolerance for float comparisons. Defaults to 1e-10.
                      Use 0.0 for exact comparison.
        """
        self.tolerance = tolerance

    def validate_dataframe_format(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame format is supported.

        Args:
            df: DataFrame to validate

        Raises:
            InvalidDataTypeException: If not a DataFrame
            UnsupportedDataFormatException: If format is unsupported
        """
        if not isinstance(df, pd.DataFrame):
            raise InvalidDataTypeException(f"Expected DataFrame, got {type(df)}")

        # Check for MultiIndex columns (not supported)
        if isinstance(df.columns, pd.MultiIndex):
            raise UnsupportedDataFormatException("MultiIndex columns are not supported")

        # Check for MultiIndex index (not supported)
        if isinstance(df.index, pd.MultiIndex):
            raise UnsupportedDataFormatException("MultiIndex index is not supported")

        logger.debug(f"DataFrame validation passed: shape {df.shape}")

    def detect_changes_detailed(
        self, key: str, new_data: pd.DataFrame, cache_manager: CacheManager
    ) -> tuple[list[Change], list[Change]]:
        """
        Detect detailed changes between cached and new data using vectorized operations.

        Args:
            key: Dataset key
            new_data: New DataFrame to compare
            cache_manager: CacheManager instance

        Returns:
            Tuple of (modifications, additions) where:
            - modifications: List of changes to existing historical data
            - additions: List of new data points added
        """
        existing_data = cache_manager.get_latest_data(key)

        # If no existing data, everything that's non-NaN in new_data is an addition
        timestamp = datetime.now()
        modifications: list[Change] = []
        additions: list[Change] = []

        if existing_data.empty:
            additions = self._create_changes_from_dataframe(
                new_data, timestamp, is_addition=True
            )
            return modifications, additions

        # Ensure indices and columns are aligned in terms of labels but don't fill missing values yet
        # We'll operate on intersections for modifications, and handle additions separately
        common_index = existing_data.index.intersection(new_data.index)
        common_cols = existing_data.columns.intersection(new_data.columns)

        # If there is an intersection, detect modifications there
        if len(common_index) > 0 and len(common_cols) > 0:
            ex_sub = existing_data.loc[common_index, common_cols]
            new_sub = new_data.loc[common_index, common_cols]

            # --- Numeric columns: use NumPy for fast absolute-diff-with-tolerance ---
            # Exclude boolean columns as they don't support arithmetic operations
            numeric_cols = [
                c
                for c in common_cols
                if pd.api.types.is_numeric_dtype(ex_sub[c].dtype)
                and pd.api.types.is_numeric_dtype(new_sub[c].dtype)
                and not pd.api.types.is_bool_dtype(ex_sub[c].dtype)
                and not pd.api.types.is_bool_dtype(new_sub[c].dtype)
            ]

            # process numeric columns with numpy (if any)
            if numeric_cols:
                a = ex_sub[numeric_cols].to_numpy(copy=False)
                b = new_sub[numeric_cols].to_numpy(copy=False)

                # Create masks for NaNs
                # np.isnan works for numeric arrays; if dtypes are object, we didn't include them
                a_na = np.isnan(a)
                b_na = np.isnan(b)

                one_nan_mask = a_na ^ b_na
                both_not_na = ~(a_na | b_na)

                if self.tolerance > 0.0:
                    # compute absolute differences only where both_not_na to avoid warnings
                    diff = np.empty_like(a, dtype=np.float64)
                    # safe subtraction where both_not_na
                    # fill diff with 0 where not both_not_na
                    diff[:, :] = 0.0
                    if np.any(both_not_na):
                        diff[both_not_na] = np.abs(a[both_not_na] - b[both_not_na])
                    mismatch_mask = one_nan_mask | (diff > self.tolerance)
                else:
                    mismatch_mask = one_nan_mask | (a != b)

                # extract coordinates of differences
                rows, cols = np.nonzero(mismatch_mask)
                # Map numeric column indices to labels
                for r, c_idx in zip(rows, cols):
                    row_label = common_index[r]
                    col_label = numeric_cols[c_idx]
                    old_v = ex_sub.loc[row_label, col_label]
                    new_v = new_sub.loc[row_label, col_label]
                    modifications.append(
                        Change((row_label, col_label), old_v, new_v, timestamp)
                    )

            # --- Non-numeric columns: vectorized pandas comparisons ---
            non_numeric_cols = [c for c in common_cols if c not in numeric_cols]
            if non_numeric_cols:
                left = ex_sub[non_numeric_cols]
                right = new_sub[non_numeric_cols]

                # Build boolean mask of differences handling NaNs
                left_na = left.isna()
                right_na = right.isna()

                # one is NaN and the other not
                one_nan = left_na ^ right_na

                # both not NaN and not equal
                both_not_na = ~(left_na | right_na)
                # Use != which is vectorized but may produce True for unequal; combine with both_not_na
                try:
                    neq = (left != right) & both_not_na
                except Exception:
                    # fallback: convert to string and compare (only if weird types)
                    neq = (left.astype(str) != right.astype(str)) & both_not_na

                diff_mask_df = one_nan | neq

                # stack to get (row, col) pairs that are True
                stacked = diff_mask_df.stack()
                changed = stacked[stacked].index  # MultiIndex of (row_label, col_label)

                # Use direct label-based lookups for simplicity and type safety
                for row_label, col_label in changed:
                    old_v = left.loc[row_label, col_label]
                    new_v = right.loc[row_label, col_label]
                    modifications.append(
                        Change((row_label, col_label), old_v, new_v, timestamp)
                    )

        # --- Additions: new-only rows and columns ---
        # New rows (entire row is new for rows not in existing_data)
        new_row_idx = new_data.index.difference(existing_data.index)
        if len(new_row_idx) > 0:
            series = new_data.loc[new_row_idx].stack()
            for coord, v in series.items():
                r, c = cast(tuple[Any, Any], coord)
                additions.append(Change((r, c), None, v, timestamp))

        # New columns (columns present in new_data but not in existing_data) for common rows
        new_cols = new_data.columns.difference(existing_data.columns)
        common_rows_for_new_cols = existing_data.index.intersection(new_data.index)
        if len(new_cols) > 0 and len(common_rows_for_new_cols) > 0:
            part = new_data.loc[common_rows_for_new_cols, new_cols].stack()
            for coord, v in part.items():
                r, c = cast(tuple[Any, Any], coord)
                additions.append(Change((r, c), None, v, timestamp))

        logger.debug(
            f"Change detection for {key}: {len(modifications)} modifications, {len(additions)} additions"
        )

        return modifications, additions

    def _create_changes_from_dataframe(
        self, df: pd.DataFrame, timestamp: datetime, is_addition: bool = True
    ) -> list[Change]:
        changes: list[Change] = []
        # iterate only non-NaN values
        for coord, v in df.stack().items():
            r, c = cast(tuple[Any, Any], coord)
            old_value = None if is_addition else v
            new_value = v
            changes.append(Change((r, c), old_value, new_value, timestamp))
        return changes

    # Keep helper methods if other parts of the codebase use them
    def _get_coordinates(self, df: pd.DataFrame) -> pd.MultiIndex:
        """
        Get all (row, column) coordinates from DataFrame using pd.MultiIndex.from_product.

        Args:
            df: DataFrame to extract coordinates from

        Returns:
            pd.MultiIndex
        """
        return pd.MultiIndex.from_product(
            [df.index, df.columns],
            names=list(df.index.names) + [df.columns.name or "column"],
        )

    def _get_value_at_coord(
        self, df: pd.DataFrame, row_idx: int | str, col_idx: int | str
    ) -> Any | None:
        """
        Safely get value at coordinate.

        Args:
            df: DataFrame
            row_idx: Row index
            col_idx: Column index

        Returns:
            Value at coordinate or None if not exists
        """
        try:
            if row_idx in df.index and col_idx in df.columns:
                return df.loc[row_idx, col_idx]
            return None
        except (KeyError, IndexError):
            return None

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """
        Compare two values handling NaN properly.

        Args:
            val1: First value
            val2: Second value

        Returns:
            True if values are equal
        """
        # Both NaN
        # Use Any-typed values so pandas.isna overloads match
        if pd.isna(val1) and pd.isna(val2):
            return True

        # One NaN, one not
        if pd.isna(val1) or pd.isna(val2):
            return False

        # Both not NaN - direct comparison
        try:
            return bool(val1 == val2)
        except Exception:
            # Fallback for complex types
            return bool(str(val1) == str(val2))
