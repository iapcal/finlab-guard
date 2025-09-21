"""Data validation and change detection for finlab-guard."""

import logging
from datetime import datetime
from typing import Any, Optional, Union

import pandas as pd

from ..utils.exceptions import (
    Change,
    InvalidDataTypeException,
    UnsupportedDataFormatException,
)
from .manager import CacheManager

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates DataFrame format and detects data changes."""

    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize DataValidator with specified tolerance for float comparisons.

        Args:
            tolerance: Absolute tolerance for float comparisons. Defaults to 1e-12.
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
        Detect detailed changes between cached and new data.

        Args:
            key: Dataset key
            new_data: New DataFrame to compare
            cache_manager: CacheManager instance

        Returns:
            Tuple of (modifications, additions) where:
            - modifications: List of changes to existing historical data
            - additions: List of new data points added
        """
        # Load existing data
        existing_data = cache_manager.get_latest_data(key)

        if existing_data.empty:
            # No existing data, everything is new
            additions = self._create_changes_from_dataframe(
                new_data, datetime.now(), is_addition=True
            )
            return [], additions

        # Handle coordinates - only consider actually existing (non-NaN) coordinates
        new_stacked = new_data.stack()
        existing_stacked = existing_data.stack()

        # Common coordinates are those that exist in both stacked DataFrames (non-NaN in both)
        common_coords = existing_stacked.index.intersection(new_stacked.index)

        # New coordinates are those that exist in new but not in existing
        new_only_coords = new_stacked.index.difference(existing_stacked.index)

        modifications = []
        additions = []
        timestamp = datetime.now()

        # Vectorized change detection
        if len(common_coords) > 0:
            # Reindex both stacked DataFrames to common coordinates
            old_stacked_common = existing_stacked.reindex(common_coords)
            new_stacked_common = new_stacked.reindex(common_coords)

            # Find differences using vectorized comparison with tolerance
            if self.tolerance > 0.0:
                # Use tolerance-based comparison for numeric data
                numeric_mask = pd.api.types.is_numeric_dtype(
                    old_stacked_common.dtype
                ) and pd.api.types.is_numeric_dtype(new_stacked_common.dtype)

                if numeric_mask:
                    # For numeric data, use absolute difference comparison
                    abs_diff = (old_stacked_common - new_stacked_common).abs()
                    significant_diff = abs_diff > self.tolerance

                    # Handle NaN values properly
                    both_nan = old_stacked_common.isna() & new_stacked_common.isna()
                    one_nan = old_stacked_common.isna() ^ new_stacked_common.isna()

                    nan_safe_diff = (significant_diff | one_nan) & ~both_nan
                else:
                    # For non-numeric data, use exact comparison
                    nan_safe_diff = (old_stacked_common != new_stacked_common) & ~(
                        old_stacked_common.isna() & new_stacked_common.isna()
                    )
            else:
                # Exact comparison when tolerance is 0
                nan_safe_diff = (old_stacked_common != new_stacked_common) & ~(
                    old_stacked_common.isna() & new_stacked_common.isna()
                )

            if nan_safe_diff.any():
                changed_coords = common_coords[nan_safe_diff]

                # Batch create Change objects
                for coord in changed_coords:
                    old_val = old_stacked_common.loc[coord]
                    new_val = new_stacked_common.loc[coord]
                    change = Change(coord, old_val, new_val, timestamp)
                    modifications.append(change)

        # Handle additions using index difference
        if len(new_only_coords) > 0:
            for coord in new_only_coords:
                new_value = new_stacked.loc[coord]
                change = Change(coord, None, new_value, timestamp)
                additions.append(change)

        logger.debug(
            f"Change detection for {key}: {len(modifications)} modifications, {len(additions)} additions"
        )
        return modifications, additions

    def _get_coordinates(self, df: pd.DataFrame) -> list[tuple]:
        """
        Get all (row, column) coordinates from DataFrame.

        Args:
            df: DataFrame to extract coordinates from

        Returns:
            List of (row_index, column_name) tuples
        """
        # Use stack() to get non-NaN coordinates
        stacked = df.stack()
        coordinates = stacked.index.tolist()
        return coordinates

    def _get_value_at_coord(
        self, df: pd.DataFrame, row_idx: Union[int, str], col_idx: Union[int, str]
    ) -> Optional[Any]:
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

    def _create_changes_from_dataframe(
        self, df: pd.DataFrame, timestamp: datetime, is_addition: bool = True
    ) -> list[Change]:
        """
        Create Change objects from all values in DataFrame.

        Args:
            df: DataFrame to process
            timestamp: Timestamp for changes
            is_addition: Whether these are additions (vs modifications)

        Returns:
            List of Change objects
        """
        changes = []
        for row_idx in df.index:
            for col_name in df.columns:
                value = df.loc[row_idx, col_name]
                if pd.notna(value):  # Only include non-NaN values
                    coord = (row_idx, col_name)
                    old_value = None if is_addition else value
                    new_value = value
                    change = Change(coord, old_value, new_value, timestamp)
                    changes.append(change)
        return changes
