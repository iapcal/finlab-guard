"""Main FinlabGuard class for managing finlab data cache."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from ..cache.manager import CacheManager
from ..utils.exceptions import (
    DataModifiedException,
    FinlabConnectionException,
    InvalidDataTypeException,
    UnsupportedDataFormatException,
)

# Global instance to ensure uniqueness
_global_guard_instance: Optional["FinlabGuard"] = None

logger = logging.getLogger(__name__)


class FinlabGuard:
    """
    Main class for managing finlab data cache with version control.

    Provides automatic caching, change detection, and time-based queries
    for finlab data to ensure reproducible backtesting results.
    """

    def __init__(
        self,
        cache_dir: str = "~/.finlab_guard",
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize FinlabGuard.

        Args:
            cache_dir: Directory to store cache files
            config: Configuration dictionary
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        self.config = {
            "compression": "snappy",
            "progress_bar": True,
            "log_level": "INFO",
            "force_hash_bypass": False,  # Force ignore hash optimization, always reconstruct for comparison
        }
        if config:
            self.config.update(config)

        # Set up logging: ensure we pass an int level. Config stores level names like 'INFO'
        log_level_name = str(
            self.config.get("log_level", "DEBUG")
        )  # Change default to DEBUG for troubleshooting
        log_level = getattr(logging, log_level_name, logging.DEBUG)
        logging.basicConfig(
            level=log_level,
            format="%(levelname)s %(name)s:%(funcName)s:%(lineno)d %(message)s",
        )

        # Initialize components
        self.cache_manager = CacheManager(self.cache_dir, self.config)

        # Time context for historical queries
        self.time_context: Optional[datetime] = None

        # Global setting for allowing historical changes
        self._allow_historical_changes: bool = True

        logger.info(f"FinlabGuard initialized with cache_dir: {self.cache_dir}")

    def set_time_context(
        self, as_of_time: Optional[Union[datetime, str]] = None
    ) -> None:
        """
        Set global time context for historical data queries.

        Args:
            as_of_time: Target datetime for historical queries. None to clear.
        """
        if isinstance(as_of_time, str):
            as_of_time = pd.to_datetime(as_of_time)

        self.time_context = as_of_time
        if as_of_time:
            logger.info(f"Time context set to: {as_of_time}")
        else:
            logger.info("Time context cleared")

    def clear_time_context(self) -> None:
        """Clear the time context to return to normal mode."""
        self.set_time_context(None)

    def get_time_context(self) -> Optional[datetime]:
        """Get current time context."""
        return self.time_context

    def _now(self) -> datetime:
        return datetime.now()

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

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame to handle problematic types."""
        result = df.copy()

        # Categorical columns are now properly supported in the cache manager
        # No preprocessing needed for categorical dtypes

        # Note: DatetimeIndex preservation is important for test compatibility
        # Only convert if we encounter specific serialization issues

        return result

    def generate_unique_timestamp(self, key: str) -> datetime:
        """
        Generate unique timestamp to avoid conflicts.

        Args:
            key: Dataset key

        Returns:
            Unique timestamp
        """
        now = self._now()

        # Check if this timestamp already exists for this key
        existing_data = self.cache_manager.load_raw_data(key)
        if existing_data is not None and not existing_data.empty:
            # Find the latest timestamp
            latest_time = existing_data["save_time"].max()
            if pd.notna(latest_time) and now <= latest_time:
                # Add 1 second to ensure uniqueness
                now = latest_time + pd.Timedelta(seconds=1)

        return now

    def get(
        self,
        dataset: str,
        save_to_storage: bool = True,
        force_download: bool = False,
        allow_historical_changes: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Get data with caching and change detection.

        Args:
            dataset: Dataset key (e.g., 'price:收盤價')
            save_to_storage: Compatibility parameter for original finlab API (bypassed to original)
            force_download: Force download parameter for original finlab API (bypassed to original)
            allow_historical_changes: Allow historical data modifications (overrides global setting if specified)

        Returns:
            DataFrame with requested data

        Raises:
            DataModifiedException: When historical data has been modified
            FinlabConnectionException: When unable to connect to finlab
        """
        # Use dataset as key for internal consistency
        key = dataset

        # Determine effective allow_historical_changes setting
        # Method parameter takes precedence over global setting
        effective_allow_changes = (
            allow_historical_changes
            if allow_historical_changes is not None
            else self._allow_historical_changes
        )

        # Check if in time context mode (historical query)
        if self.time_context:
            logger.info(f"Loading historical data for {key} as of {self.time_context}")
            return self.cache_manager.load_data(key, self.time_context)

        # Get fresh data from finlab (bypass parameters to original API)
        try:
            new_data = self._fetch_from_finlab(key, save_to_storage, force_download)
        except Exception as e:
            raise FinlabConnectionException(
                f"Cannot fetch data from finlab: {e}"
            ) from e

        # Validate data format
        self.validate_dataframe_format(new_data)

        # Preprocess data to handle problematic types
        new_data = self._preprocess_dataframe(new_data)

        # Check if cache exists
        if not self.cache_manager.exists(key):
            # First time: save directly
            timestamp = self.generate_unique_timestamp(key)
            self.cache_manager.save_data(key, new_data, timestamp)
            logger.info(f"First time caching data for {key}")
            return new_data

        # **Hash optimization: Quick comparison before detailed diff**
        # Skip hash optimization if force_hash_bypass is enabled
        if not self.config.get("force_hash_bypass", False):
            new_hash = self.cache_manager._compute_dataframe_hash(new_data)
            cached_hash = self.cache_manager._get_data_hash(key)

            if cached_hash and new_hash == cached_hash:
                # Hash match → data unchanged → return new data directly
                logger.info(
                    f"Data unchanged for {key} (hash match), returning new data"
                )
                return new_data
        else:
            logger.debug(
                f"Hash optimization bypassed for {key} (force_hash_bypass enabled)"
            )
            # Still compute hash for storage, but don't use it for comparison
            new_hash = self.cache_manager._compute_dataframe_hash(new_data)

        # Save data and get changes (includes dtype, hash, and all metadata)
        timestamp = self.generate_unique_timestamp(key)
        changes = self.cache_manager.save_data(key, new_data, timestamp)

        # Check if changes are allowed (only if we have change details)
        if changes is not None and not effective_allow_changes:
            # Check for prohibited change types
            has_cell_modifications = not changes.cell_changes.empty
            has_row_deletions = not changes.row_deletions.empty
            has_column_deletions = not changes.column_deletions.empty
            has_dtype_changes = changes.dtype_changed

            if (
                has_cell_modifications
                or has_row_deletions
                or has_column_deletions
                or has_dtype_changes
            ):
                # Build detailed error message
                error_details = []
                if has_cell_modifications:
                    error_details.append(
                        f"{len(changes.cell_changes)} cell modifications"
                    )
                if has_row_deletions:
                    error_details.append(f"{len(changes.row_deletions)} row deletions")
                if has_column_deletions:
                    error_details.append(
                        f"{len(changes.column_deletions)} column deletions"
                    )
                if has_dtype_changes:
                    error_details.append("dtype changes")

                raise DataModifiedException(
                    f"Historical data modified for {key}: {', '.join(error_details)}",
                    changes,
                )

        # Log the changes
        if changes is not None:
            total_changes = (
                len(changes.cell_changes)
                + len(changes.row_additions)
                + len(changes.row_deletions)
                + len(changes.column_additions)
                + len(changes.column_deletions)
            )

            # Include dtype changes in total count
            if changes.dtype_changed:
                total_changes += 1

            if total_changes > 0:
                change_summary = []
                if not changes.cell_changes.empty:
                    change_summary.append(f"{len(changes.cell_changes)} cell changes")
                if not changes.row_additions.empty:
                    change_summary.append(f"{len(changes.row_additions)} row additions")
                if not changes.row_deletions.empty:
                    change_summary.append(f"{len(changes.row_deletions)} row deletions")
                if not changes.column_additions.empty:
                    change_summary.append(
                        f"{len(changes.column_additions)} column additions"
                    )
                if not changes.column_deletions.empty:
                    change_summary.append(
                        f"{len(changes.column_deletions)} column deletions"
                    )
                if changes.dtype_changed:
                    change_summary.append("dtype changes")

                logger.info(f"Updated cache for {key}: {', '.join(change_summary)}")
            else:
                logger.info(f"No changes detected for {key}")
        else:
            logger.info(f"First time caching data for {key}")

        return new_data

    def _fetch_from_finlab(
        self, key: str, save_to_storage: bool = True, force_download: bool = False
    ) -> pd.DataFrame:
        """
        Fetch data from finlab using original function.

        Args:
            key: Dataset key
            save_to_storage: Parameter passed to original finlab API
            force_download: Parameter passed to original finlab API

        Returns:
            DataFrame from finlab
        """
        try:
            import finlab.data

            # If a patched original exists, call that; otherwise call get.
            # Pass through the original API parameters
            if hasattr(finlab.data, "_original_get"):
                result = finlab.data._original_get(
                    key, save_to_storage=save_to_storage, force_download=force_download
                )
            else:
                result = finlab.data.get(
                    key, save_to_storage=save_to_storage, force_download=force_download
                )

            # Ensure we return a DataFrame (finlab API should return one)
            if isinstance(result, pd.DataFrame):
                return result

            # Try to coerce to DataFrame if possible
            try:
                return pd.DataFrame(result)
            except Exception as e:
                raise FinlabConnectionException(
                    f"finlab returned unexpected type: {type(result)}"
                ) from e
        except ImportError as e:
            raise FinlabConnectionException("finlab package not found") from e

    def install_patch(self, allow_historical_changes: bool = True) -> None:
        """
        Install monkey patch for finlab.data.get.

        Args:
            allow_historical_changes: Global setting to allow historical data modifications
        """
        global _global_guard_instance

        try:
            import finlab.data
        except ImportError as e:
            raise ImportError(
                "finlab package not found. Please install finlab first."
            ) from e

        # Check if patch is already installed
        if _global_guard_instance is not None or hasattr(finlab.data, "_original_get"):
            raise RuntimeError(
                "finlab-guard already installed. Use remove_patch() first."
            )

        # Set global setting for historical changes
        self._allow_historical_changes = allow_historical_changes

        # Save original function and install patch
        finlab.data._original_get = finlab.data.get
        _global_guard_instance = self

        # Install patch - forward all arguments to maintain API compatibility
        def patched_get(*args: Any, **kwargs: Any) -> Any:
            return _global_guard_instance.get(*args, **kwargs)

        finlab.data.get = patched_get
        logger.info("Monkey patch installed successfully")

    @classmethod
    def remove_patch(cls) -> None:
        """Class method to remove monkey patch for finlab.data.get."""
        global _global_guard_instance

        # First check if finlab.data is already imported
        import sys

        if "finlab.data" not in sys.modules:
            logger.warning("No monkey patch found to remove (finlab.data not imported)")
            _global_guard_instance = None
            return

        try:
            import finlab.data

            if hasattr(finlab.data, "_original_get"):
                finlab.data.get = finlab.data._original_get
                delattr(finlab.data, "_original_get")
                _global_guard_instance = None
                logger.info("Monkey patch removed successfully")
            else:
                logger.warning("No monkey patch found to remove")
                _global_guard_instance = None
        except ImportError:
            logger.warning("finlab package not found")
        except Exception as e:
            # Handle finlab initialization errors gracefully
            logger.warning(f"Failed to access finlab during patch removal: {e}")
            _global_guard_instance = None

    def clear_cache(self, key: Optional[str] = None) -> None:
        """
        Clear cache data.

        Args:
            key: Specific dataset key to clear. If None, clear all cache.
        """
        if key:
            self.cache_manager.clear_key(key)
            logger.info(f"Cleared cache for {key}")
        else:
            self.cache_manager.clear_all()
            logger.info("Cleared all cache")

    def get_change_history(self, key: str) -> pd.DataFrame:
        """
        Get change history for a dataset.

        Args:
            key: Dataset key

        Returns:
            DataFrame containing change history
        """
        return self.cache_manager.get_change_history(key)

    def get_storage_info(self, key: Optional[str] = None) -> dict[str, Any]:
        """
        Get storage information.

        Args:
            key: Specific dataset key. If None, get info for all datasets.

        Returns:
            Dictionary with storage information
        """
        return self.cache_manager.get_storage_info(key)

    def close(self) -> None:
        """Close the underlying cache manager and its connections."""
        if hasattr(self, "cache_manager") and self.cache_manager:
            self.cache_manager.close()

    def __enter__(self) -> "FinlabGuard":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure connections are closed."""
        # Standard context manager parameters are not used
        del exc_type, exc_val, exc_tb
        self.close()
