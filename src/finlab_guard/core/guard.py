"""Main FinlabGuard class for managing finlab data cache."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from ..cache.manager import CacheManager
from ..cache.validator import DataValidator
from ..utils.exceptions import DataModifiedException, FinlabConnectionException

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
        }
        if config:
            self.config.update(config)

        # Set up logging: ensure we pass an int level. Config stores level names like 'INFO'
        log_level_name = str(self.config.get("log_level", "INFO"))
        log_level = getattr(logging, log_level_name, logging.INFO)
        logging.basicConfig(level=log_level)

        # Initialize components
        self.cache_manager = CacheManager(self.cache_dir, self.config)
        self.validator = DataValidator()

        # Time context for historical queries
        self.time_context: Optional[datetime] = None

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

    def get(self, key: str, force_download: bool = False) -> pd.DataFrame:
        """
        Get data with caching and change detection.

        Args:
            key: Dataset key (e.g., 'price:收盤價')
            force_download: Force download even if changes detected

        Returns:
            DataFrame with requested data

        Raises:
            DataModifiedException: When historical data has been modified
            FinlabConnectionException: When unable to connect to finlab
        """
        # Check if in time context mode (historical query)
        if self.time_context:
            logger.info(f"Loading historical data for {key} as of {self.time_context}")
            return self.cache_manager.load_data(key, self.time_context)

        # Get fresh data from finlab
        try:
            new_data = self._fetch_from_finlab(key)
        except Exception as e:
            raise FinlabConnectionException(
                f"Cannot fetch data from finlab: {e}"
            ) from e

        # Validate data format
        self.validator.validate_dataframe_format(new_data)

        # Check if cache exists
        if not self.cache_manager.exists(key):
            # First time: save directly
            timestamp = self.generate_unique_timestamp(key)
            self.cache_manager.save_data(key, new_data, timestamp)
            logger.info(f"First time caching data for {key}")
            return new_data

        # Detect changes
        modifications, additions = self.validator.detect_changes_detailed(
            key, new_data, self.cache_manager
        )

        if modifications:
            # Historical data was modified
            if force_download:
                timestamp = self.generate_unique_timestamp(key)
                # Use incremental save to only store changes
                self.cache_manager.save_incremental_changes(
                    key, modifications, additions, timestamp, new_data
                )
                logger.warning(
                    f"Historical data modified for {key}: {len(modifications)} cells changed"
                )
                return new_data
            else:
                raise DataModifiedException(
                    f"Historical data modified for {key}", modifications
                )
        else:
            # Normal case: save only new/updated data incrementally
            timestamp = self.generate_unique_timestamp(key)
            if additions:
                # Only save the additions, no need to save the entire dataset
                self.cache_manager.save_incremental_changes(
                    key, [], additions, timestamp, new_data
                )
                logger.info(
                    f"Updated cache for {key}: {len(additions)} new data points"
                )
            else:
                # Even if no data changes, check and save dtype changes
                self.cache_manager._save_dtype_mapping(key, new_data, timestamp)
                logger.info(f"No new data to cache for {key}")
            return new_data

    def _fetch_from_finlab(self, key: str) -> pd.DataFrame:
        """
        Fetch data from finlab using original function.

        Args:
            key: Dataset key

        Returns:
            DataFrame from finlab
        """
        try:
            import finlab.data

            # If a patched original exists, call that; otherwise call get.
            if hasattr(finlab.data, "_original_get"):
                result = finlab.data._original_get(key)
            else:
                result = finlab.data.get(key)

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

    def install_patch(self) -> None:
        """Install monkey patch for finlab.data.get."""
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

        # Save original function and install patch
        finlab.data._original_get = finlab.data.get
        _global_guard_instance = self

        # Install patch
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
