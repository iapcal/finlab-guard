"""Cache management for finlab-guard."""

import json
import logging
import tempfile
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages cache storage and retrieval for finlab data."""

    def __init__(self, cache_dir: Path, config: dict[str, Any]):
        """
        Initialize CacheManager.

        Args:
            cache_dir: Directory to store cache files
            config: Configuration dictionary
        """
        self.cache_dir = cache_dir
        self.config = config
        self.compression = config.get("compression", "snappy")

    def _get_cache_path(self, key: str) -> Path:
        """
        Get cache file path for a dataset key.

        Args:
            key: Dataset key

        Returns:
            Path to cache file
        """
        # Replace special characters for filesystem safety
        safe_key = key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.parquet"

    def _get_dtype_path(self, key: str) -> Path:
        """
        Get dtype mapping file path for a dataset key.

        Args:
            key: Dataset key

        Returns:
            Path to dtype mapping file
        """
        # Replace special characters for filesystem safety
        safe_key = key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}_dtypes.json"

    def _save_dtype_mapping(
        self, key: str, df: pd.DataFrame, timestamp: Optional[datetime] = None
    ) -> None:
        """
        Save dtype mapping for a DataFrame with versioning support.
        Only creates new entry when dtypes actually change.

        Args:
            key: Dataset key
            df: DataFrame to save dtype mapping for
            timestamp: Timestamp for this dtype entry (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        # Prepare current dtype signature
        current_signature = {
            "dtypes": {str(col): str(df[col].dtype) for col in df.columns},
            "index_dtype": str(df.index.dtype),
            "columns_dtype": str(df.columns.dtype)
            if hasattr(df.columns, "dtype")
            else "object",
            "index_name": df.index.name,
            "columns_name": df.columns.name,
            "columns_order": [str(col) for col in df.columns],
            "index_order": [str(idx) for idx in df.index],
            # pandas Index may expose freq or freqstr depending on type/version.
            # Use getattr to safely obtain a string representation when available.
            "index_freq": (
                getattr(df.index, "freqstr", None)
                if getattr(df.index, "freq", None) is not None
                else None
            ),
        }

        # Load existing mapping
        existing_mapping = self._load_dtype_mapping(key)

        # Check if we need a new entry
        if not self._needs_new_dtype_entry(current_signature, existing_mapping):
            logger.debug(f"No dtype changes detected for {key}, skipping save")
            return

        # Create new entry
        new_entry = {"timestamp": timestamp.isoformat(), **current_signature}

        # Initialize or update dtype mapping structure
        if existing_mapping:
            # Append to existing structure
            dtype_mapping = existing_mapping
            dtype_mapping["dtype_history"].append(new_entry)
            dtype_mapping["last_updated"] = new_entry["timestamp"]
        else:
            # Create new structure
            dtype_mapping = {
                "schema_version": "1.0",
                "last_updated": new_entry["timestamp"],
                "dtype_history": [new_entry],
            }

        dtype_path = self._get_dtype_path(key)
        with open(dtype_path, "w") as f:
            json.dump(dtype_mapping, f, indent=2)

        logger.debug(f"Saved new dtype entry for {key} at {new_entry['timestamp']}")

    def _needs_new_dtype_entry(
        self,
        current_signature: dict[str, Any],
        existing_mapping: Optional[dict[str, Any]],
    ) -> bool:
        """
        Check if a new dtype entry is needed based on current signature.

        Args:
            current_signature: Current DataFrame dtype signature
            existing_mapping: Existing dtype mapping (may be None)

        Returns:
            True if a new dtype entry should be created
        """
        if not existing_mapping:
            # First time save
            return True

        # Ensure we have the expected structure
        if "dtype_history" not in existing_mapping:
            return True

        dtype_history = existing_mapping.get("dtype_history", [])
        if not dtype_history:
            # Empty history
            return True

        # Get latest entry
        latest_entry = dtype_history[-1]

        # Compare each component
        changes_detected = (
            latest_entry.get("dtypes") != current_signature["dtypes"]
            or latest_entry.get("index_dtype") != current_signature["index_dtype"]
            or latest_entry.get("columns_dtype") != current_signature["columns_dtype"]
            or latest_entry.get("index_name") != current_signature["index_name"]
            or latest_entry.get("columns_name") != current_signature["columns_name"]
            or latest_entry.get("columns_order") != current_signature["columns_order"]
            or set(latest_entry.get("index_order", []))
            != set(current_signature["index_order"])
            or latest_entry.get("index_freq") != current_signature["index_freq"]
        )

        if changes_detected:
            logger.debug("Dtype changes detected - need new entry")
            # Log specific changes for debugging
            if latest_entry.get("dtypes") != current_signature["dtypes"]:
                logger.debug(
                    f"Column dtypes changed: {latest_entry.get('dtypes')} -> {current_signature['dtypes']}"
                )
            if latest_entry.get("columns_order") != current_signature["columns_order"]:
                logger.debug(
                    f"Column order changed: {latest_entry.get('columns_order')} -> {current_signature['columns_order']}"
                )

        return changes_detected

    def _load_dtype_mapping(self, key: str) -> Optional[dict[str, Any]]:
        """
        Load dtype mapping for a dataset.

        Args:
            key: Dataset key

        Returns:
            Dtype mapping dictionary or None if not found
        """
        dtype_path = self._get_dtype_path(key)
        if not dtype_path.exists():
            return None

        try:
            with open(dtype_path) as f:
                loaded_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dtype mapping for {key}: {e}")
            return None

        # Ensure we always return a mapping or None (narrow Any to dict[str, Any])
        if isinstance(loaded_data, dict):
            # mypy cannot always infer nested types from json; keep Any for values
            return loaded_data
        return None

    def _get_dtype_mapping_at_time(
        self, key: str, target_time: Optional[datetime]
    ) -> Optional[dict[str, Any]]:
        """
        Get dtype mapping for a specific time point.

        Args:
            key: Dataset key
            target_time: Target time point (None for latest)

        Returns:
            Dtype mapping entry for the specified time or None
        """
        full_mapping = self._load_dtype_mapping(key)
        if not full_mapping:
            return None

        # Ensure we have the expected structure
        if "dtype_history" not in full_mapping:
            return None

        dtype_history = full_mapping.get("dtype_history", [])
        if not dtype_history:
            return None

        # If no target time specified, return latest
        if target_time is None:
            latest_entry = dtype_history[-1]
            if isinstance(latest_entry, dict):
                return latest_entry
            return None

        # Find the most recent entry at or before target_time
        target_entry: Optional[dict[str, Any]] = None
        for entry in dtype_history:
            # entry may be Any from json; guard access
            if not isinstance(entry, dict) or "timestamp" not in entry:
                continue
            entry_time = pd.to_datetime(entry["timestamp"])
            if entry_time <= target_time:
                target_entry = entry
            # Don't break - continue to find the latest entry within time range

        # If no entry found before target_time, return the first entry
        # (this handles the case where target_time is before first entry)
        if target_entry is None and dtype_history:
            first_entry = dtype_history[0]
            if isinstance(first_entry, dict):
                target_entry = first_entry

        return target_entry

    def _apply_dtypes_to_result(
        self, result: pd.DataFrame, key: str, target_time: Optional[datetime] = None
    ) -> None:
        """
        Apply saved dtypes to reconstructed DataFrame.

        Args:
            result: DataFrame to apply dtypes to (modified in place)
            key: Dataset key to load dtype mapping for
            target_time: Target time point for dtype lookup
        """
        dtype_mapping = self._get_dtype_mapping_at_time(key, target_time)
        if not dtype_mapping or "dtypes" not in dtype_mapping:
            return

        dtypes = dtype_mapping["dtypes"]

        # Apply saved dtypes to columns
        for col, dtype_str in dtypes.items():
            if col in result.columns and dtype_str and dtype_str != "None":
                try:
                    # Since values are stored as strings, convert them back
                    if "int" in dtype_str:
                        # Convert string values to numeric first, then to target int type
                        result[col] = pd.to_numeric(
                            result[col], errors="coerce"
                        ).astype(dtype_str)
                    elif "float" in dtype_str:
                        result[col] = pd.to_numeric(
                            result[col], errors="coerce"
                        ).astype(dtype_str)
                    elif "bool" in dtype_str:
                        # Convert string boolean values back to bool
                        result[col] = (
                            result[col]
                            .map({"True": True, "False": False})
                            .astype("bool")
                        )
                    elif dtype_str == "object":
                        # Keep as string/object
                        result[col] = result[col].astype("object")
                    else:
                        # Try direct conversion
                        try:
                            target_dtype = pd.api.types.pandas_dtype(dtype_str)
                            result[col] = result[col].astype(target_dtype)
                        except (ValueError, TypeError):
                            pass
                except (ValueError, TypeError, Exception):
                    # Fallback: try to convert from string
                    try:
                        if "int" in dtype_str:
                            result[col] = pd.to_numeric(
                                result[col], errors="coerce"
                            ).astype("int64")
                        elif "float" in dtype_str:
                            result[col] = pd.to_numeric(
                                result[col], errors="coerce"
                            ).astype("float64")
                        elif "bool" in dtype_str:
                            result[col] = (
                                result[col]
                                .map({"True": True, "False": False})
                                .fillna(False)
                                .astype("bool")
                            )
                    except (ValueError, TypeError):
                        pass

        # Apply saved dtype to index
        index_dtype = dtype_mapping.get("index_dtype")
        if index_dtype and index_dtype != "None":
            try:
                result.index = result.index.astype(index_dtype)
            except (ValueError, TypeError, Exception):
                # Fallback: keep original index dtype
                logger.debug(
                    f"Failed to convert index to dtype {index_dtype}, keeping original"
                )
                pass

        # Apply saved frequency to index (for DatetimeIndex)
        index_freq = dtype_mapping.get("index_freq")
        if (
            index_freq
            and index_freq != "None"
            and isinstance(result.index, pd.DatetimeIndex)
        ):
            try:
                # Try to set the frequency using pandas' to_offset, constructing a new
                # DatetimeIndex with the same values but with the desired freq.
                try:
                    from pandas.tseries.frequencies import to_offset

                    offset = to_offset(index_freq)
                    result.index = pd.DatetimeIndex(result.index.values, freq=offset)
                except Exception:
                    # Fallback: try using asfreq on a temporary Series
                    try:
                        tmp = pd.Series([None] * len(result), index=result.index)
                        tmp = tmp.asfreq(index_freq)
                        result.index = tmp.index
                    except Exception:
                        # If we still can't set freq, continue without raising
                        pass
            except (ValueError, TypeError, Exception) as e:
                # Fallback: keep original index frequency
                logger.debug(
                    f"Failed to set index frequency to {index_freq}: {e}, keeping original"
                )
                pass

        # Apply saved dtype to columns (the columns object itself)
        columns_dtype = dtype_mapping.get("columns_dtype")
        if columns_dtype and columns_dtype != "None":
            try:
                result.columns = result.columns.astype(columns_dtype)
            except (ValueError, TypeError, Exception):
                # Fallback: keep original columns dtype
                logger.debug(
                    f"Failed to convert columns to dtype {columns_dtype}, keeping original"
                )
                pass

    def exists(self, key: str) -> bool:
        """
        Check if cache exists for a dataset.

        Args:
            key: Dataset key

        Returns:
            True if cache exists
        """
        return self._get_cache_path(key).exists()

    def save_data(self, key: str, data: pd.DataFrame, timestamp: datetime) -> None:
        """
        Save DataFrame to cache with timestamp.

        Args:
            key: Dataset key
            data: DataFrame to save
            timestamp: Save timestamp
        """
        if data.empty:
            logger.warning(f"Attempting to save empty DataFrame for {key}")
            return

        # Save dtype mapping separately
        self._save_dtype_mapping(key, data, timestamp)

        # Process and stack the data (without dtype info)
        stacked_data = self._stack_dataframe(data, timestamp)

        # Get existing data if it exists
        existing_data = self.load_raw_data(key)
        if existing_data is not None and not existing_data.empty:
            # Append to existing data
            combined_data = pd.concat([existing_data, stacked_data], ignore_index=True)
        else:
            combined_data = stacked_data

        # Save with atomic write
        self._atomic_write_parquet(key, combined_data)
        logger.debug(f"Saved {len(stacked_data)} records for {key}")

    def save_incremental_changes(
        self,
        key: str,
        modifications: list,
        additions: list,
        timestamp: datetime,
        original_df: pd.DataFrame,
    ) -> None:
        """
        Save only the incremental changes (modifications + additions) instead of full dataset.

        Args:
            key: Dataset key
            modifications: List of Change objects for modifications
            additions: List of Change objects for additions
            timestamp: Save timestamp
            original_df: Original DataFrame to get index/columns names
        """
        if not modifications and not additions:
            logger.debug(f"No changes to save for {key}")
            return

        # Update dtype mapping if there are new columns
        self._save_dtype_mapping(key, original_df, timestamp)

        # Get the correct column names from original DataFrame
        original_index_name = (
            original_df.index.name
            if original_df.index.name is not None
            else "_finlab_index_none"
        )
        original_columns_name = (
            original_df.columns.name
            if original_df.columns.name is not None
            else "_finlab_columns_none"
        )

        # Convert changes to stacked DataFrame format
        incremental_data = []

        # Add modifications
        for change in modifications:
            row_idx, col_idx = change.coord
            incremental_data.append(
                {
                    original_index_name: row_idx,
                    original_columns_name: col_idx,
                    "value": str(change.new_value),  # Convert to string for consistency
                    "save_time": timestamp,
                }
            )

        # Add additions
        for change in additions:
            row_idx, col_idx = change.coord
            incremental_data.append(
                {
                    original_index_name: row_idx,
                    original_columns_name: col_idx,
                    "value": str(change.new_value),  # Convert to string for consistency
                    "save_time": timestamp,
                }
            )

        if not incremental_data:
            return

        # Create DataFrame from incremental changes
        incremental_df = pd.DataFrame(incremental_data)

        # Get existing data if it exists
        existing_data = self.load_raw_data(key)
        if existing_data is not None and not existing_data.empty:
            # Append to existing data
            combined_data = pd.concat(
                [existing_data, incremental_df], ignore_index=True
            )
        else:
            combined_data = incremental_df

        # Save with atomic write
        self._atomic_write_parquet(key, combined_data)
        logger.debug(f"Saved {len(incremental_data)} incremental changes for {key}")

    def load_data(
        self, key: str, as_of_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load data from cache, optionally at a specific time.

        Args:
            key: Dataset key
            as_of_time: Load data as of this time. None for latest.

        Returns:
            DataFrame with requested data
        """
        raw_data = self.load_raw_data(key)
        if raw_data is None or raw_data.empty:
            logger.warning(f"No cache data found for {key}")
            return pd.DataFrame()

        return self._reconstruct_dataframe(raw_data, as_of_time, key)

    def load_raw_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Load raw stacked data from cache.

        Args:
            key: Dataset key

        Returns:
            Raw stacked DataFrame or None if not found
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.error(f"Failed to load cache for {key}: {e}")
            return None

    def get_latest_data(self, key: str) -> pd.DataFrame:
        """
        Get the latest version of data.

        Args:
            key: Dataset key

        Returns:
            Latest DataFrame
        """
        return self.load_data(key, as_of_time=None)

    def clear_key(self, key: str) -> None:
        """
        Clear cache for a specific key.

        Args:
            key: Dataset key to clear
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def clear_all(self) -> None:
        """Clear all cache files."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()

    def get_change_history(self, key: str) -> pd.DataFrame:
        """
        Get change history for a dataset.

        Args:
            key: Dataset key

        Returns:
            DataFrame with change history
        """
        raw_data = self.load_raw_data(key)
        if raw_data is None or raw_data.empty:
            return pd.DataFrame()

        # Group by coordinates and show all timestamps
        history = (
            raw_data.groupby([raw_data.columns[0], raw_data.columns[1]])
            .agg({"value": "count", "save_time": ["min", "max"]})
            .reset_index()
        )

        return history

    def get_storage_info(self, key: Optional[str] = None) -> dict[str, Any]:
        """
        Get storage information.

        Args:
            key: Specific dataset key or None for all

        Returns:
            Storage information dictionary
        """
        info: dict[str, Any] = {}

        if key:
            # Info for specific key
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                stat = cache_path.stat()
                raw_data = self.load_raw_data(key)
                info[key] = {
                    "file_size": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime),
                    "record_count": len(raw_data) if raw_data is not None else 0,
                }
        else:
            # Info for all keys
            total_size = 0
            for cache_file in self.cache_dir.glob("*.parquet"):
                stat = cache_file.stat()
                total_size += stat.st_size
                key_name = (
                    cache_file.stem
                )  # Use filename as-is since conversion is not reversible
                info[key_name] = {
                    "file_size": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime),
                }
            # total_size is an int; store explicitly as int to satisfy type checker
            info["total_size"] = int(total_size)

        return info

    def _stack_dataframe(self, df: pd.DataFrame, timestamp: datetime) -> pd.DataFrame:
        """
        Stack DataFrame and prepare for storage.

        Args:
            df: Original DataFrame
            timestamp: Save timestamp

        Returns:
            Stacked DataFrame ready for storage
        """
        # Handle None names with special identifiers
        original_index_name = (
            df.index.name if df.index.name is not None else "_finlab_index_none"
        )
        original_columns_name = (
            df.columns.name if df.columns.name is not None else "_finlab_columns_none"
        )

        # Stack the DataFrame
        stacked = df.stack()
        stacked_df = stacked.reset_index()
        stacked_df.columns = [original_index_name, original_columns_name, "value"]

        # Convert values to string for consistent parquet storage
        stacked_df["value"] = stacked_df["value"].astype(str)

        # Add timestamp
        stacked_df["save_time"] = timestamp

        return stacked_df

    def _reconstruct_dataframe(
        self, stacked_data: pd.DataFrame, target_time: Optional[datetime], key: str
    ) -> pd.DataFrame:
        """
        Reconstruct original DataFrame from stacked data.

        Args:
            stacked_data: Stacked data from cache
            target_time: Target time for reconstruction. None for latest.

        Returns:
            Reconstructed DataFrame
        """
        if stacked_data.empty:
            return pd.DataFrame()

        # Filter by time if specified
        if target_time:
            filtered = stacked_data[stacked_data["save_time"] <= target_time]
        else:
            filtered = stacked_data

        if filtered.empty:
            return pd.DataFrame()

        # Get original names
        original_index_name = stacked_data.columns[0]
        original_columns_name = stacked_data.columns[1]

        # Get latest value for each coordinate
        latest_data = filtered.groupby([original_index_name, original_columns_name])[
            "value"
        ].last()

        if latest_data.empty:
            return pd.DataFrame()

        # Reconstruct DataFrame
        # Annotate these as Index/iterables of Hashable to help the type checker
        available_columns: pd.Index = latest_data.index.get_level_values(1).unique()
        available_index: pd.Index = latest_data.index.get_level_values(0).unique()

        result = latest_data.unstack(level=1)

        # Try to preserve original order from dtype mapping at target time
        dtype_mapping = self._get_dtype_mapping_at_time(key, target_time)
        if (
            dtype_mapping
            and "columns_order" in dtype_mapping
            and "index_order" in dtype_mapping
        ):
            # Use saved order, filtering only available items
            original_col_order = dtype_mapping["columns_order"]  # These are strings
            original_idx_order = dtype_mapping["index_order"]  # These are strings

            # Create mapping dictionaries: string -> original_value
            # Map stringified representation to original Hashable values
            col_str_to_orig: dict[str, Hashable] = {
                str(col): col for col in available_columns
            }
            idx_str_to_orig: dict[str, Hashable] = {
                str(idx): idx for idx in available_index
            }

            # Create reverse mappings for remaining items detection
            available_cols_str = {str(col) for col in available_columns}
            available_idx_str = {str(idx) for idx in available_index}

            # Filter and preserve order for columns using string comparison
            preserved_cols_str = [
                col_str
                for col_str in original_col_order
                if col_str in available_cols_str
            ]
            remaining_cols_str = [
                col_str
                for col_str in available_cols_str
                if col_str not in preserved_cols_str
            ]

            # Convert back to original types
            preserved_cols: list[Hashable] = [
                col_str_to_orig[col_str] for col_str in preserved_cols_str
            ]
            remaining_cols: list[Hashable] = [
                col_str_to_orig[col_str] for col_str in sorted(remaining_cols_str)
            ]
            final_col_order = preserved_cols + remaining_cols

            # Filter and preserve order for index using string comparison
            preserved_idx_str = [
                idx_str
                for idx_str in original_idx_order
                if idx_str in available_idx_str
            ]
            remaining_idx_str = [
                idx_str
                for idx_str in available_idx_str
                if idx_str not in preserved_idx_str
            ]

            # Convert back to original types
            preserved_idx: list[Hashable] = [
                idx_str_to_orig[idx_str] for idx_str in preserved_idx_str
            ]
            remaining_idx: list[Hashable] = [
                idx_str_to_orig[idx_str] for idx_str in sorted(remaining_idx_str)
            ]
            final_idx_order: list[Hashable] = preserved_idx + remaining_idx

            result = result[final_col_order].reindex(index=final_idx_order)
        else:
            # Fallback to sorted order
            result = result[sorted(available_columns)].reindex(
                index=sorted(available_index)
            )

        # Restore data types using separate dtype mapping file at target time
        try:
            self._apply_dtypes_to_result(result, key, target_time)
        except Exception as e:
            logger.debug(f"Failed to apply dtypes for {key} at time {target_time}: {e}")
            pass

        # Restore original names
        # original names may be any Hashable; keep looser typing to satisfy type checker
        restored_index_name: Optional[object] = (
            None if original_index_name == "_finlab_index_none" else original_index_name
        )
        restored_columns_name: Optional[object] = (
            None
            if original_columns_name == "_finlab_columns_none"
            else original_columns_name
        )
        result.index.name = restored_index_name
        result.columns.name = restored_columns_name

        return result

    def _atomic_write_parquet(self, key: str, data: pd.DataFrame) -> None:
        """
        Atomically write parquet file.

        Args:
            key: Dataset key
            data: Data to write
        """
        cache_path = self._get_cache_path(key)

        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".parquet", dir=cache_path.parent
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Write data to temporary file
            data.to_parquet(tmp_path, compression=self.compression, index=False)

            # Atomic move
            tmp_path.replace(cache_path)
        except Exception:
            # Clean up on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise
