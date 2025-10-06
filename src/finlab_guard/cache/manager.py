"""Cache management for finlab-guard using DuckDB and Polars for high performance."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import duckdb
import orjson
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def _to_json_str(obj: Any) -> str:
    """Convert object to JSON string with enhanced type handling and precision preservation."""
    import numpy as np

    def _default(o: Any) -> Any:
        # Check for NaN/NaT first before type checking (pandas NaN detection)
        try:
            if pd.isna(o):  # Handle NaT and NaN values
                return None
        except (TypeError, ValueError):
            pass  # Not a pandas-compatible type

        # Handle numpy types with precision preservation
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return f"__HIGHPREC_FLOAT__{repr(float(o))}__"
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # Handle Python native types with precision preservation
        elif isinstance(o, float):
            if np.isnan(o):
                return None
            return f"__HIGHPREC_FLOAT__{repr(o)}__"
        elif isinstance(o, int):
            return o
        # Handle pandas types
        elif hasattr(o, "item"):  # numpy scalar types
            item_value = o.item()
            if isinstance(item_value, float):
                if np.isnan(item_value):
                    return None
                return f"__HIGHPREC_FLOAT__{repr(item_value)}__"
            return item_value
        # Enhanced pandas datetime handling
        elif isinstance(o, pd.Timestamp):  # Direct pandas Timestamp handling
            return o.isoformat() if not pd.isna(o) else None
        elif hasattr(o, "to_pydatetime"):  # other pandas datetime types
            return o.isoformat()
        elif hasattr(o, "isoformat"):  # standard datetime objects
            return o.isoformat()
        # Fallback to string conversion
        else:
            return str(o)

    def _has_special_values(obj: Any) -> bool:
        """Check if object contains inf/nan values that orjson can't handle."""
        if isinstance(obj, (int, float)):
            return bool(np.isinf(obj) or np.isnan(obj))
        elif isinstance(obj, dict):
            return any(_has_special_values(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return any(_has_special_values(v) for v in obj)
        elif hasattr(obj, "__array__"):  # numpy arrays/scalars
            try:
                result = np.any(np.isinf(obj)) or np.any(np.isnan(obj))
                return bool(result)
            except (TypeError, ValueError):
                return False
        return False

    # Use standard JSON for special values (orjson converts inf/nan to null)
    if _has_special_values(obj):
        return json.dumps(obj, default=_default, ensure_ascii=False)

    try:
        # Try orjson for better performance on normal values
        import orjson

        return orjson.dumps(obj, default=_default).decode("utf-8")
    except Exception:
        # Fallback to standard json
        return json.dumps(obj, default=_default, ensure_ascii=False)


@dataclass
class ChangeResult:
    """Result of diff computation between two DataFrames."""

    cell_changes: pd.DataFrame  # columns: row_key, col_key, value, save_time
    row_additions: pd.DataFrame  # columns: row_key, row_data, save_time
    row_deletions: pd.DataFrame  # columns: row_key, delete_time
    column_additions: pd.DataFrame  # columns: col_key, col_data_json, add_time
    column_deletions: pd.DataFrame  # columns: col_key, delete_time
    meta: dict[str, Any]
    dtype_changed: bool = False  # True if dtype has changed


class CacheManager:
    """High-performance cache manager using DuckDB and Polars.

    Features:
    - Cell-level diff storage using DuckDB for efficiency
    - Polars/NumPy vectorized diff computation (avoids pandas.stack())
    - Time-based reconstruction with window queries
    - Maintains API compatibility with original implementation
    """

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
        self.row_change_threshold = config.get("row_change_threshold", 200)

        # Initialize DuckDB connection
        db_path = cache_dir / "cache.duckdb"
        self.conn = duckdb.connect(str(db_path))
        self._setup_tables()

    def close(self) -> None:
        """Close the DuckDB connection."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self) -> None:
        """Ensure connection is closed when object is destroyed."""
        self.close()

    def __enter__(self) -> "CacheManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure connection is closed."""
        self.close()

    def _setup_tables(self) -> None:
        """Initialize DuckDB tables for cache storage."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rows_base (
                table_id VARCHAR,
                row_key VARCHAR,
                row_data VARCHAR,
                snapshot_time TIMESTAMP
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cell_changes (
                table_id VARCHAR,
                row_key VARCHAR,
                col_key VARCHAR,
                value VARCHAR,
                save_time TIMESTAMP
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS row_additions (
                table_id VARCHAR,
                row_key VARCHAR,
                row_data VARCHAR,
                save_time TIMESTAMP
            );
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_hashes (
                table_id VARCHAR,
                data_hash VARCHAR,
                save_time TIMESTAMP,
                PRIMARY KEY (table_id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS row_deletions (
                table_id VARCHAR,
                row_key VARCHAR,
                delete_time TIMESTAMP
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS column_deletions (
                table_id VARCHAR,
                col_key VARCHAR,
                delete_time TIMESTAMP
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS column_additions (
                table_id VARCHAR,
                col_key VARCHAR,
                col_data_json VARCHAR,
                add_time TIMESTAMP
            );
            """
        )

        # Create indexes for performance
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cell_changes_lookup ON cell_changes(table_id, save_time, row_key, col_key);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_row_additions_lookup ON row_additions(table_id, save_time, row_key);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rows_base_lookup ON rows_base(table_id, snapshot_time, row_key);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_row_deletions_lookup ON row_deletions(table_id, delete_time, row_key);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_column_deletions_lookup ON column_deletions(table_id, delete_time, col_key);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_column_additions_lookup ON column_additions(table_id, add_time, col_key);"
            )
        except Exception:
            pass  # Indexes might already exist or database doesn't support them

    def _get_cache_path(self, key: str) -> Path:
        """Legacy method for compatibility - returns DuckDB path."""
        return self.cache_dir / "cache.duckdb"

    def _get_dtype_path(self, key: str) -> Path:
        """Get dtype mapping file path for a dataset key."""
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
        # Prepare current dtype signature with enhanced categorical support
        dtypes_dict = {}
        categorical_metadata = {}

        for col in df.columns:
            col_str = str(col)
            dtype_str = str(df[col].dtype)
            dtypes_dict[col_str] = dtype_str

            # Save additional metadata for categorical dtypes
            if dtype_str == "category":
                try:
                    cat_col = df[col]
                    # Safely extract categorical metadata with error handling
                    categories_list = cat_col.cat.categories.tolist()
                    ordered = cat_col.cat.ordered
                    categories_dtype = str(cat_col.cat.categories.dtype)

                    categorical_metadata[col_str] = {
                        "categories": categories_list,
                        "ordered": ordered,
                        "categories_dtype": categories_dtype,
                    }
                except Exception as e:
                    logger.debug(
                        f"Failed to extract categorical metadata for column '{col}': {e}"
                    )
                    # Fallback: convert to object dtype to avoid categorical issues
                    logger.debug(
                        f"Converting column '{col}' from category to object dtype"
                    )
                    df[col] = df[col].astype(str)
                    dtypes_dict[col_str] = "object"

        current_signature = {
            "dtypes": dtypes_dict,
            "categorical_metadata": categorical_metadata,
            "index_dtype": str(df.index.dtype),
            "columns_dtype": str(df.columns.dtype),
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
            # Use enhanced JSON serialization to handle Timestamp objects
            dtype_json_str = _to_json_str(dtype_mapping)
            f.write(dtype_json_str)

        logger.debug(f"Saved new dtype entry for {key} at {new_entry['timestamp']}")

    def _compute_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate DataFrame hash value including dtype information.

        Args:
            df: DataFrame to hash

        Returns:
            SHA256 hex string of the DataFrame
        """
        import hashlib

        if df.empty:
            return hashlib.sha256(b"empty_dataframe").hexdigest()

        # Include dtype information to distinguish int8 vs int16 etc.
        content = (
            df.values.tobytes()
            + str(df.index.tolist()).encode()
            + str(df.columns.tolist()).encode()
            + str(
                {col: str(df[col].dtype) for col in df.columns}
            ).encode()  # Add dtype info
            + str(df.index.dtype).encode()  # Add index dtype
        )

        return hashlib.sha256(content).hexdigest()

    def _save_data_hash(self, key: str, hash_value: str, timestamp: datetime) -> None:
        """
        Save or update data hash.

        Args:
            key: Dataset key
            hash_value: SHA256 hash of the data
            timestamp: Save timestamp
        """
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO data_hashes (table_id, data_hash, save_time)
                VALUES (?, ?, ?)
                """,
                [key, hash_value, timestamp],
            )
        except Exception as e:
            logger.error(f"Failed to save hash for {key}: {e}")

    def _get_data_hash(self, key: str) -> Optional[str]:
        """
        Get cached data hash.

        Args:
            key: Dataset key

        Returns:
            Hash value if exists, None otherwise
        """
        try:
            result = self.conn.execute(
                "SELECT data_hash FROM data_hashes WHERE table_id = ?", [key]
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get hash for {key}: {e}")
            return None

    def _check_dtype_changed(self, key: str, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame dtype has changed compared to cached version.
        Only checks actual dtype changes, NOT index/row order changes.

        Args:
            key: Dataset key
            df: Current DataFrame

        Returns:
            True if dtype has changed (excluding index_order changes)
        """
        # Build current signature (reuse same logic as _save_dtype_mapping)
        dtypes_dict = {}
        categorical_metadata = {}

        for col in df.columns:
            col_str = str(col)
            dtype_str = str(df[col].dtype)
            dtypes_dict[col_str] = dtype_str

            if dtype_str == "category":
                try:
                    cat_col = df[col]
                    categorical_metadata[col_str] = {
                        "categories": cat_col.cat.categories.tolist(),
                        "ordered": cat_col.cat.ordered,
                        "categories_dtype": str(cat_col.cat.categories.dtype),
                    }
                except Exception:
                    pass

        current_signature = {
            "dtypes": dtypes_dict,
            "categorical_metadata": categorical_metadata,
            "index_dtype": str(df.index.dtype),
            "columns_dtype": str(df.columns.dtype),
            "index_name": df.index.name,
            "columns_name": df.columns.name,
            "columns_order": [str(col) for col in df.columns],
            "index_freq": (
                getattr(df.index, "freqstr", None)
                if getattr(df.index, "freq", None) is not None
                else None
            ),
        }

        # Load existing mapping
        existing_mapping = self._load_dtype_mapping(key)

        if not existing_mapping or "dtype_history" not in existing_mapping:
            # First time - no dtype change
            return False

        dtype_history = existing_mapping.get("dtype_history", [])
        if not dtype_history:
            return False

        # Get latest entry
        latest_entry = dtype_history[-1]

        # Check ONLY dtype-related changes for EXISTING columns
        # Exclude: index_order, column additions/deletions
        latest_dtypes = latest_entry.get("dtypes", {})
        current_dtypes = current_signature.get("dtypes", {})

        # Type assertion for mypy
        assert isinstance(latest_dtypes, dict)
        assert isinstance(current_dtypes, dict)

        # Find common columns (exclude additions/deletions)
        common_columns = set(latest_dtypes.keys()) & set(current_dtypes.keys())

        # Check if any common column's dtype changed
        common_dtypes_changed = any(
            latest_dtypes[col] != current_dtypes[col] for col in common_columns
        )

        # Check categorical metadata for common columns
        latest_cat_meta = latest_entry.get("categorical_metadata", {})
        current_cat_meta = current_signature.get("categorical_metadata", {})
        assert isinstance(latest_cat_meta, dict)
        assert isinstance(current_cat_meta, dict)

        common_cat_changed = any(
            latest_cat_meta.get(col) != current_cat_meta.get(col)
            for col in common_columns
            if col in latest_cat_meta or col in current_cat_meta
        )

        # Check other dtype-related properties (not related to column additions/deletions)
        dtype_changed = (
            common_dtypes_changed
            or common_cat_changed
            or latest_entry.get("index_dtype") != current_signature.get("index_dtype")
            or latest_entry.get("columns_dtype")
            != current_signature.get("columns_dtype")
            or latest_entry.get("index_name") != current_signature.get("index_name")
            or latest_entry.get("columns_name") != current_signature.get("columns_name")
            or latest_entry.get("index_freq") != current_signature.get("index_freq")
        )

        return dtype_changed

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

        # Compare each component including categorical metadata
        changes_detected = (
            latest_entry.get("dtypes") != current_signature.get("dtypes")
            or latest_entry.get("categorical_metadata", {})
            != current_signature.get("categorical_metadata", {})
            or latest_entry.get("index_dtype") != current_signature.get("index_dtype")
            or latest_entry.get("columns_dtype")
            != current_signature.get("columns_dtype")
            or latest_entry.get("index_name") != current_signature.get("index_name")
            or latest_entry.get("columns_name") != current_signature.get("columns_name")
            or latest_entry.get("columns_order")
            != current_signature.get("columns_order")
            or set(latest_entry.get("index_order", []))
            != set(current_signature.get("index_order", []))
            or latest_entry.get("index_freq") != current_signature.get("index_freq")
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

    # =================== New DuckDB Core Methods ===================

    def save_snapshot(
        self, table_id: str, df: pd.DataFrame, snapshot_time: Optional[datetime] = None
    ) -> None:
        """Save a complete DataFrame snapshot to DuckDB."""
        if snapshot_time is None:
            snapshot_time = datetime.now()

        rows = []
        for idx, row in df.iterrows():
            rows.append(
                (table_id, str(idx), _to_json_str(row.to_dict()), snapshot_time)
            )

        if not rows:
            return

        tmp = pd.DataFrame(
            rows, columns=["table_id", "row_key", "row_data", "snapshot_time"]
        )
        self.conn.register("_tmp_snapshot", tmp)
        self.conn.execute("INSERT INTO rows_base SELECT * FROM _tmp_snapshot")
        self.conn.unregister("_tmp_snapshot")

    def save_version(
        self,
        table_id: str,
        prev_df: pd.DataFrame,
        cur_df: pd.DataFrame,
        timestamp: Optional[datetime] = None,
    ) -> ChangeResult:
        """Save only the changes between prev_df and cur_df and return the changes."""
        if timestamp is None:
            timestamp = datetime.now()

        changes = self.get_changes_extended(prev_df, cur_df, timestamp)

        # Persist cell changes
        if not changes.cell_changes.empty:
            cell = changes.cell_changes.copy()
            cell.insert(0, "table_id", table_id)
            self.conn.register("_tmp_cell", cell)
            self.conn.execute("INSERT INTO cell_changes SELECT * FROM _tmp_cell")
            self.conn.unregister("_tmp_cell")

        # Persist row additions
        if not changes.row_additions.empty:
            ra = changes.row_additions.copy()
            ra.insert(0, "table_id", table_id)
            self.conn.register("_tmp_row_add", ra)
            self.conn.execute("INSERT INTO row_additions SELECT * FROM _tmp_row_add")
            self.conn.unregister("_tmp_row_add")

        # Persist row deletions
        if not changes.row_deletions.empty:
            rd = changes.row_deletions.copy()
            rd.insert(0, "table_id", table_id)
            self.conn.register("_tmp_row_del", rd)
            self.conn.execute("INSERT INTO row_deletions SELECT * FROM _tmp_row_del")
            self.conn.unregister("_tmp_row_del")

        # Persist column additions
        if not changes.column_additions.empty:
            ca = changes.column_additions.copy()
            ca.insert(0, "table_id", table_id)
            self.conn.register("_tmp_col_add", ca)
            self.conn.execute("INSERT INTO column_additions SELECT * FROM _tmp_col_add")
            self.conn.unregister("_tmp_col_add")

        # Persist column deletions
        if not changes.column_deletions.empty:
            cd = changes.column_deletions.copy()
            cd.insert(0, "table_id", table_id)
            self.conn.register("_tmp_col_del", cd)
            self.conn.execute("INSERT INTO column_deletions SELECT * FROM _tmp_col_del")
            self.conn.unregister("_tmp_col_del")

        return changes

    def get_changes_extended(
        self, prev: pd.DataFrame, cur: pd.DataFrame, timestamp: datetime
    ) -> ChangeResult:
        """Compute changes between prev and cur using Polars for high performance."""
        cell_df, row_df, row_deletions_df, col_additions_df, col_deletions_df, meta = (
            self._get_changes_extended_polars(
                prev, cur, timestamp, self.row_change_threshold
            )
        )

        return ChangeResult(
            cell_changes=cell_df,
            row_additions=row_df,
            row_deletions=row_deletions_df,
            column_additions=col_additions_df,
            column_deletions=col_deletions_df,
            meta=meta,
        )

    def _to_pdf_with_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize pandas DF: index -> __row_key__ column (string)"""
        pdf = df.copy()
        # Use str() for each index value to match save_snapshot format
        # This preserves full timestamp precision: '2023-06-04 00:00:00' instead of '2023-06-04'
        pdf.index = pd.Index([str(idx) for idx in pdf.index])
        pdf = pdf.reset_index()
        pdf.columns.values[0] = "__row_key__"
        return pdf

    def _get_changes_extended_polars(
        self,
        prev: pd.DataFrame,
        cur: pd.DataFrame,
        timestamp: datetime,
        row_change_threshold: int = 200,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Any],
    ]:
        """
        Use Polars to compute sparse cell changes, row additions, and deletions.
        Returns (cell_changes_df, row_additions_df, row_deletions_df, column_additions_df, column_deletions_df, meta)
        """
        # Prepare pandas frames with __row_key__
        cur_pdf = self._to_pdf_with_key(cur)
        prev_pdf = (
            self._to_pdf_with_key(prev)
            if (prev is not None and not prev.empty)
            else pd.DataFrame(columns=["__row_key__"] + list(cur.columns))
        )

        # Convert to Polars (preserve NaN for accurate comparison)
        # Handle mixed types in object columns by forcing them to string
        def _prepare_for_polars(df: pd.DataFrame) -> pd.DataFrame:
            """Prepare pandas DataFrame for Polars conversion by handling mixed types."""
            df_copy = df.copy()
            for col in df_copy.columns:
                if (
                    isinstance(df_copy[col].dtype, pd.CategoricalDtype)
                    and col != "__row_key__"
                ):
                    # convert the category column
                    try:
                        cat_dtype = df_copy[col].dtype
                        if hasattr(cat_dtype, "categories"):
                            df_copy[col] = df_copy[col].astype(
                                cat_dtype.categories.dtype
                            )
                    except (ValueError, TypeError):
                        pass

                # Handle string dtype: convert pd.NA to "<NA>" string
                if col != "__row_key__" and pd.api.types.is_string_dtype(df_copy[col]):
                    try:
                        # Replace pd.NA with the string "<NA>"
                        df_copy[col] = df_copy[col].fillna("<NA>")
                    except (ValueError, TypeError):
                        pass
            return df_copy

        try:
            p_prev = pl.from_pandas(_prepare_for_polars(prev_pdf), nan_to_null=False)
            p_cur = pl.from_pandas(_prepare_for_polars(cur_pdf), nan_to_null=False)
        except Exception as e:
            logger.debug(
                f"Polars conversion failed: {e}, falling back to string conversion"
            )
            # More aggressive fallback: convert all non-numeric columns to string
            prev_str = prev_pdf.copy()
            cur_str = cur_pdf.copy()
            for col in prev_str.columns:
                if prev_str[col].dtype == "object" and col != "__row_key__":
                    prev_str[col] = prev_str[col].astype(str)
            for col in cur_str.columns:
                if cur_str[col].dtype == "object" and col != "__row_key__":
                    cur_str[col] = cur_str[col].astype(str)
            p_prev = pl.from_pandas(prev_str, nan_to_null=False)
            p_cur = pl.from_pandas(cur_str, nan_to_null=False)

        # Determine union of data columns (exclude key)
        prev_cols = [c for c in prev_pdf.columns if c != "__row_key__"]
        cur_cols = [c for c in cur_pdf.columns if c != "__row_key__"]
        # Union preserves order: cur first (favor current schema), then any old-only cols
        union_cols = list(
            dict.fromkeys(cur_cols + [c for c in prev_cols if c not in cur_cols])
        )

        # Outer join on key with suffix (Polars only accepts string suffix)
        joined = p_prev.join(p_cur, on="__row_key__", how="full", suffix="_new")

        # Ensure all suffixed columns exist
        for col in union_cols:
            old_col = col  # Original columns from p_prev don't have suffix
            new_col = f"{col}_new"
            if old_col not in joined.columns:
                joined = joined.with_columns(pl.lit(None).alias(old_col))
            if new_col not in joined.columns:
                joined = joined.with_columns(pl.lit(None).alias(new_col))

        # For each column, produce changed rows (row_key, old, new, col_key)
        changed_frames = []
        for col in union_cols:
            old_col = col  # Original columns from p_prev don't have suffix
            new_col = f"{col}_new"

            # Smart comparison that preserves data types and precision
            try:
                # First, try direct comparison without forced float conversion
                # This preserves integer precision and avoids float64 precision loss
                basic_mask = (
                    ~(pl.col(old_col).is_null() & pl.col(new_col).is_null())
                ) & (pl.col(old_col) != pl.col(new_col))

                # Check if this basic comparison can work
                can_use_basic = True
                try:
                    joined.select(basic_mask).limit(1)
                except Exception:
                    can_use_basic = False

                if can_use_basic:
                    # Use smart serialization to handle various data types consistently
                    def smart_serialize(x: Any) -> Optional[str]:
                        """Smart serialization that preserves data types with explicit markers."""
                        import numpy as np

                        # Handle None explicitly
                        if x is None:
                            return "__NONE__"

                        # Check for pandas NA (pd.NA is a special singleton)
                        try:
                            # pd.NA has type pandas._libs.missing.NAType
                            if pd.isna(x):
                                # Distinguish between different NA types
                                x_type = type(x).__name__
                                if x_type == "NAType":  # pandas pd.NA
                                    return "__PD_NA__"
                                # For other NA values, continue to check type
                        except (TypeError, ValueError):
                            pass  # Not a pandas-compatible type

                        # Handle boolean first (before int, since bool is subclass of int)
                        if isinstance(x, (bool, np.bool_)):
                            return f"__BOOL__{str(x)}__"  # "__BOOL__True__" or "__BOOL__False__"
                        # Mark integers explicitly to prevent confusion with string numbers
                        elif isinstance(x, (int, np.integer)):
                            return f"__INT__{int(x)}__"  # "__INT__1__", "__INT__-42__", etc.
                        elif isinstance(x, (float, np.floating)):
                            if np.isnan(x):
                                return "__NAN__"  # Explicit NaN marker
                            return (
                                f"__FLOAT__{repr(float(x))}__"  # Unified float marker
                            )
                        else:
                            return str(x)  # Strings remain unmarked

                    df_changed = (
                        joined.filter(basic_mask)
                        .select(
                            [
                                pl.col("__row_key__").alias("row_key"),
                                # Use smart serialization to preserve original data types
                                pl.col(old_col)
                                .map_elements(smart_serialize, return_dtype=pl.Utf8)
                                .alias("old"),
                                pl.col(new_col)
                                .map_elements(smart_serialize, return_dtype=pl.Utf8)
                                .alias("new"),
                            ]
                        )
                        .with_columns(pl.lit(col).alias("col_key"))
                    )
                else:
                    # Fallback: try float comparison for mixed types
                    old_numeric = pl.col(old_col).cast(pl.Float64, strict=False)
                    new_numeric = pl.col(new_col).cast(pl.Float64, strict=False)

                    # Check if numeric conversion works
                    try:
                        joined.select(old_numeric, new_numeric).limit(1)
                        numeric_mask = (
                            ~(pl.col(old_col).is_null() & pl.col(new_col).is_null())
                        ) & (old_numeric != new_numeric)
                        df_changed = (
                            joined.filter(numeric_mask)
                            .select(
                                [
                                    pl.col("__row_key__").alias("row_key"),
                                    # Only use high-precision encoding for actual floats
                                    old_numeric.map_elements(
                                        lambda x: f"__HIGHPREC_FLOAT__{repr(x)}__"
                                        if x is not None
                                        else None,
                                        return_dtype=pl.Utf8,
                                    ).alias("old"),
                                    new_numeric.map_elements(
                                        lambda x: f"__HIGHPREC_FLOAT__{repr(x)}__"
                                        if x is not None
                                        else None,
                                        return_dtype=pl.Utf8,
                                    ).alias("new"),
                                ]
                            )
                            .with_columns(pl.lit(col).alias("col_key"))
                        )
                    except Exception:
                        # Final fallback to string comparison
                        string_mask = (
                            ~(pl.col(old_col).is_null() & pl.col(new_col).is_null())
                        ) & (
                            pl.col(old_col).cast(pl.Utf8)
                            != pl.col(new_col).cast(pl.Utf8)
                        )
                        df_changed = (
                            joined.filter(string_mask)
                            .select(
                                [
                                    pl.col("__row_key__").alias("row_key"),
                                    pl.col(old_col).cast(pl.Utf8).alias("old"),
                                    pl.col(new_col).cast(pl.Utf8).alias("new"),
                                ]
                            )
                            .with_columns(pl.lit(col).alias("col_key"))
                        )
            except Exception:
                # Fallback to string comparison if numeric comparison fails
                string_mask = (
                    ~(pl.col(old_col).is_null() & pl.col(new_col).is_null())
                ) & (pl.col(old_col).cast(pl.Utf8) != pl.col(new_col).cast(pl.Utf8))
                df_changed = (
                    joined.filter(string_mask)
                    .select(
                        [
                            pl.col("__row_key__").alias("row_key"),
                            pl.col(old_col).cast(pl.Utf8).alias("old"),
                            pl.col(new_col).cast(pl.Utf8).alias("new"),
                        ]
                    )
                    .with_columns(pl.lit(col).alias("col_key"))
                )

            changed_frames.append(df_changed)

        if changed_frames:
            all_changes_pl = pl.concat(changed_frames, how="vertical")
            all_changes_pdf = (
                all_changes_pl.to_pandas()
            )  # columns: row_key, old, new, col_key
            # Compute per-row counts to apply thresholding
            counts = (
                all_changes_pdf.groupby("row_key")
                .size()
                .rename("n_changes")
                .reset_index()
            )
            big_row_keys = counts[counts["n_changes"] > row_change_threshold][
                "row_key"
            ].tolist()
            big_rows: set[str] = set(
                big_row_keys
            )  # row_key is already string from _to_pdf_with_key
        else:
            all_changes_pdf = pd.DataFrame(columns=["row_key", "old", "new", "col_key"])
            big_rows = set()

        # Build row additions: new rows in cur not in prev
        prev_keys = (
            {str(idx) for idx in prev.index}
            if (prev is not None and not prev.empty)
            else set()
        )
        cur_keys = list(cur.index.tolist())  # Keep original types for DataFrame access
        cur_keys_str = [
            str(idx) for idx in cur.index
        ]  # String version for comparisons - same format as _to_pdf_with_key
        cur_keys_set = set(cur_keys_str)
        # Create mapping from string to original index for DataFrame access
        # Use the same string format as cur_keys_str for consistency
        str_to_orig = {cur_keys_str[i]: cur_keys[i] for i in range(len(cur_keys))}
        new_rows = [k for k in cur_keys_str if k not in prev_keys]

        # Build row deletions: rows in prev not in cur
        deleted_rows = [k for k in prev_keys if k not in cur_keys_set]

        row_adds = []
        for r in new_rows:
            # Convert row to dict and ensure JSON serializable types
            # r is string, need to map back to original index for DataFrame access
            orig_r = str_to_orig[r]
            row_dict = cur.loc[orig_r].to_dict()
            # Clean up types to avoid JSON serialization issues
            clean_row_dict: dict[str, Any] = {}
            for k, v in row_dict.items():
                if pd.isna(v):
                    clean_row_dict[k] = None
                elif hasattr(v, "item"):  # numpy scalar types
                    clean_row_dict[k] = v.item()
                else:
                    clean_row_dict[k] = v
            row_adds.append((str(r), _to_json_str(clean_row_dict), timestamp))

        row_deletions = []
        for r in deleted_rows:
            row_deletions.append((str(r), timestamp))

        # Build cell changes for non-big rows
        cell_rows = []
        if not all_changes_pdf.empty:
            for _, row in all_changes_pdf.iterrows():
                rk = str(row["row_key"])
                ck = row["col_key"]
                if rk in big_rows:
                    continue
                # Handle type conversion for cell values
                if pd.isna(row["new"]):
                    newv = None  # type: ignore[unreachable]
                elif hasattr(  # type: ignore[unreachable]
                    row["new"], "item"
                ):  # numpy scalar types
                    newv = row["new"].item()
                elif isinstance(row["new"], str) and row["new"].lower() in {
                    "true",
                    "false",
                }:
                    newv = True if row["new"].lower() == "true" else False
                else:
                    newv = row["new"]  # Already serialized by smart_serialize
                cell_rows.append(  # type: ignore[unreachable]
                    (rk, str(ck), _to_json_str(newv), timestamp)
                )  # Don't double-serialize

            # Big rows become partial row maps stored in row_additions
            for br in big_rows:
                subset = all_changes_pdf[all_changes_pdf["row_key"] == br]
                row_map: dict[str, Any] = {}
                for _, r in subset.iterrows():  # type: ignore[assignment]
                    new_val = r["new"]  # type: ignore[index]
                    col_key_val = r["col_key"]  # type: ignore[index]
                    if pd.isna(new_val):
                        row_map[str(col_key_val)] = None  # type: ignore[unreachable]
                    elif hasattr(new_val, "item"):  # numpy scalar types
                        row_map[str(col_key_val)] = new_val.item()
                    else:
                        row_map[str(col_key_val)] = new_val
                row_adds.append((str(br), _to_json_str(row_map), timestamp))

        # New columns: store as column_additions with full column data
        new_cols = [c for c in cur_cols if c not in prev_cols]
        col_additions = []
        if new_cols:
            for c in new_cols:
                # Extract the full column data as a dictionary {row_key: value}
                col_data = {}
                for (
                    r
                ) in cur_keys:  # r is now original type, can access DataFrame directly
                    v = cur.at[r, c]
                    # Handle pandas NA/NaN values and ensure JSON serializable types
                    if pd.isna(v):
                        # Convert to string for storage
                        col_data[str(r)] = None  # type: ignore[unreachable]
                    else:
                        # Convert to Python native types to avoid JSON serialization issues
                        if hasattr(v, "item"):  # numpy scalar types
                            col_data[str(r)] = v.item()  # Convert to string for storage
                        else:
                            col_data[str(r)] = v  # Convert to string for storage
                col_additions.append((str(c), _to_json_str(col_data), timestamp))

        # Deleted columns: columns in prev not in cur
        deleted_cols = [c for c in prev_cols if c not in cur_cols]
        col_deletions = []
        for c in deleted_cols:
            col_deletions.append((str(c), timestamp))

        cell_df = (
            pd.DataFrame(
                cell_rows, columns=["row_key", "col_key", "value", "save_time"]
            )
            if cell_rows
            else pd.DataFrame(columns=["row_key", "col_key", "value", "save_time"])
        )
        row_df = (
            pd.DataFrame(row_adds, columns=["row_key", "row_data", "save_time"])
            if row_adds
            else pd.DataFrame(columns=["row_key", "row_data", "save_time"])
        )

        row_deletions_df = (
            pd.DataFrame(row_deletions, columns=["row_key", "delete_time"])
            if row_deletions
            else pd.DataFrame(columns=["row_key", "delete_time"])
        )

        col_additions_df = (
            pd.DataFrame(
                col_additions, columns=["col_key", "col_data_json", "add_time"]
            )
            if col_additions
            else pd.DataFrame(columns=["col_key", "col_data_json", "add_time"])
        )

        col_deletions_df = (
            pd.DataFrame(col_deletions, columns=["col_key", "delete_time"])
            if col_deletions
            else pd.DataFrame(columns=["col_key", "delete_time"])
        )

        meta = {
            "new_rows": new_rows,
            "deleted_rows": deleted_rows,
            "new_cols": new_cols,
            "deleted_cols": deleted_cols,
            "big_rows": list(big_rows),
            "union_cols": union_cols,
        }
        return (
            cell_df,
            row_df,
            row_deletions_df,
            col_additions_df,
            col_deletions_df,
            meta,
        )

    def _parse_json_batch(self, json_strings: list[str]) -> list[dict]:
        """Parse JSON strings with orjson -> json fallback."""
        parsed = []
        for s in json_strings:
            try:
                parsed.append(orjson.loads(s) if s else {})
            except Exception:
                try:
                    parsed.append(json.loads(s) if s else {})
                except Exception:
                    parsed.append({})
        return parsed

    def reconstruct_as_of(self, table_id: str, target_time: datetime) -> pd.DataFrame:
        """
        Reconstruct DataFrame as of specific time by combining data layers.

        Returns pandas.DataFrame (index = row_key strings).
        """

        base_data, snapshot_time = self._load_base_snapshot(table_id, target_time)

        cell_changes = self._load_and_process_cell_changes(
            table_id, snapshot_time, target_time
        )

        row_additions = self._load_and_process_row_additions(
            table_id, snapshot_time, target_time
        )

        column_additions = self._load_and_process_column_additions(
            table_id, snapshot_time, target_time
        )

        deleted_rows, deleted_cols = self._load_deletions(
            table_id, snapshot_time, target_time
        )

        merged_data = self._merge_data_layers(
            base_data, cell_changes, row_additions, column_additions
        )

        result = self._finalize_dataframe(merged_data, deleted_rows, deleted_cols)

        if result.empty:
            logger.warning(f"No cache data found for {table_id}")
            return pd.DataFrame()

        result = self._apply_dtypes_to_result(result, table_id, target_time)

        return result

    def _load_base_snapshot(
        self, table_id: str, target_time: datetime
    ) -> tuple[pl.DataFrame, datetime]:
        """
        Load latest snapshot row_data for the table as Polars DataFrame.

        Returns:
            Tuple of (base_data, snapshot_time)
        """
        # First get the snapshot_time
        q_snap_time = f"""
        SELECT MAX(snapshot_time) as snap_time FROM rows_base
        WHERE table_id = '{table_id}' AND snapshot_time <= TIMESTAMP '{target_time.isoformat()}'
        """
        snap_time_result = self.conn.execute(q_snap_time).fetchone()
        snapshot_time = (
            snap_time_result[0]
            if snap_time_result and snap_time_result[0]
            else target_time
        )

        # Now get the snapshot data
        q_snap = f"""
        SELECT row_key, row_data FROM rows_base
        WHERE table_id = '{table_id}' AND snapshot_time = TIMESTAMP '{snapshot_time.isoformat()}'
        """
        base_df = self.conn.execute(q_snap).fetchdf()

        # Convert base snapshot JSON column into a wide DataFrame
        if not base_df.empty:
            row_keys = base_df["row_key"].to_list()
            raw_json = base_df["row_data"].fillna("{}").to_list()
            parsed = self._parse_json_batch(raw_json)

            # normalize to wide table (C-optimized)
            base_wide_pdf = pd.json_normalize(parsed)

            if not base_wide_pdf.empty:
                base_wide_pdf["row_key"] = row_keys
                try:
                    base_pl = pl.from_pandas(
                        base_wide_pdf, nan_to_null=False
                    ).with_columns(pl.col("row_key").cast(pl.Utf8))
                except Exception:
                    # Convert object columns to string to avoid type inference issues
                    base_wide_pdf_str = base_wide_pdf.copy()
                    for col in base_wide_pdf_str.columns:
                        if (
                            base_wide_pdf_str[col].dtype == "object"
                            and col != "row_key"
                        ):
                            base_wide_pdf_str[col] = base_wide_pdf_str[col].astype(str)
                    base_pl = pl.from_pandas(
                        base_wide_pdf_str, nan_to_null=False
                    ).with_columns(pl.col("row_key").cast(pl.Utf8))
            else:
                base_pl = pl.DataFrame({"row_key": pl.Series(row_keys, dtype=pl.Utf8)})
        else:
            base_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})

        return base_pl, snapshot_time

    def _load_and_process_cell_changes(
        self, table_id: str, snapshot_time: datetime, target_time: datetime
    ) -> pl.DataFrame:
        """
        Load and process cell changes into pivoted Polars DataFrame.

        Only loads changes where snapshot_time < save_time <= target_time.
        """
        q_changes_latest = f"""
        SELECT row_key, col_key, value FROM (
            SELECT row_key, col_key, value,
                row_number() OVER (PARTITION BY row_key, col_key ORDER BY save_time DESC) as rn
            FROM cell_changes
            WHERE table_id = '{table_id}'
              AND save_time > TIMESTAMP '{snapshot_time.isoformat()}'
              AND save_time <= TIMESTAMP '{target_time.isoformat()}'
        ) t WHERE rn = 1
        """

        # Simplified arrow handling
        try:
            changes_arrow = self.conn.execute(q_changes_latest).arrow()
            if hasattr(changes_arrow, "read_all"):
                changes_arrow = changes_arrow.read_all()
            if changes_arrow.num_rows > 0:
                changes_pl: pl.DataFrame = pl.from_arrow(changes_arrow)  # type: ignore[assignment]
            else:
                changes_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})
        except Exception:
            changes_pdf = self.conn.execute(q_changes_latest).fetchdf()
            if not changes_pdf.empty:
                try:
                    changes_pl = pl.from_pandas(changes_pdf, nan_to_null=False)
                except Exception:
                    # Convert object columns to string to avoid type inference issues
                    changes_pdf_str = changes_pdf.copy()
                    for col in changes_pdf_str.columns:
                        if changes_pdf_str[col].dtype == "object":
                            changes_pdf_str[col] = changes_pdf_str[col].astype(str)
                    changes_pl = pl.from_pandas(changes_pdf_str, nan_to_null=False)
            else:
                changes_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})

        # Process cell changes and pivot
        if not changes_pl.is_empty():
            # Parse JSON values but convert to string for pivot compatibility
            # We'll handle type conversion during final merge
            def parse_and_convert_for_pivot(value: str) -> str:
                """Parse JSON value and convert to string for pivot operation."""
                parsed = self._parse_json_value(value)
                if parsed is None:
                    return ""
                elif isinstance(parsed, bool):
                    # Keep Boolean as "true"/"false" for later type inference
                    return "true" if parsed else "false"
                else:
                    return str(parsed)

            changes_pl = changes_pl.with_columns(
                [
                    pl.col("row_key").cast(pl.Utf8),
                    pl.col("col_key").cast(pl.Utf8),
                    pl.col("value")
                    .map_elements(parse_and_convert_for_pivot, return_dtype=pl.Utf8)
                    .alias("value"),
                ]
            )

            # Simplified pivot with single try
            try:
                pivot_pl = changes_pl.pivot(
                    on="col_key",
                    index="row_key",
                    values="value",
                    aggregate_function="first",
                )
            except Exception:
                # If pivot fails, use groupby fallback
                grouped_pl = changes_pl.group_by(["row_key", "col_key"]).agg(
                    pl.col("value").first()
                )
                pivot_pl = grouped_pl.pivot(
                    on="col_key", index="row_key", values="value"
                )

            # Rename pivot columns to indicate delta
            pivot_cols = [c for c in pivot_pl.columns if c != "row_key"]
            if pivot_cols:
                pivot_rename_map = {c: f"{c}__delta" for c in pivot_cols}
                pivot_pl = pivot_pl.rename(pivot_rename_map)

            # Apply type inference to delta columns (simpler approach)
            if pivot_cols:
                delta_cols = [f"{c}__delta" for c in pivot_cols]

                # Check each column and convert if all non-null values are Boolean strings
                for col_name in delta_cols:
                    try:
                        # Get non-null values
                        non_null_mask = pivot_pl[col_name].is_not_null()

                        # Check if all non-null values are "true" or "false"
                        all_boolean = pivot_pl.select(
                            pl.col(col_name)
                            .filter(non_null_mask)
                            .is_in(["true", "false"])
                            .all()
                        ).item()

                        if all_boolean:
                            # Convert the entire column to Boolean
                            pivot_pl = pivot_pl.with_columns(
                                pl.when(pl.col(col_name) == "true")
                                .then(True)
                                .when(pl.col(col_name) == "false")
                                .then(False)
                                .otherwise(None)
                                .alias(col_name)
                            )
                    except Exception as e:
                        # If conversion fails, keep as string
                        logger.debug(
                            f"Boolean conversion failed for column {col_name}: {e}"
                        )
                        pass

            # Ensure row_key is string type
            pivot_pl = pivot_pl.with_columns(pl.col("row_key").cast(pl.Utf8))
        else:
            pivot_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})

        return pivot_pl

    def _parse_marker_string(self, s: str) -> Optional[Any]:
        """Parse type marker strings like __INT__123__, __BOOL__True__, __FLOAT__1.5__.

        Returns:
            Parsed value if a valid marker is found, None otherwise.
        """
        if not isinstance(s, str):
            return None  # type: ignore[unreachable]

        # Check for special missing value markers
        if s == "__NONE__":
            return None
        if s == "__PD_NA__":
            return pd.NA  # Return pandas NA singleton
        if s == "__NAN__":
            import numpy as np

            return np.nan

        # Check for integer markers: __INT__123__
        if s.startswith("__INT__") and s.endswith("__"):
            try:
                int_str = s[7:-2]  # Remove "__INT__" and trailing "__"
                result: int = int(int_str)
                return result
            except (ValueError, TypeError):
                return None

        # Check for boolean markers: __BOOL__True__ or __BOOL__False__
        if s.startswith("__BOOL__") and s.endswith("__"):
            try:
                bool_str = s[8:-2]  # Remove "__BOOL__" and trailing "__"
                if bool_str == "True":
                    result_bool: bool = True
                elif bool_str == "False":
                    result_bool = False
                else:
                    raise ValueError(f"Invalid boolean value: {bool_str}")
                return result_bool
            except (ValueError, TypeError):
                return None

        # Check for float markers: __FLOAT__1.5__ or __HIGHPREC_FLOAT__1.5__ (legacy)
        if (
            s.startswith("__FLOAT__") or s.startswith("__HIGHPREC_FLOAT__")
        ) and s.endswith("__"):
            try:
                if s.startswith("__FLOAT__"):
                    float_repr = s[9:-2]  # Remove "__FLOAT__" and trailing "__"
                else:  # __HIGHPREC_FLOAT__ (legacy support)
                    float_repr = s[
                        18:-2
                    ]  # Remove "__HIGHPREC_FLOAT__" and trailing "__"
                result_float: float = float(float_repr)
                return result_float
            except (ValueError, TypeError):
                return None

        # No marker found
        return None

    def _parse_json_value(self, x: Any) -> Any:
        """Parse JSON value with smart type conversion and precision preservation.

        Handles:
        - Direct marker strings: __INT__123__, __BOOL__True__, __FLOAT__1.5__
        - JSON-encoded values containing markers
        - Primitive types (int, float, bool)
        - Regular strings (returned as-is, no type guessing)
        """
        # Fast path: primitive types
        if x is None:
            return None
        if isinstance(x, (int, float, bool)):
            return x

        # String parsing
        if isinstance(x, str):
            # Check for direct marker strings first (before JSON parsing)
            marker_result = self._parse_marker_string(x)
            if marker_result is not None:
                return marker_result

            # Try JSON parsing (for JSON-encoded strings)
            try:
                parsed = orjson.loads(x) if x else None

                # If parsed result is a string, check for markers
                if isinstance(parsed, str):
                    marker_result = self._parse_marker_string(parsed)
                    if marker_result is not None:
                        return marker_result

                # Return parsed value as-is (no type guessing)
                return parsed

            except Exception:
                # No type guessing - return the original string
                return x

        # Non-string, non-primitive types
        return x

    def _load_and_process_row_additions(
        self, table_id: str, snapshot_time: datetime, target_time: datetime
    ) -> pl.DataFrame:
        """
        Load and process row additions into wide Polars DataFrame.

        Only loads additions where snapshot_time < save_time <= target_time.
        """
        q_add = f"""
        WITH latest_additions AS (
            SELECT
                row_key,
                row_data,
                ROW_NUMBER() OVER (PARTITION BY row_key ORDER BY save_time DESC) as rn
            FROM row_additions
            WHERE table_id = '{table_id}'
              AND save_time > TIMESTAMP '{snapshot_time.isoformat()}'
              AND save_time <= TIMESTAMP '{target_time.isoformat()}'
        )
        SELECT row_key, row_data
        FROM latest_additions
        WHERE rn = 1
        ORDER BY row_key
        """
        adds_df = self.conn.execute(q_add).fetchdf()
        if not adds_df.empty:
            add_row_keys = adds_df["row_key"].to_list()
            raw_adds = adds_df["row_data"].fillna("{}").to_list()
            parsed_adds = self._parse_json_batch(raw_adds)

            adds_wide_pdf = pd.json_normalize(parsed_adds)
            if not adds_wide_pdf.empty:
                adds_wide_pdf["row_key"] = add_row_keys
                try:
                    adds_pl = pl.from_pandas(
                        adds_wide_pdf, nan_to_null=False
                    ).with_columns(pl.col("row_key").cast(pl.Utf8))
                except Exception:
                    # Convert object columns to string to avoid type inference issues
                    adds_wide_pdf_str = adds_wide_pdf.copy()
                    for col in adds_wide_pdf_str.columns:
                        if (
                            adds_wide_pdf_str[col].dtype == "object"
                            and col != "row_key"
                        ):
                            adds_wide_pdf_str[col] = adds_wide_pdf_str[col].astype(str)
                    adds_pl = pl.from_pandas(
                        adds_wide_pdf_str, nan_to_null=False
                    ).with_columns(pl.col("row_key").cast(pl.Utf8))
            else:
                adds_pl = pl.DataFrame(
                    {"row_key": pl.Series(add_row_keys, dtype=pl.Utf8)}
                )
        else:
            adds_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})

        return adds_pl

    def _load_deletions(
        self, table_id: str, snapshot_time: datetime, target_time: datetime
    ) -> tuple[set[str], set[str]]:
        """
        Load row and column deletions, considering re-additions.

        Only loads deletions where snapshot_time < event_time <= target_time.
        """
        # Load deleted rows, but exclude those that were re-added later
        q_row_del = f"""
        WITH latest_row_events AS (
            SELECT
                row_key,
                'deletion' as event_type,
                delete_time as event_time
            FROM row_deletions
            WHERE table_id = '{table_id}'
              AND delete_time > TIMESTAMP '{snapshot_time.isoformat()}'
              AND delete_time <= TIMESTAMP '{target_time.isoformat()}'

            UNION ALL

            SELECT
                row_key,
                'addition' as event_type,
                save_time as event_time
            FROM row_additions
            WHERE table_id = '{table_id}'
              AND save_time > TIMESTAMP '{snapshot_time.isoformat()}'
              AND save_time <= TIMESTAMP '{target_time.isoformat()}'
        ),
        latest_per_row AS (
            SELECT
                row_key,
                event_type,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY row_key ORDER BY event_time DESC) as rn
            FROM latest_row_events
        )
        SELECT DISTINCT row_key
        FROM latest_per_row
        WHERE rn = 1 AND event_type = 'deletion'
        """
        row_del_df = self.conn.execute(q_row_del).fetchdf()
        deleted_rows = (
            set(row_del_df["row_key"].astype(str).tolist())
            if not row_del_df.empty
            else set()
        )

        # Enhanced column deletion tracking with proper multi-cycle support
        q_col_del = f"""
        WITH deleted_columns AS (
            SELECT col_key, delete_time as event_time, 'deletion' as event_type
            FROM column_deletions
            WHERE table_id = '{table_id}'
              AND delete_time > TIMESTAMP '{snapshot_time.isoformat()}'
              AND delete_time <= TIMESTAMP '{target_time.isoformat()}'
        ),
        readded_columns AS (
            -- Track column re-additions from dedicated column_additions table
            SELECT DISTINCT
                ca.col_key,
                ca.add_time as event_time,
                'addition' as event_type
            FROM column_additions ca
            WHERE ca.table_id = '{table_id}'
                AND ca.add_time > TIMESTAMP '{snapshot_time.isoformat()}'
                AND ca.add_time <= TIMESTAMP '{target_time.isoformat()}'
                AND EXISTS (
                    SELECT 1 FROM column_deletions cd
                    WHERE cd.table_id = ca.table_id
                        AND cd.col_key = ca.col_key
                        AND cd.delete_time < ca.add_time
                )
        ),
        all_column_events AS (
            SELECT * FROM deleted_columns
            UNION ALL
            SELECT * FROM readded_columns
        ),
        latest_column_events AS (
            SELECT
                col_key,
                event_type,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY col_key ORDER BY event_time DESC) as rn
            FROM all_column_events
        )
        SELECT DISTINCT col_key
        FROM latest_column_events
        WHERE rn = 1 AND event_type = 'deletion'
        """
        col_del_df = self.conn.execute(q_col_del).fetchdf()
        deleted_cols = (
            set(col_del_df["col_key"].astype(str).tolist())
            if not col_del_df.empty
            else set()
        )

        return deleted_rows, deleted_cols

    def _merge_data_layers(
        self,
        base: pl.DataFrame,
        changes: pl.DataFrame,
        row_additions: pl.DataFrame,
        column_additions: dict[str, dict[str, Any]],
    ) -> pl.DataFrame:
        """Merge four data layers: base snapshot <- row additions <- column additions <- cell changes.

        New priority order ensures cell changes (precise modifications) always have highest precedence.
        """

        # Step 1: Start with base snapshot
        merged = base

        # Step 2: Apply row additions (structural additions)
        if not row_additions.is_empty():
            # Check for duplicate row_keys in row_additions
            row_keys_in_additions = (
                row_additions.select("row_key").to_pandas()["row_key"].tolist()
            )
            unique_row_keys = len(set(row_keys_in_additions))
            total_row_keys = len(row_keys_in_additions)
            if unique_row_keys != total_row_keys:
                logger.warning(
                    f"Duplicate row keys detected in row_additions: {unique_row_keys} unique out of {total_row_keys} total"
                )

            merged = merged.join(row_additions, on="row_key", how="full", suffix="_add")

            # Handle row_key from add join
            if "row_key_add" in merged.columns:
                merged = merged.with_columns(
                    pl.coalesce(pl.col("row_key"), pl.col("row_key_add")).alias(
                        "row_key"
                    )
                ).drop("row_key_add")

            # Apply add columns (row additions take precedence over base)
            add_cols = [c for c in row_additions.columns if c != "row_key"]
            for c in add_cols:
                add_col = f"{c}_add"
                if add_col in merged.columns:
                    if c in merged.columns:
                        # Row additions have precedence over base snapshot
                        merged = merged.with_columns(
                            pl.coalesce(pl.col(add_col), pl.col(c)).alias(c)
                        )
                    else:
                        merged = merged.rename({add_col: c})

            # Clean up remaining _add columns
            remaining_adds = [c for c in merged.columns if c.endswith("_add")]
            if remaining_adds:
                merged = merged.drop(remaining_adds)

        # Step 3: Apply column additions (structural additions)
        if column_additions:
            for col_key, col_data in column_additions.items():
                if col_data:  # Skip empty column data
                    # Create a temporary dataframe with the new column
                    col_rows = []
                    for row_key, value in col_data.items():
                        # Convert value to string to ensure consistent typing
                        str_value = str(value) if value is not None else None
                        col_rows.append({"row_key": str(row_key), col_key: str_value})

                    if col_rows:
                        # Create DataFrame with explicit string type to avoid type inference errors
                        col_df = pl.DataFrame(
                            col_rows, schema={"row_key": pl.Utf8, col_key: pl.Utf8}
                        )
                        # Join the new column to the existing data
                        if col_key in merged.columns:
                            # Column already exists, merge values
                            merged = merged.join(
                                col_df, on="row_key", how="left", suffix="_col_add"
                            )
                            # Column additions have precedence over base+row_additions, but will be overridden by cell changes later
                            add_col_name = f"{col_key}_col_add"
                            if add_col_name in merged.columns:
                                merged = merged.with_columns(
                                    pl.coalesce(
                                        pl.col(add_col_name), pl.col(col_key)
                                    ).alias(col_key)
                                ).drop(add_col_name)
                        else:
                            # New column, simple join
                            merged = merged.join(col_df, on="row_key", how="left")

        # Step 4: Apply cell changes (precise modifications) - HIGHEST PRECEDENCE
        if not changes.is_empty():
            merged = merged.join(changes, on="row_key", how="full")

            # Handle row_key from joins
            if "row_key_right" in merged.columns:
                merged = merged.with_columns(
                    pl.coalesce(pl.col("row_key"), pl.col("row_key_right")).alias(
                        "row_key"
                    )
                ).drop("row_key_right")

            # Apply delta columns from pivot (cell changes override everything)
            delta_cols = [c for c in merged.columns if c.endswith("__delta")]
            for delta_col in delta_cols:
                target_col = delta_col[: -len("__delta")]
                if target_col in merged.columns:
                    # Cell changes have absolute precedence - they override everything
                    merged = merged.with_columns(
                        pl.when(pl.col(delta_col).is_not_null())
                        .then(pl.col(delta_col))
                        .otherwise(pl.col(target_col))
                        .alias(target_col)
                    )
                else:
                    merged = merged.rename({delta_col: target_col})

            # Drop all remaining __delta columns
            if delta_cols:
                remaining_deltas = [c for c in merged.columns if c.endswith("__delta")]
                if remaining_deltas:
                    merged = merged.drop(remaining_deltas)

        return merged

    def _load_and_process_column_additions(
        self, table_id: str, snapshot_time: datetime, target_time: datetime
    ) -> dict[str, dict[str, Any]]:
        """
        Load column additions and return as nested dict {col_key: {row_key: value}}.

        Only loads additions where snapshot_time < add_time <= target_time.
        """
        q_col_add = f"""
        WITH latest_column_additions AS (
            SELECT
                col_key,
                col_data_json,
                ROW_NUMBER() OVER (PARTITION BY col_key ORDER BY add_time DESC) as rn
            FROM column_additions
            WHERE table_id = '{table_id}'
              AND add_time > TIMESTAMP '{snapshot_time.isoformat()}'
              AND add_time <= TIMESTAMP '{target_time.isoformat()}'
        )
        SELECT col_key, col_data_json
        FROM latest_column_additions
        WHERE rn = 1
        """
        col_add_df = self.conn.execute(q_col_add).fetchdf()

        column_data = {}
        if not col_add_df.empty:
            for _, row in col_add_df.iterrows():
                col_key = str(row["col_key"])
                try:
                    col_data = orjson.loads(row["col_data_json"])
                except Exception:
                    try:
                        col_data = json.loads(row["col_data_json"])
                    except Exception:
                        col_data = {}
                column_data[col_key] = col_data

        return column_data

    def _finalize_dataframe(
        self,
        merged: pl.DataFrame,
        deleted_rows: Optional[set[str]] = None,
        deleted_cols: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """Convert to pandas and apply final formatting with deletion filtering."""
        if merged.is_empty():
            return pd.DataFrame()

        # Convert to pandas
        result_pdf = merged.to_pandas()

        # Clean up any remaining duplicate row_key columns
        duplicate_cols = [
            col
            for col in result_pdf.columns
            if col.startswith("row_key") and col != "row_key"
        ]
        if duplicate_cols:
            result_pdf = result_pdf.drop(columns=duplicate_cols)

        # Filter out deleted columns before setting index
        if deleted_cols:
            remaining_cols = [
                col for col in result_pdf.columns if col not in deleted_cols
            ]
            if remaining_cols:
                result_pdf = result_pdf[remaining_cols]

        # Set row_key as index with smart sorting
        if "row_key" in result_pdf.columns:
            # Smart sort: numeric if possible, otherwise string
            try:
                numeric_sort_key = pd.to_numeric(result_pdf["row_key"], errors="coerce")
                if not numeric_sort_key.isna().all():
                    result_pdf["_sort_key"] = numeric_sort_key
                    result_pdf = result_pdf.sort_values("_sort_key").drop(
                        "_sort_key", axis=1
                    )
                else:
                    result_pdf = result_pdf.sort_values("row_key")
            except Exception:
                result_pdf = result_pdf.sort_values("row_key")

            result_pdf.set_index("row_key", inplace=True)
            result_pdf.index.name = None

        # Filter out invalid rows and deleted rows
        if not result_pdf.empty:
            result_pdf = result_pdf[result_pdf.index.notna()]

            # Filter out deleted rows
            if deleted_rows:
                remaining_rows = [
                    idx for idx in result_pdf.index if str(idx) not in deleted_rows
                ]
                if remaining_rows:
                    result_pdf = result_pdf.loc[remaining_rows]
                else:
                    result_pdf = pd.DataFrame()

        return result_pdf

    def compact_up_to(
        self,
        table_id: str,
        cutoff_time: datetime,
        new_snapshot_time: Optional[datetime] = None,
    ) -> None:
        """Compact history up to cutoff_time into a new snapshot."""
        if new_snapshot_time is None:
            new_snapshot_time = datetime.now()

        df = self.reconstruct_as_of(table_id, cutoff_time)
        self.save_snapshot(table_id, df, new_snapshot_time)

        # Delete compacted changes
        self.conn.execute(f"""
            DELETE FROM cell_changes
            WHERE table_id = '{table_id}' AND save_time <= TIMESTAMP '{cutoff_time.isoformat()}';
        """)
        self.conn.execute(f"""
            DELETE FROM row_additions
            WHERE table_id = '{table_id}' AND save_time <= TIMESTAMP '{cutoff_time.isoformat()}';
        """)

    # =================== Legacy API Compatibility Methods ===================

    def _apply_dtypes_to_result(
        self, result: pd.DataFrame, key: str, target_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Apply saved dtypes to reconstructed DataFrame.

        Args:
            result: DataFrame to apply dtypes to (modified in place)
            key: Dataset key to load dtype mapping for
            target_time: Target time point for dtype lookup
        """
        dtype_mapping = self._get_dtype_mapping_at_time(key, target_time)
        if not dtype_mapping or "dtypes" not in dtype_mapping:
            return result

        dtypes = dtype_mapping["dtypes"]
        categorical_metadata = dtype_mapping.get("categorical_metadata", {})

        # Apply saved dtypes to columns
        for col, dtype_str in dtypes.items():
            if col in result.columns and dtype_str and dtype_str != "None":
                logger.debug(
                    f"Applying dtype '{dtype_str}' to column '{col}', current dtype: {result[col].dtype}"
                )
                try:
                    # Pre-processing: If column is object type and can be parsed as datetime, convert it first
                    # This prevents "2018-10-01 00:00:00" -> float conversion from producing NaN
                    if result[col].dtype == "object":
                        try:
                            # Filter out "NaN" strings and null values before datetime conversion
                            mask = (
                                (result[col] != "NaN")
                                & (result[col] != "")
                                & result[col].notna()
                            )
                            # Only convert if ALL values are valid datetime (no "NaN" strings)
                            if mask.all():
                                # Convert entire column to datetime64
                                result[col] = result[col].astype(
                                    pd.api.types.pandas_dtype("datetime64[ns]")
                                )
                            # elif mask.any():
                            #     # Has "NaN" strings - convert only non-NaN values, keep dtype as object
                            #     datetime_values = result[col][mask].astype(pd.api.types.pandas_dtype('datetime64[ns]'))
                            #     result.loc[mask, col] = datetime_values
                        except Exception:
                            pass  # Not a datetime string, continue with original logic

                    # Since values are stored as strings, convert them back
                    if "int" in dtype_str:
                        # Convert string values to numeric first, then to target int type
                        try:
                            numeric_col = pd.to_numeric(result[col], errors="coerce")
                            # Check if there are any NaN values that would cause integer conversion to fail
                            if numeric_col.isna().any():
                                # Keep as object if there are unconvertible values
                                pass
                            else:
                                result[col] = numeric_col.astype(dtype_str)
                        except (ValueError, TypeError) as e:
                            logger.debug(
                                f"Failed to convert column '{col}' to {dtype_str}: {e}"
                            )
                            # Keep original dtype if conversion fails
                    elif "float" in dtype_str:
                        try:
                            result[col] = pd.to_numeric(
                                result[col], errors="coerce"
                            ).astype(dtype_str)
                        except (ValueError, TypeError) as e:
                            logger.debug(
                                f"Failed to convert column '{col}' to {dtype_str}: {e}"
                            )
                            # Keep original dtype if conversion fails
                    elif "bool" in dtype_str:
                        # Handle both boolean values and string boolean values
                        if result[col].dtype == "bool":
                            # Already boolean, no conversion needed
                            pass
                        else:
                            # Convert string boolean values back to bool
                            result[col] = (
                                result[col]
                                .map(
                                    {
                                        "True": True,
                                        "False": False,
                                        True: True,
                                        False: False,
                                    }
                                )
                                .astype("bool")
                            )
                    elif dtype_str == "category":
                        # Handle categorical dtypes with proper metadata restoration
                        if col in categorical_metadata:
                            cat_meta = categorical_metadata[col]
                            categories = cat_meta["categories"]
                            ordered = cat_meta.get("ordered", False)
                            categories_dtype = cat_meta.get(
                                "categories_dtype", "object"
                            )

                            # Convert categories to proper dtype if specified
                            if categories_dtype != "object":
                                try:
                                    categories = pd.Index(categories).astype(
                                        categories_dtype
                                    )
                                except (ValueError, TypeError):
                                    categories = pd.Index(categories)
                            else:
                                categories = pd.Index(categories)

                            try:
                                # Check if all values in result[col] are in categories
                                # Convert unique values to match categories dtype for proper comparison
                                # Filter out None, empty strings, and string representations of null
                                series = pd.Series(result[col])
                                # Remove actual None/NaN and string representations like "nan", "None", ""
                                mask = (
                                    series.notna()
                                    & (series != "")
                                    & (series != "nan")
                                    & (series != "None")
                                    & (series != "NaN")
                                )
                                unique_values_raw = series[mask].unique()

                                if categories_dtype != "object":
                                    try:
                                        # Convert to same dtype as categories to avoid type mismatch
                                        # e.g., string "1.0" vs float 1.0
                                        unique_values = pd.Index(
                                            unique_values_raw
                                        ).astype(categories_dtype)
                                    except (ValueError, TypeError):
                                        # If conversion fails, use raw values
                                        unique_values = unique_values_raw
                                else:
                                    unique_values = unique_values_raw

                                missing_categories = set(unique_values) - set(
                                    categories
                                )

                                if missing_categories:
                                    # Extend categories to include missing values
                                    extended_categories = list(categories) + list(
                                        missing_categories
                                    )
                                    categories = pd.Index(extended_categories)

                                # Convert result[col] to match categories dtype before creating Categorical
                                # This prevents type mismatch (e.g., string "1.0" vs float 1.0)
                                if categories_dtype != "object":
                                    try:
                                        result[col] = pd.Series(result[col]).astype(
                                            categories_dtype
                                        )
                                    except (ValueError, TypeError) as e:
                                        logger.debug(
                                            f"Failed to convert column '{col}' to {categories_dtype}: {e}"
                                        )
                                        # Keep as-is if conversion fails

                                result[col] = pd.Categorical(
                                    result[col], categories=categories, ordered=ordered
                                )
                            except (ValueError, TypeError) as e:
                                logger.debug(
                                    f"Failed to restore categorical for column '{col}': {e}"
                                )
                                # Fallback to simple categorical conversion
                                result[col] = result[col].astype("category")
                        else:
                            # Fallback to simple categorical conversion
                            result[col] = result[col].astype("category")
                    elif dtype_str == "string" or dtype_str == "object":
                        # Keep as string
                        result[col] = result[col].replace("<NA>", pd.NA)
                        result[col] = result[col].astype(dtype_str)
                    else:
                        # Try direct conversion
                        try:
                            target_dtype = pd.api.types.pandas_dtype(dtype_str)
                            result[col] = result[col].astype(target_dtype)
                        except (ValueError, TypeError):
                            pass
                except (ValueError, TypeError, Exception) as e:
                    # Fallback: try to convert from string
                    logger.debug(
                        f"First dtype conversion failed for column '{col}': {e}"
                    )
                    try:
                        if "int" in dtype_str:
                            numeric_col = pd.to_numeric(result[col], errors="coerce")
                            # Check if there are any NaN values that would cause integer conversion to fail
                            if not numeric_col.isna().any():
                                result[col] = numeric_col.astype(dtype_str)
                        elif "float" in dtype_str:
                            try:
                                result[col] = pd.to_numeric(
                                    result[col], errors="coerce"
                                ).astype(dtype_str)
                            except (ValueError, TypeError):
                                pass
                        elif "bool" in dtype_str:
                            result[col] = (
                                result[col]
                                .map({"True": True, "False": False})
                                .fillna(False)
                                .astype("bool")
                            )
                    except (ValueError, TypeError) as e2:
                        logger.debug(
                            f"Fallback dtype conversion also failed for column '{col}': {e2}, keeping original dtype {result[col].dtype}"
                        )

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

        # Store frequency info before any index modifications
        index_freq = dtype_mapping.get("index_freq")

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

        # Apply saved index name
        index_name = dtype_mapping.get("index_name")
        if index_name is not None:
            result.index.name = index_name

        # Apply saved columns name
        columns_name = dtype_mapping.get("columns_name")
        if columns_name is not None:
            result.columns.name = columns_name

        # Apply saved index order
        index_order = dtype_mapping.get("index_order")
        if index_order is not None and set(index_order) == set(result.index):
            result = result.loc[index_order]

        # Apply saved columns order
        columns_order = dtype_mapping.get("columns_order")
        if columns_order is not None and set(columns_order) == set(result.columns):
            # Reorder columns to match the saved order
            result = result[columns_order]

        # Apply saved frequency to index (for DatetimeIndex) - do this after reordering
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
                    # Preserve index name when creating new DatetimeIndex
                    original_name = result.index.name
                    result.index = pd.DatetimeIndex(
                        result.index.values, freq=offset, name=original_name
                    )
                except Exception:
                    # Fallback: try using asfreq on a temporary Series
                    try:
                        tmp = pd.Series([None] * len(result), index=result.index)
                        tmp = tmp.asfreq(index_freq)
                        # Preserve index name when using asfreq fallback
                        original_name = result.index.name
                        result.index = tmp.index
                        result.index.name = original_name
                    except Exception:
                        # If we still can't set freq, continue without raising
                        pass
            except (ValueError, TypeError, Exception) as e:
                # Fallback: keep original index frequency
                logger.debug(
                    f"Failed to set index frequency to {index_freq}: {e}, keeping original"
                )
                pass

        return result

    def exists(self, key: str) -> bool:
        """
        Check if cache exists for a dataset.

        Args:
            key: Dataset key

        Returns:
            True if cache exists
        """
        raw_data = self.load_raw_data(key)
        return raw_data is not None

    def save_data(
        self, key: str, data: pd.DataFrame, timestamp: datetime
    ) -> Optional[ChangeResult]:
        """
        Save DataFrame to cache with timestamp.

        Args:
            key: Dataset key
            data: DataFrame to save
            timestamp: Save timestamp

        Returns:
            ChangeResult if incremental save occurred, None for initial save
        """
        if data.empty:
            logger.warning(f"Attempting to save empty DataFrame for {key}")
            return None

        changes = None
        dtype_changed = False

        # Check if this is the first save (no existing data)
        existing_data = self.load_raw_data(key)

        if existing_data is None or existing_data.empty:
            # First save - create snapshot
            self.save_snapshot(key, data, timestamp)
            logger.debug(f"Created initial snapshot for {key} with {len(data)} rows")
        else:
            # Check if dtype has changed - if so, force snapshot
            dtype_changed = self._check_dtype_changed(key, data)

            if dtype_changed:
                # Dtype changed - create new snapshot and return ChangeResult with dtype_changed flag
                self.save_snapshot(key, data, timestamp)
                logger.info(f"Dtype changed for {key}, created new snapshot")
                # Create ChangeResult with dtype_changed flag
                changes = ChangeResult(
                    cell_changes=pd.DataFrame(),
                    row_additions=pd.DataFrame(),
                    row_deletions=pd.DataFrame(),
                    column_additions=pd.DataFrame(),
                    column_deletions=pd.DataFrame(),
                    meta={"dtype_changed": True},
                    dtype_changed=True,
                )
            else:
                # Incremental save - compute diff and save changes
                prev_data = self.load_data(key)
                if not prev_data.empty:
                    changes = self.save_version(key, prev_data, data, timestamp)
                    logger.debug(f"Saved incremental changes for {key}")
                else:
                    # Fallback to snapshot if reconstruction failed
                    self.save_snapshot(key, data, timestamp)
                    logger.debug(f"Created fallback snapshot for {key}")

        # Calculate and save data hash with current time (separate from data timestamp)
        data_hash = self._compute_dataframe_hash(data)
        hash_timestamp = datetime.now()  # Use current time, not data timestamp
        self._save_data_hash(key, data_hash, hash_timestamp)

        # Save dtype mapping after data is successfully saved (maintain consistency)
        self._save_dtype_mapping(key, data, timestamp)

        return changes

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
        try:
            if as_of_time is None:
                # Get the latest timestamp for this table (from DuckDB)
                q_latest = f"""
                SELECT MAX(latest_time) as max_time FROM (
                    SELECT MAX(save_time) as latest_time FROM cell_changes WHERE table_id = '{key}'
                    UNION ALL
                    SELECT MAX(save_time) as latest_time FROM row_additions WHERE table_id = '{key}'
                    UNION ALL
                    SELECT MAX(snapshot_time) as latest_time FROM rows_base WHERE table_id = '{key}'
                ) combined
                """
                latest_result = self.conn.execute(q_latest).fetchone()
                db_latest_time = (
                    latest_result[0] if latest_result and latest_result[0] else None
                )

                # Also check dtype mapping file for the latest timestamp
                dtype_mapping = self._load_dtype_mapping(key)
                dtype_latest_time = None
                if dtype_mapping and "dtype_history" in dtype_mapping:
                    # Get the latest timestamp from dtype_history
                    dtype_history = dtype_mapping["dtype_history"]
                    if dtype_history:
                        last_entry = dtype_history[-1]
                        dtype_latest_time = datetime.fromisoformat(
                            last_entry["timestamp"]
                        )

                # Use the maximum of both
                if db_latest_time and dtype_latest_time:
                    as_of_time = max(db_latest_time, dtype_latest_time)
                elif db_latest_time:
                    as_of_time = db_latest_time
                elif dtype_latest_time:
                    as_of_time = dtype_latest_time
                else:
                    as_of_time = datetime.now()

            # Use new DuckDB reconstruction
            result = self.reconstruct_as_of(key, as_of_time)

            return result

        except Exception as e:
            logger.error(f"Failed to load data for {key}: {e}")
            return pd.DataFrame()

    def load_raw_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get timestamps for data that exists for this key in DuckDB.

        Args:
            key: Dataset key

        Returns:
            DataFrame with save_time column if data exists, None if not found
        """
        try:
            # Get all timestamps for this key
            q_timestamps = f"""
            SELECT DISTINCT save_time FROM (
                SELECT snapshot_time as save_time FROM rows_base WHERE table_id = '{key}'
                UNION ALL
                SELECT save_time FROM cell_changes WHERE table_id = '{key}'
                UNION ALL
                SELECT save_time FROM row_additions WHERE table_id = '{key}'
            ) combined
            ORDER BY save_time
            """
            result = self.conn.execute(q_timestamps).fetchall()

            if result:
                # Return DataFrame with save_time column for compatibility
                timestamps = [row[0] for row in result]
                return pd.DataFrame({"save_time": timestamps})
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to load raw data for {key}: {e}")
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
        try:
            # Delete from all DuckDB tables
            self.conn.execute(f"DELETE FROM rows_base WHERE table_id = '{key}'")
            self.conn.execute(f"DELETE FROM cell_changes WHERE table_id = '{key}'")
            self.conn.execute(f"DELETE FROM row_additions WHERE table_id = '{key}'")
            self.conn.execute(f"DELETE FROM data_hashes WHERE table_id = '{key}'")

            # Also clear dtype mapping file for compatibility
            dtype_path = self._get_dtype_path(key)
            if dtype_path.exists():
                dtype_path.unlink()

        except Exception as e:
            logger.error(f"Failed to clear cache for {key}: {e}")

    def clear_all(self) -> None:
        """Clear all cache data."""
        try:
            # Clear all DuckDB tables
            self.conn.execute("DELETE FROM rows_base")
            self.conn.execute("DELETE FROM cell_changes")
            self.conn.execute("DELETE FROM row_additions")
            self.conn.execute("DELETE FROM data_hashes")

            # Clear all dtype mapping files
            if self.cache_dir.exists():
                for dtype_file in self.cache_dir.glob("*_dtypes.json"):
                    dtype_file.unlink()

        except Exception as e:
            logger.error(f"Failed to clear all cache: {e}")

    def get_change_history(self, key: str) -> pd.DataFrame:
        """
        Get change history for a dataset.

        Args:
            key: Dataset key

        Returns:
            DataFrame with change history
        """
        try:
            # Get history from cell_changes table
            q_history = f"""
            SELECT row_key, col_key,
                   COUNT(*) as change_count,
                   MIN(save_time) as first_change,
                   MAX(save_time) as last_change
            FROM cell_changes
            WHERE table_id = '{key}'
            GROUP BY row_key, col_key
            ORDER BY last_change DESC
            """

            history: pd.DataFrame = self.conn.execute(q_history).fetchdf()

            if history.empty:
                return pd.DataFrame()

            return history

        except Exception as e:
            logger.error(f"Failed to get change history for {key}: {e}")
            return pd.DataFrame()

    def get_storage_info(self, key: Optional[str] = None) -> dict[str, Any]:
        """
        Get storage information.

        Args:
            key: Specific dataset key or None for all

        Returns:
            Storage information dictionary
        """
        info: dict[str, Any] = {}

        try:
            if key:
                # Info for specific key
                q_info = f"""
                SELECT
                    (SELECT COUNT(*) FROM rows_base WHERE table_id = '{key}') as snapshot_rows,
                    (SELECT COUNT(*) FROM cell_changes WHERE table_id = '{key}') as cell_changes,
                    (SELECT COUNT(*) FROM row_additions WHERE table_id = '{key}') as row_additions,
                    (SELECT MAX(save_time) FROM (
                        SELECT save_time FROM cell_changes WHERE table_id = '{key}'
                        UNION ALL
                        SELECT save_time FROM row_additions WHERE table_id = '{key}'
                        UNION ALL
                        SELECT snapshot_time as save_time FROM rows_base WHERE table_id = '{key}'
                    )) as last_modified
                """
                result = self.conn.execute(q_info).fetchone()
                if result and any(result[:3]):  # Check if any count > 0
                    info[key] = {
                        "snapshot_rows": result[0] or 0,
                        "cell_changes": result[1] or 0,
                        "row_additions": result[2] or 0,
                        "last_modified": result[3],
                        "storage_type": "duckdb",
                    }
            else:
                # Info for all keys
                q_all = """
                SELECT table_id,
                       COUNT(*) as total_records,
                       'mixed' as record_type
                FROM (
                    SELECT table_id FROM rows_base
                    UNION ALL
                    SELECT table_id FROM cell_changes
                    UNION ALL
                    SELECT table_id FROM row_additions
                ) combined
                GROUP BY table_id
                ORDER BY table_id
                """
                results = self.conn.execute(q_all).fetchall()

                for table_id, record_count, _ in results:
                    info[table_id] = {
                        "total_records": record_count,
                        "storage_type": "duckdb",
                    }

                # Get DuckDB file size
                db_path = self.cache_dir / "cache.duckdb"
                if db_path.exists():
                    info["database_file_size"] = db_path.stat().st_size

        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")

        return info
