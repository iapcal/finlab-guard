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
from tqdm import tqdm

from .codec import DataTypeCodec

logger = logging.getLogger(__name__)


def _to_json_str(obj: Any, dtypes: Optional[dict[str, str]] = None) -> str:
    """Convert object to JSON string using DataTypeCodec (JSON Schema v2).

    This function now delegates to DataTypeCodec for consistent serialization.
    Supports both single values and row dictionaries.

    Args:
        obj: Value or dict to serialize
        dtypes: Optional dtype mapping for row dict (column_name -> dtype string)

    Returns:
        JSON string in Schema v2 format

    Note:
        - For dict objects: Uses encode_row() to serialize entire row
        - For single values: Uses encode() for individual value
        - Minimal fallbacks (< 3), all documented in DataTypeCodec
    """
    codec = DataTypeCodec()

    if isinstance(obj, dict):
        # Serialize entire row (dict of column -> value)
        return codec.encode_row(obj, dtypes=dtypes)
    else:
        # Serialize single value
        # Extract dtype if available from dtypes dict
        dtype = None
        if dtypes and len(dtypes) == 1:
            dtype = list(dtypes.values())[0]
        return codec.encode(obj, dtype=dtype)


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

        # Initialize codec for serialization/deserialization
        self.codec = DataTypeCodec()

        # Initialize DuckDB connection
        db_path = cache_dir / "cache.duckdb"
        self.conn = duckdb.connect(str(db_path))
        self._setup_tables()

    def close(self) -> None:
        """Close the DuckDB connection."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
            self.conn = None  # type: ignore[assignment,unused-ignore]

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
        with open(dtype_path, "w", encoding="utf-8") as f:
            # Use standard JSON for dtype mapping (not DataFrame data)
            # dtype_mapping is just metadata, doesn't need JSON Schema v2
            json.dump(dtype_mapping, f, ensure_ascii=False, indent=2)

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
            with open(dtype_path, encoding="utf-8") as f:
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

    def _insert_snapshot_chunk(
        self, table_id: str, chunk_df: pd.DataFrame, snapshot_time: datetime
    ) -> None:
        """Insert a chunk of DataFrame into rows_base table.

        Helper function to avoid code duplication in save_snapshot.
        """
        row_keys = [str(idx) for idx in chunk_df.index]
        records = chunk_df.to_dict("records")

        # Extract dtype information for encoding
        dtypes_dict = {col: str(chunk_df[col].dtype) for col in chunk_df.columns}

        # Serialize each row with dtype information
        row_data_list = [_to_json_str(record, dtypes=dtypes_dict) for record in records]

        tmp = pd.DataFrame(
            {
                "table_id": table_id,
                "row_key": row_keys,
                "row_data": row_data_list,
                "snapshot_time": snapshot_time,
            }
        )

        self.conn.register("_tmp_snapshot", tmp)
        self.conn.execute("INSERT INTO rows_base SELECT * FROM _tmp_snapshot")
        self.conn.unregister("_tmp_snapshot")

    def save_snapshot(
        self, table_id: str, df: pd.DataFrame, snapshot_time: Optional[datetime] = None
    ) -> None:
        """Save a complete DataFrame snapshot to DuckDB.

        Optimized for large DataFrames by using chunked processing to avoid memory explosion.
        For DataFrames with >10M cells, processes in chunks with progress tracking.
        """
        if snapshot_time is None:
            snapshot_time = datetime.now()

        if df.empty:
            return

        total_cells = df.shape[0] * df.shape[1]
        chunk_size = 10000  # Process 10k rows at a time

        # Detect huge DataFrames and use chunked processing
        if total_cells > 100_000_000:  # > 100M cells
            num_rows = len(df)
            num_chunks = (num_rows + chunk_size - 1) // chunk_size

            logger.info(
                f"Processing large DataFrame: {num_rows:,} rows × {df.shape[1]} cols = {total_cells:,} cells"
            )
            logger.info(
                f"Using chunked processing: {num_chunks} chunks of {chunk_size} rows"
            )

            # Process in chunks with progress bar
            for chunk_idx in tqdm(range(num_chunks), desc="Saving snapshot chunks"):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, num_rows)

                # Extract and insert chunk
                chunk_df = df.iloc[start_idx:end_idx]
                self._insert_snapshot_chunk(table_id, chunk_df, snapshot_time)
        else:
            # Small/medium DataFrames: process all at once
            self._insert_snapshot_chunk(table_id, df, snapshot_time)

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
                # Fallback #1 (comparison): Test if Polars can execute basic comparison
                # Trigger: Type mismatch or unsupported operations in Polars
                can_use_basic = True
                try:
                    joined.select(basic_mask).limit(1)
                except Exception:
                    can_use_basic = False
                    logger.debug(f"Basic comparison failed for column {col}, trying numeric fallback")

                if can_use_basic:
                    # Use DataTypeCodec for serialization (JSON Schema v2)
                    codec = DataTypeCodec()

                    def smart_serialize(x: Any) -> Optional[str]:
                        """Serialize value using DataTypeCodec (JSON Schema v2).

                        Replaces old marker-based serialization with structured format.
                        This function is used in Polars map_elements() pipeline.
                        """
                        # Delegate to codec for consistent serialization
                        return codec.encode(x, dtype=None)

                    df_changed = (
                        joined.filter(basic_mask)
                        .select(
                            [
                                pl.col("__row_key__").alias("row_key"),
                                # Use DataTypeCodec serialization (v2 format)
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
                    # This handles cases where direct comparison fails due to type mismatch
                    old_numeric = pl.col(old_col).cast(pl.Float64, strict=False)
                    new_numeric = pl.col(new_col).cast(pl.Float64, strict=False)

                    # Check if numeric conversion works
                    # Fallback #2 (comparison): Test if numeric casting works
                    # Trigger: Basic comparison failed, trying float64 conversion
                    try:
                        joined.select(old_numeric, new_numeric).limit(1)
                        numeric_mask = (
                            ~(pl.col(old_col).is_null() & pl.col(new_col).is_null())
                        ) & (old_numeric != new_numeric)

                        # Use codec for float serialization
                        codec = DataTypeCodec()

                        def serialize_float(x: Any) -> Optional[str]:
                            """Serialize float value using codec."""
                            if x is None:
                                return codec.encode(None, dtype="float64")
                            return codec.encode(x, dtype="float64")

                        df_changed = (
                            joined.filter(numeric_mask)
                            .select(
                                [
                                    pl.col("__row_key__").alias("row_key"),
                                    # Use codec for high-precision float encoding
                                    old_numeric.map_elements(
                                        serialize_float,
                                        return_dtype=pl.Utf8,
                                    ).alias("old"),
                                    new_numeric.map_elements(
                                        serialize_float,
                                        return_dtype=pl.Utf8,
                                    ).alias("new"),
                                ]
                            )
                            .with_columns(pl.lit(col).alias("col_key"))
                        )
                    except Exception as e:
                        # Fallback #3 (comparison): Final fallback to string comparison
                        # Trigger: Numeric conversion also failed
                        logger.debug(f"Numeric comparison failed for column {col}: {e}, using string comparison")
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
            except Exception as e:
                # Fallback #4 (comparison): Top-level fallback to string comparison
                # Trigger: All previous comparison methods failed
                logger.debug(f"All comparisons failed for column {col}: {e}, final string fallback")
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
        # Extract dtype information from current DataFrame for encoding
        cur_dtypes = {col: str(cur[col].dtype) for col in cur.columns}

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
            row_adds.append((str(r), _to_json_str(clean_row_dict, dtypes=cur_dtypes), timestamp))

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
                # The "new" value is already serialized by smart_serialize (JSON Schema v2)
                # Just pass it through without double-serialization
                newv = row["new"]
                cell_rows.append((rk, str(ck), newv, timestamp))

            # Big rows become partial row maps stored in row_additions
            # For big rows, values are already serialized by smart_serialize (JSON Schema v2)
            # We need to wrap them in a simple JSON dict, not double-encode
            for br in big_rows:
                subset = all_changes_pdf[all_changes_pdf["row_key"] == br]
                row_map: dict[str, str] = {}  # Map col_key -> serialized value (already JSON Schema v2)
                for _, r in subset.iterrows():  # type: ignore[assignment]
                    new_val = r["new"]  # type: ignore[index] - Already serialized by smart_serialize
                    col_key_val = r["col_key"]  # type: ignore[index]
                    row_map[str(col_key_val)] = new_val
                # Use standard JSON (not codec) since values are already encoded
                # This creates: {"col1": "JSON-v2-string", "col2": "JSON-v2-string"}
                row_map_json = json.dumps(row_map, ensure_ascii=False)
                row_adds.append((str(br), row_map_json, timestamp))

        # New columns: store as column_additions with full column data
        new_cols = [c for c in cur_cols if c not in prev_cols]
        col_additions = []
        if new_cols:
            for c in new_cols:
                # Extract the full column data as a dictionary {row_key: value}
                col_data = {}
                col_dtype = str(cur[c].dtype)  # Get dtype for this column

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

                # Create dtype dict for this column's values
                col_dtypes_dict = {key: col_dtype for key in col_data.keys()}
                col_additions.append((str(c), _to_json_str(col_data, dtypes=col_dtypes_dict), timestamp))

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
        """Parse JSON strings and decode JSON Schema v2 format.

        Handles three cases:
        1. JSON Schema v2 full row: {"v":2,"d":{"col1":{...},"col2":{...}}}
        2. Dict with nested JSON strings: {"col1":"JSON-v2","col2":"JSON-v2"}
        3. Legacy format: direct dict {col1: val1, ...}
        """
        parsed = []
        for s in json_strings:
            if not s:
                parsed.append({})
                continue

            # Parse JSON string
            try:
                obj = orjson.loads(s)
            except Exception:
                try:
                    obj = json.loads(s)
                except Exception:
                    parsed.append({})
                    continue

            # Case 1: JSON Schema v2 full row format
            if isinstance(obj, dict) and obj.get("v") == 2 and "d" in obj:
                # Decode using codec
                decoded_row = self.codec.decode_row(s)
                parsed.append(decoded_row)
            # Case 2: Dict with values being JSON Schema v2 strings (for big rows)
            elif isinstance(obj, dict) and obj:
                # Check if first value looks like JSON Schema v2
                first_val = next(iter(obj.values()), None)
                if isinstance(first_val, str) and first_val.startswith('{"v":2'):
                    # Decode each value
                    decoded_dict = {}
                    for k, v in obj.items():
                        if isinstance(v, str):
                            decoded_dict[k] = self.codec.decode(v)
                        else:
                            decoded_dict[k] = v
                    parsed.append(decoded_dict)
                else:
                    # Case 3: Legacy format - use as is
                    parsed.append(obj)
            else:
                parsed.append(obj)

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

        IMPORTANT: Filters cell_changes by row and column lifecycles to prevent
        stale changes from affecting re-added rows/columns.

        Key insight: A cell_change is valid only if it occurred during a time
        when BOTH its row and column were "alive" (existed and not deleted).
        """
        q_changes_latest = f"""
        WITH valid_cell_changes AS (
            SELECT DISTINCT
                cc.row_key,
                cc.col_key,
                cc.value,
                cc.save_time
            FROM cell_changes cc
            WHERE cc.table_id = '{table_id}'
              AND cc.save_time > TIMESTAMP '{snapshot_time.isoformat()}'
              AND cc.save_time <= TIMESTAMP '{target_time.isoformat()}'

              -- Row lifecycle check: cell_change is valid if row was NOT deleted
              -- in the interval (save_time, target_time]
              AND NOT EXISTS (
                  SELECT 1 FROM row_deletions rd
                  WHERE rd.table_id = cc.table_id
                    AND rd.row_key = cc.row_key
                    AND rd.delete_time > cc.save_time
                    AND rd.delete_time <= TIMESTAMP '{target_time.isoformat()}'
              )

              -- Column lifecycle check: cell_change is valid if column was NOT deleted
              -- in the interval (save_time, target_time]
              AND NOT EXISTS (
                  SELECT 1 FROM column_deletions cd
                  WHERE cd.table_id = cc.table_id
                    AND cd.col_key = cc.col_key
                    AND cd.delete_time > cc.save_time
                    AND cd.delete_time <= TIMESTAMP '{target_time.isoformat()}'
              )
        )
        SELECT row_key, col_key, value FROM (
            SELECT row_key, col_key, value,
                row_number() OVER (PARTITION BY row_key, col_key ORDER BY save_time DESC) as rn
            FROM valid_cell_changes
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
        """Parse legacy type marker strings (backward compatibility).

        LEGACY SUPPORT: This function handles old marker-based serialization format.
        New data uses JSON Schema v2 format (see codec.py).

        Supported markers:
        - __INT__123__ → int
        - __BOOL__True__ / __BOOL__False__ → bool
        - __FLOAT__1.5__ / __HIGHPREC_FLOAT__1.5__ → float
        - __NAN__ → np.nan
        - __PD_NA__ → pd.NA
        - __NONE__ → None

        Args:
            s: String to check for markers

        Returns:
            Parsed value if a valid marker is found, None otherwise.

        Note:
            This function can be removed once all legacy cache data is migrated to v2 format.
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
        """Parse JSON value with support for JSON Schema v2 and legacy formats.

        Priority order:
        1. JSON Schema v2: {"v":2,"val":...,"type":...} → uses codec.decode()
        2. Legacy marker strings: __INT__123__, __BOOL__True__ → uses _parse_marker_string()
        3. JSON-encoded strings → orjson.loads()
        4. Primitive types and plain strings → returned as-is

        Args:
            x: Value to parse (any type)

        Returns:
            Parsed value with appropriate type
        """
        # Fast path: primitive types (already correct)
        if x is None:
            return None
        if isinstance(x, (int, float, bool)):
            return x

        # String parsing (most common case)
        if isinstance(x, str):
            # Priority 1: JSON Schema v2 format (new format)
            if x.startswith('{"v":2'):
                try:
                    return self.codec.decode(x)
                except Exception:
                    # Fallback #6 (parse): Continue to legacy parsing if codec fails
                    # Trigger: Malformed JSON Schema v2 string
                    pass

            # Priority 2: Legacy marker strings (backward compatibility)
            marker_result = self._parse_marker_string(x)
            if marker_result is not None:
                return marker_result

            # Priority 3: Try JSON parsing for nested JSON strings
            try:
                parsed = orjson.loads(x) if x else None

                # If parsed is still a string, check for markers again
                if isinstance(parsed, str):
                    marker_result = self._parse_marker_string(parsed)
                    if marker_result is not None:
                        return marker_result

                return parsed

            except Exception:
                # Fallback #7 (parse): Return original string if not JSON
                # Trigger: Not a valid JSON string (e.g., plain text)
                # No type guessing - explicit is better than implicit
                return x

        # Non-string, non-primitive types (pass through)
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
        deleted_rows: set[str] = (
            {str(x) for x in row_del_df["row_key"].tolist()}
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
        deleted_cols: set[str] = (
            {str(x) for x in col_del_df["col_key"].tolist()}
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
                col_data_json_str = str(row["col_data_json"])

                # Parse JSON and check if it's JSON Schema v2 format
                try:
                    obj = orjson.loads(col_data_json_str)
                except Exception:
                    try:
                        obj = json.loads(col_data_json_str)
                    except Exception:
                        column_data[col_key] = {}
                        continue

                # Decode if JSON Schema v2
                if isinstance(obj, dict) and obj.get("v") == 2 and "d" in obj:
                    # Full row JSON Schema v2 format
                    col_data = self.codec.decode_row(col_data_json_str)
                elif isinstance(obj, dict):
                    # Check if values are JSON Schema v2 strings
                    first_val = next(iter(obj.values()), None) if obj else None
                    if isinstance(first_val, str) and first_val.startswith('{"v":2'):
                        # Decode each value
                        col_data = {}
                        for k, v in obj.items():
                            if isinstance(v, str):
                                col_data[k] = self.codec.decode(v)
                            else:
                                col_data[k] = v
                    else:
                        # Legacy format
                        col_data = obj
                else:
                    col_data = obj

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
        """Apply saved metadata (index/columns properties) to reconstructed DataFrame.

        Note: Since Phase 2, column dtypes are already correct from JSON Schema v2 decoding.
        This function only handles:
        1. Categorical metadata restoration (requires categories info)
        2. Index/columns dtype, name, order
        3. DatetimeIndex frequency

        Args:
            result: DataFrame to apply metadata to
            key: Dataset key to load dtype mapping for
            target_time: Target time point for dtype lookup

        Returns:
            DataFrame with metadata applied
        """
        dtype_mapping = self._get_dtype_mapping_at_time(key, target_time)
        if not dtype_mapping or "dtypes" not in dtype_mapping:
            return result

        dtypes = dtype_mapping["dtypes"]
        categorical_metadata = dtype_mapping.get("categorical_metadata", {})

        # Apply column dtypes
        # Note: Most dtypes are already correct from codec decoding (JSON Schema v2),
        # but we still need to handle edge cases and ensure consistency
        for col, dtype_str in dtypes.items():
            if col not in result.columns or not dtype_str or dtype_str == "None":
                continue

            # Categorical dtype requires special handling (metadata restoration)
            if dtype_str == "category":
                if col in categorical_metadata:
                    cat_meta = categorical_metadata[col]
                    categories = cat_meta["categories"]
                    ordered = cat_meta.get("ordered", False)
                    categories_dtype = cat_meta.get("categories_dtype", "object")

                    # Convert categories to proper dtype if specified
                    if categories_dtype != "object":
                        try:
                            categories = pd.Index(categories).astype(categories_dtype)
                        except (ValueError, TypeError):
                            categories = pd.Index(categories)
                    else:
                        categories = pd.Index(categories)

                    try:
                        # Filter out null-like values
                        series = pd.Series(result[col])
                        mask = (
                            series.notna()
                            & (series != "")
                            & (series != "nan")
                            & (series != "None")
                            & (series != "NaN")
                        )
                        unique_values_raw = series[mask].unique()

                        # Convert unique values to match categories dtype
                        if categories_dtype != "object":
                            try:
                                unique_values = pd.Index(unique_values_raw).astype(
                                    categories_dtype
                                )
                            except (ValueError, TypeError):
                                unique_values = unique_values_raw
                        else:
                            unique_values = unique_values_raw

                        # Extend categories if needed
                        missing_categories = set(unique_values) - set(categories)
                        if missing_categories:
                            categories = pd.Index(
                                list(categories) + list(missing_categories)
                            )

                        # Convert column to match categories dtype
                        if categories_dtype != "object":
                            try:
                                result[col] = pd.Series(result[col]).astype(
                                    categories_dtype
                                )
                            except (ValueError, TypeError) as e:
                                logger.debug(
                                    f"Failed to convert column '{col}' to {categories_dtype}: {e}"
                                )

                        result[col] = pd.Categorical(
                            result[col], categories=categories, ordered=ordered
                        )
                    except (ValueError, TypeError) as e:
                        # Fallback #1 (categorical): Simple categorical conversion
                        # Trigger: Categories restoration failed
                        logger.debug(
                            f"Failed to restore categorical for column '{col}': {e}, using simple conversion"
                        )
                        result[col] = result[col].astype("category")
                else:
                    # No metadata available, use simple categorical
                    result[col] = result[col].astype("category")

            # Other dtypes: Try direct conversion if needed
            # This handles cases where:
            # 1. Data was not decoded through codec (direct API calls)
            # 2. Specific dtype precision is required (e.g., int32 vs int64)
            else:
                current_dtype = result[col].dtype
                target_dtype_obj = pd.api.types.pandas_dtype(dtype_str)

                # Skip if already correct dtype
                if current_dtype == target_dtype_obj:
                    continue

                try:
                    # Direct conversion for most types
                    result[col] = result[col].astype(dtype_str)
                except (ValueError, TypeError) as e:
                    # Fallback #2 (dtype): Keep original if conversion fails
                    # Trigger: Type conversion error (e.g., invalid string to int)
                    logger.debug(
                        f"Failed to convert column '{col}' to {dtype_str}: {e}, keeping {current_dtype}"
                    )

        # Apply index metadata
        index_dtype = dtype_mapping.get("index_dtype")
        if index_dtype and index_dtype != "None":
            try:
                result.index = result.index.astype(index_dtype)
            except (ValueError, TypeError) as e:
                # Fallback #3 (index): Keep original index dtype if conversion fails
                # Trigger: Index dtype conversion error (e.g., string to numeric)
                logger.debug(
                    f"Failed to convert index to dtype {index_dtype}: {e}, keeping original"
                )

        index_name = dtype_mapping.get("index_name")
        if index_name is not None:
            result.index.name = index_name

        index_order = dtype_mapping.get("index_order")
        if index_order is not None and set(index_order) == set(result.index):
            result = result.loc[index_order]

        # Apply columns metadata
        columns_dtype = dtype_mapping.get("columns_dtype")
        if columns_dtype and columns_dtype != "None":
            try:
                result.columns = result.columns.astype(columns_dtype)
            except (ValueError, TypeError) as e:
                # Fallback #4 (columns): Keep original columns dtype if conversion fails
                # Trigger: Columns dtype conversion error
                logger.debug(
                    f"Failed to convert columns to dtype {columns_dtype}: {e}, keeping original"
                )

        columns_name = dtype_mapping.get("columns_name")
        if columns_name is not None:
            result.columns.name = columns_name

        columns_order = dtype_mapping.get("columns_order")
        if columns_order is not None and set(columns_order) == set(result.columns):
            result = result[columns_order]

        # Apply DatetimeIndex frequency (if applicable)
        index_freq = dtype_mapping.get("index_freq")
        if (
            index_freq
            and index_freq != "None"
            and isinstance(result.index, pd.DatetimeIndex)
        ):
            try:
                from pandas.tseries.frequencies import to_offset

                offset = to_offset(index_freq)
                original_name = result.index.name
                result.index = pd.DatetimeIndex(
                    result.index.values, freq=offset, name=original_name
                )
            except Exception:
                # Fallback #4 (frequency): Try asfreq method
                # Trigger: to_offset() failed or DatetimeIndex construction failed
                try:
                    tmp = pd.Series([None] * len(result), index=result.index)
                    tmp = tmp.asfreq(index_freq)
                    original_name = result.index.name
                    result.index = tmp.index
                    result.index.name = original_name
                except Exception as e:
                    # Fallback #5 (frequency): Keep original if both methods fail
                    # Trigger: asfreq also failed (e.g., non-uniform DatetimeIndex)
                    logger.debug(
                        f"Failed to set index frequency to {index_freq}: {e}, keeping original"
                    )

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
