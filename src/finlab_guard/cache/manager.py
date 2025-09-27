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
    """Convert object to JSON string with NaN handling."""

    def _default(o: Any) -> str:
        return str(o)

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
        # Prepare current dtype signature
        current_signature = {
            "dtypes": {str(col): str(df[col].dtype) for col in df.columns},
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
            json.dump(dtype_mapping, f, indent=2)

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
            logger.debug(f"Saved hash for {key}: {hash_value[:8]}...")
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
        pdf.index = pdf.index.astype(str)
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
        p_prev = pl.from_pandas(prev_pdf, nan_to_null=False)
        p_cur = pl.from_pandas(cur_pdf, nan_to_null=False)

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
            # Mask: not(both null) and values differ
            # Cast both columns to string for comparison to handle type mismatches
            mask = (~(pl.col(old_col).is_null() & pl.col(new_col).is_null())) & (
                pl.col(old_col).cast(pl.Utf8) != pl.col(new_col).cast(pl.Utf8)
            )
            df_changed = (
                joined.filter(mask)
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
            big_row_keys = (
                counts[counts["n_changes"] > row_change_threshold]["row_key"]
                .astype(str)
                .tolist()
            )
            big_rows: set[str] = {str(k) for k in big_row_keys}
        else:
            all_changes_pdf = pd.DataFrame(columns=["row_key", "old", "new", "col_key"])
            big_rows = set()

        # Build row additions: new rows in cur not in prev
        prev_keys = (
            set(prev.index.astype(str).tolist())
            if (prev is not None and not prev.empty)
            else set()
        )
        cur_keys = list(cur.index.astype(str).tolist())
        cur_keys_set = set(cur_keys)
        new_rows = [k for k in cur_keys if k not in prev_keys]

        # Build row deletions: rows in prev not in cur
        deleted_rows = [k for k in prev_keys if k not in cur_keys_set]

        row_adds = []
        for r in new_rows:
            row_adds.append((str(r), _to_json_str(cur.loc[r].to_dict()), timestamp))

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
                newv: Any = None if pd.isna(row["new"]) else row["new"]
                cell_rows.append((rk, str(ck), _to_json_str(newv), timestamp))

            # Big rows become partial row maps stored in row_additions
            for br in big_rows:
                subset = all_changes_pdf[all_changes_pdf["row_key"] == br]
                row_map = {
                    str(r["col_key"]): (None if pd.isna(r["new"]) else r["new"])
                    for _, r in subset.iterrows()
                }
                row_adds.append((str(br), _to_json_str(row_map), timestamp))

        # New columns: store as column_additions with full column data
        new_cols = [c for c in cur_cols if c not in prev_cols]
        col_additions = []
        if new_cols:
            for c in new_cols:
                # Extract the full column data as a dictionary {row_key: value}
                col_data = {}
                for r in cur_keys:
                    v = cur.at[r, c]
                    col_data[str(r)] = None if pd.isna(v) else v
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
        base_data = self._load_base_snapshot(table_id, target_time)
        cell_changes = self._load_and_process_cell_changes(table_id, target_time)
        row_additions = self._load_and_process_row_additions(table_id, target_time)
        column_additions = self._load_and_process_column_additions(
            table_id, target_time
        )
        deleted_rows, deleted_cols = self._load_deletions(table_id, target_time)
        merged_data = self._merge_data_layers(
            base_data, cell_changes, row_additions, column_additions
        )
        return self._finalize_dataframe(merged_data, deleted_rows, deleted_cols)

    def _load_base_snapshot(self, table_id: str, target_time: datetime) -> pl.DataFrame:
        """Load latest snapshot row_data for the table as Polars DataFrame."""
        q_snap = f"""
        SELECT row_key, row_data FROM rows_base
        WHERE table_id = '{table_id}' AND snapshot_time = (
            SELECT MAX(snapshot_time) FROM rows_base
            WHERE table_id = '{table_id}' AND snapshot_time <= TIMESTAMP '{target_time.isoformat()}'
        )
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
                base_pl = pl.from_pandas(base_wide_pdf, nan_to_null=False).with_columns(
                    pl.col("row_key").cast(pl.Utf8)
                )
            else:
                base_pl = pl.DataFrame({"row_key": pl.Series(row_keys, dtype=pl.Utf8)})
        else:
            base_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})

        return base_pl

    def _load_and_process_cell_changes(
        self, table_id: str, target_time: datetime
    ) -> pl.DataFrame:
        """Load and process cell changes into pivoted Polars DataFrame."""
        q_changes_latest = f"""
        SELECT row_key, col_key, value FROM (
            SELECT row_key, col_key, value,
                row_number() OVER (PARTITION BY row_key, col_key ORDER BY save_time DESC) as rn
            FROM cell_changes
            WHERE table_id = '{table_id}' AND save_time <= TIMESTAMP '{target_time.isoformat()}'
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
            changes_pl = (
                pl.from_pandas(changes_pdf, nan_to_null=False)
                if not changes_pdf.empty
                else pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})
            )

        # Process cell changes and pivot
        if not changes_pl.is_empty():
            # Parse JSON values directly in Polars using map_elements
            # Keep the original _parse_json_value logic but ensure string output for Polars
            def parse_and_stringify(value: str) -> str:
                """Parse JSON and ensure string output for Polars compatibility."""
                parsed = self._parse_json_value(value)
                return str(parsed) if parsed is not None else ""

            changes_pl = changes_pl.with_columns(
                [
                    pl.col("row_key").cast(pl.Utf8),
                    pl.col("col_key").cast(pl.Utf8),
                    pl.col("value")
                    .map_elements(parse_and_stringify, return_dtype=pl.Utf8)
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

            # Ensure row_key is string type
            pivot_pl = pivot_pl.with_columns(pl.col("row_key").cast(pl.Utf8))
        else:
            pivot_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})

        return pivot_pl

    def _parse_json_value(self, x: Any) -> Any:
        """Parse JSON value with smart type conversion."""
        if x is None:
            return None
        if isinstance(x, (int, float, bool)):
            return x
        if isinstance(x, str):
            # Try JSON parsing with smart number conversion
            try:
                parsed = orjson.loads(x) if x else None
                if (
                    isinstance(parsed, str)
                    and parsed.replace(".", "").replace("-", "").isdigit()
                ):
                    return float(parsed) if "." in parsed else int(parsed)
                return parsed
            except Exception:
                try:
                    parsed = json.loads(x) if x else None
                    if (
                        isinstance(parsed, str)
                        and parsed.replace(".", "").replace("-", "").isdigit()
                    ):
                        return float(parsed) if "." in parsed else int(parsed)
                    return parsed
                except Exception:
                    if x.replace(".", "").replace("-", "").isdigit():
                        return float(x) if "." in x else int(x)
                    return x
        return x

    def _load_and_process_row_additions(
        self, table_id: str, target_time: datetime
    ) -> pl.DataFrame:
        """Load and process row additions into wide Polars DataFrame."""
        q_add = f"""
        WITH latest_additions AS (
            SELECT
                row_key,
                row_data,
                ROW_NUMBER() OVER (PARTITION BY row_key ORDER BY save_time DESC) as rn
            FROM row_additions
            WHERE table_id = '{table_id}' AND save_time <= TIMESTAMP '{target_time.isoformat()}'
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
                adds_pl = pl.from_pandas(adds_wide_pdf, nan_to_null=False).with_columns(
                    pl.col("row_key").cast(pl.Utf8)
                )
            else:
                adds_pl = pl.DataFrame(
                    {"row_key": pl.Series(add_row_keys, dtype=pl.Utf8)}
                )
        else:
            adds_pl = pl.DataFrame({"row_key": pl.Series([], dtype=pl.Utf8)})

        return adds_pl

    def _load_deletions(
        self, table_id: str, target_time: datetime
    ) -> tuple[set[str], set[str]]:
        """Load row and column deletions up to target_time, considering re-additions."""
        # Load deleted rows, but exclude those that were re-added later
        q_row_del = f"""
        WITH latest_row_events AS (
            SELECT
                row_key,
                'deletion' as event_type,
                delete_time as event_time
            FROM row_deletions
            WHERE table_id = '{table_id}' AND delete_time <= TIMESTAMP '{target_time.isoformat()}'

            UNION ALL

            SELECT
                row_key,
                'addition' as event_type,
                save_time as event_time
            FROM row_additions
            WHERE table_id = '{table_id}' AND save_time <= TIMESTAMP '{target_time.isoformat()}'
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
            WHERE table_id = '{table_id}' AND delete_time <= TIMESTAMP '{target_time.isoformat()}'
        ),
        readded_columns AS (
            -- Track column re-additions from dedicated column_additions table
            SELECT DISTINCT
                ca.col_key,
                ca.add_time as event_time,
                'addition' as event_type
            FROM column_additions ca
            WHERE ca.table_id = '{table_id}'
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
        """Merge four data layers: base snapshot <- cell changes <- row additions <- column additions."""
        # All DataFrames already have row_key as Utf8, so join directly
        merged = base.join(changes, on="row_key", how="full")

        # Handle row_key from joins
        if "row_key_right" in merged.columns:
            merged = merged.with_columns(
                pl.coalesce(pl.col("row_key"), pl.col("row_key_right")).alias("row_key")
            ).drop("row_key_right")

        # Apply delta columns from pivot
        delta_cols = [c for c in merged.columns if c.endswith("__delta")]
        for delta_col in delta_cols:
            target_col = delta_col[: -len("__delta")]
            if target_col in merged.columns:
                merged = merged.with_columns(
                    pl.coalesce(pl.col(delta_col), pl.col(target_col)).alias(target_col)
                )
            else:
                merged = merged.rename({delta_col: target_col})

        # Drop all remaining __delta columns
        if delta_cols:
            remaining_deltas = [c for c in merged.columns if c.endswith("__delta")]
            if remaining_deltas:
                merged = merged.drop(remaining_deltas)

        # Apply row additions: join and coalesce (row additions have higher precedence)
        if not row_additions.is_empty():
            merged = merged.join(row_additions, on="row_key", how="full", suffix="_add")

            # Handle row_key from add join
            if "row_key_add" in merged.columns:
                merged = merged.with_columns(
                    pl.coalesce(pl.col("row_key"), pl.col("row_key_add")).alias(
                        "row_key"
                    )
                ).drop("row_key_add")

            # Apply add columns
            add_cols = [c for c in row_additions.columns if c != "row_key"]
            for c in add_cols:
                add_col = f"{c}_add"
                if add_col in merged.columns:
                    if c in merged.columns:
                        # For rows that exist in row_additions, use add_col values unconditionally (including NaN)
                        # For rows that don't exist in row_additions, add_col will be null, so use original c
                        merged = merged.with_columns(
                            pl.when(pl.col(add_col).is_not_null())
                            .then(pl.col(add_col))
                            .otherwise(pl.col(c))
                            .alias(c)
                        )
                    else:
                        merged = merged.rename({add_col: c})

            # Clean up remaining _add columns
            remaining_adds = [c for c in merged.columns if c.endswith("_add")]
            if remaining_adds:
                merged = merged.drop(remaining_adds)

        # Apply column additions (fourth layer) - highest precedence for new columns
        if column_additions:
            for col_key, col_data in column_additions.items():
                if col_data:  # Skip empty column data
                    # Create a temporary dataframe with the new column
                    col_rows = []
                    for row_key, value in col_data.items():
                        col_rows.append({"row_key": str(row_key), col_key: value})

                    if col_rows:
                        # Let Polars infer the correct dtype from data
                        col_df = pl.DataFrame(col_rows)
                        # Join the new column to the existing data, handling existing column names
                        if col_key in merged.columns:
                            # Column already exists, merge values (column_additions have precedence for their rows)
                            merged = merged.join(
                                col_df, on="row_key", how="left", suffix="_col_add"
                            )
                            # Use column_additions values where available, otherwise keep existing values
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

        return merged

    def _load_and_process_column_additions(
        self, table_id: str, target_time: datetime
    ) -> dict[str, dict[str, Any]]:
        """Load column additions up to target_time and return as nested dict {col_key: {row_key: value}}."""
        q_col_add = f"""
        WITH latest_column_additions AS (
            SELECT
                col_key,
                col_data_json,
                ROW_NUMBER() OVER (PARTITION BY col_key ORDER BY add_time DESC) as rn
            FROM column_additions
            WHERE table_id = '{table_id}' AND add_time <= TIMESTAMP '{target_time.isoformat()}'
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

        # Apply saved dtypes to columns
        for col, dtype_str in dtypes.items():
            if col in result.columns and dtype_str and dtype_str != "None":
                logger.debug(
                    f"Applying dtype '{dtype_str}' to column '{col}', current dtype: {result[col].dtype}"
                )
                try:
                    # Since values are stored as strings, convert them back
                    if "int" in dtype_str:
                        # Convert string values to numeric first, then to target int type
                        result[col] = pd.to_numeric(
                            result[col], errors="coerce"
                        ).astype(dtype_str)
                        logger.debug(
                            f"Successfully converted column '{col}' to {dtype_str}"
                        )
                    elif "float" in dtype_str:
                        result[col] = pd.to_numeric(
                            result[col], errors="coerce"
                        ).astype(dtype_str)
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
                except (ValueError, TypeError, Exception) as e:
                    # Fallback: try to convert from string
                    logger.debug(
                        f"First dtype conversion failed for column '{col}': {e}"
                    )
                    try:
                        if "int" in dtype_str:
                            result[col] = pd.to_numeric(
                                result[col], errors="coerce"
                            ).astype(
                                dtype_str
                            )  # Use original dtype_str, not hardcoded int64
                            logger.debug(
                                f"Fallback conversion successful for column '{col}' to {dtype_str}"
                            )
                        elif "float" in dtype_str:
                            result[col] = pd.to_numeric(
                                result[col], errors="coerce"
                            ).astype(
                                dtype_str
                            )  # Use original dtype_str, not hardcoded float64
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

        # Save dtype mapping separately (maintain compatibility)
        self._save_dtype_mapping(key, data, timestamp)

        changes = None

        # Check if this is the first save (no existing data)
        existing_data = self.load_raw_data(key)

        if existing_data is None or existing_data.empty:
            # First save - create snapshot
            self.save_snapshot(key, data, timestamp)
            logger.debug(f"Created initial snapshot for {key} with {len(data)} rows")
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
                # Get the latest timestamp for this table
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
                if latest_result and latest_result[0]:
                    as_of_time = latest_result[0]
                else:
                    as_of_time = datetime.now()

            # Use new DuckDB reconstruction
            result = self.reconstruct_as_of(key, as_of_time)

            if result.empty:
                logger.warning(f"No cache data found for {key}")
                return pd.DataFrame()

            # Apply dtype mapping for compatibility
            result = self._apply_dtypes_to_result(result, key, as_of_time)
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

            logger.debug(f"Cleared cache for key: {key}")
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

            logger.debug("Cleared all cache data")
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
