"""DataFrame mutation utilities for random testing.

This module provides comprehensive DataFrame mutation capabilities including:
- Cell-level value changes
- Row insertion at random positions
- Column insertion at random positions
- Comprehensive dtype changes (float64/32/16, int64/32/16/8, etc.)
- Index modifications
- Extreme value testing (NaN, Inf, very large/small numbers)
"""

import random
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


class DataFrameMutator:
    """Provides random mutation capabilities for pandas DataFrames."""

    # Comprehensive dtype mapping for mutations
    NUMERIC_DTYPES = [
        np.float64,
        np.float32,
        np.float16,  # Float precision variations
        np.int64,
        np.int32,
        np.int16,
        np.int8,  # Integer precision variations
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,  # Unsigned integer variations
    ]

    OTHER_DTYPES = [
        np.object_,  # Object/string type
        np.bool_,  # Boolean type
        "category",  # Pandas categorical
        "string",  # Pandas string (nullable)
    ]

    # Extreme values for testing edge cases
    EXTREME_VALUES = [
        np.nan,
        np.inf,
        -np.inf,  # Special float values
        1e-10,
        1e10,
        -1e10,  # Very small/large numbers
        0,
        -1,
        1,  # Simple values
        2**31 - 1,
        -(2**31),  # Int32 limits
        2**63 - 1,
        -(2**63) + 1,  # Int64 limits (approximate)
    ]

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize mutator with optional random seed for reproducibility."""
        self.random = random.Random(random_seed)
        self.np_random = np.random.RandomState(random_seed)

    def mutate_cell_values(
        self, df: pd.DataFrame, n_changes: int = None
    ) -> pd.DataFrame:
        """Randomly mutate cell values in the DataFrame.

        Args:
            df: Source DataFrame to mutate
            n_changes: Number of cells to change (default: 1-5% of total cells)

        Returns:
            Mutated DataFrame copy
        """
        if df.empty:
            return df.copy()

        df_copy = df.copy()
        total_cells = df.size

        if n_changes is None:
            # Change 1-5% of cells randomly
            n_changes = max(1, self.random.randint(1, max(1, total_cells // 20)))

        for _ in range(n_changes):
            # Select random position
            row_idx = self.random.randint(0, len(df) - 1)
            col_idx = self.random.randint(0, len(df.columns) - 1)
            col_name = df.columns[col_idx]

            # Generate new value based on column dtype and data
            new_value = self._generate_value_for_column(df_copy[col_name], row_idx)
            df_copy.iloc[row_idx, col_idx] = new_value

        return df_copy

    def insert_random_rows(self, df: pd.DataFrame, n_rows: int = None) -> pd.DataFrame:
        """Insert new rows at random positions.

        Args:
            df: Source DataFrame
            n_rows: Number of rows to insert (default: 1-3)

        Returns:
            DataFrame with new rows inserted
        """
        if n_rows is None:
            n_rows = self.random.randint(1, 3)

        df_copy = df.copy()

        # Generate all unique index values at once to avoid duplicates
        new_indices = self._generate_unique_index_values(df_copy.index, n_rows)

        for i in range(n_rows):
            # Choose random insertion position (0 to len(df))
            insert_pos = self.random.randint(0, len(df_copy))

            # Generate new row data
            new_row_data = {}
            for col in df_copy.columns:
                # Use column-aware generation for categorical data
                new_row_data[col] = self._generate_value_for_column(df_copy[col], 0)

            # Create new row as DataFrame with unique index
            new_row = pd.DataFrame([new_row_data], index=[new_indices[i]]).astype(
                df_copy.dtypes
            )

            # Insert row at specified position
            if insert_pos == 0:
                df_copy = pd.concat([new_row, df_copy], ignore_index=False)
            elif insert_pos >= len(df_copy):
                df_copy = pd.concat([df_copy, new_row], ignore_index=False)
            else:
                df_top = df_copy.iloc[:insert_pos]
                df_bottom = df_copy.iloc[insert_pos:]
                df_copy = pd.concat([df_top, new_row, df_bottom], ignore_index=False)

        # Sort by index if it's a datetime index to maintain chronological order
        if pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy = df_copy.sort_index()

        return df_copy

    def insert_random_columns(
        self, df: pd.DataFrame, n_cols: int = None
    ) -> pd.DataFrame:
        """Insert new columns at random positions.

        Args:
            df: Source DataFrame
            n_cols: Number of columns to insert (default: 1-2)

        Returns:
            DataFrame with new columns inserted
        """
        if n_cols is None:
            n_cols = self.random.randint(1, 2)

        df_copy = df.copy()

        for _ in range(n_cols):
            # Choose random insertion position
            insert_pos = self.random.randint(0, len(df_copy.columns))

            # Generate new column name (avoid conflicts)
            existing_cols = set(df_copy.columns)
            col_name = self._generate_unique_column_name(existing_cols)

            # Choose random dtype for new column
            dtype = self.random.choice(self.NUMERIC_DTYPES + [np.object_])

            # Convert object dtype to string for consistency
            value_dtype = "string" if dtype == np.object_ else dtype

            # Generate column data
            col_data = [
                self._generate_value_for_dtype(value_dtype) for _ in range(len(df_copy))
            ]

            # Insert column at specified position
            if insert_pos >= len(df_copy.columns):
                df_copy[col_name] = col_data
            else:
                # Insert at specific position by reordering columns
                cols = df_copy.columns.tolist()
                cols.insert(insert_pos, col_name)
                df_copy[col_name] = col_data
                df_copy = df_copy[cols]

        return df_copy

    def mutate_dtypes(self, df: pd.DataFrame, n_columns: int = None) -> pd.DataFrame:
        """Change dtypes of random columns, ensuring entire column consistency.

        Args:
            df: Source DataFrame
            n_columns: Number of columns to change dtype (default: 1-3)

        Returns:
            DataFrame with mutated dtypes (entire columns changed consistently)
        """
        if df.empty or len(df.columns) == 0:
            return df.copy()

        if n_columns is None:
            n_columns = min(len(df.columns), self.random.randint(1, 3))

        df_copy = df.copy()
        columns_to_change = self.random.sample(list(df_copy.columns), n_columns)

        for col in columns_to_change:
            current_dtype = df_copy[col].dtype
            new_dtype = self._choose_compatible_dtype(current_dtype)

            success = self._convert_column_dtype_safely(df_copy, col, new_dtype)
            if not success:
                # If primary conversion fails, try fallback conversions
                fallback_dtypes = self._get_fallback_dtypes(current_dtype)
                for fallback_dtype in fallback_dtypes:
                    if self._convert_column_dtype_safely(df_copy, col, fallback_dtype):
                        break

        return df_copy

    def mutate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mutate the DataFrame index.

        Args:
            df: Source DataFrame

        Returns:
            DataFrame with mutated index
        """
        df_copy = df.copy()

        mutation_type = self.random.choice(
            ["rename", "reorder", "dtype_change", "new_values"]
        )

        if mutation_type == "rename" and hasattr(df_copy.index, "name"):
            # Change index name
            df_copy.index.name = f"mutated_{df_copy.index.name or 'index'}"

        elif mutation_type == "reorder" and len(df_copy) > 1:
            # Shuffle index order
            new_order = list(range(len(df_copy)))
            self.random.shuffle(new_order)
            df_copy = df_copy.iloc[new_order]

        elif mutation_type == "dtype_change":
            try:
                # Try to change index dtype
                if pd.api.types.is_numeric_dtype(df_copy.index):
                    new_dtype = self.random.choice(
                        [np.int32, np.int64, np.float32, np.float64]
                    )
                    df_copy.index = df_copy.index.astype(new_dtype)
                elif pd.api.types.is_datetime64_any_dtype(df_copy.index):
                    # For datetime index, try different freq or timezone
                    pass  # Keep as is for now
            except Exception:
                pass  # Keep original if conversion fails

        elif mutation_type == "new_values":
            # Generate completely new index values (ensure uniqueness)
            new_index = self._generate_unique_index_values(df_copy.index, len(df_copy))
            df_copy.index = new_index

        return df_copy

    def delete_random_rows(self, df: pd.DataFrame, n_rows: int = None) -> pd.DataFrame:
        """Delete random rows from the DataFrame.

        Args:
            df: Source DataFrame
            n_rows: Number of rows to delete (default: 10-30% of rows, min 1)

        Returns:
            DataFrame with rows deleted
        """
        if df.empty or len(df) <= 1:
            return df.copy()  # Can't delete from empty or single-row DataFrame

        if n_rows is None:
            # Delete 10-30% of rows, but at least 1 and leave at least 1 row
            max_deletable = len(df) - 1  # Leave at least 1 row
            n_rows = min(
                max_deletable, max(1, int(len(df) * self.random.uniform(0.1, 0.3)))
            )

        # Ensure we don't delete too many rows
        n_rows = min(n_rows, len(df) - 1)

        if n_rows <= 0:
            return df.copy()

        df_copy = df.copy()

        # Select random rows to delete
        rows_to_delete = self.random.sample(range(len(df_copy)), n_rows)

        # Delete selected rows
        df_copy = df_copy.drop(df_copy.index[rows_to_delete])

        return df_copy

    def delete_random_columns(
        self, df: pd.DataFrame, n_cols: int = None
    ) -> pd.DataFrame:
        """Delete random columns from the DataFrame.

        Args:
            df: Source DataFrame
            n_cols: Number of columns to delete (default: 10-30% of columns, min 1)

        Returns:
            DataFrame with columns deleted
        """
        if df.empty or len(df.columns) <= 1:
            return df.copy()  # Can't delete from empty or single-column DataFrame

        if n_cols is None:
            # Delete 10-30% of columns, but at least 1 and leave at least 1 column
            max_deletable = len(df.columns) - 1  # Leave at least 1 column
            n_cols = min(
                max_deletable,
                max(1, int(len(df.columns) * self.random.uniform(0.1, 0.3))),
            )

        # Ensure we don't delete too many columns
        n_cols = min(n_cols, len(df.columns) - 1)

        if n_cols <= 0:
            return df.copy()

        df_copy = df.copy()

        # Select random columns to delete
        cols_to_delete = self.random.sample(list(df_copy.columns), n_cols)

        # Delete selected columns
        df_copy = df_copy.drop(columns=cols_to_delete)

        return df_copy

    def apply_random_mutations(
        self, df: pd.DataFrame, n_mutations: int = None
    ) -> pd.DataFrame:
        """Apply a sequence of random mutations to the DataFrame.

        Args:
            df: Source DataFrame
            n_mutations: Number of mutation operations to apply (default: 1-5)

        Returns:
            DataFrame after applying all mutations
        """
        if n_mutations is None:
            n_mutations = self.random.randint(1, 5)

        df_mutated = df.copy()

        # Available mutation functions
        mutation_funcs = [
            self.mutate_cell_values,
            self.insert_random_rows,
            self.insert_random_columns,
            self.delete_random_rows,
            self.delete_random_columns,
            self.mutate_dtypes,
            self.mutate_index,
        ]

        for _ in range(n_mutations):
            # Choose random mutation
            mutation_func = self.random.choice(mutation_funcs)
            try:
                df_mutated = mutation_func(df_mutated)
            except Exception:
                # If mutation fails, continue with next one
                pass

        return df_mutated

    def _generate_value_for_column(self, column: pd.Series, row_idx: int) -> Any:
        """Generate a random value for a specific column, handling categorical properly."""
        dtype = column.dtype
        dtype_str = str(dtype)

        # Handle categorical columns with actual categories
        if "category" in dtype_str:
            # Choose from actual categories, not hardcoded values
            if hasattr(column, "cat") and hasattr(column.cat, "categories"):
                categories = column.cat.categories.tolist()
                if categories:
                    return self.random.choice(categories)
            # Fallback to existing values in the column if no categories
            existing_values = column.dropna().unique().tolist()
            if existing_values:
                return self.random.choice(existing_values)
            # Last resort fallback
            return self.random.choice(["A", "B", "C", "D"])

        # Handle object dtype with content inspection
        elif dtype_str == "object":
            non_null_values = column.dropna()
            if len(non_null_values) > 0:
                # Check if all non-null values are strings
                all_strings = all(isinstance(val, str) for val in non_null_values)
                if all_strings:
                    # Treat as string type
                    return self._generate_value_for_dtype("string")

        # For non-categorical columns, use the existing dtype-based method
        return self._generate_value_for_dtype(dtype)

    def _generate_value_for_dtype(self, dtype) -> Any:
        """Generate a random value that strictly matches the given dtype."""
        dtype_str = str(dtype)

        # Handle special pandas dtypes
        if "category" in dtype_str:
            # For categorical dtypes, we need the actual categories
            # This is a fallback - in practice, should use actual categories from the column
            return self.random.choice(["A", "B", "C", "D"])
        elif "string" in dtype_str:
            return f"random_str_{self.random.randint(1, 1000)}"
        elif dtype == np.object_:
            # For object dtype, mix different types but ensure compatibility
            return self.random.choice(
                [
                    f"str_{self.random.randint(1, 1000)}",
                    # self.random.randint(1, 100),
                    # self.random.uniform(1.0, 100.0),
                    None,  # Allow None values in object columns
                ]
            )
        elif "bool" in dtype_str:
            return bool(self.random.choice([True, False]))
        elif "datetime" in dtype_str:
            base_date = pd.Timestamp("2020-01-01")
            days_offset = self.random.randint(-1000, 1000)
            return base_date + pd.Timedelta(days=days_offset)

        # Handle numeric dtypes with precise type casting
        if pd.api.types.is_integer_dtype(dtype):
            if "int8" in dtype_str:
                value = self.random.randint(-128, 127)
                return np.int8(value)
            elif "int16" in dtype_str:
                value = self.random.randint(-32768, 32767)
                return np.int16(value)
            elif "int32" in dtype_str:
                value = self.random.randint(-(2**31), 2**31 - 1)
                return np.int32(value)
            elif "int64" in dtype_str:
                value = self.random.randint(-(2**20), 2**20)  # Reasonable range
                return np.int64(value)
            elif "uint8" in dtype_str:
                value = self.random.randint(0, 255)
                return np.uint8(value)
            elif "uint16" in dtype_str:
                value = self.random.randint(0, 65535)
                return np.uint16(value)
            elif "uint32" in dtype_str:
                value = self.random.randint(0, 2**32 - 1)
                return np.uint32(value)
            elif "uint64" in dtype_str:
                value = self.random.randint(0, 2**20)  # Reasonable range
                return np.uint64(value)
            else:
                # Default integer
                value = self.random.randint(-(2**20), 2**20)
                return np.int64(value)

        elif pd.api.types.is_float_dtype(dtype):
            # Generate values appropriate for float precision
            if self.random.random() < 0.1:  # 10% chance of extreme value
                extreme_val = self.random.choice([np.nan, np.inf, -np.inf, 0.0])
                if "float16" in dtype_str:
                    return (
                        np.float16(extreme_val)
                        if not np.isnan(extreme_val) and not np.isinf(extreme_val)
                        else extreme_val
                    )
                elif "float32" in dtype_str:
                    return (
                        np.float32(extreme_val)
                        if not np.isnan(extreme_val) and not np.isinf(extreme_val)
                        else extreme_val
                    )
                else:
                    return (
                        np.float64(extreme_val)
                        if not np.isnan(extreme_val) and not np.isinf(extreme_val)
                        else extreme_val
                    )
            else:
                value = self.random.uniform(-1000, 1000)
                if "float16" in dtype_str:
                    return np.float16(value)
                elif "float32" in dtype_str:
                    return np.float32(value)
                else:
                    return np.float64(value)

        # Default fallback - return a safe value
        return f"fallback_{self.random.randint(1, 1000)}"

    def _choose_compatible_dtype(self, current_dtype):
        """Choose a new dtype that's somewhat compatible with current dtype."""
        if pd.api.types.is_numeric_dtype(current_dtype):
            # For numeric, try other numeric types
            return self.random.choice(self.NUMERIC_DTYPES + ["category", "string"])
        elif pd.api.types.is_object_dtype(current_dtype):
            # For object, try string or category
            return self.random.choice(["category", "string"])
        else:
            # For others, try object or numeric
            return self.random.choice([np.float64, "category"])

    def _convert_column_dtype_safely(
        self, df: pd.DataFrame, col: str, new_dtype
    ) -> bool:
        """Safely convert column dtype, ensuring entire column consistency.

        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            if new_dtype == "category":
                df[col] = df[col].astype("category")
            elif new_dtype == "string":
                df[col] = df[col].astype("string")
            elif pd.api.types.is_numeric_dtype(new_dtype):
                # For numeric conversions, ensure all values are compatible
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    # Direct numeric conversion
                    df[col] = df[col].astype(new_dtype)
                else:
                    # Convert from non-numeric to numeric
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(new_dtype)
            else:
                # Generic conversion
                df[col] = df[col].astype(new_dtype)
            return True
        except (ValueError, OverflowError, TypeError):
            return False

    def _get_fallback_dtypes(self, current_dtype):
        """Get safe fallback dtypes if primary conversion fails."""
        if pd.api.types.is_numeric_dtype(current_dtype):
            # For numeric types, try safer numeric conversions
            return [np.float64, np.object_, "string"]
        elif pd.api.types.is_object_dtype(current_dtype):
            # For object types, try string or keep as object
            return ["string", np.object_]
        else:
            # For other types, convert to object as safe fallback
            return [np.object_, "string"]

    def _generate_index_value(self, current_index):
        """Generate a new index value based on current index type."""
        if pd.api.types.is_datetime64_any_dtype(current_index):
            # For datetime index, generate random timestamp
            base_date = pd.Timestamp("2020-01-01")
            days_offset = self.random.randint(0, 1000)
            return base_date + pd.Timedelta(days=days_offset)
        elif pd.api.types.is_numeric_dtype(current_index):
            # For numeric index, generate random number
            return self.random.randint(1, 10000)
        else:
            # For other types, generate random string
            return f"idx_{self.random.randint(1, 10000)}"

    def _generate_unique_index_values(self, current_index, n_values: int):
        """Generate unique index values based on current index type.

        Args:
            current_index: Current DataFrame index to determine type
            n_values: Number of unique values to generate

        Returns:
            List of unique index values (unique among themselves AND not in current_index)
        """
        # Initialize with existing index values to avoid duplicates
        generated = set()
        for idx_val in current_index:
            key = idx_val.timestamp() if isinstance(idx_val, pd.Timestamp) else idx_val
            generated.add(key)

        result = []
        max_attempts = n_values * 100  # Avoid infinite loop
        attempts = 0

        while len(result) < n_values and attempts < max_attempts:
            attempts += 1
            value = self._generate_index_value(current_index)

            # For datetime, use timestamp for uniqueness check
            key = value.timestamp() if isinstance(value, pd.Timestamp) else value

            if key not in generated:
                generated.add(key)
                result.append(value)

        # Fallback: if still not enough unique values, generate sequential ones
        if len(result) < n_values:
            if pd.api.types.is_datetime64_any_dtype(current_index):
                base = pd.Timestamp("2020-01-01")
                result = [base + pd.Timedelta(hours=i) for i in range(n_values)]
            elif pd.api.types.is_numeric_dtype(current_index):
                result = list(range(n_values))
            else:
                result = [f"idx_{i}" for i in range(n_values)]

        # Sort datetime index to maintain chronological order
        if pd.api.types.is_datetime64_any_dtype(current_index):
            result = sorted(result)

        return result

    def _generate_unique_column_name(self, existing_cols: set) -> str:
        """Generate a unique column name that doesn't conflict with existing columns."""
        for i in range(10000):  # Try up to 10000 names
            name = f"random_col_{i}"
            if name not in existing_cols:
                return name
        # Fallback
        return f"random_col_{self.random.randint(10000, 99999)}"
