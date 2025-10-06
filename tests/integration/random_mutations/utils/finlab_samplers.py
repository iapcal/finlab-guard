"""Finlab DataFrame samplers for random testing.

This module provides utilities to load and sample real finlab DataFrames
from pickle files for comprehensive random testing scenarios.
"""

import pickle
import random
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class FinlabDataSampler:
    """Samples real finlab DataFrames from pickle files for testing."""

    def __init__(
        self,
        pickle_dir: Union[str, Path, None] = None,
        random_seed: Optional[int] = None,
    ):
        """Initialize sampler with pickle directory and optional random seed.

        Args:
            pickle_dir: Directory containing finlab pickle files (defaults to fixtures directory)
            random_seed: Random seed for reproducible sampling
        """
        if pickle_dir is None:
            # Default to fixtures directory relative to this file
            pickle_dir = Path(__file__).parent.parent / "fixtures"
        self.pickle_dir = Path(pickle_dir)
        self.random = random.Random(random_seed)
        self.random_seed = random_seed
        self._cache: dict[str, pd.DataFrame] = {}

        # Discover available pickle files
        self.available_files = list(self.pickle_dir.glob("*.pkl"))

        if not self.available_files:
            raise FileNotFoundError(f"No pickle files found in {pickle_dir}")

    def get_available_datasets(self) -> list[str]:
        """Get list of available dataset names."""
        return [f.stem for f in self.available_files]

    def load_dataset(self, dataset_name: str, use_cache: bool = True) -> pd.DataFrame:
        """Load a specific dataset by name.

        Args:
            dataset_name: Name of the dataset (without .pkl extension)
            use_cache: Whether to use cached version if available

        Returns:
            Loaded DataFrame
        """
        if use_cache and dataset_name in self._cache:
            return self._cache[dataset_name].copy()

        pickle_file = self.pickle_dir / f"{dataset_name}.pkl"
        if not pickle_file.exists():
            raise FileNotFoundError(f"Dataset {dataset_name}.pkl not found")

        try:
            with open(pickle_file, "rb") as f:
                df = pickle.load(f)

            if use_cache:
                self._cache[dataset_name] = df

            return df.copy()
        except Exception as e:
            raise ValueError(f"Failed to load {dataset_name}.pkl: {e}") from e

    def sample_random_dataset(self) -> tuple[str, pd.DataFrame]:
        """Sample a random dataset from available pickle files.

        Returns:
            Tuple of (dataset_name, dataframe)
        """
        dataset_name = self.random.choice(self.get_available_datasets())
        df = self.load_dataset(dataset_name)
        return dataset_name, df

    def sample_subset(
        self,
        df: pd.DataFrame,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        min_rows: int = 5,
        min_cols: int = 2,
    ) -> pd.DataFrame:
        """Sample a random subset of the DataFrame for manageable testing.

        Args:
            df: Source DataFrame
            max_rows: Maximum number of rows (default: min(100, df.shape[0]))
            max_cols: Maximum number of columns (default: min(50, df.shape[1]))
            min_rows: Minimum number of rows to sample
            min_cols: Minimum number of columns to sample

        Returns:
            Sampled subset DataFrame
        """
        if df.empty:
            return df.copy()

        # Set reasonable defaults for testing
        if max_rows is None:
            max_rows = min(100, len(df))
        if max_cols is None:
            max_cols = min(50, len(df.columns))

        # Ensure minimums
        max_rows = max(min_rows, max_rows)
        max_cols = max(min_cols, max_cols)

        # Sample rows
        n_rows = min(len(df), self.random.randint(min_rows, max_rows))
        if n_rows < len(df):
            sampled_rows = sorted(self.random.sample(range(len(df)), n_rows))
            df_subset = df.iloc[sampled_rows].copy()
        else:
            df_subset = df.copy()

        # Sample columns
        n_cols = min(len(df.columns), self.random.randint(min_cols, max_cols))
        if n_cols < len(df.columns):
            sampled_cols = self.random.sample(list(df.columns), n_cols)
            df_subset = df_subset[sampled_cols].copy()

        return df_subset

    def create_test_scenarios(self, n_scenarios: int = 10) -> list[dict]:
        """Create multiple test scenarios with different datasets and subsets.

        Args:
            n_scenarios: Number of scenarios to create

        Returns:
            List of scenario dictionaries containing dataset info
        """
        scenarios = []

        for i in range(n_scenarios):
            # Sample dataset
            dataset_name, df = self.sample_random_dataset()

            # Create different types of subsets based on dataset characteristics
            scenario_type = self._classify_dataset(df)
            subset_params = self._get_subset_params_for_type(scenario_type)

            # Sample subset
            df_subset = self.sample_subset(df, **subset_params)

            scenario = {
                "scenario_id": i,
                "dataset_name": dataset_name,
                "scenario_type": scenario_type,
                "original_shape": df.shape,
                "subset_shape": df_subset.shape,
                "dataframe": df_subset,
                "has_datetime_index": pd.api.types.is_datetime64_any_dtype(
                    df_subset.index
                ),
                "has_numeric_data": any(
                    pd.api.types.is_numeric_dtype(df_subset[col])
                    for col in df_subset.columns
                ),
                "has_missing_values": df_subset.isnull().any().any(),
            }

            scenarios.append(scenario)

        return scenarios

    def _classify_dataset(self, df: pd.DataFrame) -> str:
        """Classify dataset type based on characteristics."""
        if pd.api.types.is_datetime64_any_dtype(df.index) and df.shape[1] > 100:
            return "large_timeseries"  # Like close.pkl, monthly_revenue.pkl
        elif pd.api.types.is_datetime64_any_dtype(df.index):
            return "small_timeseries"  # Time-indexed but smaller
        elif df.shape[1] > 100:
            return "large_tabular"  # Many columns but not time-indexed
        elif any("date" in str(col).lower() for col in df.columns):
            return "date_containing"  # Contains date columns
        else:
            return "simple_tabular"  # Simple rectangular data

    def _get_subset_params_for_type(self, scenario_type: str) -> dict:
        """Get subset parameters based on scenario type."""
        params = {
            "large_timeseries": {
                "max_rows": self.random.randint(20, 80),
                "max_cols": self.random.randint(10, 30),
                "min_rows": 10,
                "min_cols": 5,
            },
            "small_timeseries": {
                "max_rows": self.random.randint(15, 50),
                "max_cols": self.random.randint(5, 15),
                "min_rows": 8,
                "min_cols": 3,
            },
            "large_tabular": {
                "max_rows": self.random.randint(30, 100),
                "max_cols": self.random.randint(8, 25),
                "min_rows": 10,
                "min_cols": 4,
            },
            "date_containing": {
                "max_rows": self.random.randint(20, 80),
                "max_cols": None,  # Keep all columns for date analysis
                "min_rows": 10,
                "min_cols": 2,
            },
            "simple_tabular": {
                "max_rows": self.random.randint(10, 60),
                "max_cols": None,  # Keep all columns
                "min_rows": 5,
                "min_cols": 2,
            },
        }

        return params.get(scenario_type, params["simple_tabular"])


class TestDataGenerator:
    """Generate synthetic test data based on real finlab patterns."""

    def __init__(self, sampler: FinlabDataSampler):
        self.sampler = sampler
        self.random = sampler.random

    def create_mutation_sequence(
        self, base_df: pd.DataFrame, n_steps: int = 5
    ) -> list[tuple[str, pd.DataFrame]]:
        """Create a sequence of DataFrame mutations with dtype consistency.

        Args:
            base_df: Base DataFrame to start mutations from
            n_steps: Number of mutation steps to create

        Returns:
            List of (description, mutated_dataframe) tuples representing the sequence
        """
        from tests.integration.random_mutations.utils.dataframe_mutators import (
            DataFrameMutator,
        )

        # Initialize mutator with deterministic seed derived from current random state
        # This ensures mutations are reproducible
        current_state = self.random.getstate()
        # Extract seed directly from the random state tuple (more stable than hash)
        # current_state is typically (version, tuple_of_ints, gauss_next)
        # We use the first few integers from the state tuple
        state_tuple = current_state[1]  # Get the integer tuple part
        mutator_seed = (state_tuple[0] + state_tuple[1]) % (
            2**31
        )  # Combine first two integers
        mutator = DataFrameMutator(random_seed=mutator_seed)

        sequence = [("initial", base_df.copy())]
        current_df = base_df.copy()

        # Create a balanced mutation plan that avoids dtype conflicts
        mutation_plan = self._create_balanced_mutation_plan(n_steps)

        for step, mutation_type in enumerate(mutation_plan):
            try:
                if mutation_type == "dtype_changes":
                    # Apply dtype changes to entire columns
                    n_cols = min(len(current_df.columns), self.random.randint(1, 3))
                    current_df = mutator.mutate_dtypes(current_df, n_columns=n_cols)
                    description = f"step_{step + 1}_dtype_changes_{n_cols}_columns"

                elif mutation_type == "cell_values":
                    # Apply cell value changes that respect current dtypes
                    n_changes = self.random.randint(1, min(5, current_df.size))
                    current_df = mutator.mutate_cell_values(
                        current_df, n_changes=n_changes
                    )
                    description = f"step_{step + 1}_cell_values_{n_changes}_changes"

                elif mutation_type == "new_rows":
                    # Add new rows with values matching current dtypes
                    n_rows = self.random.randint(1, 3)
                    current_df = mutator.insert_random_rows(current_df, n_rows=n_rows)
                    description = f"step_{step + 1}_new_rows_{n_rows}_added"

                elif mutation_type == "new_columns":
                    # Add new columns with consistent dtypes
                    n_cols = self.random.randint(1, 2)
                    current_df = mutator.insert_random_columns(
                        current_df, n_cols=n_cols
                    )
                    description = f"step_{step + 1}_new_columns_{n_cols}_added"

                elif mutation_type == "delete_rows":
                    # Delete random rows (conservative approach)
                    n_rows = min(len(current_df) - 1, self.random.randint(1, 2))
                    current_df = mutator.delete_random_rows(current_df, n_rows=n_rows)
                    description = f"step_{step + 1}_delete_rows_{n_rows}_removed"

                elif mutation_type == "delete_columns":
                    # Delete random columns (conservative approach)
                    n_cols = min(len(current_df.columns) - 1, self.random.randint(1, 2))
                    current_df = mutator.delete_random_columns(
                        current_df, n_cols=n_cols
                    )
                    description = f"step_{step + 1}_delete_columns_{n_cols}_removed"

                elif mutation_type == "index_mutation":
                    # Mutate the index
                    current_df = mutator.mutate_index(current_df)
                    description = f"step_{step + 1}_index_mutation"

                else:
                    # Fallback to safe cell value mutation
                    current_df = mutator.mutate_cell_values(current_df, n_changes=1)
                    description = f"step_{step + 1}_fallback_cell_change"

                # Add to sequence
                sequence.append((description, current_df.copy()))

            except Exception:
                # If mutation fails, try a safer fallback
                try:
                    current_df = mutator.mutate_cell_values(current_df, n_changes=1)
                    description = f"step_{step + 1}_exception_fallback"
                    sequence.append((description, current_df.copy()))
                except Exception:
                    # If even fallback fails, skip this step but continue
                    pass

        return sequence

    def _create_balanced_mutation_plan(self, n_steps: int) -> list[str]:
        """Create a conservative mutation plan that maximizes success rate.

        Based on empirical testing, this strategy focuses on mutations that
        are most likely to succeed with finlab-guard's current implementation.
        Now includes deletion operations with careful timing.

        Returns:
            List of mutation type names in execution order
        """
        # Define mutation categories by reliability (based on test results)
        safe_mutations = ["cell_values"]  # 100% reliable
        moderate_mutations = [
            "new_columns",
            "delete_rows",
            "delete_columns",
        ]  # Moderately reliable
        risky_mutations = [
            "new_rows",
            "index_mutation",
            "dtype_changes",
        ]  # High failure rate

        plan = []

        for step in range(n_steps):
            # Conservative strategy: favor safe mutations, use moderate sparingly, avoid risky
            if step == 0:
                # First step: always use safe mutations to establish baseline
                mutation = self.random.choice(safe_mutations)
            elif step == n_steps - 1:
                # Last step: end with safe mutations for verification
                mutation = self.random.choice(safe_mutations)
            else:
                # Middle steps: mostly safe, occasionally moderate, avoid risky
                # Special handling for deletion operations - only in middle steps and not too early
                if step == 1 and n_steps > 3:
                    # Second step: avoid deletions to ensure we have enough data to work with
                    weights = {
                        "safe": 0.8,  # 80% safe mutations
                        "moderate_non_delete": 0.15,  # 15% moderate non-delete mutations
                        "risky": 0.05,  # 5% risky mutations
                    }
                else:
                    weights = {
                        "safe": 0.6,  # 60% safe mutations
                        "moderate": 0.3,  # 30% moderate mutations (including deletions)
                        "risky": 0.1,  # 10% risky mutations
                    }

                category = self.random.choices(
                    list(weights.keys()), weights=list(weights.values())
                )[0]

                if category == "safe":
                    mutation = self.random.choice(safe_mutations)
                elif category == "moderate":
                    mutation = self.random.choice(moderate_mutations)
                elif category == "moderate_non_delete":
                    # Only non-deletion moderate mutations
                    mutation = self.random.choice(["new_columns"])
                else:  # risky
                    mutation = self.random.choice(risky_mutations)

            plan.append(mutation)

        return plan
