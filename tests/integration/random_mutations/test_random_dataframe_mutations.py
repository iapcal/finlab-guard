"""Random DataFrame mutation tests for comprehensive finlab-guard validation.

This module performs extensive random testing with 10 iterations per seed:
- Loading real finlab DataFrames from pickle files
- Applying random mutations (cell changes, row/column insertions, dtype changes)
- Verifying as-of-time query correctness
- Testing historical data reconstruction

With 50 different seeds, this provides 500 total test iterations with better
parallelization compared to 5 seeds × 100 iterations.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

from finlab_guard import FinlabGuard
from tests.integration.random_mutations.utils.finlab_samplers import FinlabDataSampler, TestDataGenerator
from tests.integration.random_mutations.utils.dataframe_mutators import DataFrameMutator
from tests.integration.random_mutations.utils.verification_helpers import AsOfTimeVerifier


class TestRandomDataFrameMutations:
    """Comprehensive random mutation testing for finlab-guard."""

    def test_10_random_mutations_per_seed(
        self,
        random_guard: FinlabGuard,
        seeded_sampler: FinlabDataSampler,
        seeded_mutator: DataFrameMutator,
        as_of_verifier: AsOfTimeVerifier,
        test_seed: int
    ):
        """Test comprehensive random DataFrame mutations with dtype consistency.

        This test performs enhanced validation of finlab-guard's ability to:
        1. Handle dtype-consistent DataFrame mutations correctly
        2. Maintain data integrity across different data types and structures
        3. Provide accurate as-of-time queries for all mutation scenarios
        4. Preserve dtype information through the cache system
        5. Ensure column-level dtype consistency in mutations
        6. Validate cell-level value compatibility with column dtypes
        """
        print(f"\n🎲 Starting dtype-consistent random mutations test with seed: {test_seed}")
        pd.set_option("display.width", None)
        # Statistics tracking
        total_iterations = 10
        successful_iterations = 0
        failed_iterations = 0
        total_mutations_applied = 0
        total_verifications_passed = 0
        total_verifications_failed = 0
        dtype_consistency_checks = 0
        dtype_consistency_failures = 0
        deletion_operations_applied = 0
        deletion_operations_successful = 0

        iteration_results = []

        # Test generator for creating mutation sequences
        test_generator = TestDataGenerator(seeded_sampler)

        for iteration in range(total_iterations):
            iteration_start_time = time.time()

            try:
                print(f"\n📊 Iteration {iteration + 1}/{total_iterations}")

                # 1. Sample random dataset and create subset
                dataset_name, base_df = seeded_sampler.sample_random_dataset()
                df_subset = seeded_sampler.sample_subset(base_df)

                print(f"   Dataset: {dataset_name} (shape: {df_subset.shape})")
                print(f"   Data types: {dict(df_subset.dtypes)}")

                # 2. Create random mutation sequence (3-7 steps)
                n_mutation_steps = seeded_sampler.random.randint(3, 7)
                mutation_sequence = test_generator.create_mutation_sequence(
                    df_subset, n_steps=n_mutation_steps
                )

                total_mutations_applied += len(mutation_sequence) - 1  # -1 for initial state

                print(f"   Mutation steps: {n_mutation_steps}")
                print(f"   Mutations: {[desc for desc, _ in mutation_sequence[1:]]}")

                # Track deletion operations
                deletion_ops_in_sequence = [desc for desc, _ in mutation_sequence[1:] if 'delete' in desc]
                deletion_operations_applied += len(deletion_ops_in_sequence)
                if deletion_ops_in_sequence:
                    print(f"   Deletion ops: {deletion_ops_in_sequence}")

                # 3. Apply mutations and store in finlab-guard
                dataset_key = f"random_test_{iteration}_{dataset_name}"

                # Store each step with time intervals (1 second apart) using proper mock pattern
                base_time = datetime.now()
                for step_idx, (description, df_step) in enumerate(mutation_sequence):
                    timestamp = base_time + timedelta(seconds=step_idx)

                    try:
                        # Validate dtype consistency for this step
                        if step_idx > 0:
                            consistency_check = self._check_dtype_consistency(df_step, description)
                            dtype_consistency_checks += 1
                            if not consistency_check['is_consistent']:
                                dtype_consistency_failures += 1
                                print(f"     ⚠️  Dtype inconsistency in {description}: {consistency_check['issues']}")

                        # Store the data using mock pattern with time control
                        with patch.object(random_guard, "_now", return_value=timestamp):
                            with patch.object(type(random_guard), "_fetch_from_finlab", return_value=df_step):
                                # Random mutations almost always modify existing data
                                allow_changes = (step_idx > 0)
                                random_guard.get(dataset_key, allow_historical_changes=allow_changes)
                        print(f"     ✓ Stored: {description} (shape: {df_step.shape})")
                    except Exception as e:
                        print(f"     ❌ Failed to store {description}: {e}")
                        raise

                # 4. Verify as-of-time queries for all steps
                verification_results = as_of_verifier.verify_mutation_sequence(
                    dataset_key,
                    mutation_sequence,
                    time_interval_seconds=1,
                    base_timestamp=base_time
                )

                # Update statistics
                total_verifications_passed += verification_results['verified_steps']
                total_verifications_failed += verification_results['failed_steps']

                # 5. Test additional as-of-time scenarios using time context
                # Query between timestamps to test historical consistency
                if len(mutation_sequence) > 1:
                    mid_time = base_time + timedelta(seconds=0.5)
                    try:
                        random_guard.set_time_context(mid_time)
                        try:
                            mid_result = random_guard.get(dataset_key, allow_historical_changes=False)
                            # Should get the initial state
                            print(f"     ✓ Mid-timestamp query successful (shape: {mid_result.shape})")
                        finally:
                            random_guard.clear_time_context()
                    except Exception as e:
                        print(f"     ❌ Mid-timestamp query failed: {e}")

                # 6. Test edge case queries
                # Query before first timestamp (should fail or return empty)
                before_time = base_time - timedelta(seconds=1)
                try:
                    random_guard.set_time_context(before_time)
                    try:
                        before_result = random_guard.get(dataset_key, allow_historical_changes=False)
                        print(f"     ⚠️  Before-time query returned data: {before_result.shape}")
                    finally:
                        random_guard.clear_time_context()
                except Exception:
                    print(f"     ✓ Before-time query correctly failed/empty")

                # Query after last timestamp (should get latest)
                after_time = base_time + timedelta(seconds=len(mutation_sequence) + 1)
                try:
                    random_guard.set_time_context(after_time)
                    try:
                        after_result = random_guard.get(dataset_key, allow_historical_changes=False)
                        print(f"     ✓ After-time query successful (shape: {after_result.shape})")
                    finally:
                        random_guard.clear_time_context()
                except Exception as e:
                    print(f"     ❌ After-time query failed: {e}")

                # Record iteration results
                iteration_time = time.time() - iteration_start_time
                iteration_result = {
                    'iteration': iteration + 1,
                    'dataset_name': dataset_name,
                    'original_shape': base_df.shape,
                    'subset_shape': df_subset.shape,
                    'mutation_steps': n_mutation_steps,
                    'verification_success_rate': (
                        verification_results['verified_steps'] / verification_results['total_steps']
                        if verification_results['total_steps'] > 0 else 0
                    ),
                    'time_taken': iteration_time,
                    'errors': verification_results.get('errors', [])
                }

                iteration_results.append(iteration_result)

                # Determine if iteration was successful
                if verification_results['failed_steps'] == 0:
                    successful_iterations += 1
                    deletion_operations_successful += len(deletion_ops_in_sequence)
                    print(f"   ✅ Iteration {iteration + 1} PASSED ({iteration_time:.2f}s)")
                else:
                    failed_iterations += 1
                    print(f"   ❌ Iteration {iteration + 1} FAILED: {verification_results['failed_steps']} verification failures")
                    print(f"      Errors: {verification_results['errors'][:3]}...")  # Show first 3 errors

            except Exception as e:
                failed_iterations += 1
                iteration_time = time.time() - iteration_start_time
                error_msg = f"Iteration {iteration + 1} crashed: {str(e)}"
                print(f"   💥 {error_msg}")

                iteration_results.append({
                    'iteration': iteration + 1,
                    'dataset_name': 'CRASHED',
                    'original_shape': (0, 0),
                    'subset_shape': (0, 0),
                    'mutation_steps': 0,
                    'verification_success_rate': 0,
                    'time_taken': iteration_time,
                    'errors': [error_msg]
                })

        # Final reporting
        self._print_final_report(
            total_iterations,
            successful_iterations,
            failed_iterations,
            total_mutations_applied,
            total_verifications_passed,
            total_verifications_failed,
            dtype_consistency_checks,
            dtype_consistency_failures,
            deletion_operations_applied,
            deletion_operations_successful,
            iteration_results
        )

        # Test assertions
        success_rate = successful_iterations / total_iterations
        verification_rate = (
            total_verifications_passed / (total_verifications_passed + total_verifications_failed)
            if (total_verifications_passed + total_verifications_failed) > 0 else 0
        )

        # Require 100% success - no failures allowed
        # Calculate dtype consistency rate
        dtype_consistency_rate = (
            (dtype_consistency_checks - dtype_consistency_failures) / dtype_consistency_checks
            if dtype_consistency_checks > 0 else 1.0
        )

        # Collect detailed error information for failed iterations
        failed_iteration_details = [
            f"Iteration {r['iteration']}: {r.get('errors', ['Unknown error'])}"
            for r in iteration_results if r.get('errors')
        ]

        assert failed_iterations == 0, (
            f"Expected 0 failed iterations, but got {failed_iterations}/{total_iterations}.\n"
            f"Failed iterations:\n" + "\n".join(failed_iteration_details[:5])
        )

        assert total_verifications_failed == 0, (
            f"Expected 0 verification failures, but got {total_verifications_failed}/"
            f"{total_verifications_passed + total_verifications_failed}."
        )

        assert dtype_consistency_failures == 0, (
            f"Expected 0 dtype consistency failures, but got {dtype_consistency_failures}/"
            f"{dtype_consistency_checks}."
        )

        print(f"\n🎉 Dtype-consistent random mutations test PASSED!")
        print(f"   Success rate: {success_rate:.2%}")
        print(f"   Verification rate: {verification_rate:.2%}")
        print(f"   Dtype consistency rate: {dtype_consistency_rate:.2%}")

    def test_extreme_mutation_scenarios(
        self,
        random_guard: FinlabGuard,
        seeded_sampler: FinlabDataSampler,
        seeded_mutator: DataFrameMutator,
        as_of_verifier: AsOfTimeVerifier
    ):
        """Test extreme mutation scenarios that might break the system."""
        print(f"\n🔥 Testing extreme mutation scenarios")

        extreme_scenarios = [
            ("massive_dtype_changes", lambda df: self._apply_massive_dtype_changes(df, seeded_mutator)),
            ("extreme_values", lambda df: self._apply_extreme_values(df, seeded_mutator)),
            ("large_insertions", lambda df: self._apply_large_insertions(df, seeded_mutator)),
            ("massive_deletions", lambda df: self._apply_massive_deletions(df, seeded_mutator)),
            ("alternating_add_delete", lambda df: self._apply_alternating_add_delete(df, seeded_mutator)),
            ("index_chaos", lambda df: self._apply_index_chaos(df, seeded_mutator)),
        ]

        for scenario_name, scenario_func in extreme_scenarios:
            try:
                print(f"\n🧪 Testing scenario: {scenario_name}")

                # Get a test dataset
                dataset_name, base_df = seeded_sampler.sample_random_dataset()
                df_subset = seeded_sampler.sample_subset(base_df, max_rows=20, max_cols=10)

                # Apply extreme scenario
                mutated_df = scenario_func(df_subset)
                dataset_key = f"extreme_{scenario_name}"

                # Store original and mutated versions using mock pattern
                base_time = datetime.now()

                # Store original
                with patch.object(random_guard, "_now", return_value=base_time):
                    with patch.object(type(random_guard), "_fetch_from_finlab", return_value=df_subset):
                        random_guard.get(dataset_key, allow_historical_changes=False)

                # Store mutated version
                mutated_time = base_time + timedelta(seconds=1)
                with patch.object(random_guard, "_now", return_value=mutated_time):
                    with patch.object(type(random_guard), "_fetch_from_finlab", return_value=mutated_df):
                        random_guard.get(dataset_key, allow_historical_changes=True)

                # Verify both can be retrieved using time context
                random_guard.set_time_context(base_time)
                try:
                    retrieved_original = random_guard.get(dataset_key, allow_historical_changes=False)
                finally:
                    random_guard.clear_time_context()

                random_guard.set_time_context(mutated_time)
                try:
                    retrieved_mutated = random_guard.get(dataset_key, allow_historical_changes=False)
                finally:
                    random_guard.clear_time_context()

                print(f"     ✅ {scenario_name} survived: {df_subset.shape} -> {mutated_df.shape}")

            except Exception as e:
                print(f"     ❌ {scenario_name} failed: {e}")
                # Require 100% success - no failures allowed
                raise AssertionError(
                    f"Extreme scenario '{scenario_name}' must pass. Error: {str(e)}"
                ) from e

    def _apply_massive_dtype_changes(self, df, mutator):
        """Apply dtype changes to all columns."""
        return mutator.mutate_dtypes(df, n_columns=len(df.columns))

    def _apply_extreme_values(self, df, mutator):
        """Apply extreme values (NaN, Inf, very large numbers)."""
        return mutator.mutate_cell_values(df, n_changes=min(df.size, 20))

    def _apply_large_insertions(self, df, mutator):
        """Apply large numbers of row/column insertions."""
        df_with_rows = mutator.insert_random_rows(df, n_rows=min(10, len(df)))
        return mutator.insert_random_columns(df_with_rows, n_cols=min(5, len(df.columns)))

    def _apply_massive_deletions(self, df, mutator):
        """Apply aggressive deletion operations."""
        # Start with a larger dataset to allow for significant deletions
        df_with_extra_rows = mutator.insert_random_rows(df, n_rows=10)
        df_with_extra_cols = mutator.insert_random_columns(df_with_extra_rows, n_cols=5)

        # Now apply aggressive deletions
        df_deleted_rows = mutator.delete_random_rows(df_with_extra_cols, n_rows=7)
        df_deleted_cols = mutator.delete_random_columns(df_deleted_rows, n_cols=3)

        return df_deleted_cols

    def _apply_alternating_add_delete(self, df, mutator):
        """Apply alternating add and delete operations to stress the system."""
        current_df = df.copy()

        # Perform multiple rounds of add/delete cycles
        for _ in range(3):
            # Add rows and columns
            current_df = mutator.insert_random_rows(current_df, n_rows=3)
            current_df = mutator.insert_random_columns(current_df, n_cols=2)

            # Then delete some (but not all) of what we added
            if len(current_df) > 5:  # Ensure we don't delete too much
                current_df = mutator.delete_random_rows(current_df, n_rows=2)
            if len(current_df.columns) > 3:  # Ensure we don't delete too much
                current_df = mutator.delete_random_columns(current_df, n_cols=1)

        return current_df

    def _apply_index_chaos(self, df, mutator):
        """Apply chaotic index mutations."""
        return mutator.mutate_index(df)

    def _check_dtype_consistency(self, df: pd.DataFrame, description: str) -> Dict:
        """Check dtype consistency within the DataFrame.

        Args:
            df: DataFrame to check
            description: Description of the mutation step

        Returns:
            Dict with consistency check results
        """
        issues = []

        # Check for mixed types within object columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # For object columns, check if types are reasonable
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    types_found = set(type(val).__name__ for val in non_null_values)
                    if len(types_found) > 3:  # Too many different types
                        issues.append(f"Column '{col}' has too many types: {types_found}")

            # Check for NaN in integer columns (shouldn't happen with proper mutations)
            elif pd.api.types.is_integer_dtype(df[col].dtype):
                if df[col].isna().any():
                    issues.append(f"Integer column '{col}' contains NaN values")

        # Check for extreme dtype changes that might indicate issues
        if "dtype_changes" in description:
            for col in df.columns:
                # Ensure dtype is as expected
                if df[col].dtype == 'object' and df[col].notna().any():
                    # Check if object column actually contains consistent types
                    sample_values = df[col].dropna().iloc[:5]
                    if len(sample_values) > 0:
                        first_type = type(sample_values.iloc[0])
                        mixed_types = any(type(val) != first_type for val in sample_values)
                        if mixed_types and "cell_values" not in description:
                            issues.append(f"Object column '{col}' has inconsistent types after dtype change")

        return {
            'is_consistent': len(issues) == 0,
            'issues': issues,
            'total_columns': len(df.columns),
            'dtype_distribution': df.dtypes.value_counts().to_dict()
        }

    def _print_final_report(
        self,
        total_iterations: int,
        successful_iterations: int,
        failed_iterations: int,
        total_mutations_applied: int,
        total_verifications_passed: int,
        total_verifications_failed: int,
        dtype_consistency_checks: int,
        dtype_consistency_failures: int,
        deletion_operations_applied: int,
        deletion_operations_successful: int,
        iteration_results: List[Dict]
    ):
        """Print comprehensive final report."""
        print(f"\n" + "="*60)
        print(f"🎯 RANDOM MUTATIONS TEST FINAL REPORT")
        print(f"="*60)

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Iterations: {total_iterations}")
        print(f"   Successful Iterations: {successful_iterations}")
        print(f"   Failed Iterations: {failed_iterations}")
        print(f"   Success Rate: {successful_iterations/total_iterations:.2%}")

        print(f"\n🔬 MUTATION STATISTICS:")
        print(f"   Total Mutations Applied: {total_mutations_applied}")
        print(f"   Total Verifications: {total_verifications_passed + total_verifications_failed}")
        print(f"   Verifications Passed: {total_verifications_passed}")
        print(f"   Verifications Failed: {total_verifications_failed}")
        verification_rate = (
            total_verifications_passed / (total_verifications_passed + total_verifications_failed)
            if (total_verifications_passed + total_verifications_failed) > 0 else 0
        )
        print(f"   Verification Rate: {verification_rate:.2%}")

        print(f"\n🧬 DTYPE CONSISTENCY STATISTICS:")
        print(f"   Total Dtype Consistency Checks: {dtype_consistency_checks}")
        print(f"   Dtype Consistency Failures: {dtype_consistency_failures}")
        dtype_consistency_rate = (
            (dtype_consistency_checks - dtype_consistency_failures) / dtype_consistency_checks
            if dtype_consistency_checks > 0 else 1.0
        )
        print(f"   Dtype Consistency Rate: {dtype_consistency_rate:.2%}")

        print(f"\n🗑️  DELETION OPERATIONS STATISTICS:")
        print(f"   Total Deletion Operations Applied: {deletion_operations_applied}")
        print(f"   Deletion Operations Successful: {deletion_operations_successful}")
        deletion_success_rate = (
            deletion_operations_successful / deletion_operations_applied
            if deletion_operations_applied > 0 else 1.0
        )
        print(f"   Deletion Success Rate: {deletion_success_rate:.2%}")
        if deletion_operations_applied > 0:
            print(f"   Average Deletions per Iteration: {deletion_operations_applied / total_iterations:.1f}")

        # Dataset coverage
        datasets_tested = set()
        for result in iteration_results:
            if result['dataset_name'] != 'CRASHED':
                datasets_tested.add(result['dataset_name'])

        print(f"\n📚 DATASET COVERAGE:")
        print(f"   Unique Datasets Tested: {len(datasets_tested)}")
        print(f"   Datasets: {', '.join(sorted(datasets_tested))}")

        # Performance statistics
        times = [r['time_taken'] for r in iteration_results if r['time_taken'] > 0]
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"\n⏱️  PERFORMANCE STATISTICS:")
            print(f"   Average Time per Iteration: {avg_time:.2f}s")
            print(f"   Min Time: {min_time:.2f}s")
            print(f"   Max Time: {max_time:.2f}s")
            print(f"   Total Test Time: {sum(times):.1f}s")

        # Top failures
        failed_results = [r for r in iteration_results if r['errors']]
        if failed_results:
            print(f"\n❌ TOP FAILURE PATTERNS:")
            error_counts = {}
            for result in failed_results[:5]:  # Show top 5
                for error in result['errors'][:1]:  # Show first error per iteration
                    error_key = str(error)[:100]  # Truncate for readability
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1

            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {count}x: {error}")

        print(f"\n" + "="*60)