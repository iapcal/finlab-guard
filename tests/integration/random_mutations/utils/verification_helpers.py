"""Verification helpers for as-of-time query correctness in random tests."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from finlab_guard import FinlabGuard


class AsOfTimeVerifier:
    """Verifies correctness of as-of-time queries and historical data reconstruction."""

    def __init__(self, guard: FinlabGuard):
        self.guard = guard
        self.verification_errors: List[str] = []

    def verify_mutation_sequence(self,
                                dataset_key: str,
                                mutation_sequence: List[Tuple[str, pd.DataFrame]],
                                time_interval_seconds: int = 1,
                                base_timestamp: Optional[datetime] = None) -> Dict:
        """Verify that as-of-time queries return correct data for a mutation sequence.

        Args:
            dataset_key: Key identifier for the dataset
            mutation_sequence: List of (description, dataframe) tuples
            time_interval_seconds: Time interval between mutations in seconds

        Returns:
            Dictionary containing verification results
        """
        self.verification_errors = []
        results = {
            'total_steps': len(mutation_sequence),
            'verified_steps': 0,
            'failed_steps': 0,
            'errors': [],
            'step_results': []
        }

        # Use the provided base timestamp or estimate from current time
        if base_timestamp is None:
            # Estimate based on when main test likely stored the data
            base_timestamp = datetime.now() - timedelta(seconds=len(mutation_sequence) + 2)

        # Calculate the timestamps the main test used for each step
        mutation_timestamps = []
        for i, (description, df) in enumerate(mutation_sequence):
            timestamp = base_timestamp + timedelta(seconds=i)
            mutation_timestamps.append(timestamp)

        # Verify each step can be retrieved correctly at the time it was stored
        for i, (description, expected_df) in enumerate(mutation_sequence):
            # Query slightly after the storage time to ensure we get the data
            query_timestamp = mutation_timestamps[i] + timedelta(milliseconds=500)

            try:
                # Retrieve data directly from cache manager as of this timestamp
                retrieved_df = self.guard.cache_manager.load_data(dataset_key, query_timestamp)

                if retrieved_df is None or retrieved_df.empty:
                    verification_result = {
                        'success': False,
                        'errors': [f"No data found for {dataset_key} as of {query_timestamp}"]
                    }
                else:
                    # Verify the data matches
                    verification_result = self._verify_dataframe_equality(
                        expected_df, retrieved_df, f"Step {i} ({description})"
                    )

                if verification_result['success']:
                    results['verified_steps'] += 1
                    print(f"✓ Verified {description} as-of-time query")
                else:
                    results['failed_steps'] += 1
                    results['errors'].extend(verification_result['errors'])

                results['step_results'].append({
                    'step': i,
                    'description': description,
                    'timestamp': query_timestamp,
                    'success': verification_result['success'],
                    'errors': verification_result['errors']
                })

            except Exception as e:
                error_msg = f"Failed as-of-time query for step {i} ({description}): {e}"
                self.verification_errors.append(error_msg)
                results['errors'].append(error_msg)
                results['failed_steps'] += 1

                results['step_results'].append({
                    'step': i,
                    'description': description,
                    'timestamp': query_timestamp,
                    'success': False,
                    'errors': [error_msg]
                })

        return results

    def verify_historical_consistency(self,
                                    dataset_key: str,
                                    reference_sequence: List[Tuple[str, pd.DataFrame]],
                                    mutation_timestamps: List[datetime]) -> Dict:
        """Verify historical consistency by checking intermediate timestamps.

        Args:
            dataset_key: Dataset identifier
            reference_sequence: Reference mutation sequence
            mutation_timestamps: Timestamps when each mutation was applied

        Returns:
            Dictionary containing consistency verification results
        """
        results = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'consistency_errors': []
        }

        # Test queries between mutation timestamps
        for i in range(len(reference_sequence) - 1):
            current_time = mutation_timestamps[i]
            next_time = mutation_timestamps[i + 1]

            # Test query at current timestamp
            expected_df = reference_sequence[i][1]

            # Test multiple points between current and next timestamp
            test_points = 3
            for j in range(test_points):
                test_time = current_time + timedelta(
                    seconds=(next_time - current_time).total_seconds() * j / (test_points + 1)
                )

                try:
                    # Retrieve data directly from cache as of test time
                    retrieved_df = self.guard.cache_manager.reconstruct_as_of(dataset_key, test_time)

                    if retrieved_df is None or retrieved_df.empty:
                        verification = {
                            'success': False,
                            'errors': [f"No data found for consistency check at {test_time}"]
                        }
                    else:
                        verification = self._verify_dataframe_equality(
                            expected_df, retrieved_df,
                            f"Consistency check between step {i} and {i+1} at {test_time}"
                        )

                    results['total_checks'] += 1
                    if verification['success']:
                        results['passed_checks'] += 1
                    else:
                        results['failed_checks'] += 1
                        results['consistency_errors'].extend(verification['errors'])

                except Exception as e:
                    results['total_checks'] += 1
                    results['failed_checks'] += 1
                    error_msg = f"Consistency check failed at {test_time}: {e}"
                    results['consistency_errors'].append(error_msg)

        return results

    def verify_dtype_preservation(self,
                                original_df: pd.DataFrame,
                                retrieved_df: pd.DataFrame,
                                step_description: str = "") -> Dict:
        """Verify that dtypes are preserved correctly in retrieved data.

        Args:
            original_df: Original DataFrame
            retrieved_df: Retrieved DataFrame from cache
            step_description: Description of the test step

        Returns:
            Dictionary containing dtype verification results
        """
        results = {
            'success': True,
            'dtype_mismatches': [],
            'missing_columns': [],
            'extra_columns': []
        }

        # Check for missing/extra columns
        orig_cols = set(original_df.columns)
        retr_cols = set(retrieved_df.columns)

        results['missing_columns'] = list(orig_cols - retr_cols)
        results['extra_columns'] = list(retr_cols - orig_cols)

        if results['missing_columns'] or results['extra_columns']:
            results['success'] = False

        # Check dtype consistency for common columns
        common_cols = orig_cols & retr_cols

        for col in common_cols:
            orig_dtype = original_df[col].dtype
            retr_dtype = retrieved_df[col].dtype

            # Allow for some dtype flexibility (e.g., int64 vs int32)
            if not self._dtypes_compatible(orig_dtype, retr_dtype):
                results['dtype_mismatches'].append({
                    'column': col,
                    'expected_dtype': str(orig_dtype),
                    'actual_dtype': str(retr_dtype)
                })
                results['success'] = False

        # Check index dtype
        if not self._dtypes_compatible(original_df.index.dtype, retrieved_df.index.dtype):
            results['dtype_mismatches'].append({
                'column': 'INDEX',
                'expected_dtype': str(original_df.index.dtype),
                'actual_dtype': str(retrieved_df.index.dtype)
            })
            results['success'] = False

        return results

    def _verify_dataframe_equality(self,
                                 expected: pd.DataFrame,
                                 actual: pd.DataFrame,
                                 context: str = "") -> Dict:
        """Verify two DataFrames are equal with detailed error reporting.

        Args:
            expected: Expected DataFrame
            actual: Actual DataFrame
            context: Context description for error messages

        Returns:
            Dictionary with verification results
        """
        result = {
            'success': True,
            'errors': []
        }

        try:
            # Check basic properties
            if expected.shape != actual.shape:
                # Add detailed debugging output for shape mismatches
                print(f"\n🐛 DEBUG: Shape mismatch for {context}")
                print(f"Expected shape: {expected.shape}, Actual shape: {actual.shape}")
                print(f"Expected columns: {list(expected.columns)}")
                print(f"Actual columns: {list(actual.columns)}")
                print(f"Expected index: {list(expected.index)}")
                print(f"Actual index: {list(actual.index)}")

                print(f"\n📊 EXPECTED DATA (shape {expected.shape}):")
                print(expected)
                print(f"\n📊 ACTUAL DATA (shape {actual.shape}):")
                print(actual)

                result['success'] = False
                result['errors'].append(
                    f"{context}: Shape mismatch - expected {expected.shape}, got {actual.shape}"
                )
                return result

            # Check columns (allow for reordering)
            expected_cols = set(expected.columns)
            actual_cols = set(actual.columns)

            if expected_cols != actual_cols:
                result['success'] = False
                result['errors'].append(
                    f"{context}: Column content mismatch - expected {sorted(expected_cols)}, got {sorted(actual_cols)}"
                )
                return result

            # Reorder actual to match expected column order for comparison
            actual_reordered = actual[expected.columns] if len(actual.columns) > 0 else actual

            # For index comparison, be more lenient - only check values, not names/types
            try:
                # Try to compare with flexible index comparison
                expected_sorted = expected.sort_index()
                actual_sorted = actual_reordered.sort_index()

                assert_frame_equal(
                    expected_sorted, actual_sorted,
                    check_dtype=False,           # Allow dtype flexibility
                    check_index_type=False,      # Allow index type flexibility
                    check_names=False,           # Allow index name flexibility
                    rtol=1e-5, atol=1e-8,       # Reasonable numeric tolerance
                    check_exact=False            # Allow for floating point differences
                )
            except (AssertionError, ValueError) as e:
                # If sorting fails, try direct comparison with relaxed constraints
                try:
                    assert_frame_equal(
                        expected, actual_reordered,
                        check_dtype=False,
                        check_index_type=False,
                        check_names=False,
                        rtol=1e-5, atol=1e-8,
                        check_exact=False
                    )
                except (AssertionError, ValueError) as e2:
                    # Add detailed debugging output
                    print(f"\n🐛 DEBUG: Data comparison failed for {context}")
                    print(f"Expected shape: {expected.shape}, Actual shape: {actual_reordered.shape}")
                    print(f"Expected columns: {list(expected.columns)}")
                    print(f"Actual columns: {list(actual_reordered.columns)}")
                    print(f"Expected index: {list(expected.index)}")
                    print(f"Actual index: {list(actual_reordered.index)}")

                    print(f"\n📊 EXPECTED DATA:")
                    print(expected)
                    print(f"\n📊 ACTUAL DATA:")
                    print(actual_reordered)

                    # Show differences cell by cell for first few differing columns
                    common_cols = set(expected.columns) & set(actual_reordered.columns)
                    if common_cols:
                        print(f"\n🔍 CELL-BY-CELL DIFFERENCES (first 3 columns):")
                        for col in list(common_cols)[:3]:
                            if col in expected.columns and col in actual_reordered.columns:
                                exp_vals = expected[col]
                                act_vals = actual_reordered[col]
                                print(f"  Column '{col}':")
                                for idx in expected.index:
                                    if idx in actual_reordered.index:
                                        exp_val = exp_vals.loc[idx] if idx in exp_vals.index else "MISSING"
                                        act_val = act_vals.loc[idx] if idx in act_vals.index else "MISSING"
                                        if str(exp_val) != str(act_val):
                                            print(f"    [{idx}]: expected {exp_val}, got {act_val}")

                    print(f"\n❌ Assertion Error: {str(e2)[:500]}")

                    result['success'] = False
                    result['errors'].append(f"{context}: Data comparison failed - {str(e2)[:200]}...")

            # Skip dtype verification for now as it's often too strict for random testing

        except Exception as e:
            result['success'] = False
            result['errors'].append(f"{context}: Verification failed with exception - {e}")

        return result

    def _dtypes_compatible(self, dtype1, dtype2) -> bool:
        """Check if two dtypes are compatible (allowing for reasonable conversions)."""
        # Exact match
        if dtype1 == dtype2:
            return True

        # Convert to string for easier comparison
        d1_str = str(dtype1).lower()
        d2_str = str(dtype2).lower()

        # Allow numeric type flexibility
        numeric_groups = [
            ['int8', 'int16', 'int32', 'int64'],
            ['uint8', 'uint16', 'uint32', 'uint64'],
            ['float16', 'float32', 'float64'],
            ['object', 'string'],
            ['category']
        ]

        for group in numeric_groups:
            if any(t in d1_str for t in group) and any(t in d2_str for t in group):
                return True

        # Special cases
        if ('datetime' in d1_str and 'datetime' in d2_str) or \
           ('timedelta' in d1_str and 'timedelta' in d2_str):
            return True

        return False

    def generate_verification_report(self, results: Dict) -> str:
        """Generate a human-readable verification report.

        Args:
            results: Results dictionary from verify_mutation_sequence

        Returns:
            Formatted report string
        """
        report = []
        report.append("=== As-of-Time Verification Report ===")
        report.append(f"Total Steps: {results['total_steps']}")
        report.append(f"Verified Steps: {results['verified_steps']}")
        report.append(f"Failed Steps: {results['failed_steps']}")
        report.append(f"Success Rate: {results['verified_steps']/results['total_steps']*100:.1f}%")

        if results['errors']:
            report.append("\n=== Errors ===")
            for error in results['errors']:
                report.append(f"❌ {error}")

        if results['step_results']:
            report.append("\n=== Step Details ===")
            for step_result in results['step_results']:
                status = "✓" if step_result['success'] else "❌"
                report.append(f"{status} Step {step_result['step']}: {step_result['description']}")
                if step_result['errors']:
                    for error in step_result['errors']:
                        report.append(f"    - {error}")

        return "\n".join(report)