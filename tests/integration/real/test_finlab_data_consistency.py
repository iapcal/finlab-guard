"""Test specific finlab data calls for consistency with pandas.compare() implementation."""

import tempfile
from pathlib import Path

import pytest

from finlab_guard import FinlabGuard
from finlab_guard.utils.exceptions import DataModifiedException


class TestFinlabDataConsistency:
    """Test consistency of specific finlab data calls that previously caused issues."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup is handled by tempfile

    @pytest.fixture
    def guard(self, temp_cache_dir):
        """Create FinlabGuard instance with temporary cache."""
        guard_instance = FinlabGuard(cache_dir=temp_cache_dir)
        yield guard_instance
        # Ensure DuckDB connection is closed to prevent Windows file locking
        guard_instance.close()

    def test_etl_full_cash_delivery_stock_filter_consistency(self, guard):
        """
        Test that etl:full_cash_delivery_stock_filter can be called twice without DataModifiedException.

        This test specifically targets data that may have precision issues or type inconsistencies.
        """
        try:
            guard.install_patch()

            # Import finlab.data after patch is installed
            import finlab.data as data

            # First call - should cache the data
            result1 = data.get("etl:full_cash_delivery_stock_filter")
            assert result1 is not None
            print(f"First call dtype: {result1.dtypes}")

            # Second call - should NOT trigger DataModifiedException
            # This is the critical test for our pandas.compare() implementation
            result2 = data.get("etl:full_cash_delivery_stock_filter")
            assert result2 is not None
            print(f"Second call dtype: {result2.dtypes}")

            # Results should be equivalent (not necessarily identical due to potential precision)
            assert result1.shape == result2.shape
            assert list(result1.columns) == list(result2.columns)

        except DataModifiedException as e:
            pytest.fail(
                f"DataModifiedException should not be raised on consecutive calls: {e}"
            )
        finally:
            guard.remove_patch()

    def test_etl_adj_close_consistency(self, guard):
        """
        Test that etl:adj_close can be called twice without DataModifiedException.

        This test targets float64 data that commonly has precision issues.
        """
        try:
            guard.install_patch()

            # Import finlab.data after patch is installed
            import finlab.data as data

            # First call - should cache the data
            result1 = data.get("etl:adj_close")
            assert result1 is not None
            print(f"First call dtype: {result1.dtypes}")
            print(f"First call sample values: {result1.iloc[:2, :2]}")

            # Second call - should NOT trigger DataModifiedException
            # This is critical for float64 precision handling
            result2 = data.get("etl:adj_close")
            assert result2 is not None
            print(f"Second call dtype: {result2.dtypes}")
            print(f"Second call sample values: {result2.iloc[:2, :2]}")

            # Results should be equivalent
            assert result1.shape == result2.shape
            assert list(result1.columns) == list(result2.columns)

        except DataModifiedException as e:
            pytest.fail(
                f"DataModifiedException should not be raised on consecutive calls: {e}"
            )
        finally:
            guard.remove_patch()

    def test_delisted_companies_otc_consistency(self, guard):
        """
        Test that delisted_companies_otc can be called twice without DataModifiedException.

        This test targets data that may have mixed types or special formatting.
        """
        try:
            guard.install_patch()

            # Import finlab.data after patch is installed
            import finlab.data as data

            # First call - should cache the data
            result1 = data.get("delisted_companies_otc")
            assert result1 is not None
            print(f"First call dtype: {result1.dtypes}")

            # Second call - should NOT trigger DataModifiedException
            result2 = data.get("delisted_companies_otc")
            assert result2 is not None
            print(f"Second call dtype: {result2.dtypes}")

            # Results should be equivalent
            assert result1.shape == result2.shape
            assert list(result1.columns) == list(result2.columns)

        except DataModifiedException as e:
            pytest.fail(
                f"DataModifiedException should not be raised on consecutive calls: {e}"
            )
        finally:
            guard.remove_patch()

    def test_all_three_datasets_consecutive_calls(self, guard):
        """
        Test all three datasets in sequence to ensure comprehensive compatibility.

        This is the integration test that covers the complete workflow.
        """
        datasets = [
            "etl:full_cash_delivery_stock_filter",
            "etl:adj_close",
            "delisted_companies_otc",
        ]

        try:
            guard.install_patch()

            # Import finlab.data after patch is installed
            import finlab.data as data

            results = {}

            # First round - cache all datasets
            for dataset in datasets:
                print(f"\nFirst call to {dataset}")
                result = data.get(dataset)
                assert result is not None
                results[dataset] = {
                    "shape": result.shape,
                    "dtypes": result.dtypes,
                    "columns": list(result.columns),
                }
                print(f"  Shape: {result.shape}, Dtypes: {result.dtypes}")

            # Second round - should all hit cache without issues
            for dataset in datasets:
                print(f"\nSecond call to {dataset}")
                result = data.get(dataset)
                assert result is not None

                # Verify consistency
                cached_info = results[dataset]
                assert result.shape == cached_info["shape"]
                assert list(result.columns) == cached_info["columns"]
                print(f"  ✅ Consistent: {dataset}")

        except DataModifiedException as e:
            pytest.fail(
                f"DataModifiedException should not be raised on consecutive calls: {e}"
            )
        finally:
            guard.remove_patch()

    def test_precision_tolerance_effectiveness(self, guard):
        """
        Test that our tolerance-based comparison prevents false positives.

        This test specifically validates the pandas.compare() + tolerance logic.
        """
        try:
            guard.install_patch()

            # Import finlab.data after patch is installed
            import finlab.data as data

            # Test with adj_close which is most likely to have precision issues
            print("Testing precision tolerance with etl:adj_close")

            # Multiple consecutive calls should all succeed
            for i in range(3):
                print(f"Call {i + 1}/3")
                result = data.get("etl:adj_close")
                assert result is not None
                print(f"  Shape: {result.shape}")

            print("✅ All calls succeeded - tolerance working correctly")

        except DataModifiedException as e:
            pytest.fail(
                f"Precision tolerance failed - DataModifiedException raised: {e}"
            )
        finally:
            guard.remove_patch()
