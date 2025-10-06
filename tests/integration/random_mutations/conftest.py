"""Configuration for random DataFrame mutation tests."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from finlab_guard import FinlabGuard
from tests.integration.random_mutations.utils.dataframe_mutators import DataFrameMutator
from tests.integration.random_mutations.utils.finlab_samplers import FinlabDataSampler
from tests.integration.random_mutations.utils.verification_helpers import (
    AsOfTimeVerifier,
)


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create temporary cache directory for random tests."""
    temp_dir = tempfile.mkdtemp(prefix="finlab_guard_random_")
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def random_guard(temp_cache_dir: Path) -> Generator[FinlabGuard, None, None]:
    """Create FinlabGuard instance for random testing."""
    config = {
        "compression": None,  # Disable compression for faster testing
        "allow_historical_changes": True,  # Enable historical changes for mutation testing
    }

    guard = FinlabGuard(cache_dir=temp_cache_dir, config=config)
    try:
        yield guard
    finally:
        # Ensure cleanup
        try:
            guard.close()
        except Exception:
            pass


@pytest.fixture
def finlab_sampler() -> FinlabDataSampler:
    """Create FinlabDataSampler for loading real finlab data."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    return FinlabDataSampler(pickle_dir=fixtures_dir)


@pytest.fixture
def dataframe_mutator() -> DataFrameMutator:
    """Create DataFrameMutator for DataFrame mutations."""
    return DataFrameMutator()


@pytest.fixture
def as_of_verifier(random_guard: FinlabGuard) -> AsOfTimeVerifier:
    """Create AsOfTimeVerifier for verification."""
    return AsOfTimeVerifier(random_guard)


@pytest.fixture(params=range(50), scope="session")
def test_seed(request) -> int:
    """Provide different random seeds for test reproducibility."""
    # Generate 50 diverse seeds for better parallel testing
    return 42 + request.param * 100


@pytest.fixture
def seeded_sampler(test_seed: int) -> FinlabDataSampler:
    """Create seeded FinlabDataSampler for reproducible tests."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    return FinlabDataSampler(pickle_dir=fixtures_dir, random_seed=test_seed)


@pytest.fixture
def seeded_mutator(test_seed: int) -> DataFrameMutator:
    """Create seeded DataFrameMutator for reproducible mutations."""
    return DataFrameMutator(random_seed=test_seed)
