"""Test configuration for finlab-guard.

This is the main test configuration file. The specific test directories
have their own conftest.py files with specialized configurations:

- tests/integration/mock/conftest.py: Mock finlab environment
- tests/integration/real/conftest.py: Real finlab integration
"""

import os
from pathlib import Path

import pytest

# Load environment variables from .env file if it exists
# Only load if the environment variable is not already set
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # Only set if not already in environment (allows override)
                if key not in os.environ:
                    os.environ[key] = value


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "real_finlab: marks tests that require real finlab connection"
    )
    # Note: mock_only marker is deprecated in favor of directory structure
