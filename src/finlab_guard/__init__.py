"""
finlab-guard: A lightweight package for managing local finlab data cache with version control.

This package provides automatic caching and version control for finlab data,
ensuring reproducible backtesting results by detecting and managing data changes.
"""

from typing import Optional

__version__ = "0.2.0"

from .core.guard import FinlabGuard


# 便利的安裝函數
def install(
    cache_dir: str = "~/.finlab_guard", config: Optional[dict] = None
) -> FinlabGuard:
    """
    Install finlab-guard with monkey patching.

    Args:
        cache_dir: Directory to store cache files
        config: Configuration dictionary

    Returns:
        FinlabGuard instance
    """
    guard = FinlabGuard(cache_dir=cache_dir, config=config)
    guard.install_patch()
    return guard


__all__ = ["FinlabGuard", "install"]
